"""
Pipeline Runner
通用流水线执行器：根据 PipelineConfig 串联 steps，并将结果写入指定存储后端。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any, Dict, List, Optional

from .config import ProcessorSettings, get_processor_settings
from .pipelines import StepType, get_pipeline_config, PipelineConfig, StepConfig
from .steps import (
    ProcessingContext,
    ProcessingStep,
    DocumentReader,
    HTMLProcessor,
    TextChunker,
    EmbeddingStep,
)
from .storage import StorageResult, create_storage_backend


@dataclass
class PipelineResult:
    """流水线运行结果"""

    success: bool = True
    backend: str = ""
    pipeline_name: str = ""
    duration_seconds: float = 0.0
    stats: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    storage_result: Optional[StorageResult] = None

    def __str__(self) -> str:
        lines = [
            "✅ Pipeline succeeded" if self.success else "❌ Pipeline failed",
            f"   Backend: {self.backend}",
        ]
        if self.pipeline_name:
            lines.append(f"   Pipeline: {self.pipeline_name}")
        if self.duration_seconds:
            lines.append(f"   Duration: {self.duration_seconds:.2f}s")
        if self.stats:
            lines.append("📊 Stats:")
            for k, v in self.stats.items():
                lines.append(f"   - {k}: {v}")
        if self.storage_result is not None:
            lines.append(f"💾 Storage: {self.storage_result}")
        if self.errors:
            lines.append("⚠️ Errors:")
            for e in self.errors[:10]:
                lines.append(f"   - {e}")
            if len(self.errors) > 10:
                lines.append(f"   ... and {len(self.errors) - 10} more")
        return "\n".join(lines)


class PipelineRunner:
    """
    流水线执行器

    Usage:
        runner = PipelineRunner()
        result = await runner.run(data_dir="./data/zh-cn", storage_backend="milvus")
    """

    def __init__(self, settings: Optional[ProcessorSettings] = None):
        self.settings = settings or get_processor_settings()

    async def run(self, dry_run: bool = False, **overrides) -> PipelineResult:
        """
        运行流水线

        Args:
            dry_run: 试运行模式，不进行存储
            **overrides: 覆盖 ProcessorSettings 的字段，例如 data_dir, storage_backend...
        """

        settings = self._merge_settings(self.settings, overrides)
        started = time.time()

        backend_name = settings.storage_backend
        pipeline_config = get_pipeline_config(backend_name)

        # Decide whether we need embedding based on storage backend.
        storage_probe = create_storage_backend(backend_name, **self._storage_kwargs(settings, vector_dim=1))
        requires_embedding = bool(getattr(storage_probe, "requires_embedding", True))

        # Validate pipeline vs storage expectations (no magic auto-step injection).
        self._validate_pipeline(pipeline_config, requires_embedding=requires_embedding)

        context = ProcessingContext(data_dir=settings.data_dir)

        # Build steps from pipeline config and run them sequentially.
        steps: List[ProcessingStep] = self._build_steps(pipeline_config, settings)
        for step in steps:
            context = await step(context)

        storage_result: Optional[StorageResult] = None
        if not dry_run:
            docs_to_store = self._prepare_documents_for_storage(context, requires_embedding=requires_embedding)
            vector_dim = self._infer_vector_dim(context, fallback_step=self._find_embedder(steps))
            async with create_storage_backend(
                backend_name, **self._storage_kwargs(settings, vector_dim=vector_dim)
            ) as storage:
                storage_result = await storage.store(docs_to_store)

            # Merge storage result back into stats/errors
            context.stats["chunks_stored"] = int(storage_result.stored_count)
            for e in storage_result.errors:
                context.add_error(e)

        duration = time.time() - started
        success = (context.stats.get("errors", 0) == 0) and (storage_result.success if storage_result else True)

        return PipelineResult(
            success=success,
            backend=backend_name,
            pipeline_name=pipeline_config.name,
            duration_seconds=duration,
            stats=dict(context.stats),
            errors=list(context.errors),
            storage_result=storage_result,
        )

    def _merge_settings(self, base: ProcessorSettings, overrides: Dict[str, Any]) -> ProcessorSettings:
        if not overrides:
            return base
        kwargs: Dict[str, Any] = {f.name: getattr(base, f.name) for f in dataclass_fields(ProcessorSettings)}
        kwargs.update(overrides)
        return ProcessorSettings(**kwargs)

    def _validate_pipeline(self, cfg: PipelineConfig, *, requires_embedding: bool) -> None:
        enabled = cfg.get_enabled_steps()
        enabled_types = {s.step_type for s in enabled}
        
        # 只有需要 embedding 的存储后端才需要 EMBEDDER 步骤
        if requires_embedding and StepType.EMBEDDER not in enabled_types:
            raise ValueError(
                f"Pipeline '{cfg.name}' missing EMBEDDER step for backend storage. "
                f"Either update pipeline config or use a storage backend that doesn't require embeddings."
            )

    def _build_steps(self, cfg: PipelineConfig, settings: ProcessorSettings) -> List[ProcessingStep]:
        steps: List[ProcessingStep] = []

        for step_cfg in cfg.get_enabled_steps():
            step_type = step_cfg.step_type
            params = dict(step_cfg.params or {})

            if step_type == StepType.READER:
                steps.append(
                    DocumentReader(
                        extensions=params.get("extensions") or settings.required_extensions,
                        exclude_patterns=params.get("exclude_patterns") or settings.exclude_patterns,
                        recursive=bool(params.get("recursive", True)),
                    )
                )
            elif step_type == StepType.HTML_PROCESSOR:
                steps.append(
                    HTMLProcessor(
                        extract_codes=bool(params.get("extract_codes", True)),
                        content_selector=params.get("content_selector", "div.td-content"),
                        code_blocks_dir=params.get("code_blocks_dir") or settings.code_blocks_dir,
                        skip_if_missing_selector=bool(params.get("skip_if_missing_selector", True)),
                        html2text_ignore_links=bool(params.get("html2text_ignore_links", True)),
                        html2text_body_width=int(params.get("html2text_body_width", 0)),
                    )
                )
            elif step_type == StepType.CHUNKER:
                steps.append(
                    TextChunker(
                        chunk_size=int(params.get("chunk_size", settings.chunk_size)),
                        chunk_overlap=int(params.get("chunk_overlap", settings.chunk_overlap)),
                        min_chunk_length=int(params.get("min_chunk_length", settings.min_text_length)),
                        separators=params.get("separators"),
                    )
                )
            elif step_type == StepType.EMBEDDER:
                steps.append(EmbeddingStep(batch_size=int(params.get("batch_size", settings.batch_size))))
            else:
                # Unknown/unsupported step type: ignore for forward compatibility.
                continue

        return steps

    def _find_embedder(self, steps: List[ProcessingStep]) -> Optional[EmbeddingStep]:
        for s in steps:
            if isinstance(s, EmbeddingStep):
                return s
        return None

    def _infer_vector_dim(self, context: ProcessingContext, fallback_step: Optional[EmbeddingStep]) -> int:
        # Prefer actual produced embeddings.
        for c in context.chunks:
            if c.embedding is not None and len(c.embedding) > 0:
                return int(len(c.embedding))
        # Fall back to embedding service configured dimension.
        if fallback_step is not None:
            try:
                return int(fallback_step.get_embedding_dimension())
            except Exception:
                pass
        # Conservative default (bge-small commonly 384/512; we default to 512)
        return 512

    def _prepare_documents_for_storage(self, context: ProcessingContext, *, requires_embedding: bool) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []

        # 使用分块存储（更好的检索粒度，且与 Milvus/ES 保持一致便于 reranker 融合）
        if context.chunks:
            for chunk in context.chunks:
                if not chunk.is_valid:
                    continue
                # 只有需要 embedding 的存储才跳过没有 embedding 的分块
                if requires_embedding and not chunk.embedding:
                    continue
                
                doc_dict = chunk.to_dict()
                # 确保 doc_id 字段存在（用于 reranker 融合）
                if "doc_id" not in doc_dict.get("metadata", {}):
                    # 如果 metadata 中没有 doc_id，使用 chunk 的 id
                    doc_dict.setdefault("metadata", {})["doc_id"] = doc_dict.get("id", "")
                
                docs.append(doc_dict)
            return docs

        # Fallback: store whole documents (only works if storage doesn't require embedding).
        for doc in context.documents:
            content = doc.get("content") or ""
            if not isinstance(content, str):
                content = str(content)
            if not content.strip():
                continue
            
            metadata = doc.get("metadata", {}) or {}
            # 优先使用相对路径（相对于 --data-dir），确保 doc_id 一致性
            relative_path = (
                metadata.get("relative_path", "") or 
                metadata.get("file_path", "") or 
                doc.get("file_path", "")
            )
            # 生成统一的 doc_id，使用相对路径
            doc_id = metadata.get("doc_id") or f"{relative_path}#0"
            
            docs.append(
                {
                    "id": doc_id,
                    "content": content,
                    "metadata": {
                        **metadata,
                        "doc_id": doc_id,
                        "file_path": relative_path,  # 使用相对路径
                        "relative_path": relative_path,
                    },
                }
            )
        return docs

    def _storage_kwargs(self, settings: ProcessorSettings, *, vector_dim: int) -> Dict[str, Any]:
        if settings.storage_backend == "milvus":
            return {
                "uri": settings.milvus_uri,
                "collection_name": settings.collection_name,
                "vector_dim": int(vector_dim),
                "overwrite": bool(settings.milvus_overwrite),
                "similarity_metric": "COSINE",  # 统一使用 COSINE 度量类型
            }
        if settings.storage_backend == "elasticsearch":
            return {
                "es_url": settings.es_host,
                "index_name": settings.es_index,
                "username": settings.es_user,
                "password": settings.es_password,
            }
        return {}


async def run_pipeline(dry_run: bool = False, **kwargs) -> PipelineResult:
    """
    便捷函数：按 kwargs 创建/覆盖 ProcessorSettings 并执行流水线。
    """

    settings = get_processor_settings(**{k: v for k, v in kwargs.items() if k in {f.name for f in dataclass_fields(ProcessorSettings)}})
    runner = PipelineRunner(settings)
    # Pass through non-settings kwargs (e.g., dry_run handled explicitly).
    passthrough = {k: v for k, v in kwargs.items() if k not in {f.name for f in dataclass_fields(ProcessorSettings)}}
    return await runner.run(dry_run=dry_run, **passthrough)


