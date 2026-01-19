"""
Text Chunker Step
文本分块步骤
"""

import uuid
import math
from typing import List, Tuple

from .base import ProcessingStep, ProcessingContext, DocumentChunk


class TextChunker(ProcessingStep):
    """
    文本分块器
    将长文本分割成适合向量化的小块
    """
    
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        min_chunk_length: int = 10,
        separators: List[str] = None,
    ):
        """
        初始化文本分块器
        
        Args:
            chunk_size: 块大小（字符数）
            chunk_overlap: 块之间的重叠量
            min_chunk_length: 最小块长度
            separators: 分隔符列表，按优先级排序
        """
        super().__init__("TextChunker")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "]
    
    async def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        分块处理
        
        Args:
            context: 包含文档的处理上下文
            
        Returns:
            ProcessingContext: 包含文本块
        """
        self.logger.info(
            f"🔄 Chunking {len(context.documents)} documents "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})..."
        )

        # Light diagnostics to help spot unexpectedly large inputs
        # (this is the most common reason for huge chunk counts).
        try:
            doc_sizes: List[Tuple[int, str]] = []
            total_chars = 0
            for doc in context.documents:
                content = doc.get("content") or ""
                if not isinstance(content, str):
                    content = str(content)
                n = len(content)
                total_chars += n
                path = (
                    doc.get("metadata", {}).get("relative_path")
                    or doc.get("file_path", "unknown")
                )
                doc_sizes.append((n, path))

            if doc_sizes:
                doc_sizes.sort(reverse=True, key=lambda x: x[0])
                avg_chars = total_chars / max(1, len(doc_sizes))
                self.logger.info(
                    f"📏 Input size: total_chars={total_chars}, avg_chars/doc={avg_chars:.0f}, "
                    f"max_chars/doc={doc_sizes[0][0]}"
                )
                # Only print the top offenders when inputs are large enough to plausibly explode.
                if total_chars >= 50_000_000:  # ~50MB of text
                    top_n = 10
                    self.logger.info(f"📌 Top {top_n} largest documents (by chars):")
                    for n, path in doc_sizes[:top_n]:
                        self.logger.info(f"   - chars={n}  path={path}")
        except Exception:
            # Never fail chunking due to diagnostics
            pass

        top_by_chunks: List[Tuple[int, int, str]] = []  # (chunks, chars, path)
        
        for doc in context.documents:
            try:
                content = doc.get("content") or ""
                if not isinstance(content, str):
                    content = str(content)

                metadata = doc.get("metadata", {}) or {}

                chunks = self._chunk_text(content, metadata)
                # Adaptive fallback: paragraph-aware chunking can be inefficient when the
                # extracted text is composed of many medium-sized "paragraphs" that don't
                # pack well. If a doc produces far more chunks than a fixed-size window
                # lower bound, switch to window splitting for that doc to reduce chunk count.
                try:
                    step = self.chunk_size - self.chunk_overlap if self.chunk_overlap > 0 else self.chunk_size
                    step = max(1, step)
                    expected_min = max(1, math.ceil(len(content) / step))
                    # Only consider fallback for large docs where the difference matters.
                    if len(content) >= 50_000 and len(chunks) > expected_min * 1.25:
                        chunks = self._chunk_text_window(content, metadata)
                except Exception:
                    pass
                # Track top chunk producers for debugging
                try:
                    path = (
                        doc.get("metadata", {}).get("relative_path")
                        or doc.get("file_path", "unknown")
                    )
                    top_by_chunks.append((len(chunks), len(content), path))
                except Exception:
                    pass
                for chunk in chunks:
                    context.add_chunk(chunk)
                    
            except Exception as e:
                context.add_error(
                    f"Failed to chunk {doc.get('file_path', 'unknown')}: {str(e)}"
                )
        
        created = context.stats["chunks_created"]
        self.logger.info(f"✅ Created {created} chunks")

        # If the chunk count is unusually high, summarize the biggest contributors.
        if created >= 100_000 and top_by_chunks:
            try:
                top_by_chunks.sort(reverse=True, key=lambda x: x[0])
                top_n = 20
                self.logger.info(f"📌 Top {top_n} documents by chunk count:")
                for c, chars, path in top_by_chunks[:top_n]:
                    self.logger.info(f"   - chunks={c} chars={chars} path={path}")
            except Exception:
                pass
        return context
    
    def _chunk_text(self, text: str, metadata: dict) -> List[DocumentChunk]:
        """
        将文本分割成块
        
        使用段落感知的分块策略
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        
        buffer = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_length = len(para)
            
            # 如果当前段落本身就超过块大小，需要进一步分割
            if para_length > self.chunk_size:
                # 先刷新当前缓冲区
                if buffer:
                    chunk_text = "\n\n".join(buffer)
                    if len(chunk_text.strip()) >= self.min_chunk_length:
                        chunks.append(self._create_chunk(chunk_text, metadata, len(chunks)))
                    buffer = []
                    current_length = 0
                
                # 分割大段落
                sub_chunks = self._split_large_paragraph(para)
                for sub_chunk in sub_chunks:
                    if len(sub_chunk.strip()) >= self.min_chunk_length:
                        chunks.append(self._create_chunk(sub_chunk, metadata, len(chunks)))
                continue
            
            # 检查是否需要刷新缓冲区
            new_length = current_length + para_length + (2 if buffer else 0)  # +2 for \n\n
            if new_length > self.chunk_size and buffer:
                # 刷新缓冲区
                chunk_text = "\n\n".join(buffer)
                if len(chunk_text.strip()) >= self.min_chunk_length:
                    chunks.append(self._create_chunk(chunk_text, metadata, len(chunks)))
                
                # 创建重叠
                if self.chunk_overlap > 0:
                    overlap_text = chunk_text[-self.chunk_overlap:]
                    buffer = [overlap_text]
                    current_length = len(overlap_text)
                else:
                    buffer = []
                    current_length = 0
            
            # 添加段落到缓冲区
            buffer.append(para)
            current_length = len("\n\n".join(buffer))
        
        # 处理剩余的缓冲区
        if buffer:
            chunk_text = "\n\n".join(buffer)
            if len(chunk_text.strip()) >= self.min_chunk_length:
                chunks.append(self._create_chunk(chunk_text, metadata, len(chunks)))
        
        return chunks

    def _chunk_text_window(self, text: str, metadata: dict) -> List[DocumentChunk]:
        """
        Fixed-size sliding window chunking.

        This is a robust fallback for very large documents where paragraph packing is
        inefficient, producing too many small-ish chunks.
        """
        if not text or not text.strip():
            return []

        chunks: List[DocumentChunk] = []
        step = self.chunk_size - self.chunk_overlap if self.chunk_overlap > 0 else self.chunk_size
        step = max(1, step)

        for i in range(0, len(text), step):
            chunk_text = text[i:i + self.chunk_size]
            if len(chunk_text.strip()) >= self.min_chunk_length:
                chunks.append(self._create_chunk(chunk_text, metadata, len(chunks)))

        return chunks
    
    def _split_large_paragraph(self, para: str) -> List[str]:
        """分割超大段落"""
        chunks = []
        
        # 尝试使用分隔符分割
        for separator in self.separators:
            if separator in para:
                parts = para.split(separator)
                buffer = ""
                
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    
                    test_text = buffer + separator + part if buffer else part
                    if len(test_text) <= self.chunk_size:
                        buffer = test_text
                    else:
                        if buffer:
                            chunks.append(buffer)
                        buffer = part
                
                if buffer:
                    chunks.append(buffer)
                
                if chunks:
                    # Heuristic: if separator-based splitting produces unusually small chunks
                    # (common when the HTML->text result has very long "lines"), fall back to
                    # fixed-size sliding window to reduce chunk count and stabilize sizes.
                    try:
                        if len(para) >= self.chunk_size * 8:
                            avg_len = sum(len(c) for c in chunks) / max(1, len(chunks))
                            # If average chunk is far below target, prefer fixed-size split.
                            if avg_len < self.chunk_size * 0.75:
                                break
                    except Exception:
                        pass
                    return chunks
        
        # 如果没有合适的分隔符，按字符数强制分割
        for i in range(0, len(para), self.chunk_size - self.chunk_overlap):
            chunk = para[i:i + self.chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, text: str, metadata: dict, index: int) -> DocumentChunk:
        """创建文档块"""
        chunk_metadata = metadata.copy()
        chunk_metadata["chunk_index"] = index
        chunk_metadata["chunk_id"] = str(uuid.uuid4())
        
        # 生成统一的 doc_id，格式: relative_path#chunk_index
        # 优先使用相对路径（相对于 --data-dir），确保无论数据目录放在哪里，doc_id 都是一致的
        # 这样可以在 Milvus 和 Elasticsearch 之间保持一致，便于 reranker 融合
        relative_path = metadata.get("relative_path", "") or metadata.get("file_path", "")
        doc_id = f"{relative_path}#{index}"
        chunk_metadata["doc_id"] = doc_id
        
        return DocumentChunk(
            id=doc_id,  # 使用统一的 doc_id 作为主键
            content=text.strip(),
            metadata=chunk_metadata,
        )

