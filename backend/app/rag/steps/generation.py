"""
Generation Step
LLM-based response generation
"""

import re
from typing import List, Dict, Any, Optional, AsyncIterator

from app.core.logging import get_logger
from .base import BaseStep
from ..pipeline import PipelineContext


logger = get_logger(__name__)


class GenerationStep(BaseStep):
    """
    LLM response generation step
    
    Generates a response based on retrieved documents and user query.
    """
    
    def __init__(
        self, 
        name: str = "Generation",
        enabled: bool = True,
        system_prompt: Optional[str] = None
    ):
        super().__init__(name, enabled)
        self.system_prompt = system_prompt or self._default_system_prompt()
        self._llm_provider = None
    
    def _default_system_prompt(self) -> str:
        """Get the default system prompt"""
        return """你是一个专业的 Kubernetes 助手，基于提供的文档内容回答用户问题。

请严格遵循以下原则：
1. 只能基于提供的文档内容回答问题，绝对不能使用你自身的知识或经验
2. 如果文档中没有相关信息来回答用户的问题，你必须明确回复："很抱歉，文档里没有找到相关信息"
3. 不要尝试推测、推理或提供任何不在文档中的信息
4. 回答要准确、简洁、实用，且必须来源于提供的文档
5. 如果涉及代码示例，请提供完整的 YAML 或命令行示例（仅限文档中存在的）
6. 使用中文回答，除非用户特别要求使用英文
7. 如果问题涉及多个方面，请分点说明（仅限文档中涵盖的方面）
8. 在回答末尾可以推荐相关的文档链接（如果文档中有提供）

重要提醒：当文档内容不足以回答用户问题时，请直接回复"很抱歉，文档里没有找到相关信息"，不要尝试补充任何其他信息。

请开始回答用户的问题。"""
    
    async def _get_llm_provider(self):
        """Get or create LLM provider"""
        if self._llm_provider is None:
            from shared.llm_providers import create_llm_provider
            self._llm_provider = create_llm_provider()
            await self._llm_provider.initialize()
        return self._llm_provider
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute response generation
        
        Args:
            context: Pipeline context with final_documents
            
        Returns:
            Context with generated_response populated
        """
        provider = await self._get_llm_provider()
        
        # Build context from documents
        context_text = self._build_context_text(context.final_documents)
        
        # Build user message
        user_message = self._build_user_message(context.query, context_text)
        
        # Build message list
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history if available
        if context.conversation_history:
            messages.extend(context.conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        response = await provider.generate(
            messages,
            temperature=context.temperature,
            max_tokens=context.max_tokens
        )
        
        context.generated_response = response.content
        context.metadata["generation"] = {
            "model": response.model,
            "usage": response.usage,
            "finish_reason": response.finish_reason
        }
        
        self.logger.info(f"Generated response of length {len(response.content)}")
        
        return context
    
    def _build_context_text(self, documents: List[Dict[str, Any]]) -> str:
        """Build context text from retrieved documents"""
        if not documents:
            return "没有找到相关的文档内容。"

        # Apply prompt-size controls to avoid LLM timeouts with huge top_k
        try:
            from shared.config import get_settings
            settings = get_settings()
            max_docs = int(getattr(settings, "RAG_GENERATION_MAX_DOCS", 10))
            max_chars_per_doc = int(getattr(settings, "RAG_GENERATION_MAX_CHARS_PER_DOC", 2000))
            max_total_chars = int(getattr(settings, "RAG_GENERATION_MAX_TOTAL_CHARS", 20000))
        except Exception:
            max_docs = 10
            max_chars_per_doc = 2000
            max_total_chars = 20000
        
        context_parts: List[str] = []
        total_chars = 0
        for i, doc in enumerate(documents[:max_docs], 1):
            file_path = doc.get("file_path", f"文档 {i}")
            content = doc.get("content", "")
            url = doc.get("url", "")
            score = doc.get("rerank_score", doc.get("rrf_score", doc.get("score", 0)))
            metadata = doc.get("metadata", {})
            
            # Restore code blocks if any
            restored_content = self._restore_code_blocks(content, metadata)
            if max_chars_per_doc > 0 and len(restored_content) > max_chars_per_doc:
                restored_content = restored_content[:max_chars_per_doc] + "\n...[内容已截断]"
            
            context_part = f"文档 {i}: {file_path}\n"
            context_part += f"相关度: {score:.3f}\n"
            if url:
                context_part += f"来源: {url}\n"
            context_part += f"内容: {restored_content}\n"
            context_part += "-" * 50
            
            # Best-effort total cap
            if max_total_chars > 0 and total_chars + len(context_part) > max_total_chars:
                remaining = max_total_chars - total_chars
                if remaining <= 0:
                    break
                context_part = context_part[:remaining] + "\n...[上下文已截断]"
                context_parts.append(context_part)
                total_chars = max_total_chars
                break

            context_parts.append(context_part)
            total_chars += len(context_part)
        
        return "\n\n".join(context_parts)
    
    def _restore_code_blocks(self, content: str, metadata: Dict[str, Any]) -> str:
        """Restore code blocks from metadata"""
        if not content or not metadata:
            return content
        
        code_blocks = metadata.get("code_blocks", [])
        if not code_blocks:
            return content
        
        restored_content = content
        code_marker_pattern = r'\[CODE_BLOCK:\s*([^\]]+)\]'
        
        def replace_code_marker(match):
            codeid = match.group(1)
            for code_block in code_blocks:
                if code_block.get("id") == codeid:
                    try:
                        code_path = code_block.get("path")
                        if code_path:
                            with open(code_path, "r", encoding="utf-8") as f:
                                code_content = f.read()
                            if code_content:
                                return f"\n```{code_content}\n```\n"
                    except Exception as e:
                        logger.error(f"Failed to read code block file: {e}")
                        return match.group(0)
            return match.group(0)
        
        restored_content = re.sub(code_marker_pattern, replace_code_marker, restored_content)
        return restored_content
    
    def _build_user_message(self, query: str, context_text: str) -> str:
        """Build the user message with context"""
        return f"""基于以下文档内容回答用户问题：

{context_text}

用户问题：{query}

请基于上述文档内容提供准确、详细的回答。"""


class StreamingGenerationStep(GenerationStep):
    """
    Streaming LLM response generation step
    
    Same as GenerationStep but supports streaming output.
    """
    
    def __init__(
        self, 
        name: str = "StreamingGeneration",
        enabled: bool = True,
        system_prompt: Optional[str] = None
    ):
        super().__init__(name, enabled, system_prompt)
    
    async def generate_stream(
        self, 
        context: PipelineContext
    ) -> AsyncIterator[str]:
        """
        Generate streaming response
        
        Args:
            context: Pipeline context with final_documents
            
        Yields:
            Response chunks
        """
        provider = await self._get_llm_provider()
        
        # Build context and messages
        context_text = self._build_context_text(context.final_documents)
        user_message = self._build_user_message(context.query, context_text)
        
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if context.conversation_history:
            messages.extend(context.conversation_history)
        
        messages.append({"role": "user", "content": user_message})
        
        # Stream response
        async for chunk in provider.generate_stream(
            messages,
            temperature=context.temperature,
            max_tokens=context.max_tokens
        ):
            yield chunk

