"""
大语言模型服务
Wraps the shared LLM provider for backward compatibility
"""

import re
from typing import List, Dict, Any, Optional
import logging

# Import from shared module
from shared.llm_providers import create_llm_provider, BaseLLMProvider


logger = logging.getLogger(__name__)


class LLMService:
    """
    大语言模型服务类
    
    This class wraps the shared LLM provider for backward compatibility
    with existing code that uses LLMService directly.
    """
    
    def __init__(self, provider_type: Optional[str] = None):
        """
        Initialize LLM service
        
        Args:
            provider_type: Optional provider type override
        """
        self.logger = logging.getLogger("LLMService")
        self._provider: Optional[BaseLLMProvider] = None
        self._provider_type = provider_type
        self._initialized = False
    
    async def initialize(self):
        """Initialize the LLM provider"""
        if self._initialized:
            return
        
        self._provider = create_llm_provider(self._provider_type)
        await self._provider.initialize()
        self._initialized = True
        self.logger.info("✅ LLM service initialized")
    
    async def _ensure_initialized(self):
        """Ensure the service is initialized"""
        if not self._initialized:
            await self.initialize()
    
    async def generate_refine_query(
        self,
        message: str,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate refined query for better retrieval
        
        Args:
            message: Original query
            temperature: Generation temperature
            max_tokens: Max tokens for response
            
        Returns:
            Refined query
        """
        await self._ensure_initialized()
        
        query_rewrite_template = """
你是一个查询优化专家。文档库里面包含了Kubernetes的技术文档。用户的原始查询可能表达不清楚或不适合检索,
还有可能是一个复杂的问题，请将用户的查询重写为一个更清晰、更具体、更适合搜索的查询。

原始查询: {original_query}

重写后的查询应该:
1. 明确具体，避免模糊表达
2. 包含关键术语和概念
3. 适合语义搜索
4. 保持原始意图不变

重写后的查询:
        """
        
        messages = [{"role": "user", "content": query_rewrite_template.format(original_query=message)}]
        
        response = await self._provider.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        content = response.content
        self.logger.info(f"Original query: {message}")
        self.logger.info(f"Rewrite query: {content}")
        return content
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ):
        """
        Generate response
        
        Args:
            messages: Message list
            temperature: Generation temperature
            max_tokens: Max token count
            stream: Whether to stream output
            
        Returns:
            Generated response text or async generator for streaming
        """
        await self._ensure_initialized()
        
        if stream:
            return self._provider.generate_stream(messages, temperature, max_tokens)
        else:
            response = await self._provider.generate(messages, temperature, max_tokens)
            return response.content
    
    async def generate_rag_response(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate RAG-based response
        
        Args:
            query: User query
            context_docs: Retrieved documents
            conversation_history: Conversation history
            
        Returns:
            Generated response
        """
        await self._ensure_initialized()
        
        # Build system prompt
        system_prompt = self._build_system_prompt()
        
        # Build context
        context_text = self._build_context_text(context_docs)
        
        # Build user message
        user_message = self._build_user_message(query, context_text)
        
        # Build message list
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        response = await self._provider.generate(messages)
        
        return response.content
    
    def _build_system_prompt(self) -> str:
        """Build system prompt"""
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
    
    def _build_context_text(self, context_docs: List[Dict[str, Any]]) -> str:
        """Build context text with code block restoration"""
        if not context_docs:
            return "没有找到相关的文档内容。"
        
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            file_path = doc.get("file_path", f"文档 {i}")
            content = doc.get("content", "")
            url = doc.get("url", "")
            score = doc.get("score", 0)
            metadata = doc.get("metadata", {})
            
            # Restore code blocks
            restored_content = self._restore_code_blocks(content, metadata)
            
            context_part = f"文档 {i}: {file_path}\n"
            context_part += f"相关度: {score:.3f}\n"
            if url:
                context_part += f"来源: {url}\n"
            context_part += f"内容: {restored_content}\n"
            context_part += "-" * 50
            
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def _restore_code_blocks(self, content: str, metadata: Dict[str, Any]) -> str:
        """Restore code blocks in document"""
        if not content or not metadata:
            return content
        
        code_blocks = metadata.get('code_blocks', [])
        if not code_blocks:
            return content
        
        restored_content = content
        code_marker_pattern = r'\[CODE_BLOCK:\s*([^\]]+)\]'
        
        def replace_code_marker(match):
            codeid = match.group(1)
            for code_block in code_blocks:
                if code_block.get('id') == codeid:
                    try:
                        code_path = code_block.get('path')
                        if code_path:
                            with open(code_path, "r", encoding='utf-8') as f:
                                code_content = f.read()
                            if code_content:
                                return f"\n```{code_content}\n```\n"
                    except Exception as e:
                        self.logger.error(f"Failed to read code block file {code_path}: {e}")
                        return match.group(0)
            return match.group(0)
        
        restored_content = re.sub(code_marker_pattern, replace_code_marker, restored_content)
        return restored_content
    
    def _build_user_message(self, query: str, context_text: str) -> str:
        """Build user message"""
        return f"""基于以下文档内容回答用户问题：

{context_text}

用户问题：{query}

请基于上述文档内容提供准确、详细的回答。"""
    
    async def test_connection(self) -> bool:
        """Test LLM connection"""
        await self._ensure_initialized()
        return await self._provider.test_connection()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self._provider:
            return self._provider.get_model_info()
        return {}
    
    async def close(self):
        """Clean up resources"""
        if self._provider:
            await self._provider.close()
        self._initialized = False
