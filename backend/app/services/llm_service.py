"""
大语言模型服务
"""

import asyncio
from typing import List, Dict, Any, Optional
import openai
from app.core.config import settings
from app.core.logging import get_logger


class LLMService:
    """大语言模型服务类"""
    
    def __init__(self):
        self.logger = get_logger("LLMService")
        self.api_key = settings.LLM_API_KEY
        self.base_url = settings.LLM_BASE_URL
        self.model = settings.LLM_MODEL
        self.max_tokens = settings.MAX_TOKENS
        self.temperature = settings.TEMPERATURE
        
        # 初始化 OpenAI 客户端
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        生成回复
        
        Args:
            messages: 消息列表
            temperature: 生成温度
            max_tokens: 最大token数
            stream: 是否流式输出
            
        Returns:
            生成的回复文本
        """
        try:
            # 使用默认参数或传入的参数
            temp = temperature if temperature is not None else self.temperature
            max_toks = max_tokens if max_tokens is not None else self.max_tokens
            
            self.logger.info(f"🤖 开始生成回复，模型: {self.model}")
            
            if stream:
                return await self._generate_stream_response(messages, temp, max_toks)
            else:
                return await self._generate_sync_response(messages, temp, max_toks)
                
        except Exception as e:
            self.logger.error(f"❌ 生成回复失败: {e}")
            raise
    
    async def _generate_sync_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> str:
        """同步生成回复"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            self.logger.info(f"✅ 回复生成完成，长度: {len(content)}")
            return content
            
        except Exception as e:
            self.logger.error(f"❌ 同步生成回复失败: {e}")
            raise
    
    async def _generate_stream_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ):
        """流式生成回复"""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self.logger.error(f"❌ 流式生成回复失败: {e}")
            raise
    
    async def generate_rag_response(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        基于检索增强生成回复
        
        Args:
            query: 用户查询
            context_docs: 检索到的相关文档
            conversation_history: 对话历史
            
        Returns:
            生成的回复
        """
        try:
            # 构建系统提示
            system_prompt = self._build_system_prompt()
            
            # 构建上下文
            context_text = self._build_context_text(context_docs)
            
            # 构建用户消息
            user_message = self._build_user_message(query, context_text)
            
            # 构建消息列表
            messages = [{"role": "system", "content": system_prompt}]
            
            # 添加对话历史
            if conversation_history:
                messages.extend(conversation_history)
            
            # 添加当前用户消息
            messages.append({"role": "user", "content": user_message})
            
            # 生成回复
            response = await self.generate_response(messages)
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ RAG 回复生成失败: {e}")
            raise
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
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
        """构建上下文文本，恢复代码块"""
        if not context_docs:
            return "没有找到相关的文档内容。"
        
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            title = doc.get("title", f"文档 {i}")
            file_path = doc.get("file_path", f"文档 {i}")
            # doc_id = doc.get("doc_id", f"unknown")
            content = doc.get("content", "")
            url = doc.get("url", "")
            score = doc.get("score", 0)
            metadata = doc.get("metadata", {})
            
            # 恢复代码块
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
        """
        恢复文档中的代码块
        
        Args:
            content: 文档内容
            metadata: 文档元数据
            
        Returns:
            恢复代码块后的内容
        """
        if not content or not metadata:
            return content
        
        # 获取代码块信息
        code_blocks = metadata.get('code_blocks', [])
        if not code_blocks:
            return content
        
        restored_content = content
        
        # 查找并替换代码标记
        import re
        code_marker_pattern = r'\[代码示例:\s*([^\]]+)\]'
        
        def replace_code_marker(match):
            language = match.group(1)
            # 查找对应的代码块
            for code_block in code_blocks:
                if code_block.get('language') == language:
                    code_content = code_block.get('content', '')
                    if code_content:
                        return f"\n```{language}\n{code_content}\n```\n"
            return match.group(0)  # 如果找不到对应的代码块，保持原样
        
        restored_content = re.sub(code_marker_pattern, replace_code_marker, restored_content)
        
        return restored_content
    
    def _build_user_message(self, query: str, context_text: str) -> str:
        """构建用户消息"""
        return f"""基于以下文档内容回答用户问题：

{context_text}

用户问题：{query}

请基于上述文档内容提供准确、详细的回答。"""
    
    async def test_connection(self) -> bool:
        """测试 LLM 连接"""
        try:
            test_message = [{"role": "user", "content": "Hello"}]
            response = await self.generate_response(test_message, max_tokens=10)
            self.logger.info("✅ LLM 连接测试成功")
            return True
        except Exception as e:
            self.logger.error(f"❌ LLM 连接测试失败: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
