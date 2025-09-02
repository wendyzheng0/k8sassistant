"""
聊天 API 端点
"""

import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.models.chat import (
    ChatRequest, ChatResponse, ChatHistoryResponse,
    Conversation, ChatMessage, MessageRole, MessageType
)
from app.services.milvus_service import MilvusService
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger("ChatAPI")


def get_milvus_service(request: Request) -> MilvusService:
    """获取 Milvus 服务实例"""
    return request.app.state.milvus_service


def get_embedding_service() -> EmbeddingService:
    """获取嵌入服务实例"""
    return EmbeddingService()


def get_llm_service() -> LLMService:
    """获取 LLM 服务实例"""
    return LLMService()


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    milvus_service: MilvusService = Depends(get_milvus_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    发送聊天消息并获取回复
    """
    try:
        logger.info(f"💬 收到聊天请求: {request.message[:50]}...")
        
        # 生成对话ID（如果未提供）
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # 1. 将用户查询编码为向量
        query_embedding = embedding_service.encode(request.message)[0]
        
        # 2. 在向量数据库中搜索相似文档
        similar_docs = await milvus_service.search_similar(
            query_embedding=query_embedding,
            top_k=5
        )
        
        logger.info(f"🔍 找到 {len(similar_docs)} 个相关文档")
        
        # 3. 构建对话历史（如果提供）
        conversation_history = None
        if request.context:
            conversation_history = [
                {"role": msg.role.value, "content": msg.content}
                for msg in request.context
            ]
        
        # 4. 使用 LLM 生成回复
        response_content = await llm_service.generate_rag_response(
            query=request.message,
            context_docs=similar_docs,
            conversation_history=conversation_history
        )
        
        # 5. 构建响应
        response = ChatResponse(
            message_id=str(uuid.uuid4()),
            content=response_content,
            conversation_id=conversation_id,
            timestamp=datetime.utcnow(),
            sources=[
                {
                    "title": doc.get("metadata", {}).get("title", "未知文档"),
                    "url": doc.get("metadata", {}).get("url", ""),
                    "score": doc.get("score", 0)
                }
                for doc in similar_docs
            ]
        )
        
        logger.info(f"✅ 聊天回复生成完成，长度: {len(response_content)}")
        return response
        
    except Exception as e:
        logger.error(f"❌ 聊天处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"聊天处理失败: {str(e)}")


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    milvus_service: MilvusService = Depends(get_milvus_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    流式聊天回复
    """
    try:
        logger.info(f"💬 收到流式聊天请求: {request.message[:50]}...")
        
        # 生成对话ID（如果未提供）
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # 1. 将用户查询编码为向量
        query_embedding = embedding_service.encode(request.message)[0]
        
        # 2. 在向量数据库中搜索相似文档
        similar_docs = await milvus_service.search_similar(
            query_embedding=query_embedding,
            top_k=5
        )
        
        logger.info(f"🔍 找到 {len(similar_docs)} 个相关文档")
        
        # 3. 构建对话历史（如果提供）
        conversation_history = None
        if request.context:
            conversation_history = [
                {"role": msg.role.value, "content": msg.content}
                for msg in request.context
            ]
        
        # 4. 使用 LLM 生成流式回复
        async def generate_stream():
            try:
                # 构建系统提示
                system_prompt = llm_service._build_system_prompt()
                
                # 构建上下文
                context_text = llm_service._build_context_text(similar_docs)
                
                # 构建用户消息
                user_message = llm_service._build_user_message(request.message, context_text)
                
                # 构建消息列表
                messages = [{"role": "system", "content": system_prompt}]
                
                # 添加对话历史
                if conversation_history:
                    messages.extend(conversation_history)
                
                # 添加当前用户消息
                messages.append({"role": "user", "content": user_message})
                
                # 流式生成回复
                async for chunk in llm_service._generate_stream_response(
                    messages, request.temperature or 0.7, request.max_tokens or 4096
                ):
                    yield f"data: {chunk}\n\n"
                
                # 发送结束标记
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"❌ 流式生成失败: {e}")
                yield f"data: 错误: {str(e)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ 流式聊天处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"流式聊天处理失败: {str(e)}")


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    page: int = 1,
    size: int = 20,
    conversation_id: Optional[str] = None
):
    """
    获取聊天历史
    """
    try:
        # TODO: 实现从数据库获取聊天历史的逻辑
        # 这里暂时返回空结果
        logger.info(f"📚 获取聊天历史，页码: {page}, 大小: {size}")
        
        return ChatHistoryResponse(
            conversations=[],
            total=0,
            page=page,
            size=size
        )
        
    except Exception as e:
        logger.error(f"❌ 获取聊天历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取聊天历史失败: {str(e)}")


@router.delete("/history/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    删除指定对话
    """
    try:
        # TODO: 实现删除对话的逻辑
        logger.info(f"🗑️ 删除对话: {conversation_id}")
        
        return {"message": "对话删除成功", "conversation_id": conversation_id}
        
    except Exception as e:
        logger.error(f"❌ 删除对话失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除对话失败: {str(e)}")
