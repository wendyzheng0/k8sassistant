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
from app.services.complex_retrieval_service import ComplexRetrievalService
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger("ChatAPI")


def get_milvus_service(request: Request) -> MilvusService:
    """获取 Milvus 服务实例"""
    return request.app.state.milvus_service


def get_embedding_service(request: Request) -> EmbeddingService:
    """获取嵌入服务实例"""
    return request.app.state.embedding_service


def get_llm_service(request: Request) -> LLMService:
    """获取 LLM 服务实例"""
    return request.app.state.llm_service


def get_complex_retrieval_service(request: Request) -> ComplexRetrievalService:
    """获取复杂检索服务实例"""
    return request.app.state.complex_retrieval_service



@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    complex_retrieval_service: ComplexRetrievalService = Depends(get_complex_retrieval_service),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    发送聊天消息并获取回复
    """
    try:
        logger.info(f"💬 Received chat request: {request.message[:50]}...")
        
        # 生成对话ID（如果未提供）
        conversation_id = request.conversation_id or str(uuid.uuid4())

        refined_query = await llm_service.generate_refine_query(
            request.message, 0.1, 1024)
        
        # 1. 使用复杂检索服务进行信息检索
        retrieval_result = await complex_retrieval_service.search(
            query=refined_query,
            top_k=10,
            milvus_weight=0.6,  # Milvus向量检索权重
            elasticsearch_weight=0.4,  # Elasticsearch关键字检索权重
            use_reranking=True  # 启用重排序
        )
        
        similar_docs = retrieval_result.documents
        logger.info(f"🔍 Found {len(similar_docs)} relevant documents using complex retrieval")
        logger.info(f"📊 Retrieval stats: {retrieval_result.metadata}")
        
        # 3. 构建对话历史（如果提供）
        conversation_history = None
        if request.context:
            conversation_history = [
                {"role": msg.role.value, "content": msg.content}
                for msg in request.context
            ]
        
        # 4. 使用 LLM 生成回复
        try:
            response_content = await llm_service.generate_rag_response(
                query=request.message,
                context_docs=similar_docs,
                conversation_history=conversation_history
            )
        except Exception as llm_error:
            logger.error(f"❌ LLM service failed: {llm_error}")
            # 如果LLM失败，返回错误信息
            response_content = f"抱歉，AI服务暂时不可用。错误信息：{str(llm_error)}"
        
        # 5. 构建响应
        response = ChatResponse(
            message_id=str(uuid.uuid4()),
            content=response_content,
            conversation_id=conversation_id,
            timestamp=datetime.utcnow(),
            sources=[
                {
                    "title": doc.get("file_path", "unknown"),
                    "doc_id": doc.get("id", "unknown"),
                    "content": doc.get("content", "unknown"),
                    "url": doc.get("metadata", {}).get("url", ""),
                    "score": doc.get("combined_score", doc.get("rerank_score", 0))
                }
                for doc in similar_docs
            ]
        )
        
        logger.info(f"✅ Chat response generated, length: {len(response_content)}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Chat processing failed: {e}")
        # 返回更友好的错误信息而不是抛出异常
        return ChatResponse(
            message_id=str(uuid.uuid4()),
            content=f"抱歉，处理您的请求时出现错误。请稍后重试。错误信息：{str(e)}",
            conversation_id=request.conversation_id or str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            sources=[],
            metadata={'error': True, 'error_message': str(e)}
        )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    complex_retrieval_service: ComplexRetrievalService = Depends(get_complex_retrieval_service),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    流式聊天回复
    """
    try:
        logger.info(f"💬 Received streaming chat request: {request.message[:50]}...")
        
        # 生成对话ID（如果未提供）
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # 优化查询以提高检索质量
        refined_query = await llm_service.generate_refine_query(
            request.message, 0.1, 1024)
        
        # 1. 使用复杂检索服务进行信息检索
        retrieval_result = await complex_retrieval_service.search(
            query=refined_query,
            top_k=10,
            milvus_weight=0.6,  # Milvus向量检索权重
            elasticsearch_weight=0.4,  # Elasticsearch关键字检索权重
            use_reranking=True  # 启用重排序
        )
        
        similar_docs = retrieval_result.documents
        logger.info(f"🔍 Found {len(similar_docs)} relevant documents using complex retrieval")
        logger.info(f"📊 Retrieval stats: {retrieval_result.metadata}")
        
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
                logger.error(f"❌ Streaming generation failed: {e}")
                yield f"data: Error: {str(e)}\n\n"
        
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
        logger.error(f"❌ Streaming chat processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming chat processing failed: {str(e)}")


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
        logger.info(f"📚 Getting chat history, page: {page}, size: {size}")
        
        return ChatHistoryResponse(
            conversations=[],
            total=0,
            page=page,
            size=size
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chat history: {str(e)}")


@router.delete("/history/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    删除指定对话
    """
    try:
        # TODO: 实现删除对话的逻辑
        logger.info(f"🗑️ Deleting conversation: {conversation_id}")
        
        return {"message": "Conversation deleted successfully", "conversation_id": conversation_id}
        
    except Exception as e:
        logger.error(f"❌ Failed to delete conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")
