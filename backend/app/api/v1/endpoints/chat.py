"""
聊天 API 端点
"""

import uuid
from datetime import datetime
from typing import List, Optional
import json
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
from app.rag import RAGPipeline, PipelineContext
from app.rag.steps import (
    QueryRewriteStep,
    HybridRetrievalStep,
    RRFRerankStep,
    CrossEncoderRerankStep,
    GenerationStep,
    StreamingGenerationStep
)
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger("ChatAPI")


# Global RAG pipeline instance
_rag_pipeline: Optional[RAGPipeline] = None
_streaming_generation_step: Optional[StreamingGenerationStep] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = (RAGPipeline("K8sAssistantPipeline")
            .add_step(QueryRewriteStep())
            .add_step(HybridRetrievalStep())
            .add_step(RRFRerankStep())
            .add_step(CrossEncoderRerankStep())
            .add_step(GenerationStep()))
    return _rag_pipeline


def get_streaming_generation_step() -> StreamingGenerationStep:
    """Get the streaming generation step"""
    global _streaming_generation_step
    if _streaming_generation_step is None:
        _streaming_generation_step = StreamingGenerationStep()
    return _streaming_generation_step


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
async def chat(request: ChatRequest):
    """
    发送聊天消息并获取回复
    Uses the new RAG pipeline for processing
    """
    try:
        logger.info(f"💬 Received chat request: {request.message[:50]}...")
        
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Build conversation history
        conversation_history = None
        if request.context:
            conversation_history = [
                {"role": msg.role.value, "content": msg.content}
                for msg in request.context
            ]
        
        # Create pipeline context
        context = PipelineContext(
            query=request.message,
            conversation_id=conversation_id,
            conversation_history=conversation_history,
            top_k=10,
            use_reranking=True,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 4096
        )
        
        # Run the RAG pipeline
        pipeline = get_rag_pipeline()
        logger.info(f"Start to run pipeline: {pipeline.get_pipeline_info()}")
        result_context = await pipeline.run(context)
        
        # Build response
        response = ChatResponse(
            message_id=str(uuid.uuid4()),
            content=result_context.generated_response or "抱歉，无法生成回复。",
            conversation_id=conversation_id,
            timestamp=datetime.utcnow(),
            sources=[
                {
                    "title": doc.get("file_path", "unknown"),
                    "doc_id": doc.get("id", "unknown"),
                    "content": doc.get("content", ""),
                    "url": doc.get("metadata", {}).get("url", ""),
                    "score": doc.get("rerank_score", doc.get("rrf_score", 0))
                }
                for doc in result_context.final_documents
            ],
            metadata={
                "pipeline_info": result_context.metadata,
                "steps_executed": list(result_context.step_results.keys())
            }
        )
        
        logger.info(f"✅ Chat response generated, length: {len(response.content)}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Chat processing failed: {e}")
        return ChatResponse(
            message_id=str(uuid.uuid4()),
            content=f"抱歉，处理您的请求时出现错误。请稍后重试。错误信息：{str(e)}",
            conversation_id=request.conversation_id or str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            sources=[],
            metadata={'error': True, 'error_message': str(e)}
        )


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    流式聊天回复
    Uses the new RAG pipeline with streaming generation
    """
    try:
        logger.info(f"💬 Received streaming chat request: {request.message[:50]}...")
        
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Build conversation history
        conversation_history = None
        if request.context:
            conversation_history = [
                {"role": msg.role.value, "content": msg.content}
                for msg in request.context
            ]
        
        # Create pipeline context
        context = PipelineContext(
            query=request.message,
            conversation_id=conversation_id,
            conversation_history=conversation_history,
            top_k=10,
            use_reranking=True,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 4096
        )
        
        # Run pipeline steps except generation
        pipeline = RAGPipeline("StreamingPipeline")
        pipeline.add_step(QueryRewriteStep())
        pipeline.add_step(HybridRetrievalStep())
        pipeline.add_step(RRFRerankStep())
        pipeline.add_step(CrossEncoderRerankStep())
        # Don't add generation step - we'll do streaming generation
        
        result_context = await pipeline.run(context)
        
        # Streaming generation
        async def generate_stream():
            try:
                streaming_step = get_streaming_generation_step()
                
                # Send meta first (conversation_id + sources) so the UI can render references immediately
                sources = [
                    {
                        "title": doc.get("file_path", "unknown"),
                        "doc_id": doc.get("id", "unknown"),
                        "content": doc.get("content", ""),
                        "url": doc.get("metadata", {}).get("url", ""),
                        "score": doc.get("rerank_score", doc.get("rrf_score", 0))
                    }
                    for doc in result_context.final_documents
                ]
                meta_payload = {
                    "conversation_id": result_context.conversation_id,
                    "sources": sources
                }
                yield "event: meta\n" + f"data: {json.dumps(meta_payload, ensure_ascii=False)}\n\n"
                
                # Stream response deltas
                async for chunk in streaming_step.generate_stream(result_context):
                    delta_payload = {"delta": chunk}
                    yield "event: delta\n" + f"data: {json.dumps(delta_payload, ensure_ascii=False)}\n\n"
                
                # Send end marker
                yield "event: done\ndata: {}\n\n"
                
            except Exception as e:
                logger.error(f"❌ Streaming generation failed: {e}")
                err_payload = {"error": str(e)}
                yield "event: error\n" + f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                # Best-effort: disable proxy buffering (e.g. nginx)
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Streaming chat processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming chat processing failed: {str(e)}")


@router.post("/legacy", response_model=ChatResponse)
async def chat_legacy(
    request: ChatRequest,
    complex_retrieval_service: ComplexRetrievalService = Depends(get_complex_retrieval_service),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Legacy chat endpoint (using old implementation)
    Kept for backward compatibility during migration
    """
    try:
        logger.info(f"💬 Received legacy chat request: {request.message[:50]}...")
        
        conversation_id = request.conversation_id or str(uuid.uuid4())

        refined_query = await llm_service.generate_refine_query(
            request.message, 0.1, 1024)
        
        retrieval_result = await complex_retrieval_service.search(
            query=refined_query,
            top_k=10,
            milvus_weight=0.6,
            elasticsearch_weight=0.4,
            use_reranking=True
        )
        
        similar_docs = retrieval_result.documents
        logger.info(f"🔍 Found {len(similar_docs)} relevant documents using complex retrieval")
        
        conversation_history = None
        if request.context:
            conversation_history = [
                {"role": msg.role.value, "content": msg.content}
                for msg in request.context
            ]
        
        try:
            response_content = await llm_service.generate_rag_response(
                query=request.message,
                context_docs=similar_docs,
                conversation_history=conversation_history
            )
        except Exception as llm_error:
            logger.error(f"❌ LLM service failed: {llm_error}")
            response_content = f"抱歉，AI服务暂时不可用。错误信息：{str(llm_error)}"
        
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
        
        logger.info(f"✅ Legacy chat response generated, length: {len(response_content)}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Legacy chat processing failed: {e}")
        return ChatResponse(
            message_id=str(uuid.uuid4()),
            content=f"抱歉，处理您的请求时出现错误。请稍后重试。错误信息：{str(e)}",
            conversation_id=request.conversation_id or str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            sources=[],
            metadata={'error': True, 'error_message': str(e)}
        )


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
        logger.info(f"🗑️ Deleting conversation: {conversation_id}")
        
        return {"message": "Conversation deleted successfully", "conversation_id": conversation_id}
        
    except Exception as e:
        logger.error(f"❌ Failed to delete conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")


@router.get("/pipeline/info")
async def get_pipeline_info():
    """
    Get information about the current RAG pipeline configuration
    """
    pipeline = get_rag_pipeline()
    return pipeline.get_pipeline_info()
