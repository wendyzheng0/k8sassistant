"""
Query Rewrite Step
Refines the user query for better retrieval results
"""

from typing import Optional

from app.core.logging import get_logger
from .base import BaseStep
from ..pipeline import PipelineContext
import json


logger = get_logger(__name__)


class QueryRewriteStep(BaseStep):
    """
    Query rewriting step
    
    Uses LLM to refine the user's query for better retrieval.
    The refined query is stored in context.refined_query.
    """
    
    def __init__(
        self, 
        name: str = "QueryRewrite",
        enabled: bool = True,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ):
        super().__init__(name, enabled)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm_provider = None
    
    async def _get_llm_provider(self):
        """Get or create LLM provider"""
        if self._llm_provider is None:
            from shared.llm_providers import create_llm_provider
            self._llm_provider = create_llm_provider()
            await self._llm_provider.initialize()
        return self._llm_provider
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Rewrite the query for better retrieval
        
        Args:
            context: Pipeline context with the original query
            
        Returns:
            Context with refined_query set
        """
        provider = await self._get_llm_provider()
        
        # Build the query rewrite prompt
        prompt = self._build_rewrite_prompt(context.query)
        
        messages = [{"role": "user", "content": prompt}]
        
        response = await provider.generate(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        refined_query = response.content.strip()
        
        self.logger.info(f"Original query: {context.query}")
        self.logger.info(f"Refined query: {refined_query}")
        refined_query = json.loads(refined_query)
        
        context.refined_query = refined_query
        context.metadata["query_rewrite"] = {
            "original": context.query,
            "refined": refined_query
        }
        
        return context
    
    def _build_rewrite_prompt(self, query: str) -> str:
        """Build the query rewrite prompt"""
        return f"""# 角色与任务
你是专注于Kubernetes技术文档检索的查询优化专家。你的唯一任务是：接收用户原始查询，输出一个JSON格式的结果，**无任何其他文本、解释或说明**。

# 输入
原始查询: {query}

# 重写规则（必须严格遵守）
1.  保留原始查询核心意图，不增删、不歪曲需求。
2.  使用Kubernetes领域标准术语（如Pod、Deployment、HPA、etcd、Namespace等）替代模糊表述。
3.  语义搜索查询需为**完整通顺的字符串**，适合语义检索场景；关键词列表需提取**3-5个核心检索词**，适合关键词匹配场景。

# 输出格式（严格遵循，不得修改字段名和格式）
{{
    "semantic_search_query": "优化后的语义搜索完整查询语句",
    "keyword_list": ["核心关键词1", "核心关键词2", "核心关键词3"]
}}
"""


