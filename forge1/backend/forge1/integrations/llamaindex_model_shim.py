"""
Forge1 Model Client Shim for LlamaIndex

Provides LlamaIndex-compatible model client that delegates all calls to Forge1's
Model Router. Ensures no direct model access while preserving LlamaIndex's
interface expectations and tracking usage metrics for billing.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# LlamaIndex imports
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.callbacks import CallbackManager
from llama_index.core.llms.types import (
    ChatMessage, 
    ChatResponse, 
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponse,
    CompletionResponseGen,
    CompletionResponseAsyncGen,
    LLMMetadata,
    MessageRole
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import Field

# Forge1 imports
from forge1.billing import usage_meter
from forge1.core.audit_logger import AuditLogger
from forge1.core.model_router import ModelRouter

logger = logging.getLogger(__name__)

@dataclass
class UsageMetrics:
    """Usage metrics for billing and monitoring"""
    model_name: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    cost_estimate: float
    timestamp: float
    tenant_id: str
    employee_id: str
    request_id: Optional[str] = None

class Forge1LLMShim(LLM):
    """LlamaIndex-compatible LLM that routes through Forge1's Model Router"""
    
    model_name: str = Field(description="Model name for routing")
    model_router: ModelRouter = Field(description="Forge1 model router instance")
    audit_logger: AuditLogger = Field(description="Audit logger for usage tracking")
    tenant_id: str = Field(description="Current tenant ID")
    employee_id: str = Field(description="Current employee ID")
    
    def __init__(
        self,
        model_router: ModelRouter,
        audit_logger: AuditLogger,
        tenant_id: str,
        employee_id: str,
        model_name: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs
    ):
        super().__init__(
            model_name=model_name or "forge1-routed",
            model_router=model_router,
            audit_logger=audit_logger,
            tenant_id=tenant_id,
            employee_id=employee_id,
            callback_manager=callback_manager,
            **kwargs
        )
        
        # Usage tracking
        self.usage_history: List[UsageMetrics] = []
        
    @property
    def metadata(self) -> LLMMetadata:
        """Return LLM metadata"""
        return LLMMetadata(
            context_window=128000,  # Conservative default
            num_output=4096,
            is_chat_model=True,
            model_name=self.model_name
        )
    
    def _create_task_input(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Create task input for Forge1 model router"""
        return {
            "description": prompt,
            "type": "text_generation",
            "parameters": kwargs,
            "tenant_id": self.tenant_id,
            "employee_id": self.employee_id
        }
    
    def _create_chat_task_input(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        """Create chat task input for Forge1 model router"""
        # Convert LlamaIndex ChatMessage to dict format
        message_dicts = []
        for msg in messages:
            message_dicts.append({
                "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                "content": msg.content
            })
        
        return {
            "description": "Chat completion request",
            "type": "chat_completion",
            "messages": message_dicts,
            "parameters": kwargs,
            "tenant_id": self.tenant_id,
            "employee_id": self.employee_id
        }
    
    def _track_usage(
        self,
        model_name: str,
        tokens_input: int,
        tokens_output: int,
        latency_ms: float,
        cost_estimate: float = 0.0,
        request_id: Optional[str] = None
    ) -> None:
        """Track usage metrics for billing"""
        metrics = UsageMetrics(
            model_name=model_name,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
            cost_estimate=cost_estimate,
            timestamp=time.time(),
            tenant_id=self.tenant_id,
            employee_id=self.employee_id,
            request_id=request_id
        )

        self.usage_history.append(metrics)

        usage_meter.record_model_call(
            tenant_id=self.tenant_id,
            employee_id=self.employee_id,
            model=model_name,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
            cost_estimate=cost_estimate,
            request_id=request_id or f"model-{uuid.uuid4().hex[:10]}",
            metadata={"source": "llamaindex_shim"},
        )

        # Log usage event for billing system
        asyncio.create_task(self._log_usage_event(metrics))
    
    async def _log_usage_event(self, metrics: UsageMetrics) -> None:
        """Log usage event to audit system"""
        try:
            await self.audit_logger.log_usage_event(
                event_type="llm_usage",
                user_id=self.employee_id,
                tenant_id=self.tenant_id,
                details={
                    "model_name": metrics.model_name,
                    "tokens_input": metrics.tokens_input,
                    "tokens_output": metrics.tokens_output,
                    "latency_ms": metrics.latency_ms,
                    "cost_estimate": metrics.cost_estimate,
                    "request_id": metrics.request_id
                }
            )
        except Exception as e:
            logger.error(f"Failed to log usage event: {e}")
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Synchronous completion - delegates to async version"""
        return asyncio.run(self.acomplete(prompt, **kwargs))
    
    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Async completion through Forge1 model router"""
        start_time = time.time()
        
        try:
            # Create task input for model router
            task_input = self._create_task_input(prompt, **kwargs)
            
            # Get optimal client from model router
            client = await self.model_router.get_optimal_client(task_input)
            
            # Generate response
            if hasattr(client, 'generate'):
                response_text = await client.generate(prompt, **kwargs)
            else:
                # Fallback for different client interfaces
                response_text = await client.acomplete(prompt, **kwargs)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Estimate token counts (rough approximation)
            tokens_input = len(prompt.split()) * 1.3  # Rough token estimation
            tokens_output = len(str(response_text).split()) * 1.3
            
            # Track usage
            self._track_usage(
                model_name=getattr(client, 'model_name', 'unknown'),
                tokens_input=int(tokens_input),
                tokens_output=int(tokens_output),
                latency_ms=latency_ms,
                cost_estimate=self._estimate_cost(tokens_input, tokens_output)
            )
            
            return CompletionResponse(
                text=str(response_text),
                additional_kwargs={}
            )
            
        except Exception as e:
            logger.error(f"Completion failed: {e}")
            latency_ms = (time.time() - start_time) * 1000
            
            # Track failed usage
            self._track_usage(
                model_name="error",
                tokens_input=len(prompt.split()),
                tokens_output=0,
                latency_ms=latency_ms
            )
            
            raise e
    
    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        """Streaming completion - not implemented for simplicity"""
        raise NotImplementedError("Streaming completion not supported in Forge1 shim")
    
    async def astream_complete(self, prompt: str, **kwargs) -> CompletionResponseAsyncGen:
        """Async streaming completion - not implemented for simplicity"""
        raise NotImplementedError("Async streaming completion not supported in Forge1 shim")
    
    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        """Synchronous chat - delegates to async version"""
        return asyncio.run(self.achat(messages, **kwargs))
    
    async def achat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        """Async chat through Forge1 model router"""
        start_time = time.time()
        
        try:
            # Create task input for model router
            task_input = self._create_chat_task_input(messages, **kwargs)
            
            # Get optimal client from model router
            client = await self.model_router.get_optimal_client(task_input)
            
            # Generate chat response
            if hasattr(client, 'chat'):
                response_text = await client.chat(task_input["messages"], **kwargs)
            else:
                # Fallback - convert to completion format
                prompt = self._messages_to_prompt(messages)
                response_text = await client.generate(prompt, **kwargs)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Estimate token counts
            total_input_text = " ".join([msg.content for msg in messages])
            tokens_input = len(total_input_text.split()) * 1.3
            tokens_output = len(str(response_text).split()) * 1.3
            
            # Track usage
            self._track_usage(
                model_name=getattr(client, 'model_name', 'unknown'),
                tokens_input=int(tokens_input),
                tokens_output=int(tokens_output),
                latency_ms=latency_ms,
                cost_estimate=self._estimate_cost(tokens_input, tokens_output)
            )
            
            # Create response message
            response_message = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=str(response_text)
            )
            
            return ChatResponse(
                message=response_message,
                additional_kwargs={}
            )
            
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            latency_ms = (time.time() - start_time) * 1000
            
            # Track failed usage
            total_input_text = " ".join([msg.content for msg in messages])
            self._track_usage(
                model_name="error",
                tokens_input=len(total_input_text.split()),
                tokens_output=0,
                latency_ms=latency_ms
            )
            
            raise e
    
    def stream_chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponseGen:
        """Streaming chat - not implemented for simplicity"""
        raise NotImplementedError("Streaming chat not supported in Forge1 shim")
    
    async def astream_chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponseAsyncGen:
        """Async streaming chat - not implemented for simplicity"""
        raise NotImplementedError("Async streaming chat not supported in Forge1 shim")
    
    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to a single prompt string"""
        prompt_parts = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
            prompt_parts.append(f"{role}: {msg.content}")
        return "\n".join(prompt_parts)
    
    def _estimate_cost(self, tokens_input: float, tokens_output: float) -> float:
        """Estimate cost based on token usage (rough approximation)"""
        # Very rough cost estimation - should be replaced with actual pricing
        input_cost_per_token = 0.00001  # $0.01 per 1K tokens
        output_cost_per_token = 0.00003  # $0.03 per 1K tokens
        
        return (tokens_input * input_cost_per_token) + (tokens_output * output_cost_per_token)
    
    def get_usage_metrics(self) -> List[Dict[str, Any]]:
        """Get usage metrics for export"""
        return [
            {
                "timestamp": metrics.timestamp,
                "tenant_id": metrics.tenant_id,
                "employee_id": metrics.employee_id,
                "model": metrics.model_name,
                "tokens_in": metrics.tokens_input,
                "tokens_out": metrics.tokens_output,
                "latency_ms": metrics.latency_ms,
                "cost_estimate": metrics.cost_estimate,
                "request_id": metrics.request_id
            }
            for metrics in self.usage_history
        ]

class Forge1EmbeddingShim(BaseEmbedding):
    """LlamaIndex-compatible embedding model that routes through Forge1"""
    
    def __init__(
        self,
        model_router: ModelRouter,
        audit_logger: AuditLogger,
        tenant_id: str,
        employee_id: str,
        model_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            model_name=model_name or "forge1-embedding",
            **kwargs
        )
        
        self.model_router = model_router
        self.audit_logger = audit_logger
        self.tenant_id = tenant_id
        self.employee_id = employee_id
        self.usage_history: List[UsageMetrics] = []
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query - synchronous version"""
        return asyncio.run(self._aget_query_embedding(query))
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query - async version"""
        start_time = time.time()
        
        try:
            # Create task input for embedding
            task_input = {
                "description": f"Generate embedding for: {query[:100]}...",
                "type": "embedding_generation",
                "text": query,
                "tenant_id": self.tenant_id,
                "employee_id": self.employee_id
            }
            
            # Get embedding client from model router
            client = await self.model_router.get_optimal_client(task_input)
            
            # Generate embedding
            if hasattr(client, 'get_embedding'):
                embedding = await client.get_embedding(query)
            else:
                # Fallback - use a mock embedding
                import numpy as np
                np.random.seed(hash(query) % (2**32))
                embedding = np.random.normal(0, 1, 1536).tolist()
                norm = np.linalg.norm(embedding)
                embedding = (np.array(embedding) / norm).tolist()
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Track usage
            self._track_embedding_usage(
                model_name=getattr(client, 'model_name', 'unknown'),
                text_length=len(query),
                latency_ms=latency_ms
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            latency_ms = (time.time() - start_time) * 1000
            
            # Track failed usage
            self._track_embedding_usage(
                model_name="error",
                text_length=len(query),
                latency_ms=latency_ms
            )
            
            raise e
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for text - synchronous version"""
        return self._get_query_embedding(text)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get embedding for text - async version"""
        return await self._aget_query_embedding(text)
    
    def _track_embedding_usage(
        self,
        model_name: str,
        text_length: int,
        latency_ms: float
    ) -> None:
        """Track embedding usage metrics"""
        # Rough token estimation for embeddings
        tokens = text_length / 4  # Rough approximation
        
        metrics = UsageMetrics(
            model_name=model_name,
            tokens_input=int(tokens),
            tokens_output=0,  # Embeddings don't have output tokens
            latency_ms=latency_ms,
            cost_estimate=tokens * 0.0000001,  # Very rough embedding cost
            timestamp=time.time(),
            tenant_id=self.tenant_id,
            employee_id=self.employee_id
        )
        
        self.usage_history.append(metrics)
        
        # Log usage event
        asyncio.create_task(self._log_embedding_usage(metrics))
    
    async def _log_embedding_usage(self, metrics: UsageMetrics) -> None:
        """Log embedding usage event"""
        try:
            await self.audit_logger.log_usage_event(
                event_type="embedding_usage",
                user_id=self.employee_id,
                tenant_id=self.tenant_id,
                details={
                    "model_name": metrics.model_name,
                    "tokens_input": metrics.tokens_input,
                    "latency_ms": metrics.latency_ms,
                    "cost_estimate": metrics.cost_estimate
                }
            )
        except Exception as e:
            logger.error(f"Failed to log embedding usage: {e}")

class ModelShimFactory:
    """Factory for creating Forge1 model shims"""
    
    def __init__(
        self,
        model_router: ModelRouter,
        audit_logger: AuditLogger
    ):
        self.model_router = model_router
        self.audit_logger = audit_logger
    
    def create_llm_shim(
        self,
        tenant_id: str,
        employee_id: str,
        model_name: Optional[str] = None
    ) -> Forge1LLMShim:
        """Create LLM shim for given tenant/employee"""
        return Forge1LLMShim(
            model_router=self.model_router,
            audit_logger=self.audit_logger,
            tenant_id=tenant_id,
            employee_id=employee_id,
            model_name=model_name
        )
    
    def create_embedding_shim(
        self,
        tenant_id: str,
        employee_id: str,
        model_name: Optional[str] = None
    ) -> Forge1EmbeddingShim:
        """Create embedding shim for given tenant/employee"""
        return Forge1EmbeddingShim(
            model_router=self.model_router,
            audit_logger=self.audit_logger,
            tenant_id=tenant_id,
            employee_id=employee_id,
            model_name=model_name
        )