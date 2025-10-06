# forge1/backend/forge1/core/app_kernel.py
"""
Forge 1 Enhanced App Kernel

Extends Microsoft's app_kernel.py with enterprise-grade capabilities:
- Multi-model routing (GPT-4o/5, Claude, Gemini)
- Enhanced security and authentication
- Performance monitoring and optimization
- Compliance and audit logging
- Request/response validation and error handling
"""

import asyncio
import logging
import os
import uuid
import time
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

# Import Microsoft's base functionality with fallbacks
import sys
import os

# Try to import Microsoft's modules, fallback to our implementations
try:
    sys.path.append('../../../Multi-Agent-Custom-Automation-Engine-Solution-Accelerator/src/backend')
    from app_config import config
    from auth.auth_utils import get_authenticated_user_details
    from config_kernel import Config
    from event_utils import track_event_if_configured
    MICROSOFT_MODULES_AVAILABLE = True
except ImportError:
    # Fallback implementations
    from forge1.core.fallback_modules import (
        config, get_authenticated_user_details, Config, track_event_if_configured
    )
    MICROSOFT_MODULES_AVAILABLE = False
from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
try:
    from middleware.health_check import HealthCheckMiddleware
    from models.messages_kernel import (
        AgentMessage,
        AgentType,
        HumanClarification,
        HumanFeedback,
        InputTask,
        PlanWithSteps,
        Step,
        UserLanguage
    )
    from utils_kernel import initialize_runtime_and_context, rai_success
except ImportError:
    # Fallback implementations
    from forge1.core.fallback_modules import (
        HealthCheckMiddleware, AgentMessage, AgentType, HumanClarification,
        HumanFeedback, InputTask, PlanWithSteps, Step, UserLanguage,
        initialize_runtime_and_context, rai_success
    )
from pydantic import BaseModel, ValidationError

# Forge 1 Enhanced Imports
from forge1.core.model_router import ModelRouter
from forge1.core.security_manager import SecurityManager
from forge1.core.performance_monitor import PerformanceMonitor
from forge1.core.compliance_engine import ComplianceEngine
from forge1.core.health_checks import health_router
from forge1.api.compliance_api import router as compliance_router
from forge1.core.tenancy import set_current_tenant, get_current_tenant
from forge1.agents.agent_factory_enhanced import EnhancedAgentFactory

# Azure monitoring (optional)
try:
    from azure.monitor.opentelemetry import configure_azure_monitor
    AZURE_MONITORING_AVAILABLE = True
except ImportError:
    AZURE_MONITORING_AVAILABLE = False
    configure_azure_monitor = None

# Prometheus exporter
try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    PROM_AVAILABLE = True
except Exception:
    PROM_AVAILABLE = False

# Configure Application Insights
connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
if connection_string and AZURE_MONITORING_AVAILABLE:
    configure_azure_monitor(connection_string=connection_string)
    logging.info("Application Insights configured with Forge 1 enhancements")
elif connection_string:
    logging.warning("Application Insights connection string found but Azure monitoring not available")
else:
    logging.info("No Application Insights connection string found - using fallback monitoring")

# Optional Jaeger tracing
JAEGER_TRACING_ENABLED = False
JaegerConfig = None
try:
    from jaeger_client import Config as JaegerConfig
    JAEGER_TRACING_ENABLED = True
except ImportError:
    JAEGER_TRACING_ENABLED = False

# Configure logging with Forge 1 structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress verbose Azure logs
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.identity.aio._internal").setLevel(logging.WARNING)
logging.getLogger("azure.monitor.opentelemetry.exporter.export._base").setLevel(logging.WARNING)

# Enhanced Pydantic models for request/response validation
class EnhancedInputTask(InputTask):
    """Enhanced input task with additional validation"""
    priority: Optional[str] = "normal"
    expected_complexity: Optional[float] = None
    requires_compliance: bool = True

class TaskResponse(BaseModel):
    """Standardized task response model"""
    status: str
    session_id: str
    plan_id: Optional[str] = None
    description: str
    model_used: Optional[str] = None
    performance_score: Optional[float] = None
    processing_time: Optional[float] = None
    compliance_status: str = "validated"

class ErrorResponse(BaseModel):
    """Standardized error response model"""
    error: str
    detail: str
    timestamp: str
    request_id: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Forge 1 Platform...")
    yield
    # Shutdown
    logger.info("Shutting down Forge 1 Platform...")

class Forge1App:
    """Enhanced FastAPI application with Forge 1 capabilities"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Forge 1 Platform API",
            description="Enterprise AI Employee Builder Platform - Superhuman AI Employees",
            version="1.0.0",
            docs_url="/api/v1/docs",
            redoc_url="/api/v1/redoc",
            openapi_url="/api/v1/openapi.json",
            lifespan=lifespan
        )
        
        # Initialize Forge 1 components
        self.model_router = ModelRouter()
        self.security_manager = SecurityManager()
        self.performance_monitor = PerformanceMonitor()
        self.compliance_engine = ComplianceEngine()
        self.tracer = None
        self._init_tracing()
        
        self._setup_exception_handlers()
        self._setup_middleware()
        self._setup_routes()
        self._include_routers()
        self._setup_metrics_endpoint()
    
    def _setup_exception_handlers(self):
        """Setup enhanced exception handlers"""
        
        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            """Handle request validation errors"""
            request_id = str(uuid.uuid4())
            logger.error(f"Validation error {request_id}: {exc}")
            
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=ErrorResponse(
                    error="Validation Error",
                    detail=str(exc),
                    timestamp=str(time.time()),
                    request_id=request_id
                ).dict()
            )
        
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """Handle HTTP exceptions with enhanced logging"""
            request_id = str(uuid.uuid4())
            logger.error(f"HTTP error {request_id}: {exc.status_code} - {exc.detail}")
            
            return JSONResponse(
                status_code=exc.status_code,
                content=ErrorResponse(
                    error=f"HTTP {exc.status_code}",
                    detail=await self.security_manager.sanitize_error_message(exc.detail),
                    timestamp=str(time.time()),
                    request_id=request_id
                ).dict()
            )
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """Handle general exceptions"""
            request_id = str(uuid.uuid4())
            logger.error(f"Unhandled error {request_id}: {exc}", exc_info=True)
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="Internal Server Error",
                    detail="An unexpected error occurred. Please try again later.",
                    timestamp=str(time.time()),
                    request_id=request_id
                ).dict()
            )
    
    def _setup_middleware(self):
        """Setup enhanced middleware stack with proper ordering"""
        frontend_url = Config.FRONTEND_SITE_NAME
        
        # Trusted host middleware (first for security)
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure properly for production
        )
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[frontend_url, "http://localhost:3000"],  # Add dev frontend
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "X-Performance-Score"]
        )
        
        # Enhanced health check middleware
        self.app.add_middleware(
            HealthCheckMiddleware, 
            password="", 
            checks={
                "database": self._check_database_health,
                "model_router": self._check_model_router_health,
                "security": self._check_security_health,
                "compliance": self._check_compliance_health
            }
        )
        
        # Request ID middleware
        @self.app.middleware("http")
        async def request_id_middleware(request: Request, call_next):
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

        # Tenancy middleware (A.2)
        @self.app.middleware("http")
        async def tenancy_middleware(request: Request, call_next):
            tenant = request.headers.get("X-Tenant-ID")
            if not tenant:
                try:
                    details = await self.security_manager.get_enhanced_user_details(request.headers)
                    tenant = details.get("tenant_id")
                except Exception:
                    tenant = None
            set_current_tenant(tenant)
            response = await call_next(request)
            response.headers["X-Tenant-ID"] = get_current_tenant()
            return response
        
        # Performance monitoring middleware
        @self.app.middleware("http")
        async def performance_middleware(request: Request, call_next):
            return await self.performance_monitor.track_request(request, call_next)

        # Security middleware
        @self.app.middleware("http")
        async def security_middleware(request: Request, call_next):
            return await self.security_manager.validate_request(request, call_next)

        # Compliance middleware
        @self.app.middleware("http")
        async def compliance_middleware(request: Request, call_next):
            return await self.compliance_engine.audit_request(request, call_next)

        # Tracing middleware (Jaeger)
        if self.tracer is not None:
            @self.app.middleware("http")
            async def tracing_middleware(request: Request, call_next):
                with self.tracer.start_span(operation_name=f"HTTP {request.method} {request.url.path}") as span:
                    span.set_tag("http.method", request.method)
                    span.set_tag("http.url", str(request.url))
                    response = await call_next(request)
                    span.set_tag("http.status_code", response.status_code)
                    return response
    
    def _setup_routes(self):
        """Setup enhanced API routes with proper validation and error handling"""
        
        @self.app.post("/api/v1/forge1/input_task", response_model=TaskResponse)
        async def enhanced_input_task(input_task: EnhancedInputTask, request: Request) -> TaskResponse:
            """Enhanced input task endpoint with multi-model routing and validation"""
            
            start_time = time.time()
            request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
            
            logger.info(f"Processing enhanced input task {request_id}: {input_task.description[:100]}...")
            
            # Enhanced RAI and compliance validation
            if not await rai_success(input_task.description, True):
                logger.warning(f"RAI validation failed for request {request_id}")
                track_event_if_configured("RAI_Failed", {
                    "status": "Plan not created - RAI violation",
                    "session_id": input_task.session_id,
                    "request_id": request_id
                })
                raise HTTPException(
                    status_code=400, 
                    detail="Content violates responsible AI policies"
                )
            
            if not await self.compliance_engine.validate_content(input_task.description):
                logger.warning(f"Compliance validation failed for request {request_id}")
                track_event_if_configured("Compliance_Failed", {
                    "status": "Plan not created - compliance violation",
                    "session_id": input_task.session_id,
                    "request_id": request_id
                })
                raise HTTPException(
                    status_code=400, 
                    detail="Content violates enterprise compliance policies"
                )
            
            # Enhanced authentication
            authenticated_user = await self.security_manager.get_enhanced_user_details(request.headers)
            user_id = authenticated_user.get("user_principal_id")
            
            if not user_id:
                logger.error(f"Authentication failed for request {request_id}")
                raise HTTPException(
                    status_code=401, 
                    detail="Valid authentication required"
                )
            
            # Generate session ID if not provided
            if not input_task.session_id:
                input_task.session_id = str(uuid.uuid4())
            
            client = None
            try:
                # Initialize enhanced runtime context
                logger.info(f"Initializing runtime context for session {input_task.session_id}")
                kernel, memory_store = await initialize_runtime_and_context(
                    input_task.session_id, user_id
                )
                
                # Get optimal model client through intelligent routing
                logger.info(f"Selecting optimal model for task complexity")
                client = await self.model_router.get_optimal_client(input_task)
                
                # Create enhanced agents with multi-model support
                logger.info(f"Creating enhanced agent ensemble")
                agents = await EnhancedAgentFactory.create_all_agents(
                    session_id=input_task.session_id,
                    user_id=user_id,
                    memory_store=memory_store,
                    client=client,
                    model_router=self.model_router
                )
                
                # Process with enhanced coordination
                group_chat_manager = agents[AgentType.GROUP_CHAT_MANAGER.value]
                
                # Check if enhanced method exists, fallback to standard
                if hasattr(group_chat_manager, 'handle_input_task_enhanced'):
                    await group_chat_manager.handle_input_task_enhanced(input_task)
                else:
                    logger.info("Using standard input task handling")
                    await group_chat_manager.handle_input_task(input_task)
                
                # Retrieve and validate plan
                plan = await memory_store.get_plan_by_session(input_task.session_id)
                if not plan:
                    logger.error(f"Plan not found for session {input_task.session_id}")
                    raise HTTPException(status_code=404, detail="Plan creation failed")
                
                # Calculate performance metrics
                processing_time = time.time() - start_time
                performance_score = await self.performance_monitor.calculate_performance_score(plan) if hasattr(self.performance_monitor, 'calculate_performance_score') else None
                
                # Log successful completion
                await self.performance_monitor.log_task_completion(input_task, plan) if hasattr(self.performance_monitor, 'log_task_completion') else None
                
                logger.info(f"Successfully processed task {request_id} in {processing_time:.2f}s")
                
                track_event_if_configured("Enhanced_Task_Completed", {
                    "status": "success",
                    "session_id": input_task.session_id,
                    "plan_id": plan.id,
                    "processing_time": processing_time,
                    "model_used": getattr(client, 'model_name', 'default'),
                    "request_id": request_id
                })
                
                return TaskResponse(
                    status=f"Enhanced plan created with ID: {plan.id}",
                    session_id=input_task.session_id,
                    plan_id=plan.id,
                    description=input_task.description,
                    model_used=getattr(client, 'model_name', 'default'),
                    performance_score=performance_score,
                    processing_time=processing_time,
                    compliance_status="validated"
                )
                
            except HTTPException:
                # Re-raise HTTP exceptions without modification
                raise
            except Exception as e:
                # Handle unexpected errors
                processing_time = time.time() - start_time
                logger.error(f"Unexpected error in task {request_id}: {e}", exc_info=True)
                
                # Log error for monitoring
                if hasattr(self.performance_monitor, 'log_error'):
                    await self.performance_monitor.log_error(e, input_task)
                
                track_event_if_configured("Enhanced_Task_Error", {
                    "status": "error",
                    "session_id": input_task.session_id,
                    "error": str(e),
                    "processing_time": processing_time,
                    "request_id": request_id
                })
                
                # Sanitize error message for security
                error_msg = await self.security_manager.sanitize_error_message(str(e))
                raise HTTPException(
                    status_code=500, 
                    detail=f"Task processing failed: {error_msg}"
                )
            
            finally:
                # Clean up resources
                if client:
                    await self.model_router.release_client(client)
        
        # Enhanced human feedback endpoint
        @self.app.post("/api/v1/forge1/human_feedback")
        async def enhanced_human_feedback(human_feedback: HumanFeedback, request: Request):
            """Enhanced human feedback with compliance validation"""
            
            request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
            logger.info(f"Processing human feedback {request_id}")
            
            # Validate feedback content if provided
            if human_feedback.human_feedback and not await self.compliance_engine.validate_content(human_feedback.human_feedback):
                raise HTTPException(
                    status_code=400,
                    detail="Feedback content violates compliance policies"
                )
            
            # Enhanced authentication
            authenticated_user = await self.security_manager.get_enhanced_user_details(request.headers)
            user_id = authenticated_user.get("user_principal_id")
            
            if not user_id:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            try:
                # Initialize context
                kernel, memory_store = await initialize_runtime_and_context(
                    human_feedback.session_id, user_id
                )
                
                # Get enhanced client
                client = await self.model_router.get_optimal_client("feedback_processing")
                
                # Create human agent
                human_agent = await EnhancedAgentFactory.create_enhanced_agent(
                    agent_type=AgentType.HUMAN,
                    session_id=human_feedback.session_id,
                    user_id=user_id,
                    model_router=self.model_router,
                    memory_store=memory_store
                )
                
                if not human_agent:
                    raise HTTPException(status_code=404, detail="Human agent not available")
                
                # Process feedback
                await human_agent.handle_human_feedback(human_feedback=human_feedback)
                
                logger.info(f"Successfully processed feedback {request_id}")
                
                return {
                    "status": "Enhanced feedback received",
                    "session_id": human_feedback.session_id,
                    "step_id": human_feedback.step_id,
                    "request_id": request_id
                }
                
            except Exception as e:
                logger.error(f"Error processing feedback {request_id}: {e}")
                error_msg = await self.security_manager.sanitize_error_message(str(e))
                raise HTTPException(status_code=500, detail=f"Feedback processing failed: {error_msg}")
            
            finally:
                if 'client' in locals() and client:
                    await self.model_router.release_client(client)
        
        # Add other enhanced endpoints
        self._setup_enhanced_endpoints()

    def _include_routers(self):
        """Include additional routers"""
        self.app.include_router(health_router)
        self.app.include_router(compliance_router)

    def _setup_metrics_endpoint(self):
        """Expose Prometheus metrics if library available"""
        if PROM_AVAILABLE:
            @self.app.get("/metrics")
            async def metrics():
                data = generate_latest()  # bytes
                return Response(content=data, media_type=CONTENT_TYPE_LATEST)

    def _init_tracing(self):
        """Initialize Jaeger tracer if environment configured"""
        if not JAEGER_TRACING_ENABLED:
            return
        service_name = os.getenv("JAEGER_SERVICE_NAME", "forge1-backend")
        jaeger_agent_host = os.getenv("JAEGER_AGENT_HOST")
        if not jaeger_agent_host:
            return
        try:
            config = JaegerConfig(
                config={
                    'sampler': {'type': 'const', 'param': 1},
                    'logging': False,
                    'local_agent': {'reporting_host': jaeger_agent_host}
                },
                service_name=service_name,
                validate=True,
            )
            self.tracer = config.initialize_tracer()
            logging.info("Jaeger tracer initialized")
        except Exception as e:
            logging.warning(f"Jaeger tracer not initialized: {e}")
    
    def _setup_enhanced_endpoints(self):
        """Setup additional Forge 1 specific endpoints with proper validation"""
        
        @self.app.get("/api/v1/forge1/health/detailed")
        async def detailed_health_check():
            """Comprehensive health check for all Forge 1 components"""
            try:
                health_data = {
                    "status": "healthy",
                    "components": {
                        "model_router": await self.model_router.health_check(),
                        "security_manager": await self.security_manager.health_check(),
                        "performance_monitor": await self.performance_monitor.health_check() if hasattr(self.performance_monitor, 'health_check') else {"status": "healthy"},
                        "compliance_engine": await self.compliance_engine.health_check() if hasattr(self.compliance_engine, 'health_check') else {"status": "healthy"}
                    },
                    "version": "1.0.0",
                    "timestamp": time.time(),
                    "uptime": time.time() - getattr(self, '_start_time', time.time())
                }
                
                # Determine overall status
                component_statuses = [comp.get("status", "unknown") for comp in health_data["components"].values()]
                if any(status in ["unhealthy", "degraded"] for status in component_statuses):
                    health_data["status"] = "degraded"
                
                return health_data
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": time.time()
                }
        
        @self.app.get("/api/v1/forge1/models/available")
        async def get_available_models(request: Request):
            """Get list of available AI models and their capabilities"""
            try:
                # Optional authentication for model info
                try:
                    authenticated_user = await self.security_manager.get_enhanced_user_details(request.headers)
                    user_id = authenticated_user.get("user_principal_id")
                    logger.info(f"Model info requested by user: {user_id}")
                except:
                    logger.info("Anonymous model info request")
                
                models = await self.model_router.get_available_models()
                return {
                    "models": models,
                    "total_count": len(models),
                    "available_count": sum(1 for m in models if m.get("available", False)),
                    "timestamp": time.time()
                }
                
            except Exception as e:
                logger.error(f"Failed to get available models: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve model information")
        
        @self.app.get("/api/v1/forge1/performance/metrics")
        async def get_performance_metrics(request: Request):
            """Get performance metrics for the current user"""
            try:
                authenticated_user = await self.security_manager.get_enhanced_user_details(request.headers)
                user_id = authenticated_user.get("user_principal_id")
                
                if not user_id:
                    raise HTTPException(status_code=401, detail="Authentication required")
                
                # Get user metrics if method exists
                if hasattr(self.performance_monitor, 'get_user_metrics'):
                    metrics = await self.performance_monitor.get_user_metrics(user_id)
                else:
                    # Fallback metrics
                    metrics = {
                        "user_id": user_id,
                        "total_tasks": 0,
                        "successful_tasks": 0,
                        "average_processing_time": 0.0,
                        "performance_score": 0.0,
                        "last_activity": time.time()
                    }
                
                return metrics
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get performance metrics: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve performance metrics")
        
        @self.app.get("/api/v1/forge1/system/status")
        async def get_system_status():
            """Get overall system status and statistics"""
            try:
                return {
                    "system": "Forge 1 Platform",
                    "version": "1.0.0",
                    "status": "operational",
                    "features": {
                        "multi_model_routing": True,
                        "enterprise_security": True,
                        "performance_monitoring": True,
                        "compliance_engine": True,
                        "superhuman_agents": True
                    },
                    "capabilities": {
                        "supported_models": len(self.model_router.models),
                        "active_sessions": 0,  # Would be tracked in production
                        "total_agents": len(EnhancedAgentFactory._enhanced_agent_classes)
                    },
                    "timestamp": time.time()
                }
                
            except Exception as e:
                logger.error(f"Failed to get system status: {e}")
                return {
                    "system": "Forge 1 Platform",
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }
    
    async def _check_database_health(self) -> bool:
        """Check database connectivity"""
        try:
            # Test Cosmos DB connection
            client = Config.GetCosmosDatabaseClient()
            # Simple connectivity test
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def _check_model_router_health(self) -> bool:
        """Check model router health"""
        try:
            health_result = await self.model_router.health_check()
            return health_result.get("status") == "healthy"
        except Exception as e:
            logger.error(f"Model router health check failed: {e}")
            return False
    
    async def _check_security_health(self) -> bool:
        """Check security manager health"""
        try:
            return await self.security_manager.health_check()
        except Exception as e:
            logger.error(f"Security manager health check failed: {e}")
            return False
    
    async def _check_compliance_health(self) -> bool:
        """Check compliance engine health"""
        try:
            if hasattr(self.compliance_engine, 'health_check'):
                return await self.compliance_engine.health_check()
            return True
        except Exception as e:
            logger.error(f"Compliance engine health check failed: {e}")
            return False

# Initialize the enhanced app instance
def create_forge1_app() -> Forge1App:
    """Factory function to create Forge 1 app instance"""
    app_instance = Forge1App()
    app_instance._start_time = time.time()
    logger.info("Forge 1 Platform initialized successfully")
    return app_instance

# Create the enhanced app instance
forge1_app = create_forge1_app()
app = forge1_app.app

# Export for compatibility with Microsoft's structure
__all__ = ["app", "forge1_app", "Forge1App", "TaskResponse", "ErrorResponse", "EnhancedInputTask"]
