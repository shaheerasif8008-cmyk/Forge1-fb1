# Multi-Tenant Memory/Context and Billing Isolation Design

## Overview

This design addresses the critical challenge of ensuring complete isolation of memory/context and accurate billing attribution when hundreds of AI employees from different clients share the same API keys. The system must guarantee that:

1. **Memory Isolation**: Each AI employee's context is completely isolated from others
2. **Billing Attribution**: Every API call is accurately tracked and billed to the correct client
3. **Security**: No data leakage between tenants/clients
4. **Performance**: Minimal overhead for isolation mechanisms

## Current Implementation Analysis

### ✅ **Existing Tenant Isolation** (forge1/core/tenancy.py)
```python
# Thread-local storage for tenant context
_tenant_context = threading.local()

def set_current_tenant(tenant_id: Optional[str]):
    """Set the current tenant ID for the request context"""
    _tenant_context.tenant_id = tenant_id
```

### ✅ **Memory Isolation** (forge1/core/memory_manager.py)
```python
# Tenant tag enforcement in memory storage
tenant_tag = f"tenant:{get_current_tenant()}"
if tenant_tag not in memory.tags:
    memory.tags.append(tenant_tag)

# Tenant filtering in memory search
tenant_tag = f"tenant:{get_current_tenant()}"
sql_conditions.append(f"tags && ${param_count}")
sql_params.append([tenant_tag])
```

### ✅ **Usage Tracking** (forge1/billing/pricing_engine.py)
```python
async def record_usage(
    self,
    customer_id: str,
    metric_id: str,
    quantity: Decimal,
    # ... tracking per customer
):
```

## Enhanced Multi-Tenant Isolation Architecture

### 1. **Request Context Isolation Layer**

```python
class RequestContext:
    """Enhanced request context with complete isolation"""
    
    def __init__(self):
        self.tenant_id: str = None
        self.client_id: str = None
        self.employee_id: str = None
        self.session_id: str = None
        self.user_id: str = None
        self.request_id: str = str(uuid.uuid4())
        self.created_at: datetime = datetime.utcnow()
        
        # Billing context
        self.billing_context = BillingContext()
        
        # Security context
        self.security_level: SecurityLevel = SecurityLevel.INTERNAL
        self.data_classification: str = "internal"
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics()

class ContextManager:
    """Thread-safe context management with complete isolation"""
    
    def __init__(self):
        self._context_storage = contextvars.ContextVar('request_context')
        self._context_stack = []
    
    def set_context(self, context: RequestContext) -> None:
        """Set request context with validation"""
        if not context.tenant_id or not context.client_id:
            raise ValueError("Tenant ID and Client ID are required")
        
        # Validate context integrity
        self._validate_context(context)
        
        # Set in context variable (thread-safe)
        self._context_storage.set(context)
        
        # Add to stack for nested contexts
        self._context_stack.append(context)
    
    def get_context(self) -> RequestContext:
        """Get current request context"""
        try:
            return self._context_storage.get()
        except LookupError:
            raise RuntimeError("No request context set - potential security violation")
    
    def clear_context(self) -> None:
        """Clear request context"""
        if self._context_stack:
            self._context_stack.pop()
        
        if self._context_stack:
            # Restore previous context
            self._context_storage.set(self._context_stack[-1])
        else:
            # Clear completely
            try:
                self._context_storage.delete()
            except LookupError:
                pass
    
    def _validate_context(self, context: RequestContext) -> None:
        """Validate context for security and completeness"""
        required_fields = ['tenant_id', 'client_id', 'employee_id', 'user_id']
        
        for field in required_fields:
            if not getattr(context, field):
                raise ValueError(f"Required context field missing: {field}")
        
        # Validate tenant-client relationship
        if not self._validate_tenant_client_relationship(context.tenant_id, context.client_id):
            raise SecurityError(f"Invalid tenant-client relationship: {context.tenant_id}-{context.client_id}")

# Global context manager instance
context_manager = ContextManager()
```

### 2. **Enhanced Memory Isolation System**

```python
class IsolatedMemoryManager(MemoryManager):
    """Memory manager with enhanced tenant isolation"""
    
    async def store_memory(self, memory: MemoryContext) -> str:
        """Store memory with strict tenant isolation"""
        
        # Get current context
        context = context_manager.get_context()
        
        # Enforce tenant isolation
        memory = self._enforce_tenant_isolation(memory, context)
        
        # Add isolation metadata
        memory.metadata = {
            **memory.metadata,
            'tenant_id': context.tenant_id,
            'client_id': context.client_id,
            'employee_id': context.employee_id,
            'isolation_level': 'strict',
            'created_by_request': context.request_id
        }
        
        # Store with tenant-specific encryption
        encrypted_memory = await self._encrypt_for_tenant(memory, context.tenant_id)
        
        return await super().store_memory(encrypted_memory)
    
    async def retrieve_memory(self, memory_id: str, user_id: str) -> Optional[MemoryContext]:
        """Retrieve memory with tenant validation"""
        
        context = context_manager.get_context()
        
        # Retrieve memory
        memory = await super().retrieve_memory(memory_id, user_id)
        
        if not memory:
            return None
        
        # Validate tenant access
        if not self._validate_tenant_access(memory, context):
            logger.warning(f"Tenant isolation violation: {context.tenant_id} attempted to access memory from different tenant")
            return None
        
        # Decrypt for tenant
        decrypted_memory = await self._decrypt_for_tenant(memory, context.tenant_id)
        
        return decrypted_memory
    
    def _enforce_tenant_isolation(self, memory: MemoryContext, context: RequestContext) -> MemoryContext:
        """Enforce strict tenant isolation on memory"""
        
        # Add tenant tags
        tenant_tags = [
            f"tenant:{context.tenant_id}",
            f"client:{context.client_id}",
            f"employee:{context.employee_id}"
        ]
        
        for tag in tenant_tags:
            if tag not in memory.tags:
                memory.tags.append(tag)
        
        # Set owner to tenant-scoped user
        memory.owner_id = f"{context.tenant_id}:{context.user_id}"
        
        # Ensure security level
        if memory.security_level == SecurityLevel.PUBLIC:
            memory.security_level = SecurityLevel.INTERNAL  # No public memories in multi-tenant
        
        return memory
    
    def _validate_tenant_access(self, memory: MemoryContext, context: RequestContext) -> bool:
        """Validate that current tenant can access memory"""
        
        # Check tenant tags
        tenant_tag = f"tenant:{context.tenant_id}"
        if tenant_tag not in memory.tags:
            return False
        
        # Check owner
        expected_owner_prefix = f"{context.tenant_id}:"
        if not memory.owner_id.startswith(expected_owner_prefix):
            return False
        
        return True
    
    async def _encrypt_for_tenant(self, memory: MemoryContext, tenant_id: str) -> MemoryContext:
        """Encrypt memory content for specific tenant"""
        
        # Get tenant-specific encryption key
        encryption_key = await self._get_tenant_encryption_key(tenant_id)
        
        # Encrypt sensitive fields
        if memory.content:
            memory.content = await self._encrypt_data(memory.content, encryption_key)
        
        if memory.summary:
            memory.summary = await self._encrypt_string(memory.summary, encryption_key)
        
        return memory
    
    async def _decrypt_for_tenant(self, memory: MemoryContext, tenant_id: str) -> MemoryContext:
        """Decrypt memory content for specific tenant"""
        
        # Get tenant-specific encryption key
        encryption_key = await self._get_tenant_encryption_key(tenant_id)
        
        # Decrypt sensitive fields
        if memory.content:
            memory.content = await self._decrypt_data(memory.content, encryption_key)
        
        if memory.summary:
            memory.summary = await self._decrypt_string(memory.summary, encryption_key)
        
        return memory
```

### 3. **API Call Tracking and Attribution System**

```python
class APICallTracker:
    """Track and attribute API calls to specific tenants/employees"""
    
    def __init__(self, billing_engine: PricingEngine):
        self.billing_engine = billing_engine
        self.call_registry = {}
        self.metrics_collector = MetricsCollector()
    
    async def track_api_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        response_time_ms: float,
        cost: Decimal
    ) -> str:
        """Track API call with complete attribution"""
        
        # Get current context
        context = context_manager.get_context()
        
        # Generate tracking ID
        call_id = f"api_{context.tenant_id}_{context.employee_id}_{int(time.time() * 1000)}"
        
        # Create detailed call record
        call_record = APICallRecord(
            call_id=call_id,
            tenant_id=context.tenant_id,
            client_id=context.client_id,
            employee_id=context.employee_id,
            session_id=context.session_id,
            user_id=context.user_id,
            request_id=context.request_id,
            
            # API details
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            response_time_ms=response_time_ms,
            cost=cost,
            
            # Timing
            timestamp=datetime.utcnow(),
            billing_period=self._get_billing_period(),
            
            # Metadata
            api_key_pool_id=self._get_current_api_key_pool(),
            rate_limit_bucket=self._get_rate_limit_bucket(context.tenant_id),
            
            # Security
            data_classification=context.data_classification,
            security_level=context.security_level.value
        )
        
        # Store call record
        self.call_registry[call_id] = call_record
        await self._store_call_record(call_record)
        
        # Record usage for billing
        await self.billing_engine.record_usage(
            customer_id=context.client_id,
            metric_id="ai_tokens",
            quantity=Decimal(str(input_tokens + output_tokens)),
            resource_id=context.employee_id,
            user_id=context.user_id,
            session_id=context.session_id,
            metadata={
                'call_id': call_id,
                'provider': provider,
                'model': model,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'response_time_ms': response_time_ms
            }
        )
        
        # Update metrics
        self.metrics_collector.increment(f"api_calls_{provider}_{model}")
        self.metrics_collector.record_metric(f"api_cost_{provider}", float(cost))
        self.metrics_collector.record_metric(f"api_tokens_{provider}", input_tokens + output_tokens)
        
        logger.info(f"Tracked API call {call_id}: {provider}/{model} - {input_tokens + output_tokens} tokens - ${cost}")
        
        return call_id

@dataclass
class APICallRecord:
    """Complete record of an API call for billing and auditing"""
    call_id: str
    tenant_id: str
    client_id: str
    employee_id: str
    session_id: str
    user_id: str
    request_id: str
    
    # API details
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    response_time_ms: float
    cost: Decimal
    
    # Timing
    timestamp: datetime
    billing_period: str
    
    # Infrastructure
    api_key_pool_id: str
    rate_limit_bucket: str
    
    # Security
    data_classification: str
    security_level: str
```

### 4. **Enhanced Model Router with Isolation**

```python
class IsolatedModelRouter(ModelRouter):
    """Model router with tenant isolation and billing attribution"""
    
    def __init__(self, api_key_manager: APIKeyManager, call_tracker: APICallTracker):
        super().__init__()
        self.api_key_manager = api_key_manager
        self.call_tracker = call_tracker
    
    async def route_request(self, request: ModelRequest) -> ModelResponse:
        """Route request with complete isolation and tracking"""
        
        # Get current context
        context = context_manager.get_context()
        
        # Validate tenant permissions
        if not await self._validate_tenant_permissions(context, request):
            raise SecurityError(f"Tenant {context.tenant_id} not authorized for model {request.model}")
        
        # Get tenant-appropriate API key
        api_key = await self.api_key_manager.get_tenant_api_key(
            provider=request.provider,
            tenant_id=context.tenant_id,
            client_tier=await self._get_client_tier(context.client_id)
        )
        
        if not api_key:
            raise ResourceError(f"No available API key for tenant {context.tenant_id}")
        
        try:
            # Execute request with tracking
            start_time = time.time()
            
            response = await self._execute_with_api_key(request, api_key)
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Calculate cost
            cost = await self._calculate_cost(
                provider=request.provider,
                model=request.model,
                input_tokens=request.input_tokens,
                output_tokens=response.output_tokens
            )
            
            # Track the API call
            call_id = await self.call_tracker.track_api_call(
                provider=request.provider,
                model=request.model,
                input_tokens=request.input_tokens,
                output_tokens=response.output_tokens,
                response_time_ms=response_time_ms,
                cost=cost
            )
            
            # Add tracking info to response
            response.metadata = {
                **response.metadata,
                'call_id': call_id,
                'tenant_id': context.tenant_id,
                'client_id': context.client_id,
                'employee_id': context.employee_id,
                'cost': str(cost),
                'api_key_pool': api_key.pool_id
            }
            
            return response
            
        except Exception as e:
            # Track failed calls too
            await self.call_tracker.track_failed_call(
                context=context,
                request=request,
                error=str(e),
                response_time_ms=(time.time() - start_time) * 1000
            )
            raise
        
        finally:
            # Return API key to pool
            await self.api_key_manager.return_api_key(api_key)
```

### 5. **Database Schema for Isolation**

```sql
-- Enhanced tenant isolation tables
CREATE SCHEMA IF NOT EXISTS forge1_isolation;

-- API call tracking table
CREATE TABLE forge1_isolation.api_call_records (
    call_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    client_id VARCHAR(255) NOT NULL,
    employee_id UUID NOT NULL,
    session_id UUID,
    user_id VARCHAR(255) NOT NULL,
    request_id VARCHAR(255) NOT NULL,
    
    -- API details
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    response_time_ms FLOAT NOT NULL,
    cost DECIMAL(10,6) NOT NULL,
    
    -- Timing
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    billing_period VARCHAR(20) NOT NULL,
    
    -- Infrastructure
    api_key_pool_id VARCHAR(255),
    rate_limit_bucket VARCHAR(255),
    
    -- Security
    data_classification VARCHAR(50),
    security_level VARCHAR(20),
    
    -- Indexes
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Tenant encryption keys table
CREATE TABLE forge1_isolation.tenant_encryption_keys (
    tenant_id VARCHAR(255) PRIMARY KEY,
    encryption_key_hash VARCHAR(255) NOT NULL,
    key_version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    rotated_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Tenant access audit table
CREATE TABLE forge1_isolation.tenant_access_audit (
    audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(255) NOT NULL,
    client_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(255) NOT NULL,
    action VARCHAR(50) NOT NULL,
    result VARCHAR(20) NOT NULL, -- success, denied, error
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- Indexes for performance
CREATE INDEX idx_api_calls_tenant_time ON forge1_isolation.api_call_records(tenant_id, timestamp);
CREATE INDEX idx_api_calls_client_billing ON forge1_isolation.api_call_records(client_id, billing_period);
CREATE INDEX idx_api_calls_employee ON forge1_isolation.api_call_records(employee_id, timestamp);
CREATE INDEX idx_api_calls_cost ON forge1_isolation.api_call_records(cost, timestamp);

CREATE INDEX idx_audit_tenant_time ON forge1_isolation.tenant_access_audit(tenant_id, timestamp);
CREATE INDEX idx_audit_resource ON forge1_isolation.tenant_access_audit(resource_type, resource_id);
```

### 6. **Middleware for Automatic Context Management**

```python
class TenantIsolationMiddleware:
    """FastAPI middleware for automatic tenant context management"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Extract tenant information from request
        request = Request(scope, receive)
        
        # Get tenant info from headers, JWT, or API key
        tenant_info = await self._extract_tenant_info(request)
        
        if not tenant_info:
            # Return 401 Unauthorized
            response = JSONResponse(
                status_code=401,
                content={"error": "Missing or invalid tenant information"}
            )
            await response(scope, receive, send)
            return
        
        # Create and set request context
        context = RequestContext()
        context.tenant_id = tenant_info['tenant_id']
        context.client_id = tenant_info['client_id']
        context.employee_id = tenant_info.get('employee_id')
        context.user_id = tenant_info['user_id']
        context.session_id = tenant_info.get('session_id')
        
        # Set security context
        context.security_level = SecurityLevel(tenant_info.get('security_level', 'internal'))
        context.data_classification = tenant_info.get('data_classification', 'internal')
        
        try:
            # Set context for this request
            context_manager.set_context(context)
            
            # Process request
            await self.app(scope, receive, send)
            
        finally:
            # Always clear context
            context_manager.clear_context()
    
    async def _extract_tenant_info(self, request: Request) -> Optional[Dict[str, str]]:
        """Extract tenant information from request"""
        
        # Try JWT token first
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            tenant_info = await self._decode_jwt_token(token)
            if tenant_info:
                return tenant_info
        
        # Try API key
        api_key = request.headers.get("x-api-key")
        if api_key:
            tenant_info = await self._lookup_api_key(api_key)
            if tenant_info:
                return tenant_info
        
        # Try tenant headers (for internal services)
        tenant_id = request.headers.get("x-tenant-id")
        client_id = request.headers.get("x-client-id")
        user_id = request.headers.get("x-user-id")
        
        if tenant_id and client_id and user_id:
            return {
                'tenant_id': tenant_id,
                'client_id': client_id,
                'user_id': user_id,
                'employee_id': request.headers.get("x-employee-id"),
                'session_id': request.headers.get("x-session-id")
            }
        
        return None
```

## Implementation Benefits

### **Complete Isolation Guarantees**

1. **Memory Isolation**: Each tenant's memory is encrypted with tenant-specific keys and tagged with tenant identifiers
2. **Context Isolation**: Thread-safe context management ensures no cross-tenant data leakage
3. **API Attribution**: Every API call is tracked and attributed to the correct tenant/client/employee
4. **Database Isolation**: Row-level security with tenant tags in all queries

### **Accurate Billing**

1. **Token-Level Tracking**: Every input/output token is tracked per tenant
2. **Cost Attribution**: Real-time cost calculation and attribution
3. **Usage Analytics**: Detailed usage patterns per client/employee
4. **Audit Trail**: Complete audit trail for billing disputes

### **Security Compliance**

1. **Encryption**: Tenant-specific encryption for all stored data
2. **Access Control**: Multi-level access validation
3. **Audit Logging**: Complete audit trail of all access attempts
4. **Data Classification**: Automatic data classification and handling

### **Performance Optimization**

1. **Context Caching**: Efficient context management with minimal overhead
2. **Connection Pooling**: Shared API key pools with tenant attribution
3. **Lazy Loading**: Memory and context loaded only when needed
4. **Batch Processing**: Efficient batch processing of usage records

This design ensures that hundreds of AI employees can safely share API keys while maintaining complete isolation and accurate billing attribution.