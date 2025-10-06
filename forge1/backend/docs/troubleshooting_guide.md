# Employee Lifecycle System - Troubleshooting Guide

Comprehensive troubleshooting guide for diagnosing and resolving common issues in the Employee Lifecycle System.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Database Issues](#database-issues)
3. [Redis/Cache Issues](#rediscache-issues)
4. [API Performance Issues](#api-performance-issues)
5. [Employee Interaction Issues](#employee-interaction-issues)
6. [Memory Management Issues](#memory-management-issues)
7. [Authentication and Authorization](#authentication-and-authorization)
8. [Deployment Issues](#deployment-issues)
9. [Monitoring and Alerting](#monitoring-and-alerting)
10. [Emergency Procedures](#emergency-procedures)

## Quick Diagnostics

### Health Check Commands

```bash
# Basic health check
curl -f http://localhost:8000/health

# Detailed system health
curl -f http://localhost:8000/performance/health

# Check specific components
curl -f http://localhost:8000/performance/database/status
curl -f http://localhost:8000/performance/cache/status
```

### System Status Overview

```bash
# Check all services
docker-compose ps

# Check Kubernetes pods
kubectl get pods -n employee-lifecycle

# Check application logs
docker-compose logs -f app
kubectl logs -f deployment/employee-api -n employee-lifecycle
```

### Quick Performance Check

```bash
# Response time test
time curl -s http://localhost:8000/health

# Load test (basic)
ab -n 100 -c 10 http://localhost:8000/health

# Memory usage
docker stats employee-api
kubectl top pods -n employee-lifecycle
```

## Database Issues

### Connection Problems

#### Symptoms
- "Connection refused" errors
- "Too many connections" errors
- Slow database queries
- Connection timeouts

#### Diagnosis
```bash
# Check PostgreSQL status
pg_isready -h localhost -p 5432 -U employee_user

# Check connection count
psql -h localhost -U employee_user -d employee_db -c "
SELECT count(*) as active_connections, 
       max_conn, 
       max_conn - count(*) as available_connections
FROM pg_stat_activity, 
     (SELECT setting::int as max_conn FROM pg_settings WHERE name='max_connections') mc;"

# Check for blocking queries
psql -h localhost -U employee_user -d employee_db -c "
SELECT blocked_locks.pid AS blocked_pid,
       blocked_activity.usename AS blocked_user,
       blocking_locks.pid AS blocking_pid,
       blocking_activity.usename AS blocking_user,
       blocked_activity.query AS blocked_statement,
       blocking_activity.query AS current_statement_in_blocking_process
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;"
```

#### Solutions
```bash
# Restart PostgreSQL
sudo systemctl restart postgresql
# or for Docker
docker-compose restart db

# Increase connection limit (postgresql.conf)
max_connections = 200
shared_buffers = 256MB

# Kill long-running queries
psql -h localhost -U employee_user -d employee_db -c "
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE state = 'active' 
AND query_start < now() - interval '5 minutes';"

# Optimize connection pooling
# In application configuration
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 30
DATABASE_POOL_TIMEOUT = 30
```

### Performance Issues

#### Symptoms
- Slow query execution
- High CPU usage on database server
- Lock contention
- Memory issues

#### Diagnosis
```sql
-- Check slow queries
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- Check table sizes
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check index usage
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public'
ORDER BY n_distinct DESC;
```

#### Solutions
```sql
-- Create missing indexes
CREATE INDEX CONCURRENTLY idx_employees_client_status 
ON employees(client_id, status) WHERE status = 'active';

CREATE INDEX CONCURRENTLY idx_interactions_employee_created 
ON interactions(employee_id, created_at DESC);

-- Update table statistics
ANALYZE employees;
ANALYZE interactions;
ANALYZE memories;

-- Vacuum tables
VACUUM ANALYZE employees;
VACUUM ANALYZE interactions;
```

### Data Integrity Issues

#### Symptoms
- Foreign key constraint violations
- Duplicate key errors
- Data inconsistencies
- Orphaned records

#### Diagnosis
```sql
-- Check for orphaned employees
SELECT e.id, e.client_id, e.name
FROM employees e
LEFT JOIN clients c ON e.client_id = c.id
WHERE c.id IS NULL;

-- Check for orphaned interactions
SELECT i.id, i.employee_id
FROM interactions i
LEFT JOIN employees e ON i.employee_id = e.id
WHERE e.id IS NULL;

-- Check data consistency
SELECT client_id, 
       COUNT(*) as employee_count,
       MAX(created_at) as last_employee_created
FROM employees
WHERE status = 'active'
GROUP BY client_id
HAVING COUNT(*) > (
    SELECT max_employees 
    FROM clients 
    WHERE id = employees.client_id
);
```

#### Solutions
```sql
-- Clean up orphaned records
DELETE FROM interactions 
WHERE employee_id NOT IN (SELECT id FROM employees);

-- Fix data inconsistencies
UPDATE employees 
SET status = 'inactive' 
WHERE client_id IN (
    SELECT id FROM clients WHERE status = 'inactive'
);

-- Add constraints to prevent future issues
ALTER TABLE interactions 
ADD CONSTRAINT fk_interactions_employee 
FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE;
```

## Redis/Cache Issues

### Connection Problems

#### Symptoms
- Redis connection timeouts
- "Connection refused" errors
- Cache misses increasing
- Memory warnings

#### Diagnosis
```bash
# Check Redis status
redis-cli ping

# Check Redis info
redis-cli info server
redis-cli info memory
redis-cli info clients

# Monitor Redis commands
redis-cli monitor

# Check slow log
redis-cli slowlog get 10
```

#### Solutions
```bash
# Restart Redis
sudo systemctl restart redis
# or for Docker
docker-compose restart redis

# Increase memory limit (redis.conf)
maxmemory 2gb
maxmemory-policy allkeys-lru

# Optimize Redis configuration
timeout 300
tcp-keepalive 60
save 900 1
save 300 10
```

### Performance Issues

#### Symptoms
- High Redis CPU usage
- Slow cache operations
- Memory fragmentation
- Eviction warnings

#### Diagnosis
```bash
# Check Redis performance
redis-cli --latency -h localhost -p 6379

# Check memory usage
redis-cli info memory | grep used_memory

# Check key distribution
redis-cli --scan --pattern "*" | head -20

# Check fragmentation
redis-cli info memory | grep mem_fragmentation_ratio
```

#### Solutions
```bash
# Optimize memory usage
redis-cli config set maxmemory-policy allkeys-lru

# Defragment memory
redis-cli memory doctor

# Clear specific patterns
redis-cli --scan --pattern "session:*" | xargs redis-cli del

# Restart Redis if fragmentation is high (>1.5)
docker-compose restart redis
```

## API Performance Issues

### High Response Times

#### Symptoms
- API responses taking >2 seconds
- Timeout errors
- High CPU/memory usage
- Queue buildup

#### Diagnosis
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health

# Monitor API metrics
curl http://localhost:8000/metrics | grep http_request_duration

# Check application logs for slow operations
grep "processing_time_ms" /var/log/employee-api.log | sort -k5 -nr | head -10

# Profile specific endpoints
time curl -X POST http://localhost:8000/api/v1/employees/clients/test/employees/test/interact \
  -H "Content-Type: application/json" \
  -d '{"message": "test", "session_id": "test"}'
```

#### Solutions
```python
# Optimize database queries
async def get_employee_optimized(client_id: str, employee_id: str):
    # Use select_related to reduce queries
    query = select(Employee).options(
        selectinload(Employee.personality),
        selectinload(Employee.model_preferences)
    ).where(
        Employee.client_id == client_id,
        Employee.id == employee_id
    )
    return await db.execute(query)

# Implement caching
@lru_cache(maxsize=1000)
async def get_cached_employee(employee_id: str):
    return await employee_manager.get_employee(employee_id)

# Add connection pooling
DATABASE_CONFIG = {
    "pool_size": 50,
    "max_overflow": 100,
    "pool_timeout": 30
}
```

### Rate Limiting Issues

#### Symptoms
- 429 "Too Many Requests" errors
- Legitimate requests being blocked
- Uneven rate limiting across clients

#### Diagnosis
```bash
# Check rate limit headers
curl -I http://localhost:8000/api/v1/employees/clients

# Monitor rate limit metrics
curl http://localhost:8000/metrics | grep rate_limit

# Check Redis for rate limit keys
redis-cli keys "rate_limit:*"
```

#### Solutions
```python
# Adjust rate limits
RATE_LIMITS = {
    "default": "1000/hour",
    "premium": "5000/hour",
    "enterprise": "10000/hour"
}

# Implement sliding window rate limiting
async def sliding_window_rate_limit(key: str, limit: int, window: int):
    now = time.time()
    pipeline = redis.pipeline()
    pipeline.zremrangebyscore(key, 0, now - window)
    pipeline.zcard(key)
    pipeline.zadd(key, {str(uuid.uuid4()): now})
    pipeline.expire(key, window)
    results = await pipeline.execute()
    
    return results[1] < limit
```

## Employee Interaction Issues

### LLM API Problems

#### Symptoms
- OpenAI API errors
- High latency in responses
- Token limit exceeded errors
- Cost escalation

#### Diagnosis
```bash
# Check OpenAI API status
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models

# Monitor API usage
grep "openai_api" /var/log/employee-api.log | tail -20

# Check token usage
curl http://localhost:8000/metrics | grep tokens_used
```

#### Solutions
```python
# Implement retry logic with exponential backoff
import backoff

@backoff.on_exception(
    backoff.expo,
    openai.error.RateLimitError,
    max_tries=3,
    max_time=60
)
async def call_openai_with_retry(prompt: str, **kwargs):
    return await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        **kwargs
    )

# Implement token counting
def count_tokens(text: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Add fallback models
MODEL_FALLBACKS = {
    "gpt-4": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
    "gpt-3.5-turbo": ["gpt-3.5-turbo-16k"]
}
```

### Response Quality Issues

#### Symptoms
- Inconsistent employee responses
- Off-topic or inappropriate responses
- Memory not being used effectively
- Poor context understanding

#### Diagnosis
```python
# Check employee configuration
async def diagnose_employee_config(employee_id: str):
    employee = await employee_manager.get_employee(employee_id)
    
    # Check personality configuration
    if not employee.personality:
        logger.warning(f"Employee {employee_id} missing personality config")
    
    # Check model preferences
    if employee.model_preferences.temperature > 0.9:
        logger.warning(f"Employee {employee_id} has high temperature: {employee.model_preferences.temperature}")
    
    # Check memory usage
    memory_count = await memory_manager.get_memory_count(employee_id)
    if memory_count == 0:
        logger.warning(f"Employee {employee_id} has no memories")

# Analyze response patterns
async def analyze_response_quality(employee_id: str, days: int = 7):
    interactions = await get_recent_interactions(employee_id, days)
    
    avg_response_length = sum(len(i.response) for i in interactions) / len(interactions)
    avg_processing_time = sum(i.processing_time_ms for i in interactions) / len(interactions)
    
    return {
        "avg_response_length": avg_response_length,
        "avg_processing_time": avg_processing_time,
        "total_interactions": len(interactions)
    }
```

#### Solutions
```python
# Optimize prompt engineering
def build_optimized_prompt(employee: Employee, message: str, context: dict, memories: List[Memory]) -> str:
    prompt_parts = [
        f"You are {employee.name}, a {employee.role}.",
        f"Communication style: {employee.personality.communication_style}",
        f"Empathy level: {employee.personality.empathy_level}",
    ]
    
    if memories:
        relevant_memories = [m.content for m in memories[:5]]
        prompt_parts.append(f"Relevant context: {'; '.join(relevant_memories)}")
    
    prompt_parts.append(f"User message: {message}")
    prompt_parts.append("Respond helpfully and in character.")
    
    return "\n\n".join(prompt_parts)

# Implement response validation
async def validate_response(response: str, employee: Employee) -> bool:
    # Check response length
    if len(response) < 10 or len(response) > 2000:
        return False
    
    # Check for inappropriate content
    if any(word in response.lower() for word in INAPPROPRIATE_WORDS):
        return False
    
    # Check consistency with personality
    if employee.personality.communication_style == "professional" and "lol" in response.lower():
        return False
    
    return True
```

## Memory Management Issues

### Vector Store Problems

#### Symptoms
- Memory search returning irrelevant results
- High latency in memory retrieval
- Vector store connection errors
- Memory storage failures

#### Diagnosis
```python
# Test vector store connection
async def test_vector_store():
    try:
        # Test Pinecone connection
        import pinecone
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        index = pinecone.Index("employee-memories")
        
        # Test query
        results = index.query(
            vector=[0.1] * 1536,  # Test vector
            top_k=1,
            include_metadata=True
        )
        
        logger.info(f"Vector store test successful: {len(results.matches)} results")
        return True
    except Exception as e:
        logger.error(f"Vector store test failed: {e}")
        return False

# Check memory quality
async def analyze_memory_quality(employee_id: str):
    memories = await memory_manager.get_memories(employee_id, limit=100)
    
    # Check for duplicate memories
    content_hashes = {}
    duplicates = 0
    for memory in memories:
        content_hash = hashlib.md5(memory.content.encode()).hexdigest()
        if content_hash in content_hashes:
            duplicates += 1
        content_hashes[content_hash] = memory.id
    
    # Check importance score distribution
    importance_scores = [m.importance_score for m in memories]
    avg_importance = sum(importance_scores) / len(importance_scores)
    
    return {
        "total_memories": len(memories),
        "duplicates": duplicates,
        "avg_importance": avg_importance,
        "low_importance_count": sum(1 for s in importance_scores if s < 0.3)
    }
```

#### Solutions
```python
# Implement memory deduplication
async def deduplicate_memories(employee_id: str):
    memories = await memory_manager.get_memories(employee_id)
    
    seen_hashes = set()
    duplicates_to_remove = []
    
    for memory in memories:
        content_hash = hashlib.md5(memory.content.encode()).hexdigest()
        if content_hash in seen_hashes:
            duplicates_to_remove.append(memory.id)
        else:
            seen_hashes.add(content_hash)
    
    for memory_id in duplicates_to_remove:
        await memory_manager.delete_memory(memory_id)
    
    logger.info(f"Removed {len(duplicates_to_remove)} duplicate memories for employee {employee_id}")

# Optimize memory search
async def optimized_memory_search(employee_id: str, query: str, limit: int = 10):
    # Use semantic search with filters
    query_embedding = await get_embedding(query)
    
    results = await vector_store.query(
        vector=query_embedding,
        filter={"employee_id": employee_id},
        top_k=limit * 2,  # Get more results for filtering
        include_metadata=True
    )
    
    # Filter by relevance threshold
    relevant_results = [
        r for r in results.matches 
        if r.score > 0.7  # Adjust threshold as needed
    ]
    
    return relevant_results[:limit]
```

### Memory Storage Issues

#### Symptoms
- Memory not being stored
- Storage errors in logs
- Inconsistent memory retrieval
- Memory corruption

#### Diagnosis
```python
# Test memory storage pipeline
async def test_memory_storage(employee_id: str):
    test_memory = {
        "content": "Test memory content for diagnostics",
        "memory_type": "test",
        "importance_score": 0.8,
        "context": {"test": True}
    }
    
    try:
        # Store memory
        memory_id = await memory_manager.store_memory(employee_id, test_memory)
        logger.info(f"Test memory stored: {memory_id}")
        
        # Retrieve memory
        retrieved = await memory_manager.get_memory(memory_id)
        if retrieved and retrieved.content == test_memory["content"]:
            logger.info("Memory retrieval test passed")
        else:
            logger.error("Memory retrieval test failed")
        
        # Clean up
        await memory_manager.delete_memory(memory_id)
        
    except Exception as e:
        logger.error(f"Memory storage test failed: {e}")
```

#### Solutions
```python
# Implement robust memory storage with retry
async def store_memory_with_retry(employee_id: str, memory_data: dict, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            # Store in database
            memory_id = await db_store_memory(employee_id, memory_data)
            
            # Store in vector store
            embedding = await get_embedding(memory_data["content"])
            await vector_store.upsert(
                vectors=[(memory_id, embedding, {"employee_id": employee_id})]
            )
            
            return memory_id
            
        except Exception as e:
            logger.warning(f"Memory storage attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

# Add memory validation
def validate_memory_data(memory_data: dict) -> bool:
    required_fields = ["content", "memory_type", "importance_score"]
    
    for field in required_fields:
        if field not in memory_data:
            logger.error(f"Missing required field: {field}")
            return False
    
    if not isinstance(memory_data["importance_score"], (int, float)):
        logger.error("importance_score must be numeric")
        return False
    
    if not 0 <= memory_data["importance_score"] <= 1:
        logger.error("importance_score must be between 0 and 1")
        return False
    
    return True
```

## Authentication and Authorization

### Token Issues

#### Symptoms
- "Invalid token" errors
- Token expiration issues
- Authentication failures
- Unauthorized access attempts

#### Diagnosis
```python
# Validate JWT token
import jwt

def diagnose_token(token: str):
    try:
        # Decode without verification first to see contents
        unverified = jwt.decode(token, options={"verify_signature": False})
        logger.info(f"Token contents: {unverified}")
        
        # Check expiration
        exp = unverified.get("exp")
        if exp and exp < time.time():
            logger.error("Token has expired")
        
        # Verify signature
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
        logger.info("Token signature valid")
        
        return decoded
        
    except jwt.ExpiredSignatureError:
        logger.error("Token has expired")
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid token: {e}")
    except Exception as e:
        logger.error(f"Token validation error: {e}")
```

#### Solutions
```python
# Implement token refresh mechanism
async def refresh_token(refresh_token: str) -> str:
    try:
        payload = jwt.decode(refresh_token, JWT_SECRET_KEY, algorithms=["HS256"])
        
        # Generate new access token
        new_payload = {
            "user_id": payload["user_id"],
            "tenant_id": payload["tenant_id"],
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        
        return jwt.encode(new_payload, JWT_SECRET_KEY, algorithm="HS256")
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired")

# Add token blacklist
BLACKLISTED_TOKENS = set()

async def blacklist_token(token: str):
    BLACKLISTED_TOKENS.add(token)
    # Also store in Redis for distributed systems
    await redis.sadd("blacklisted_tokens", token)

async def is_token_blacklisted(token: str) -> bool:
    return token in BLACKLISTED_TOKENS or await redis.sismember("blacklisted_tokens", token)
```

### Tenant Isolation Issues

#### Symptoms
- Cross-tenant data access
- Tenant ID validation failures
- Data leakage between tenants
- Authorization bypass

#### Diagnosis
```python
# Audit tenant access
async def audit_tenant_access(user_id: str, requested_tenant: str):
    # Check user's tenant permissions
    user_tenants = await get_user_tenants(user_id)
    
    if requested_tenant not in user_tenants:
        logger.warning(f"User {user_id} attempted to access unauthorized tenant {requested_tenant}")
        return False
    
    # Log access for audit trail
    await log_tenant_access(user_id, requested_tenant, datetime.utcnow())
    return True

# Check for data leakage
async def check_data_isolation():
    # Check for employees without proper tenant isolation
    query = """
    SELECT e.id, e.client_id, c.tenant_id
    FROM employees e
    JOIN clients c ON e.client_id = c.id
    WHERE e.tenant_id != c.tenant_id OR e.tenant_id IS NULL
    """
    
    results = await db.execute(query)
    if results:
        logger.error(f"Found {len(results)} employees with tenant isolation issues")
        return results
    
    return []
```

#### Solutions
```python
# Implement strict tenant filtering
class TenantFilter:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
    
    def apply_to_query(self, query):
        return query.where(Employee.tenant_id == self.tenant_id)

# Add tenant validation middleware
@app.middleware("http")
async def tenant_validation_middleware(request: Request, call_next):
    tenant_id = request.headers.get("X-Tenant-ID")
    user_id = request.state.user_id  # Set by auth middleware
    
    if tenant_id and not await validate_user_tenant_access(user_id, tenant_id):
        return JSONResponse(
            status_code=403,
            content={"error": "Unauthorized tenant access"}
        )
    
    request.state.tenant_id = tenant_id
    return await call_next(request)
```

## Deployment Issues

### Container Problems

#### Symptoms
- Container startup failures
- Health check failures
- Resource constraints
- Image build issues

#### Diagnosis
```bash
# Check container status
docker ps -a
kubectl get pods -n employee-lifecycle

# Check container logs
docker logs employee-api
kubectl logs deployment/employee-api -n employee-lifecycle

# Check resource usage
docker stats
kubectl top pods -n employee-lifecycle

# Check health checks
docker exec employee-api curl -f http://localhost:8000/health
```

#### Solutions
```dockerfile
# Optimize Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash employee \
    && chown -R employee:employee /app
USER employee

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "forge1.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Issues

#### Symptoms
- Pod crashes and restarts
- Service discovery failures
- Ingress configuration issues
- Resource quota exceeded

#### Diagnosis
```bash
# Check pod events
kubectl describe pod <pod-name> -n employee-lifecycle

# Check service endpoints
kubectl get endpoints -n employee-lifecycle

# Check ingress status
kubectl describe ingress employee-api-ingress -n employee-lifecycle

# Check resource quotas
kubectl describe resourcequota -n employee-lifecycle
```

#### Solutions
```yaml
# Add resource limits and requests
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi"
    cpu: "1000m"

# Configure proper probes
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3

# Add pod disruption budget
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: employee-api-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: employee-api
```

## Monitoring and Alerting

### Metrics Collection Issues

#### Symptoms
- Missing metrics data
- Prometheus scraping failures
- Grafana dashboard errors
- Alert notification failures

#### Diagnosis
```bash
# Check Prometheus targets
curl http://prometheus:9090/api/v1/targets

# Check metrics endpoint
curl http://localhost:8000/metrics

# Test alert rules
curl http://prometheus:9090/api/v1/rules

# Check Grafana data sources
curl -u admin:password http://grafana:3000/api/datasources
```

#### Solutions
```python
# Fix metrics collection
from prometheus_client import Counter, Histogram, Gauge

# Add missing metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_EMPLOYEES = Gauge('active_employees_total', 'Number of active employees')

# Middleware to collect metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    
    return response
```

### Alert Configuration

#### Common Alert Rules
```yaml
# alerts.yml
groups:
  - name: employee-api-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: DatabaseConnectionsHigh
        expr: database_connections_active / database_connections_max > 0.8
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Database connection pool usage high"

      - alert: MemoryUsageHigh
        expr: process_resident_memory_bytes / 1024 / 1024 > 1000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
```

## Emergency Procedures

### System Outage Response

#### Immediate Actions (0-5 minutes)
1. **Assess Impact**
   ```bash
   # Check system status
   curl -f https://api.forge1.com/health
   
   # Check monitoring dashboards
   # Check error rates and response times
   ```

2. **Notify Stakeholders**
   ```bash
   # Update status page
   # Send notifications to team
   # Escalate if needed
   ```

3. **Initial Diagnosis**
   ```bash
   # Check recent deployments
   kubectl rollout history deployment/employee-api -n employee-lifecycle
   
   # Check system resources
   kubectl top nodes
   kubectl top pods -n employee-lifecycle
   
   # Check logs for errors
   kubectl logs -f deployment/employee-api -n employee-lifecycle --tail=100
   ```

#### Recovery Actions (5-30 minutes)
1. **Database Issues**
   ```bash
   # Check database connectivity
   pg_isready -h db-host -p 5432
   
   # Check for blocking queries
   # Kill long-running queries if needed
   
   # Restart database if necessary
   kubectl rollout restart statefulset/postgres -n employee-lifecycle
   ```

2. **Application Issues**
   ```bash
   # Rollback to previous version
   kubectl rollout undo deployment/employee-api -n employee-lifecycle
   
   # Scale up replicas
   kubectl scale deployment/employee-api --replicas=5 -n employee-lifecycle
   
   # Clear cache if needed
   redis-cli flushall
   ```

3. **Infrastructure Issues**
   ```bash
   # Check node health
   kubectl get nodes
   
   # Drain problematic nodes
   kubectl drain <node-name> --ignore-daemonsets
   
   # Scale cluster if needed
   ```

### Data Recovery Procedures

#### Database Recovery
```bash
# Stop application to prevent further writes
kubectl scale deployment/employee-api --replicas=0 -n employee-lifecycle

# Restore from backup
pg_restore -h db-host -U postgres -d employee_db /path/to/backup.dump

# Verify data integrity
python scripts/verify_data_integrity.py

# Restart application
kubectl scale deployment/employee-api --replicas=3 -n employee-lifecycle
```

#### Cache Recovery
```bash
# Clear corrupted cache
redis-cli flushall

# Warm up cache with critical data
python scripts/warm_cache.py

# Monitor cache hit rates
redis-cli info stats | grep keyspace_hits
```

### Communication Templates

#### Incident Notification
```
Subject: [INCIDENT] Employee Lifecycle System - Service Degradation

We are currently experiencing issues with the Employee Lifecycle System API.

Impact: API response times increased, some requests may fail
Start Time: [TIME]
Current Status: Investigating

We are actively working to resolve this issue and will provide updates every 15 minutes.

Status Page: https://status.forge1.com
```

#### Resolution Notification
```
Subject: [RESOLVED] Employee Lifecycle System - Service Restored

The Employee Lifecycle System API has been restored to normal operation.

Resolution: Database connection pool was optimized and additional replicas deployed
Duration: [DURATION]
Root Cause: High database connection usage during peak traffic

Post-mortem will be available within 24 hours.
```

This troubleshooting guide provides comprehensive procedures for diagnosing and resolving issues in the Employee Lifecycle System. Regular review and practice of these procedures will help ensure quick resolution of any problems that arise.