# Employee Lifecycle System - Deployment Guide

Comprehensive deployment guide for the Employee Lifecycle System covering development, staging, and production environments.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Environment Configuration](#environment-configuration)
4. [Database Setup](#database-setup)
5. [Application Deployment](#application-deployment)
6. [Monitoring and Health Checks](#monitoring-and-health-checks)
7. [Security Configuration](#security-configuration)
8. [Scaling and Performance](#scaling-and-performance)
9. [Troubleshooting](#troubleshooting)

## Overview

The Employee Lifecycle System is designed for cloud-native deployment with support for:
- Docker containerization
- Kubernetes orchestration
- Multi-environment configuration
- Horizontal scaling
- Health monitoring
- Security best practices

### Architecture Components

- **API Server**: FastAPI application serving REST endpoints
- **Database**: PostgreSQL for persistent data storage
- **Cache**: Redis for session and performance caching
- **Vector Store**: Pinecone/Weaviate for memory embeddings
- **Message Queue**: Redis/RabbitMQ for async processing
- **Monitoring**: Prometheus + Grafana for metrics
- **Logging**: ELK stack for centralized logging

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 2 cores
- RAM: 4GB
- Storage: 20GB SSD
- Network: 1Gbps

**Recommended for Production:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 100GB+ SSD
- Network: 10Gbps
- Load Balancer: HAProxy/NGINX

### Software Dependencies

- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.20+ (for K8s deployment)
- PostgreSQL 13+
- Redis 6.0+
- Python 3.9+

### External Services

- **LLM Provider**: OpenAI API access
- **Vector Database**: Pinecone or Weaviate
- **Monitoring**: Datadog/New Relic (optional)
- **Email Service**: SendGrid/AWS SES (optional)

## Environment Configuration

### Environment Variables

Create environment-specific configuration files:

#### Development (.env.dev)
```bash
# Application
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
API_HOST=0.0.0.0
API_PORT=8000

# Database
DATABASE_URL=postgresql://dev_user:dev_pass@localhost:5432/employee_dev
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_POOL_SIZE=10

# Security
SECRET_KEY=dev-secret-key-change-in-production
JWT_SECRET_KEY=dev-jwt-secret-change-in-production
ENCRYPTION_KEY=dev-encryption-key-32-chars-long

# External Services
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-west1-gcp

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

#### Staging (.env.staging)
```bash
# Application
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# Database
DATABASE_URL=postgresql://staging_user:secure_pass@db-staging:5432/employee_staging
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
DATABASE_SSL_MODE=require

# Redis
REDIS_URL=redis://redis-staging:6379/0
REDIS_POOL_SIZE=20
REDIS_SSL=true

# Security
SECRET_KEY=${SECRET_KEY}
JWT_SECRET_KEY=${JWT_SECRET_KEY}
ENCRYPTION_KEY=${ENCRYPTION_KEY}

# External Services
OPENAI_API_KEY=${OPENAI_API_KEY}
PINECONE_API_KEY=${PINECONE_API_KEY}
PINECONE_ENVIRONMENT=us-west1-gcp

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
SENTRY_DSN=${SENTRY_DSN}
```

#### Production (.env.prod)
```bash
# Application
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
API_HOST=0.0.0.0
API_PORT=8000

# Database
DATABASE_URL=${DATABASE_URL}
DATABASE_POOL_SIZE=50
DATABASE_MAX_OVERFLOW=100
DATABASE_SSL_MODE=require
DATABASE_SSL_CERT_PATH=/certs/client-cert.pem
DATABASE_SSL_KEY_PATH=/certs/client-key.pem
DATABASE_SSL_CA_PATH=/certs/ca-cert.pem

# Redis
REDIS_URL=${REDIS_URL}
REDIS_POOL_SIZE=50
REDIS_SSL=true
REDIS_SSL_CERT_PATH=/certs/redis-cert.pem

# Security
SECRET_KEY=${SECRET_KEY}
JWT_SECRET_KEY=${JWT_SECRET_KEY}
ENCRYPTION_KEY=${ENCRYPTION_KEY}
CORS_ORIGINS=${CORS_ORIGINS}

# External Services
OPENAI_API_KEY=${OPENAI_API_KEY}
PINECONE_API_KEY=${PINECONE_API_KEY}
PINECONE_ENVIRONMENT=${PINECONE_ENVIRONMENT}

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
SENTRY_DSN=${SENTRY_DSN}
DATADOG_API_KEY=${DATADOG_API_KEY}

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=1000
```

## Database Setup

### PostgreSQL Configuration

#### Development Setup
```bash
# Using Docker
docker run -d \
  --name employee-db-dev \
  -e POSTGRES_DB=employee_dev \
  -e POSTGRES_USER=dev_user \
  -e POSTGRES_PASSWORD=dev_pass \
  -p 5432:5432 \
  postgres:13

# Create database schema
python -m alembic upgrade head
```

#### Production Setup
```sql
-- Create database and user
CREATE DATABASE employee_prod;
CREATE USER employee_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE employee_prod TO employee_user;

-- Configure connection limits
ALTER USER employee_user CONNECTION LIMIT 100;

-- Enable required extensions
\c employee_prod
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
```

#### Database Migration
```bash
# Run migrations
python -m alembic upgrade head

# Create initial data
python scripts/create_initial_data.py

# Verify setup
python scripts/verify_database.py
```

### Redis Configuration

#### Development
```bash
# Using Docker
docker run -d \
  --name employee-redis-dev \
  -p 6379:6379 \
  redis:6-alpine
```

#### Production
```bash
# Redis configuration (redis.conf)
bind 0.0.0.0
port 6379
requirepass secure_redis_password
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

## Application Deployment

### Docker Deployment

#### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash employee
RUN chown -R employee:employee /app
USER employee

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "forge1.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose (Development)
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://dev_user:dev_pass@db:5432/employee_dev
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - .:/app
    command: uvicorn forge1.main:app --host 0.0.0.0 --port 8000 --reload

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=employee_dev
      - POSTGRES_USER=dev_user
      - POSTGRES_PASSWORD=dev_pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

#### Docker Compose (Production)
```yaml
version: '3.8'

services:
  app:
    image: employee-lifecycle:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - SECRET_KEY=${SECRET_KEY}
    env_file:
      - .env.prod
    depends_on:
      - db
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=${POSTGRES_DB}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgresql.conf:/etc/postgresql/postgresql.conf
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment

#### Namespace
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: employee-lifecycle
```

#### ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: employee-config
  namespace: employee-lifecycle
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  DATABASE_POOL_SIZE: "50"
  REDIS_POOL_SIZE: "50"
```

#### Secret
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: employee-secrets
  namespace: employee-lifecycle
type: Opaque
data:
  DATABASE_URL: <base64-encoded-database-url>
  REDIS_URL: <base64-encoded-redis-url>
  SECRET_KEY: <base64-encoded-secret-key>
  OPENAI_API_KEY: <base64-encoded-openai-key>
```

#### Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: employee-api
  namespace: employee-lifecycle
spec:
  replicas: 3
  selector:
    matchLabels:
      app: employee-api
  template:
    metadata:
      labels:
        app: employee-api
    spec:
      containers:
      - name: employee-api
        image: employee-lifecycle:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: employee-config
        - secretRef:
            name: employee-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: employee-api-service
  namespace: employee-lifecycle
spec:
  selector:
    app: employee-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

#### Ingress
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: employee-api-ingress
  namespace: employee-lifecycle
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.forge1.com
    secretName: employee-api-tls
  rules:
  - host: api.forge1.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: employee-api-service
            port:
              number: 80
```

## Monitoring and Health Checks

### Health Check Endpoints

The application provides several health check endpoints:

- `GET /health` - Basic health check
- `GET /ready` - Readiness check (includes dependencies)
- `GET /metrics` - Prometheus metrics
- `GET /performance/health` - Detailed system health

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'employee-api'
    static_configs:
      - targets: ['employee-api-service:9090']
    metrics_path: /metrics
    scrape_interval: 30s
```

### Grafana Dashboard

Key metrics to monitor:
- Request rate and response time
- Error rate and status codes
- Database connection pool usage
- Redis cache hit rate
- Memory and CPU usage
- Employee interaction metrics

### Alerting Rules

```yaml
# alerts.yml
groups:
  - name: employee-api
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: High error rate detected

      - alert: DatabaseConnectionsHigh
        expr: database_connections_active / database_connections_max > 0.8
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: Database connection pool usage high

      - alert: ResponseTimeHigh
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: 95th percentile response time is high
```

## Security Configuration

### SSL/TLS Configuration

#### NGINX SSL Configuration
```nginx
server {
    listen 443 ssl http2;
    server_name api.forge1.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    location / {
        proxy_pass http://app:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Security Headers

```python
# In FastAPI middleware
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

### Database Security

- Use SSL connections for all database traffic
- Implement connection pooling with limits
- Regular security updates and patches
- Database user with minimal required permissions
- Network isolation using VPCs/security groups

### API Security

- JWT token authentication with short expiration
- Rate limiting per client/IP
- Input validation and sanitization
- CORS configuration for allowed origins
- API key rotation policies

## Scaling and Performance

### Horizontal Scaling

#### Auto-scaling Configuration (Kubernetes)
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: employee-api-hpa
  namespace: employee-lifecycle
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: employee-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Database Scaling

#### Read Replicas
```python
# Database configuration for read replicas
DATABASE_URLS = {
    "write": "postgresql://user:pass@primary-db:5432/employee",
    "read": [
        "postgresql://user:pass@replica1-db:5432/employee",
        "postgresql://user:pass@replica2-db:5432/employee"
    ]
}
```

#### Connection Pooling
```python
# Advanced connection pool configuration
DATABASE_CONFIG = {
    "pool_size": 50,
    "max_overflow": 100,
    "pool_timeout": 30,
    "pool_recycle": 3600,
    "pool_pre_ping": True
}
```

### Caching Strategy

#### Redis Configuration
```python
# Multi-level caching
CACHE_CONFIG = {
    "employee_data": {"ttl": 3600, "max_size": 10000},
    "interaction_history": {"ttl": 1800, "max_size": 50000},
    "analytics_data": {"ttl": 300, "max_size": 5000}
}
```

## Troubleshooting

### Common Issues

#### Database Connection Issues
```bash
# Check database connectivity
pg_isready -h db-host -p 5432 -U username

# Monitor connection pool
SELECT count(*) as active_connections FROM pg_stat_activity;

# Check for long-running queries
SELECT query, state, query_start 
FROM pg_stat_activity 
WHERE state = 'active' 
ORDER BY query_start;
```

#### Redis Connection Issues
```bash
# Test Redis connectivity
redis-cli -h redis-host -p 6379 ping

# Monitor Redis memory usage
redis-cli info memory

# Check Redis slow log
redis-cli slowlog get 10
```

#### Application Performance Issues
```bash
# Check application logs
kubectl logs -f deployment/employee-api -n employee-lifecycle

# Monitor resource usage
kubectl top pods -n employee-lifecycle

# Check metrics endpoint
curl http://api.forge1.com/metrics
```

### Log Analysis

#### Structured Logging Format
```json
{
  "timestamp": "2024-01-20T18:00:00Z",
  "level": "INFO",
  "logger": "forge1.services.employee_manager",
  "message": "Employee created successfully",
  "client_id": "client_123",
  "employee_id": "emp_456",
  "processing_time_ms": 1250,
  "request_id": "req_789"
}
```

#### Log Aggregation (ELK Stack)
```yaml
# Filebeat configuration
filebeat.inputs:
- type: container
  paths:
    - '/var/lib/docker/containers/*/*.log'
  processors:
  - add_kubernetes_metadata:
      host: ${NODE_NAME}
      matchers:
      - logs_path:
          logs_path: "/var/lib/docker/containers/"

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "employee-api-%{+yyyy.MM.dd}"
```

### Performance Optimization

#### Database Optimization
```sql
-- Create indexes for common queries
CREATE INDEX CONCURRENTLY idx_employees_client_status 
ON employees(client_id, status) WHERE status = 'active';

CREATE INDEX CONCURRENTLY idx_interactions_employee_created 
ON interactions(employee_id, created_at DESC);

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM employees WHERE client_id = 'client_123';
```

#### Application Optimization
```python
# Connection pool tuning
async def optimize_db_pool():
    # Monitor pool usage
    pool_stats = await db_manager.get_pool_stats()
    
    if pool_stats.utilization > 0.8:
        logger.warning("High database pool utilization", extra=pool_stats)
    
    # Adjust pool size dynamically
    if pool_stats.queue_size > 10:
        await db_manager.increase_pool_size()
```

### Backup and Recovery

#### Database Backup
```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME \
  --format=custom \
  --compress=9 \
  --file="$BACKUP_DIR/employee_db_$(date +%H%M%S).dump"

# Upload to S3
aws s3 cp $BACKUP_DIR s3://employee-backups/ --recursive
```

#### Disaster Recovery
```bash
# Database restoration
pg_restore -h $DB_HOST -U $DB_USER -d $DB_NAME \
  --clean --if-exists \
  /path/to/backup.dump

# Application state recovery
python scripts/restore_application_state.py \
  --backup-date 2024-01-20 \
  --verify-integrity
```

This deployment guide provides comprehensive instructions for deploying the Employee Lifecycle System across different environments with proper security, monitoring, and scaling considerations.