# Employee Lifecycle System - API Documentation

Comprehensive API documentation for the Employee Lifecycle System, providing detailed information about all endpoints, request/response formats, and usage examples.

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Client Management](#client-management)
4. [Employee Management](#employee-management)
5. [Employee Interactions](#employee-interactions)
6. [Memory Management](#memory-management)
7. [Analytics and Monitoring](#analytics-and-monitoring)
8. [Performance and Health](#performance-and-health)
9. [Error Handling](#error-handling)
10. [Rate Limiting](#rate-limiting)
11. [SDK Examples](#sdk-examples)

## Overview

The Employee Lifecycle System API provides comprehensive endpoints for managing AI employees throughout their entire lifecycle, from creation and configuration to interactions and analytics.

### Base URL
```
Production: https://api.forge1.com/v1
Staging: https://staging-api.forge1.com/v1
Development: http://localhost:8000/api/v1
```

### API Version
Current version: `v1`

### Content Type
All requests and responses use `application/json` content type.

## Authentication

### Bearer Token Authentication
All API requests require a valid Bearer token in the Authorization header.

```http
Authorization: Bearer <your_api_token>
```

### Required Headers
```http
Content-Type: application/json
Authorization: Bearer <your_api_token>
X-Tenant-ID: <your_tenant_id>
X-Client-ID: <your_client_id>
X-User-ID: <your_user_id>
```

### Tenant Isolation
The system enforces strict tenant isolation. All requests must include the appropriate tenant and client identifiers.

## Client Management

### Create Client

Create a new client organization.

**Endpoint:** `POST /employees/clients`

**Request Body:**
```json
{
  "name": "Acme Corporation",
  "industry": "Technology",
  "tier": "enterprise",
  "max_employees": 50,
  "allowed_models": ["gpt-4", "gpt-3.5-turbo"],
  "security_level": "high",
  "compliance_requirements": ["SOC2", "GDPR"]
}
```

**Response:**
```json
{
  "id": "client_abc123",
  "name": "Acme Corporation",
  "industry": "Technology",
  "tier": "enterprise",
  "max_employees": 50,
  "current_employees": 0,
  "allowed_models": ["gpt-4", "gpt-3.5-turbo"],
  "security_level": "high",
  "compliance_requirements": ["SOC2", "GDPR"],
  "created_at": "2024-01-15T10:30:00Z",
  "status": "active"
}
```

### Get Client Details

Retrieve client information.

**Endpoint:** `GET /employees/clients/{client_id}`

**Response:**
```json
{
  "id": "client_abc123",
  "name": "Acme Corporation",
  "industry": "Technology",
  "tier": "enterprise",
  "max_employees": 50,
  "current_employees": 5,
  "allowed_models": ["gpt-4", "gpt-3.5-turbo"],
  "security_level": "high",
  "compliance_requirements": ["SOC2", "GDPR"],
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-20T14:22:00Z",
  "status": "active"
}
```

## Employee Management

### Create Employee

Create a new AI employee for a client.

**Endpoint:** `POST /employees/clients/{client_id}/employees`

**Request Body:**
```json
{
  "role": "Customer Support Specialist",
  "industry": "Technology",
  "expertise_areas": [
    "customer_service",
    "technical_support",
    "product_knowledge"
  ],
  "communication_style": "friendly",
  "tools_needed": [
    "email",
    "chat",
    "knowledge_base"
  ],
  "knowledge_domains": [
    "product_documentation",
    "troubleshooting_guides"
  ],
  "personality_traits": {
    "empathy_level": 0.9,
    "patience_level": 0.95,
    "technical_depth": 0.8
  },
  "model_preferences": {
    "primary_model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 2000
  }
}
```

**Response:**
```json
{
  "id": "emp_xyz789",
  "client_id": "client_abc123",
  "name": "Sarah Support",
  "role": "Customer Support Specialist",
  "status": "active",
  "personality": {
    "communication_style": "friendly",
    "formality_level": "casual",
    "expertise_level": "intermediate",
    "response_length": "detailed",
    "creativity_level": 0.7,
    "empathy_level": 0.9,
    "custom_traits": {
      "patience_level": 0.95,
      "technical_depth": 0.8
    }
  },
  "model_preferences": {
    "primary_model": "gpt-4",
    "fallback_models": ["gpt-3.5-turbo"],
    "temperature": 0.7,
    "max_tokens": 2000,
    "specialized_models": {}
  },
  "tool_access": [
    "email",
    "chat",
    "knowledge_base"
  ],
  "knowledge_sources": [
    "kb_001",
    "kb_002"
  ],
  "created_at": "2024-01-15T11:00:00Z",
  "updated_at": "2024-01-15T11:00:00Z"
}
```

### Get Employee Details

Retrieve detailed information about an employee.

**Endpoint:** `GET /employees/clients/{client_id}/employees/{employee_id}`

**Response:**
```json
{
  "id": "emp_xyz789",
  "client_id": "client_abc123",
  "name": "Sarah Support",
  "role": "Customer Support Specialist",
  "status": "active",
  "personality": {
    "communication_style": "friendly",
    "formality_level": "casual",
    "expertise_level": "intermediate",
    "response_length": "detailed",
    "creativity_level": 0.7,
    "empathy_level": 0.9,
    "custom_traits": {
      "patience_level": 0.95,
      "technical_depth": 0.8
    }
  },
  "model_preferences": {
    "primary_model": "gpt-4",
    "fallback_models": ["gpt-3.5-turbo"],
    "temperature": 0.7,
    "max_tokens": 2000,
    "specialized_models": {}
  },
  "tool_access": [
    "email",
    "chat",
    "knowledge_base"
  ],
  "knowledge_sources": [
    "kb_001",
    "kb_002"
  ],
  "created_at": "2024-01-15T11:00:00Z",
  "updated_at": "2024-01-20T09:15:00Z",
  "last_interaction_at": "2024-01-20T16:45:00Z"
}
```

### Update Employee

Update employee configuration.

**Endpoint:** `PUT /employees/clients/{client_id}/employees/{employee_id}`

**Request Body:**
```json
{
  "name": "Sarah Advanced Support",
  "personality": {
    "empathy_level": 0.95,
    "technical_depth": 0.9
  },
  "model_preferences": {
    "temperature": 0.6
  }
}
```

### List Employees

Get a list of all employees for a client.

**Endpoint:** `GET /employees/clients/{client_id}/employees`

**Query Parameters:**
- `status` (optional): Filter by status (`active`, `inactive`, `archived`)
- `role` (optional): Filter by role
- `limit` (optional): Number of results (default: 50, max: 100)
- `offset` (optional): Pagination offset (default: 0)

**Response:**
```json
{
  "employees": [
    {
      "id": "emp_xyz789",
      "name": "Sarah Support",
      "role": "Customer Support Specialist",
      "status": "active",
      "created_at": "2024-01-15T11:00:00Z",
      "last_interaction_at": "2024-01-20T16:45:00Z"
    }
  ],
  "pagination": {
    "total": 5,
    "limit": 50,
    "offset": 0,
    "has_more": false
  }
}
```

### Search Employees

Search employees by various criteria.

**Endpoint:** `GET /employees/clients/{client_id}/employees/search`

**Query Parameters:**
- `query` (required): Search query
- `limit` (optional): Number of results (default: 10, max: 50)

**Response:**
```json
{
  "results": [
    {
      "id": "emp_xyz789",
      "name": "Sarah Support",
      "role": "Customer Support Specialist",
      "relevance_score": 0.95,
      "match_reasons": [
        "Role matches 'support'",
        "Expertise in customer service"
      ]
    }
  ],
  "query": "customer support",
  "total_results": 1
}
```

### Delete Employee

Soft delete or archive an employee.

**Endpoint:** `DELETE /employees/clients/{client_id}/employees/{employee_id}`

**Query Parameters:**
- `archive_only` (optional): If true, archives instead of deleting (default: true)

**Response:**
```json
{
  "success": true,
  "message": "Employee archived successfully",
  "employee_id": "emp_xyz789",
  "archived_at": "2024-01-20T17:00:00Z"
}
```

## Employee Interactions

### Interact with Employee

Send a message to an employee and receive a response.

**Endpoint:** `POST /employees/clients/{client_id}/employees/{employee_id}/interact`

**Request Body:**
```json
{
  "message": "Hello, I need help with my account setup",
  "session_id": "session_abc123",
  "context": {
    "user_type": "customer",
    "urgency": "medium",
    "previous_interactions": 2
  },
  "include_memory": true,
  "memory_limit": 10
}
```

**Response:**
```json
{
  "interaction_id": "int_def456",
  "message": "Hello! I'd be happy to help you with your account setup. Could you please tell me what specific aspect you're having trouble with?",
  "processing_time_ms": 1250,
  "tokens_used": 45,
  "cost": 0.0023,
  "model_used": "gpt-4",
  "confidence_score": 0.92,
  "memory_used": [
    {
      "id": "mem_001",
      "relevance": 0.85,
      "content": "Previous account setup question"
    }
  ],
  "session_id": "session_abc123",
  "timestamp": "2024-01-20T17:15:00Z"
}
```

## Memory Management

### Get Employee Memory

Retrieve employee's memory entries.

**Endpoint:** `GET /employees/clients/{client_id}/employees/{employee_id}/memory`

**Query Parameters:**
- `limit` (optional): Number of memories (default: 20, max: 100)
- `memory_type` (optional): Filter by type (`interaction`, `knowledge`, `context`)
- `query` (optional): Search memories by content
- `importance_threshold` (optional): Minimum importance score (0.0-1.0)

**Response:**
```json
{
  "memories": [
    {
      "id": "mem_001",
      "content": "Customer asked about account setup procedures",
      "memory_type": "interaction",
      "importance_score": 0.85,
      "context": {
        "topic": "account_setup",
        "sentiment": "neutral",
        "resolution": "in_progress"
      },
      "created_at": "2024-01-20T17:15:00Z"
    }
  ],
  "pagination": {
    "total": 25,
    "limit": 20,
    "has_more": true
  }
}
```

### Add Knowledge Source

Add a knowledge source to an employee.

**Endpoint:** `POST /employees/clients/{client_id}/employees/{employee_id}/knowledge-sources`

**Request Body:**
```json
{
  "title": "Product User Manual",
  "content": "Comprehensive guide covering all product features...",
  "source_type": "document",
  "keywords": ["setup", "features", "troubleshooting"],
  "metadata": {
    "version": "2.1",
    "last_updated": "2024-01-15"
  }
}
```

**Response:**
```json
{
  "knowledge_id": "kb_003",
  "title": "Product User Manual",
  "source_type": "document",
  "status": "processed",
  "chunks_created": 15,
  "processing_time_ms": 2500,
  "created_at": "2024-01-20T17:30:00Z"
}
```

## Analytics and Monitoring

### Employee Metrics

Get performance metrics for an employee.

**Endpoint:** `GET /analytics/employees/{client_id}/{employee_id}/metrics`

**Query Parameters:**
- `days` (optional): Number of days to include (default: 7, max: 90)
- `granularity` (optional): Data granularity (`hour`, `day`, `week`)

**Response:**
```json
{
  "metrics": {
    "total_interactions": 150,
    "avg_response_time_ms": 1200,
    "success_rate": 0.96,
    "customer_satisfaction": 4.7,
    "cost_per_interaction": 0.025,
    "tokens_per_interaction": 85,
    "performance_trend": "improving"
  },
  "time_series": [
    {
      "timestamp": "2024-01-20T00:00:00Z",
      "interactions": 25,
      "avg_response_time": 1150,
      "success_rate": 0.96
    }
  ],
  "period": {
    "start": "2024-01-14T00:00:00Z",
    "end": "2024-01-20T23:59:59Z",
    "days": 7
  }
}
```

### Employee Health

Get health status and diagnostics for an employee.

**Endpoint:** `GET /analytics/employees/{client_id}/{employee_id}/health`

**Response:**
```json
{
  "overall_health_score": 0.92,
  "health_status": "excellent",
  "component_scores": {
    "response_quality": 0.94,
    "response_time": 0.88,
    "memory_efficiency": 0.91,
    "cost_efficiency": 0.95,
    "error_rate": 0.98
  },
  "alerts": [],
  "recommendations": [
    "Consider optimizing memory retrieval for faster responses"
  ],
  "last_check": "2024-01-20T17:45:00Z"
}
```

### Client Usage Analytics

Get usage analytics for a client.

**Endpoint:** `GET /analytics/clients/{client_id}/usage`

**Query Parameters:**
- `days` (optional): Number of days to include (default: 30)

**Response:**
```json
{
  "summary": {
    "total_employees": 5,
    "active_employees": 4,
    "total_interactions": 1250,
    "total_cost": 31.25,
    "avg_cost_per_interaction": 0.025,
    "customer_satisfaction": 4.6
  },
  "employee_breakdown": [
    {
      "employee_id": "emp_xyz789",
      "name": "Sarah Support",
      "interactions": 450,
      "cost": 11.25,
      "satisfaction": 4.8
    }
  ],
  "trends": {
    "interaction_growth": 0.15,
    "cost_trend": "stable",
    "satisfaction_trend": "improving"
  }
}
```

## Performance and Health

### System Health

Check overall system health.

**Endpoint:** `GET /performance/health`

**Response:**
```json
{
  "overall_score": 0.95,
  "status": "healthy",
  "components": {
    "database": {
      "status": "healthy",
      "response_time_ms": 25,
      "connections": 15
    },
    "cache": {
      "status": "healthy",
      "hit_rate": 0.87,
      "memory_usage": 0.65
    },
    "llm_services": {
      "status": "healthy",
      "avg_response_time_ms": 1200,
      "success_rate": 0.99
    }
  },
  "timestamp": "2024-01-20T18:00:00Z"
}
```

### Performance Metrics

Get system performance metrics.

**Endpoint:** `GET /performance/metrics`

**Response:**
```json
{
  "metrics": {
    "requests_per_second": 45.2,
    "avg_response_time_ms": 850,
    "error_rate": 0.002,
    "cpu_usage": 0.35,
    "memory_usage": 0.68,
    "disk_usage": 0.42
  },
  "thresholds": {
    "response_time_warning": 2000,
    "response_time_critical": 5000,
    "error_rate_warning": 0.01,
    "error_rate_critical": 0.05
  },
  "timestamp": "2024-01-20T18:00:00Z"
}
```

## Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
  "error": {
    "code": "EMPLOYEE_NOT_FOUND",
    "message": "Employee with ID 'emp_invalid' not found",
    "details": {
      "employee_id": "emp_invalid",
      "client_id": "client_abc123"
    },
    "timestamp": "2024-01-20T18:15:00Z",
    "request_id": "req_123456"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Request validation failed |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `CLIENT_NOT_FOUND` | 404 | Client not found |
| `EMPLOYEE_NOT_FOUND` | 404 | Employee not found |
| `EMPLOYEE_LIMIT_EXCEEDED` | 409 | Client employee limit reached |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

## Rate Limiting

### Rate Limits

| Endpoint Type | Limit | Window |
|---------------|-------|--------|
| Authentication | 10 requests | 1 minute |
| Employee Creation | 5 requests | 1 minute |
| Employee Interactions | 100 requests | 1 minute |
| Analytics | 50 requests | 1 minute |
| General API | 1000 requests | 1 hour |

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642694400
X-RateLimit-Window: 60
```

## SDK Examples

### Python SDK Example

```python
from forge1_client import Forge1Client

# Initialize client
client = Forge1Client(
    api_key="your_api_key",
    tenant_id="your_tenant_id",
    base_url="https://api.forge1.com/v1"
)

# Create a client
client_data = {
    "name": "My Company",
    "industry": "Technology",
    "tier": "enterprise"
}
new_client = client.clients.create(client_data)

# Create an employee
employee_requirements = {
    "role": "Customer Support",
    "communication_style": "friendly",
    "expertise_areas": ["customer_service"]
}
employee = client.employees.create(
    client_id=new_client.id,
    requirements=employee_requirements
)

# Interact with employee
response = client.employees.interact(
    client_id=new_client.id,
    employee_id=employee.id,
    message="Hello, I need help",
    session_id="session_123"
)

print(f"Employee response: {response.message}")
```

### JavaScript SDK Example

```javascript
import { Forge1Client } from '@forge1/client';

// Initialize client
const client = new Forge1Client({
  apiKey: 'your_api_key',
  tenantId: 'your_tenant_id',
  baseUrl: 'https://api.forge1.com/v1'
});

// Create a client
const clientData = {
  name: 'My Company',
  industry: 'Technology',
  tier: 'enterprise'
};
const newClient = await client.clients.create(clientData);

// Create an employee
const employeeRequirements = {
  role: 'Customer Support',
  communicationStyle: 'friendly',
  expertiseAreas: ['customer_service']
};
const employee = await client.employees.create(
  newClient.id,
  employeeRequirements
);

// Interact with employee
const response = await client.employees.interact(
  newClient.id,
  employee.id,
  {
    message: 'Hello, I need help',
    sessionId: 'session_123'
  }
);

console.log(`Employee response: ${response.message}`);
```

### cURL Examples

#### Create Client
```bash
curl -X POST https://api.forge1.com/v1/employees/clients \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: your_tenant_id" \
  -d '{
    "name": "My Company",
    "industry": "Technology",
    "tier": "enterprise"
  }'
```

#### Create Employee
```bash
curl -X POST https://api.forge1.com/v1/employees/clients/client_123/employees \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: your_tenant_id" \
  -d '{
    "role": "Customer Support",
    "communication_style": "friendly",
    "expertise_areas": ["customer_service"]
  }'
```

#### Interact with Employee
```bash
curl -X POST https://api.forge1.com/v1/employees/clients/client_123/employees/emp_456/interact \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: your_tenant_id" \
  -d '{
    "message": "Hello, I need help with my account",
    "session_id": "session_123"
  }'
```

## Support and Resources

- **API Status**: https://status.forge1.com
- **Documentation**: https://docs.forge1.com
- **Support**: support@forge1.com
- **Community**: https://community.forge1.com

For additional help or questions about the API, please contact our support team or visit our documentation portal.