# LlamaIndex and workflows-py Integration Guide

## Overview

This document describes the integration of LlamaIndex and workflows-py into the Forge1 platform as first-class citizens. The integration maintains strict architectural boundaries while providing powerful document processing, knowledge search, and workflow orchestration capabilities.

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Forge1 Core Systems                      │
├─────────────────────────────────────────────────────────────┤
│  Model Router │ Memory Manager │ Tenancy │ RBAC │ Secrets   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                 MCAE Orchestration Layer                    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│              workflows-py Sub-Orchestration                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │ Step 1  │→ │ Step 2  │→ │ Step 3  │→ │ Step 4  │       │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                LlamaIndex Tool Provider                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Doc Parser  │ │  KB Search  │ │ Drive Fetch │ ...      │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **MCAE Remains Top-Level Orchestrator**: All multi-employee workflows are coordinated by MCAE
2. **workflows-py for Sub-Orchestration**: Used within MCAE stages for explicit step graphs
3. **LlamaIndex as Tool Provider**: Provides document processing and retrieval capabilities
4. **Forge1 Authority**: Maintains control over tenancy, memory, routing, RBAC, and billing

## Component Locations

### Core Adapters

- **LlamaIndex Adapter**: `forge1/integrations/llamaindex_adapter.py`
  - Main integration point for LlamaIndex functionality
  - Handles tool creation, execution, and lifecycle management
  - Enforces RBAC and tenant isolation

- **workflows-py Adapter**: `forge1/integrations/workflows_adapter.py`
  - Manages workflow definitions and execution
  - Provides Forge1-aware workflow context
  - Handles error recovery and observability

- **Model Shim**: `forge1/integrations/llamaindex_model_shim.py`
  - Routes all LLM calls through Forge1's Model Router
  - Tracks usage metrics for billing
  - Maintains tenant/employee context

### Tools Implementation

- **Tool Implementations**: `forge1/integrations/llamaindex_tools.py`
  - Document Parser (PDF/DOCX with OCR fallback)
  - Knowledge Base Search (vector/hybrid retrieval)
  - Google Drive Fetch (read-only access)
  - Slack Post (tenant-scoped channels)

### Workflow Definitions

- **NDA Workflow**: `forge1/integrations/nda_workflow.py`
  - Complete legal document review workflow
  - Demonstrates all integration capabilities
  - Production-ready with error handling

### Supporting Systems

- **Error Handling**: `forge1/integrations/workflow_error_handling.py`
  - Circuit breakers, retries, dead-letter queues
  - Comprehensive error categorization and recovery

- **Observability**: `forge1/integrations/otel_integration.py`
  - OpenTelemetry tracing and metrics
  - Tenant/employee context propagation

- **Usage Tracking**: `forge1/integrations/usage_tracking.py`
  - LLM and tool usage metrics
  - CSV export for billing integration

## Context Flow

### Execution Context Structure

```python
@dataclass
class ExecutionContext:
    tenant_id: str          # Tenant isolation
    employee_id: str        # Employee identification
    role: str              # RBAC role
    request_id: str        # Request tracing
    case_id: str           # Business context
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
```

### Context Propagation Flow

1. **MCAE Stage Entry**: Context created with tenant/employee information
2. **workflows-py Handoff**: Context passed to workflow adapter
3. **Step Execution**: Each step receives full context
4. **Tool Invocation**: Tools enforce RBAC using context
5. **Model Routing**: LLM calls include context for routing decisions
6. **Memory Operations**: All reads/writes scoped by tenant/employee

## Adding New Tools

### 1. Create Tool Class

```python
class MyCustomTool(LlamaIndexTool):
    def __init__(self, **kwargs):
        super().__init__(tool_type=ToolType.MY_TOOL, **kwargs)
    
    async def acall(self, context: ExecutionContext, **kwargs) -> Dict[str, Any]:
        # Enforce RBAC
        await self._enforce_rbac(context, "my_resource:action")
        
        # Get tenant secrets if needed
        secret = await self._get_tenant_secret("my_secret", context)
        
        # Perform tool operation
        result = await self._do_work(kwargs)
        
        # Apply DLP redaction
        redacted_result = await self._redact_output(result)
        
        return {"success": True, "data": redacted_result}
```

### 2. Register Tool Type

```python
# In llamaindex_adapter.py
class ToolType(Enum):
    # ... existing tools
    MY_TOOL = "my_tool"

# In tool_types mapping
self.tool_types = {
    # ... existing mappings
    ToolType.MY_TOOL: MyCustomTool
}
```

### 3. Configure RBAC Permissions

```python
# Define required permission
PERMISSION = "my_resource:action"

# Add to employee role permissions
employee_permissions = [
    "my_resource:action",  # New permission
    # ... other permissions
]
```

## Adding New Workflow Steps

### 1. Define Step Function

```python
@step
async def my_custom_step(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Custom workflow step"""
    try:
        # Get context
        context = self.context.execution_context
        
        # Use tools
        tool_result = await self.my_tool.acall(
            context=context,
            **input_data
        )
        
        # Update handoff packet
        packet = self.context.get_handoff_packet()
        packet.metadata['my_step_output'] = tool_result
        
        return tool_result
        
    except Exception as e:
        logger.error(f"Custom step failed: {e}")
        raise
```

### 2. Add to Workflow Class

```python
@workflow
class MyWorkflow(Forge1Workflow):
    async def _execute_steps(self) -> bool:
        # ... existing steps
        
        # Add new step
        my_result = await self._execute_step_with_retry(
            "my_custom_step",
            self.my_custom_step,
            input_data
        )
        
        if my_result.status != StepStatus.COMPLETED:
            return False
        
        # ... continue workflow
        return True
```

### 3. Register Workflow

```python
# Register with workflows adapter
await workflows_adapter.register_workflow_definition(
    workflow_name="my_workflow",
    workflow_class=MyWorkflow
)
```

## Configuration

### Environment Variables

```bash
# LlamaIndex Configuration
LLAMAINDEX_MODEL_PREFERENCE=gpt-4o
LLAMAINDEX_EMBEDDING_MODEL=text-embedding-ada-002

# workflows-py Configuration  
WORKFLOWS_MAX_RETRIES=3
WORKFLOWS_RETRY_DELAY=1.0
WORKFLOWS_TIMEOUT=300

# Integration Settings
ENABLE_OCR_FALLBACK=true
CIRCUIT_BREAKER_THRESHOLD=5
DEAD_LETTER_QUEUE_ENABLED=true
```

### Tenant Secrets Configuration

```python
# Required secrets per tenant
REQUIRED_SECRETS = {
    "google_drive_credentials": "Google Drive service account JSON",
    "slack_bot_token": "Slack bot token for workspace",
    "openai_api_key": "OpenAI API key (if not using global)",
}
```

## Monitoring and Observability

### OpenTelemetry Spans

All operations create spans with standardized attributes:

```python
span_attributes = {
    "tenant.id": context.tenant_id,
    "employee.id": context.employee_id,
    "service.name": "forge1-workflows",
    "operation.type": "tool_execution",
    "tool.name": tool_name,
    # ... operation-specific attributes
}
```

### Usage Metrics

Track the following metrics for billing and monitoring:

- **LLM Usage**: model, tokens_in, tokens_out, latency_ms, cost_estimate
- **Tool Usage**: tool_name, execution_time_ms, success_rate
- **Workflow Usage**: workflow_type, step_count, completion_rate

### Health Checks

Monitor integration health via endpoints:

```bash
GET /health/llamaindex    # LlamaIndex adapter status
GET /health/workflows     # workflows-py adapter status
GET /health/integration   # Overall integration status
```

## Security Considerations

### RBAC Enforcement

Every tool operation checks permissions:

```python
# Permission format: {resource}:{action}
PERMISSIONS = {
    "document:read",           # Document parsing
    "knowledge_base:search",   # KB search
    "drive:read",             # Google Drive access
    "slack:post",             # Slack posting
}
```

### Tenant Isolation

- All memory operations use tenant-prefixed namespaces
- Secrets are scoped per tenant
- Cross-tenant access is prevented at the adapter level
- Audit logs track all access attempts

### PII/DLP Protection

- All tool outputs processed through redaction hooks
- Configurable redaction rules per tenant
- No PII in logs or error messages
- Sensitive data patterns automatically detected

## Troubleshooting

### Common Issues

1. **Tool Permission Denied**
   - Check employee role has required permission
   - Verify tenant context is set correctly
   - Review audit logs for RBAC violations

2. **Workflow Step Failures**
   - Check circuit breaker status for external services
   - Review dead letter queue for failed workflows
   - Verify secrets are configured for tenant

3. **Model Routing Issues**
   - Check employee model preferences
   - Verify model router configuration
   - Review usage tracking for routing decisions

### Debug Commands

```bash
# Check integration status
python -m forge1.integrations.llamaindex_adapter health_check

# Verify tool permissions
python -m forge1.integrations.rbac_checker check_permissions

# Export usage metrics
python -m forge1.integrations.usage_tracking export_csv

# Run integration tests
pytest tests/test_llamaindex_workflows_integration.py -v
```

### Log Analysis

Key log patterns to monitor:

```bash
# Successful tool execution
grep "Tool execution completed" /var/log/forge1/integration.log

# RBAC violations
grep "Permission denied" /var/log/forge1/security.log

# Circuit breaker events
grep "Circuit breaker" /var/log/forge1/resilience.log

# Workflow completions
grep "Workflow.*completed" /var/log/forge1/workflows.log
```

## Performance Optimization

### Caching Strategy

- **Model Responses**: 1 hour TTL for identical prompts
- **Document Parsing**: 24 hour TTL for parsed content
- **KB Search Results**: 30 minute TTL for common queries

### Resource Limits

- **Connection Pools**: Max 10 concurrent per external service
- **Memory Limits**: Max 100MB per workflow execution
- **Timeouts**: 30s per tool call, 5min per workflow

### Scaling Considerations

- Tools are stateless and horizontally scalable
- Workflow state persisted in Forge1 memory system
- Circuit breakers prevent cascade failures
- Dead letter queues handle overflow

## Migration and Upgrades

### Library Version Updates

1. Update `pyproject.toml` with new versions
2. Run compatibility verification script
3. Update integration tests
4. Deploy with canary rollout

### Breaking Changes

When updating major versions:

1. Review API compatibility in adapters
2. Update tool implementations if needed
3. Test with existing workflows
4. Update documentation and examples

### Rollback Procedures

1. Revert to previous library versions
2. Restart integration services
3. Verify existing workflows continue
4. Check health endpoints for status

## Support and Maintenance

### Regular Maintenance Tasks

- Monitor usage metrics and costs
- Review and rotate tenant secrets
- Update knowledge base content
- Optimize workflow performance

### Escalation Procedures

1. **Tool Failures**: Check external service status
2. **Workflow Errors**: Review dead letter queue
3. **Performance Issues**: Analyze usage patterns
4. **Security Incidents**: Review audit logs

For additional support, consult the Forge1 platform documentation or contact the integration team.