# Azure Monitor Integration for Forge1

This document provides comprehensive information about the Microsoft Azure Monitor OpenTelemetry integration in Forge1, including setup, configuration, usage, and troubleshooting.

## Overview

The Azure Monitor integration provides enterprise-grade observability, monitoring, and analytics capabilities for Forge1 using Microsoft's official OpenTelemetry Azure Monitor Python SDK. This integration enables:

- **Comprehensive Telemetry**: Automatic collection of traces, metrics, logs, and custom events
- **Business Intelligence**: Advanced analytics and dashboards for tenant insights
- **Real-time Monitoring**: Live metrics and alerting capabilities
- **Tenant Isolation**: Secure, multi-tenant telemetry with proper data isolation
- **Cost Tracking**: Detailed usage and cost analytics per tenant

## Architecture

### Components

1. **Azure Monitor Adapter** (`forge1/integrations/observability/azure_monitor.py`)
   - Core integration with Azure Monitor OpenTelemetry SDK
   - Handles telemetry export and configuration
   - Provides health checks and metrics

2. **Analytics Service** (`forge1/services/analytics/azure_monitor_analytics.py`)
   - Business intelligence and analytics capabilities
   - Pre-defined KQL queries for common insights
   - Dashboard data generation and cost analysis

3. **API Endpoints** (`forge1/api/v1/analytics.py`)
   - REST API for accessing analytics data
   - Custom query execution and dashboard endpoints
   - Alert rule management

4. **Middleware** (`forge1/middleware/azure_monitor_middleware.py`)
   - Automatic request telemetry collection
   - Performance metrics and error tracking
   - Tenant-aware context injection

## Installation and Setup

### Prerequisites

1. **Azure Monitor Workspace**: You need an Azure Monitor workspace with either:
   - Application Insights connection string, or
   - Application Insights instrumentation key

2. **Python Dependencies**: The integration requires:
   ```
   azure-monitor-opentelemetry>=1.2.0
   azure-monitor-query>=1.2.0
   azure-core>=1.29.0
   ```

### Configuration

#### Environment Variables

Set the following environment variables:

```bash
# Primary configuration (choose one)
export AZURE_MONITOR_CONNECTION_STRING="InstrumentationKey=your-key;IngestionEndpoint=https://your-region.in.applicationinsights.azure.com/"
# OR
export AZURE_MONITOR_INSTRUMENTATION_KEY="your-instrumentation-key"

# Optional configuration
export AZURE_MONITOR_SAMPLING_RATIO="1.0"  # 0.0 to 1.0
export AZURE_MONITOR_LIVE_METRICS="true"
export AZURE_MONITOR_STORAGE_DIR="/tmp/azure_monitor"
export AZURE_MONITOR_DISABLE_OFFLINE="false"
```

#### Configuration File

Alternatively, configure via the settings manager:

```python
from forge1.config.integration_settings import settings_manager, IntegrationType

config = {
    "azure_monitor": {
        "connection_string": "your-connection-string",
        "sampling_ratio": 1.0,
        "enable_live_metrics": True,
        "storage_directory": "/tmp/azure_monitor",
        "resource_attributes": {
            "service.name": "forge1-backend",
            "service.version": "1.0.0",
            "deployment.environment": "production"
        }
    }
}

settings_manager.update_config(IntegrationType.OBSERVABILITY, config)
```

## Usage

### Automatic Telemetry Collection

The integration automatically collects:

- **HTTP Requests**: All FastAPI requests with response times, status codes, and tenant context
- **Database Operations**: PostgreSQL queries and connection metrics
- **Cache Operations**: Redis operations and performance metrics
- **Custom Events**: Business events and user interactions
- **Errors and Exceptions**: Detailed error tracking with stack traces

### Manual Telemetry

#### Custom Events

```python
from forge1.integrations.observability.azure_monitor import azure_monitor_integration
from forge1.integrations.base_adapter import ExecutionContext, TenantContext

# Create context
context = ExecutionContext(
    tenant_context=TenantContext(
        tenant_id="tenant_123",
        user_id="user_456",
        employee_id="emp_789"
    ),
    request_id="req_001"
)

# Send custom event
await azure_monitor_integration.send_custom_event(
    "employee_interaction",
    {
        "interaction_type": "chat",
        "employee_name": "Customer Support Bot",
        "user_message": "How can I reset my password?",
        "response_time_ms": 250,
        "success": True
    },
    context
)
```

#### Custom Metrics

```python
# Send custom metric
await azure_monitor_integration.send_custom_metric(
    "employee_response_time",
    250.0,  # value in milliseconds
    {
        "employee_id": "emp_789",
        "interaction_type": "chat",
        "tenant_id": "tenant_123"
    },
    context
)
```

#### Custom Logs

```python
# Send structured log
await azure_monitor_integration.log_to_azure_monitor(
    "info",
    "Employee interaction completed successfully",
    {
        "employee_id": "emp_789",
        "user_id": "user_456",
        "interaction_duration_ms": 250,
        "tokens_used": 150,
        "cost_usd": 0.002
    },
    context
)
```

### Analytics and Business Intelligence

#### Pre-defined Analytics Queries

The integration includes several pre-defined KQL queries:

1. **Tenant Activity Summary**: Overall activity metrics per tenant
2. **Request Performance Analysis**: API performance and error rates
3. **Error Analysis**: Error patterns and affected users
4. **Usage Patterns**: User behavior and feature adoption
5. **Cost Analysis**: Usage costs and resource consumption

#### Execute Analytics Query

```python
from forge1.services.analytics.azure_monitor_analytics import (
    azure_monitor_analytics_service, AnalyticsTimeRange
)

# Execute predefined query
result = await azure_monitor_analytics_service.execute_analytics_query(
    "tenant_activity_summary",
    "tenant_123",
    AnalyticsTimeRange.LAST_24_HOURS
)

print(f"Query returned {result.total_records} records")
print(f"Execution time: {result.execution_time_ms}ms")
```

#### Custom Analytics Query

```python
# Execute custom KQL query
custom_query = """
customEvents
| where timestamp >= ago(1d)
| where customDimensions.tenant_id == "tenant_123"
| where name startswith "employee_"
| summarize count() by name, bin(timestamp, 1h)
| order by timestamp desc
"""

result = await azure_monitor_analytics_service.execute_custom_query(
    custom_query,
    "tenant_123",
    "employee_activity_hourly"
)
```

#### Dashboard Data

```python
# Get comprehensive dashboard data
dashboard_data = await azure_monitor_analytics_service.get_tenant_dashboard_data(
    "tenant_123",
    AnalyticsTimeRange.LAST_7_DAYS
)

print(f"Activity Summary: {len(dashboard_data['activity_summary'])} data points")
print(f"Performance Analysis: {len(dashboard_data['performance_analysis'])} endpoints")
print(f"Summary Metrics: {dashboard_data['summary_metrics']}")
```

### API Endpoints

#### Analytics API

```bash
# Get available queries
GET /analytics/queries/available

# Execute predefined query
GET /analytics/query/tenant_activity_summary?time_range=24h

# Execute custom query
POST /analytics/query/custom
{
    "query_name": "custom_analysis",
    "kql_query": "customEvents | where name == 'employee_interaction' | count",
    "time_range": "24h"
}

# Get tenant dashboard
GET /analytics/dashboard?time_range=7d

# Get cost insights
GET /analytics/cost-insights?time_range=30d

# Create alert rule
POST /analytics/alerts
{
    "alert_name": "High Error Rate",
    "kql_query": "requests | where success == false | count",
    "threshold": 10.0,
    "alert_type": "metric"
}
```

#### Azure Monitor API

```bash
# Health check
GET /api/v1/azure-monitor/health

# Send custom event
POST /api/v1/azure-monitor/events/custom
{
    "name": "business_event",
    "properties": {"event_type": "user_signup"},
    "measurements": {"signup_time_ms": 1500}
}

# Track employee interaction
POST /api/v1/azure-monitor/track/employee-interaction
{
    "employee_id": "emp_123",
    "employee_name": "Sales Assistant",
    "interaction_type": "chat",
    "user_message": "I need help with pricing",
    "response_message": "I'd be happy to help with pricing information...",
    "processing_time_ms": 300,
    "model_used": "gpt-4",
    "tokens_used": 150,
    "cost": 0.003,
    "success": true,
    "user_id": "user_456"
}
```

## Monitoring and Alerting

### Health Monitoring

```python
# Check Azure Monitor health
from forge1.integrations.observability.azure_monitor import azure_monitor_integration

health_result = await azure_monitor_integration.health_check()
print(f"Status: {health_result.status.value}")
print(f"Message: {health_result.message}")
print(f"Response Time: {health_result.response_time_ms}ms")
```

### Metrics and Statistics

```python
# Get integration metrics
metrics = azure_monitor_integration.get_metrics()
print(f"Azure Monitor Available: {metrics['azure_monitor_available']}")
print(f"Total Telemetry Items: {metrics['total_telemetry_items']}")
print(f"Export Failures: {metrics['export_failures']}")

# Get analytics service statistics
from forge1.services.analytics.azure_monitor_analytics import azure_monitor_analytics_service

stats = azure_monitor_analytics_service.get_service_statistics()
print(f"Queries Executed: {stats['queries_executed']}")
print(f"Cache Hit Rate: {stats['cache_hit_rate']:.2%}")
```

### Alert Rules

Create alert rules for important metrics:

```python
# High error rate alert
await azure_monitor_analytics_service.create_alert_rule(
    "tenant_123",
    "High Error Rate Alert",
    """
    requests
    | where timestamp >= ago(5m)
    | where customDimensions.tenant_id == "tenant_123"
    | where success == false
    | count
    """,
    10.0,  # Alert if more than 10 errors in 5 minutes
    "metric"
)

# High cost alert
await azure_monitor_analytics_service.create_alert_rule(
    "tenant_123",
    "High Usage Cost Alert",
    """
    customMetrics
    | where timestamp >= ago(1h)
    | where customDimensions.tenant_id == "tenant_123"
    | where name contains "cost"
    | summarize total_cost = sum(value)
    """,
    100.0,  # Alert if hourly cost exceeds $100
    "metric"
)
```

## Troubleshooting

### Common Issues

#### 1. Connection String Not Working

**Problem**: Azure Monitor not receiving telemetry

**Solutions**:
- Verify connection string format: `InstrumentationKey=xxx;IngestionEndpoint=https://xxx.in.applicationinsights.azure.com/`
- Check network connectivity to Azure endpoints
- Verify Application Insights resource is active

#### 2. High Memory Usage

**Problem**: Memory consumption increases over time

**Solutions**:
- Adjust sampling ratio: `AZURE_MONITOR_SAMPLING_RATIO=0.1`
- Enable offline storage: `AZURE_MONITOR_DISABLE_OFFLINE=false`
- Monitor telemetry volume and optimize custom events

#### 3. Slow Query Performance

**Problem**: Analytics queries taking too long

**Solutions**:
- Use time range filters in KQL queries
- Enable query result caching
- Optimize KQL queries with proper indexing

#### 4. Missing Telemetry Data

**Problem**: Expected telemetry not appearing in Azure Monitor

**Solutions**:
- Check health status: `GET /analytics/health`
- Verify tenant context is properly set
- Check export failure metrics
- Review application logs for errors

### Debugging

#### Enable Debug Logging

```python
import logging
logging.getLogger("forge1.integrations.observability").setLevel(logging.DEBUG)
logging.getLogger("azure.monitor").setLevel(logging.DEBUG)
```

#### Test Connectivity

```python
# Test Azure Monitor connectivity
from forge1.integrations.observability.azure_monitor import azure_monitor_integration

# Send test event
success = await azure_monitor_integration.send_custom_event(
    "connectivity_test",
    {"test": True, "timestamp": datetime.now().isoformat()}
)

print(f"Connectivity test: {'✅ Success' if success else '❌ Failed'}")
```

#### Run Integration Tests

```bash
# Run comprehensive test suite
python test_azure_monitor_integration.py
```

### Performance Optimization

#### Sampling Configuration

```python
# Configure sampling for high-volume environments
azure_monitor_config = {
    "sampling_ratio": 0.1,  # Sample 10% of telemetry
    "enable_live_metrics": False,  # Disable for production
    "disable_offline_storage": True  # Reduce disk usage
}
```

#### Batch Configuration

```python
# Optimize batch export settings
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Custom batch processor with optimized settings
batch_processor = BatchSpanProcessor(
    azure_monitor_trace_exporter,
    max_queue_size=2048,
    schedule_delay_millis=5000,
    max_export_batch_size=512
)
```

## Security Considerations

### Data Privacy

- **Tenant Isolation**: All telemetry includes tenant context for proper data isolation
- **PII Handling**: Avoid sending personally identifiable information in telemetry
- **Data Retention**: Configure appropriate retention policies in Azure Monitor

### Access Control

- **API Authentication**: Secure analytics endpoints with proper authentication
- **Role-Based Access**: Implement role-based access to analytics data
- **Audit Logging**: All analytics queries and configuration changes are audited

### Compliance

- **GDPR Compliance**: Ensure telemetry collection complies with data protection regulations
- **Data Residency**: Configure Azure Monitor regions for data residency requirements
- **Encryption**: All telemetry is encrypted in transit and at rest

## Best Practices

### Telemetry Design

1. **Structured Events**: Use consistent event schemas with proper dimensions
2. **Meaningful Names**: Use descriptive names for events and metrics
3. **Appropriate Sampling**: Balance telemetry volume with observability needs
4. **Context Enrichment**: Include relevant business context in telemetry

### Query Optimization

1. **Time Filters**: Always include time range filters in KQL queries
2. **Tenant Filters**: Filter by tenant ID early in queries
3. **Result Limits**: Use appropriate limits to prevent large result sets
4. **Caching**: Leverage query result caching for frequently accessed data

### Cost Management

1. **Sampling Strategy**: Implement intelligent sampling based on event importance
2. **Data Retention**: Configure appropriate retention periods
3. **Query Optimization**: Optimize KQL queries to reduce compute costs
4. **Monitoring**: Monitor Azure Monitor costs and usage patterns

## Support and Resources

### Documentation

- [Azure Monitor OpenTelemetry Documentation](https://docs.microsoft.com/en-us/azure/azure-monitor/app/opentelemetry-python)
- [KQL Query Language Reference](https://docs.microsoft.com/en-us/azure/data-explorer/kusto/query/)
- [OpenTelemetry Python Documentation](https://opentelemetry-python.readthedocs.io/)

### Monitoring

- Health Check Endpoint: `GET /analytics/health`
- Statistics Endpoint: `GET /analytics/statistics`
- Integration Test Suite: `python test_azure_monitor_integration.py`

### Support

For issues with the Azure Monitor integration:

1. Check the health endpoints and logs
2. Run the integration test suite
3. Review Azure Monitor ingestion logs
4. Contact the Forge1 development team with detailed error information

## Changelog

### Version 1.2.0
- Added comprehensive analytics service with pre-defined queries
- Implemented tenant-aware dashboard generation
- Added cost analysis and insights capabilities
- Enhanced API endpoints with custom query support
- Improved error handling and graceful degradation

### Version 1.1.0
- Added Azure Monitor middleware for automatic request telemetry
- Implemented custom event and metric tracking
- Added health checks and monitoring capabilities
- Enhanced tenant isolation and context management

### Version 1.0.0
- Initial Azure Monitor OpenTelemetry integration
- Basic telemetry export functionality
- Configuration management and environment variable support