# Implementation Plan

- [x] 1. Setup project structure and core integration interfaces
  - Create directory structure for all OSS integration modules
  - Define base adapter interfaces that establish integration boundaries
  - Create configuration management for all OSS systems
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 2. Implement Celery and Redis queue integration
- [x] 2.1 Create Celery adapter with tenant-aware configuration
  - Write CeleryAdapter class with tenant context isolation
  - Implement task serialization with tenant and compliance metadata
  - Create retry policies with exponential backoff and circuit breaker
  - Write unit tests for adapter functionality
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 2.2 Implement Redis adapter for broker and caching
  - Write RedisAdapter class with tenant-scoped key namespacing
  - Implement connection pooling and failover mechanisms
  - Create tenant isolation verification tests
  - Write performance benchmarks for Redis operations
  - _Requirements: 1.1, 1.4_

- [x] 2.3 Create Celery task definitions for MCAE and workflows
  - Implement execute_mcae_stage task with context preservation
  - Create process_document_embedding task for heavy IO operations
  - Write execute_llamaindex_tool task wrapper
  - Implement dead letter queue handling for failed tasks
  - Write integration tests with failure injection scenarios
  - _Requirements: 1.1, 1.2, 1.5_

- [x] 3. Implement Weaviate vector database integration
- [x] 3.1 Create Weaviate adapter with tenant-aware schemas
  - Write WeaviateAdapter class with tenant-scoped collections
  - Implement schema creation and management per tenant
  - Create vector storage and retrieval methods with tenant filtering
  - Write unit tests for tenant isolation verification
  - _Requirements: 2.1, 2.3_

- [x] 3.2 Implement MemGPT memory compression service
  - Write MemGPTSummarizer class with compression algorithms
  - Implement TTL policy enforcement and archival mechanisms
  - Create Celery tasks for scheduled memory compression
  - Write compression fidelity verification tests
  - _Requirements: 2.2, 2.5_

- [x] 3.3 Create enhanced memory store adapter
  - Write WeaviateMemoryAdapter implementing MemoryAdapter interface
  - Integrate MemGPT compression with memory lifecycle management
  - Implement cross-tenant isolation verification
  - Write performance tests for memory operations
  - _Requirements: 2.1, 2.3, 2.4_

- [x] 4. Implement OpenTelemetry observability integration
- [x] 4.1 Create OTEL instrumentation setup
  - Write OTELIntegration class with tracer and meter providers
  - Implement FastAPI instrumentation with tenant context injection
  - Create Celery instrumentation for task tracing
  - Write database call instrumentation
  - _Requirements: 3.1, 3.5_

- [x] 4.2 Implement custom business metrics collection
  - Write ForgeMetrics class with business-specific metrics
  - Create metrics for task execution, model usage, and policy decisions
  - Implement tenant-scoped metric labeling
  - Write unit tests for metric collection accuracy
  - _Requirements: 3.2, 3.4_

- [x] 4.3 Create Prometheus metrics exposure
  - Implement /metrics endpoint with proper security
  - Create custom Prometheus exporters for Forge1 metrics
  - Write scrape configuration for all services
  - Implement metric aggregation and retention policies
  - _Requirements: 3.2, 3.4_

- [x] 5. Implement Grafana dashboard integration
- [x] 5.1 Create Grafana dashboard management
  - Write GrafanaDashboardManager for programmatic dashboard creation
  - Implement tenant-specific dashboard provisioning
  - Create dashboard templates for common monitoring scenarios
  - Write automated dashboard update mechanisms
  - _Requirements: 3.2, 3.4_

- [x] 5.2 Implement alerting and notification system
  - Create alert rule definitions for critical system metrics
  - Implement tenant-aware alerting with proper routing
  - Write alert escalation and notification logic
  - Create integration tests for alert triggering scenarios
  - _Requirements: 3.2, 3.4_

- [x] 6. Implement OpenMeter usage metering integration
- [x] 6.1 Create OpenMeter adapter and client
  - Write OpenMeterAdapter class with HTTP/Kafka support
  - Implement usage event emission with tenant context
  - Create batch processing for high-volume events
  - Write unit tests for event serialization and transmission
  - _Requirements: 4.1, 4.3_

- [x] 6.2 Implement usage event service
  - Write UsageEventService for standardized event generation
  - Create event types for model usage, tool execution, and storage
  - Implement cost estimation algorithms for different resource types
  - Write integration tests for event accuracy verification
  - _Requirements: 4.1, 4.2_

- [x] 6.3 Create usage reporting and export functionality
  - Implement monthly CSV export per tenant and employee
  - Create /reports/usage API endpoint with proper authentication
  - Write usage aggregation and summary generation
  - Create automated report generation and delivery
  - _Requirements: 4.2, 4.4_

- [x] 7. Implement OPA policy engine integration
- [x] 7.1 Create OPA adapter and client
  - Write OPAAdapter class with policy evaluation capabilities
  - Implement policy bundle loading and management
  - Create policy caching for performance optimization
  - Write unit tests for policy evaluation accuracy
  - _Requirements: 5.1, 5.5_

- [x] 7.2 Implement policy enforcement middleware
  - Write PolicyEnforcementMiddleware for FastAPI integration
  - Create policy evaluation for tool access and document access
  - Implement audit logging for all policy decisions
  - Write integration tests for policy enforcement scenarios
  - _Requirements: 5.1, 5.2, 5.4_

- [x] 7.3 Create Rego policy definitions
  - Write tool_access.rego policy for tool execution permissions
  - Create doc_access.rego policy for document access control
  - Implement routing_constraints.rego for model and resource routing
  - Write policy unit tests with positive and negative test cases
  - _Requirements: 5.1, 5.3_

- [x] 8. Implement DLP and security integration
- [x] 8.1 Enhance DLP redaction for all integrated systems
  - Update redaction to work with Celery task payloads
  - Implement trace payload sanitization for OTEL
  - Create redaction for usage events and policy inputs
  - Write comprehensive DLP effectiveness tests
  - _Requirements: 7.1, 7.2, 7.5_

- [x] 8.2 Implement secrets management integration
  - Update SecretManager to handle all OSS system credentials
  - Create secure configuration loading for all adapters
  - Implement credential rotation mechanisms
  - Write security tests for credential handling
  - _Requirements: 7.3, 7.5_

- [x] 9. Create configuration and deployment setup
- [x] 9.1 Implement configuration management
  - Create configuration classes for all OSS integrations
  - Implement environment-specific configuration loading
  - Write configuration validation and error handling
  - Create configuration documentation and examples
  - _Requirements: 6.1, 7.3_

- [x] 9.2 Create Docker and Kubernetes deployment configurations
  - Write Docker Compose configuration for local development
  - Create Kubernetes manifests for production deployment
  - Implement resource quotas and limits per tenant
  - Write deployment automation scripts
  - _Requirements: 6.1, 6.4_

- [x] 10. Implement comprehensive testing suite
- [x] 10.1 Create unit tests for all adapters
  - Write unit tests for each adapter interface implementation
  - Create mock implementations for external dependencies
  - Implement tenant isolation verification tests
  - Write performance benchmark tests for critical paths
  - _Requirements: 6.5, 7.5_

- [x] 10.2 Create integration tests for end-to-end workflows
  - Write law firm NDA workflow integration test
  - Create finance P&L workflow integration test
  - Implement cross-tenant isolation verification tests
  - Write failure injection and recovery tests
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [x] 10.3 Create performance and load tests
  - Write load tests for Celery task execution scaling
  - Create vector search performance tests under load
  - Implement policy evaluation latency tests
  - Write usage event throughput tests
  - _Requirements: 8.5_

- [x] 11. Implement observability and monitoring setup
- [x] 11.1 Create Prometheus scrape configuration
  - Write prometheus.yml configuration for all services
  - Implement service discovery for dynamic scaling
  - Create metric retention and aggregation rules
  - Write monitoring setup automation scripts
  - _Requirements: 3.2, 3.3_

- [x] 11.2 Create Grafana dashboard definitions
  - Write JSON dashboard definitions for all key metrics
  - Create tenant-specific dashboard templates
  - Implement dashboard provisioning automation
  - Write dashboard screenshot generation for documentation
  - _Requirements: 3.2, 3.4_

- [x] 12. Create demo workflows and validation
- [x] 12.1 Implement law firm NDA demo workflow
  - Create NDA document processing pipeline with MCAE integration
  - Implement drive_fetch → document_parser → kb_search → compose → slack_post workflow
  - Write OTEL trace validation for complete workflow visibility
  - Create demo data and test scenarios
  - _Requirements: 8.1, 8.2_

- [x] 12.2 Implement finance P&L demo workflow
  - Create ERP data ingestion and analysis pipeline
  - Implement analysis → report_builder → email_sender workflow
  - Write Weaviate vector operations validation
  - Create MemGPT summarization job execution validation
  - _Requirements: 8.3, 8.4_

- [x] 12.3 Create demo validation and evidence collection
  - Write automated demo execution scripts
  - Create trace and metric collection validation
  - Implement usage event verification and CSV export validation
  - Write policy enforcement demonstration with negative tests
  - _Requirements: 8.2, 8.4, 8.5_

- [x] 13. Create documentation and overlap resolution
- [x] 13.1 Write integration documentation
  - Create README for adding new queue tasks
  - Write guide for creating new policies
  - Document metric creation and usage event emission
  - Create troubleshooting guide for common integration issues
  - _Requirements: 6.5_

- [x] 13.2 Create no-overlap technical report
  - Document how LlamaIndex memory/prompting conflicts were resolved
  - Explain workflows-py integration within MCAE stages
  - Detail legacy pricing/permission code consolidation
  - Write architectural decision records for all integration choices
  - _Requirements: 6.5_

- [x] 14. Final integration and system validation
- [x] 14.1 Integrate all components into main application
  - Update main.py to initialize all OSS integrations
  - Create startup health checks for all integrated systems
  - Implement graceful shutdown procedures
  - Write system-wide integration tests
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 14.2 Validate complete system functionality
  - Run comprehensive end-to-end test suite
  - Validate tenant isolation across all integrated systems
  - Verify compliance and security requirements
  - Create system performance baseline measurements
  - _Requirements: 7.5, 8.5_