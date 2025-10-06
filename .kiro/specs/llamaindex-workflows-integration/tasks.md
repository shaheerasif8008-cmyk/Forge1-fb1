# Implementation Plan

- [x] 1. Set up project dependencies and library integration
  - Pin LlamaIndex to v0.9.x and workflows-py to v0.1.x in pyproject.toml
  - Add required dependencies for document processing, vector operations, and workflow execution
  - Create compatibility verification script to ensure library versions work together
  - _Requirements: 1.1, 1.2, 6.4_

- [ ] 2. Create core adapter infrastructure
- [x] 2.1 Implement LlamaIndex adapter with Forge1 integration
  - Create `forge1/integrations/llamaindex_adapter.py` with base adapter class
  - Implement context propagation for tenant_id, employee_id, role, request_id, case_id
  - Add RBAC enforcement wrapper for all tool operations
  - Integrate with Forge1's Key/Secret broker for credential management
  - _Requirements: 1.1, 1.3, 2.1, 2.2, 3.5, 6.1_

- [x] 2.2 Implement workflows-py adapter with MCAE integration
  - Create `forge1/integrations/workflows_adapter.py` with workflow orchestration
  - Implement standardized handoff packet structure between workflow steps
  - Add OpenTelemetry span creation with proper tenant/employee tagging
  - Implement idempotent step execution with retry/backoff mechanisms
  - _Requirements: 1.2, 2.1, 4.2, 4.3, 4.4, 5.4_

- [x] 2.3 Create Forge1 model client shim for LlamaIndex
  - Implement `forge1/integrations/llamaindex_model_shim.py` with LlamaIndex LLM interface
  - Route all LLM calls through Forge1's Model Router with no direct model access
  - Preserve tenant/employee context in all model routing decisions
  - Track usage metrics for billing and cost estimation
  - _Requirements: 1.4, 1.5, 5.2, 5.3_

- [ ] 3. Implement production-ready tool belt via LlamaIndex
- [x] 3.1 Create document parser tool with OCR fallback
  - Implement PDF/DOCX parsing using LlamaIndex document loaders
  - Add OCR fallback capability for image-based documents
  - Enforce `document:read` RBAC permission with audited 403 responses
  - Apply PII/DLP redaction hooks before logging or persistence
  - _Requirements: 3.1, 3.5, 6.1, 6.3_

- [x] 3.2 Implement knowledge base search tool with vector retrieval
  - Create vector/hybrid search using Forge1's vector store and memory manager
  - Enforce tenant isolation in all search operations with namespace filtering
  - Require `knowledge_base:search` RBAC permission with audit logging
  - Implement result caching with 30-minute TTL for performance
  - _Requirements: 1.5, 2.3, 3.2, 3.5_

- [x] 3.3 Build Google Drive fetch tool with read-only access
  - Implement Google Drive API integration using tenant-scoped credentials
  - Consume credentials via Forge1's Key/Secret broker with no plaintext storage
  - Enforce `drive:read` RBAC permission with comprehensive audit trail
  - Add connection pooling and rate limiting for external API calls
  - _Requirements: 2.5, 3.3, 3.5, 6.2_

- [x] 3.4 Create Slack post tool for tenant-scoped channels
  - Implement Slack API integration with tenant-specific channel posting
  - Use tenant-scoped Slack tokens from Key/Secret broker
  - Require `slack:post` RBAC permission with message audit logging
  - Add retry logic with exponential backoff for API failures
  - _Requirements: 2.5, 3.4, 3.5, 6.2_

- [ ] 4. Build NDA workflow using workflows-py sub-orchestration
- [x] 4.1 Design and implement NDA workflow graph structure
  - Create workflow definition with steps: drive_fetch → document_parser → kb_search → compose_draft → slack_post
  - Implement standardized handoff packet with tenant_id, employee_id, case_id, summary, citations, attachment_refs
  - Add workflow registration with MCAE adapter for lawyer stage integration
  - Ensure each step is independently testable and idempotent
  - _Requirements: 4.1, 4.2, 7.2_

- [x] 4.2 Implement workflow step execution with error handling
  - Add comprehensive error handling with circuit breaker pattern for external services
  - Implement dead-letter queue for failed workflows with visible status reporting
  - Create workflow status tracking and progress monitoring
  - Add timeout handling (30s per tool, 5min per workflow) with graceful degradation
  - _Requirements: 4.4, 6.2, 6.5_

- [ ] 5. Integrate observability and usage tracking
- [x] 5.1 Implement OpenTelemetry integration for comprehensive tracing
  - Add span creation for every tool call and workflow step execution
  - Tag all spans with tenant_id, employee_id, request_id for complete traceability
  - Bridge LlamaIndex and workflows-py tracing into Forge1's OTEL pipeline
  - Create custom metrics for tool usage patterns and workflow success rates
  - _Requirements: 1.4, 5.1, 5.4_

- [x] 5.2 Build usage tracking and metrics collection system
  - Emit usage events with model, tokens_in/out, latency_ms, cost_estimate for all LLM calls
  - Track per-step and per-tool execution metrics in usage pipeline
  - Create CSV export functionality with timestamp, tenant_id, employee_id, model, tokens, latency, cost, tool_calls
  - Implement real-time usage monitoring dashboard integration
  - _Requirements: 5.2, 5.5, 9.3_

- [ ] 6. Implement comprehensive security and compliance features
- [x] 6.1 Add PII/DLP protection with configurable redaction
  - Integrate PII/DLP redaction hooks for all tool outputs before logging
  - Implement configurable redaction rules per tenant with policy enforcement
  - Add sensitive data pattern detection and masking capabilities
  - Ensure no PII leakage in error messages or debug logs
  - _Requirements: 6.1, 6.3_

- [x] 6.2 Implement production resilience patterns
  - Add timeouts, retries with exponential backoff, and circuit breakers for all external calls
  - Implement connection pooling with limits (max 10 concurrent per service)
  - Add rate limiting to respect external service constraints
  - Create health check endpoints for all adapters and tools
  - _Requirements: 6.2, 6.5_

- [ ] 7. Create comprehensive test suite for validation
- [x] 7.1 Build isolation and RBAC test suite
  - Create isolation tests with two tenants and two employees each to verify no cross-contamination
  - Implement RBAC tests that attempt disallowed operations and verify 403 responses with audit logs
  - Add test data cleanup automation to prevent test pollution
  - Create mock services for external dependencies (Drive, Slack) with configurable responses
  - _Requirements: 8.1, 8.2_

- [x] 7.2 Implement resilience and routing test suite
  - Create resilience tests that inject Slack/Drive failures and verify retry/dead-letter behavior
  - Build routing tests that change employee model preferences and verify router selection in logs
  - Add performance tests to measure latency and throughput under load
  - Implement concurrent multi-tenant workflow testing
  - _Requirements: 8.3, 8.4_

- [ ] 8. Build end-to-end demonstration system
- [x] 8.1 Create demo tenant and employee setup
  - Set up "Hartwell & Associates" tenant with four employees (Intake, Lawyer, Research, HR)
  - Configure employee roles, permissions, and model preferences for realistic demo
  - Create sample documents and knowledge base content for NDA workflow testing
  - Set up tenant-scoped Slack channels and Google Drive folders
  - _Requirements: 7.1, 7.2_

- [x] 8.2 Implement complete NDA workflow demonstration
  - Execute full workflow: Intake → MCAE handoff → Lawyer stage (workflows-py subgraph) → Research check → final Slack message
  - Capture and display logs/spans showing correct tenant_id, employee_id, case_id at each step
  - Generate Grafana dashboard or log dump showing step-by-step execution spans
  - Create usage CSV export demonstrating per-call metrics collection
  - _Requirements: 7.2, 7.3, 9.4_

- [ ] 9. Create documentation and maintainability artifacts
- [x] 9.1 Write comprehensive integration documentation
  - Create "how it works" guide explaining adapter locations, context flow, and architecture
  - Document process for adding new tools and workflow steps
  - Write troubleshooting guide for common integration issues
  - Create API reference documentation for all adapters and tools
  - _Requirements: 9.1, 9.2_

- [x] 9.2 Implement configuration and deployment support
  - Create environment-specific configuration templates
  - Add health check endpoints and monitoring integration
  - Implement graceful shutdown and cleanup procedures
  - Create deployment verification scripts and smoke tests
  - _Requirements: 9.2, 9.5_

- [ ] 10. Final integration testing and validation
- [x] 10.1 Execute comprehensive integration validation
  - Run complete test suite including isolation, RBAC, resilience, and routing tests
  - Perform end-to-end NDA workflow demonstration with full observability
  - Validate usage tracking and CSV export functionality
  - Verify no architectural boundary violations (no bypassing of Forge1 Router, Memory, or RBAC)
  - _Requirements: 8.5, 7.4, 9.5_

- [x] 10.2 Performance optimization and production readiness
  - Optimize caching strategies for model responses, document parsing, and knowledge base queries
  - Tune connection pools, timeouts, and resource limits for production load
  - Validate error handling and recovery mechanisms under failure conditions
  - Complete security audit of credential handling and tenant isolation
  - _Requirements: 6.4, 6.5, 8.5_