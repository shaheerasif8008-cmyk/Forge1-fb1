# Requirements Document

## Introduction

This feature integrates nine critical open-source systems (Celery, Redis, Weaviate, MemGPT, OpenTelemetry, Prometheus, Grafana, OpenMeter, and OPA) into Forge1 as first-class production components. The integration maintains Forge1's architectural principles of centralized tenancy, RBAC, routing, memory namespaces, and compliance while avoiding duplication with existing modules. The systems will be integrated behind Forge1 adapters/services to preserve the existing orchestration hierarchy where MCAE remains top-level orchestration and workflows-py stays as sub-orchestration.

## Requirements

### Requirement 1

**User Story:** As a Forge1 system administrator, I want reliable asynchronous execution capabilities through Celery and Redis integration, so that long-running tasks can be processed efficiently without blocking the main application flow.

#### Acceptance Criteria

1. WHEN a long-running MCAE stage is executed THEN the system SHALL dispatch the task through Celery with tenant_id, employee_id, role, request_id, and case_id context
2. WHEN a Celery task fails THEN the system SHALL implement exponential backoff retry policy with circuit breaker and DLQ routing
3. WHEN heavy IO operations (document parsing, embeddings, Drive/Slack integrations) are requested THEN the system SHALL route them through Celery queues
4. WHEN short tasks are executed THEN the system SHALL maintain existing in-process execution for optimal performance
5. IF a simulated outage occurs in Slack/Drive integrations THEN the system SHALL retry according to policy, route to DLQ, and surface status in dashboard

### Requirement 2

**User Story:** As a Forge1 developer, I want long-term memory and retrieval capabilities through Weaviate and MemGPT integration, so that the system can efficiently store, compress, and retrieve contextual information with proper tenant isolation.

#### Acceptance Criteria

1. WHEN memory data is stored THEN the system SHALL use namespace convention {tenant_id}:{employee_id}:{memory_type} with enforced query filters
2. WHEN long contexts exceed threshold THEN MemGPT summarization job SHALL compress contexts via Celery with TTL/archival policies
3. WHEN LlamaIndex tools require persistence THEN they SHALL call through Forge1 MemoryAdapter exclusively
4. WHEN two tenants ingest data simultaneously THEN the system SHALL prevent cross-tenant data reads
5. WHEN summarization reduces token footprint THEN the system SHALL maintain fidelity checks to ensure data integrity

### Requirement 3

**User Story:** As a Forge1 operations team member, I want comprehensive observability through OpenTelemetry, Prometheus, and Grafana integration, so that I can monitor system performance, trace requests, and visualize metrics across all components.

#### Acceptance Criteria

1. WHEN API handlers, MCAE nodes, workflows-py steps, LlamaIndex tools, and ModelRouter calls execute THEN the system SHALL generate OTEL traces
2. WHEN metrics are collected THEN the system SHALL expose /metrics endpoints in API and worker components
3. WHEN Grafana dashboards are accessed THEN they SHALL display latency p95, error rate, queue depth, token usage, and cost with tenant/employee/request_id labels
4. WHEN demo flows run THEN traces SHALL span MCAE → workflows-py → tools with complete visibility
5. IF ad-hoc logging duplicates metrics THEN the system SHALL consolidate on OTEL + Prometheus exclusively

### Requirement 4

**User Story:** As a Forge1 billing administrator, I want usage metering and export capabilities through OpenMeter integration, so that I can track resource consumption and generate accurate billing reports per tenant and employee.

#### Acceptance Criteria

1. WHEN model or tool calls are made THEN the system SHALL emit metering events with {tenant_id, employee_id, model, tokens_in/out, latency, cost_estimate, request_id}
2. WHEN monthly reports are requested THEN the system SHALL provide /reports/usage CSV endpoint with aggregated data
3. WHEN sample month usage is generated THEN CSV totals SHALL match emitted events exactly
4. IF existing pricing_engine stubs exist THEN they SHALL be deprecated in favor of OpenMeter events as source of truth
5. WHEN usage data is exported THEN it SHALL include per-tenant and per-employee breakdowns

### Requirement 5

**User Story:** As a Forge1 security administrator, I want policy enforcement and RBAC through OPA integration, so that access controls are consistently applied across all system components with auditable decisions.

#### Acceptance Criteria

1. WHEN tool execution, document access, or cross-role resource requests occur THEN OPA SHALL enforce policy checks
2. WHEN policy violations are detected THEN the system SHALL deny access with audited 403 responses
3. WHEN policy input is evaluated THEN it SHALL include {tenant_id, employee_id, role, tool_name, doc_tags, pii_flags, request_id}
4. WHEN disallowed tool/document access is attempted THEN the system SHALL return 403 with auditable reason and redacted logs
5. IF existing PermissionEnforcer logic exists THEN it SHALL be adapted to call OPA while removing duplicate hard-coded policy branches

### Requirement 6

**User Story:** As a Forge1 architect, I want proper integration architecture that preserves existing system boundaries, so that new components don't violate the established orchestration hierarchy or create conflicting ownership.

#### Acceptance Criteria

1. WHEN new components are integrated THEN Forge1 SHALL remain the source of truth for tenancy, RBAC, routing, memory namespaces, and compliance
2. WHEN orchestration occurs THEN MCAE SHALL remain top-level orchestration and workflows-py SHALL stay as sub-orchestration inside stages
3. WHEN LlamaIndex is used THEN it SHALL remain a tool provider with all model calls going through ModelRouter and memory I/O through Forge1 Memory adapters
4. WHEN new components require global state THEN they SHALL be integrated behind Forge1 adapters/services exclusively
5. IF conflicts exist with existing modules THEN they SHALL be resolved with documented "no-overlap" report

### Requirement 7

**User Story:** As a Forge1 compliance officer, I want DLP and security measures maintained across all integrated systems, so that sensitive data is protected and regulatory requirements are met.

#### Acceptance Criteria

1. WHEN data is persisted or logged THEN DLP redaction SHALL be applied before storage
2. WHEN traces are generated THEN they SHALL NOT include sensitive payloads
3. WHEN secrets and configuration are managed THEN existing SecretManager SHALL be used with no plaintext secrets in code/logs
4. WHEN dependencies are added THEN they SHALL be version pinned with requirements lockfile and compatibility documentation
5. WHEN two-tenant test suite runs THEN it SHALL demonstrate no cross-reads with RBAC denials audited and DLP redaction verified

### Requirement 8

**User Story:** As a Forge1 end user, I want to see the integrated systems working in realistic scenarios, so that I can validate the complete functionality through law firm NDA processing and finance report generation workflows.

#### Acceptance Criteria

1. WHEN law firm NDA flow executes THEN it SHALL complete Intake → MCAE handoff → Lawyer workflows (drive_fetch → document_parser → kb_search → compose → slack_post)
2. WHEN NDA flow runs THEN OTEL traces SHALL show all steps with {tenant_id, employee_id, case_id} and Prometheus SHALL show queue depth and p95 latency
3. WHEN finance P&L flow executes THEN it SHALL complete ERP ingest → analysis → report_builder → email_sender
4. WHEN P&L flow runs THEN Weaviate vector operations SHALL be confirmed and MemGPT summarization job SHALL execute
5. WHEN both demo flows complete THEN OpenMeter SHALL show usage events, /reports/usage SHALL return monthly CSV, and OPA SHALL demonstrate policy enforcement with negative tests