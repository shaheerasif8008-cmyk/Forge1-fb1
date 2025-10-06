# Requirements Document

## Introduction

This feature integrates both LlamaIndex and workflows-py into the Forge1 platform as first-class citizens, maintaining the existing architecture where MCAE remains the top-level orchestrator, workflows-py provides sub-orchestration within MCAE stages, and LlamaIndex serves as a tool provider. The integration must be tenant/employee aware, production-safe, and fully tested with comprehensive observability.

## Requirements

### Requirement 1: Architecture Compliance and Integration Boundaries

**User Story:** As a platform architect, I want LlamaIndex and workflows-py integrated while preserving Forge1's architectural boundaries, so that we maintain clear separation of concerns and avoid overlapping responsibilities.

#### Acceptance Criteria

1. WHEN integrating LlamaIndex THEN the system SHALL treat it as a tool provider only and SHALL NOT allow it to choose models or own memory/routing policy
2. WHEN integrating workflows-py THEN the system SHALL use it only as sub-orchestration inside single MCAE stages and SHALL NOT duplicate MCAE's top-level orchestration responsibilities
3. WHEN any integration component needs tenant context THEN Forge1 SHALL remain the authoritative source for tenancy, isolation, memory namespaces, model routing, RBAC, compliance, and billing events
4. WHEN LlamaIndex or workflows-py components make LLM calls THEN all calls SHALL be routed through Forge1's model router with no direct model client access
5. WHEN memory operations occur THEN all reads/writes SHALL go through Forge1's memory abstraction using per-employee namespaces with no direct global state access

### Requirement 2: Context Propagation and Tenant Isolation

**User Story:** As a security engineer, I want all LlamaIndex pipelines and workflows-py steps to receive proper tenant context, so that data isolation is maintained across all operations.

#### Acceptance Criteria

1. WHEN any LlamaIndex pipeline executes THEN it SHALL receive context containing {tenant_id, employee_id, role, request_id, case_id}
2. WHEN any workflows-py step executes THEN it SHALL receive context containing {tenant_id, employee_id, role, request_id, case_id}
3. WHEN context is propagated THEN adapters SHALL ensure no cross-tenant or cross-employee data access occurs
4. WHEN memory operations occur THEN they SHALL use per-employee namespaces enforced by Forge1's memory abstraction
5. WHEN secrets or credentials are needed THEN they SHALL be consumed via Forge1's Key/Secret broker with no plaintext in code or logs

### Requirement 3: Production-Ready Tool Belt via LlamaIndex

**User Story:** As a legal professional, I want access to document processing, knowledge base search, and communication tools through LlamaIndex integration, so that I can efficiently handle legal workflows with proper security controls.

#### Acceptance Criteria

1. WHEN document processing is needed THEN the system SHALL provide a document_parser tool supporting PDF/DOCX with OCR fallback
2. WHEN knowledge base search is required THEN the system SHALL provide a kb_search tool with vector/hybrid retrieval via Forge1's vector store
3. WHEN Google Drive access is needed THEN the system SHALL provide a drive_fetch tool with read-only access
4. WHEN Slack communication is required THEN the system SHALL provide a slack_post tool for tenant-scoped channels
5. WHEN any tool is accessed THEN RBAC SHALL be enforced with audited 403 responses for insufficient permissions
6. WHEN tools are called from MCAE or workflows-py steps THEN they SHALL be accessible through standardized interfaces

### Requirement 4: Sub-Orchestration with workflows-py

**User Story:** As a workflow designer, I want to use workflows-py for explicit step graphs within MCAE stages, so that complex multi-step processes can be clearly defined and executed with proper error handling.

#### Acceptance Criteria

1. WHEN the Lawyer/NDA pipeline stage executes THEN it SHALL use a workflows-py graph with steps: drive_fetch → document_parser → kb_search → compose_draft → slack_post
2. WHEN steps execute in the workflow THEN they SHALL preserve a standardized handoff packet: {tenant_id, employee_id, case_id, summary, citations, attachment_refs}
3. WHEN workflow steps execute THEN they SHALL be idempotent and support retries with exponential backoff
4. WHEN workflow steps execute THEN they SHALL emit OpenTelemetry spans with proper tenant/employee tagging
5. WHEN workflow failures occur THEN the system SHALL handle them gracefully with visible status reporting

### Requirement 5: Observability and Usage Tracking

**User Story:** As a platform operator, I want comprehensive observability and usage tracking for all LlamaIndex and workflows-py operations, so that I can monitor performance, costs, and compliance.

#### Acceptance Criteria

1. WHEN any operation executes THEN it SHALL emit OpenTelemetry spans tagged with {tenant_id, employee_id, request_id}
2. WHEN LLM or embedding calls are made THEN usage events SHALL be emitted with {model, tokens_in/out, latency_ms, cost_estimate}
3. WHEN tools are used THEN per-step and per-tool metrics SHALL be recorded in the usage pipeline
4. WHEN tracing occurs THEN any extra tracing from LlamaIndex/workflows-py SHALL be bridged into Forge1's OTEL pipeline
5. WHEN usage data is needed THEN it SHALL be exportable in CSV format with all relevant metrics

### Requirement 6: Security and Compliance Integration

**User Story:** As a compliance officer, I want all LlamaIndex and workflows-py operations to respect our security policies and data protection requirements, so that sensitive information is properly handled.

#### Acceptance Criteria

1. WHEN tool outputs are logged or persisted THEN PII/DLP redaction hooks SHALL be applied to prevent sensitive data leaks
2. WHEN RBAC violations occur THEN they SHALL result in audited 403 responses with clear reasons
3. WHEN external calls are made THEN they SHALL include timeouts, retries with exponential backoff, and circuit breakers
4. WHEN library versions are used THEN they SHALL be pinned and documented with compatibility notes
5. WHEN failures occur THEN they SHALL be handled with dead-letter queues and visible status reporting

### Requirement 7: End-to-End Workflow Demonstration

**User Story:** As a stakeholder, I want to see a complete demonstration of the integrated system handling a real legal workflow, so that I can verify the integration works as intended.

#### Acceptance Criteria

1. WHEN the demo runs THEN it SHALL use tenant "Hartwell & Associates" with four employees (Intake, Lawyer, Research, HR)
2. WHEN the NDA workflow executes THEN it SHALL flow: Intake → MCAE handoff → Lawyer stage (workflows-py subgraph) → Research check → final Slack message
3. WHEN the workflow executes THEN logs/spans SHALL show correct {tenant_id, employee_id, case_id} at each step
4. WHEN the demo completes THEN it SHALL produce artifacts showing successful execution with proper context propagation
5. WHEN observability is checked THEN Grafana or logs SHALL show step-by-step spans for the complete NDA run

### Requirement 8: Comprehensive Testing and Validation

**User Story:** As a quality assurance engineer, I want comprehensive tests that validate isolation, RBAC, resilience, and routing behavior, so that the integration is production-ready.

#### Acceptance Criteria

1. WHEN isolation tests run THEN they SHALL verify no cross-tenant or cross-employee reads/writes with two tenants and two employees each
2. WHEN RBAC tests run THEN they SHALL verify 403 responses with auditable reasons for disallowed tool and document class access
3. WHEN resilience tests run THEN they SHALL inject Slack/Drive failures and verify retries then dead-letter behavior with visible status
4. WHEN routing tests run THEN they SHALL change employee model preferences and verify router selection in logs
5. WHEN any test fails THEN it SHALL provide clear diagnostic information for debugging

### Requirement 9: Documentation and Maintainability

**User Story:** As a developer, I want clear documentation and modular code structure for the integration, so that I can understand, maintain, and extend the system.

#### Acceptance Criteria

1. WHEN documentation is provided THEN it SHALL include a "how it works" note explaining adapter locations, context flow, and new tool/step addition process
2. WHEN code is structured THEN integrations SHALL be modular to allow swapping or version-bumping without system-wide ripple effects
3. WHEN usage data is needed THEN sample CSV exports SHALL be provided showing per-call metrics
4. WHEN artifacts are delivered THEN they SHALL include screenshots or dumps of observability data
5. WHEN maintenance is needed THEN the integration SHALL support independent updates of LlamaIndex and workflows-py versions