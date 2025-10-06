# Forge 1 – Roadmap v2 (Wholesale Replacement)

Purpose: A progressive task plan to evolve Forge 1 from the current MVP into a production‑grade platform capable of wholesale replacement of skilled employees across functions (e.g., F500 ops, legal, banking, software). This plan builds on the existing tasks.md, adds durability, safety, policy, vertical depth, and SRE maturity.

Guiding principles
- Production first: durability, isolation, security, observability before breadth.
- Vertical depth over generic breadth; ship per‑function playbooks with measurable SLAs.
- Safety by design: DLP/classification/redaction + policy enforcement + approvals.
- Evaluate continuously: gold datasets, SLOs, perf/quality gates in CI/CD.

## Phase A — Production Foundations (Durability & Isolation)

- [x] A.1 Durable Orchestration Layer
  - Add a persistent job queue (e.g., Celery/RQ/Arq) and outbox pattern for all external calls.
  - Implement saga/compensation for long‑running, multi‑step workflows.
  - Enforce idempotency keys and exponential backoff with jitter; circuit breakers.
  - Acceptance: Workflows survive pod restarts; retries are idempotent; recovery playbooks documented; chaos test passes for failure injection.

- [x] A.2 Multi‑Tenant Isolation & Data Boundaries
  - Per‑tenant DB schemas/roles; per‑tenant Redis namespaces; per‑tenant vector indexes with KMS‑scoped keys.
  - Enforce tenant scoping in query builders and memory retrieval; add tenancy unit tests.
  - Acceptance: Cross‑tenant data access impossible by construction; tenancy tests and audits pass; data export per tenant available.

- [x] A.3 Identity, Access, Secrets
  - SSO/SAML/OIDC; SCIM for user lifecycle; fine‑grained RBAC/ABAC; session management.
  - Secrets via Key Vault with rotation, usage auditing, and break‑glass flows.
  - Acceptance: IdP integrated; roles/attributes enforced in policy engine; secret rotation drills successful.

- [x] A.4 Ops Maturity: Migrations, DR, SLOs, Incident Response
  - Migrations (Alembic) + seed fixtures; backup/restore runbooks (DB/Redis/vector stores).
  - Define SLOs (availability/latency/error rate/backlog) with Prometheus alerts; on‑call runbooks.
  - Acceptance: Disaster recovery drill passes (RPO/RTO targets); SLO dashboards and alert policies live.

## Phase B — Safety, Compliance, and Policy

- [x] B.1 DLP/Classification/Redaction Service
  - Integrate a DLP engine (or build a microservice) for PII/PHI/PCI detection; automatic redaction before storage/transit.
  - Wire into FastAPI middleware and storage gateways (memory manager, file ingestion).
  - Acceptance: Redaction verified in tests; sensitive payloads never persist unredacted; audit logs capture actions.

- [x] B.2 Policy Engine & Approvals
  - Add OPA/Rego (or equivalent) for action/data access/tool execution policies; ABAC + RBAC.
  - Human‑in‑the‑loop approvals for high‑risk actions (finance, legal, security), with queues and SLAs.
  - Acceptance: Policies centrally managed; denial/approval paths observable; coverage tests for critical actions.

- [x] B.3 Audit, E‑Discovery, Data Residency & Retention
  - Structured, immutable audit store; e‑discovery exports; jurisdiction tagging and residency enforcement.
  - Automated retention/deletion schedules per regulation.
  - Acceptance: Residency tests pass; retention jobs verifiably delete; export produces complete chain of evidence.

- [x] B.4 Legal/Risk Reporting Pipelines
  - Executive compliance dashboards; breach notification runbooks; DPIA/PIA tooling hooks.
  - Acceptance: Quarterly compliance reporting pipeline operational; drills completed.

## Phase C — Vertical Playbooks & Connectors

For each target function, deliver a production‑ready “AI employee” with SOPs, tools, KPIs, evals, and guardrails. Suggested order:

- [x] C.1 Customer Experience (CX)
  - Connectors: CRM (Salesforce/HubSpot), CCaaS, ticketing (Zendesk/ServiceNow), n8n/Zapier flows.
  - Playbooks: Triage, resolution, escalation, upsell; SLAs; disputes handling with approvals.
  - Acceptance: 95%+ deflection on scoped intents; first‑response < 5s; CSAT target met; HITL audit trail.

- [x] C.2 Revenue Ops (RevOps/Sales Ops)
  - Connectors: CRM, CPQ, billing; enrichment (Clearbit/ZoomInfo); BI (Looker/Power BI).
  - Playbooks: Pipeline hygiene, forecasting, quoting, renewal motions with approvals.
  - Acceptance: Data hygiene KPIs, forecast delta thresholds, quote approval turnaround.

- [x] C.3 Finance/FP&A
  - Connectors: ERP (SAP/Oracle/Netsuite), banks, payroll; spreadsheet APIs.
  - Playbooks: Close assistance, variance analysis, budget planning; approvals for postings.
  - Acceptance: Close cycle time reduction; variance accuracy; audit‑ready traces.

- [x] C.4 Legal
  - Connectors: CLM (Icertis/Conga/DocuSign), e‑discovery; docket feeds.
  - Playbooks: NDA/MSA/SOW templating, clause extraction, risk scoring, negotiation assistance (HITL gated).
  - Acceptance: Contract cycle time; risk scoring precision; attorney approvals logged.

- [x] C.5 IT Ops / SecOps
  - Connectors: ITSM (ServiceNow/Jira), IdP/MDM, SIEM/SOAR.
  - Playbooks: Provisioning, incident triage, vulnerability patch workflows; emergency approvals.
  - Acceptance: MTTR, false‑positive reduction, change‑risk approvals.

- [x] C.6 Software Engineering
  - Connectors: GitHub/GitLab, CI/CD, issue trackers, artifact registries.
  - Playbooks: Spec → code drafts, PR reviews, test generation, release notes; protected branches/approvals.
  - Acceptance: PR throughput, defect escape rate, build stability; human approvals for merges.

For each vertical task:
- SOPs codified; tools configured; data mapping finalized; gold datasets and eval suites added; SLAs and HITL points documented; dashboards instrumented.

## Phase D — Multi‑Agent Excellence & Tool Reliability

- [x] D.1 Planning & Tool‑Use Reliability
  - Benchmarks for LangChain/CrewAI/AutoGen tasks; tool‑call success/latency tracking; automatic tool retry/backoff.
  - Acceptance: Tool success > 99% on gold suites; planning errors < target; self‑healing retries.

- [x] D.2 Code‑Execution Sandboxes (Dev Agent)
  - Secure sandboxes (Firecracker/K8s jobs) for code agents; resource/time limits; artifact capture.
  - Acceptance: No breakout; reproducible artifacts; audit of executions.

- [x] D.3 Self‑Play & Adversarial Evaluation
  - Synthetic adversarial tasks; model routing under stress; conflict resolution metrics.
  - Acceptance: Degradation bounded; failover effective; escalation protocols trigger.

## Phase E — Platform Marketplace & Customization

- [x] E.1 Agent/Tool Marketplace
  - Versioned agent templates; dependency graphs; tenant‑scoped publishing; approvals to install.
  - Acceptance: Install/upgrade/rollback safe; provenance tracked; SBOMs generated.

- [x] E.2 Tenant Customization & Guardrails
  - Policy and prompt guardrails per tenant; cost caps; execution quotas.
  - Acceptance: Guardrails enforced; quota breaches alerted; budget reports.

## Phase F — Security, Supply Chain & Certifications

- [x] F.1 Supply Chain Security
  - SBOMs (CycloneDX), signature verification (Sigstore), dependency pinning; container scanning.
  - Acceptance: CI blocks on critical vulns; attestations published; provenance verifiable.

- [x] F.2 Penetration Testing & Bug Bounty
  - Automated DAST/SAST, external pen‑tests; triage and remediation SLAs.
  - Acceptance: Findings tracked to closure; severity SLAs met.

- [x] F.3 Compliance Programs
  - SOC 2 Type II / ISO 27001 scaffolding; evidence collection automation; policy docs.
  - Acceptance: Audit readiness checklists pass; evidence pipelines populated.

## Phase G — Launch, Pricing, and Scale

- [x] G.1 Pricing, Billing, Entitlements
  - Usage metering; plan limits; cost visibility; overage policies.
  - Acceptance: Invoices reconcile with usage; entitlements enforced end‑to‑end.

- [x] G.2 Capacity & Cost Optimization
  - Predictive scaling; model cost routing; budget alerts; FinOps dashboards.
  - Acceptance: Unit‑economics targets met; autoscaling events correlate to SLOs.

- [x] G.3 Support, Rollback, and Playbooks
  - Tiered support; incident/rollback playbooks; customer runbooks; onboarding flows.
  - Acceptance: DR drills successful; customer NPS/CSAT targets met in pilots.

---

Milestone sequencing (suggested)
1) A.1–A.4 (4–6 weeks): durability, isolation, identity, ops maturity.
2) B.1–B.3 (3–5 weeks): DLP, policy engine, residency/retention; approvals.
3) C.1–C.2 (3–4 weeks): CX + RevOps verticals to production SLAs.
4) D.1–D.2 (2–3 weeks): tool reliability + dev sandbox; expand C.3–C.6 thereafter.
5) E–G (parallel over 6–12 weeks): marketplace, certifications, pricing & scale.

Acceptance artifacts per task
- Design doc updates; threat models; migration plans.
- Tests (unit/integration/load/chaos); CI gates with thresholds.
- Runbooks (ops/compliance/legal); dashboards; drill reports.

Dependencies on current repo
- Extend: `forge1/backend/forge1/core/*` (orchestration, policy, compliance, memory), `integrations/*` (connectors), and CI/CD.
- Add: workflow queue service, DLP service, policy service, tenancy middleware, approval UIs.

This plan intentionally prioritizes depth, safety, and reliability to credibly achieve wholesale replacement in high‑stakes environments.
