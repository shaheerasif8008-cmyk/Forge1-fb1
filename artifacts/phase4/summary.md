# Phase 4 – Policy & DLP

- Implemented FastAPI `PolicyEnforcementMiddleware` backed by the new `OPAAdapter`, enforcing `tool_access`, `doc_access`, and `routing_constraints` Rego policies before protected operations.
- Added decision logging to `artifacts/policy/decision_log.jsonl` and redaction utilities that scrub vector metadata, logs, and export payloads.
- Created focused unit tests covering cross-tenant / wrong-role denials and DLP behaviour across vector writes, logging, and export helpers.

## Evidence
- Policy + DLP tests: see `artifacts/phase4/pytest_policy_dlp.txt`.
- Sample policy decision log: `artifacts/policy/decision_log.jsonl`.
