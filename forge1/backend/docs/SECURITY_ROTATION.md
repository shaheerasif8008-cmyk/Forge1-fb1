# Credential Rotation Playbook

1. Generate new secrets using the approved vault tooling.
2. Update the environment by copying `forge1/.env.example` to `.env` and filling in the new values.
3. Apply the secrets to every deployment target (Docker Compose, Kubernetes, CI) using the secure secret manager.
4. Invalidate the previous credentials (database users, Redis auth, Grafana admin) immediately after rollout.
5. Run `pre-commit run gitleaks --all-files` to confirm no secrets remain in the repository.
