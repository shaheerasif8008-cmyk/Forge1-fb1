Forge 1 Terraform (Azure)

This minimal Terraform configuration provisions:
- Resource Group
- Log Analytics Workspace (monitoring)
- AKS Cluster (System-assigned identity, OMS agent)
- Azure Key Vault (for secrets)

Quick start (requires az login and Terraform >= 1.5):
1. cd infra/terraform
2. terraform init
3. terraform apply -auto-approve

Outputs include the AKS cluster name and Key Vault name. Integrate ACR, Redis, and Postgres as managed services in production.

