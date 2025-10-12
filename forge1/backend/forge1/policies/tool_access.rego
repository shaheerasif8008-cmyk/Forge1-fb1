package forge1.policies.tool_access

default allow = false

# Admins and security engineers can execute any tool within their tenant
allow {
  input.subject.role == "admin"
  input.tenant_id == input.resource.tenant_id
}

allow {
  input.subject.role == "security"
  input.tenant_id == input.resource.tenant_id
}

# Standard users may execute tools that are not flagged as high sensitivity
allow {
  input.subject.role == "user"
  input.tenant_id == input.resource.tenant_id
  input.resource.sensitivity != "high"
}
