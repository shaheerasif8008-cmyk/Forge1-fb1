package forge1.policies.doc_access

default allow = false

# Documents may only be accessed by principals from the same tenant
allow {
  input.tenant_id == input.resource.tenant_id
  input.resource.classification == "public"
}

allow {
  input.tenant_id == input.resource.tenant_id
  input.resource.classification == "internal"
}

# Restricted and confidential documents require elevated roles
allow {
  input.tenant_id == input.resource.tenant_id
  input.resource.classification == "restricted"
  input.subject.role == "manager"
}

allow {
  input.tenant_id == input.resource.tenant_id
  input.resource.classification == "confidential"
  input.subject.role == "admin"
}
