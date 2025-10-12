package forge1.policies.routing_constraints

default allow = false

allow {
  input.subject.role == "admin"
}

allow {
  input.subject.role == "manager"
  input.environment.target_model == "gpt-3.5"
}

allow {
  input.subject.role == "user"
  input.environment.target_model == "gpt-3.5"
}
