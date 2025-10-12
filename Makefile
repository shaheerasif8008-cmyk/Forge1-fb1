DOCKER_COMPOSE ?= docker compose
ARTIFACT_DIR ?= artifacts/boot

.PHONY: up down logs migrate health seed-mock

up:
$(DOCKER_COMPOSE) up -d --build

down:
$(DOCKER_COMPOSE) down -v

logs:
$(DOCKER_COMPOSE) logs -f api

migrate:
$(DOCKER_COMPOSE) exec api alembic upgrade head

health:
mkdir -p $(ARTIFACT_DIR)
curl -fsS localhost:8000/health | tee $(ARTIFACT_DIR)/health.json
curl -fsS localhost:8000/metrics | head -n 50 | tee $(ARTIFACT_DIR)/metrics.txt

seed-mock:
@echo "No mock seed implemented yet. Add a script under forge1/scripts and update this target."
