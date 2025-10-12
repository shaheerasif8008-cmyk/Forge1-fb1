"""Forge1 FastAPI application entrypoint focused on Phase 1 foundations."""

from __future__ import annotations

import os
import secrets
from datetime import timedelta
from typing import Any, Dict

from fastapi import Depends, FastAPI, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest

from forge1.billing import usage_meter
from forge1.auth.jwt import UserRole, issue_token, require_role
from forge1.core.database_config import DatabaseManager
from forge1.core.dlp import redact_for_export
from forge1.core.logging_config import init_logger
from forge1.core.security import SecretManager
from forge1.middleware.policy_enforcer import PolicyEnforcementMiddleware
from forge1.middleware.tenant_context import TenantContextMiddleware
from forge1.observability import setup_observability, shutdown_observability

logger = init_logger("forge1.main")

app = FastAPI(title="Forge1 Backend", version="0.1.0")

observability_state = setup_observability(app=app)

secret_manager = SecretManager()
jwt_secret = secret_manager.get_secret("FORGE1_JWT_SECRET")
if not jwt_secret:
    jwt_secret = secrets.token_urlsafe(32)
    os.environ.setdefault("FORGE1_JWT_SECRET", jwt_secret)
    logger.warning("FORGE1_JWT_SECRET not provided; generated ephemeral secret for runtime use")

db_manager = DatabaseManager()

app.add_middleware(PolicyEnforcementMiddleware)

app.add_middleware(
    TenantContextMiddleware,
    secret_manager=secret_manager,
    allow_anonymous_paths={"/", "/health", "/metrics", "/auth/token"},
)

HEALTH_GAUGE = Gauge("forge1_health_status", "Overall Forge1 health (1=ok,0=degraded)")
POSTGRES_GAUGE = Gauge("forge1_postgres_health", "Postgres availability (1=ok,0=down)")
REDIS_GAUGE = Gauge("forge1_redis_health", "Redis availability (1=ok,0=down)")


@app.on_event("startup")
async def startup() -> None:  # pragma: no cover - exercised in integration tests
    await db_manager.start()


@app.on_event("shutdown")
async def shutdown() -> None:  # pragma: no cover
    await db_manager.stop()
    shutdown_observability(observability_state)


class TokenRequest(BaseModel):
    user_id: str
    tenant_id: str
    role: UserRole = UserRole.USER
    expires_in_seconds: int = 1800


class TokenResponse(BaseModel):
    token: str


class ExportRequest(BaseModel):
    payload: Dict[str, Any]


class ExportResponse(BaseModel):
    redacted: Dict[str, Any]
    violations: int


@app.post("/auth/token", response_model=TokenResponse)
async def create_token(request: TokenRequest) -> TokenResponse:
    token = issue_token(
        secret=jwt_secret,
        user_id=request.user_id,
        tenant_id=request.tenant_id,
        role=request.role,
        expires_in=timedelta(seconds=request.expires_in_seconds),
    )
    return TokenResponse(token=token)


@app.get("/")
async def root() -> Dict[str, Any]:
    return {"service": "forge1", "status": "ok"}


@app.get("/health")
async def health() -> JSONResponse:
    status = await db_manager.health()
    HEALTH_GAUGE.set(1 if status["overall"] else 0)
    POSTGRES_GAUGE.set(1 if status["postgres"] else 0)
    REDIS_GAUGE.set(1 if status["redis"] else 0)
    payload = {
        "status": "ok" if status["overall"] else "degraded",
        "services": {
            "postgres": "ok" if status["postgres"] else "down",
            "redis": "ok" if status["redis"] else "down",
        },
    }
    return JSONResponse(payload)


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/whoami")
async def whoami(payload=Depends(require_role(UserRole.USER, UserRole.ADMIN, UserRole.SERVICE))) -> Dict[str, Any]:
    return {
        "tenant_id": payload.tenant_id,
        "user_id": payload.user_id,
        "role": payload.role.value,
    }


@app.get("/reports/usage")
async def usage_report(
    month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
    format: str = Query("json", pattern=r"^(json|csv)$"),
    payload=Depends(require_role(UserRole.ADMIN, UserRole.SERVICE)),
) -> Response:
    if format == "csv":
        csv_body = usage_meter.export_month_csv(month)
        return Response(content=csv_body, media_type="text/csv")
    summary = usage_meter.month_summary(month)
    return JSONResponse(summary)


@app.post("/exports/snapshot", response_model=ExportResponse)
async def export_snapshot(
    request: ExportRequest,
    payload=Depends(require_role(UserRole.USER, UserRole.ADMIN, UserRole.SERVICE)),
) -> ExportResponse:
    safe_payload, violations = redact_for_export(request.payload)
    return ExportResponse(redacted=safe_payload, violations=len(violations))
