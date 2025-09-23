# forge1/backend/forge1/api/compliance_api.py
"""
FastAPI endpoints for compliance monitoring and reporting
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Request
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import logging
from datetime import datetime

from forge1.core.compliance_engine import ComplianceEngine, AlertPriority
from forge1.core.security_manager import SecurityManager, Permission
from forge1.core.database_config import get_database_manager

logger = logging.getLogger(__name__)

# Pydantic models
class ContentValidationRequest(BaseModel):
    content: str = Field(..., description="Text content to analyze for compliance")
    framework: Optional[str] = Field(None, description="Optional framework hint (GDPR, HIPAA, SOX, PCI_DSS, CCPA)")

class ResolveAlertRequest(BaseModel):
    resolved_by: Optional[str] = Field(None, description="User ID who resolved the alert")
    resolution_note: Optional[str] = Field(None, description="Notes about resolution")

# Singletons
_engine: Optional[ComplianceEngine] = None
_security: Optional[SecurityManager] = None


def get_engine() -> ComplianceEngine:
    global _engine
    if _engine is None:
        _engine = ComplianceEngine()
    return _engine


def get_security() -> SecurityManager:
    global _security
    if _security is None:
        _security = SecurityManager()
    return _security


router = APIRouter(prefix="/api/v1/compliance", tags=["compliance"])


@router.get("/overview", response_model=Dict[str, Any])
async def get_overview(engine: ComplianceEngine = Depends(get_engine)) -> Dict[str, Any]:
    try:
        overview = await engine.health_check()
        # Include framework scores
        overview.update({
            "framework_scores": engine.framework_scores,
        })
        return overview
    except Exception as e:
        logger.error(f"Failed to get compliance overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve compliance overview")


@router.get("/audit-trail", response_model=Dict[str, Any])
async def get_audit_trail(
    request: Request,
    framework: Optional[str] = Query(None, description="Filter by framework"),
    limit: int = Query(100, ge=1, le=1000, description="Max entries to return"),
    engine: ComplianceEngine = Depends(get_engine),
    security: SecurityManager = Depends(get_security)
) -> Dict[str, Any]:
    try:
        # Require permission to view compliance logs
        user = await security.get_enhanced_user_details(request.headers)
        await security.require_permission(user.get("user_principal_id", ""), Permission.COMPLIANCE_VIEW)

        entries = await engine.get_audit_trail(framework=framework, limit=limit)
        # Serialize
        items = [
            {
                "id": e.id,
                "timestamp": e.timestamp.isoformat(),
                "framework": e.framework,
                "event_type": e.event_type,
                "user_id": e.user_id,
                "resource": e.resource,
                "action": e.action,
                "result": e.result,
                "ip_address": e.ip_address,
                "user_agent": e.user_agent,
                "details": e.details,
            }
            for e in entries
        ]
        return {"items": items, "count": len(items)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get audit trail: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve audit trail")


@router.get("/alerts", response_model=Dict[str, Any])
async def get_alerts(
    framework: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    engine: ComplianceEngine = Depends(get_engine)
) -> Dict[str, Any]:
    try:
        prio = AlertPriority(priority) if priority else None
        alerts = await engine.get_active_alerts(framework=framework, priority=prio)
        # Serialize
        items = [
            {
                "id": a.id,
                "framework": a.framework,
                "title": a.title,
                "description": a.description,
                "priority": a.priority.value,
                "created_at": a.created_at.isoformat(),
                "due_date": a.due_date.isoformat() if a.due_date else None,
                "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
                "assigned_to": a.assigned_to,
            }
            for a in alerts
        ]
        return {"items": items, "count": len(items)}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid priority value")
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")


@router.post("/validate-content", response_model=Dict[str, Any])
async def validate_content(
    payload: ContentValidationRequest,
    engine: ComplianceEngine = Depends(get_engine)
) -> Dict[str, Any]:
    try:
        analysis = await engine.analyze_content(payload.content, payload.framework)
        return analysis
    except Exception as e:
        logger.error(f"Failed to validate content: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze content")

@router.get("/audit/export", response_model=Dict[str, Any])
async def export_audit(
    request: Request,
    hours: int = Query(24, ge=1, le=720),
    security: SecurityManager = Depends(get_security)
) -> Dict[str, Any]:
    """Export audit events for e-discovery (B.3)"""
    try:
        user = await security.get_enhanced_user_details(request.headers)
        await security.require_permission(user.get("user_principal_id", ""), Permission.COMPLIANCE_VIEW)

        db = await get_database_manager()
        async with db.postgres.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT event_type, user_id, event_data, ip_address, user_agent, created_at
                FROM forge1_audit.audit_log
                WHERE created_at >= NOW() - ($1 || ' hours')::interval
                ORDER BY created_at DESC
                """,
                hours,
            )
        items = [
            {
                "event_type": r["event_type"],
                "user_id": r["user_id"],
                "event_data": r["event_data"],
                "ip_address": str(r["ip_address"]) if r["ip_address"] else None,
                "user_agent": r["user_agent"],
                "created_at": r["created_at"].isoformat(),
            }
            for r in rows
        ]
        return {"items": items, "count": len(items)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export audit: {e}")
        raise HTTPException(status_code=500, detail="Failed to export audit events")

@router.get("/reports/summary", response_model=Dict[str, Any])
async def compliance_summary(engine: ComplianceEngine = Depends(get_engine)) -> Dict[str, Any]:
    """Legal/Risk reporting summary (B.4)"""
    try:
        health = await engine.health_check()
        return {
            "status": health.get("status"),
            "framework_scores": engine.framework_scores,
            "active_alerts": len(await engine.get_active_alerts()),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to compute compliance summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute compliance summary")


@router.post("/alerts/{alert_id}/resolve", response_model=Dict[str, Any])
async def resolve_alert(
    alert_id: str,
    payload: ResolveAlertRequest,
    engine: ComplianceEngine = Depends(get_engine)
) -> Dict[str, Any]:
    try:
        success = await engine.resolve_alert(alert_id, resolved_by=payload.resolved_by, resolution_note=payload.resolution_note)
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found or already resolved")
        return {"success": True, "alert_id": alert_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve alert")
