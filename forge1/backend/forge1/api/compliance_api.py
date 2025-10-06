"""
Forge 1 Compliance API
Enterprise compliance management and automation endpoints
"""

import time
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from forge1.core.compliance_engine import ComplianceEngine, ComplianceFramework

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/compliance", tags=["compliance"])

# Initialize compliance engine
compliance_engine = ComplianceEngine()

# Pydantic models
class DataSubjectRequest(BaseModel):
    request_type: str  # access, deletion, portability
    user_id: str
    email: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

class ComplianceReportRequest(BaseModel):
    framework: str
    start_date: Optional[float] = None
    end_date: Optional[float] = None
    include_evidence: bool = True

@router.get("/status")
async def get_compliance_status():
    """Get overall compliance status"""
    try:
        status = await compliance_engine.get_compliance_status()
        return status
    except Exception as e:
        logger.error(f"Error getting compliance status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get compliance status")

@router.get("/frameworks")
async def get_supported_frameworks():
    """Get list of supported compliance frameworks"""
    return {
        "frameworks": [
            {
                "id": framework.value,
                "name": framework.value.upper(),
                "description": f"{framework.value.upper()} compliance framework"
            }
            for framework in ComplianceFramework
        ]
    }

@router.get("/frameworks/{framework}/status")
async def get_framework_status(framework: str):
    """Get status for specific compliance framework"""
    try:
        framework_enum = ComplianceFramework(framework.lower())
        report = await compliance_engine.generate_compliance_report(framework_enum)
        return report
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unsupported framework: {framework}")
    except Exception as e:
        logger.error(f"Error getting framework status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get framework status")

@router.post("/data-subject-request")
async def handle_data_subject_request(request: DataSubjectRequest):
    """Handle GDPR data subject requests"""
    try:
        if request.request_type not in ["access", "deletion", "portability"]:
            raise HTTPException(
                status_code=400, 
                detail="Invalid request type. Must be: access, deletion, or portability"
            )
        
        result = await compliance_engine.handle_data_subject_request(
            request.request_type, 
            request.user_id
        )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling data subject request: {e}")
        raise HTTPException(status_code=500, detail="Failed to process data subject request")

@router.get("/audit-log")
async def get_audit_log(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    event_type: Optional[str] = None
):
    """Get compliance audit log"""
    try:
        # Filter audit log
        audit_log = compliance_engine.audit_log
        
        if event_type:
            filtered_log = [
                entry for entry in audit_log 
                if entry.get("event_type") == event_type
            ]
        else:
            filtered_log = audit_log
        
        # Apply pagination
        total = len(filtered_log)
        paginated_log = filtered_log[offset:offset + limit]
        
        return {
            "audit_log": paginated_log,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        }
        
    except Exception as e:
        logger.error(f"Error getting audit log: {e}")
        raise HTTPException(status_code=500, detail="Failed to get audit log")

@router.post("/reports/generate")
async def generate_compliance_report(request: ComplianceReportRequest):
    """Generate compliance report for specific framework"""
    try:
        framework_enum = ComplianceFramework(request.framework.lower())
        report = await compliance_engine.generate_compliance_report(framework_enum)
        
        # Add requested date range if provided
        if request.start_date or request.end_date:
            report["date_range"] = {
                "start": request.start_date,
                "end": request.end_date
            }
        
        return report
        
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unsupported framework: {request.framework}")
    except Exception as e:
        logger.error(f"Error generating compliance report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate compliance report")

@router.get("/privacy-controls")
async def get_privacy_controls():
    """Get current privacy controls configuration"""
    return {
        "controls": compliance_engine.privacy_controls,
        "last_updated": time.time()
    }

@router.post("/privacy-controls")
async def update_privacy_controls(controls: Dict[str, bool]):
    """Update privacy controls configuration"""
    try:
        # Validate controls
        valid_controls = set(compliance_engine.privacy_controls.keys())
        invalid_controls = set(controls.keys()) - valid_controls
        
        if invalid_controls:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid controls: {list(invalid_controls)}"
            )
        
        # Update controls
        compliance_engine.privacy_controls.update(controls)
        
        # Log the change
        await compliance_engine._log_compliance_event(
            "privacy_controls_updated",
            {"updated_controls": controls}
        )
        
        return {
            "status": "updated",
            "controls": compliance_engine.privacy_controls,
            "updated_at": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating privacy controls: {e}")
        raise HTTPException(status_code=500, detail="Failed to update privacy controls")

@router.get("/data-inventory")
async def get_data_inventory():
    """Get data inventory for compliance tracking"""
    return {
        "inventory": compliance_engine.data_inventory,
        "total_records": len(compliance_engine.data_inventory),
        "last_updated": time.time()
    }

@router.post("/validate-content")
async def validate_content(content: Dict[str, str]):
    """Validate content for compliance violations"""
    try:
        if "content" not in content:
            raise HTTPException(status_code=400, detail="Content field is required")
        
        is_compliant = await compliance_engine.validate_content(content["content"])
        
        return {
            "compliant": is_compliant,
            "content_length": len(content["content"]),
            "validated_at": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating content: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate content")

@router.get("/health")
async def compliance_health_check():
    """Health check for compliance engine"""
    try:
        health = await compliance_engine.health_check()
        return health
    except Exception as e:
        logger.error(f"Compliance health check failed: {e}")
        raise HTTPException(status_code=500, detail="Compliance health check failed")