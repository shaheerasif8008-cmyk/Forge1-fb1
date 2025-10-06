"""
Usage Reporting and Export Service

Provides comprehensive usage reporting, CSV export functionality,
and automated report generation for billing and analytics.
"""

import asyncio
import csv
import io
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from fastapi import HTTPException
from forge1.integrations.metering.openmeter_client import OpenMeterAdapter, TimePeriod, UsageSummary
from forge1.integrations.base_adapter import ExecutionContext, TenantContext
from forge1.core.tenancy import get_current_tenant

logger = logging.getLogger(__name__)

@dataclass
class UsageReport:
    """Usage report structure"""
    tenant_id: str
    employee_id: Optional[str]
    period_start: datetime
    period_end: datetime
    total_cost: float
    total_events: int
    resource_breakdown: Dict[str, Dict[str, Any]]
    generated_at: datetime

@dataclass
class BillingExportConfig:
    """Configuration for billing exports"""
    include_detailed_events: bool = False
    include_cost_breakdown: bool = True
    include_metadata: bool = False
    currency: str = "USD"
    decimal_places: int = 6

class UsageReportExporter:
    """Service for usage reporting and export functionality"""
    
    def __init__(self, openmeter_adapter: Optional[OpenMeterAdapter] = None):
        self.openmeter = openmeter_adapter or OpenMeterAdapter()
        self.export_config = BillingExportConfig()
        
        # Report generation statistics
        self._reports_generated = 0
        self._exports_created = 0
        self._total_export_size_bytes = 0
    
    async def generate_monthly_report(self, tenant_id: str, year: int, month: int, 
                                    employee_id: Optional[str] = None) -> UsageReport:
        """Generate comprehensive monthly usage report"""
        
        try:
            # Calculate period boundaries
            period_start = datetime(year, month, 1, tzinfo=timezone.utc)
            
            if month == 12:
                period_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                period_end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
            
            # Create time period
            time_period = TimePeriod(start=period_start, end=period_end)
            
            # Get usage summary from OpenMeter
            usage_summary = await self.openmeter.get_usage_summary(
                tenant_id=tenant_id,
                period=time_period,
                employee_id=employee_id
            )
            
            if not usage_summary:
                # Create empty report if no data
                usage_summary = UsageSummary(
                    tenant_id=tenant_id,
                    employee_id=employee_id,
                    period_start=period_start,
                    period_end=period_end,
                    total_events=0,
                    total_cost=0.0,
                    resource_breakdown={}
                )
            
            # Create usage report
            report = UsageReport(
                tenant_id=tenant_id,
                employee_id=employee_id,
                period_start=period_start,
                period_end=period_end,
                total_cost=usage_summary.total_cost,
                total_events=usage_summary.total_events,
                resource_breakdown=usage_summary.resource_breakdown,
                generated_at=datetime.now(timezone.utc)
            )
            
            self._reports_generated += 1
            
            logger.info(f"Generated monthly report for tenant {tenant_id}, "
                       f"period {year}-{month:02d}, cost: ${usage_summary.total_cost:.2f}")
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate monthly report for tenant {tenant_id}: {e}")
            raise
    
    async def export_usage_csv(self, tenant_id: str, year: int, month: int, 
                             employee_id: Optional[str] = None) -> str:
        """Export usage data as CSV for a specific month"""
        
        try:
            # Generate report
            report = await self.generate_monthly_report(tenant_id, year, month, employee_id)
            
            # Create CSV content
            csv_content = await self._generate_csv_content(report)
            
            self._exports_created += 1
            self._total_export_size_bytes += len(csv_content.encode('utf-8'))
            
            logger.info(f"Exported CSV for tenant {tenant_id}, period {year}-{month:02d}, "
                       f"size: {len(csv_content)} characters")
            
            return csv_content
            
        except Exception as e:
            logger.error(f"Failed to export CSV for tenant {tenant_id}: {e}")
            raise
    
    async def export_usage_json(self, tenant_id: str, year: int, month: int, 
                              employee_id: Optional[str] = None) -> str:
        """Export usage data as JSON for a specific month"""
        
        try:
            # Generate report
            report = await self.generate_monthly_report(tenant_id, year, month, employee_id)
            
            # Convert to JSON
            report_dict = asdict(report)
            
            # Convert datetime objects to ISO format
            report_dict["period_start"] = report.period_start.isoformat()
            report_dict["period_end"] = report.period_end.isoformat()
            report_dict["generated_at"] = report.generated_at.isoformat()
            
            json_content = json.dumps(report_dict, indent=2, default=str)
            
            self._exports_created += 1
            self._total_export_size_bytes += len(json_content.encode('utf-8'))
            
            logger.info(f"Exported JSON for tenant {tenant_id}, period {year}-{month:02d}")
            
            return json_content
            
        except Exception as e:
            logger.error(f"Failed to export JSON for tenant {tenant_id}: {e}")
            raise
    
    async def get_usage_summary_for_period(self, tenant_id: str, start_date: datetime, 
                                         end_date: datetime, employee_id: Optional[str] = None) -> UsageSummary:
        """Get usage summary for a custom period"""
        
        try:
            time_period = TimePeriod(start=start_date, end=end_date)
            
            usage_summary = await self.openmeter.get_usage_summary(
                tenant_id=tenant_id,
                period=time_period,
                employee_id=employee_id
            )
            
            if not usage_summary:
                usage_summary = UsageSummary(
                    tenant_id=tenant_id,
                    employee_id=employee_id,
                    period_start=start_date,
                    period_end=end_date,
                    total_events=0,
                    total_cost=0.0,
                    resource_breakdown={}
                )
            
            return usage_summary
            
        except Exception as e:
            logger.error(f"Failed to get usage summary for tenant {tenant_id}: {e}")
            raise
    
    async def generate_tenant_billing_report(self, tenant_id: str, year: int, month: int) -> Dict[str, Any]:
        """Generate comprehensive billing report for a tenant"""
        
        try:
            # Get overall tenant report
            tenant_report = await self.generate_monthly_report(tenant_id, year, month)
            
            # Get per-employee breakdown
            employee_reports = []
            
            # This would typically query for all employees in the tenant
            # For now, we'll simulate with the main report
            if tenant_report.employee_id:
                employee_reports.append(tenant_report)
            
            # Calculate billing summary
            billing_summary = {
                "tenant_id": tenant_id,
                "billing_period": f"{year}-{month:02d}",
                "total_cost": tenant_report.total_cost,
                "total_events": tenant_report.total_events,
                "currency": self.export_config.currency,
                "employee_count": len(employee_reports) if employee_reports else 1,
                "cost_breakdown": tenant_report.resource_breakdown,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Add employee details if available
            if employee_reports:
                billing_summary["employee_breakdown"] = [
                    {
                        "employee_id": report.employee_id,
                        "cost": report.total_cost,
                        "events": report.total_events
                    }
                    for report in employee_reports
                ]
            
            return billing_summary
            
        except Exception as e:
            logger.error(f"Failed to generate billing report for tenant {tenant_id}: {e}")
            raise
    
    async def _generate_csv_content(self, report: UsageReport) -> str:
        """Generate CSV content from usage report"""
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header information
        writer.writerow(["Forge1 Usage Report"])
        writer.writerow(["Tenant ID", report.tenant_id])
        if report.employee_id:
            writer.writerow(["Employee ID", report.employee_id])
        writer.writerow(["Period Start", report.period_start.isoformat()])
        writer.writerow(["Period End", report.period_end.isoformat()])
        writer.writerow(["Generated At", report.generated_at.isoformat()])
        writer.writerow([])  # Empty row
        
        # Write summary
        writer.writerow(["Summary"])
        writer.writerow(["Total Cost", f"${report.total_cost:.{self.export_config.decimal_places}f}"])
        writer.writerow(["Total Events", report.total_events])
        writer.writerow(["Currency", self.export_config.currency])
        writer.writerow([])  # Empty row
        
        # Write resource breakdown
        if report.resource_breakdown:
            writer.writerow(["Resource Breakdown"])
            writer.writerow(["Resource Type", "Events", "Cost", "Unit", "Quantity"])
            
            for resource_type, details in report.resource_breakdown.items():
                writer.writerow([
                    resource_type,
                    details.get("events", 0),
                    f"${details.get('cost', 0.0):.{self.export_config.decimal_places}f}",
                    details.get("unit", ""),
                    details.get("quantity", 0)
                ])
            
            writer.writerow([])  # Empty row
        
        # Write cost breakdown by category
        if self.export_config.include_cost_breakdown and report.resource_breakdown:
            writer.writerow(["Cost Breakdown by Category"])
            writer.writerow(["Category", "Percentage", "Amount"])
            
            total_cost = report.total_cost
            if total_cost > 0:
                for resource_type, details in report.resource_breakdown.items():
                    cost = details.get("cost", 0.0)
                    percentage = (cost / total_cost) * 100
                    writer.writerow([
                        resource_type,
                        f"{percentage:.2f}%",
                        f"${cost:.{self.export_config.decimal_places}f}"
                    ])
        
        return output.getvalue()
    
    async def schedule_automated_reports(self, tenant_id: str, schedule_config: Dict[str, Any]) -> bool:
        """Schedule automated report generation (placeholder for future implementation)"""
        
        try:
            # This would integrate with a job scheduler like Celery
            # For now, just log the scheduling request
            
            logger.info(f"Scheduled automated reports for tenant {tenant_id} with config: {schedule_config}")
            
            # Example schedule config:
            # {
            #     "frequency": "monthly",
            #     "day_of_month": 1,
            #     "format": "csv",
            #     "email_recipients": ["billing@company.com"],
            #     "include_employees": True
            # }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule automated reports for tenant {tenant_id}: {e}")
            return False
    
    def update_export_config(self, config: BillingExportConfig):
        """Update export configuration"""
        self.export_config = config
        logger.info("Updated billing export configuration")
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """Get export service statistics"""
        
        return {
            "reports_generated": self._reports_generated,
            "exports_created": self._exports_created,
            "total_export_size_bytes": self._total_export_size_bytes,
            "average_export_size_bytes": self._total_export_size_bytes / max(1, self._exports_created),
            "supported_formats": ["csv", "json"],
            "export_config": asdict(self.export_config)
        }
    
    def reset_statistics(self):
        """Reset export statistics"""
        
        self._reports_generated = 0
        self._exports_created = 0
        self._total_export_size_bytes = 0

# Global usage report exporter
usage_report_exporter = UsageReportExporter()