"""
Azure Monitor Analytics Service

Provides business intelligence and analytics capabilities using Azure Monitor
with advanced querying, dashboards, and tenant-aware insights for Forge1.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from forge1.integrations.observability.azure_monitor import azure_monitor_integration
from forge1.integrations.base_adapter import ExecutionContext, TenantContext
from forge1.core.tenancy import get_current_tenant

logger = logging.getLogger(__name__)

class AnalyticsTimeRange(Enum):
    """Time ranges for analytics queries"""
    LAST_HOUR = "1h"
    LAST_24_HOURS = "24h"
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"
    LAST_90_DAYS = "90d"
    CUSTOM = "custom"

@dataclass
class AnalyticsQuery:
    """Analytics query structure"""
    query_name: str
    kql_query: str
    time_range: AnalyticsTimeRange
    parameters: Dict[str, Any]
    tenant_filter: bool = True

@dataclass
class AnalyticsResult:
    """Analytics query result"""
    query_name: str
    data: List[Dict[str, Any]]
    total_records: int
    execution_time_ms: float
    timestamp: datetime
    tenant_id: Optional[str] = None

class AzureMonitorAnalyticsService:
    """Service for business analytics using Azure Monitor data"""
    
    def __init__(self, azure_monitor=None):
        self.azure_monitor = azure_monitor or azure_monitor_integration
        
        # Pre-defined analytics queries
        self.predefined_queries = {
            "tenant_activity_summary": AnalyticsQuery(
                query_name="tenant_activity_summary",
                kql_query="""
                customEvents
                | where timestamp >= ago({time_range})
                | where customDimensions.tenant_id == "{tenant_id}"
                | summarize 
                    TotalEvents = count(),
                    UniqueUsers = dcount(customDimensions.user_id),
                    UniqueEmployees = dcount(customDimensions.employee_id),
                    AvgProcessingTime = avg(todouble(customDimensions.processing_time_ms))
                by bin(timestamp, 1h)
                | order by timestamp desc
                """,
                time_range=AnalyticsTimeRange.LAST_24_HOURS,
                parameters={}
            ),
            
            "request_performance_analysis": AnalyticsQuery(
                query_name="request_performance_analysis",
                kql_query="""
                requests
                | where timestamp >= ago({time_range})
                | where customDimensions.tenant_id == "{tenant_id}"
                | summarize 
                    RequestCount = count(),
                    AvgDuration = avg(duration),
                    P95Duration = percentile(duration, 95),
                    P99Duration = percentile(duration, 99),
                    ErrorRate = countif(success == false) * 100.0 / count()
                by url, bin(timestamp, 1h)
                | order by RequestCount desc
                """,
                time_range=AnalyticsTimeRange.LAST_24_HOURS,
                parameters={}
            ),
            
            "error_analysis": AnalyticsQuery(
                query_name="error_analysis",
                kql_query="""
                exceptions
                | where timestamp >= ago({time_range})
                | where customDimensions.tenant_id == "{tenant_id}"
                | summarize 
                    ErrorCount = count(),
                    UniqueErrors = dcount(type),
                    AffectedUsers = dcount(customDimensions.user_id)
                by type, bin(timestamp, 1h)
                | order by ErrorCount desc
                """,
                time_range=AnalyticsTimeRange.LAST_24_HOURS,
                parameters={}
            ),
            
            "usage_patterns": AnalyticsQuery(
                query_name="usage_patterns",
                kql_query="""
                customEvents
                | where timestamp >= ago({time_range})
                | where customDimensions.tenant_id == "{tenant_id}"
                | where name startswith "forge1_business_"
                | summarize 
                    EventCount = count(),
                    UniqueUsers = dcount(customDimensions.user_id)
                by name, customDimensions.employee_id, bin(timestamp, 1d)
                | order by timestamp desc, EventCount desc
                """,
                time_range=AnalyticsTimeRange.LAST_7_DAYS,
                parameters={}
            ),
            
            "cost_analysis": AnalyticsQuery(
                query_name="cost_analysis",
                kql_query="""
                customMetrics
                | where timestamp >= ago({time_range})
                | where customDimensions.tenant_id == "{tenant_id}"
                | where name contains "cost" or name contains "usage"
                | summarize 
                    TotalCost = sum(value),
                    AvgCost = avg(value),
                    MaxCost = max(value)
                by name, customDimensions.resource_type, bin(timestamp, 1d)
                | order by timestamp desc, TotalCost desc
                """,
                time_range=AnalyticsTimeRange.LAST_30_DAYS,
                parameters={}
            )
        }
        
        # Analytics cache
        self._query_cache: Dict[str, AnalyticsResult] = {}
        self._cache_ttl_seconds = 300  # 5 minutes
        
        # Statistics
        self._queries_executed = 0
        self._cache_hits = 0
        self._cache_misses = 0
    
    async def execute_analytics_query(self, query_name: str, tenant_id: str, 
                                    time_range: Optional[AnalyticsTimeRange] = None,
                                    custom_parameters: Optional[Dict[str, Any]] = None) -> Optional[AnalyticsResult]:
        """Execute a predefined analytics query"""
        
        try:
            # Get query definition
            if query_name not in self.predefined_queries:
                logger.error(f"Unknown analytics query: {query_name}")
                return None
            
            query_def = self.predefined_queries[query_name]
            
            # Use provided time range or default
            effective_time_range = time_range or query_def.time_range
            
            # Check cache first
            cache_key = f"{query_name}_{tenant_id}_{effective_time_range.value}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self._cache_hits += 1
                return cached_result
            
            self._cache_misses += 1
            
            # Prepare query parameters
            parameters = query_def.parameters.copy()
            if custom_parameters:
                parameters.update(custom_parameters)
            
            # Format the KQL query
            formatted_query = query_def.kql_query.format(
                time_range=effective_time_range.value,
                tenant_id=tenant_id,
                **parameters
            )
            
            # Execute the query (simulated - in real implementation would use Azure Monitor Query API)
            start_time = datetime.now()
            result_data = await self._execute_kql_query(formatted_query, tenant_id)
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create result
            result = AnalyticsResult(
                query_name=query_name,
                data=result_data,
                total_records=len(result_data),
                execution_time_ms=execution_time,
                timestamp=datetime.now(timezone.utc),
                tenant_id=tenant_id
            )
            
            # Cache the result
            self._cache_result(cache_key, result)
            
            self._queries_executed += 1
            logger.info(f"Executed analytics query {query_name} for tenant {tenant_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute analytics query {query_name}: {e}")
            return None
    
    async def execute_custom_query(self, kql_query: str, tenant_id: str, 
                                 query_name: str = "custom") -> Optional[AnalyticsResult]:
        """Execute a custom KQL query"""
        
        try:
            # Add tenant filter to custom query if not present
            if "tenant_id" not in kql_query and tenant_id != "system":
                kql_query = f"{kql_query}\n| where customDimensions.tenant_id == '{tenant_id}'"
            
            # Execute the query
            start_time = datetime.now()
            result_data = await self._execute_kql_query(kql_query, tenant_id)
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create result
            result = AnalyticsResult(
                query_name=query_name,
                data=result_data,
                total_records=len(result_data),
                execution_time_ms=execution_time,
                timestamp=datetime.now(timezone.utc),
                tenant_id=tenant_id
            )
            
            self._queries_executed += 1
            logger.info(f"Executed custom analytics query for tenant {tenant_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute custom analytics query: {e}")
            return None
    
    async def get_tenant_dashboard_data(self, tenant_id: str, 
                                      time_range: AnalyticsTimeRange = AnalyticsTimeRange.LAST_24_HOURS) -> Dict[str, Any]:
        """Get comprehensive dashboard data for a tenant"""
        
        try:
            # Execute multiple queries in parallel
            tasks = [
                self.execute_analytics_query("tenant_activity_summary", tenant_id, time_range),
                self.execute_analytics_query("request_performance_analysis", tenant_id, time_range),
                self.execute_analytics_query("error_analysis", tenant_id, time_range),
                self.execute_analytics_query("usage_patterns", tenant_id, time_range)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            dashboard_data = {
                "tenant_id": tenant_id,
                "time_range": time_range.value,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "activity_summary": results[0].data if results[0] and not isinstance(results[0], Exception) else [],
                "performance_analysis": results[1].data if results[1] and not isinstance(results[1], Exception) else [],
                "error_analysis": results[2].data if results[2] and not isinstance(results[2], Exception) else [],
                "usage_patterns": results[3].data if results[3] and not isinstance(results[3], Exception) else [],
                "summary_metrics": await self._calculate_summary_metrics(results, tenant_id)
            }
            
            logger.info(f"Generated dashboard data for tenant {tenant_id}")
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to get tenant dashboard data: {e}")
            return {
                "tenant_id": tenant_id,
                "error": str(e),
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
    
    async def get_cost_insights(self, tenant_id: str, 
                              time_range: AnalyticsTimeRange = AnalyticsTimeRange.LAST_30_DAYS) -> Dict[str, Any]:
        """Get cost and usage insights for a tenant"""
        
        try:
            cost_result = await self.execute_analytics_query("cost_analysis", tenant_id, time_range)
            
            if not cost_result:
                return {"error": "Failed to retrieve cost data"}
            
            # Calculate cost insights
            total_cost = sum(record.get("TotalCost", 0) for record in cost_result.data)
            avg_daily_cost = total_cost / max(1, len(cost_result.data))
            
            # Find top cost drivers
            cost_by_resource = {}
            for record in cost_result.data:
                resource_type = record.get("resource_type", "unknown")
                cost_by_resource[resource_type] = cost_by_resource.get(resource_type, 0) + record.get("TotalCost", 0)
            
            top_cost_drivers = sorted(cost_by_resource.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "tenant_id": tenant_id,
                "time_range": time_range.value,
                "total_cost": total_cost,
                "avg_daily_cost": avg_daily_cost,
                "top_cost_drivers": top_cost_drivers,
                "cost_trend": cost_result.data,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get cost insights: {e}")
            return {"error": str(e)}
    
    async def create_alert_rule(self, tenant_id: str, alert_name: str, 
                              kql_query: str, threshold: float, 
                              alert_type: str = "metric") -> bool:
        """Create an alert rule in Azure Monitor"""
        
        try:
            # In a real implementation, this would use Azure Monitor Alert Rules API
            alert_rule = {
                "name": alert_name,
                "tenant_id": tenant_id,
                "query": kql_query,
                "threshold": threshold,
                "alert_type": alert_type,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "enabled": True
            }
            
            # Send alert rule creation event
            await self.azure_monitor.send_custom_event(
                "alert_rule_created",
                alert_rule,
                ExecutionContext(
                    tenant_context=TenantContext(tenant_id=tenant_id),
                    request_id=f"alert_{int(datetime.now().timestamp())}",
                    metadata={"alert_name": alert_name}
                )
            )
            
            logger.info(f"Created alert rule {alert_name} for tenant {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create alert rule: {e}")
            return False
    
    def _get_cached_result(self, cache_key: str) -> Optional[AnalyticsResult]:
        """Get cached analytics result if still valid"""
        
        if cache_key in self._query_cache:
            cached_result = self._query_cache[cache_key]
            cache_age = (datetime.now(timezone.utc) - cached_result.timestamp).total_seconds()
            
            if cache_age < self._cache_ttl_seconds:
                return cached_result
            else:
                # Remove expired cache entry
                del self._query_cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: AnalyticsResult):
        """Cache analytics result"""
        self._query_cache[cache_key] = result
        
        # Clean up old cache entries if cache gets too large
        if len(self._query_cache) > 100:
            oldest_key = min(self._query_cache.keys(), 
                           key=lambda k: self._query_cache[k].timestamp)
            del self._query_cache[oldest_key]
    
    async def _execute_kql_query(self, kql_query: str, tenant_id: str) -> List[Dict[str, Any]]:
        """Execute KQL query against Azure Monitor (simulated)"""
        
        # In a real implementation, this would use Azure Monitor Query API
        # For now, return simulated data based on query type
        
        if "tenant_activity_summary" in kql_query:
            return [
                {
                    "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
                    "TotalEvents": 100 - i * 5,
                    "UniqueUsers": 20 - i,
                    "UniqueEmployees": 15 - i,
                    "AvgProcessingTime": 150 + i * 10
                }
                for i in range(24)
            ]
        
        elif "request_performance_analysis" in kql_query:
            return [
                {
                    "url": f"/api/endpoint{i}",
                    "RequestCount": 500 - i * 20,
                    "AvgDuration": 200 + i * 50,
                    "P95Duration": 400 + i * 100,
                    "P99Duration": 800 + i * 200,
                    "ErrorRate": i * 0.5
                }
                for i in range(10)
            ]
        
        elif "error_analysis" in kql_query:
            return [
                {
                    "type": f"Error{i}",
                    "ErrorCount": 50 - i * 5,
                    "UniqueErrors": 10 - i,
                    "AffectedUsers": 20 - i * 2
                }
                for i in range(5)
            ]
        
        else:
            # Generic response
            return [
                {
                    "timestamp": datetime.now().isoformat(),
                    "value": 100,
                    "tenant_id": tenant_id
                }
            ]
    
    async def _calculate_summary_metrics(self, query_results: List, tenant_id: str) -> Dict[str, Any]:
        """Calculate summary metrics from query results"""
        
        try:
            summary = {
                "total_events": 0,
                "unique_users": 0,
                "avg_response_time": 0,
                "error_rate": 0,
                "uptime_percentage": 99.9
            }
            
            # Extract metrics from activity summary
            if query_results[0] and not isinstance(query_results[0], Exception):
                activity_data = query_results[0].data
                if activity_data:
                    summary["total_events"] = sum(record.get("TotalEvents", 0) for record in activity_data)
                    summary["unique_users"] = max(record.get("UniqueUsers", 0) for record in activity_data)
            
            # Extract metrics from performance analysis
            if query_results[1] and not isinstance(query_results[1], Exception):
                perf_data = query_results[1].data
                if perf_data:
                    avg_durations = [record.get("AvgDuration", 0) for record in perf_data]
                    summary["avg_response_time"] = sum(avg_durations) / len(avg_durations) if avg_durations else 0
            
            # Extract metrics from error analysis
            if query_results[2] and not isinstance(query_results[2], Exception):
                error_data = query_results[2].data
                if error_data:
                    total_errors = sum(record.get("ErrorCount", 0) for record in error_data)
                    summary["error_rate"] = min(total_errors / max(1, summary["total_events"]) * 100, 100)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to calculate summary metrics: {e}")
            return {"error": str(e)}
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get analytics service statistics"""
        
        return {
            "queries_executed": self._queries_executed,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses),
            "cached_queries": len(self._query_cache),
            "available_queries": list(self.predefined_queries.keys())
        }
    
    def clear_cache(self):
        """Clear analytics query cache"""
        self._query_cache.clear()
        logger.info("Analytics query cache cleared")

# Global analytics service instance
azure_monitor_analytics_service = AzureMonitorAnalyticsService()