"""
Analytics Services

Provides business intelligence and analytics capabilities for Forge1
using various data sources including Azure Monitor, Prometheus, and custom metrics.
"""

from forge1.services.analytics.azure_monitor_analytics import azure_monitor_analytics_service

__all__ = [
    "azure_monitor_analytics_service"
]