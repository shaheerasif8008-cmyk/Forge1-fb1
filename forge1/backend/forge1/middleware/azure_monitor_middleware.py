"""
Azure Monitor Middleware Stub

Simplified middleware for Azure Monitor telemetry collection.
This version handles missing Azure Monitor SDK gracefully.
"""

import asyncio
import logging
import time
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class AzureMonitorMiddleware(BaseHTTPMiddleware):
    """Simplified Azure Monitor middleware with graceful degradation"""
    
    def __init__(self, app, azure_monitor=None):
        super().__init__(app)
        self.azure_monitor = azure_monitor
        
        # Telemetry collection settings
        self.collect_request_telemetry = True
        self.collect_performance_metrics = True
        self.collect_error_telemetry = True
        self.collect_custom_events = True
        
        # Statistics
        self._requests_processed = 0
        self._telemetry_sent = 0
        self._telemetry_failures = 0
        
        logger.info("Azure Monitor middleware initialized (simplified version)")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect basic telemetry"""
        
        start_time = time.time()
        request_id = request.headers.get("X-Request-ID", f"req_{int(start_time)}")
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Log basic telemetry
            if self.collect_request_telemetry:
                logger.info(f"Request {request.method} {request.url.path} - "
                           f"{response.status_code} - {processing_time:.2f}ms")
            
            self._requests_processed += 1
            return response
            
        except Exception as e:
            # Calculate processing time for failed requests
            processing_time = (time.time() - start_time) * 1000
            
            # Log error
            if self.collect_error_telemetry:
                logger.error(f"Request {request.method} {request.url.path} failed - "
                           f"{processing_time:.2f}ms - {str(e)}")
            
            self._requests_processed += 1
            raise
    
    async def send_custom_business_event(self, event_name: str, properties: Dict[str, Any], 
                                       context: Optional[Any] = None):
        """Send custom business event (stub implementation)"""
        
        try:
            logger.info(f"Business event: {event_name} - {properties}")
            self._telemetry_sent += 1
            return True
            
        except Exception as e:
            self._telemetry_failures += 1
            logger.error(f"Failed to send custom business event {event_name}: {e}")
            return False
    
    async def send_tenant_activity_metric(self, activity_type: str, value: float, 
                                        context: Optional[Any] = None):
        """Send tenant activity metric (stub implementation)"""
        
        try:
            logger.info(f"Tenant activity metric: {activity_type} = {value}")
            self._telemetry_sent += 1
            return True
            
        except Exception as e:
            self._telemetry_failures += 1
            logger.error(f"Failed to send tenant activity metric: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get middleware statistics"""
        
        return {
            "requests_processed": self._requests_processed,
            "telemetry_sent": self._telemetry_sent,
            "telemetry_failures": self._telemetry_failures,
            "success_rate": self._telemetry_sent / max(1, self._telemetry_sent + self._telemetry_failures),
            "collect_request_telemetry": self.collect_request_telemetry,
            "collect_performance_metrics": self.collect_performance_metrics,
            "collect_error_telemetry": self.collect_error_telemetry,
            "collect_custom_events": self.collect_custom_events
        }
    
    def configure_telemetry_collection(self, request_telemetry: bool = True, 
                                     performance_metrics: bool = True,
                                     error_telemetry: bool = True, 
                                     custom_events: bool = True):
        """Configure telemetry collection settings"""
        
        self.collect_request_telemetry = request_telemetry
        self.collect_performance_metrics = performance_metrics
        self.collect_error_telemetry = error_telemetry
        self.collect_custom_events = custom_events
        
        logger.info(f"Azure Monitor telemetry collection configured: "
                   f"requests={request_telemetry}, performance={performance_metrics}, "
                   f"errors={error_telemetry}, events={custom_events}")

# Global Azure Monitor middleware instance
azure_monitor_middleware = AzureMonitorMiddleware