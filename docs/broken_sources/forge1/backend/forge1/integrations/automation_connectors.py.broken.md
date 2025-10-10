```python
# forge1/backend/forge1/integrations/automation_connectors.py
"""
Automation Platform Connectors for Forge 1

Comprehensive automation platform integration providing:
- n8n webhook integration and workflow triggers
- Zapier connector for popular automation workflows
- Custom webhook support for enterprise integrations
- Event-driven automation and workflow orchestration
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import json
import uuid
import hmac
import hashlib
import aiohttp
from urllib.parse import urlencode

from forge1.core.security_manager import SecurityManager
from forge1.core.performance_monitor import PerformanceMonitor
from forge1.auth.authentication_manager import AuthenticationManager

logger = logging.getLogger(__name__)

class AutomationPlatform(Enum):
    """Supported automation platforms"""
    N8N = "n8n"
    ZAPIER = "zapier"
    MAKE = "make"
    CUSTOM_WEBHOOK = "custom_webhook"
    MICROSOFT_POWER_AUTOMATE = "power_automate"
    IFTTT = "ifttt"

class WebhookMethod(Enum):
    """HTTP methods for webhooks"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"

class TriggerType(Enum):
    """Types of automation triggers"""
    WEBHOOK = "webhook"
    SCHEDULE = "schedule"
    EVENT = "event"
    CONDITION = "condition"
    MANUAL = "manual"

class AutomationStatus(Enum):
    """Status of automation workflows"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    ERROR = "error"
    TESTING = "testing"c
lass BaseAutomationConnector:
    """Base class for automation platform connectors"""
    
    def __init__(
        self,
        platform: AutomationPlatform,
        name: str,
        base_url: str,
        auth_manager: AuthenticationManager,
        security_manager: SecurityManager,
        performance_monitor: PerformanceMonitor,
        **kwargs
    ):
        self.platform = platform
        self.name = name
        self.base_url = base_url
        self.auth_manager = auth_manager
        self.security_manager = security_manager
        self.performance_monitor = performance_monitor
        
        self.connector_id = f"{platform.value}_{uuid.uuid4().hex[:8]}"
        self.created_at = datetime.now(timezone.utc)
        self.config = kwargs
        
        # Connector state
        self.active_workflows = {}
        self.webhook_endpoints = {}
        self.trigger_handlers = {}
        
        # Performance metrics
        self.metrics = {
            "workflows_created": 0,
            "webhooks_triggered": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "total_execution_time": 0.0
        }
        
        logger.info(f"Automation connector {self.name} ({platform.value}) initialized")
    
    async def create_workflow(
        self,
        workflow_name: str,
        trigger_config: Dict[str, Any],
        actions: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Create automation workflow
        
        Args:
            workflow_name: Name of the workflow
            trigger_config: Trigger configuration
            actions: List of actions to perform
            **kwargs: Additional workflow configuration
            
        Returns:
            Workflow creation result
        """
        
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        
        try:
            # Validate workflow configuration
            await self._validate_workflow_config(trigger_config, actions)
            
            # Create workflow
            workflow = {
                "id": workflow_id,
                "name": workflow_name,
                "platform": self.platform.value,
                "connector_id": self.connector_id,
                "trigger": trigger_config,
                "actions": actions,
                "status": AutomationStatus.ACTIVE,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config": kwargs
            }
            
            # Store workflow
            self.active_workflows[workflow_id] = workflow
            
            # Set up trigger if needed
            if trigger_config.get("type") == TriggerType.WEBHOOK.value:
                await self._setup_webhook_trigger(workflow_id, trigger_config)
            
            # Update metrics
            self.metrics["workflows_created"] += 1
            
            logger.info(f"Workflow {workflow_name} created with ID {workflow_id}")
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                "platform": self.platform.value,
                "status": AutomationStatus.ACTIVE.value,
                "webhook_url": self.webhook_endpoints.get(workflow_id)
            }
            
        except Exception as e:
            logger.error(f"Failed to create workflow {workflow_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "workflow_name": workflow_name
            }
    
    async def _validate_workflow_config(
        self,
        trigger_config: Dict[str, Any],
        actions: List[Dict[str, Any]]
    ) -> None:
        """Validate workflow configuration"""
        
        # Validate trigger
        if not trigger_config.get("type"):
            raise ValueError("Trigger type is required")
        
        # Validate actions
        if not actions:
            raise ValueError("At least one action is required")
        
        for action in actions:
            if not action.get("type"):
                raise ValueError("Action type is required")
    
    async def _setup_webhook_trigger(
        self,
        workflow_id: str,
        trigger_config: Dict[str, Any]
    ) -> None:
        """Set up webhook trigger for workflow"""
        
        webhook_path = trigger_config.get("path", f"/webhook/{workflow_id}")
        webhook_url = f"{self.base_url}{webhook_path}"
        
        self.webhook_endpoints[workflow_id] = webhook_url
        
        # Register webhook handler
        self.trigger_handlers[webhook_path] = {
            "workflow_id": workflow_id,
            "method": trigger_config.get("method", WebhookMethod.POST.value),
            "auth_required": trigger_config.get("auth_required", True),
            "secret": trigger_config.get("secret")
        }
        
        logger.info(f"Webhook trigger set up for workflow {workflow_id}: {webhook_url}")
    
    async def trigger_workflow(
        self,
        workflow_id: str,
        payload: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Trigger workflow execution
        
        Args:
            workflow_id: Workflow ID to trigger
            payload: Trigger payload data
            context: Execution context
            
        Returns:
            Workflow execution result
        """
        
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now(timezone.utc)
        
        try:
            if workflow_id not in self.active_workflows:
                return {
                    "success": False,
                    "error": f"Workflow {workflow_id} not found",
                    "execution_id": execution_id
                }
            
            workflow = self.active_workflows[workflow_id]
            
            if workflow["status"] != AutomationStatus.ACTIVE.value:
                return {
                    "success": False,
                    "error": f"Workflow {workflow_id} is not active",
                    "execution_id": execution_id,
                    "status": workflow["status"]
                }
            
            # Security validation
            await self.security_manager.validate_workflow_execution(
                workflow_id=workflow_id,
                payload=payload or {},
                context=context or {}
            )
            
            # Execute workflow actions
            action_results = []
            for i, action in enumerate(workflow["actions"]):
                action_result = await self._execute_action(
                    action, payload or {}, context or {}, execution_id, i
                )
                action_results.append(action_result)
                
                # Stop on failure if configured
                if not action_result.get("success", False) and action.get("stop_on_failure", True):
                    break
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Determine overall success
            success = all(result.get("success", False) for result in action_results)
            
            # Update metrics
            await self._update_execution_metrics(execution_time, success)
            
            # Track performance
            await self.performance_monitor.track_workflow_execution(
                workflow_id=workflow_id,
                execution_id=execution_id,
                execution_time=execution_time,
                success=success,
                platform=self.platform.value
            )
            
            result = {
                "success": success,
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "workflow_name": workflow["name"],
                "platform": self.platform.value,
                "execution_time": execution_time,
                "actions_executed": len(action_results),
                "action_results": action_results,
                "triggered_at": start_time.isoformat()
            }
            
            logger.info(f"Workflow {workflow_id} executed: {success}")
            return result
            
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            await self._update_execution_metrics(execution_time, False)
            
            logger.error(f"Workflow execution failed for {workflow_id}: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "execution_time": execution_time
            }
    
    async def _execute_action(
        self,
        action: Dict[str, Any],
        payload: Dict[str, Any],
        context: Dict[str, Any],
        execution_id: str,
        action_index: int
    ) -> Dict[str, Any]:
        """Execute individual workflow action"""
        
        action_id = f"{execution_id}_action_{action_index}"
        action_start = datetime.now(timezone.utc)
        
        try:
            action_type = action["type"]
            
            if action_type == "http_request":
                result = await self._execute_http_action(action, payload, context)
            elif action_type == "webhook":
                result = await self._execute_webhook_action(action, payload, context)
            elif action_type == "email":
                result = await self._execute_email_action(action, payload, context)
            elif action_type == "delay":
                result = await self._execute_delay_action(action, payload, context)
            elif action_type == "condition":
                result = await self._execute_condition_action(action, payload, context)
            else:
                result = await self._execute_custom_action(action, payload, context)
            
            execution_time = (datetime.now(timezone.utc) - action_start).total_seconds()
            
            result.update({
                "action_id": action_id,
                "action_type": action_type,
                "execution_time": execution_time,
                "executed_at": action_start.isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Action execution failed for {action_id}: {e}")
            return {
                "action_id": action_id,
                "action_type": action.get("type", "unknown"),
                "success": False,
                "error": str(e),
                "execution_time": (datetime.now(timezone.utc) - action_start).total_seconds()
            }
    
    async def _execute_http_action(
        self,
        action: Dict[str, Any],
        payload: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute HTTP request action"""
        
        url = action["url"]
        method = action.get("method", "POST")
        headers = action.get("headers", {})
        data = action.get("data", payload)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response_data = await response.text()
                    
                    return {
                        "success": response.status < 400,
                        "status_code": response.status,
                        "response": response_data,
                        "url": url,
                        "method": method
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "method": method
            }
    
    async def _execute_webhook_action(
        self,
        action: Dict[str, Any],
        payload: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute webhook action"""
        
        webhook_url = action["webhook_url"]
        method = action.get("method", "POST")
        headers = action.get("headers", {"Content-Type": "application/json"})
        
        # Add webhook signature if secret is provided
        if action.get("secret"):
            signature = self._generate_webhook_signature(
                json.dumps(payload), action["secret"]
            )
            headers["X-Webhook-Signature"] = signature
        
        return await self._execute_http_action({
            "url": webhook_url,
            "method": method,
            "headers": headers,
            "data": payload
        }, payload, context)
    
    async def _execute_email_action(
        self,
        action: Dict[str, Any],
        payload: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute email action"""
        
        # Mock email sending for demonstration
        return {
            "success": True,
            "recipient": action.get("to", "unknown"),
            "subject": action.get("subject", "Automation Notification"),
            "message_id": f"email_{uuid.uuid4().hex[:8]}"
        }
    
    async def _execute_delay_action(
        self,
        action: Dict[str, Any],
        payload: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute delay action"""
        
        delay_seconds = action.get("seconds", 1)
        await asyncio.sleep(delay_seconds)
        
        return {
            "success": True,
            "delay_seconds": delay_seconds
        }
    
    async def _execute_condition_action(
        self,
        action: Dict[str, Any],
        payload: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute condition action"""
        
        condition = action.get("condition", "true")
        
        # Simple condition evaluation (in production, use safe expression evaluator)
        try:
            # Mock condition evaluation
            result = True  # Simplified for demonstration
            
            return {
                "success": True,
                "condition": condition,
                "result": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "condition": condition,
                "error": str(e)
            }
    
    async def _execute_custom_action(
        self,
        action: Dict[str, Any],
        payload: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute custom action"""
        
        return {
            "success": True,
            "action_type": action.get("type", "custom"),
            "message": "Custom action executed successfully"
        }
    
    def _generate_webhook_signature(self, payload: str, secret: str) -> str:
        """Generate webhook signature for security"""
        
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"
    
    async def _update_execution_metrics(self, execution_time: float, success: bool) -> None:
        """Update execution metrics"""
        
        if success:
            self.metrics["successful_executions"] += 1
        else:
            self.metrics["failed_executions"] += 1
        
        # Update average execution time
        total_executions = self.metrics["successful_executions"] + self.metrics["failed_executions"]
        self.metrics["total_execution_time"] += execution_time
        self.metrics["average_execution_time"] = (
            self.metrics["total_execution_time"] / total_executions
        )
    
    async def handle_webhook(
        self,
        path: str,
        method: str,
        headers: Dict[str, str],
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle incoming webhook request
        
        Args:
            path: Webhook path
            method: HTTP method
            headers: Request headers
            payload: Request payload
            
        Returns:
            Webhook handling result
        """
        
        if path not in self.trigger_handlers:
            return {
                "success": False,
                "error": "Webhook endpoint not found",
                "path": path
            }
        
        handler = self.trigger_handlers[path]
        workflow_id = handler["workflow_id"]
        
        try:
            # Validate method
            if method.upper() != handler["method"].upper():
                return {
                    "success": False,
                    "error": f"Method {method} not allowed",
                    "expected_method": handler["method"]
                }
            
            # Validate signature if secret is configured
            if handler.get("secret"):
                signature = headers.get("X-Webhook-Signature", "")
                expected_signature = self._generate_webhook_signature(
                    json.dumps(payload), handler["secret"]
                )
                
                if not hmac.compare_digest(signature, expected_signature):
                    return {
                        "success": False,
                        "error": "Invalid webhook signature"
                    }
            
            # Update metrics
            self.metrics["webhooks_triggered"] += 1
            
            # Trigger workflow
            result = await self.trigger_workflow(workflow_id, payload, {
                "trigger_type": "webhook",
                "webhook_path": path,
                "headers": headers
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Webhook handling failed for {path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "path": path,
                "workflow_id": workflow_id
            }
    
    def get_connector_info(self) -> Dict[str, Any]:
        """Get connector information and metrics"""
        
        return {
            "connector_id": self.connector_id,
            "name": self.name,
            "platform": self.platform.value,
            "base_url": self.base_url,
            "created_at": self.created_at.isoformat(),
            "active_workflows": len(self.active_workflows),
            "webhook_endpoints": len(self.webhook_endpoints),
            "metrics": self.metrics.copy(),
            "config": self.config
        }
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows"""
        
        workflows = []
        for workflow in self.active_workflows.values():
            workflow_info = {
                "id": workflow["id"],
                "name": workflow["name"],
                "status": workflow["status"],
                "created_at": workflow["created_at"],
                "trigger_type": workflow["trigger"].get("type"),
                "actions_count": len(workflow["actions"]),
                "webhook_url": self.webhook_endpoints.get(workflow["id"])
            }
            workflows.append(workflow_info)
        
        return workflows
    
    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete workflow
        
        Args:
            workflow_id: Workflow ID to delete
            
        Returns:
            True if deleted successfully
        """
        
        if workflow_id not in self.active_workflows:
            return False
        
        try:
            # Remove webhook endpoint if exists
            if workflow_id in self.webhook_endpoints:
                webhook_url = self.webhook_endpoints[workflow_id]
                del self.webhook_endpoints[workflow_id]
                
                # Remove trigger handler
                for path, handler in list(self.trigger_handlers.items()):
                    if handler["workflow_id"] == workflow_id:
                        del self.trigger_handlers[path]
                        break
            
            # Remove workflow
            del self.active_workflows[workflow_id]
            
            logger.info(f"Workflow {workflow_id} deleted")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete workflow {workflow_id}: {e}")
            return False      
          workflow_id=workflow_id,
                execution_id=execution_id,
                execution_time=execution_time,
                success=False,
                error=str(e)
            )
            
            logger.error(f"Zapier Zap execution failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "execution_time": execution_time
            }
    
    async def register_webhook(
        self,
        webhook_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Register webhook endpoint for Zapier"""
        
        webhook_id = f"zapier_webhook_{uuid.uuid4().hex[:8]}"
        
        try:
            webhook_url = webhook_config.get("url", self.zapier_webhook_url)
            
            webhook_info = {
                "webhook_id": webhook_id,
                "url": webhook_url,
                "method": webhook_config.get("method", "POST"),
                "secret": webhook_config.get("secret"),
                "workflow_id": webhook_config.get("workflow_id"),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "active"
            }
            
            self.webhook_endpoints[webhook_id] = webhook_info
            
            logger.info(f"Registered Zapier webhook {webhook_id}: {webhook_url}")
            
            return {
                "success": True,
                "webhook_id": webhook_id,
                "webhook_url": webhook_url,
                "webhook_info": webhook_info
            }
            
        except Exception as e:
            logger.error(f"Failed to register Zapier webhook: {e}")
            return {
                "success": False,
                "error": str(e),
                "webhook_id": webhook_id
            }
    
    def _update_average_execution_time(self, execution_time: float) -> None:
        """Update average execution time metric"""
        current_avg = self.metrics["average_execution_time"]
        executions = self.metrics["workflows_executed"]
        
        self.metrics["average_execution_time"] = (
            (current_avg * (executions - 1) + execution_time) / executions
        )
    
    async def close(self) -> None:
        """Close connector and cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None

class CustomWebhookConnector(BaseAutomationConnector):
    """Custom webhook connector for enterprise integrations"""
    
    def __init__(
        self,
        auth_manager: AuthenticationManager,
        security_manager: SecurityManager,
        performance_monitor: PerformanceMonitor,
        webhook_base_url: str = "https://api.forge1.com/webhooks",
        **kwargs
    ):
        super().__init__(
            AutomationPlatform.CUSTOM_WEBHOOK,
            auth_manager,
            security_manager,
            performance_monitor,
            **kwargs
        )
        self.webhook_base_url = webhook_base_url.rstrip('/')
        self.webhook_handlers = {}
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get HTTP session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def create_workflow(
        self,
        name: str,
        triggers: List[Dict[str, Any]],
        actions: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Create custom webhook workflow"""
        
        workflow_id = f"webhook_workflow_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create webhook workflow definition
            workflow_definition = {
                "name": name,
                "description": kwargs.get("description", f"Custom webhook workflow: {name}"),
                "triggers": triggers,
                "actions": actions,
                "security": {
                    "require_signature": kwargs.get("require_signature", True),
                    "allowed_ips": kwargs.get("allowed_ips", []),
                    "rate_limit": kwargs.get("rate_limit", {"requests": 100, "window": 3600})
                },
                "retry_policy": {
                    "max_retries": kwargs.get("max_retries", 3),
                    "backoff_factor": kwargs.get("backoff_factor", 2.0),
                    "timeout": kwargs.get("timeout", 30)
                }
            }
            
            # Store workflow
            self.active_workflows[workflow_id] = {
                "id": workflow_id,
                "name": name,
                "definition": workflow_definition,
                "triggers": triggers,
                "actions": actions,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "active"
            }
            
            # Create webhook handlers for triggers
            for trigger in triggers:
                if trigger.get("type") == TriggerType.WEBHOOK.value:
                    webhook_path = trigger.get("path", f"/webhook/{workflow_id}")
                    self.webhook_handlers[webhook_path] = {
                        "workflow_id": workflow_id,
                        "handler": self._create_webhook_handler(workflow_id, trigger)
                    }
            
            # Update metrics
            self.metrics["workflows_created"] += 1
            
            logger.info(f"Created custom webhook workflow {workflow_id}: {name}")
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "platform": "custom_webhook",
                "name": name,
                "webhook_endpoints": list(self.webhook_handlers.keys()),
                "definition": workflow_definition
            }
            
        except Exception as e:
            logger.error(f"Failed to create custom webhook workflow: {e}")
            return {
                "success": False,
                "error": str(e),
                "workflow_id": workflow_id
            }
    
    def _create_webhook_handler(
        self,
        workflow_id: str,
        trigger_config: Dict[str, Any]
    ) -> Callable:
        """Create webhook handler function"""
        
        async def webhook_handler(request_data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle incoming webhook request"""
            
            execution_id = f"exec_{uuid.uuid4().hex[:8]}"
            start_time = datetime.now(timezone.utc)
            
            try:
                # Validate webhook signature if required
                if trigger_config.get("require_signature"):
                    signature_valid = await self._validate_webhook_signature(
                        request_data, trigger_config.get("secret")
                    )
                    if not signature_valid:
                        return {
                            "success": False,
                            "error": "Invalid webhook signature",
                            "execution_id": execution_id
                        }
                
                # Execute workflow
                result = await self.execute_workflow(workflow_id, request_data)
                
                # Update webhook metrics
                self.metrics["webhooks_received"] += 1
                
                return result
                
            except Exception as e:
                logger.error(f"Webhook handler error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "execution_id": execution_id
                }
        
        return webhook_handler
    
    async def _validate_webhook_signature(
        self,
        request_data: Dict[str, Any],
        secret: str
    ) -> bool:
        """Validate webhook signature"""
        
        if not secret:
            return True  # No signature validation required
        
        try:
            # Get signature from headers
            signature = request_data.get("headers", {}).get("X-Webhook-Signature")
            if not signature:
                return False
            
            # Calculate expected signature
            payload = json.dumps(request_data.get("body", {}), sort_keys=True)
            expected_signature = hmac.new(
                secret.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures
            return hmac.compare_digest(signature, f"sha256={expected_signature}")
            
        except Exception as e:
            logger.error(f"Signature validation error: {e}")
            return False
    
    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute custom webhook workflow"""
        
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now(timezone.utc)
        
        try:
            if workflow_id not in self.active_workflows:
                return {
                    "success": False,
                    "error": "Workflow not found",
                    "workflow_id": workflow_id,
                    "execution_id": execution_id
                }
            
            workflow = self.active_workflows[workflow_id]
            actions = workflow["actions"]
            
            # Execute actions sequentially
            action_results = []
            for i, action in enumerate(actions):
                action_result = await self._execute_action(action, input_data, i)
                action_results.append(action_result)
                
                # Stop on action failure if configured
                if not action_result.get("success") and not action.get("continue_on_failure", False):
                    break
            
            # Compile execution result
            execution_result = {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "workflow_name": workflow["name"],
                "input_data": input_data or {},
                "action_results": action_results,
                "status": "completed" if all(r.get("success") for r in action_results) else "partial",
                "started_at": start_time.isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Update metrics
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.metrics["workflows_executed"] += 1
            self.metrics["actions_performed"] += len(action_results)
            self._update_average_execution_time(execution_time)
            
            # Track performance
            await self.performance_monitor.track_automation_execution(
                platform="custom_webhook",
                workflow_id=workflow_id,
                execution_id=execution_id,
                execution_time=execution_time,
                success=execution_result["status"] == "completed"
            )
            
            logger.info(f"Executed custom webhook workflow {workflow_id} in {execution_time:.2f}s")
            
            return {
                "success": True,
                "execution_result": execution_result,
                "execution_time": execution_time
            }
            
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.metrics["errors_encountered"] += 1
            
            await self.performance_monitor.track_automation_execution(
                platform="custom_webhook",
                workflow_id=workflow_id,
                execution_id=execution_id,
                execution_time=execution_time,
                success=False,
                error=str(e)
            )
            
            logger.error(f"Custom webhook workflow execution failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "execution_time": execution_time
            }
    
    async def _execute_action(
        self,
        action: Dict[str, Any],
        input_data: Dict[str, Any],
        action_index: int
    ) -> Dict[str, Any]:
        """Execute individual action"""
        
        action_type = action.get("type", ActionType.HTTP_REQUEST.value)
        action_id = f"action_{action_index}"
        
        try:
            if action_type == ActionType.HTTP_REQUEST.value:
                return await self._execute_http_request(action, input_data)
            elif action_type == ActionType.EMAIL_SEND.value:
                return await self._execute_email_send(action, input_data)
            elif action_type == ActionType.NOTIFICATION.value:
                return await self._execute_notification(action, input_data)
            elif action_type == ActionType.AGENT_EXECUTION.value:
                return await self._execute_agent_action(action, input_data)
            else:
                # Generic action execution
                return {
                    "success": True,
                    "action_id": action_id,
                    "action_type": action_type,
                    "result": f"Mock execution of {action_type} action",
                    "input_data": input_data
                }
                
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return {
                "success": False,
                "action_id": action_id,
                "action_type": action_type,
                "error": str(e)
            }
    
    async def _execute_http_request(
        self,
        action: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute HTTP request action"""
        
        session = await self._get_session()
        
        url = action.get("url", "")
        method = action.get("method", "POST").upper()
        headers = action.get("headers", {})
        data = action.get("data", input_data)
        timeout = action.get("timeout", 30)
        
        try:
            async with session.request(
                method=method,
                url=url,
                json=data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                response_data = await response.text()
                
                return {
                    "success": response.status < 400,
                    "action_type": "http_request",
                    "status_code": response.status,
                    "response_data": response_data,
                    "url": url,
                    "method": method
                }
                
        except Exception as e:
            return {
                "success": False,
                "action_type": "http_request",
                "error": str(e),
                "url": url,
                "method": method
            }
    
    async def _execute_email_send(
        self,
        action: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute email send action (mock implementation)"""
        
        return {
            "success": True,
            "action_type": "email_send",
            "to": action.get("to", ""),
            "subject": action.get("subject", ""),
            "result": "Email sent successfully (mock)"
        }
    
    async def _execute_notification(
        self,
        action: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute notification action (mock implementation)"""
        
        return {
            "success": True,
            "action_type": "notification",
            "channel": action.get("channel", ""),
            "message": action.get("message", ""),
            "result": "Notification sent successfully (mock)"
        }
    
    async def _execute_agent_action(
        self,
        action: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute agent action (integration with Forge 1 agents)"""
        
        return {
            "success": True,
            "action_type": "agent_execution",
            "agent_type": action.get("agent_type", ""),
            "task": action.get("task", ""),
            "result": "Agent execution completed successfully (mock)"
        }
    
    async def register_webhook(
        self,
        webhook_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Register custom webhook endpoint"""
        
        webhook_id = f"custom_webhook_{uuid.uuid4().hex[:8]}"
        
        try:
            webhook_path = webhook_config.get("path", f"/webhook/{webhook_id}")
            webhook_url = f"{self.webhook_base_url}{webhook_path}"
            
            webhook_info = {
                "webhook_id": webhook_id,
                "url": webhook_url,
                "path": webhook_path,
                "method": webhook_config.get("method", "POST"),
                "secret": webhook_config.get("secret"),
                "workflow_id": webhook_config.get("workflow_id"),
                "security": {
                    "require_signature": webhook_config.get("require_signature", True),
                    "allowed_ips": webhook_config.get("allowed_ips", []),
                    "rate_limit": webhook_config.get("rate_limit", {"requests": 100, "window": 3600})
                },
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "active"
            }
            
            self.webhook_endpoints[webhook_id] = webhook_info
            
            logger.info(f"Registered custom webhook {webhook_id}: {webhook_url}")
            
            return {
                "success": True,
                "webhook_id": webhook_id,
                "webhook_url": webhook_url,
                "webhook_info": webhook_info
            }
            
        except Exception as e:
            logger.error(f"Failed to register custom webhook: {e}")
            return {
                "success": False,
                "error": str(e),
                "webhook_id": webhook_id
            }
    
    def _update_average_execution_time(self, execution_time: float) -> None:
        """Update average execution time metric"""
        current_avg = self.metrics["average_execution_time"]
        executions = self.metrics["workflows_executed"]
        
        if executions > 0:
            self.metrics["average_execution_time"] = (
                (current_avg * (executions - 1) + execution_time) / executions
            )
    
    async def close(self) -> None:
        """Close connector and cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None

class AutomationConnectorManager:
    """Manager for automation platform connectors"""
    
    def __init__(
        self,
        auth_manager: AuthenticationManager,
        security_manager: SecurityManager,
        performance_monitor: PerformanceMonitor
    ):
        self.auth_manager = auth_manager
        self.security_manager = security_manager
        self.performance_monitor = performance_monitor
        self.connectors = {}
        self.connector_configs = {}
    
    async def create_connector(
        self,
        platform: AutomationPlatform,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create automation platform connector"""
        
        connector_id = f"{platform.value}_{uuid.uuid4().hex[:8]}"
        
        try:
            if platform == AutomationPlatform.N8N:
                connector = N8NConnector(
                    self.auth_manager,
                    self.security_manager,
                    self.performance_monitor,
                    **config
                )
            elif platform == AutomationPlatform.ZAPIER:
                connector = ZapierConnector(
                    self.auth_manager,
                    self.security_manager,
                    self.performance_monitor,
                    **config
                )
            elif platform == AutomationPlatform.CUSTOM_WEBHOOK:
                connector = CustomWebhookConnector(
                    self.auth_manager,
                    self.security_manager,
                    self.performance_monitor,
                    **config
                )
            else:
                return {
                    "success": False,
                    "error": f"Unsupported platform: {platform.value}",
                    "connector_id": connector_id
                }
            
            self.connectors[connector_id] = connector
            self.connector_configs[connector_id] = {
                "platform": platform,
                "config": config,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"Created {platform.value} connector {connector_id}")
            
            return {
                "success": True,
                "connector_id": connector_id,
                "platform": platform.value,
                "connector_info": connector.get_connector_info()
            }
            
        except Exception as e:
            logger.error(f"Failed to create {platform.value} connector: {e}")
            return {
                "success": False,
                "error": str(e),
                "connector_id": connector_id
            }
    
    async def get_connector(self, connector_id: str) -> Optional[BaseAutomationConnector]:
        """Get connector by ID"""
        return self.connectors.get(connector_id)
    
    async def list_connectors(self) -> Dict[str, Any]:
        """List all connectors"""
        
        connector_list = []
        for connector_id, connector in self.connectors.items():
            connector_info = connector.get_connector_info()
            connector_info["connector_id"] = connector_id
            connector_info["config"] = self.connector_configs.get(connector_id, {})
            connector_list.append(connector_info)
        
        return {
            "connectors": connector_list,
            "total_count": len(connector_list)
        }
    
    async def remove_connector(self, connector_id: str) -> Dict[str, Any]:
        """Remove connector"""
        
        try:
            if connector_id not in self.connectors:
                return {
                    "success": False,
                    "error": "Connector not found",
                    "connector_id": connector_id
                }
            
            connector = self.connectors[connector_id]
            await connector.close()
            
            del self.connectors[connector_id]
            del self.connector_configs[connector_id]
            
            logger.info(f"Removed connector {connector_id}")
            
            return {
                "success": True,
                "connector_id": connector_id,
                "message": "Connector removed successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to remove connector {connector_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "connector_id": connector_id
            }
    
    async def execute_workflow(
        self,
        connector_id: str,
        workflow_id: str,
        input_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute workflow on specific connector"""
        
        try:
            connector = self.connectors.get(connector_id)
            if not connector:
                return {
                    "success": False,
                    "error": "Connector not found",
                    "connector_id": connector_id
                }
            
            return await connector.execute_workflow(workflow_id, input_data)
            
        except Exception as e:
            logger.error(f"Failed to execute workflow: {e}")
            return {
                "success": False,
                "error": str(e),
                "connector_id": connector_id,
                "workflow_id": workflow_id
            }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics from all connectors"""
        
        total_metrics = {
            "total_connectors": len(self.connectors),
            "connectors_by_platform": {},
            "total_workflows": 0,
            "total_executions": 0,
            "total_webhooks": 0,
            "total_errors": 0,
            "average_execution_time": 0.0
        }
        
        execution_times = []
        
        for connector in self.connectors.values():
            platform = connector.platform.value
            metrics = connector.metrics
            
            # Count by platform
            if platform not in total_metrics["connectors_by_platform"]:
                total_metrics["connectors_by_platform"][platform] = 0
            total_metrics["connectors_by_platform"][platform] += 1
            
            # Aggregate metrics
            total_metrics["total_workflows"] += metrics["workflows_created"]
            total_metrics["total_executions"] += metrics["workflows_executed"]
            total_metrics["total_webhooks"] += metrics["webhooks_received"]
            total_metrics["total_errors"] += metrics["errors_encountered"]
            
            if metrics["workflows_executed"] > 0:
                execution_times.append(metrics["average_execution_time"])
        
        # Calculate overall average execution time
        if execution_times:
            total_metrics["average_execution_time"] = sum(execution_times) / len(execution_times)
        
        return total_metrics
    
    async def close_all(self) -> None:
        """Close all connectors"""
        
        for connector in self.connectors.values():
            try:
                await connector.close()
            except Exception as e:
                logger.error(f"Error closing connector: {e}")
        
        self.connectors.clear()
        self.connector_configs.clear()
```
