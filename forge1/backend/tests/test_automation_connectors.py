# forge1/backend/tests/test_automation_connectors.py

"""
Test suite for automation platform connectors
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from forge1.integrations.automation_connectors import (
    AutomationPlatform,
    TriggerType,
    ActionType,
    N8NConnector,
    ZapierConnector,
    CustomWebhookConnector,
    AutomationConnectorManager
)
from forge1.auth.authentication_manager import AuthenticationManager
from forge1.core.security_manager import SecurityManager
from forge1.core.performance_monitor import PerformanceMonitor

@pytest.fixture
def mock_auth_manager():
    """Mock authentication manager"""
    auth_manager = Mock(spec=AuthenticationManager)
    auth_manager.authenticate = AsyncMock(return_value={
        "success": True,
        "headers": {"Authorization": "Bearer test-token"}
    })
    return auth_manager

@pytest.fixture
def mock_security_manager():
    """Mock security manager"""
    return Mock(spec=SecurityManager)

@pytest.fixture
def mock_performance_monitor():
    """Mock performance monitor"""
    monitor = Mock(spec=PerformanceMonitor)
    monitor.track_automation_execution = AsyncMock()
    return monitor

@pytest.fixture
def n8n_connector(mock_auth_manager, mock_security_manager, mock_performance_monitor):
    """N8N connector fixture"""
    return N8NConnector(
        auth_manager=mock_auth_manager,
        security_manager=mock_security_manager,
        performance_monitor=mock_performance_monitor,
        n8n_url="https://n8n.example.com",
        credential_id="test-cred"
    )

@pytest.fixture
def zapier_connector(mock_auth_manager, mock_security_manager, mock_performance_monitor):
    """Zapier connector fixture"""
    return ZapierConnector(
        auth_manager=mock_auth_manager,
        security_manager=mock_security_manager,
        performance_monitor=mock_performance_monitor,
        zapier_webhook_url="https://hooks.zapier.com/test",
        credential_id="test-cred"
    )

@pytest.fixture
def webhook_connector(mock_auth_manager, mock_security_manager, mock_performance_monitor):
    """Custom webhook connector fixture"""
    return CustomWebhookConnector(
        auth_manager=mock_auth_manager,
        security_manager=mock_security_manager,
        performance_monitor=mock_performance_monitor,
        webhook_base_url="https://api.forge1.com/webhooks"
    )

@pytest.fixture
def connector_manager(mock_auth_manager, mock_security_manager, mock_performance_monitor):
    """Automation connector manager fixture"""
    return AutomationConnectorManager(
        auth_manager=mock_auth_manager,
        security_manager=mock_security_manager,
        performance_monitor=mock_performance_monitor
    )

class TestN8NConnector:
    """Test N8N connector functionality"""
    
    @pytest.mark.asyncio
    async def test_create_workflow(self, n8n_connector):
        """Test N8N workflow creation"""
        
        triggers = [{
            "type": TriggerType.WEBHOOK.value,
            "method": "POST",
            "path": "/test-webhook"
        }]
        
        actions = [{
            "type": ActionType.HTTP_REQUEST.value,
            "url": "https://api.example.com/endpoint",
            "method": "POST",
            "data": {"test": "data"}
        }]
        
        result = await n8n_connector.create_workflow(
            name="Test Workflow",
            triggers=triggers,
            actions=actions
        )
        
        assert result["success"] is True
        assert result["platform"] == "n8n"
        assert result["name"] == "Test Workflow"
        assert "workflow_id" in result
        assert "definition" in result
        
        # Check workflow is stored
        workflow_id = result["workflow_id"]
        assert workflow_id in n8n_connector.active_workflows
        
        # Check metrics updated
        assert n8n_connector.metrics["workflows_created"] == 1
    
    @pytest.mark.asyncio
    async def test_execute_workflow(self, n8n_connector):
        """Test N8N workflow execution"""
        
        # First create a workflow
        triggers = [{"type": TriggerType.WEBHOOK.value}]
        actions = [{"type": ActionType.HTTP_REQUEST.value}]
        
        create_result = await n8n_connector.create_workflow(
            name="Test Workflow",
            triggers=triggers,
            actions=actions
        )
        
        workflow_id = create_result["workflow_id"]
        
        # Execute the workflow
        input_data = {"test": "input"}
        result = await n8n_connector.execute_workflow(workflow_id, input_data)
        
        assert result["success"] is True
        assert "execution_result" in result
        assert "execution_time" in result
        
        execution_result = result["execution_result"]
        assert execution_result["workflow_id"] == workflow_id
        assert execution_result["input_data"] == input_data
        assert execution_result["status"] == "completed"
        
        # Check metrics updated
        assert n8n_connector.metrics["workflows_executed"] == 1
    
    @pytest.mark.asyncio
    async def test_register_webhook(self, n8n_connector):
        """Test N8N webhook registration"""
        
        webhook_config = {
            "path": "test-webhook",
            "method": "POST",
            "secret": "test-secret",
            "workflow_id": "test-workflow"
        }
        
        result = await n8n_connector.register_webhook(webhook_config)
        
        assert result["success"] is True
        assert "webhook_id" in result
        assert "webhook_url" in result
        assert result["webhook_url"].endswith("/test-webhook")
        
        # Check webhook is stored
        webhook_id = result["webhook_id"]
        assert webhook_id in n8n_connector.webhook_endpoints
    
    @pytest.mark.asyncio
    async def test_connector_info(self, n8n_connector):
        """Test connector info retrieval"""
        
        info = n8n_connector.get_connector_info()
        
        assert "connector_id" in info
        assert info["platform"] == "n8n"
        assert "created_at" in info
        assert "metrics" in info
        assert info["active_workflows"] == 0
        assert info["webhook_endpoints"] == 0

class TestZapierConnector:
    """Test Zapier connector functionality"""
    
    @pytest.mark.asyncio
    async def test_create_workflow(self, zapier_connector):
        """Test Zapier Zap creation"""
        
        triggers = [{
            "type": TriggerType.WEBHOOK.value,
            "url": "https://hooks.zapier.com/test"
        }]
        
        actions = [{
            "type": ActionType.EMAIL_SEND.value,
            "to": "test@example.com",
            "subject": "Test Email",
            "body": "Test message"
        }]
        
        result = await zapier_connector.create_workflow(
            name="Test Zap",
            triggers=triggers,
            actions=actions,
            description="Test Zapier integration"
        )
        
        assert result["success"] is True
        assert result["platform"] == "zapier"
        assert result["name"] == "Test Zap"
        assert "workflow_id" in result
        assert "zap_definition" in result
        
        # Check Zap definition structure
        zap_def = result["zap_definition"]
        assert zap_def["title"] == "Test Zap"
        assert "trigger" in zap_def
        assert "actions" in zap_def
        assert zap_def["status"] == "on"
    
    @pytest.mark.asyncio
    async def test_execute_workflow_with_webhook(self, zapier_connector):
        """Test Zapier Zap execution with webhook"""
        
        # Create a Zap first
        triggers = [{"type": TriggerType.WEBHOOK.value}]
        actions = [{"type": ActionType.NOTIFICATION.value}]
        
        create_result = await zapier_connector.create_workflow(
            name="Test Zap",
            triggers=triggers,
            actions=actions
        )
        
        workflow_id = create_result["workflow_id"]
        
        # Execute the Zap
        input_data = {"trigger_data": "test"}
        result = await zapier_connector.execute_workflow(workflow_id, input_data)
        
        assert result["success"] is True
        assert "execution_result" in result
        
        execution_result = result["execution_result"]
        assert execution_result["workflow_id"] == workflow_id
        assert execution_result["input_data"] == input_data
        assert execution_result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_register_webhook(self, zapier_connector):
        """Test Zapier webhook registration"""
        
        webhook_config = {
            "url": "https://hooks.zapier.com/custom",
            "method": "POST",
            "secret": "zapier-secret"
        }
        
        result = await zapier_connector.register_webhook(webhook_config)
        
        assert result["success"] is True
        assert "webhook_id" in result
        assert result["webhook_url"] == webhook_config["url"]

class TestCustomWebhookConnector:
    """Test custom webhook connector functionality"""
    
    @pytest.mark.asyncio
    async def test_create_workflow(self, webhook_connector):
        """Test custom webhook workflow creation"""
        
        triggers = [{
            "type": TriggerType.WEBHOOK.value,
            "path": "/custom-webhook",
            "require_signature": True,
            "secret": "webhook-secret"
        }]
        
        actions = [
            {
                "type": ActionType.HTTP_REQUEST.value,
                "url": "https://api.example.com/process",
                "method": "POST"
            },
            {
                "type": ActionType.EMAIL_SEND.value,
                "to": "admin@example.com",
                "subject": "Webhook Triggered"
            }
        ]
        
        result = await webhook_connector.create_workflow(
            name="Custom Webhook Flow",
            triggers=triggers,
            actions=actions,
            require_signature=True,
            max_retries=5
        )
        
        assert result["success"] is True
        assert result["platform"] == "custom_webhook"
        assert result["name"] == "Custom Webhook Flow"
        assert "webhook_endpoints" in result
        assert len(result["webhook_endpoints"]) == 1
        
        # Check workflow definition
        definition = result["definition"]
        assert definition["security"]["require_signature"] is True
        assert definition["retry_policy"]["max_retries"] == 5
    
    @pytest.mark.asyncio
    async def test_execute_workflow_with_actions(self, webhook_connector):
        """Test custom webhook workflow execution with multiple actions"""
        
        triggers = [{"type": TriggerType.WEBHOOK.value}]
        actions = [
            {"type": ActionType.HTTP_REQUEST.value, "url": "https://api.test.com"},
            {"type": ActionType.EMAIL_SEND.value, "to": "test@example.com"},
            {"type": ActionType.NOTIFICATION.value, "channel": "#alerts"}
        ]
        
        create_result = await webhook_connector.create_workflow(
            name="Multi-Action Flow",
            triggers=triggers,
            actions=actions
        )
        
        workflow_id = create_result["workflow_id"]
        
        # Mock HTTP session for action execution
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value='{"success": true}')
            
            mock_session.return_value.__aenter__.return_value.request.return_value.__aenter__.return_value = mock_response
            
            input_data = {"webhook_data": "test"}
            result = await webhook_connector.execute_workflow(workflow_id, input_data)
            
            assert result["success"] is True
            
            execution_result = result["execution_result"]
            assert len(execution_result["action_results"]) == 3
            assert execution_result["status"] == "completed"
            
            # Check all actions succeeded
            for action_result in execution_result["action_results"]:
                assert action_result["success"] is True
    
    @pytest.mark.asyncio
    async def test_webhook_signature_validation(self, webhook_connector):
        """Test webhook signature validation"""
        
        secret = "test-secret"
        request_data = {
            "headers": {"X-Webhook-Signature": "sha256=invalid"},
            "body": {"test": "data"}
        }
        
        # Test invalid signature
        is_valid = await webhook_connector._validate_webhook_signature(request_data, secret)
        assert is_valid is False
        
        # Test no signature required
        is_valid = await webhook_connector._validate_webhook_signature(request_data, None)
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_register_webhook(self, webhook_connector):
        """Test custom webhook registration"""
        
        webhook_config = {
            "path": "/enterprise-webhook",
            "method": "POST",
            "secret": "enterprise-secret",
            "require_signature": True,
            "allowed_ips": ["192.168.1.0/24"],
            "rate_limit": {"requests": 1000, "window": 3600}
        }
        
        result = await webhook_connector.register_webhook(webhook_config)
        
        assert result["success"] is True
        assert "webhook_id" in result
        assert result["webhook_url"].endswith("/enterprise-webhook")
        
        webhook_info = result["webhook_info"]
        assert webhook_info["security"]["require_signature"] is True
        assert webhook_info["security"]["allowed_ips"] == ["192.168.1.0/24"]

class TestAutomationConnectorManager:
    """Test automation connector manager functionality"""
    
    @pytest.mark.asyncio
    async def test_create_n8n_connector(self, connector_manager):
        """Test creating N8N connector through manager"""
        
        config = {
            "n8n_url": "https://n8n.example.com",
            "credential_id": "n8n-cred"
        }
        
        result = await connector_manager.create_connector(
            AutomationPlatform.N8N,
            config
        )
        
        assert result["success"] is True
        assert result["platform"] == "n8n"
        assert "connector_id" in result
        assert "connector_info" in result
        
        # Check connector is stored
        connector_id = result["connector_id"]
        connector = await connector_manager.get_connector(connector_id)
        assert connector is not None
        assert connector.platform == AutomationPlatform.N8N
    
    @pytest.mark.asyncio
    async def test_create_zapier_connector(self, connector_manager):
        """Test creating Zapier connector through manager"""
        
        config = {
            "zapier_webhook_url": "https://hooks.zapier.com/test",
            "credential_id": "zapier-cred"
        }
        
        result = await connector_manager.create_connector(
            AutomationPlatform.ZAPIER,
            config
        )
        
        assert result["success"] is True
        assert result["platform"] == "zapier"
    
    @pytest.mark.asyncio
    async def test_create_custom_webhook_connector(self, connector_manager):
        """Test creating custom webhook connector through manager"""
        
        config = {
            "webhook_base_url": "https://api.forge1.com/webhooks"
        }
        
        result = await connector_manager.create_connector(
            AutomationPlatform.CUSTOM_WEBHOOK,
            config
        )
        
        assert result["success"] is True
        assert result["platform"] == "custom_webhook"
    
    @pytest.mark.asyncio
    async def test_list_connectors(self, connector_manager):
        """Test listing all connectors"""
        
        # Create multiple connectors
        await connector_manager.create_connector(
            AutomationPlatform.N8N,
            {"n8n_url": "https://n8n.example.com"}
        )
        
        await connector_manager.create_connector(
            AutomationPlatform.ZAPIER,
            {"zapier_webhook_url": "https://hooks.zapier.com/test"}
        )
        
        result = await connector_manager.list_connectors()
        
        assert "connectors" in result
        assert result["total_count"] == 2
        assert len(result["connectors"]) == 2
        
        # Check connector platforms
        platforms = [conn["platform"] for conn in result["connectors"]]
        assert "n8n" in platforms
        assert "zapier" in platforms
    
    @pytest.mark.asyncio
    async def test_remove_connector(self, connector_manager):
        """Test removing connector"""
        
        # Create a connector
        create_result = await connector_manager.create_connector(
            AutomationPlatform.N8N,
            {"n8n_url": "https://n8n.example.com"}
        )
        
        connector_id = create_result["connector_id"]
        
        # Remove the connector
        result = await connector_manager.remove_connector(connector_id)
        
        assert result["success"] is True
        assert result["connector_id"] == connector_id
        
        # Check connector is removed
        connector = await connector_manager.get_connector(connector_id)
        assert connector is None
    
    @pytest.mark.asyncio
    async def test_execute_workflow_through_manager(self, connector_manager):
        """Test executing workflow through manager"""
        
        # Create connector and workflow
        create_result = await connector_manager.create_connector(
            AutomationPlatform.CUSTOM_WEBHOOK,
            {"webhook_base_url": "https://api.forge1.com/webhooks"}
        )
        
        connector_id = create_result["connector_id"]
        connector = await connector_manager.get_connector(connector_id)
        
        # Create workflow
        workflow_result = await connector.create_workflow(
            name="Test Workflow",
            triggers=[{"type": TriggerType.WEBHOOK.value}],
            actions=[{"type": ActionType.HTTP_REQUEST.value}]
        )
        
        workflow_id = workflow_result["workflow_id"]
        
        # Execute through manager
        result = await connector_manager.execute_workflow(
            connector_id,
            workflow_id,
            {"test": "data"}
        )
        
        assert result["success"] is True
        assert "execution_result" in result
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, connector_manager):
        """Test getting aggregated metrics"""
        
        # Create connectors and execute some workflows
        n8n_result = await connector_manager.create_connector(
            AutomationPlatform.N8N,
            {"n8n_url": "https://n8n.example.com"}
        )
        
        zapier_result = await connector_manager.create_connector(
            AutomationPlatform.ZAPIER,
            {"zapier_webhook_url": "https://hooks.zapier.com/test"}
        )
        
        # Create and execute workflows to generate metrics
        n8n_connector = await connector_manager.get_connector(n8n_result["connector_id"])
        workflow_result = await n8n_connector.create_workflow(
            name="Test",
            triggers=[{"type": TriggerType.WEBHOOK.value}],
            actions=[{"type": ActionType.HTTP_REQUEST.value}]
        )
        
        await n8n_connector.execute_workflow(workflow_result["workflow_id"])
        
        # Get metrics
        metrics = await connector_manager.get_metrics()
        
        assert metrics["total_connectors"] == 2
        assert "n8n" in metrics["connectors_by_platform"]
        assert "zapier" in metrics["connectors_by_platform"]
        assert metrics["total_workflows"] >= 1
        assert metrics["total_executions"] >= 1
    
    @pytest.mark.asyncio
    async def test_unsupported_platform(self, connector_manager):
        """Test creating connector for unsupported platform"""
        
        # Mock an unsupported platform
        with patch('forge1.integrations.automation_connectors.AutomationPlatform') as mock_platform:
            mock_platform.UNSUPPORTED = "unsupported"
            
            result = await connector_manager.create_connector(
                mock_platform.UNSUPPORTED,
                {}
            )
            
            assert result["success"] is False
            assert "Unsupported platform" in result["error"]

if __name__ == "__main__":
    pytest.main([__file__])