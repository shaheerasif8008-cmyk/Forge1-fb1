"""
Customer Experience (CX) AI Employee Tests
Comprehensive test suite for CX vertical
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from forge1.backend.forge1.verticals.cx.employee import CXAIEmployee, CXPerformanceMetrics
from forge1.backend.forge1.verticals.cx.connectors import CXTicket, CustomerProfile, SalesforceConnector, ZendeskConnector
from forge1.backend.forge1.verticals.cx.playbooks import CXPlaybooks, TriageResult, TicketPriority, TicketCategory


class TestCXConnectors:
    """Test CX connector implementations"""
    
    @pytest.fixture
    def mock_secret_manager(self):
        secret_manager = Mock()
        secret_manager.get_secret = AsyncMock(return_value={
            "client_id": "test_client",
            "client_secret": "test_secret"
        })
        return secret_manager
    
    @pytest.fixture
    def mock_metrics(self):
        metrics = Mock()
        metrics.increment = Mock()
        metrics.record_metric = Mock()
        return metrics
    
    @pytest.mark.asyncio
    async def test_salesforce_authentication(self, mock_secret_manager, mock_metrics):
        """Test Salesforce connector authentication"""
        connector = SalesforceConnector(mock_secret_manager, mock_metrics)
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"access_token": "test_token"})
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await connector.authenticate()
            
            assert result is True
            assert connector.access_token == "test_token"
            mock_metrics.increment.assert_called_with("salesforce_auth_success")
    
    @pytest.mark.asyncio
    async def test_zendesk_ticket_retrieval(self, mock_secret_manager, mock_metrics):
        """Test Zendesk ticket retrieval"""
        mock_secret_manager.get_secret = AsyncMock(return_value={
            "subdomain": "test",
            "api_token": "test_token",
            "email": "test@example.com"
        })
        
        connector = ZendeskConnector(mock_secret_manager, mock_metrics)
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "ticket": {
                    "id": 123,
                    "requester_id": 456,
                    "subject": "Test Issue",
                    "description": "Test description",
                    "priority": "high",
                    "status": "open",
                    "type": "problem",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z"
                }
            })
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            ticket = await connector.get_ticket("123")
            
            assert ticket is not None
            assert ticket.id == "123"
            assert ticket.subject == "Test Issue"
            assert ticket.priority == "high"


class TestCXPlaybooks:
    """Test CX playbook implementations"""
    
    @pytest.fixture
    def mock_dependencies(self):
        workflow_engine = Mock()
        memory_manager = Mock()
        model_router = Mock()
        connector_factory = Mock()
        
        return workflow_engine, memory_manager, model_router, connector_factory
    
    @pytest.fixture
    def cx_playbooks(self, mock_dependencies):
        workflow_engine, memory_manager, model_router, connector_factory = mock_dependencies
        return CXPlaybooks(workflow_engine, memory_manager, model_router, connector_factory)
    
    @pytest.fixture
    def sample_ticket(self):
        return CXTicket(
            id="TICKET-123",
            customer_id="CUST-456",
            subject="Login issues",
            description="Cannot log into my account",
            priority="normal",
            status="new",
            category="technical",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    @pytest.fixture
    def sample_customer(self):
        return CustomerProfile(
            id="CUST-456",
            name="John Doe",
            email="john@example.com",
            tier="gold",
            lifetime_value=50000.0,
            support_history=[],
            preferences={}
        )
    
    @pytest.mark.asyncio
    async def test_ticket_triage(self, cx_playbooks, sample_ticket, sample_customer):
        """Test intelligent ticket triage"""
        # Mock AI model response
        cx_playbooks.model_router.generate_response = AsyncMock(
            return_value="Priority: high, Category: technical, Sentiment: negative, Urgency: 0.8"
        )
        
        triage_result = await cx_playbooks.triage_ticket(sample_ticket, sample_customer)
        
        assert isinstance(triage_result, TriageResult)
        assert triage_result.priority in [p for p in TicketPriority]
        assert triage_result.category in [c for c in TicketCategory]
        assert triage_result.sentiment in ["positive", "neutral", "negative"]
        assert 0.0 <= triage_result.urgency_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_ticket_resolution(self, cx_playbooks, sample_ticket, sample_customer):
        """Test ticket resolution process"""
        # Mock dependencies
        cx_playbooks.memory_manager.retrieve_context = AsyncMock(return_value=[
            {"content": "Password reset instructions: Go to login page and click 'Forgot Password'"}
        ])
        
        cx_playbooks.model_router.generate_response = AsyncMock(
            return_value="To resolve your login issue, please try resetting your password by clicking 'Forgot Password' on the login page."
        )
        
        triage_result = TriageResult(
            priority=TicketPriority.HIGH,
            category=TicketCategory.TECHNICAL,
            sentiment="negative",
            urgency_score=0.8,
            escalation_required=False,
            suggested_response="We'll help you with your login issue.",
            confidence=0.9
        )
        
        resolution_result = await cx_playbooks.resolve_ticket(sample_ticket, sample_customer, triage_result)
        
        assert resolution_result.resolution_text is not None
        assert len(resolution_result.resolution_text) > 0
        assert isinstance(resolution_result.resolved, bool)
        assert 0.0 <= resolution_result.customer_satisfaction_predicted <= 1.0
    
    @pytest.mark.asyncio
    async def test_escalation_handling(self, cx_playbooks, sample_ticket, sample_customer):
        """Test ticket escalation to human agents"""
        # Mock n8n connector
        mock_n8n = Mock()
        mock_n8n.trigger_workflow = AsyncMock(return_value=True)
        cx_playbooks.connector_factory.create_n8n_connector.return_value = mock_n8n
        
        # Mock ticket connector
        mock_connector = Mock()
        mock_connector.update_ticket = AsyncMock(return_value=True)
        cx_playbooks.connector_factory.create_connector.return_value = mock_connector
        
        # Mock AI response for escalation summary
        cx_playbooks.model_router.generate_response = AsyncMock(
            return_value="Customer experiencing login issues. Requires immediate attention due to gold tier status."
        )
        
        result = await cx_playbooks.handle_escalation(sample_ticket, sample_customer, "Complex technical issue")
        
        assert result is True
        mock_n8n.trigger_workflow.assert_called_once()
        mock_connector.update_ticket.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_performance_measurement(self, cx_playbooks):
        """Test performance metrics calculation"""
        # Mock performance data
        cx_playbooks._query_performance_metrics = AsyncMock(return_value={
            "deflection_rate": 0.96,
            "first_response_time": 3.2,
            "resolution_time": 1800,
            "csat_score": 4.2,
            "escalation_rate": 0.04,
            "sla_compliance": 0.98
        })
        
        metrics = await cx_playbooks.measure_performance(timedelta(hours=24))
        
        assert metrics["deflection_rate"] == 0.96
        assert metrics["first_response_time_avg"] == 3.2
        assert metrics["customer_satisfaction"] == 4.2
        assert metrics["sla_compliance"] == 0.98


class TestCXAIEmployee:
    """Test CX AI Employee implementation"""
    
    @pytest.fixture
    def mock_dependencies(self):
        workflow_engine = Mock()
        memory_manager = Mock()
        model_router = Mock()
        metrics_collector = Mock()
        secret_manager = Mock()
        
        return workflow_engine, memory_manager, model_router, metrics_collector, secret_manager
    
    @pytest.fixture
    def cx_employee(self, mock_dependencies):
        workflow_engine, memory_manager, model_router, metrics_collector, secret_manager = mock_dependencies
        
        employee = CXAIEmployee(
            employee_id="CX-001",
            tenant_id="TENANT-123",
            workflow_engine=workflow_engine,
            memory_manager=memory_manager,
            model_router=model_router,
            metrics_collector=metrics_collector,
            secret_manager=secret_manager
        )
        
        return employee
    
    @pytest.fixture
    def sample_ticket_data(self):
        return {
            "id": "TICKET-789",
            "customer_id": "CUST-123",
            "subject": "Billing question",
            "description": "I was charged twice for my subscription",
            "priority": "high",
            "status": "new",
            "category": "billing",
            "created_at": datetime.utcnow().isoformat()
        }
    
    @pytest.mark.asyncio
    async def test_ticket_handling_success(self, cx_employee, sample_ticket_data):
        """Test successful ticket handling end-to-end"""
        # Mock playbook responses
        cx_employee.playbooks.triage_ticket = AsyncMock(return_value=TriageResult(
            priority=TicketPriority.HIGH,
            category=TicketCategory.BILLING,
            sentiment="negative",
            urgency_score=0.7,
            escalation_required=False,
            suggested_response="I'll help you with your billing issue.",
            confidence=0.9
        ))
        
        cx_employee.playbooks.resolve_ticket = AsyncMock(return_value=Mock(
            resolved=True,
            resolution_text="I've reviewed your account and processed a refund for the duplicate charge.",
            follow_up_required=True,
            escalation_needed=False,
            customer_satisfaction_predicted=0.85
        ))
        
        # Mock customer profile retrieval
        cx_employee._get_customer_profile = AsyncMock(return_value=CustomerProfile(
            id="CUST-123",
            name="Jane Smith",
            email="jane@example.com",
            tier="platinum",
            lifetime_value=100000.0,
            support_history=[],
            preferences={}
        ))
        
        # Mock other methods
        cx_employee._update_external_ticket = AsyncMock(return_value=True)
        cx_employee._get_interaction_history = AsyncMock(return_value=[])
        cx_employee._store_interaction_memory = AsyncMock()
        
        result = await cx_employee.handle_ticket(sample_ticket_data)
        
        assert result["status"] == "resolved"
        assert result["ticket_id"] == "TICKET-789"
        assert result["sla_met"] is True  # Should be under 5 seconds
        assert "response" in result
        assert result["priority"] == "high"
        assert result["category"] == "billing"
    
    @pytest.mark.asyncio
    async def test_performance_metrics_retrieval(self, cx_employee):
        """Test performance metrics retrieval"""
        # Mock playbook performance measurement
        cx_employee.playbooks.measure_performance = AsyncMock(return_value={
            "deflection_rate": 0.97,
            "first_response_time_avg": 2.8,
            "resolution_time_avg": 1200,
            "customer_satisfaction": 4.3,
            "escalation_rate": 0.03,
            "sla_compliance": 0.99
        })
        
        metrics = await cx_employee.get_performance_metrics()
        
        assert isinstance(metrics, CXPerformanceMetrics)
        assert metrics.deflection_rate == 0.97
        assert metrics.first_response_time == 2.8
        assert metrics.customer_satisfaction == 4.3
        assert metrics.sla_compliance == 0.99
    
    @pytest.mark.asyncio
    async def test_health_check(self, cx_employee):
        """Test health check functionality"""
        # Mock connector authentication
        mock_sf_connector = Mock()
        mock_sf_connector.authenticate = AsyncMock(return_value=True)
        
        mock_zd_connector = Mock()
        mock_zd_connector.authenticate = AsyncMock(return_value=True)
        
        mock_n8n_connector = Mock()
        mock_n8n_connector.authenticate = AsyncMock(return_value=True)
        
        cx_employee.connector_factory.create_connector = Mock(side_effect=[mock_sf_connector, mock_zd_connector])
        cx_employee.connector_factory.create_n8n_connector = Mock(return_value=mock_n8n_connector)
        
        health = await cx_employee.health_check()
        
        assert health["status"] == "healthy"
        assert health["employee_id"] == "CX-001"
        assert "uptime_seconds" in health
        assert health["salesforce_connected"] is True
        assert health["zendesk_connected"] is True
        assert health["n8n_connected"] is True
    
    @pytest.mark.asyncio
    async def test_sla_compliance_validation(self, cx_employee):
        """Test SLA compliance validation"""
        # Mock performance metrics
        cx_employee.get_performance_metrics = AsyncMock(return_value=CXPerformanceMetrics(
            deflection_rate=0.96,
            first_response_time=3.5,
            resolution_time=1500,
            customer_satisfaction=4.1,
            escalation_rate=0.04,
            sla_compliance=0.99,
            tickets_handled=150,
            uptime=0.999
        ))
        
        compliance = await cx_employee.validate_sla_compliance()
        
        assert compliance["first_response_time"] is True  # 3.5s < 5s target
        assert compliance["deflection_rate"] is True  # 0.96 > 0.95 target
        assert compliance["customer_satisfaction"] is True  # 4.1 > 4.0 target
        assert compliance["escalation_rate"] is True  # 0.04 < 0.05 target
        assert compliance["overall_compliance"] is True  # 0.99 >= 0.99 target
    
    @pytest.mark.asyncio
    async def test_error_handling(self, cx_employee, sample_ticket_data):
        """Test error handling in ticket processing"""
        # Force an exception in triage
        cx_employee.playbooks.triage_ticket = AsyncMock(side_effect=Exception("Triage failed"))
        
        result = await cx_employee.handle_ticket(sample_ticket_data)
        
        assert result["status"] == "error"
        assert "error" in result
        assert result["escalated"] is True


class TestCXIntegration:
    """Integration tests for CX vertical"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_ticket_flow(self):
        """Test complete ticket processing flow"""
        # This would test the entire flow from ticket creation to resolution
        # Including external system integrations (mocked)
        pass
    
    @pytest.mark.asyncio
    async def test_sla_performance_under_load(self):
        """Test SLA compliance under high ticket volume"""
        # This would simulate high load and verify SLA targets are met
        pass
    
    @pytest.mark.asyncio
    async def test_escalation_workflow(self):
        """Test complete escalation workflow to human agents"""
        # This would test the full escalation process including notifications
        pass


if __name__ == "__main__":
    pytest.main([__file__])