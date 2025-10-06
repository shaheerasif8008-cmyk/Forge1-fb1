"""
Customer Experience (CX) Connectors
Integrations for CRM, ticketing systems, and automation platforms
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio
import aiohttp
from datetime import datetime

from forge1.backend.forge1.core.security import SecretManager
from forge1.backend.forge1.core.monitoring import MetricsCollector


@dataclass
class CXTicket:
    """Customer support ticket data model"""
    id: str
    customer_id: str
    subject: str
    description: str
    priority: str
    status: str
    category: str
    created_at: datetime
    updated_at: datetime
    assigned_agent: Optional[str] = None
    resolution: Optional[str] = None
    satisfaction_score: Optional[float] = None


@dataclass
class CustomerProfile:
    """Customer profile data model"""
    id: str
    name: str
    email: str
    phone: Optional[str]
    company: Optional[str]
    tier: str  # bronze, silver, gold, platinum
    lifetime_value: float
    support_history: List[str]
    preferences: Dict[str, Any]


class CXConnector(ABC):
    """Base class for CX system connectors"""
    
    def __init__(self, secret_manager: SecretManager, metrics: MetricsCollector):
        self.secret_manager = secret_manager
        self.metrics = metrics
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the external system"""
        pass
    
    @abstractmethod
    async def get_ticket(self, ticket_id: str) -> Optional[CXTicket]:
        """Retrieve a specific ticket"""
        pass
    
    @abstractmethod
    async def create_ticket(self, ticket: CXTicket) -> str:
        """Create a new ticket and return its ID"""
        pass
    
    @abstractmethod
    async def update_ticket(self, ticket_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing ticket"""
        pass


class SalesforceConnector(CXConnector):
    """Salesforce CRM connector for customer data and case management"""
    
    def __init__(self, secret_manager: SecretManager, metrics: MetricsCollector):
        super().__init__(secret_manager, metrics)
        self.base_url = "https://api.salesforce.com"
        self.access_token = None
    
    async def authenticate(self) -> bool:
        """Authenticate using OAuth2 client credentials flow"""
        try:
            credentials = await self.secret_manager.get_secret("salesforce_oauth")
            
            auth_data = {
                "grant_type": "client_credentials",
                "client_id": credentials["client_id"],
                "client_secret": credentials["client_secret"]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/services/oauth2/token",
                    data=auth_data
                ) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        self.access_token = token_data["access_token"]
                        self.metrics.increment("salesforce_auth_success")
                        return True
                    else:
                        self.metrics.increment("salesforce_auth_failure")
                        return False
        except Exception as e:
            self.metrics.increment("salesforce_auth_error")
            return False
    
    async def get_ticket(self, ticket_id: str) -> Optional[CXTicket]:
        """Retrieve a Salesforce case as a CX ticket"""
        if not self.access_token:
            await self.authenticate()
        
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/services/data/v58.0/sobjects/Case/{ticket_id}",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        case_data = await response.json()
                        return CXTicket(
                            id=case_data["Id"],
                            customer_id=case_data["ContactId"],
                            subject=case_data["Subject"],
                            description=case_data["Description"] or "",
                            priority=case_data["Priority"],
                            status=case_data["Status"],
                            category=case_data["Type"] or "General",
                            created_at=datetime.fromisoformat(case_data["CreatedDate"].replace('Z', '+00:00')),
                            updated_at=datetime.fromisoformat(case_data["LastModifiedDate"].replace('Z', '+00:00'))
                        )
        except Exception as e:
            self.metrics.increment("salesforce_get_ticket_error")
            return None
    
    async def create_ticket(self, ticket: CXTicket) -> str:
        """Create a new Salesforce case"""
        if not self.access_token:
            await self.authenticate()
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        case_data = {
            "Subject": ticket.subject,
            "Description": ticket.description,
            "Priority": ticket.priority,
            "Status": ticket.status,
            "Type": ticket.category,
            "ContactId": ticket.customer_id
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/services/data/v58.0/sobjects/Case",
                    headers=headers,
                    json=case_data
                ) as response:
                    if response.status == 201:
                        result = await response.json()
                        self.metrics.increment("salesforce_create_ticket_success")
                        return result["id"]
        except Exception as e:
            self.metrics.increment("salesforce_create_ticket_error")
            return ""
    
    async def update_ticket(self, ticket_id: str, updates: Dict[str, Any]) -> bool:
        """Update a Salesforce case"""
        if not self.access_token:
            await self.authenticate()
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.patch(
                    f"{self.base_url}/services/data/v58.0/sobjects/Case/{ticket_id}",
                    headers=headers,
                    json=updates
                ) as response:
                    success = response.status == 204
                    if success:
                        self.metrics.increment("salesforce_update_ticket_success")
                    else:
                        self.metrics.increment("salesforce_update_ticket_failure")
                    return success
        except Exception as e:
            self.metrics.increment("salesforce_update_ticket_error")
            return False


class ZendeskConnector(CXConnector):
    """Zendesk ticketing system connector"""
    
    def __init__(self, secret_manager: SecretManager, metrics: MetricsCollector):
        super().__init__(secret_manager, metrics)
        self.base_url = None
        self.api_token = None
        self.email = None
    
    async def authenticate(self) -> bool:
        """Authenticate using API token"""
        try:
            credentials = await self.secret_manager.get_secret("zendesk_api")
            self.base_url = f"https://{credentials['subdomain']}.zendesk.com"
            self.api_token = credentials["api_token"]
            self.email = credentials["email"]
            self.metrics.increment("zendesk_auth_success")
            return True
        except Exception as e:
            self.metrics.increment("zendesk_auth_error")
            return False
    
    async def get_ticket(self, ticket_id: str) -> Optional[CXTicket]:
        """Retrieve a Zendesk ticket"""
        if not self.api_token:
            await self.authenticate()
        
        auth = aiohttp.BasicAuth(f"{self.email}/token", self.api_token)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/v2/tickets/{ticket_id}.json",
                    auth=auth
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        ticket_data = data["ticket"]
                        return CXTicket(
                            id=str(ticket_data["id"]),
                            customer_id=str(ticket_data["requester_id"]),
                            subject=ticket_data["subject"],
                            description=ticket_data["description"],
                            priority=ticket_data["priority"],
                            status=ticket_data["status"],
                            category=ticket_data.get("type", "question"),
                            created_at=datetime.fromisoformat(ticket_data["created_at"].replace('Z', '+00:00')),
                            updated_at=datetime.fromisoformat(ticket_data["updated_at"].replace('Z', '+00:00'))
                        )
        except Exception as e:
            self.metrics.increment("zendesk_get_ticket_error")
            return None
    
    async def create_ticket(self, ticket: CXTicket) -> str:
        """Create a new Zendesk ticket"""
        if not self.api_token:
            await self.authenticate()
        
        auth = aiohttp.BasicAuth(f"{self.email}/token", self.api_token)
        
        ticket_data = {
            "ticket": {
                "subject": ticket.subject,
                "comment": {"body": ticket.description},
                "priority": ticket.priority,
                "status": ticket.status,
                "type": ticket.category,
                "requester_id": int(ticket.customer_id) if ticket.customer_id.isdigit() else None
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/v2/tickets.json",
                    auth=auth,
                    json=ticket_data
                ) as response:
                    if response.status == 201:
                        result = await response.json()
                        self.metrics.increment("zendesk_create_ticket_success")
                        return str(result["ticket"]["id"])
        except Exception as e:
            self.metrics.increment("zendesk_create_ticket_error")
            return ""
    
    async def update_ticket(self, ticket_id: str, updates: Dict[str, Any]) -> bool:
        """Update a Zendesk ticket"""
        if not self.api_token:
            await self.authenticate()
        
        auth = aiohttp.BasicAuth(f"{self.email}/token", self.api_token)
        
        update_data = {"ticket": updates}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    f"{self.base_url}/api/v2/tickets/{ticket_id}.json",
                    auth=auth,
                    json=update_data
                ) as response:
                    success = response.status == 200
                    if success:
                        self.metrics.increment("zendesk_update_ticket_success")
                    else:
                        self.metrics.increment("zendesk_update_ticket_failure")
                    return success
        except Exception as e:
            self.metrics.increment("zendesk_update_ticket_error")
            return False


class N8nConnector:
    """n8n automation platform connector for CX workflows"""
    
    def __init__(self, secret_manager: SecretManager, metrics: MetricsCollector):
        self.secret_manager = secret_manager
        self.metrics = metrics
        self.base_url = None
        self.api_key = None
    
    async def authenticate(self) -> bool:
        """Authenticate with n8n instance"""
        try:
            credentials = await self.secret_manager.get_secret("n8n_api")
            self.base_url = credentials["base_url"]
            self.api_key = credentials["api_key"]
            return True
        except Exception as e:
            return False
    
    async def trigger_workflow(self, workflow_id: str, data: Dict[str, Any]) -> bool:
        """Trigger an n8n workflow with customer data"""
        if not self.api_key:
            await self.authenticate()
        
        headers = {
            "X-N8N-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/v1/workflows/{workflow_id}/execute",
                    headers=headers,
                    json=data
                ) as response:
                    success = response.status == 200
                    if success:
                        self.metrics.increment("n8n_workflow_trigger_success")
                    else:
                        self.metrics.increment("n8n_workflow_trigger_failure")
                    return success
        except Exception as e:
            self.metrics.increment("n8n_workflow_trigger_error")
            return False


class CXConnectorFactory:
    """Factory for creating CX connectors"""
    
    def __init__(self, secret_manager: SecretManager, metrics: MetricsCollector):
        self.secret_manager = secret_manager
        self.metrics = metrics
    
    def create_connector(self, connector_type: str) -> Optional[CXConnector]:
        """Create a connector instance based on type"""
        connectors = {
            "salesforce": SalesforceConnector,
            "zendesk": ZendeskConnector
        }
        
        connector_class = connectors.get(connector_type.lower())
        if connector_class:
            return connector_class(self.secret_manager, self.metrics)
        return None
    
    def create_n8n_connector(self) -> N8nConnector:
        """Create an n8n connector instance"""
        return N8nConnector(self.secret_manager, self.metrics)