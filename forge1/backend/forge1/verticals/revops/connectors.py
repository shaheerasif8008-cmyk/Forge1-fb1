"""
Revenue Operations (RevOps) Connectors
Integrations for CRM, CPQ, billing, and BI systems
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime, date
from decimal import Decimal
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta

from forge1.backend.forge1.core.security import SecretManager
from forge1.backend.forge1.core.monitoring import MetricsCollector


@dataclass
class Deal:
    """Sales deal/opportunity data model"""
    id: str
    name: str
    account_id: str
    owner_id: str
    stage: str
    amount: Decimal
    probability: float
    close_date: date
    created_date: datetime
    last_activity: Optional[datetime]
    source: str
    product_line: Optional[str]
    forecast_category: str


@dataclass
class Account:
    """Customer account data model"""
    id: str
    name: str
    type: str  # prospect, customer, partner
    industry: str
    size: str  # SMB, Mid-Market, Enterprise
    annual_revenue: Optional[Decimal]
    employee_count: Optional[int]
    owner_id: str
    health_score: float
    renewal_date: Optional[date]
    expansion_potential: float


@dataclass
class Quote:
    """Sales quote data model"""
    id: str
    deal_id: str
    account_id: str
    total_amount: Decimal
    discount_percent: float
    status: str  # draft, sent, approved, rejected
    valid_until: date
    created_date: datetime
    line_items: List[Dict[str, Any]]


@dataclass
class ForecastData:
    """Sales forecast data model"""
    period: str  # Q1 2024, Jan 2024, etc.
    rep_id: str
    pipeline_amount: Decimal
    weighted_amount: Decimal
    commit_amount: Decimal
    best_case_amount: Decimal
    closed_won_amount: Decimal
    quota: Decimal
    attainment: float


class RevOpsConnector(ABC):
    """Base class for RevOps system connectors"""
    
    def __init__(self, secret_manager: SecretManager, metrics: MetricsCollector):
        self.secret_manager = secret_manager
        self.metrics = metrics
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the external system"""
        pass
    
    @abstractmethod
    async def get_deals(self, filters: Dict[str, Any] = None) -> List[Deal]:
        """Retrieve deals/opportunities"""
        pass
    
    @abstractmethod
    async def update_deal(self, deal_id: str, updates: Dict[str, Any]) -> bool:
        """Update deal information"""
        pass
    
    @abstractmethod
    async def get_accounts(self, filters: Dict[str, Any] = None) -> List[Account]:
        """Retrieve account information"""
        pass


class SalesforceCRMConnector(RevOpsConnector):
    """Salesforce CRM connector for deals and accounts"""
    
    def __init__(self, secret_manager: SecretManager, metrics: MetricsCollector):
        super().__init__(secret_manager, metrics)
        self.base_url = "https://api.salesforce.com"
        self.access_token = None
    
    async def authenticate(self) -> bool:
        """Authenticate using OAuth2 with comprehensive error handling and token refresh"""
        try:
            credentials = await self.secret_manager.get_secret("salesforce_oauth")
            
            # Support both client credentials and refresh token flows
            if "refresh_token" in credentials:
                auth_data = {
                    "grant_type": "refresh_token",
                    "refresh_token": credentials["refresh_token"],
                    "client_id": credentials["client_id"],
                    "client_secret": credentials["client_secret"]
                }
            else:
                auth_data = {
                    "grant_type": "client_credentials",
                    "client_id": credentials["client_id"],
                    "client_secret": credentials["client_secret"]
                }
            
            # Add retry logic with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    timeout = aiohttp.ClientTimeout(total=30)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(
                            f"{self.base_url}/services/oauth2/token",
                            data=auth_data,
                            headers={"Content-Type": "application/x-www-form-urlencoded"}
                        ) as response:
                            if response.status == 200:
                                token_data = await response.json()
                                self.access_token = token_data["access_token"]
                                
                                # Store refresh token if provided
                                if "refresh_token" in token_data:
                                    await self._store_refresh_token(token_data["refresh_token"])
                                
                                # Set token expiration
                                if "expires_in" in token_data:
                                    self.token_expires_at = datetime.utcnow() + timedelta(seconds=token_data["expires_in"])
                                
                                self.metrics.increment("salesforce_crm_auth_success")
                                return True
                            elif response.status == 401:
                                error_data = await response.json()
                                if error_data.get("error") == "invalid_grant" and attempt < max_retries - 1:
                                    # Try client credentials flow if refresh token failed
                                    auth_data = {
                                        "grant_type": "client_credentials",
                                        "client_id": credentials["client_id"],
                                        "client_secret": credentials["client_secret"]
                                    }
                                    continue
                                else:
                                    self.metrics.increment("salesforce_crm_auth_unauthorized")
                                    return False
                            else:
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                    continue
                                else:
                                    self.metrics.increment("salesforce_crm_auth_failure")
                                    return False
                except aiohttp.ClientError as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        raise e
                        
        except Exception as e:
            self.metrics.increment("salesforce_crm_auth_error")
            logging.error(f"Salesforce authentication error: {e}")
            return False
        
        return False
    
    async def _store_refresh_token(self, refresh_token: str) -> None:
        """Store refresh token securely"""
        try:
            # In production, this would update the stored credentials
            # For now, we'll log that we received a refresh token
            logging.info("Received new refresh token from Salesforce")
        except Exception as e:
            logging.error(f"Error storing refresh token: {e}")
    
    async def _ensure_authenticated(self) -> bool:
        """Ensure we have a valid access token"""
        if not self.access_token:
            return await self.authenticate()
        
        # Check if token is expired (if we have expiration info)
        if hasattr(self, 'token_expires_at') and self.token_expires_at:
            if datetime.utcnow() >= self.token_expires_at - timedelta(minutes=5):  # Refresh 5 min early
                return await self.authenticate()
        
        return True
    
    async def get_deals(self, filters: Dict[str, Any] = None) -> List[Deal]:
        """Retrieve Salesforce opportunities as deals"""
        if not self.access_token:
            await self.authenticate()
        
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        # Build SOQL query
        query = """
        SELECT Id, Name, AccountId, OwnerId, StageName, Amount, Probability, 
               CloseDate, CreatedDate, LastActivityDate, LeadSource, 
               Product_Line__c, ForecastCategoryName
        FROM Opportunity
        WHERE IsDeleted = false
        """
        
        if filters:
            conditions = []
            if filters.get("stage"):
                conditions.append(f"StageName = '{filters['stage']}'")
            if filters.get("owner_id"):
                conditions.append(f"OwnerId = '{filters['owner_id']}'")
            if filters.get("close_date_after"):
                conditions.append(f"CloseDate >= {filters['close_date_after']}")
            
            if conditions:
                query += " AND " + " AND ".join(conditions)
        
        query += " ORDER BY CloseDate ASC LIMIT 1000"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/services/data/v58.0/query",
                    headers=headers,
                    params={"q": query}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        deals = []
                        
                        for record in data.get("records", []):
                            deal = Deal(
                                id=record["Id"],
                                name=record["Name"],
                                account_id=record["AccountId"],
                                owner_id=record["OwnerId"],
                                stage=record["StageName"],
                                amount=Decimal(str(record["Amount"] or 0)),
                                probability=record["Probability"] / 100.0,
                                close_date=datetime.strptime(record["CloseDate"], "%Y-%m-%d").date(),
                                created_date=datetime.fromisoformat(record["CreatedDate"].replace('Z', '+00:00')),
                                last_activity=datetime.fromisoformat(record["LastActivityDate"].replace('Z', '+00:00')) if record["LastActivityDate"] else None,
                                source=record["LeadSource"] or "Unknown",
                                product_line=record.get("Product_Line__c"),
                                forecast_category=record["ForecastCategoryName"]
                            )
                            deals.append(deal)
                        
                        self.metrics.record_metric("salesforce_deals_retrieved", len(deals))
                        return deals
        except Exception as e:
            self.metrics.increment("salesforce_get_deals_error")
            return []
    
    async def update_deal(self, deal_id: str, updates: Dict[str, Any]) -> bool:
        """Update Salesforce opportunity"""
        if not self.access_token:
            await self.authenticate()
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        # Map updates to Salesforce fields
        sf_updates = {}
        if "stage" in updates:
            sf_updates["StageName"] = updates["stage"]
        if "amount" in updates:
            sf_updates["Amount"] = float(updates["amount"])
        if "probability" in updates:
            sf_updates["Probability"] = updates["probability"] * 100
        if "close_date" in updates:
            sf_updates["CloseDate"] = updates["close_date"].isoformat()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.patch(
                    f"{self.base_url}/services/data/v58.0/sobjects/Opportunity/{deal_id}",
                    headers=headers,
                    json=sf_updates
                ) as response:
                    success = response.status == 204
                    if success:
                        self.metrics.increment("salesforce_deal_update_success")
                    else:
                        self.metrics.increment("salesforce_deal_update_failure")
                    return success
        except Exception as e:
            self.metrics.increment("salesforce_deal_update_error")
            return False
    
    async def get_accounts(self, filters: Dict[str, Any] = None) -> List[Account]:
        """Retrieve Salesforce accounts"""
        if not self.access_token:
            await self.authenticate()
        
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        query = """
        SELECT Id, Name, Type, Industry, AnnualRevenue, NumberOfEmployees,
               OwnerId, Health_Score__c, Renewal_Date__c, Expansion_Potential__c
        FROM Account
        WHERE IsDeleted = false
        """
        
        if filters:
            conditions = []
            if filters.get("type"):
                conditions.append(f"Type = '{filters['type']}'")
            if filters.get("owner_id"):
                conditions.append(f"OwnerId = '{filters['owner_id']}'")
            
            if conditions:
                query += " AND " + " AND ".join(conditions)
        
        query += " ORDER BY Name ASC LIMIT 1000"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/services/data/v58.0/query",
                    headers=headers,
                    params={"q": query}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        accounts = []
                        
                        for record in data.get("records", []):
                            # Determine size based on employee count
                            emp_count = record.get("NumberOfEmployees", 0)
                            if emp_count < 100:
                                size = "SMB"
                            elif emp_count < 1000:
                                size = "Mid-Market"
                            else:
                                size = "Enterprise"
                            
                            account = Account(
                                id=record["Id"],
                                name=record["Name"],
                                type=record.get("Type", "Customer"),
                                industry=record.get("Industry", "Unknown"),
                                size=size,
                                annual_revenue=Decimal(str(record["AnnualRevenue"])) if record.get("AnnualRevenue") else None,
                                employee_count=record.get("NumberOfEmployees"),
                                owner_id=record["OwnerId"],
                                health_score=record.get("Health_Score__c", 0.5),
                                renewal_date=datetime.strptime(record["Renewal_Date__c"], "%Y-%m-%d").date() if record.get("Renewal_Date__c") else None,
                                expansion_potential=record.get("Expansion_Potential__c", 0.0)
                            )
                            accounts.append(account)
                        
                        return accounts
        except Exception as e:
            self.metrics.increment("salesforce_get_accounts_error")
            return []


class HubSpotConnector(RevOpsConnector):
    """HubSpot CRM connector"""
    
    def __init__(self, secret_manager: SecretManager, metrics: MetricsCollector):
        super().__init__(secret_manager, metrics)
        self.base_url = "https://api.hubapi.com"
        self.api_key = None
    
    async def authenticate(self) -> bool:
        """Authenticate using API key"""
        try:
            credentials = await self.secret_manager.get_secret("hubspot_api")
            self.api_key = credentials["api_key"]
            return True
        except Exception as e:
            return False
    
    async def get_deals(self, filters: Dict[str, Any] = None) -> List[Deal]:
        """Retrieve HubSpot deals"""
        if not self.api_key:
            await self.authenticate()
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        params = {
            "properties": "dealname,amount,dealstage,probability,closedate,createdate,hs_lastmodifieddate,pipeline,dealtype",
            "limit": 100
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/crm/v3/objects/deals",
                    headers=headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        deals = []
                        
                        for result in data.get("results", []):
                            props = result["properties"]
                            
                            deal = Deal(
                                id=result["id"],
                                name=props.get("dealname", ""),
                                account_id=props.get("associatedcompanyid", ""),
                                owner_id=props.get("hubspot_owner_id", ""),
                                stage=props.get("dealstage", ""),
                                amount=Decimal(str(props.get("amount", 0))),
                                probability=float(props.get("probability", 0)) / 100.0,
                                close_date=datetime.strptime(props["closedate"], "%Y-%m-%d").date() if props.get("closedate") else date.today(),
                                created_date=datetime.fromisoformat(props["createdate"].replace('Z', '+00:00')),
                                last_activity=datetime.fromisoformat(props["hs_lastmodifieddate"].replace('Z', '+00:00')) if props.get("hs_lastmodifieddate") else None,
                                source=props.get("hs_analytics_source", "Unknown"),
                                product_line=props.get("dealtype"),
                                forecast_category=self._map_stage_to_forecast(props.get("dealstage", ""))
                            )
                            deals.append(deal)
                        
                        return deals
        except Exception as e:
            self.metrics.increment("hubspot_get_deals_error")
            return []
    
    async def update_deal(self, deal_id: str, updates: Dict[str, Any]) -> bool:
        """Update HubSpot deal"""
        if not self.api_key:
            await self.authenticate()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Map updates to HubSpot properties
        hs_updates = {"properties": {}}
        if "stage" in updates:
            hs_updates["properties"]["dealstage"] = updates["stage"]
        if "amount" in updates:
            hs_updates["properties"]["amount"] = str(updates["amount"])
        if "probability" in updates:
            hs_updates["properties"]["probability"] = str(updates["probability"] * 100)
        if "close_date" in updates:
            hs_updates["properties"]["closedate"] = updates["close_date"].isoformat()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.patch(
                    f"{self.base_url}/crm/v3/objects/deals/{deal_id}",
                    headers=headers,
                    json=hs_updates
                ) as response:
                    success = response.status == 200
                    if success:
                        self.metrics.increment("hubspot_deal_update_success")
                    else:
                        self.metrics.increment("hubspot_deal_update_failure")
                    return success
        except Exception as e:
            self.metrics.increment("hubspot_deal_update_error")
            return False
    
    async def get_accounts(self, filters: Dict[str, Any] = None) -> List[Account]:
        """Retrieve HubSpot companies as accounts"""
        if not self.api_key:
            await self.authenticate()
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        params = {
            "properties": "name,industry,annualrevenue,numberofemployees,hubspot_owner_id,lifecyclestage",
            "limit": 100
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/crm/v3/objects/companies",
                    headers=headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        accounts = []
                        
                        for result in data.get("results", []):
                            props = result["properties"]
                            
                            # Determine size and type
                            emp_count = int(props.get("numberofemployees", 0)) if props.get("numberofemployees") else 0
                            if emp_count < 100:
                                size = "SMB"
                            elif emp_count < 1000:
                                size = "Mid-Market"
                            else:
                                size = "Enterprise"
                            
                            lifecycle_stage = props.get("lifecyclestage", "lead")
                            account_type = "Customer" if lifecycle_stage == "customer" else "Prospect"
                            
                            account = Account(
                                id=result["id"],
                                name=props.get("name", ""),
                                type=account_type,
                                industry=props.get("industry", "Unknown"),
                                size=size,
                                annual_revenue=Decimal(str(props["annualrevenue"])) if props.get("annualrevenue") else None,
                                employee_count=emp_count,
                                owner_id=props.get("hubspot_owner_id", ""),
                                health_score=0.5,  # HubSpot doesn't have default health score
                                renewal_date=None,  # Would need custom property
                                expansion_potential=0.0  # Would need custom property
                            )
                            accounts.append(account)
                        
                        return accounts
        except Exception as e:
            self.metrics.increment("hubspot_get_accounts_error")
            return []
    
    def _map_stage_to_forecast(self, stage: str) -> str:
        """Map HubSpot deal stage to forecast category"""
        stage_mapping = {
            "appointmentscheduled": "Pipeline",
            "qualifiedtobuy": "Best Case",
            "presentationscheduled": "Commit",
            "decisionmakerboughtin": "Commit",
            "contractsent": "Commit",
            "closedwon": "Closed Won",
            "closedlost": "Closed Lost"
        }
        return stage_mapping.get(stage.lower(), "Pipeline")


class PowerBIConnector:
    """Power BI connector for revenue analytics"""
    
    def __init__(self, secret_manager: SecretManager, metrics: MetricsCollector):
        self.secret_manager = secret_manager
        self.metrics = metrics
        self.base_url = "https://api.powerbi.com"
        self.access_token = None
    
    async def authenticate(self) -> bool:
        """Authenticate using Azure AD"""
        try:
            credentials = await self.secret_manager.get_secret("powerbi_oauth")
            
            auth_data = {
                "grant_type": "client_credentials",
                "client_id": credentials["client_id"],
                "client_secret": credentials["client_secret"],
                "scope": "https://analysis.windows.net/powerbi/api/.default"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"https://login.microsoftonline.com/{credentials['tenant_id']}/oauth2/v2.0/token",
                    data=auth_data
                ) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        self.access_token = token_data["access_token"]
                        return True
                    return False
        except Exception as e:
            return False
    
    async def get_revenue_metrics(self, dataset_id: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Retrieve revenue metrics from Power BI dataset"""
        if not self.access_token:
            await self.authenticate()
        
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        # DAX query for revenue metrics
        dax_query = """
        EVALUATE
        SUMMARIZE(
            Sales,
            "Total_Revenue", SUM(Sales[Amount]),
            "Deal_Count", COUNT(Sales[DealID]),
            "Avg_Deal_Size", AVERAGE(Sales[Amount]),
            "Win_Rate", DIVIDE(COUNTROWS(FILTER(Sales, Sales[Stage] = "Closed Won")), COUNTROWS(Sales))
        )
        """
        
        query_data = {
            "queries": [{"query": dax_query}],
            "serializerSettings": {"includeNulls": True}
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1.0/myorg/datasets/{dataset_id}/executeQueries",
                    headers=headers,
                    json=query_data
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Parse Power BI response
                        results = data.get("results", [{}])[0]
                        tables = results.get("tables", [{}])[0]
                        rows = tables.get("rows", [])
                        
                        if rows:
                            row = rows[0]
                            return {
                                "total_revenue": row[0],
                                "deal_count": row[1],
                                "avg_deal_size": row[2],
                                "win_rate": row[3]
                            }
        except Exception as e:
            self.metrics.increment("powerbi_query_error")
            return {}


class RevOpsConnectorFactory:
    """Factory for creating RevOps connectors"""
    
    def __init__(self, secret_manager: SecretManager, metrics: MetricsCollector):
        self.secret_manager = secret_manager
        self.metrics = metrics
    
    def create_crm_connector(self, crm_type: str) -> Optional[RevOpsConnector]:
        """Create a CRM connector instance"""
        connectors = {
            "salesforce": SalesforceCRMConnector,
            "hubspot": HubSpotConnector
        }
        
        connector_class = connectors.get(crm_type.lower())
        if connector_class:
            return connector_class(self.secret_manager, self.metrics)
        return None
    
    def create_bi_connector(self, bi_type: str):
        """Create a BI connector instance"""
        if bi_type.lower() == "powerbi":
            return PowerBIConnector(self.secret_manager, self.metrics)
        return None