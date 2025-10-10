```python
# forge1/backend/forge1/tools/example_tools.py
"""
Example Tools for Forge 1

Demonstration tools showing how to implement the BaseTool interface
for various categories and use cases.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json
import re
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

from .tool_registry import BaseTool, ToolCategory, AuthenticationType

logger = logging.getLogger(__name__)

class EmailTool(BaseTool):
    """Tool for sending emails"""
    
    def __init__(self, smtp_config: Dict[str, Any] = None):
        super().__init__(
            name="email_sender",
            description="Send emails via SMTP with support for HTML and attachments",
            category=ToolCategory.COMMUNICATION,
            version="1.0.0",
            authentication_type=AuthenticationType.BASIC_AUTH,
            smtp_config=smtp_config or {}
        )
    
    async def execute(self, to: str, subject: str, body: str, html: bool = False, **kwargs) -> Dict[str, Any]:
        """Send an email
        
        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body content
            html: Whether body is HTML format
            **kwargs: Additional email parameters
            
        Returns:
            Email sending result
        """
        
        try:
            # Mock email sending for demonstration
            # In production, this would use actual SMTP configuration
            
            email_data = {
                "to": to,
                "subject": subject,
                "body": body,
                "html": html,
                "sent_at": datetime.now(timezone.utc).isoformat(),
                "message_id": f"msg_{hash(f'{to}{subject}{body}')}"
            }
            
            # Simulate email sending delay
            await asyncio.sleep(0.1)
            
            logger.info(f"Email sent to {to} with subject: {subject}")
            
            return {
                "success": True,
                "message_id": email_data["message_id"],
                "recipient": to,
                "sent_at": email_data["sent_at"]
            }
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate email parameters"""
        
        required_params = ["to", "subject", "body"]
        
        for param in required_params:
            if param not in parameters:
                return False
        
        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, parameters["to"]):
            return False
        
        return True
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to": {"type": "string", "format": "email", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body content"},
                "html": {"type": "boolean", "default": False, "description": "Whether body is HTML format"}
            },
            "required": ["to", "subject", "body"]
        }

class WebScrapingTool(BaseTool):
    """Tool for web scraping and data extraction"""
    
    def __init__(self):
        super().__init__(
            name="web_scraper",
            description="Extract data from web pages with support for various formats",
            category=ToolCategory.DATA_ANALYSIS,
            version="1.0.0",
            authentication_type=AuthenticationType.NONE
        )
    
    async def execute(self, url: str, selector: str = None, format: str = "text", **kwargs) -> Dict[str, Any]:
        """Scrape data from a web page
        
        Args:
            url: URL to scrape
            selector: CSS selector for specific elements (optional)
            format: Output format (text, html, json)
            **kwargs: Additional scraping parameters
            
        Returns:
            Scraped data
        """
        
        try:
            # Mock web scraping for demonstration
            # In production, this would use libraries like BeautifulSoup, Scrapy, etc.
            
            scraped_data = {
                "url": url,
                "title": f"Mock Title for {url}",
                "content": f"Mock content scraped from {url}",
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "format": format
            }
            
            if selector:
                scraped_data["selector"] = selector
                scraped_data["selected_content"] = f"Mock selected content using {selector}"
            
            # Simulate scraping delay
            await asyncio.sleep(0.2)
            
            logger.info(f"Successfully scraped data from {url}")
            
            return {
                "success": True,
                "data": scraped_data,
                "url": url,
                "scraped_at": scraped_data["scraped_at"]
            }
            
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate web scraping parameters"""
        
        if "url" not in parameters:
            return False
        
        # Basic URL validation
        url = parameters["url"]
        if not url.startswith(("http://", "https://")):
            return False
        
        # Validate format if provided
        if "format" in parameters:
            valid_formats = ["text", "html", "json"]
            if parameters["format"] not in valid_formats:
                return False
        
        return True
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "format": "uri", "description": "URL to scrape"},
                "selector": {"type": "string", "description": "CSS selector for specific elements"},
                "format": {"type": "string", "enum": ["text", "html", "json"], "default": "text", "description": "Output format"}
            },
            "required": ["url"]
        }

class DataAnalysisTool(BaseTool):
    """Tool for basic data analysis operations"""
    
    def __init__(self):
        super().__init__(
            name="data_analyzer",
            description="Perform basic data analysis operations on datasets",
            category=ToolCategory.DATA_ANALYSIS,
            version="1.0.0",
            authentication_type=AuthenticationType.NONE
        )
    
    async def execute(self, data: List[Dict[str, Any]], operation: str, **kwargs) -> Dict[str, Any]:
        """Perform data analysis operation
        
        Args:
            data: Dataset to analyze
            operation: Analysis operation (summary, filter, aggregate, etc.)
            **kwargs: Additional operation parameters
            
        Returns:
            Analysis results
        """
        
        try:
            if operation == "summary":
                result = await self._generate_summary(data)
            elif operation == "filter":
                result = await self._filter_data(data, kwargs.get("filter_criteria", {}))
            elif operation == "aggregate":
                result = await self._aggregate_data(data, kwargs.get("group_by"), kwargs.get("aggregation"))
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            return {
                "success": True,
                "operation": operation,
                "result": result,
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
                "records_processed": len(data)
            }
            
        except Exception as e:
            logger.error(f"Data analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "operation": operation
            }
    
    async def _generate_summary(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate data summary"""
        
        if not data:
            return {"total_records": 0}
        
        summary = {
            "total_records": len(data),
            "columns": list(data[0].keys()) if data else [],
            "sample_record": data[0] if data else None
        }
        
        # Basic statistics for numeric columns
        numeric_stats = {}
        for column in summary["columns"]:
            values = [record.get(column) for record in data if isinstance(record.get(column), (int, float))]
            if values:
                numeric_stats[column] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values)
                }
        
        summary["numeric_statistics"] = numeric_stats
        return summary
    
    async def _filter_data(self, data: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter data based on criteria"""
        
        filtered_data = []
        
        for record in data:
            match = True
            for key, value in criteria.items():
                if key not in record or record[key] != value:
                    match = False
                    break
            
            if match:
                filtered_data.append(record)
        
        return filtered_data
    
    async def _aggregate_data(self, data: List[Dict[str, Any]], group_by: str, aggregation: str) -> Dict[str, Any]:
        """Aggregate data by grouping"""
        
        if not group_by or not aggregation:
            raise ValueError("group_by and aggregation parameters are required")
        
        groups = {}
        for record in data:
            group_key = record.get(group_by)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(record)
        
        aggregated = {}
        for group_key, group_records in groups.items():
            if aggregation == "count":
                aggregated[group_key] = len(group_records)
            elif aggregation == "sum":
                # Sum numeric values (simplified)
                aggregated[group_key] = len(group_records)  # Placeholder
            else:
                aggregated[group_key] = len(group_records)
        
        return aggregated
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate data analysis parameters"""
        
        required_params = ["data", "operation"]
        
        for param in required_params:
            if param not in parameters:
                return False
        
        # Validate data format
        if not isinstance(parameters["data"], list):
            return False
        
        # Validate operation
        valid_operations = ["summary", "filter", "aggregate"]
        if parameters["operation"] not in valid_operations:
            return False
        
        return True
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {"type": "array", "items": {"type": "object"}, "description": "Dataset to analyze"},
                "operation": {"type": "string", "enum": ["summary", "filter", "aggregate"], "description": "Analysis operation"},
                "filter_criteria": {"type": "object", "description": "Filter criteria for filter operation"},
                "group_by": {"type": "string", "description": "Column to group by for aggregate operation"},
                "aggregation": {"type": "string", "enum": ["count", "sum", "avg"], "description": "Aggregation function"}
            },
            "required": ["data", "operation"]
        }

class SlackTool(BaseTool):
    """Tool for Slack integration"""
    
    def __init__(self, bot_token: str = None):
        super().__init__(
            name="slack_messenger",
            description="Send messages and interact with Slack channels",
            category=ToolCategory.COMMUNICATION,
            version="1.0.0",
            authentication_type=AuthenticationType.BEARER_TOKEN,
            bot_token=bot_token
        )
    
    async def execute(self, channel: str, message: str, thread_ts: str = None, **kwargs) -> Dict[str, Any]:
        """Send message to Slack channel
        
        Args:
            channel: Slack channel ID or name
            message: Message content
            thread_ts: Thread timestamp for replies (optional)
            **kwargs: Additional Slack API parameters
            
        Returns:
            Message sending result
        """
        
        try:
            # Mock Slack API call for demonstration
            # In production, this would use the Slack SDK
            
            message_data = {
                "channel": channel,
                "message": message,
                "thread_ts": thread_ts,
                "sent_at": datetime.now(timezone.utc).isoformat(),
                "message_ts": f"ts_{hash(f'{channel}{message}')}"
            }
            
            # Simulate API call delay
            await asyncio.sleep(0.1)
            
            logger.info(f"Slack message sent to {channel}: {message[:50]}...")
            
            return {
                "success": True,
                "channel": channel,
                "message_ts": message_data["message_ts"],
                "sent_at": message_data["sent_at"]
            }
            
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return {
                "success": False,
                "error": str(e),
                "channel": channel
            }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate Slack parameters"""
        
        required_params = ["channel", "message"]
        
        for param in required_params:
            if param not in parameters:
                return False
        
        # Validate channel format (basic)
        channel = parameters["channel"]
        if not (channel.startswith("#") or channel.startswith("C") or channel.startswith("D")):
            return False
        
        return True
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "channel": {"type": "string", "description": "Slack channel ID or name"},
                "message": {"type": "string", "description": "Message content"},
                "thread_ts": {"type": "string", "description": "Thread timestamp for replies"}
            },
            "required": ["channel", "message"]
        }

class CalendarTool(BaseTool):
    """Tool for calendar operations"""
    
    def __init__(self, calendar_config: Dict[str, Any] = None):
        super().__init__(
            name="calendar_manager",
            description="Manage calendar events and scheduling",
            category=ToolCategory.PRODUCTIVITY,
            version="1.0.0",
            authentication_type=AuthenticationType.OAUTH2,
            calendar_config=calendar_config or {}
        )
    
    async def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Perform calendar operation
        
        Args:
            action: Calendar action (create_event, list_events, update_event, delete_event)
            **kwargs: Action-specific parameters
            
        Returns:
            Calendar operation result
        """
        
        try:
            if action == "create_event":
                result = await self._create_event(kwargs)
            elif action == "list_events":
                result = await self._list_events(kwargs)
            elif action == "update_event":
                result = await self._update_event(kwargs)
            elif action == "delete_event":
                result = await self._delete_event(kwargs)
            else:
                raise ValueError(f"Unsupported calendar action: {action}")
            
            return {
                "success": True,
                "action": action,
                "result": result,
                "executed_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Calendar operation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "action": action
            }
    
    async def _create_event(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create calendar event"""
        
        event_data = {
            "event_id": f"event_{hash(f'{params.get(\"title\", \"\")}{params.get(\"start_time\", \"\")}')}", 
            "title": params.get("title", "Untitled Event"),
            "start_time": params.get("start_time"),
            "end_time": params.get("end_time"),
            "description": params.get("description", ""),
            "attendees": params.get("attendees", []),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Simulate calendar API call
        await asyncio.sleep(0.1)
        
        logger.info(f"Calendar event created: {event_data['title']}")
        
        return event_data
    
    async def _list_events(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """List calendar events"""
        
        # Mock event list
        events = [
            {
                "event_id": "event_1",
                "title": "Team Meeting",
                "start_time": "2024-01-15T10:00:00Z",
                "end_time": "2024-01-15T11:00:00Z"
            },
            {
                "event_id": "event_2", 
                "title": "Project Review",
                "start_time": "2024-01-15T14:00:00Z",
                "end_time": "2024-01-15T15:00:00Z"
            }
        ]
        
        return events
    
    async def _update_event(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update calendar event"""
        
        event_id = params.get("event_id")
        if not event_id:
            raise ValueError("event_id is required for update")
        
        updated_event = {
            "event_id": event_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "changes": {k: v for k, v in params.items() if k != "event_id"}
        }
        
        return updated_event
    
    async def _delete_event(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete calendar event"""
        
        event_id = params.get("event_id")
        if not event_id:
            raise ValueError("event_id is required for delete")
        
        return {
            "event_id": event_id,
            "deleted": True,
            "deleted_at": datetime.now(timezone.utc).isoformat()
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate calendar parameters"""
        
        if "action" not in parameters:
            return False
        
        action = parameters["action"]
        valid_actions = ["create_event", "list_events", "update_event", "delete_event"]
        
        if action not in valid_actions:
            return False
        
        # Action-specific validation
        if action == "create_event":
            required = ["title", "start_time", "end_time"]
            for param in required:
                if param not in parameters:
                    return False
        
        elif action in ["update_event", "delete_event"]:
            if "event_id" not in parameters:
                return False
        
        return True
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["create_event", "list_events", "update_event", "delete_event"], "description": "Calendar action"},
                "title": {"type": "string", "description": "Event title"},
                "start_time": {"type": "string", "format": "date-time", "description": "Event start time"},
                "end_time": {"type": "string", "format": "date-time", "description": "Event end time"},
                "description": {"type": "string", "description": "Event description"},
                "attendees": {"type": "array", "items": {"type": "string"}, "description": "Event attendees"},
                "event_id": {"type": "string", "description": "Event ID for update/delete operations"}
            },
            "required": ["action"]
        }
```
