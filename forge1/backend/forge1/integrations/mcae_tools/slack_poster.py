"""
Tenant-Aware Slack Poster

Slack posting tool with tenant-specific credentials and channel access
that ensures employees can only post to authorized Slack workspaces.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any, Union
import uuid
from datetime import datetime, timezone

from forge1.core.tenancy import get_current_tenant, set_current_tenant
from forge1.core.secret_manager import SecretManager

logger = logging.getLogger(__name__)


class SlackPostingError(Exception):
    """Exception raised when Slack posting fails"""
    pass


class TenantAccessError(Exception):
    """Exception raised when tenant access is denied"""
    pass


class TenantAwareSlackPoster:
    """
    Slack posting tool with tenant-specific credentials.
    
    Provides Slack integration while ensuring strict tenant isolation
    and employee-specific channel access controls.
    """
    
    def __init__(self, tenant_id: str, employee_id: str, secret_manager: Optional[SecretManager] = None):
        """
        Initialize the tenant-aware Slack poster.
        
        Args:
            tenant_id: Tenant ID for isolation
            employee_id: Employee ID for access control
            secret_manager: Forge1's secret manager for credentials
        """
        self.tenant_id = tenant_id
        self.employee_id = employee_id
        self.secret_manager = secret_manager
        
        # Slack client (will be initialized with tenant credentials)
        self.slack_client = None
        self.workspace_info = None
        
        # Posting statistics
        self.stats = {
            "messages_posted": 0,
            "channels_used": set(),
            "total_characters": 0,
            "errors": 0,
            "access_denied": 0
        }
        
        # Channel access cache
        self._channel_cache = {}
        self._cache_ttl = 600  # 10 minutes
        
    async def initialize(self):
        """Initialize Slack client with tenant credentials"""
        try:
            # Set tenant context
            set_current_tenant(self.tenant_id)
            
            # Get tenant-specific Slack credentials
            slack_token = await self._get_tenant_slack_token()
            if not slack_token:
                raise SlackPostingError(f"No Slack credentials found for tenant {self.tenant_id}")
            
            # Initialize mock Slack client (in real implementation, use slack_sdk)
            self.slack_client = MockSlackClient(slack_token, self.tenant_id)
            
            # Get workspace information
            self.workspace_info = await self.slack_client.get_workspace_info()
            
            logger.info(f"Initialized Slack poster for tenant {self.tenant_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Slack poster: {e}")
            raise SlackPostingError(f"Slack initialization failed: {e}")
    
    async def post_message(self, channel: str, message: str, 
                          thread_ts: Optional[str] = None,
                          attachments: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Post message to tenant-specific Slack workspace.
        
        Args:
            channel: Channel name or ID to post to
            message: Message text to post
            thread_ts: Optional thread timestamp for replies
            attachments: Optional message attachments
            
        Returns:
            Dictionary containing post result and metadata
            
        Raises:
            TenantAccessError: If channel access is denied
            SlackPostingError: If posting fails
        """
        try:
            # Ensure client is initialized
            if not self.slack_client:
                await self.initialize()
            
            # Set tenant context
            set_current_tenant(self.tenant_id)
            
            # Validate channel access
            await self._validate_channel_access(channel)
            
            # Prepare message data
            message_data = {
                "channel": channel,
                "text": message,
                "thread_ts": thread_ts,
                "attachments": attachments or [],
                "metadata": {
                    "tenant_id": self.tenant_id,
                    "employee_id": self.employee_id,
                    "posted_at": datetime.now(timezone.utc).isoformat()
                }
            }
            
            # Post message through Slack client
            response = await self.slack_client.post_message(message_data)
            
            # Update statistics
            self.stats["messages_posted"] += 1
            self.stats["channels_used"].add(channel)
            self.stats["total_characters"] += len(message)
            
            # Prepare result
            result = {
                "message_id": response.get("ts"),
                "channel": channel,
                "message": message,
                "thread_ts": thread_ts,
                "tenant_id": self.tenant_id,
                "employee_id": self.employee_id,
                "posted_at": datetime.now(timezone.utc).isoformat(),
                "workspace": self.workspace_info.get("name") if self.workspace_info else "unknown",
                "success": response.get("ok", False),
                "response": response
            }
            
            logger.info(f"Posted message to Slack channel {channel} for tenant {self.tenant_id}")
            return result
            
        except TenantAccessError:
            self.stats["access_denied"] += 1
            logger.warning(f"Access denied to Slack channel {channel} for tenant {self.tenant_id}")
            raise
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to post Slack message: {e}")
            raise SlackPostingError(f"Message posting failed: {e}")
    
    async def post_file(self, channel: str, file_path: str, 
                       title: Optional[str] = None,
                       comment: Optional[str] = None) -> Dict[str, Any]:
        """
        Post file to Slack channel.
        
        Args:
            channel: Channel to post file to
            file_path: Path to file to upload
            title: Optional file title
            comment: Optional file comment
            
        Returns:
            Dictionary containing upload result
        """
        try:
            # Ensure client is initialized
            if not self.slack_client:
                await self.initialize()
            
            # Set tenant context
            set_current_tenant(self.tenant_id)
            
            # Validate channel access
            await self._validate_channel_access(channel)
            
            # Validate file path (ensure it's within tenant scope)
            validated_path = await self._validate_file_path(file_path)
            
            # Upload file through Slack client
            response = await self.slack_client.upload_file(
                channel=channel,
                file_path=validated_path,
                title=title,
                comment=comment
            )
            
            # Update statistics
            self.stats["messages_posted"] += 1  # Count file uploads as posts
            self.stats["channels_used"].add(channel)
            
            result = {
                "file_id": response.get("file", {}).get("id"),
                "channel": channel,
                "file_path": file_path,
                "title": title,
                "comment": comment,
                "tenant_id": self.tenant_id,
                "employee_id": self.employee_id,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "success": response.get("ok", False),
                "response": response
            }
            
            logger.info(f"Uploaded file to Slack channel {channel} for tenant {self.tenant_id}")
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to upload file to Slack: {e}")
            raise SlackPostingError(f"File upload failed: {e}")
    
    async def list_channels(self, include_private: bool = False) -> List[Dict[str, Any]]:
        """
        List available Slack channels for the tenant.
        
        Args:
            include_private: Whether to include private channels
            
        Returns:
            List of channel information
        """
        try:
            # Ensure client is initialized
            if not self.slack_client:
                await self.initialize()
            
            # Set tenant context
            set_current_tenant(self.tenant_id)
            
            # Get channels from Slack
            channels = await self.slack_client.list_channels(include_private)
            
            # Filter channels based on employee permissions
            accessible_channels = []
            for channel in channels:
                if await self._check_channel_access(channel["id"], raise_error=False):
                    accessible_channels.append({
                        "id": channel["id"],
                        "name": channel["name"],
                        "is_private": channel.get("is_private", False),
                        "is_member": channel.get("is_member", False),
                        "purpose": channel.get("purpose", {}).get("value", ""),
                        "topic": channel.get("topic", {}).get("value", ""),
                        "member_count": channel.get("num_members", 0)
                    })
            
            logger.debug(f"Listed {len(accessible_channels)} accessible channels for tenant {self.tenant_id}")
            return accessible_channels
            
        except Exception as e:
            logger.error(f"Failed to list Slack channels: {e}")
            raise SlackPostingError(f"Channel listing failed: {e}")
    
    async def get_channel_history(self, channel: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent messages from a Slack channel.
        
        Args:
            channel: Channel to get history from
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of recent messages
        """
        try:
            # Ensure client is initialized
            if not self.slack_client:
                await self.initialize()
            
            # Set tenant context
            set_current_tenant(self.tenant_id)
            
            # Validate channel access
            await self._validate_channel_access(channel)
            
            # Get channel history
            messages = await self.slack_client.get_channel_history(channel, limit)
            
            # Format messages
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "ts": msg.get("ts"),
                    "user": msg.get("user"),
                    "text": msg.get("text", ""),
                    "type": msg.get("type", "message"),
                    "timestamp": datetime.fromtimestamp(float(msg.get("ts", 0))).isoformat()
                })
            
            logger.debug(f"Retrieved {len(formatted_messages)} messages from channel {channel}")
            return formatted_messages
            
        except Exception as e:
            logger.error(f"Failed to get channel history: {e}")
            raise SlackPostingError(f"Channel history retrieval failed: {e}")
    
    async def _get_tenant_slack_token(self) -> Optional[str]:
        """Get Slack token for the tenant"""
        try:
            if self.secret_manager:
                # Get from secret manager
                secret_key = f"slack_token_tenant_{self.tenant_id}"
                return await self.secret_manager.get_secret(secret_key)
            else:
                # Fallback to environment variable
                return os.getenv(f"SLACK_TOKEN_TENANT_{self.tenant_id.upper()}")
        except Exception as e:
            logger.error(f"Failed to get Slack token: {e}")
            return None
    
    async def _validate_channel_access(self, channel: str):
        """Validate that employee has access to the channel"""
        if not await self._check_channel_access(channel):
            raise TenantAccessError(f"Access denied to Slack channel: {channel}")
    
    async def _check_channel_access(self, channel: str, raise_error: bool = True) -> bool:
        """Check if employee has access to the channel"""
        try:
            # Check cache first
            cache_key = f"{self.tenant_id}_{self.employee_id}_{channel}"
            if cache_key in self._channel_cache:
                cached_data = self._channel_cache[cache_key]
                age = (datetime.now(timezone.utc) - cached_data["timestamp"]).total_seconds()
                if age < self._cache_ttl:
                    return cached_data["access"]
            
            # Check channel access through Slack client
            has_access = await self.slack_client.check_channel_access(channel)
            
            # Cache the result
            self._channel_cache[cache_key] = {
                "access": has_access,
                "timestamp": datetime.now(timezone.utc)
            }
            
            return has_access
            
        except Exception as e:
            logger.error(f"Failed to check channel access: {e}")
            if raise_error:
                raise TenantAccessError(f"Channel access check failed: {e}")
            return False
    
    async def _validate_file_path(self, file_path: str) -> str:
        """Validate file path is within tenant scope"""
        # In a real implementation, this would validate the file path
        # is within the tenant's allowed directories
        if not os.path.exists(file_path):
            raise SlackPostingError(f"File not found: {file_path}")
        
        if not os.access(file_path, os.R_OK):
            raise TenantAccessError(f"Access denied to file: {file_path}")
        
        return file_path
    
    def get_stats(self) -> Dict[str, Any]:
        """Get posting statistics"""
        return {
            "messages_posted": self.stats["messages_posted"],
            "channels_used": len(self.stats["channels_used"]),
            "channel_list": list(self.stats["channels_used"]),
            "total_characters": self.stats["total_characters"],
            "errors": self.stats["errors"],
            "access_denied": self.stats["access_denied"],
            "tenant_id": self.tenant_id,
            "employee_id": self.employee_id,
            "workspace": self.workspace_info.get("name") if self.workspace_info else "unknown"
        }
    
    def reset_stats(self):
        """Reset posting statistics"""
        self.stats = {
            "messages_posted": 0,
            "channels_used": set(),
            "total_characters": 0,
            "errors": 0,
            "access_denied": 0
        }
        
        logger.info(f"Reset Slack poster stats for tenant {self.tenant_id}")
    
    def clear_cache(self):
        """Clear channel access cache"""
        self._channel_cache.clear()
        logger.info(f"Cleared Slack channel cache for tenant {self.tenant_id}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Test Slack connection
            if not self.slack_client:
                await self.initialize()
            
            # Test API call
            workspace_info = await self.slack_client.get_workspace_info()
            
            return {
                "status": "healthy",
                "tenant_id": self.tenant_id,
                "employee_id": self.employee_id,
                "workspace_name": workspace_info.get("name", "unknown"),
                "workspace_id": workspace_info.get("id", "unknown"),
                "stats": self.get_stats()
            }
        except Exception as e:
            logger.error(f"Slack poster health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "tenant_id": self.tenant_id,
                "employee_id": self.employee_id
            }


class MockSlackClient:
    """Mock Slack client for testing and development"""
    
    def __init__(self, token: str, tenant_id: str):
        self.token = token
        self.tenant_id = tenant_id
        self.workspace_info = {
            "id": f"T{tenant_id.upper()}",
            "name": f"Tenant {tenant_id} Workspace"
        }
    
    async def get_workspace_info(self) -> Dict[str, Any]:
        """Get workspace information"""
        return self.workspace_info
    
    async def post_message(self, message_data: Dict) -> Dict[str, Any]:
        """Mock message posting"""
        await asyncio.sleep(0.1)  # Simulate API delay
        
        return {
            "ok": True,
            "ts": f"{datetime.now().timestamp()}",
            "channel": message_data["channel"],
            "message": {
                "text": message_data["text"],
                "user": "U123456",
                "ts": f"{datetime.now().timestamp()}"
            }
        }
    
    async def upload_file(self, channel: str, file_path: str, 
                         title: Optional[str] = None,
                         comment: Optional[str] = None) -> Dict[str, Any]:
        """Mock file upload"""
        await asyncio.sleep(0.2)  # Simulate upload delay
        
        return {
            "ok": True,
            "file": {
                "id": f"F{uuid.uuid4().hex[:8].upper()}",
                "name": os.path.basename(file_path),
                "title": title or os.path.basename(file_path),
                "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
        }
    
    async def list_channels(self, include_private: bool = False) -> List[Dict[str, Any]]:
        """Mock channel listing"""
        channels = [
            {
                "id": "C123456",
                "name": "general",
                "is_private": False,
                "is_member": True,
                "purpose": {"value": "General discussion"},
                "topic": {"value": "Welcome to the workspace"},
                "num_members": 10
            },
            {
                "id": "C789012",
                "name": f"tenant-{self.tenant_id}",
                "is_private": False,
                "is_member": True,
                "purpose": {"value": f"Tenant {self.tenant_id} specific channel"},
                "topic": {"value": "Tenant communications"},
                "num_members": 5
            }
        ]
        
        if include_private:
            channels.append({
                "id": "G345678",
                "name": "private-group",
                "is_private": True,
                "is_member": True,
                "purpose": {"value": "Private discussions"},
                "topic": {"value": ""},
                "num_members": 3
            })
        
        return channels
    
    async def get_channel_history(self, channel: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Mock channel history"""
        messages = []
        base_time = datetime.now().timestamp()
        
        for i in range(min(limit, 5)):  # Return up to 5 mock messages
            messages.append({
                "ts": str(base_time - (i * 3600)),  # 1 hour apart
                "user": f"U{i:06d}",
                "text": f"Mock message {i + 1} in channel {channel}",
                "type": "message"
            })
        
        return messages
    
    async def check_channel_access(self, channel: str) -> bool:
        """Mock channel access check"""
        # Allow access to channels that start with tenant ID or are general
        return channel.startswith(f"tenant-{self.tenant_id}") or channel in ["general", "C123456"]