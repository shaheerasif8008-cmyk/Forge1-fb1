"""
Tenant-Aware Google Drive Fetcher

Google Drive integration tool with tenant-scoped access that ensures
employees can only access files from their tenant's authorized Drive accounts.
"""

import asyncio
import logging
import os
import mimetypes
from typing import Dict, List, Optional, Any, Union
import uuid
from datetime import datetime, timezone
from pathlib import Path

from forge1.core.tenancy import get_current_tenant, set_current_tenant
from forge1.core.secret_manager import SecretManager

logger = logging.getLogger(__name__)


class DriveAccessError(Exception):
    """Exception raised when Drive access fails"""
    pass


class TenantAccessError(Exception):
    """Exception raised when tenant access is denied"""
    pass


class TenantAwareDriveFetcher:
    """
    Google Drive fetcher with tenant-scoped access.
    
    Provides Google Drive integration while ensuring strict tenant isolation
    and employee-specific file access controls.
    """
    
    def __init__(self, tenant_id: str, employee_id: str, secret_manager: Optional[SecretManager] = None):
        """
        Initialize the tenant-aware Drive fetcher.
        
        Args:
            tenant_id: Tenant ID for isolation
            employee_id: Employee ID for access control
            secret_manager: Forge1's secret manager for credentials
        """
        self.tenant_id = tenant_id
        self.employee_id = employee_id
        self.secret_manager = secret_manager
        
        # Drive client (will be initialized with tenant credentials)
        self.drive_client = None
        self.drive_info = None
        
        # Local cache directory for downloaded files
        self.cache_dir = f"/var/forge1/drive_cache/tenant_{tenant_id}"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Fetching statistics
        self.stats = {
            "files_fetched": 0,
            "total_bytes_downloaded": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "access_denied": 0
        }
        
        # File cache for metadata
        self._file_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
    async def initialize(self):
        """Initialize Drive client with tenant credentials"""
        try:
            # Set tenant context
            set_current_tenant(self.tenant_id)
            
            # Get tenant-specific Drive credentials
            credentials = await self._get_tenant_drive_credentials()
            if not credentials:
                raise DriveAccessError(f"No Drive credentials found for tenant {self.tenant_id}")
            
            # Initialize mock Drive client (in real implementation, use google-api-python-client)
            self.drive_client = MockDriveClient(credentials, self.tenant_id)
            
            # Get Drive information
            self.drive_info = await self.drive_client.get_drive_info()
            
            logger.info(f"Initialized Drive fetcher for tenant {self.tenant_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Drive fetcher: {e}")
            raise DriveAccessError(f"Drive initialization failed: {e}")
    
    async def fetch_file(self, file_id: str, download: bool = True) -> Dict[str, Any]:
        """
        Fetch file from tenant's authorized Drive.
        
        Args:
            file_id: Google Drive file ID
            download: Whether to download file content
            
        Returns:
            Dictionary containing file metadata and content path
            
        Raises:
            TenantAccessError: If file access is denied
            DriveAccessError: If fetching fails
        """
        try:
            # Ensure client is initialized
            if not self.drive_client:
                await self.initialize()
            
            # Set tenant context
            set_current_tenant(self.tenant_id)
            
            # Validate file access
            await self._validate_file_access(file_id)
            
            # Check cache first
            cache_key = f"{self.tenant_id}_{file_id}"
            cached_file = self._get_cached_file_info(cache_key)
            
            if cached_file and not download:
                self.stats["cache_hits"] += 1
                return cached_file
            
            self.stats["cache_misses"] += 1
            
            # Get file metadata from Drive
            file_metadata = await self.drive_client.get_file_metadata(file_id)
            
            # Prepare result
            result = {
                "file_id": file_id,
                "name": file_metadata["name"],
                "mime_type": file_metadata["mimeType"],
                "size": file_metadata.get("size", 0),
                "created_time": file_metadata.get("createdTime"),
                "modified_time": file_metadata.get("modifiedTime"),
                "owners": file_metadata.get("owners", []),
                "tenant_id": self.tenant_id,
                "employee_id": self.employee_id,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "cached": False,
                "local_path": None
            }
            
            # Download file content if requested
            if download:
                local_path = await self._download_file(file_id, file_metadata)
                result["local_path"] = local_path
                result["cached"] = True
                
                # Update download statistics
                file_size = int(file_metadata.get("size", 0))
                self.stats["total_bytes_downloaded"] += file_size
            
            # Cache file metadata
            self._cache_file_info(cache_key, result)
            
            # Update statistics
            self.stats["files_fetched"] += 1
            
            logger.info(f"Fetched file {file_id} for tenant {self.tenant_id}")
            return result
            
        except TenantAccessError:
            self.stats["access_denied"] += 1
            logger.warning(f"Access denied to Drive file {file_id} for tenant {self.tenant_id}")
            raise
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to fetch Drive file: {e}")
            raise DriveAccessError(f"File fetch failed: {e}")
    
    async def list_files(self, folder_id: Optional[str] = None, 
                        query: Optional[str] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """
        List files in tenant's Drive.
        
        Args:
            folder_id: Optional folder ID to list files from
            query: Optional search query
            limit: Maximum number of files to return
            
        Returns:
            List of file metadata dictionaries
        """
        try:
            # Ensure client is initialized
            if not self.drive_client:
                await self.initialize()
            
            # Set tenant context
            set_current_tenant(self.tenant_id)
            
            # Build Drive query
            drive_query = self._build_drive_query(folder_id, query)
            
            # List files from Drive
            files = await self.drive_client.list_files(drive_query, limit)
            
            # Process and format results
            formatted_files = []
            for file_data in files:
                # Check if we have access to this file
                if await self._check_file_access(file_data["id"], raise_error=False):
                    formatted_files.append({
                        "id": file_data["id"],
                        "name": file_data["name"],
                        "mime_type": file_data["mimeType"],
                        "size": file_data.get("size", 0),
                        "created_time": file_data.get("createdTime"),
                        "modified_time": file_data.get("modifiedTime"),
                        "is_folder": file_data["mimeType"] == "application/vnd.google-apps.folder",
                        "parents": file_data.get("parents", []),
                        "web_view_link": file_data.get("webViewLink"),
                        "tenant_id": self.tenant_id
                    })
            
            logger.debug(f"Listed {len(formatted_files)} files for tenant {self.tenant_id}")
            return formatted_files
            
        except Exception as e:
            logger.error(f"Failed to list Drive files: {e}")
            raise DriveAccessError(f"File listing failed: {e}")
    
    async def search_files(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search files in tenant's Drive.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching files
        """
        try:
            # Build search query
            search_query = f"name contains '{query}' or fullText contains '{query}'"
            
            return await self.list_files(query=search_query, limit=limit)
            
        except Exception as e:
            logger.error(f"Failed to search Drive files: {e}")
            raise DriveAccessError(f"File search failed: {e}")
    
    async def get_file_content(self, file_id: str) -> bytes:
        """
        Get file content as bytes.
        
        Args:
            file_id: Google Drive file ID
            
        Returns:
            File content as bytes
        """
        try:
            # Fetch file with download
            file_info = await self.fetch_file(file_id, download=True)
            
            if not file_info["local_path"]:
                raise DriveAccessError("File was not downloaded")
            
            # Read file content
            with open(file_info["local_path"], "rb") as f:
                content = f.read()
            
            return content
            
        except Exception as e:
            logger.error(f"Failed to get file content: {e}")
            raise DriveAccessError(f"File content retrieval failed: {e}")
    
    async def get_folder_contents(self, folder_id: str) -> Dict[str, Any]:
        """
        Get contents of a specific folder.
        
        Args:
            folder_id: Google Drive folder ID
            
        Returns:
            Dictionary containing folder info and contents
        """
        try:
            # Get folder metadata
            folder_info = await self.fetch_file(folder_id, download=False)
            
            # Verify it's actually a folder
            if folder_info["mime_type"] != "application/vnd.google-apps.folder":
                raise DriveAccessError(f"File {folder_id} is not a folder")
            
            # List folder contents
            contents = await self.list_files(folder_id=folder_id)
            
            return {
                "folder": folder_info,
                "contents": contents,
                "file_count": len([f for f in contents if not f["is_folder"]]),
                "folder_count": len([f for f in contents if f["is_folder"]]),
                "total_size": sum(int(f.get("size", 0)) for f in contents)
            }
            
        except Exception as e:
            logger.error(f"Failed to get folder contents: {e}")
            raise DriveAccessError(f"Folder contents retrieval failed: {e}")
    
    async def _get_tenant_drive_credentials(self) -> Optional[Dict]:
        """Get Drive credentials for the tenant"""
        try:
            if self.secret_manager:
                # Get from secret manager
                secret_key = f"drive_credentials_tenant_{self.tenant_id}"
                credentials_json = await self.secret_manager.get_secret(secret_key)
                if credentials_json:
                    import json
                    return json.loads(credentials_json)
            else:
                # Fallback to environment variable
                creds_env = os.getenv(f"DRIVE_CREDENTIALS_TENANT_{self.tenant_id.upper()}")
                if creds_env:
                    import json
                    return json.loads(creds_env)
            
            return None
        except Exception as e:
            logger.error(f"Failed to get Drive credentials: {e}")
            return None
    
    async def _validate_file_access(self, file_id: str):
        """Validate that employee has access to the file"""
        if not await self._check_file_access(file_id):
            raise TenantAccessError(f"Access denied to Drive file: {file_id}")
    
    async def _check_file_access(self, file_id: str, raise_error: bool = True) -> bool:
        """Check if employee has access to the file"""
        try:
            # In a real implementation, this would check:
            # 1. File permissions in Drive
            # 2. Tenant-specific access rules
            # 3. Employee-specific permissions
            
            # For now, simulate access check
            file_metadata = await self.drive_client.get_file_metadata(file_id)
            
            # Check if file is in tenant's authorized domain
            owners = file_metadata.get("owners", [])
            for owner in owners:
                email_domain = owner.get("emailAddress", "").split("@")[-1]
                if await self._is_authorized_domain(email_domain):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check file access: {e}")
            if raise_error:
                raise TenantAccessError(f"File access check failed: {e}")
            return False
    
    async def _is_authorized_domain(self, domain: str) -> bool:
        """Check if domain is authorized for the tenant"""
        # In a real implementation, this would check against tenant's authorized domains
        # For now, allow common domains and tenant-specific domains
        authorized_domains = [
            "gmail.com",
            "googlemail.com",
            f"{self.tenant_id}.com",
            f"tenant{self.tenant_id}.com"
        ]
        
        return domain.lower() in authorized_domains
    
    async def _download_file(self, file_id: str, file_metadata: Dict) -> str:
        """Download file to local cache"""
        try:
            # Generate local file path
            file_name = file_metadata["name"]
            safe_name = "".join(c for c in file_name if c.isalnum() or c in "._- ")
            local_path = os.path.join(self.cache_dir, f"{file_id}_{safe_name}")
            
            # Check if file already exists in cache
            if os.path.exists(local_path):
                # Check if cached file is still valid
                cached_mtime = datetime.fromtimestamp(os.path.getmtime(local_path), timezone.utc)
                drive_mtime = datetime.fromisoformat(file_metadata.get("modifiedTime", "1970-01-01T00:00:00Z").replace("Z", "+00:00"))
                
                if cached_mtime >= drive_mtime:
                    logger.debug(f"Using cached file: {local_path}")
                    return local_path
            
            # Download file from Drive
            file_content = await self.drive_client.download_file(file_id)
            
            # Save to local cache
            with open(local_path, "wb") as f:
                f.write(file_content)
            
            logger.debug(f"Downloaded file to: {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            raise DriveAccessError(f"File download failed: {e}")
    
    def _build_drive_query(self, folder_id: Optional[str], query: Optional[str]) -> str:
        """Build Google Drive API query"""
        query_parts = []
        
        # Add folder filter
        if folder_id:
            query_parts.append(f"'{folder_id}' in parents")
        
        # Add custom query
        if query:
            query_parts.append(query)
        
        # Add tenant-specific filters (exclude trashed files)
        query_parts.append("trashed = false")
        
        return " and ".join(query_parts) if query_parts else "trashed = false"
    
    def _get_cached_file_info(self, cache_key: str) -> Optional[Dict]:
        """Get cached file information"""
        if cache_key in self._file_cache:
            cached_data = self._file_cache[cache_key]
            age = (datetime.now(timezone.utc) - cached_data["timestamp"]).total_seconds()
            
            if age < self._cache_ttl:
                return cached_data["info"]
            else:
                # Remove expired cache entry
                del self._file_cache[cache_key]
        
        return None
    
    def _cache_file_info(self, cache_key: str, file_info: Dict):
        """Cache file information"""
        self._file_cache[cache_key] = {
            "info": file_info,
            "timestamp": datetime.now(timezone.utc)
        }
        
        # Limit cache size
        if len(self._file_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(
                self._file_cache.keys(),
                key=lambda k: self._file_cache[k]["timestamp"]
            )[:200]
            
            for key in oldest_keys:
                del self._file_cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fetching statistics"""
        return {
            **self.stats,
            "tenant_id": self.tenant_id,
            "employee_id": self.employee_id,
            "cache_size": len(self._file_cache),
            "cache_directory": self.cache_dir,
            "cache_hit_rate": (
                self.stats["cache_hits"] / 
                max(self.stats["cache_hits"] + self.stats["cache_misses"], 1)
            )
        }
    
    def clear_cache(self):
        """Clear file cache"""
        self._file_cache.clear()
        
        # Optionally clear local file cache
        try:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to clear local cache: {e}")
        
        logger.info(f"Cleared Drive cache for tenant {self.tenant_id}")
    
    def reset_stats(self):
        """Reset fetching statistics"""
        self.stats = {
            "files_fetched": 0,
            "total_bytes_downloaded": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "access_denied": 0
        }
        
        logger.info(f"Reset Drive fetcher stats for tenant {self.tenant_id}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Test Drive connection
            if not self.drive_client:
                await self.initialize()
            
            # Test API call
            drive_info = await self.drive_client.get_drive_info()
            
            return {
                "status": "healthy",
                "tenant_id": self.tenant_id,
                "employee_id": self.employee_id,
                "drive_name": drive_info.get("name", "unknown"),
                "drive_id": drive_info.get("id", "unknown"),
                "stats": self.get_stats()
            }
        except Exception as e:
            logger.error(f"Drive fetcher health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "tenant_id": self.tenant_id,
                "employee_id": self.employee_id
            }


class MockDriveClient:
    """Mock Google Drive client for testing and development"""
    
    def __init__(self, credentials: Dict, tenant_id: str):
        self.credentials = credentials
        self.tenant_id = tenant_id
        self.drive_info = {
            "id": f"drive_{tenant_id}",
            "name": f"Tenant {tenant_id} Drive"
        }
    
    async def get_drive_info(self) -> Dict[str, Any]:
        """Get Drive information"""
        return self.drive_info
    
    async def get_file_metadata(self, file_id: str) -> Dict[str, Any]:
        """Mock file metadata retrieval"""
        await asyncio.sleep(0.1)  # Simulate API delay
        
        return {
            "id": file_id,
            "name": f"document_{file_id}.pdf",
            "mimeType": "application/pdf",
            "size": "1024000",
            "createdTime": "2024-01-01T00:00:00Z",
            "modifiedTime": "2024-01-01T12:00:00Z",
            "owners": [
                {
                    "emailAddress": f"user@{self.tenant_id}.com",
                    "displayName": f"Tenant {self.tenant_id} User"
                }
            ],
            "parents": ["root"],
            "webViewLink": f"https://drive.google.com/file/d/{file_id}/view"
        }
    
    async def list_files(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Mock file listing"""
        await asyncio.sleep(0.2)  # Simulate API delay
        
        # Generate mock files
        files = []
        for i in range(min(limit, 10)):  # Return up to 10 mock files
            file_id = f"file_{self.tenant_id}_{i:03d}"
            files.append({
                "id": file_id,
                "name": f"document_{i}.pdf",
                "mimeType": "application/pdf",
                "size": str(1024 * (i + 1)),
                "createdTime": "2024-01-01T00:00:00Z",
                "modifiedTime": "2024-01-01T12:00:00Z",
                "parents": ["root"],
                "webViewLink": f"https://drive.google.com/file/d/{file_id}/view"
            })
        
        return files
    
    async def download_file(self, file_id: str) -> bytes:
        """Mock file download"""
        await asyncio.sleep(0.5)  # Simulate download delay
        
        # Return mock file content
        content = f"Mock file content for {file_id} in tenant {self.tenant_id}\n"
        content += "This is a test document with some sample text.\n"
        content += f"File ID: {file_id}\n"
        content += f"Tenant: {self.tenant_id}\n"
        
        return content.encode('utf-8')