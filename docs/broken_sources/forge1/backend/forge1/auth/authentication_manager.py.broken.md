```python
# forge1/backend/forge1/auth/authentication_manager.py
"""
Authentication Manager for Forge 1

Comprehensive authentication system providing:
- Secure credential handling and storage
- OAuth2, API key, and certificate-based authentication
- Azure KeyVault integration for secure token storage
- Multi-tenant authentication support
- Enterprise security standards compliance
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from abc import ABC, abstractmethod
import json
import uuid
import base64
import hashlib
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

class AuthenticationType(Enum):
    """Types of authentication supported"""
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BASIC_AUTH = "basic_auth"
    BEARER_TOKEN = "bearer_token"
    CERTIFICATE = "certificate"
    JWT = "jwt"
    SAML = "saml"

class CredentialStatus(Enum):
    """Status of stored credentials"""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING = "pending"
    SUSPENDED = "suspended"c
lass BaseCredential(ABC):
    """Base class for all credential types"""
    
    def __init__(
        self,
        credential_id: str,
        name: str,
        auth_type: AuthenticationType,
        tenant_id: str = None,
        expires_at: datetime = None,
        **kwargs
    ):
        self.credential_id = credential_id
        self.name = name
        self.auth_type = auth_type
        self.tenant_id = tenant_id
        self.expires_at = expires_at
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        self.status = CredentialStatus.ACTIVE
        self.metadata = kwargs
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert credential to dictionary"""
        pass
    
    @abstractmethod
    def is_valid(self) -> bool:
        """Check if credential is valid"""
        pass
    
    def is_expired(self) -> bool:
        """Check if credential is expired"""
        if self.expires_at:
            return datetime.now(timezone.utc) > self.expires_at
        return False

class APIKeyCredential(BaseCredential):
    """API Key credential implementation"""
    
    def __init__(self, credential_id: str, name: str, api_key: str, **kwargs):
        super().__init__(credential_id, name, AuthenticationType.API_KEY, **kwargs)
        self.api_key = api_key
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "credential_id": self.credential_id,
            "name": self.name,
            "auth_type": self.auth_type.value,
            "tenant_id": self.tenant_id,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "api_key": self.api_key  # In production, this would be encrypted
        }
    
    def is_valid(self) -> bool:
        return (
            self.status == CredentialStatus.ACTIVE and
            not self.is_expired() and
            bool(self.api_key)
        )

class OAuth2Credential(BaseCredential):
    """OAuth2 credential implementation"""
    
    def __init__(
        self,
        credential_id: str,
        name: str,
        client_id: str,
        client_secret: str,
        access_token: str = None,
        refresh_token: str = None,
        token_url: str = None,
        **kwargs
    ):
        super().__init__(credential_id, name, AuthenticationType.OAUTH2, **kwargs)
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.token_url = token_url
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "credential_id": self.credential_id,
            "name": self.name,
            "auth_type": self.auth_type.value,
            "tenant_id": self.tenant_id,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "client_id": self.client_id,
            "client_secret": self.client_secret,  # In production, encrypted
            "access_token": self.access_token,    # In production, encrypted
            "refresh_token": self.refresh_token,  # In production, encrypted
            "token_url": self.token_url
        }
    
    def is_valid(self) -> bool:
        return (
            self.status == CredentialStatus.ACTIVE and
            not self.is_expired() and
            bool(self.client_id) and
            bool(self.client_secret)
        )

class BasicAuthCredential(BaseCredential):
    """Basic Authentication credential implementation"""
    
    def __init__(self, credential_id: str, name: str, username: str, password: str, **kwargs):
        super().__init__(credential_id, name, AuthenticationType.BASIC_AUTH, **kwargs)
        self.username = username
        self.password = password
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "credential_id": self.credential_id,
            "name": self.name,
            "auth_type": self.auth_type.value,
            "tenant_id": self.tenant_id,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "username": self.username,
            "password": self.password  # In production, this would be encrypted
        }
    
    def is_valid(self) -> bool:
        return (
            self.status == CredentialStatus.ACTIVE and
            not self.is_expired() and
            bool(self.username) and
            bool(self.password)
        )clas
s EncryptionManager:
    """Handles encryption and decryption of sensitive data"""
    
    def __init__(self, encryption_key: bytes = None):
        if encryption_key:
            self.fernet = Fernet(encryption_key)
        else:
            # Generate a new key for demonstration
            self.fernet = Fernet(Fernet.generate_key())
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    @staticmethod
    def generate_key() -> bytes:
        """Generate a new encryption key"""
        return Fernet.generate_key()

class AzureKeyVaultManager:
    """Azure KeyVault integration for secure credential storage"""
    
    def __init__(self, vault_url: str = None, credential: Any = None):
        self.vault_url = vault_url
        self.credential = credential
        self.client = None
        
        # Mock KeyVault for demonstration
        self.mock_vault = {}
    
    async def store_secret(self, name: str, value: str, **kwargs) -> Dict[str, Any]:
        """Store secret in Azure KeyVault"""
        try:
            # Mock implementation for demonstration
            secret_id = f"secret_{uuid.uuid4().hex[:8]}"
            self.mock_vault[name] = {
                "id": secret_id,
                "value": value,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": kwargs
            }
            
            logger.info(f"Secret {name} stored in KeyVault")
            return {"id": secret_id, "name": name, "version": "1"}
            
        except Exception as e:
            logger.error(f"Failed to store secret {name}: {e}")
            raise
    
    async def get_secret(self, name: str) -> Optional[str]:
        """Retrieve secret from Azure KeyVault"""
        try:
            # Mock implementation
            if name in self.mock_vault:
                return self.mock_vault[name]["value"]
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve secret {name}: {e}")
            raise
    
    async def delete_secret(self, name: str) -> bool:
        """Delete secret from Azure KeyVault"""
        try:
            # Mock implementation
            if name in self.mock_vault:
                del self.mock_vault[name]
                logger.info(f"Secret {name} deleted from KeyVault")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete secret {name}: {e}")
            raise

class AuthenticationManager:
    """Comprehensive authentication management system"""
    
    def __init__(
        self,
        encryption_key: bytes = None,
        azure_keyvault_url: str = None,
        azure_credential: Any = None,
        enable_keyvault: bool = False
    ):
        self.encryption_manager = EncryptionManager(encryption_key)
        self.enable_keyvault = enable_keyvault
        
        if enable_keyvault:
            self.keyvault_manager = AzureKeyVaultManager(azure_keyvault_url, azure_credential)
        else:
            self.keyvault_manager = None
        
        # In-memory credential storage (in production, use secure database)
        self.credentials: Dict[str, BaseCredential] = {}
        self.tenant_credentials: Dict[str, List[str]] = {}  # tenant_id -> [credential_ids]
        
        # Authentication metrics
        self.auth_metrics = {
            "credentials_stored": 0,
            "authentications_performed": 0,
            "successful_authentications": 0,
            "failed_authentications": 0,
            "tokens_refreshed": 0,
            "credentials_expired": 0
        }
        
        logger.info("Authentication Manager initialized")
    
    async def store_credential(
        self,
        name: str,
        auth_type: AuthenticationType,
        credential_data: Dict[str, Any],
        tenant_id: str = None,
        expires_at: datetime = None,
        use_keyvault: bool = None
    ) -> str:
        """Store authentication credential securely
        
        Args:
            name: Credential name
            auth_type: Type of authentication
            credential_data: Credential data (keys, tokens, etc.)
            tenant_id: Tenant ID for multi-tenancy
            expires_at: Expiration datetime
            use_keyvault: Whether to use Azure KeyVault
            
        Returns:
            Credential ID
        """
        
        credential_id = f"cred_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create credential object based on type
            if auth_type == AuthenticationType.API_KEY:
                credential = APIKeyCredential(
                    credential_id=credential_id,
                    name=name,
                    api_key=credential_data["api_key"],
                    tenant_id=tenant_id,
                    expires_at=expires_at
                )
            elif auth_type == AuthenticationType.OAUTH2:
                credential = OAuth2Credential(
                    credential_id=credential_id,
                    name=name,
                    client_id=credential_data["client_id"],
                    client_secret=credential_data["client_secret"],
                    access_token=credential_data.get("access_token"),
                    refresh_token=credential_data.get("refresh_token"),
                    token_url=credential_data.get("token_url"),
                    tenant_id=tenant_id,
                    expires_at=expires_at
                )
            elif auth_type == AuthenticationType.BASIC_AUTH:
                credential = BasicAuthCredential(
                    credential_id=credential_id,
                    name=name,
                    username=credential_data["username"],
                    password=credential_data["password"],
                    tenant_id=tenant_id,
                    expires_at=expires_at
                )
            else:
                raise ValueError(f"Unsupported authentication type: {auth_type}")
            
            # Store credential
            self.credentials[credential_id] = credential
            
            # Update tenant mapping
            if tenant_id:
                if tenant_id not in self.tenant_credentials:
                    self.tenant_credentials[tenant_id] = []
                self.tenant_credentials[tenant_id].append(credential_id)
            
            # Store in KeyVault if enabled
            if (use_keyvault or (use_keyvault is None and self.enable_keyvault)) and self.keyvault_manager:
                await self._store_in_keyvault(credential)
            
            # Update metrics
            self.auth_metrics["credentials_stored"] += 1
            
            logger.info(f"Credential {name} stored with ID {credential_id}")
            return credential_id
            
        except Exception as e:
            logger.error(f"Failed to store credential {name}: {e}")
            raise
    
    async def _store_in_keyvault(self, credential: BaseCredential) -> None:
        """Store credential in Azure KeyVault"""
        
        if not self.keyvault_manager:
            return
        
        # Store sensitive parts in KeyVault
        if isinstance(credential, APIKeyCredential):
            await self.keyvault_manager.store_secret(
                f"{credential.credential_id}-api-key",
                credential.api_key
            )
        elif isinstance(credential, OAuth2Credential):
            await self.keyvault_manager.store_secret(
                f"{credential.credential_id}-client-secret",
                credential.client_secret
            )
            if credential.access_token:
                await self.keyvault_manager.store_secret(
                    f"{credential.credential_id}-access-token",
                    credential.access_token
                )
        elif isinstance(credential, BasicAuthCredential):
            await self.keyvault_manager.store_secret(
                f"{credential.credential_id}-password",
                credential.password
            )
    
    async def get_credential(self, credential_id: str) -> Optional[BaseCredential]:
        """Retrieve credential by ID
        
        Args:
            credential_id: Credential ID
            
        Returns:
            Credential object or None if not found
        """
        
        credential = self.credentials.get(credential_id)
        
        if credential and self.keyvault_manager:
            # Retrieve sensitive data from KeyVault
            await self._load_from_keyvault(credential)
        
        return credential
    
    async def _load_from_keyvault(self, credential: BaseCredential) -> None:
        """Load sensitive credential data from KeyVault"""
        
        if not self.keyvault_manager:
            return
        
        try:
            if isinstance(credential, APIKeyCredential):
                api_key = await self.keyvault_manager.get_secret(
                    f"{credential.credential_id}-api-key"
                )
                if api_key:
                    credential.api_key = api_key
            elif isinstance(credential, OAuth2Credential):
                client_secret = await self.keyvault_manager.get_secret(
                    f"{credential.credential_id}-client-secret"
                )
                if client_secret:
                    credential.client_secret = client_secret
                
                access_token = await self.keyvault_manager.get_secret(
                    f"{credential.credential_id}-access-token"
                )
                if access_token:
                    credential.access_token = access_token
            elif isinstance(credential, BasicAuthCredential):
                password = await self.keyvault_manager.get_secret(
                    f"{credential.credential_id}-password"
                )
                if password:
                    credential.password = password
                    
        except Exception as e:
            logger.error(f"Failed to load credential from KeyVault: {e}")
    
    async def authenticate(
        self,
        credential_id: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Perform authentication using stored credential
        
        Args:
            credential_id: Credential ID to use
            context: Authentication context
            
        Returns:
            Authentication result
        """
        
        auth_id = f"auth_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now(timezone.utc)
        
        try:
            credential = await self.get_credential(credential_id)
            
            if not credential:
                return {
                    "success": False,
                    "error": "Credential not found",
                    "auth_id": auth_id,
                    "credential_id": credential_id
                }
            
            if not credential.is_valid():
                return {
                    "success": False,
                    "error": "Credential is invalid or expired",
                    "auth_id": auth_id,
                    "credential_id": credential_id
                }
            
            # Perform authentication based on type
            if credential.auth_type == AuthenticationType.API_KEY:
                result = await self._authenticate_api_key(credential, context or {})
            elif credential.auth_type == AuthenticationType.OAUTH2:
                result = await self._authenticate_oauth2(credential, context or {})
            elif credential.auth_type == AuthenticationType.BASIC_AUTH:
                result = await self._authenticate_basic_auth(credential, context or {})
            else:
                return {
                    "success": False,
                    "error": f"Unsupported authentication type: {credential.auth_type.value}",
                    "auth_id": auth_id,
                    "credential_id": credential_id
                }
            
            # Update metrics
            self.auth_metrics["authentications_performed"] += 1
            if result["success"]:
                self.auth_metrics["successful_authentications"] += 1
            else:
                self.auth_metrics["failed_authentications"] += 1
            
            result.update({
                "auth_id": auth_id,
                "credential_id": credential_id,
                "auth_type": credential.auth_type.value,
                "duration": (datetime.now(timezone.utc) - start_time).total_seconds()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Authentication failed for credential {credential_id}: {e}")
            self.auth_metrics["failed_authentications"] += 1
            
            return {
                "success": False,
                "error": str(e),
                "auth_id": auth_id,
                "credential_id": credential_id,
                "duration": (datetime.now(timezone.utc) - start_time).total_seconds()
            }
    
    async def _authenticate_api_key(
        self,
        credential: APIKeyCredential,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Authenticate using API key"""
        
        # Mock API key authentication
        return {
            "success": True,
            "auth_method": "api_key",
            "headers": {
                "Authorization": f"Bearer {credential.api_key}",
                "X-API-Key": credential.api_key
            }
        }
    
    async def _authenticate_oauth2(
        self,
        credential: OAuth2Credential,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Authenticate using OAuth2"""
        
        # Check if access token needs refresh
        if credential.is_expired() and credential.refresh_token:
            refresh_result = await self._refresh_oauth2_token(credential)
            if not refresh_result["success"]:
                return refresh_result
        
        # Mock OAuth2 authentication
        return {
            "success": True,
            "auth_method": "oauth2",
            "headers": {
                "Authorization": f"Bearer {credential.access_token}"
            },
            "token_type": "Bearer",
            "access_token": credential.access_token
        }
    
    async def _authenticate_basic_auth(
        self,
        credential: BasicAuthCredential,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Authenticate using Basic Auth"""
        
        # Create Basic Auth header
        auth_string = f"{credential.username}:{credential.password}"
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
        
        return {
            "success": True,
            "auth_method": "basic_auth",
            "headers": {
                "Authorization": f"Basic {auth_b64}"
            },
            "username": credential.username
        }
    
    async def _refresh_oauth2_token(self, credential: OAuth2Credential) -> Dict[str, Any]:
        """Refresh OAuth2 access token"""
        
        try:
            # Mock token refresh (in production, make actual OAuth2 request)
            new_access_token = f"new_token_{uuid.uuid4().hex[:16]}"
            new_expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
            
            # Update credential
            credential.access_token = new_access_token
            credential.expires_at = new_expires_at
            credential.updated_at = datetime.now(timezone.utc)
            
            # Update in KeyVault if enabled
            if self.keyvault_manager:
                await self.keyvault_manager.store_secret(
                    f"{credential.credential_id}-access-token",
                    new_access_token
                )
            
            # Update metrics
            self.auth_metrics["tokens_refreshed"] += 1
            
            logger.info(f"OAuth2 token refreshed for credential {credential.credential_id}")
            
            return {
                "success": True,
                "access_token": new_access_token,
                "expires_at": new_expires_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to refresh OAuth2 token: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_credentials(self, tenant_id: str = None) -> List[Dict[str, Any]]:
        """List stored credentials
        
        Args:
            tenant_id: Filter by tenant ID (optional)
            
        Returns:
            List of credential information (without sensitive data)
        """
        
        credentials = []
        
        for credential in self.credentials.values():
            if tenant_id and credential.tenant_id != tenant_id:
                continue
            
            cred_info = {
                "credential_id": credential.credential_id,
                "name": credential.name,
                "auth_type": credential.auth_type.value,
                "tenant_id": credential.tenant_id,
                "status": credential.status.value,
                "created_at": credential.created_at.isoformat(),
                "updated_at": credential.updated_at.isoformat(),
                "expires_at": credential.expires_at.isoformat() if credential.expires_at else None,
                "is_expired": credential.is_expired(),
                "is_valid": credential.is_valid()
            }
            
            credentials.append(cred_info)
        
        return credentials
    
    async def delete_credential(self, credential_id: str) -> bool:
        """Delete stored credential
        
        Args:
            credential_id: Credential ID to delete
            
        Returns:
            True if deleted successfully
        """
        
        try:
            if credential_id not in self.credentials:
                return False
            
            credential = self.credentials[credential_id]
            
            # Remove from KeyVault if enabled
            if self.keyvault_manager:
                await self._delete_from_keyvault(credential)
            
            # Remove from tenant mapping
            if credential.tenant_id and credential.tenant_id in self.tenant_credentials:
                if credential_id in self.tenant_credentials[credential.tenant_id]:
                    self.tenant_credentials[credential.tenant_id].remove(credential_id)
            
            # Remove credential
            del self.credentials[credential_id]
            
            logger.info(f"Credential {credential_id} deleted")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete credential {credential_id}: {e}")
            return False
    
    async def _delete_from_keyvault(self, credential: BaseCredential) -> None:
        """Delete credential from Azure KeyVault"""
        
        if not self.keyvault_manager:
            return
        
        try:
            if isinstance(credential, APIKeyCredential):
                await self.keyvault_manager.delete_secret(
                    f"{credential.credential_id}-api-key"
                )
            elif isinstance(credential, OAuth2Credential):
                await self.keyvault_manager.delete_secret(
                    f"{credential.credential_id}-client-secret"
                )
                await self.keyvault_manager.delete_secret(
                    f"{credential.credential_id}-access-token"
                )
            elif isinstance(credential, BasicAuthCredential):
                await self.keyvault_manager.delete_secret(
                    f"{credential.credential_id}-password"
                )
                
        except Exception as e:
            logger.error(f"Failed to delete credential from KeyVault: {e}")
    
    def get_auth_metrics(self) -> Dict[str, Any]:
        """Get authentication metrics
        
        Returns:
            Authentication metrics
        """
        
        return {
            "metrics": self.auth_metrics.copy(),
            "total_credentials": len(self.credentials),
            "active_credentials": len([
                c for c in self.credentials.values()
                if c.status == CredentialStatus.ACTIVE
            ]),
            "expired_credentials": len([
                c for c in self.credentials.values()
                if c.is_expired()
            ]),
            "tenants": len(self.tenant_credentials),
            "keyvault_enabled": self.enable_keyvault
        }
    
    async def cleanup_expired_credentials(self) -> Dict[str, Any]:
        """Clean up expired credentials
        
        Returns:
            Cleanup result
        """
        
        cleanup_result = {
            "cleaned_up": 0,
            "errors": 0,
            "credential_ids": []
        }
        
        expired_credentials = [
            cred_id for cred_id, cred in self.credentials.items()
            if cred.is_expired()
        ]
        
        for cred_id in expired_credentials:
            try:
                success = await self.delete_credential(cred_id)
                if success:
                    cleanup_result["cleaned_up"] += 1
                    cleanup_result["credential_ids"].append(cred_id)
                else:
                    cleanup_result["errors"] += 1
            except Exception as e:
                logger.error(f"Failed to cleanup credential {cred_id}: {e}")
                cleanup_result["errors"] += 1
        
        # Update metrics
        self.auth_metrics["credentials_expired"] += cleanup_result["cleaned_up"]
        
        logger.info(f"Cleaned up {cleanup_result['cleaned_up']} expired credentials")
        return cleanup_result
```
