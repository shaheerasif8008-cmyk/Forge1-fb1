# forge1/backend/forge1/tests/test_authentication_manager.py
"""
Tests for Authentication Manager

Comprehensive tests for the authentication system including credential management,
Azure KeyVault integration, and multi-tenant support.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch

from forge1.auth.authentication_manager import (
    AuthenticationManager,
    APIKeyCredential,
    OAuth2Credential,
    BasicAuthCredential,
    AuthenticationType,
    CredentialStatus,
    EncryptionManager,
    AzureKeyVaultManager
)


class TestEncryptionManager:
    """Test cases for EncryptionManager"""
    
    @pytest.fixture
    def encryption_manager(self):
        """Create encryption manager for testing"""
        return EncryptionManager()
    
    def test_encrypt_decrypt(self, encryption_manager):
        """Test encryption and decryption"""
        
        original_data = "sensitive_password_123"
        
        # Encrypt data
        encrypted_data = encryption_manager.encrypt(original_data)
        assert encrypted_data != original_data
        assert len(encrypted_data) > len(original_data)
        
        # Decrypt data
        decrypted_data = encryption_manager.decrypt(encrypted_data)
        assert decrypted_data == original_data
    
    def test_generate_key(self):
        """Test key generation"""
        
        key = EncryptionManager.generate_key()
        assert isinstance(key, bytes)
        assert len(key) > 0
        
        # Keys should be different
        key2 = EncryptionManager.generate_key()
        assert key != key2


class TestAzureKeyVaultManager:
    """Test cases for AzureKeyVaultManager"""
    
    @pytest.fixture
    def keyvault_manager(self):
        """Create KeyVault manager for testing"""
        return AzureKeyVaultManager()
    
    @pytest.mark.asyncio
    async def test_store_and_get_secret(self, keyvault_manager):
        """Test storing and retrieving secrets"""
        
        secret_name = "test-secret"
        secret_value = "secret_value_123"
        
        # Store secret
        result = await keyvault_manager.store_secret(secret_name, secret_value)
        
        assert "id" in result
        assert result["name"] == secret_name
        
        # Retrieve secret
        retrieved_value = await keyvault_manager.get_secret(secret_name)
        assert retrieved_value == secret_value
        
        # Retrieve non-existent secret
        non_existent = await keyvault_manager.get_secret("non-existent")
        assert non_existent is None
    
    @pytest.mark.asyncio
    async def test_delete_secret(self, keyvault_manager):
        """Test deleting secrets"""
        
        secret_name = "test-delete-secret"
        secret_value = "delete_me"
        
        # Store secret
        await keyvault_manager.store_secret(secret_name, secret_value)
        
        # Verify it exists
        retrieved = await keyvault_manager.get_secret(secret_name)
        assert retrieved == secret_value
        
        # Delete secret
        deleted = await keyvault_manager.delete_secret(secret_name)
        assert deleted == True
        
        # Verify it's gone
        retrieved = await keyvault_manager.get_secret(secret_name)
        assert retrieved is None
        
        # Delete non-existent secret
        deleted = await keyvault_manager.delete_secret("non-existent")
        assert deleted == False


class TestCredentialClasses:
    """Test cases for credential classes"""
    
    def test_api_key_credential(self):
        """Test API key credential"""
        
        credential = APIKeyCredential(
            credential_id="test_id",
            name="Test API Key",
            api_key="test_api_key_123"
        )
        
        assert credential.credential_id == "test_id"
        assert credential.name == "Test API Key"
        assert credential.auth_type == AuthenticationType.API_KEY
        assert credential.api_key == "test_api_key_123"
        assert credential.is_valid() == True
        assert credential.is_expired() == False
        
        # Test to_dict
        cred_dict = credential.to_dict()
        assert cred_dict["credential_id"] == "test_id"
        assert cred_dict["auth_type"] == "api_key"
        assert cred_dict["api_key"] == "test_api_key_123"
    
    def test_oauth2_credential(self):
        """Test OAuth2 credential"""
        
        credential = OAuth2Credential(
            credential_id="oauth_id",
            name="Test OAuth2",
            client_id="client_123",
            client_secret="secret_456",
            access_token="access_789",
            refresh_token="refresh_abc"
        )
        
        assert credential.credential_id == "oauth_id"
        assert credential.auth_type == AuthenticationType.OAUTH2
        assert credential.client_id == "client_123"
        assert credential.client_secret == "secret_456"
        assert credential.access_token == "access_789"
        assert credential.refresh_token == "refresh_abc"
        assert credential.is_valid() == True
    
    def test_basic_auth_credential(self):
        """Test Basic Auth credential"""
        
        credential = BasicAuthCredential(
            credential_id="basic_id",
            name="Test Basic Auth",
            username="testuser",
            password="testpass"
        )
        
        assert credential.credential_id == "basic_id"
        assert credential.auth_type == AuthenticationType.BASIC_AUTH
        assert credential.username == "testuser"
        assert credential.password == "testpass"
        assert credential.is_valid() == True
    
    def test_credential_expiration(self):
        """Test credential expiration"""
        
        # Create expired credential
        expired_time = datetime.now(timezone.utc) - timedelta(hours=1)
        credential = APIKeyCredential(
            credential_id="expired_id",
            name="Expired Credential",
            api_key="expired_key",
            expires_at=expired_time
        )
        
        assert credential.is_expired() == True
        assert credential.is_valid() == False  # Should be invalid due to expiration
        
        # Create future expiration
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        credential.expires_at = future_time
        
        assert credential.is_expired() == False
        assert credential.is_valid() == True


class TestAuthenticationManager:
    """Test cases for AuthenticationManager"""
    
    @pytest.fixture
    def auth_manager(self):
        """Create authentication manager for testing"""
        return AuthenticationManager()
    
    @pytest.fixture
    def auth_manager_with_keyvault(self):
        """Create authentication manager with KeyVault enabled"""
        return AuthenticationManager(enable_keyvault=True)
    
    @pytest.mark.asyncio
    async def test_store_api_key_credential(self, auth_manager):
        """Test storing API key credential"""
        
        credential_data = {
            "api_key": "test_api_key_123"
        }
        
        credential_id = await auth_manager.store_credential(
            name="Test API Key",
            auth_type=AuthenticationType.API_KEY,
            credential_data=credential_data,
            tenant_id="tenant_1"
        )
        
        assert credential_id.startswith("cred_")
        assert credential_id in auth_manager.credentials
        
        # Verify credential
        credential = auth_manager.credentials[credential_id]
        assert isinstance(credential, APIKeyCredential)
        assert credential.name == "Test API Key"
        assert credential.api_key == "test_api_key_123"
        assert credential.tenant_id == "tenant_1"
        
        # Check metrics
        assert auth_manager.auth_metrics["credentials_stored"] == 1
    
    @pytest.mark.asyncio
    async def test_store_oauth2_credential(self, auth_manager):
        """Test storing OAuth2 credential"""
        
        credential_data = {
            "client_id": "client_123",
            "client_secret": "secret_456",
            "access_token": "access_789",
            "refresh_token": "refresh_abc",
            "token_url": "https://oauth.example.com/token"
        }
        
        credential_id = await auth_manager.store_credential(
            name="Test OAuth2",
            auth_type=AuthenticationType.OAUTH2,
            credential_data=credential_data
        )
        
        credential = auth_manager.credentials[credential_id]
        assert isinstance(credential, OAuth2Credential)
        assert credential.client_id == "client_123"
        assert credential.client_secret == "secret_456"
        assert credential.token_url == "https://oauth.example.com/token"
    
    @pytest.mark.asyncio
    async def test_store_basic_auth_credential(self, auth_manager):
        """Test storing Basic Auth credential"""
        
        credential_data = {
            "username": "testuser",
            "password": "testpass"
        }
        
        credential_id = await auth_manager.store_credential(
            name="Test Basic Auth",
            auth_type=AuthenticationType.BASIC_AUTH,
            credential_data=credential_data
        )
        
        credential = auth_manager.credentials[credential_id]
        assert isinstance(credential, BasicAuthCredential)
        assert credential.username == "testuser"
        assert credential.password == "testpass"
    
    @pytest.mark.asyncio
    async def test_get_credential(self, auth_manager):
        """Test retrieving credential"""
        
        # Store credential
        credential_data = {"api_key": "test_key"}
        credential_id = await auth_manager.store_credential(
            name="Test Credential",
            auth_type=AuthenticationType.API_KEY,
            credential_data=credential_data
        )
        
        # Retrieve credential
        retrieved = await auth_manager.get_credential(credential_id)
        
        assert retrieved is not None
        assert retrieved.credential_id == credential_id
        assert retrieved.name == "Test Credential"
        
        # Try to get non-existent credential
        non_existent = await auth_manager.get_credential("non_existent")
        assert non_existent is None
    
    @pytest.mark.asyncio
    async def test_authenticate_api_key(self, auth_manager):
        """Test API key authentication"""
        
        # Store API key credential
        credential_data = {"api_key": "test_api_key_123"}
        credential_id = await auth_manager.store_credential(
            name="Test API Key",
            auth_type=AuthenticationType.API_KEY,
            credential_data=credential_data
        )
        
        # Authenticate
        result = await auth_manager.authenticate(credential_id)
        
        assert result["success"] == True
        assert result["auth_method"] == "api_key"
        assert result["credential_id"] == credential_id
        assert result["auth_type"] == "api_key"
        assert "headers" in result
        assert "Authorization" in result["headers"]
        
        # Check metrics
        assert auth_manager.auth_metrics["authentications_performed"] == 1
        assert auth_manager.auth_metrics["successful_authentications"] == 1
    
    @pytest.mark.asyncio
    async def test_authenticate_oauth2(self, auth_manager):
        """Test OAuth2 authentication"""
        
        # Store OAuth2 credential
        credential_data = {
            "client_id": "client_123",
            "client_secret": "secret_456",
            "access_token": "access_789"
        }
        credential_id = await auth_manager.store_credential(
            name="Test OAuth2",
            auth_type=AuthenticationType.OAUTH2,
            credential_data=credential_data
        )
        
        # Authenticate
        result = await auth_manager.authenticate(credential_id)
        
        assert result["success"] == True
        assert result["auth_method"] == "oauth2"
        assert result["token_type"] == "Bearer"
        assert result["access_token"] == "access_789"
    
    @pytest.mark.asyncio
    async def test_authenticate_basic_auth(self, auth_manager):
        """Test Basic Auth authentication"""
        
        # Store Basic Auth credential
        credential_data = {
            "username": "testuser",
            "password": "testpass"
        }
        credential_id = await auth_manager.store_credential(
            name="Test Basic Auth",
            auth_type=AuthenticationType.BASIC_AUTH,
            credential_data=credential_data
        )
        
        # Authenticate
        result = await auth_manager.authenticate(credential_id)
        
        assert result["success"] == True
        assert result["auth_method"] == "basic_auth"
        assert result["username"] == "testuser"
        assert "Authorization" in result["headers"]
        assert result["headers"]["Authorization"].startswith("Basic ")
    
    @pytest.mark.asyncio
    async def test_authenticate_invalid_credential(self, auth_manager):
        """Test authentication with invalid credential"""
        
        # Try to authenticate with non-existent credential
        result = await auth_manager.authenticate("non_existent")
        
        assert result["success"] == False
        assert "not found" in result["error"]
        assert result["credential_id"] == "non_existent"
        
        # Check metrics
        assert auth_manager.auth_metrics["failed_authentications"] == 1
    
    @pytest.mark.asyncio
    async def test_authenticate_expired_credential(self, auth_manager):
        """Test authentication with expired credential"""
        
        # Store expired credential
        expired_time = datetime.now(timezone.utc) - timedelta(hours=1)
        credential_data = {"api_key": "expired_key"}
        credential_id = await auth_manager.store_credential(
            name="Expired Credential",
            auth_type=AuthenticationType.API_KEY,
            credential_data=credential_data,
            expires_at=expired_time
        )
        
        # Try to authenticate
        result = await auth_manager.authenticate(credential_id)
        
        assert result["success"] == False
        assert "invalid or expired" in result["error"]
    
    def test_list_credentials(self, auth_manager):
        """Test listing credentials"""
        
        # Initially empty
        credentials = auth_manager.list_credentials()
        assert len(credentials) == 0
        
        # Store some credentials
        asyncio.run(auth_manager.store_credential(
            name="Credential 1",
            auth_type=AuthenticationType.API_KEY,
            credential_data={"api_key": "key1"},
            tenant_id="tenant_1"
        ))
        
        asyncio.run(auth_manager.store_credential(
            name="Credential 2",
            auth_type=AuthenticationType.OAUTH2,
            credential_data={"client_id": "client", "client_secret": "secret"},
            tenant_id="tenant_2"
        ))
        
        # List all credentials
        all_credentials = auth_manager.list_credentials()
        assert len(all_credentials) == 2
        
        # List credentials for specific tenant
        tenant1_credentials = auth_manager.list_credentials(tenant_id="tenant_1")
        assert len(tenant1_credentials) == 1
        assert tenant1_credentials[0]["name"] == "Credential 1"
        
        # Check credential info structure
        cred_info = all_credentials[0]
        assert "credential_id" in cred_info
        assert "name" in cred_info
        assert "auth_type" in cred_info
        assert "status" in cred_info
        assert "is_valid" in cred_info
        assert "is_expired" in cred_info
    
    @pytest.mark.asyncio
    async def test_delete_credential(self, auth_manager):
        """Test deleting credential"""
        
        # Store credential
        credential_data = {"api_key": "test_key"}
        credential_id = await auth_manager.store_credential(
            name="Test Credential",
            auth_type=AuthenticationType.API_KEY,
            credential_data=credential_data,
            tenant_id="tenant_1"
        )
        
        # Verify it exists
        assert credential_id in auth_manager.credentials
        assert "tenant_1" in auth_manager.tenant_credentials
        assert credential_id in auth_manager.tenant_credentials["tenant_1"]
        
        # Delete credential
        deleted = await auth_manager.delete_credential(credential_id)
        
        assert deleted == True
        assert credential_id not in auth_manager.credentials
        assert credential_id not in auth_manager.tenant_credentials["tenant_1"]
        
        # Try to delete non-existent credential
        deleted = await auth_manager.delete_credential("non_existent")
        assert deleted == False
    
    def test_get_auth_metrics(self, auth_manager):
        """Test getting authentication metrics"""
        
        metrics = auth_manager.get_auth_metrics()
        
        assert "metrics" in metrics
        assert "total_credentials" in metrics
        assert "active_credentials" in metrics
        assert "expired_credentials" in metrics
        assert "tenants" in metrics
        assert "keyvault_enabled" in metrics
        
        # Initially should be zero
        assert metrics["total_credentials"] == 0
        assert metrics["active_credentials"] == 0
        assert metrics["keyvault_enabled"] == False
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_credentials(self, auth_manager):
        """Test cleaning up expired credentials"""
        
        # Store active credential
        active_data = {"api_key": "active_key"}
        active_id = await auth_manager.store_credential(
            name="Active Credential",
            auth_type=AuthenticationType.API_KEY,
            credential_data=active_data
        )
        
        # Store expired credential
        expired_time = datetime.now(timezone.utc) - timedelta(hours=1)
        expired_data = {"api_key": "expired_key"}
        expired_id = await auth_manager.store_credential(
            name="Expired Credential",
            auth_type=AuthenticationType.API_KEY,
            credential_data=expired_data,
            expires_at=expired_time
        )
        
        # Verify both exist
        assert len(auth_manager.credentials) == 2
        
        # Cleanup expired credentials
        result = await auth_manager.cleanup_expired_credentials()
        
        assert result["cleaned_up"] == 1
        assert result["errors"] == 0
        assert expired_id in result["credential_ids"]
        
        # Verify only active credential remains
        assert len(auth_manager.credentials) == 1
        assert active_id in auth_manager.credentials
        assert expired_id not in auth_manager.credentials
    
    @pytest.mark.asyncio
    async def test_keyvault_integration(self, auth_manager_with_keyvault):
        """Test KeyVault integration"""
        
        auth_manager = auth_manager_with_keyvault
        
        # Store credential with KeyVault
        credential_data = {"api_key": "secret_key_123"}
        credential_id = await auth_manager.store_credential(
            name="KeyVault Test",
            auth_type=AuthenticationType.API_KEY,
            credential_data=credential_data,
            use_keyvault=True
        )
        
        # Verify credential was stored
        assert credential_id in auth_manager.credentials
        
        # Verify KeyVault was used (mock implementation)
        assert auth_manager.keyvault_manager is not None
        
        # Retrieve credential (should load from KeyVault)
        retrieved = await auth_manager.get_credential(credential_id)
        assert retrieved is not None
        assert retrieved.api_key == "secret_key_123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])