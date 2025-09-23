# forge1/backend/forge1/auth/__init__.py
"""
Forge 1 Authentication Module

Enterprise authentication system providing secure credential management,
multi-tenant support, and Azure KeyVault integration.
"""

from .authentication_manager import (
    AuthenticationManager,
    BaseCredential,
    APIKeyCredential,
    OAuth2Credential,
    BasicAuthCredential,
    AuthenticationType,
    CredentialStatus,
    EncryptionManager,
    AzureKeyVaultManager
)

__all__ = [
    "AuthenticationManager",
    "BaseCredential",
    "APIKeyCredential", 
    "OAuth2Credential",
    "BasicAuthCredential",
    "AuthenticationType",
    "CredentialStatus",
    "EncryptionManager",
    "AzureKeyVaultManager"
]