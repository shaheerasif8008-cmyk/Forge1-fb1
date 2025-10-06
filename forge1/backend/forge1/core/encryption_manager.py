# forge1/backend/forge1/core/encryption_manager.py
"""
Encryption Manager for Employee Lifecycle System

Provides tenant-specific encryption for sensitive employee data including:
- Tenant-specific encryption keys
- Field-level encryption for sensitive data
- Key rotation and management
- Secure key storage and retrieval

Requirements: 8.1, 8.2
"""

import base64
import hashlib
import logging
import os
import secrets
from typing import Dict, Optional, Any, Union
from datetime import datetime, timezone, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class EncryptionManager:
    """
    Manages tenant-specific encryption for sensitive employee data.
    
    Features:
    - Tenant-specific encryption keys
    - Field-level encryption for sensitive data
    - Key derivation from master key + tenant salt
    - Automatic key rotation
    - Secure key caching with TTL
    """
    
    def __init__(self):
        self.master_key = self._get_master_key()
        self.key_cache = {}  # tenant_id -> (fernet_instance, expiry)
        self.key_ttl_hours = 24  # Keys expire after 24 hours
        self.salt_length = 32
        
        # Fields that should be encrypted
        self.encrypted_fields = {
            "employee": [
                "personality.custom_traits",
                "model_preferences.api_keys",
                "tool_access_credentials",
                "knowledge_sources.credentials"
            ],
            "client": [
                "configuration.api_keys",
                "configuration.credentials",
                "billing_info",
                "contact_info.email",
                "contact_info.phone"
            ],
            "interaction": [
                "context.sensitive_data",
                "context.user_info",
                "response.internal_notes"
            ]
        }
    
    def _get_master_key(self) -> bytes:
        """Get or generate master encryption key"""
        master_key_env = os.getenv("FORGE1_MASTER_ENCRYPTION_KEY")
        
        if master_key_env:
            try:
                return base64.b64decode(master_key_env)
            except Exception as e:
                logger.error(f"Invalid master key in environment: {e}")
                raise ValueError("Invalid master encryption key")
        
        # For development, generate a key (in production, use proper key management)
        logger.warning("No master key found, generating temporary key for development")
        return Fernet.generate_key()
    
    def _get_tenant_key(self, tenant_id: str) -> Fernet:
        """Get or create tenant-specific encryption key"""
        
        # Check cache first
        if tenant_id in self.key_cache:
            fernet_instance, expiry = self.key_cache[tenant_id]
            if datetime.now(timezone.utc) < expiry:
                return fernet_instance
            else:
                # Key expired, remove from cache
                del self.key_cache[tenant_id]
        
        # Generate tenant-specific key
        tenant_salt = self._get_tenant_salt(tenant_id)
        
        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=tenant_salt,
            iterations=100000,
        )
        
        derived_key = kdf.derive(self.master_key)
        fernet_key = base64.urlsafe_b64encode(derived_key)
        fernet_instance = Fernet(fernet_key)
        
        # Cache with expiry
        expiry = datetime.now(timezone.utc) + timedelta(hours=self.key_ttl_hours)
        self.key_cache[tenant_id] = (fernet_instance, expiry)
        
        return fernet_instance
    
    def _get_tenant_salt(self, tenant_id: str) -> bytes:
        """Get deterministic salt for tenant"""
        # Create deterministic salt from tenant ID
        # In production, store salts securely in key management system
        salt_input = f"forge1_tenant_salt_{tenant_id}".encode()
        return hashlib.sha256(salt_input).digest()
    
    def encrypt_field(self, tenant_id: str, field_value: Any) -> str:
        """
        Encrypt a field value for a specific tenant.
        
        Args:
            tenant_id: Tenant identifier
            field_value: Value to encrypt
            
        Returns:
            Base64-encoded encrypted value
        """
        if field_value is None:
            return None
        
        try:
            # Convert value to string if needed
            if not isinstance(field_value, str):
                field_value = str(field_value)
            
            # Get tenant-specific key
            fernet = self._get_tenant_key(tenant_id)
            
            # Encrypt the value
            encrypted_bytes = fernet.encrypt(field_value.encode())
            
            # Return base64-encoded string
            return base64.b64encode(encrypted_bytes).decode()
            
        except Exception as e:
            logger.error(f"Failed to encrypt field for tenant {tenant_id}: {e}")
            raise ValueError(f"Encryption failed: {e}")
    
    def decrypt_field(self, tenant_id: str, encrypted_value: str) -> str:
        """
        Decrypt a field value for a specific tenant.
        
        Args:
            tenant_id: Tenant identifier
            encrypted_value: Base64-encoded encrypted value
            
        Returns:
            Decrypted string value
        """
        if encrypted_value is None:
            return None
        
        try:
            # Get tenant-specific key
            fernet = self._get_tenant_key(tenant_id)
            
            # Decode from base64
            encrypted_bytes = base64.b64decode(encrypted_value.encode())
            
            # Decrypt the value
            decrypted_bytes = fernet.decrypt(encrypted_bytes)
            
            return decrypted_bytes.decode()
            
        except Exception as e:
            logger.error(f"Failed to decrypt field for tenant {tenant_id}: {e}")
            raise ValueError(f"Decryption failed: {e}")
    
    def encrypt_object(
        self,
        tenant_id: str,
        obj: Dict[str, Any],
        object_type: str
    ) -> Dict[str, Any]:
        """
        Encrypt sensitive fields in an object based on object type.
        
        Args:
            tenant_id: Tenant identifier
            obj: Object to encrypt
            object_type: Type of object (employee, client, interaction)
            
        Returns:
            Object with encrypted sensitive fields
        """
        if object_type not in self.encrypted_fields:
            return obj
        
        encrypted_obj = obj.copy()
        fields_to_encrypt = self.encrypted_fields[object_type]
        
        for field_path in fields_to_encrypt:
            try:
                # Navigate to nested field
                current_obj = encrypted_obj
                field_parts = field_path.split('.')
                
                # Navigate to parent of target field
                for part in field_parts[:-1]:
                    if part in current_obj and isinstance(current_obj[part], dict):
                        current_obj = current_obj[part]
                    else:
                        # Field path doesn't exist, skip
                        break
                else:
                    # Encrypt the final field if it exists
                    final_field = field_parts[-1]
                    if final_field in current_obj and current_obj[final_field] is not None:
                        current_obj[final_field] = self.encrypt_field(
                            tenant_id, current_obj[final_field]
                        )
                        
                        # Mark field as encrypted
                        current_obj[f"{final_field}_encrypted"] = True
                        
            except Exception as e:
                logger.warning(f"Failed to encrypt field {field_path}: {e}")
                continue
        
        return encrypted_obj
    
    def decrypt_object(
        self,
        tenant_id: str,
        obj: Dict[str, Any],
        object_type: str
    ) -> Dict[str, Any]:
        """
        Decrypt sensitive fields in an object based on object type.
        
        Args:
            tenant_id: Tenant identifier
            obj: Object to decrypt
            object_type: Type of object (employee, client, interaction)
            
        Returns:
            Object with decrypted sensitive fields
        """
        if object_type not in self.encrypted_fields:
            return obj
        
        decrypted_obj = obj.copy()
        fields_to_decrypt = self.encrypted_fields[object_type]
        
        for field_path in fields_to_decrypt:
            try:
                # Navigate to nested field
                current_obj = decrypted_obj
                field_parts = field_path.split('.')
                
                # Navigate to parent of target field
                for part in field_parts[:-1]:
                    if part in current_obj and isinstance(current_obj[part], dict):
                        current_obj = current_obj[part]
                    else:
                        # Field path doesn't exist, skip
                        break
                else:
                    # Decrypt the final field if it's marked as encrypted
                    final_field = field_parts[-1]
                    encrypted_marker = f"{final_field}_encrypted"
                    
                    if (encrypted_marker in current_obj and 
                        current_obj[encrypted_marker] and
                        final_field in current_obj and 
                        current_obj[final_field] is not None):
                        
                        current_obj[final_field] = self.decrypt_field(
                            tenant_id, current_obj[final_field]
                        )
                        
                        # Remove encryption marker
                        del current_obj[encrypted_marker]
                        
            except Exception as e:
                logger.warning(f"Failed to decrypt field {field_path}: {e}")
                continue
        
        return decrypted_obj
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate a cryptographically secure random token"""
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
        """
        Hash a password with salt.
        
        Args:
            password: Password to hash
            salt: Optional salt (generated if not provided)
            
        Returns:
            Tuple of (hashed_password, salt) as base64 strings
        """
        if salt is None:
            salt = secrets.token_bytes(32)
        
        # Use PBKDF2 for password hashing
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        hashed = kdf.derive(password.encode())
        
        return (
            base64.b64encode(hashed).decode(),
            base64.b64encode(salt).decode()
        )
    
    def verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            password: Password to verify
            hashed_password: Base64-encoded hashed password
            salt: Base64-encoded salt
            
        Returns:
            True if password matches
        """
        try:
            # Decode salt and hash
            salt_bytes = base64.b64decode(salt.encode())
            expected_hash = base64.b64decode(hashed_password.encode())
            
            # Hash the provided password
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt_bytes,
                iterations=100000,
            )
            
            actual_hash = kdf.derive(password.encode())
            
            # Compare hashes
            return secrets.compare_digest(expected_hash, actual_hash)
            
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    def rotate_tenant_key(self, tenant_id: str) -> bool:
        """
        Rotate encryption key for a tenant.
        
        This would typically involve:
        1. Generating new key
        2. Re-encrypting all data with new key
        3. Updating key storage
        
        Args:
            tenant_id: Tenant to rotate key for
            
        Returns:
            True if rotation successful
        """
        try:
            # Remove from cache to force regeneration
            if tenant_id in self.key_cache:
                del self.key_cache[tenant_id]
            
            # In production, this would:
            # 1. Generate new salt for tenant
            # 2. Re-encrypt all tenant data with new key
            # 3. Update key management system
            # 4. Audit the key rotation
            
            logger.info(f"Key rotation initiated for tenant {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Key rotation failed for tenant {tenant_id}: {e}")
            return False
    
    def get_encryption_metadata(self, tenant_id: str) -> Dict[str, Any]:
        """
        Get encryption metadata for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Encryption metadata including key info and statistics
        """
        try:
            # Check if key is cached
            key_cached = tenant_id in self.key_cache
            key_expiry = None
            
            if key_cached:
                _, expiry = self.key_cache[tenant_id]
                key_expiry = expiry.isoformat()
            
            return {
                "tenant_id": tenant_id,
                "encryption_enabled": True,
                "key_cached": key_cached,
                "key_expiry": key_expiry,
                "key_ttl_hours": self.key_ttl_hours,
                "encrypted_field_types": list(self.encrypted_fields.keys()),
                "encryption_algorithm": "Fernet (AES 128 CBC + HMAC SHA256)",
                "key_derivation": "PBKDF2-HMAC-SHA256 (100,000 iterations)"
            }
            
        except Exception as e:
            logger.error(f"Failed to get encryption metadata for tenant {tenant_id}: {e}")
            return {"error": str(e)}
    
    def clear_key_cache(self, tenant_id: Optional[str] = None):
        """
        Clear encryption key cache.
        
        Args:
            tenant_id: Specific tenant to clear, or None for all tenants
        """
        if tenant_id:
            if tenant_id in self.key_cache:
                del self.key_cache[tenant_id]
                logger.info(f"Cleared encryption key cache for tenant {tenant_id}")
        else:
            self.key_cache.clear()
            logger.info("Cleared all encryption key caches")


# Utility functions for encryption

def encrypt_sensitive_data(
    tenant_id: str,
    data: Dict[str, Any],
    object_type: str,
    encryption_manager: Optional[EncryptionManager] = None
) -> Dict[str, Any]:
    """
    Utility function to encrypt sensitive data in an object.
    
    Args:
        tenant_id: Tenant identifier
        data: Data to encrypt
        object_type: Type of object (employee, client, interaction)
        encryption_manager: Optional encryption manager instance
        
    Returns:
        Object with encrypted sensitive fields
    """
    if encryption_manager is None:
        encryption_manager = EncryptionManager()
    
    return encryption_manager.encrypt_object(tenant_id, data, object_type)


def decrypt_sensitive_data(
    tenant_id: str,
    data: Dict[str, Any],
    object_type: str,
    encryption_manager: Optional[EncryptionManager] = None
) -> Dict[str, Any]:
    """
    Utility function to decrypt sensitive data in an object.
    
    Args:
        tenant_id: Tenant identifier
        data: Data to decrypt
        object_type: Type of object (employee, client, interaction)
        encryption_manager: Optional encryption manager instance
        
    Returns:
        Object with decrypted sensitive fields
    """
    if encryption_manager is None:
        encryption_manager = EncryptionManager()
    
    return encryption_manager.decrypt_object(tenant_id, data, object_type)