"""
API Key Encryption Service

Provides secure encryption/decryption for API keys stored in MongoDB.
Uses Fernet symmetric encryption (AES-128-CBC with HMAC).
"""

import base64
import os
import secrets
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class EncryptionService:
    """
    Service for encrypting and decrypting sensitive data like API keys.

    Usage:
        service = EncryptionService()
        encrypted = service.encrypt("sk-1234567890abcdef")
        decrypted = service.decrypt(encrypted)
    """

    _instance: Optional["EncryptionService"] = None
    _fernet: Optional[Fernet] = None

    def __new__(cls) -> "EncryptionService":
        """Singleton pattern to ensure consistent encryption key."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the Fernet cipher with the encryption key."""
        encryption_key = os.getenv("ENCRYPTION_KEY")

        if not encryption_key:
            # Generate a new key if not provided (for development)
            # In production, ENCRYPTION_KEY should always be set
            print("⚠️  ENCRYPTION_KEY not set. Generating temporary key.")
            print("   For production, set ENCRYPTION_KEY in .env")
            encryption_key = self._generate_key()
            os.environ["ENCRYPTION_KEY"] = encryption_key

        # Derive a proper Fernet key from the provided key
        self._fernet = self._create_fernet(encryption_key)

    def _generate_key(self) -> str:
        """Generate a new random encryption key."""
        return secrets.token_hex(32)

    def _create_fernet(self, key: str) -> Fernet:
        """
        Create a Fernet cipher from a hex key.

        Uses PBKDF2 to derive a proper 32-byte key for Fernet.
        """
        # Use a fixed salt (can be stored separately for extra security)
        salt = b"beanllm_encryption_salt_v1"

        # Derive a proper key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )

        # Convert hex key to bytes
        key_bytes = key.encode() if isinstance(key, str) else key
        derived_key = kdf.derive(key_bytes)

        # Fernet requires a URL-safe base64-encoded 32-byte key
        fernet_key = base64.urlsafe_b64encode(derived_key)

        return Fernet(fernet_key)

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a plaintext string.

        Args:
            plaintext: The string to encrypt (e.g., API key)

        Returns:
            Base64-encoded encrypted string
        """
        if not plaintext:
            raise ValueError("Cannot encrypt empty string")

        if not self._fernet:
            raise RuntimeError("Encryption service not initialized")

        # Encrypt and return as string
        encrypted_bytes = self._fernet.encrypt(plaintext.encode())
        return encrypted_bytes.decode()

    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt an encrypted string.

        Args:
            ciphertext: Base64-encoded encrypted string

        Returns:
            Decrypted plaintext string

        Raises:
            ValueError: If decryption fails (invalid key or corrupted data)
        """
        if not ciphertext:
            raise ValueError("Cannot decrypt empty string")

        if not self._fernet:
            raise RuntimeError("Encryption service not initialized")

        try:
            decrypted_bytes = self._fernet.decrypt(ciphertext.encode())
            return decrypted_bytes.decode()
        except InvalidToken:
            raise ValueError("Failed to decrypt: invalid key or corrupted data")

    def get_key_hint(self, api_key: str) -> str:
        """
        Get a hint for the API key (last 4 characters).

        Args:
            api_key: The full API key

        Returns:
            Last 4 characters of the key for identification
        """
        if not api_key or len(api_key) < 4:
            return "****"
        return api_key[-4:]

    def mask_key(self, api_key: str) -> str:
        """
        Mask an API key for display.

        Args:
            api_key: The full API key

        Returns:
            Masked key like "sk-****...7890"
        """
        if not api_key:
            return "****"

        if len(api_key) <= 8:
            return "****" + api_key[-4:] if len(api_key) >= 4 else "****"

        prefix = api_key[:4] if api_key.startswith(("sk-", "sk_")) else api_key[:2]
        suffix = api_key[-4:]
        return f"{prefix}****...{suffix}"

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None
        cls._fernet = None


# Singleton instance
encryption_service = EncryptionService()
