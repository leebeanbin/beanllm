"""
Encryption Module - 민감 데이터 암호화

Fernet 대칭 암호화를 사용하여 민감한 데이터를 안전하게 저장합니다.
API 키, 비밀번호, 토큰 등을 암호화하여 파일 시스템에 저장할 때 사용합니다.
"""

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
except ImportError:
    raise ImportError(
        "cryptography is required for encryption. "
        "Install it with: pip install cryptography"
    )


class SecureStorage:
    """
    안전한 암호화 스토리지

    Features:
        - Fernet 대칭 암호화 (AES 128-bit)
        - PBKDF2 키 유도 함수
        - 파일 기반 저장소
        - JSON 직렬화 지원
        - 키 회전 지원

    Security:
        - Fernet: AES 128-bit CBC mode with HMAC authentication
        - PBKDF2: 100,000 iterations for key derivation
        - Salt: Random 16-byte salt for each password

    Example:
        ```python
        from beanllm.infrastructure.security import SecureStorage

        # 초기화 (비밀번호 기반)
        storage = SecureStorage.from_password("my-secret-password")

        # 데이터 암호화 및 저장
        storage.set("api_key", "sk-1234567890")
        storage.set("database_password", "super-secret")

        # 데이터 복호화 및 조회
        api_key = storage.get("api_key")
        print(api_key)  # "sk-1234567890"

        # 파일로 저장
        storage.save("secrets.enc")

        # 파일에서 로드
        loaded = SecureStorage.load("secrets.enc", "my-secret-password")
        print(loaded.get("api_key"))  # "sk-1234567890"
        ```
    """

    def __init__(self, key: bytes):
        """
        Args:
            key: Fernet 암호화 키 (32 bytes, base64 encoded)
        """
        self.fernet = Fernet(key)
        self._data: Dict[str, str] = {}

    @classmethod
    def generate_key(cls) -> bytes:
        """
        새로운 Fernet 키 생성

        Returns:
            32-byte Fernet key (base64 encoded)
        """
        return Fernet.generate_key()

    @classmethod
    def from_password(
        cls, password: str, salt: Optional[bytes] = None, iterations: int = 100000
    ) -> "SecureStorage":
        """
        비밀번호로부터 SecureStorage 생성

        PBKDF2를 사용하여 비밀번호로부터 암호화 키를 유도합니다.

        Args:
            password: 비밀번호
            salt: Salt (None이면 자동 생성, 16 bytes)
            iterations: PBKDF2 반복 횟수 (기본: 100,000)

        Returns:
            SecureStorage 인스턴스

        Example:
            >>> storage = SecureStorage.from_password("my-password")
        """
        if salt is None:
            salt = os.urandom(16)

        # PBKDF2로 키 유도
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))

        instance = cls(key)
        instance._salt = salt
        instance._iterations = iterations
        return instance

    def encrypt(self, data: str) -> bytes:
        """
        데이터 암호화

        Args:
            data: 암호화할 문자열

        Returns:
            암호화된 데이터 (bytes)
        """
        return self.fernet.encrypt(data.encode())

    def decrypt(self, encrypted_data: bytes) -> str:
        """
        데이터 복호화

        Args:
            encrypted_data: 암호화된 데이터

        Returns:
            복호화된 문자열

        Raises:
            cryptography.fernet.InvalidToken: 복호화 실패 (잘못된 키 또는 데이터)
        """
        return self.fernet.decrypt(encrypted_data).decode()

    def set(self, key: str, value: Any) -> None:
        """
        암호화된 데이터 저장

        Args:
            key: 데이터 키
            value: 저장할 값 (JSON 직렬화 가능해야 함)

        Example:
            >>> storage.set("api_key", "sk-123")
            >>> storage.set("config", {"host": "localhost", "port": 5432})
        """
        # JSON 직렬화
        json_str = json.dumps(value)

        # 암호화
        encrypted = self.encrypt(json_str)

        # Base64 인코딩하여 저장 (문자열로)
        self._data[key] = base64.urlsafe_b64encode(encrypted).decode()

    def get(self, key: str, default: Any = None) -> Any:
        """
        암호화된 데이터 조회

        Args:
            key: 데이터 키
            default: 키가 없을 때 반환할 기본값

        Returns:
            복호화된 값

        Example:
            >>> storage.get("api_key")
            'sk-123'
            >>> storage.get("missing_key", "default")
            'default'
        """
        if key not in self._data:
            return default

        try:
            # Base64 디코딩
            encrypted = base64.urlsafe_b64decode(self._data[key].encode())

            # 복호화
            json_str = self.decrypt(encrypted)

            # JSON 역직렬화
            return json.loads(json_str)

        except Exception:
            return default

    def delete(self, key: str) -> bool:
        """
        데이터 삭제

        Args:
            key: 삭제할 키

        Returns:
            삭제 성공 여부
        """
        if key in self._data:
            del self._data[key]
            return True
        return False

    def keys(self) -> list:
        """
        저장된 모든 키 반환

        Returns:
            키 리스트
        """
        return list(self._data.keys())

    def clear(self) -> None:
        """모든 데이터 삭제"""
        self._data.clear()

    def save(self, file_path: Union[str, Path], password: Optional[str] = None) -> None:
        """
        암호화된 데이터를 파일로 저장

        Args:
            file_path: 저장할 파일 경로
            password: 추가 비밀번호 (double encryption)

        Example:
            >>> storage.save("secrets.enc")
            >>> storage.save("secrets.enc", password="extra-security")
        """
        file_path = Path(file_path)

        # 데이터 준비
        payload = {
            "data": self._data,
            "salt": base64.urlsafe_b64encode(getattr(self, "_salt", os.urandom(16))).decode(),
            "iterations": getattr(self, "_iterations", 100000),
        }

        # Double encryption (선택적)
        if password:
            double_storage = SecureStorage.from_password(password)
            json_str = json.dumps(payload)
            encrypted = double_storage.encrypt(json_str)
            payload = {
                "double_encrypted": True,
                "data": base64.urlsafe_b64encode(encrypted).decode(),
            }

        # 파일 저장
        with open(file_path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(
        cls, file_path: Union[str, Path], password: str, double_password: Optional[str] = None
    ) -> "SecureStorage":
        """
        파일에서 암호화된 데이터 로드

        Args:
            file_path: 파일 경로
            password: 비밀번호
            double_password: 추가 비밀번호 (double encryption 사용 시)

        Returns:
            SecureStorage 인스턴스

        Example:
            >>> storage = SecureStorage.load("secrets.enc", "my-password")
            >>> storage = SecureStorage.load("secrets.enc", "pwd", "extra-pwd")
        """
        file_path = Path(file_path)

        # 파일 로드
        with open(file_path, "r") as f:
            payload = json.load(f)

        # Double decryption 처리
        if payload.get("double_encrypted"):
            if not double_password:
                raise ValueError("Double encryption used but no double_password provided")

            double_storage = SecureStorage.from_password(double_password)
            encrypted = base64.urlsafe_b64decode(payload["data"].encode())
            json_str = double_storage.decrypt(encrypted)
            payload = json.loads(json_str)

        # Storage 복원
        salt = base64.urlsafe_b64decode(payload["salt"].encode())
        iterations = payload["iterations"]

        instance = cls.from_password(password, salt, iterations)
        instance._data = payload["data"]

        return instance


class SecureConfigManager:
    """
    설정 파일 암호화 관리자

    환경변수 또는 설정 파일에서 민감한 정보를 안전하게 관리합니다.

    Example:
        ```python
        from beanllm.infrastructure.security import SecureConfigManager

        # 초기화
        manager = SecureConfigManager("config.enc", "my-password")

        # 설정 저장
        manager.set_config("openai_api_key", "sk-123")
        manager.set_config("database", {
            "host": "localhost",
            "password": "secret"
        })
        manager.save()

        # 설정 조회
        api_key = manager.get_config("openai_api_key")
        db_config = manager.get_config("database")
        ```
    """

    def __init__(self, config_path: Union[str, Path], password: str):
        """
        Args:
            config_path: 설정 파일 경로
            password: 암호화 비밀번호
        """
        self.config_path = Path(config_path)
        self.password = password

        # 기존 파일 로드 또는 새로 생성
        if self.config_path.exists():
            self.storage = SecureStorage.load(self.config_path, password)
        else:
            self.storage = SecureStorage.from_password(password)

    def set_config(self, key: str, value: Any) -> None:
        """설정 저장"""
        self.storage.set(key, value)

    def get_config(self, key: str, default: Any = None) -> Any:
        """설정 조회"""
        return self.storage.get(key, default)

    def delete_config(self, key: str) -> bool:
        """설정 삭제"""
        return self.storage.delete(key)

    def save(self) -> None:
        """설정 파일 저장"""
        self.storage.save(self.config_path)

    def list_keys(self) -> list:
        """모든 설정 키 반환"""
        return self.storage.keys()
