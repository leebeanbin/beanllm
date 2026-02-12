"""
Google OAuth 2.0 Service

Google 서비스 인증을 위한 OAuth 2.0 플로우 관리:
- Authorization URL 생성
- 콜백 처리 및 토큰 교환
- 토큰 저장 (MongoDB, 암호화)
- 토큰 갱신
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from services.encryption_service import encryption_service

logger = logging.getLogger(__name__)

# Google OAuth 설정
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv(
    "GOOGLE_OAUTH_REDIRECT_URI", "http://localhost:8000/api/auth/google/callback"
)

# Google OAuth 스코프 (각 서비스별)
GOOGLE_SCOPES = {
    "drive": [
        "https://www.googleapis.com/auth/drive.file",  # 앱이 생성한 파일만
        "https://www.googleapis.com/auth/drive.readonly",  # 읽기 전용
    ],
    "docs": [
        "https://www.googleapis.com/auth/documents",  # Docs 읽기/쓰기
    ],
    "gmail": [
        "https://www.googleapis.com/auth/gmail.send",  # 이메일 전송
        "https://www.googleapis.com/auth/gmail.readonly",  # 이메일 읽기
    ],
    "calendar": [
        "https://www.googleapis.com/auth/calendar",  # 캘린더 읽기/쓰기
        "https://www.googleapis.com/auth/calendar.events",  # 이벤트 관리
    ],
    "sheets": [
        "https://www.googleapis.com/auth/spreadsheets",  # Sheets 읽기/쓰기
    ],
}

# 기본 스코프 (사용자 정보)
DEFAULT_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]


class GoogleOAuthService:
    """
    Google OAuth 2.0 서비스

    Usage:
        oauth = GoogleOAuthService()
        # 또는 DB/설정에서 읽은 값으로:
        oauth = GoogleOAuthService(client_id="...", client_secret="...", redirect_uri="...")

        # 1. 인증 URL 생성
        auth_url = oauth.get_authorization_url(["drive", "docs"])

        # 2. 콜백 처리
        tokens = await oauth.handle_callback(code, db)

        # 3. 토큰 사용
        access_token = await oauth.get_valid_access_token(user_id, db)
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
    ):
        self._client_id = (client_id or "").strip() or GOOGLE_CLIENT_ID
        self._client_secret = (client_secret or "").strip() or GOOGLE_CLIENT_SECRET
        self._redirect_uri = (redirect_uri or "").strip() or GOOGLE_REDIRECT_URI

    @property
    def is_configured(self) -> bool:
        """OAuth가 설정되어 있는지 확인"""
        return bool(self._client_id and self._client_secret)

    def get_authorization_url(
        self,
        services: List[str] = None,
        user_id: str = "default",
        extra_scopes: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Google OAuth 인증 URL 생성

        Args:
            services: 요청할 서비스 목록 (drive, docs, gmail, calendar, sheets)
            user_id: 사용자 ID (state에 포함)
            extra_scopes: 추가 스코프

        Returns:
            {"auth_url": "...", "state": "..."}
        """
        if not self.is_configured:
            raise ValueError(
                "Google OAuth is not configured. Set GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET."
            )

        # 스코프 수집
        scopes = list(DEFAULT_SCOPES)

        if services:
            for service in services:
                if service in GOOGLE_SCOPES:
                    scopes.extend(GOOGLE_SCOPES[service])

        if extra_scopes:
            scopes.extend(extra_scopes)

        # 중복 제거
        scopes = list(set(scopes))

        # State 생성 (CSRF 방지 + 사용자 ID)
        import secrets

        state = f"{user_id}:{secrets.token_urlsafe(16)}"

        # URL 생성
        import urllib.parse

        params = {
            "client_id": self._client_id,
            "redirect_uri": self._redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
            "access_type": "offline",  # refresh_token 받기 위해
            "prompt": "consent",  # 항상 동의 화면 표시 (refresh_token 보장)
            "state": state,
        }

        auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"

        logger.info(f"Generated Google OAuth URL for user: {user_id}, services: {services}")

        return {
            "auth_url": auth_url,
            "state": state,
            "scopes": scopes,
            "services": services or [],
        }

    async def handle_callback(
        self,
        code: str,
        state: str,
        db,
    ) -> Dict[str, Any]:
        """
        OAuth 콜백 처리 및 토큰 교환

        Args:
            code: Authorization code
            state: State 파라미터
            db: MongoDB 데이터베이스

        Returns:
            {"success": True, "user_id": "...", "scopes": [...]}
        """
        if not self.is_configured:
            raise ValueError("Google OAuth is not configured")

        try:
            from utils.http_client import get_http_client

            # State에서 user_id 추출
            user_id = state.split(":")[0] if ":" in state else "default"

            # 토큰 교환
            client = get_http_client()
            response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": self._redirect_uri,
                },
            )

            if response.status_code != 200:
                error_data = response.json()
                raise ValueError(f"Token exchange failed: {error_data}")

            token_data = response.json()

            # 토큰 정보 추출
            access_token = token_data["access_token"]
            refresh_token = token_data.get("refresh_token")
            expires_in = token_data.get("expires_in", 3600)
            scope = token_data.get("scope", "")

            # 만료 시간 계산
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

            # 토큰 암호화
            access_token_encrypted = encryption_service.encrypt(access_token)
            refresh_token_encrypted = (
                encryption_service.encrypt(refresh_token) if refresh_token else None
            )

            # MongoDB에 저장 (upsert)
            now = datetime.now(timezone.utc)
            await db.google_oauth_tokens.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "access_token_encrypted": access_token_encrypted,
                        "refresh_token_encrypted": refresh_token_encrypted,
                        "token_type": token_data.get("token_type", "Bearer"),
                        "expires_at": expires_at,
                        "scopes": scope.split(" ") if scope else [],
                        "updated_at": now,
                    },
                    "$setOnInsert": {
                        "created_at": now,
                    },
                },
                upsert=True,
            )

            logger.info(f"Saved Google OAuth tokens for user: {user_id}")

            return {
                "success": True,
                "user_id": user_id,
                "scopes": scope.split(" ") if scope else [],
                "expires_at": expires_at.isoformat(),
            }

        except Exception as e:
            logger.error(f"OAuth callback failed: {e}")
            raise

    async def get_valid_access_token(
        self,
        user_id: str,
        db,
    ) -> Optional[str]:
        """
        유효한 액세스 토큰 가져오기 (필요시 자동 갱신)

        Args:
            user_id: 사용자 ID
            db: MongoDB 데이터베이스

        Returns:
            액세스 토큰 또는 None
        """
        try:
            # MongoDB에서 토큰 조회
            token_doc = await db.google_oauth_tokens.find_one({"user_id": user_id})

            if not token_doc:
                logger.warning(f"No OAuth token found for user: {user_id}")
                return None

            expires_at = token_doc.get("expires_at")
            now = datetime.now(timezone.utc)

            # 만료 10분 전이면 갱신
            if expires_at and (expires_at - now) < timedelta(minutes=10):
                refresh_token_encrypted = token_doc.get("refresh_token_encrypted")

                if refresh_token_encrypted:
                    try:
                        await self._refresh_token(user_id, refresh_token_encrypted, db)
                        # 갱신 후 다시 조회
                        token_doc = await db.google_oauth_tokens.find_one({"user_id": user_id})
                    except Exception as e:
                        logger.error(f"Token refresh failed: {e}")
                        return None
                else:
                    logger.warning(
                        f"Token expired and no refresh token available for user: {user_id}"
                    )
                    return None

            # 복호화하여 반환
            access_token_encrypted = token_doc.get("access_token_encrypted")
            if access_token_encrypted:
                return encryption_service.decrypt(access_token_encrypted)

            return None

        except Exception as e:
            logger.error(f"Failed to get access token: {e}")
            return None

    async def _refresh_token(
        self,
        user_id: str,
        refresh_token_encrypted: str,
        db,
    ):
        """토큰 갱신"""
        from utils.http_client import get_http_client

        refresh_token = encryption_service.decrypt(refresh_token_encrypted)

        client = get_http_client()
        response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
        )

        if response.status_code != 200:
            error_data = response.json()
            raise ValueError(f"Token refresh failed: {error_data}")

        token_data = response.json()

        # 새 토큰 저장
        access_token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 3600)
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

        access_token_encrypted = encryption_service.encrypt(access_token)

        # 새 refresh_token이 있으면 저장 (Google은 가끔 새 refresh_token 발급)
        new_refresh_token = token_data.get("refresh_token")
        update_data = {
            "access_token_encrypted": access_token_encrypted,
            "expires_at": expires_at,
            "updated_at": datetime.now(timezone.utc),
        }

        if new_refresh_token:
            update_data["refresh_token_encrypted"] = encryption_service.encrypt(new_refresh_token)

        await db.google_oauth_tokens.update_one(
            {"user_id": user_id},
            {"$set": update_data},
        )

        logger.info(f"Refreshed Google OAuth token for user: {user_id}")

    async def get_auth_status(
        self,
        user_id: str,
        db,
    ) -> Dict[str, Any]:
        """
        인증 상태 확인

        Args:
            user_id: 사용자 ID
            db: MongoDB 데이터베이스

        Returns:
            인증 상태 정보
        """
        try:
            token_doc = await db.google_oauth_tokens.find_one({"user_id": user_id})

            if not token_doc:
                return {
                    "is_authenticated": False,
                    "scopes": [],
                    "expires_at": None,
                    "available_services": [],
                }

            expires_at = token_doc.get("expires_at")
            scopes = token_doc.get("scopes", [])

            # 만료 확인
            now = datetime.now(timezone.utc)
            is_valid = expires_at and expires_at > now

            # 사용 가능한 서비스 확인
            available_services = []
            for service, service_scopes in GOOGLE_SCOPES.items():
                if any(scope in scopes for scope in service_scopes):
                    available_services.append(service)

            return {
                "is_authenticated": is_valid,
                "scopes": scopes,
                "expires_at": expires_at.isoformat() if expires_at else None,
                "available_services": available_services,
            }

        except Exception as e:
            logger.error(f"Failed to get auth status: {e}")
            return {
                "is_authenticated": False,
                "scopes": [],
                "expires_at": None,
                "available_services": [],
                "error": str(e),
            }

    async def revoke_token(
        self,
        user_id: str,
        db,
    ) -> bool:
        """
        OAuth 토큰 취소 (로그아웃)

        Args:
            user_id: 사용자 ID
            db: MongoDB 데이터베이스

        Returns:
            성공 여부
        """
        try:
            # 토큰 조회
            token_doc = await db.google_oauth_tokens.find_one({"user_id": user_id})

            if token_doc:
                access_token_encrypted = token_doc.get("access_token_encrypted")

                if access_token_encrypted:
                    access_token = encryption_service.decrypt(access_token_encrypted)

                    # Google에 토큰 취소 요청
                    from utils.http_client import get_http_client

                    revoke_client = get_http_client()
                    await revoke_client.post(
                        "https://oauth2.googleapis.com/revoke",
                        params={"token": access_token},
                    )

                # MongoDB에서 삭제
                await db.google_oauth_tokens.delete_one({"user_id": user_id})

            logger.info(f"Revoked Google OAuth token for user: {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            return False


# Singleton instance (env vars)
google_oauth_service = GoogleOAuthService()


async def get_effective_google_oauth_service(db) -> "GoogleOAuthService":
    """
    DB에 저장된 Google OAuth 설정이 있으면 해당 값으로 서비스를 반환하고,
    없으면 환경변수 기반 기본 인스턴스를 반환합니다.
    """
    if db is None:
        return google_oauth_service
    try:
        doc = await db["google_oauth_config"].find_one({})
        if not doc:
            return google_oauth_service
        client_id = encryption_service.decrypt(doc.get("client_id_encrypted")) or ""
        client_secret = encryption_service.decrypt(doc.get("client_secret_encrypted")) or ""
        redirect_uri = (doc.get("redirect_uri") or "").strip()
        if client_id and client_secret:
            return GoogleOAuthService(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri or None,
            )
    except Exception as e:
        logger.warning(f"Failed to load Google OAuth config from DB: {e}")
    return google_oauth_service
