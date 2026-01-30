"""
Google OAuth Router

Google OAuth 2.0 인증 플로우 엔드포인트:
- /api/auth/google/start - 인증 시작 (Auth URL 생성)
- /api/auth/google/callback - OAuth 콜백 처리
- /api/auth/google/status - 인증 상태 확인
- /api/auth/google/logout - 로그아웃 (토큰 취소)
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import List, Optional

from database import get_mongodb_database
from services.google_oauth_service import (
    GOOGLE_SCOPES,
    get_effective_google_oauth_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth/google", tags=["Google OAuth"])


def get_db():
    """MongoDB DB dependency (Google OAuth 설정·토큰 조회용)"""
    return get_mongodb_database()


# ===========================================
# Request/Response Models
# ===========================================

class GoogleAuthStartRequest(BaseModel):
    """Google OAuth 시작 요청"""
    services: List[str] = Field(
        default=["drive", "docs"],
        description="요청할 Google 서비스 목록 (drive, docs, gmail, calendar, sheets)"
    )
    user_id: str = Field(default="default", description="사용자 ID")


class GoogleAuthStartResponse(BaseModel):
    """Google OAuth 시작 응답"""
    auth_url: str = Field(..., description="Google 인증 URL")
    state: str = Field(..., description="CSRF 방지 state")
    scopes: List[str] = Field(..., description="요청된 스코프")
    services: List[str] = Field(..., description="요청된 서비스")


class GoogleAuthStatusResponse(BaseModel):
    """Google 인증 상태 응답"""
    is_authenticated: bool = Field(..., description="인증 여부")
    scopes: List[str] = Field(default_factory=list, description="승인된 스코프")
    expires_at: Optional[str] = Field(None, description="토큰 만료 시간")
    available_services: List[str] = Field(default_factory=list, description="사용 가능한 서비스")


class GoogleAuthCallbackResponse(BaseModel):
    """OAuth 콜백 응답"""
    success: bool
    user_id: str
    scopes: List[str]
    expires_at: str


# ===========================================
# Endpoints
# ===========================================

@router.get("/services")
async def list_google_services(db=Depends(get_db)):
    """
    사용 가능한 Google 서비스 목록

    각 서비스별 필요한 스코프를 반환합니다.
    is_configured는 환경변수 또는 DB에 저장된 설정을 사용합니다.
    """
    service = await get_effective_google_oauth_service(db)
    return {
        "services": [
            {
                "name": svc,
                "scopes": scopes,
                "description": {
                    "drive": "Google Drive 파일 관리",
                    "docs": "Google Docs 문서 관리",
                    "gmail": "Gmail 이메일 전송/읽기",
                    "calendar": "Google Calendar 일정 관리",
                    "sheets": "Google Sheets 스프레드시트 관리",
                }.get(svc, ""),
            }
            for svc, scopes in GOOGLE_SCOPES.items()
        ],
        "is_configured": service.is_configured,
    }


@router.post("/start", response_model=GoogleAuthStartResponse)
async def start_google_auth(request: GoogleAuthStartRequest, db=Depends(get_db)):
    """
    Google OAuth 인증 시작

    인증 URL을 생성하여 반환합니다.
    클라이언트는 이 URL로 리다이렉트하여 Google 로그인을 진행합니다.
    DB에 저장된 Google OAuth 설정이 있으면 그 값을 사용합니다.
    """
    try:
        service = await get_effective_google_oauth_service(db)
        result = service.get_authorization_url(
            services=request.services,
            user_id=request.user_id,
        )
        return GoogleAuthStartResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start Google auth: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/callback")
async def google_oauth_callback(
    code: str = Query(..., description="Authorization code"),
    state: str = Query(..., description="State parameter"),
    error: Optional[str] = Query(None, description="Error if any"),
):
    """
    Google OAuth 콜백

    Google 인증 완료 후 리다이렉트되는 엔드포인트입니다.
    Authorization code를 access token으로 교환하고 저장합니다.
    """
    if error:
        # 프론트엔드로 에러와 함께 리다이렉트
        frontend_url = "http://localhost:3000"
        return RedirectResponse(
            url=f"{frontend_url}/settings?error={error}"
        )

    try:
        db = get_mongodb_database()
        if db is None:
            raise HTTPException(
                status_code=503,
                detail="MongoDB not configured"
            )

        service = await get_effective_google_oauth_service(db)
        result = await service.handle_callback(
            code=code,
            state=state,
            db=db,
        )

        # 성공 시 프론트엔드로 리다이렉트
        frontend_url = "http://localhost:3000"
        return RedirectResponse(
            url=f"{frontend_url}/settings?google_auth=success&user_id={result['user_id']}"
        )

    except ValueError as e:
        frontend_url = "http://localhost:3000"
        return RedirectResponse(
            url=f"{frontend_url}/settings?error={str(e)}"
        )
    except Exception as e:
        logger.error(f"OAuth callback failed: {e}")
        frontend_url = "http://localhost:3000"
        return RedirectResponse(
            url=f"{frontend_url}/settings?error=callback_failed"
        )


@router.get("/status", response_model=GoogleAuthStatusResponse)
async def get_google_auth_status(
    user_id: str = Query(default="default", description="사용자 ID"),
):
    """
    Google 인증 상태 확인

    현재 사용자의 Google OAuth 인증 상태를 반환합니다.
    """
    try:
        db = get_mongodb_database()
        if db is None:
            return GoogleAuthStatusResponse(
                is_authenticated=False,
                scopes=[],
                expires_at=None,
                available_services=[],
            )

        service = await get_effective_google_oauth_service(db)
        status = await service.get_auth_status(
            user_id=user_id,
            db=db,
        )

        return GoogleAuthStatusResponse(**status)

    except Exception as e:
        logger.error(f"Failed to get auth status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/logout")
async def google_logout(
    user_id: str = Query(default="default", description="사용자 ID"),
):
    """
    Google 로그아웃

    OAuth 토큰을 취소하고 삭제합니다.
    """
    try:
        db = get_mongodb_database()
        if db is None:
            raise HTTPException(
                status_code=503,
                detail="MongoDB not configured"
            )

        service = await get_effective_google_oauth_service(db)
        success = await service.revoke_token(
            user_id=user_id,
            db=db,
        )

        return {
            "success": success,
            "message": "Logged out from Google" if success else "Logout failed",
        }

    except Exception as e:
        logger.error(f"Failed to logout: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/token")
async def get_access_token(
    user_id: str = Query(default="default", description="사용자 ID"),
):
    """
    Google 액세스 토큰 가져오기 (내부 사용)

    유효한 액세스 토큰을 반환합니다. 만료된 경우 자동으로 갱신합니다.

    Note: 이 엔드포인트는 보안상 내부에서만 사용해야 합니다.
    """
    try:
        db = get_mongodb_database()
        if db is None:
            raise HTTPException(
                status_code=503,
                detail="MongoDB not configured"
            )

        service = await get_effective_google_oauth_service(db)
        access_token = await service.get_valid_access_token(
            user_id=user_id,
            db=db,
        )

        if not access_token:
            raise HTTPException(
                status_code=401,
                detail="Not authenticated with Google"
            )

        # 토큰은 마스킹하여 반환 (보안)
        masked_token = access_token[:10] + "..." + access_token[-4:]

        return {
            "has_token": True,
            "token_preview": masked_token,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get token: {e}")
        raise HTTPException(status_code=500, detail=str(e))
