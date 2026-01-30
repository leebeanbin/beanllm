"""
Config & Health Router

System health checks, configuration, and API key management endpoints.
"""

import logging
import os
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from schemas.database import (
    ApiKeyCreate,
    ApiKeyResponse,
    ApiKeyListResponse,
    ApiKeyValidationResult,
    ProviderInfo,
    ProviderListResponse,
    PROVIDER_CONFIG,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Config & Health"])


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "ok",
        "service": "beanllm-playground-api",
        "version": "1.0.0",
        "features": [
            "chat",
            "rag",
            "multi-agent",
            "knowledge-graph",
            "audio",
            "ocr",
            "google-workspace",
            "web-search",
            "evaluation",
            "finetuning",
        ],
    }


@router.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Import here to avoid circular dependency
        from common import (
            _client,
            _kg,
            _rag_chains,
            _web_search,
            _rag_debugger,
            _optimizer,
            _multi_agent,
            _orchestrator,
        )
        from database import ping_mongodb

        mongodb_status = await ping_mongodb()

        return {
            "status": "healthy",
            "services": {
                "mongodb": mongodb_status,
                "client": _client is not None,
                "kg": _kg is not None,
                "rag_chains": len(_rag_chains),
                "web_search": _web_search is not None,
                "rag_debugger": _rag_debugger is not None,
                "optimizer": _optimizer is not None,
                "multi_agent": _multi_agent is not None,
                "orchestrator": _orchestrator is not None,
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@router.get("/api/config/providers")
async def get_active_providers():
    """
    Get active providers based on available API keys.

    Returns:
        {
            "providers": ["openai", "anthropic", "ollama"],
            "config": {
                "openai_api_key": "***MASKED***",
                "anthropic_api_key": None,
                ...
            }
        }
    """
    try:
        from beanllm.utils.config import EnvConfig

        return {
            "providers": EnvConfig.get_active_providers(),
            "config": EnvConfig.get_safe_config_dict(),
        }
    except Exception as e:
        logger.error(f"Failed to get active providers: {e}")
        # Fallback: Ollama is always available
        return {
            "providers": ["ollama"],
            "config": {},
        }


@router.get("/api/config/models")
async def get_available_models():
    """
    Get available models per active provider.

    Returns:
        {
            "openai": ["gpt-4o", "gpt-4o-mini", ...],
            "anthropic": ["claude-sonnet-4", ...],
            "ollama": ["qwen2.5:0.5b", ...]
        }
    """
    try:
        from beanllm.utils.config import EnvConfig

        available_models = {}

        # OpenAI models
        if EnvConfig.is_provider_available("openai"):
            available_models["openai"] = [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
            ]

        # Anthropic models
        if EnvConfig.is_provider_available("anthropic"):
            available_models["anthropic"] = [
                "claude-sonnet-4-20250514",
                "claude-opus-4-20250514",
                "claude-haiku-4-20250514",
            ]

        # Google models
        if EnvConfig.is_provider_available("google"):
            available_models["google"] = [
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-1.5-pro",
            ]

        # DeepSeek models
        if EnvConfig.is_provider_available("deepseek"):
            available_models["deepseek"] = [
                "deepseek-chat",
                "deepseek-coder",
            ]

        # Perplexity models
        if EnvConfig.is_provider_available("perplexity"):
            available_models["perplexity"] = [
                "llama-3.1-sonar-large-128k-online",
                "llama-3.1-sonar-small-128k-online",
            ]

        # Ollama models (always available) - 실제 설치된 모델만 반환
        try:
            from beanllm.providers.ollama_provider import OllamaProvider

            ollama = OllamaProvider()
            ollama_models = await ollama.list_models()
            # 실제 설치된 모델만 반환 (하드코딩된 fallback 제거)
            if ollama_models:
                available_models["ollama"] = [m["name"] if isinstance(m, dict) else str(m) for m in ollama_models]
            else:
                # 모델이 없으면 빈 배열 (하드코딩된 모델 제거)
                available_models["ollama"] = []
                logger.info("No Ollama models installed. User should install a model first.")
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
            # 에러 발생 시에도 하드코딩된 모델 반환하지 않음
            available_models["ollama"] = []

        return available_models

    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        # 에러 발생 시에도 하드코딩된 모델 반환하지 않음
        return {"ollama": []}


# ===========================================
# API Key Management Endpoints
# ===========================================

def get_db():
    """Get MongoDB database connection."""
    from database import get_mongodb_database
    return get_mongodb_database()


@router.get("/api/config/keys", response_model=ApiKeyListResponse)
async def list_api_keys(db=Depends(get_db)):
    """
    List all configured API keys (without revealing the actual keys).

    Returns:
        List of API keys with provider, hint, validation status.
    """
    try:
        from services.encryption_service import encryption_service

        keys_collection = db["api_keys"]
        cursor = keys_collection.find({})
        keys = []

        async for doc in cursor:
            keys.append(ApiKeyResponse(
                provider=doc["provider"],
                key_hint=doc.get("key_hint", "****"),
                is_valid=doc.get("is_valid", False),
                last_validated=doc.get("last_validated"),
                created_at=doc.get("created_at"),  # Use stored value, None if not present
                updated_at=doc.get("updated_at"),  # Use stored value, None if not present
                metadata=doc.get("metadata", {}),
            ))

        return ApiKeyListResponse(keys=keys, total=len(keys))

    except Exception as e:
        logger.error(f"Failed to list API keys: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/config/keys/{provider}", response_model=ApiKeyResponse)
async def get_api_key(provider: str, db=Depends(get_db)):
    """
    Get a specific API key by provider.

    Args:
        provider: Provider name (openai, anthropic, etc.)

    Returns:
        API key info (without the actual key).
    """
    try:
        keys_collection = db["api_keys"]
        doc = await keys_collection.find_one({"provider": provider})

        if not doc:
            raise HTTPException(status_code=404, detail=f"API key for {provider} not found")

        return ApiKeyResponse(
            provider=doc["provider"],
            key_hint=doc.get("key_hint", "****"),
            is_valid=doc.get("is_valid", False),
            last_validated=doc.get("last_validated"),
            created_at=doc.get("created_at"),  # Use stored value, None if not present
            updated_at=doc.get("updated_at"),  # Use stored value, None if not present
            metadata=doc.get("metadata", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get API key for {provider}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/config/keys", response_model=ApiKeyResponse)
async def save_api_key(request: ApiKeyCreate, db=Depends(get_db)):
    """
    Save or update an API key.

    The key is encrypted before storage. If a key for this provider
    already exists, it will be updated.

    Args:
        request: Provider and API key to save.

    Returns:
        Saved API key info.
    """
    try:
        from services.encryption_service import encryption_service

        # Validate provider
        if request.provider not in PROVIDER_CONFIG:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider: {request.provider}. Valid providers: {list(PROVIDER_CONFIG.keys())}"
            )

        # Encrypt the key
        encrypted_key = encryption_service.encrypt(request.api_key)
        key_hint = encryption_service.get_key_hint(request.api_key)

        now = datetime.now(timezone.utc)
        keys_collection = db["api_keys"]

        # Check if key exists
        existing = await keys_collection.find_one({"provider": request.provider})

        doc = {
            "provider": request.provider,
            "key_encrypted": encrypted_key,
            "key_hint": key_hint,
            "is_valid": False,  # Will be validated separately
            "last_validated": None,
            "updated_at": now,
            "metadata": request.metadata or {},
        }

        if existing:
            # Update existing
            await keys_collection.update_one(
                {"provider": request.provider},
                {"$set": doc}
            )
            doc["created_at"] = existing.get("created_at", now)
        else:
            # Insert new
            doc["created_at"] = now
            await keys_collection.insert_one(doc)

        # Inject into environment
        env_var = PROVIDER_CONFIG[request.provider]["env_var"]
        os.environ[env_var] = request.api_key
        logger.info(f"API key for {request.provider} saved and injected into environment")

        return ApiKeyResponse(
            provider=doc["provider"],
            key_hint=doc["key_hint"],
            is_valid=doc["is_valid"],
            last_validated=doc["last_validated"],
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
            metadata=doc["metadata"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/config/keys/{provider}")
async def delete_api_key(provider: str, db=Depends(get_db)):
    """
    Delete an API key.

    Args:
        provider: Provider name to delete.

    Returns:
        Success message.
    """
    try:
        keys_collection = db["api_keys"]

        result = await keys_collection.delete_one({"provider": provider})

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"API key for {provider} not found")

        # Remove from environment
        if provider in PROVIDER_CONFIG:
            env_var = PROVIDER_CONFIG[provider]["env_var"]
            if env_var in os.environ:
                del os.environ[env_var]

        logger.info(f"API key for {provider} deleted")
        return {"message": f"API key for {provider} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete API key for {provider}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/config/keys/{provider}/validate", response_model=ApiKeyValidationResult)
async def validate_api_key(provider: str, db=Depends(get_db)):
    """
    Validate an API key by making a test API call.

    Args:
        provider: Provider name to validate.

    Returns:
        Validation result with available models if successful.
    """
    try:
        from services.encryption_service import encryption_service
        from services.key_validator import key_validator

        keys_collection = db["api_keys"]
        doc = await keys_collection.find_one({"provider": provider})

        if not doc:
            raise HTTPException(status_code=404, detail=f"API key for {provider} not found")

        # Decrypt the key
        api_key = encryption_service.decrypt(doc["key_encrypted"])

        # Validate the key
        result = await key_validator.validate(provider, api_key)

        # Update validation status in DB
        now = datetime.now(timezone.utc)
        await keys_collection.update_one(
            {"provider": provider},
            {"$set": {
                "is_valid": result.is_valid,
                "last_validated": now,
                "updated_at": now,
            }}
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate API key for {provider}: {e}")
        return ApiKeyValidationResult(
            provider=provider,
            is_valid=False,
            error=str(e),
        )


@router.get("/api/config/providers/all", response_model=ProviderListResponse)
async def list_all_providers(db=Depends(get_db)):
    """
    List all supported providers with their configuration status.

    Returns:
        List of providers with name, env_var, and whether they're configured.
    """
    try:
        keys_collection = db["api_keys"]

        # Get all configured keys
        configured_keys = {}
        async for doc in keys_collection.find({}):
            configured_keys[doc["provider"]] = {
                "is_valid": doc.get("is_valid", False),
            }

        providers = []
        for provider_id, config in PROVIDER_CONFIG.items():
            is_configured = provider_id in configured_keys or bool(os.getenv(config["env_var"]))
            is_valid = configured_keys.get(provider_id, {}).get("is_valid")

            providers.append(ProviderInfo(
                id=provider_id,
                name=config["name"],
                env_var=config["env_var"],
                placeholder=config["placeholder"],
                description=config["description"],
                is_configured=is_configured,
                is_valid=is_valid if is_configured else None,
            ))

        return ProviderListResponse(providers=providers)

    except Exception as e:
        logger.error(f"Failed to list providers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================
# Google OAuth 설정 (모달에서 입력·저장)
# ===========================================


class GoogleOAuthConfigCreate(BaseModel):
    """Google OAuth 설정 저장 요청"""
    client_id: str
    client_secret: str
    redirect_uri: Optional[str] = None


@router.post("/api/config/google-oauth")
async def save_google_oauth_config(request: GoogleOAuthConfigCreate, db=Depends(get_db)):
    """
    Google OAuth Client ID / Secret / Redirect URI를 저장합니다.
    모달에서 설정 입력 후 호출하며, 암호화되어 MongoDB에 저장됩니다.
    """
    try:
        from services.encryption_service import encryption_service

        if not request.client_id.strip() or not request.client_secret.strip():
            raise HTTPException(
                status_code=400,
                detail="client_id와 client_secret은 필수입니다",
            )
        redirect_uri = (request.redirect_uri or "").strip() or "http://localhost:8000/api/auth/google/callback"
        client_id_enc = encryption_service.encrypt(request.client_id.strip())
        client_secret_enc = encryption_service.encrypt(request.client_secret.strip())

        coll = db["google_oauth_config"]
        await coll.update_one(
            {},
            {
                "$set": {
                    "client_id_encrypted": client_id_enc,
                    "client_secret_encrypted": client_secret_enc,
                    "redirect_uri": redirect_uri,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )
        logger.info("Google OAuth config saved from UI")
        return {"ok": True, "message": "Google OAuth 설정이 저장되었습니다"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save Google OAuth config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/config/google-oauth/status")
async def get_google_oauth_config_status(db=Depends(get_db)):
    """
    Google OAuth가 설정되어 있는지 반환합니다.
    환경변수 또는 DB에 저장된 값이 있으면 True입니다.
    """
    try:
        if os.getenv("GOOGLE_OAUTH_CLIENT_ID") and os.getenv("GOOGLE_OAUTH_CLIENT_SECRET"):
            return {"is_configured": True, "source": "env"}
        if db:
            doc = await db["google_oauth_config"].find_one({})
            if doc and doc.get("client_id_encrypted") and doc.get("client_secret_encrypted"):
                return {"is_configured": True, "source": "db"}
        return {"is_configured": False, "source": None}
    except Exception as e:
        logger.warning(f"Failed to check Google OAuth config: {e}")
        return {"is_configured": False, "source": None}


@router.post("/api/config/keys/load-all")
async def load_all_keys_to_env(db=Depends(get_db)):
    """
    Load all API keys from MongoDB and inject into environment variables.

    This is typically called on application startup.

    Returns:
        Number of keys loaded.
    """
    try:
        from services.encryption_service import encryption_service

        keys_collection = db["api_keys"]
        count = 0

        async for doc in keys_collection.find({}):
            provider = doc["provider"]
            if provider in PROVIDER_CONFIG:
                env_var = PROVIDER_CONFIG[provider]["env_var"]
                try:
                    api_key = encryption_service.decrypt(doc["key_encrypted"])
                    os.environ[env_var] = api_key
                    count += 1
                    logger.info(f"Loaded API key for {provider}")
                except Exception as e:
                    logger.warning(f"Failed to load API key for {provider}: {e}")

        return {"message": f"Loaded {count} API keys into environment", "count": count}

    except Exception as e:
        logger.error(f"Failed to load API keys: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Provider SDK 상태 확인 및 설치
class ProviderSDKStatus(BaseModel):
    provider: str
    installed: bool
    package_name: str
    install_command: str
    error: Optional[str] = None


class ProviderSDKStatusResponse(BaseModel):
    providers: List[ProviderSDKStatus]
    warnings: List[str] = []


@router.get("/api/config/provider-sdks", response_model=ProviderSDKStatusResponse)
async def get_provider_sdk_status():
    """
    Provider SDK 설치 상태 확인

    Returns:
        설치되지 않은 Provider SDK 목록 및 설치 명령어
    """
    try:
        from beanllm.providers.provider_factory import (
            OpenAIProvider,
            ClaudeProvider,
            GeminiProvider,
            OllamaProvider,
            DeepSeekProvider,
            PerplexityProvider,
        )

        providers = []
        warnings = []

        # Provider별 SDK 상태 확인
        provider_configs = [
            {
                "provider": "openai",
                "package_name": "openai",
                "install_command": "poetry add --group web openai",
                "installed": OpenAIProvider is not None,
            },
            {
                "provider": "anthropic",
                "package_name": "anthropic",
                "install_command": "poetry add --group web anthropic",
                "installed": ClaudeProvider is not None,
            },
            {
                "provider": "gemini",
                "package_name": "google-generativeai",
                "install_command": "poetry add --group web google-generativeai",
                "installed": GeminiProvider is not None,
            },
            {
                "provider": "ollama",
                "package_name": "ollama",
                "install_command": "poetry add ollama",  # 기본 의존성
                "installed": OllamaProvider is not None,
            },
            {
                "provider": "deepseek",
                "package_name": "openai",  # DeepSeek uses OpenAI SDK
                "install_command": "poetry add --group web openai",
                "installed": DeepSeekProvider is not None,
            },
            {
                "provider": "perplexity",
                "package_name": "openai",  # Perplexity uses OpenAI SDK
                "install_command": "poetry add --group web openai",
                "installed": PerplexityProvider is not None,
            },
        ]

        for config in provider_configs:
            try:
                providers.append(
                    ProviderSDKStatus(
                        provider=config["provider"],
                        installed=config["installed"],
                        package_name=config["package_name"],
                        install_command=config["install_command"],
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to check {config['provider']} SDK: {e}")
                providers.append(
                    ProviderSDKStatus(
                        provider=config["provider"],
                        installed=False,
                        package_name=config["package_name"],
                        install_command=config["install_command"],
                        error=str(e),
                    )
                )

        # 경고 메시지 생성
        missing_providers = [p for p in providers if not p.installed]
        if missing_providers:
            warnings.append(
                f"{len(missing_providers)}개의 Provider SDK가 설치되지 않았습니다. "
                "모델을 사용하려면 SDK를 설치해야 합니다."
            )

        return ProviderSDKStatusResponse(providers=providers, warnings=warnings)

    except Exception as e:
        logger.error(f"Failed to get provider SDK status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class PackageInstallRequest(BaseModel):
    package_group: str  # e.g., "openai", "gemini", "ollama"


class PackageInstallResponse(BaseModel):
    success: bool
    message: str
    output: Optional[str] = None
    error: Optional[str] = None


@router.post("/api/config/install-package", response_model=PackageInstallResponse)
async def install_package(request: PackageInstallRequest):
    """
    Provider SDK 패키지 설치 (Poetry 사용)

    Args:
        request: 설치할 패키지 그룹

    Returns:
        설치 결과
    """
    import subprocess
    import sys

    try:
        # 패키지 그룹 매핑
        package_map = {
            "openai": "openai",
            "anthropic": "anthropic",
            "gemini": "google-generativeai",
            "ollama": "ollama",
            "deepseek": "openai",  # DeepSeek uses OpenAI SDK
            "perplexity": "openai",  # Perplexity uses OpenAI SDK
        }

        package_name = package_map.get(request.package_group)
        if not package_name:
            return PackageInstallResponse(
                success=False,
                message=f"Unknown package group: {request.package_group}",
            )

        # Poetry로 패키지 설치
        # ollama는 기본 의존성이므로 일반 설치, 나머지는 web 그룹에 추가
        if request.package_group == "ollama":
            cmd = ["poetry", "add", package_name]
        else:
            cmd = ["poetry", "add", "--group", "web", package_name]

        logger.info(f"Installing package: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5분 타임아웃
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # 프로젝트 루트
        )

        if result.returncode == 0:
            return PackageInstallResponse(
                success=True,
                message=f"{package_name} 패키지가 성공적으로 설치되었습니다.",
                output=result.stdout,
            )
        else:
            return PackageInstallResponse(
                success=False,
                message=f"{package_name} 패키지 설치 실패",
                error=result.stderr or result.stdout,
                output=result.stdout,
            )

    except subprocess.TimeoutExpired:
        return PackageInstallResponse(
            success=False,
            message="패키지 설치 시간 초과 (5분)",
            error="설치가 너무 오래 걸렸습니다. 터미널에서 직접 설치해주세요.",
        )
    except Exception as e:
        logger.error(f"Failed to install package: {e}")
        return PackageInstallResponse(
            success=False,
            message="패키지 설치 중 오류 발생",
            error=str(e),
        )
