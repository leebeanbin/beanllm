"""
Chain Router

Chain building and execution endpoints.
Uses Python best practices: factory pattern, duck typing.
"""

import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from common import get_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chain", tags=["Chain"])

# In-memory chain storage
_chains: Dict[str, Any] = {}


# ============================================================================
# Request/Response Models
# ============================================================================

class ChainRequest(BaseModel):
    """Request to run or build a chain"""
    input: str = Field(..., description="Input text for the chain")
    chain_id: Optional[str] = Field(None, description="Chain ID to use or create")
    chain_type: str = Field(default="basic", description="Chain type: basic, prompt")
    template: Optional[str] = Field(None, description="Prompt template for prompt chains")
    model: Optional[str] = Field(None, description="LLM model to use")


class ChainRunResponse(BaseModel):
    """Response from chain execution"""
    chain_id: str
    input: str
    output: str
    steps: list = Field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class ChainBuildResponse(BaseModel):
    """Response from chain building"""
    chain_id: str
    chain_type: str
    status: str = "success"


# ============================================================================
# Helper Functions
# ============================================================================

def _get_or_create_chain(
    chain_id: str,
    chain_type: str,
    template: Optional[str],
    client: Any
) -> Any:
    """Get existing chain or create new one using factory pattern"""
    from beanllm.facade.core.chain_facade import Chain, PromptChain

    if chain_id in _chains:
        return _chains[chain_id]

    # Factory pattern for chain creation
    chain_factories = {
        "prompt": lambda: PromptChain(client=client, template=template) if template else Chain(client=client),
        "basic": lambda: Chain(client=client),
    }

    factory = chain_factories.get(chain_type, chain_factories["basic"])
    chain = factory()
    _chains[chain_id] = chain

    return chain


def _extract_chain_result(result: Any) -> Dict[str, Any]:
    """Extract result using duck typing"""
    return {
        "output": getattr(result, "output", str(result)),
        "steps": getattr(result, "steps", []),
        "success": getattr(result, "success", True),
        "error": getattr(result, "error", None),
    }


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/run", response_model=ChainRunResponse)
async def chain_run(request: ChainRequest) -> ChainRunResponse:
    """
    Run a chain with input.

    Creates chain if it doesn't exist, then executes with provided input.
    """
    try:
        from beanllm.facade.core.client_facade import Client

        client = Client(model=request.model) if request.model else get_client()
        chain_id = request.chain_id or "default"

        chain = _get_or_create_chain(
            chain_id=chain_id,
            chain_type=request.chain_type,
            template=request.template,
            client=client,
        )

        result = await chain.run(user_input=request.input)
        extracted = _extract_chain_result(result)

        return ChainRunResponse(
            chain_id=chain_id,
            input=request.input,
            **extracted,
        )

    except Exception as e:
        logger.error(f"Chain error: {e}", exc_info=True)
        raise HTTPException(500, f"Chain error: {str(e)}")


@router.post("/build", response_model=ChainBuildResponse)
async def chain_build(request: ChainRequest) -> ChainBuildResponse:
    """
    Build a new chain using ChainBuilder.

    Stores the chain for later use with chain_run.
    """
    try:
        from beanllm.facade.core.client_facade import Client
        from beanllm.facade.core.chain_facade import ChainBuilder

        client = Client(model=request.model) if request.model else get_client()

        # Use builder pattern
        builder = ChainBuilder(client=client)

        if request.template:
            builder.with_template(request.template)

        chain = builder.build()

        # Store chain
        chain_id = request.chain_id or f"chain_{len(_chains)}"
        _chains[chain_id] = chain

        return ChainBuildResponse(
            chain_id=chain_id,
            chain_type=request.chain_type,
            status="success",
        )

    except Exception as e:
        logger.error(f"Chain build error: {e}", exc_info=True)
        raise HTTPException(500, f"Chain build error: {str(e)}")


@router.get("/list")
async def chain_list() -> Dict[str, Any]:
    """List all stored chains"""
    return {
        "chains": list(_chains.keys()),
        "total": len(_chains),
    }


@router.delete("/{chain_id}")
async def chain_delete(chain_id: str) -> Dict[str, str]:
    """Delete a stored chain"""
    if chain_id not in _chains:
        raise HTTPException(404, f"Chain '{chain_id}' not found")

    del _chains[chain_id]
    return {"status": "deleted", "chain_id": chain_id}
