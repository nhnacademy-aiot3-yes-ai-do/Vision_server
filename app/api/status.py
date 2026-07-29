"""Liveness and model-readiness probes without inference."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.model_registry import RegistryState
from app.schemas.status import LiveStatusResponse, ReadyStatusResponse


router = APIRouter(prefix="/health", tags=["service-health"])


@router.get(
    "/live",
    response_model=LiveStatusResponse,
    summary="서비스 프로세스 liveness 확인",
    description="모델이나 추론기를 호출하지 않고 FastAPI 프로세스 상태만 확인합니다.",
)
async def live() -> LiveStatusResponse:
    """Return UP without accessing the model registry."""

    return LiveStatusResponse(status="UP")


def _registry_is_ready(request: Request) -> bool:
    """Read only the registry lifecycle state and hide internal failures."""

    try:
        registry = getattr(request.app.state, "model_registry", None)
        return (
            registry is not None
            and getattr(registry, "state", None) is RegistryState.READY
        )
    except Exception:
        return False


@router.get(
    "/ready",
    response_model=ReadyStatusResponse,
    summary="모델 readiness 확인",
    description=(
        "두 모델을 보관하는 registry가 READY인지 확인합니다. "
        "모델 조회나 추론은 실행하지 않습니다."
    ),
    responses={
        503: {
            "model": ReadyStatusResponse,
            "description": "모델 registry가 아직 요청을 받을 준비가 되지 않음",
        }
    },
)
async def ready(request: Request) -> ReadyStatusResponse | JSONResponse:
    """Return readiness without exposing model paths or internal exceptions."""

    if _registry_is_ready(request):
        return ReadyStatusResponse(status="READY")
    response = ReadyStatusResponse(status="NOT_READY")
    return JSONResponse(
        status_code=503,
        content=response.model_dump(mode="json"),
    )
