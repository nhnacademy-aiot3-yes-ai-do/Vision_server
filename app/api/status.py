# Kubernetes 등이 프로세스 생존 여부와 모델 준비 여부를 추론 없이 확인하는 상태 API이다.
"""Liveness and model-readiness probes without inference."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.model_registry import RegistryState
from app.schemas.status import LiveStatusResponse, ReadyStatusResponse


# 서비스 상태 점검용 엔드포인트를 실제 비즈니스 API와 분리된 경로에 묶는다.
router = APIRouter(prefix="/health", tags=["service-health"])


# 프로세스가 HTTP 요청에 응답할 수 있는지만 확인하며 모델 상태는 읽지 않는다.
@router.get(
    "/live",
    response_model=LiveStatusResponse,
    summary="서비스 프로세스 liveness 확인",
    description="모델이나 추론기를 호출하지 않고 FastAPI 프로세스 상태만 확인합니다.",
)
async def live() -> LiveStatusResponse:
    """Return UP without accessing the model registry."""

    return LiveStatusResponse(status="UP")


# 애플리케이션 상태에 저장된 레지스트리의 상태값만 읽어 readiness를 판단한다.
def _registry_is_ready(request: Request) -> bool:
    """Read only the registry lifecycle state and hide internal failures."""

    try:
        registry = getattr(request.app.state, "model_registry", None)
        return (
            registry is not None
            and getattr(registry, "state", None) is RegistryState.READY
        )
    except Exception:
        # 레지스트리 조회 자체가 실패해도 내부 예외를 노출하지 않고 준비되지 않은 것으로 처리한다.
        return False


# 레지스트리가 READY일 때만 트래픽을 받을 수 있도록 200, 그 외에는 503을 반환한다.
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
    # 503 응답도 ReadyStatusResponse와 같은 JSON 모양을 유지한다.
    response = ReadyStatusResponse(status="NOT_READY")
    return JSONResponse(
        status_code=503,
        content=response.model_dump(mode="json"),
    )
