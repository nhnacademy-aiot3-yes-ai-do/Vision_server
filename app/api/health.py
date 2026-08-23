# 버섯 이미지 한 장을 받아 검증·추론하고 공개 응답 형태로 돌려주는 REST API 라우터이다.
"""Internal mushroom health-check endpoint."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.schemas.health import HealthCheckResponse, safe_error_response
from app.services.mushroom_health_service import (
    HealthServiceError,
    MushroomHealthService,
)


LOGGER = logging.getLogger(__name__)
# 모든 버섯 건강 판별 API는 이 공통 URL 접두사 아래에 등록된다.
router = APIRouter(prefix="/api/internal/mushrooms", tags=["mushroom-health"])


# FastAPI 의존성 주입으로 lifespan에서 만든 단일 서비스 인스턴스를 꺼낸다.
async def get_health_service(request: Request) -> MushroomHealthService:
    """Return the lifespan-managed service or a safe availability error."""

    service = getattr(request.app.state, "health_service", None)
    # 시작이 끝나지 않았거나 종료 중이면 추론 요청을 받지 않고 503으로 차단한다.
    if service is None:
        raise HTTPException(status_code=503, detail="Service unavailable")
    return service


# 서비스 계층 오류의 공개 정보만 골라 동일한 HealthCheckResponse 계약으로 직렬화한다.
def _error_json(
    service: MushroomHealthService,
    error: HealthServiceError,
) -> JSONResponse:
    response = safe_error_response(
        status=error.status,
        detection_threshold=service.settings.detection_confidence,
        min_detection_confidence=(
            service.settings.min_detection_confidence
        ),
        health_threshold=service.settings.health_uncertain_threshold,
        public_message=error.public_message,
    )
    return JSONResponse(
        status_code=error.http_status,
        content=response.model_dump(mode="json", by_alias=True),
    )


# 성공과 예상 가능한 실패 모두 동일한 공개 응답 구조를 사용하는 이미지 분석 엔드포인트이다.
@router.post(
    "/health-check",
    response_model=HealthCheckResponse,
    response_model_by_alias=True,
    summary="버섯 품종 탐지 및 건강 상태 확인",
    description=(
        "업로드 이미지를 저장하지 않고 품종별 객체를 탐지한 뒤 union crop으로 "
        "HEALTHY, DISEASE_SUSPECTED 또는 UNCERTAIN 상태를 반환합니다. "
        "AI 참고 결과이며 확정 진단이 아닙니다."
    ),
    responses={
        400: {"model": HealthCheckResponse, "description": "빈 파일 또는 손상 이미지"},
        413: {"model": HealthCheckResponse, "description": "업로드 크기 제한 초과"},
        415: {"model": HealthCheckResponse, "description": "지원하지 않는 이미지 형식"},
        422: {"description": "multipart image 필드 누락"},
        500: {"model": HealthCheckResponse, "description": "안전하게 숨긴 추론 실패"},
    },
)
async def health_check(
    image: Annotated[
        UploadFile,
        File(description="JPG, JPEG, PNG 또는 WEBP 이미지"),
    ],
    service: Annotated[
        MushroomHealthService,
        Depends(get_health_service),
    ],
) -> HealthCheckResponse | JSONResponse:
    """Analyze one upload without persisting it or exposing internal errors."""

    try:
        # 업로드 검증과 모델 실행은 서비스 계층에 위임하고 내부 snake_case 결과를 DTO로 변환한다.
        internal = await service.analyze_upload(image)
        return HealthCheckResponse.from_internal(internal)
    except HealthServiceError as exc:
        # 서버 내부 장애만 traceback을 기록하고, 클라이언트에는 안전한 공개 메시지만 반환한다.
        if exc.http_status >= 500:
            LOGGER.exception("Mushroom health inference failed")
        return _error_json(service, exc)
    except Exception:
        # 분류하지 못한 예외도 경로·모델 정보가 노출되지 않는 일반 500 응답으로 치환한다.
        LOGGER.exception("Unexpected mushroom health API failure")
        safe = HealthServiceError(
            http_status=500,
            status="INFERENCE_FAILED",
            public_message="모델 추론을 완료하지 못했습니다.",
        )
        return _error_json(service, safe)
