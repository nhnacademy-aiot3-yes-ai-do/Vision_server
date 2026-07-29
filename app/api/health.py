"""Versioned mushroom health-check endpoint."""

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
router = APIRouter(prefix="/api/v1/mushroom", tags=["mushroom-health"])


async def get_health_service(request: Request) -> MushroomHealthService:
    service = getattr(request.app.state, "health_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Service unavailable")
    return service


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
    try:
        internal = await service.analyze_upload(image)
        return HealthCheckResponse.from_internal(internal)
    except HealthServiceError as exc:
        if exc.http_status >= 500:
            LOGGER.exception("Mushroom health inference failed")
        return _error_json(service, exc)
    except Exception:
        LOGGER.exception("Unexpected mushroom health API failure")
        safe = HealthServiceError(
            http_status=500,
            status="INFERENCE_FAILED",
            public_message="모델 추론을 완료하지 못했습니다.",
        )
        return _error_json(service, safe)
