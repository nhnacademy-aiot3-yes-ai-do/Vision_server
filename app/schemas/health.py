# 내부 추론 딕셔너리를 camelCase 기반의 검증된 공개 REST 응답으로 변환한다.
"""Validated public DTOs for mushroom health-check responses."""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from scripts import predict_mushroom_health as predictor


SpeciesCode = Literal[
    "OYSTER",
    "BUTTON",
    "KING_OYSTER",
    "ENOKI",
    "SHIITAKE",
]

# 모델 class id는 모델 내부 구현값이고, speciesCode는 서비스 간 업무 계약이다.
_SPECIES_CONTRACT: dict[int, tuple[str, SpeciesCode]] = {
    0: ("느타리", "OYSTER"),
    1: ("양송이", "BUTTON"),
    2: ("큰느타리", "KING_OYSTER"),
    3: ("팽이", "ENOKI"),
    4: ("표고", "SHIITAKE"),
}


def _species_code(class_id: object, species: object) -> SpeciesCode:
    contract = _SPECIES_CONTRACT.get(class_id) if isinstance(class_id, int) else None
    if contract is None or contract[0] != species:
        raise ValueError("내부 품종 식별 계약이 유효하지 않습니다")
    return contract[1]


# 모든 공개 모델에 camelCase 별칭, 필드명 입력 허용, 알 수 없는 필드 거부 정책을 적용한다.
class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )


# 실제 요청에 적용된 탐지·판정 임계값을 클라이언트가 해석할 수 있게 공개한다.
class HealthThresholds(CamelModel):
    detection: float = Field(ge=0.0, le=1.0)
    min_detection_confidence: float = Field(ge=0.0, le=1.0)
    health_uncertain: float = Field(ge=0.0, le=1.0)


# 같은 품종의 탐지 결과를 합친 뒤 건강 분류 결과와 영역 좌표를 담는 단위 결과이다.
class MushroomHealthResult(CamelModel):
    species: str
    species_code: SpeciesCode
    species_class_id: int = Field(ge=0, le=4)
    detected_count: int = Field(ge=1)
    detection_confidence: float = Field(ge=0.0, le=1.0)
    detection_confidence_min: float = Field(ge=0.0, le=1.0)
    health_status: Literal[
        "HEALTHY",
        "DISEASE_SUSPECTED",
        "UNCERTAIN",
    ]
    health_confidence: float | None = Field(ge=0.0, le=1.0)
    healthy_probability: float | None = Field(ge=0.0, le=1.0)
    disease_suspected_probability: float | None = Field(ge=0.0, le=1.0)
    bbox: list[int] = Field(min_length=4, max_length=4)
    crop_bbox: list[int] = Field(min_length=4, max_length=4)


# 성공·미탐지·안전한 오류가 공통으로 따르는 최상위 응답 계약이다.
class HealthCheckResponse(CamelModel):
    analysis_type: Literal["MUSHROOM_HEALTH_CHECK_V1"]
    status: str
    detector_model: str
    health_model: str
    thresholds: HealthThresholds
    results: list[MushroomHealthResult]
    warnings: list[str]

    # predictor의 내부 snake_case 응답에서 허용된 공개 필드만 골라 Pydantic으로 재검증한다.
    @classmethod
    def from_internal(
        cls,
        internal: Mapping[str, Any],
    ) -> "HealthCheckResponse":
        # 변환 전에도 로컬 경로 등 식별 가능 정보가 없는지 방어적으로 확인한다.
        predictor.assert_deidentified_response(internal)
        thresholds = internal.get("thresholds")
        if not isinstance(thresholds, Mapping):
            raise ValueError("내부 threshold 응답이 유효하지 않습니다")
        public_results: list[dict[str, Any]] = []
        for raw in internal.get("results", []):
            # 예상하지 못한 자료형을 조용히 직렬화하지 않고 서버 오류로 드러내 계약을 지킨다.
            if not isinstance(raw, Mapping):
                raise ValueError("내부 result 응답이 유효하지 않습니다")
            class_id = raw.get("class_id")
            species = raw.get("species")
            public_results.append(
                {
                    "species": species,
                    "species_code": _species_code(class_id, species),
                    "species_class_id": class_id,
                    "detected_count": raw.get("detected_count"),
                    "detection_confidence": raw.get(
                        "detection_confidence"
                    ),
                    "detection_confidence_min": raw.get(
                        "detection_confidence_min"
                    ),
                    "health_status": raw.get("health_status"),
                    "health_confidence": raw.get("health_confidence"),
                    "healthy_probability": raw.get("healthy_probability"),
                    "disease_suspected_probability": raw.get(
                        "disease_suspected_probability"
                    ),
                    "bbox": raw.get("bbox"),
                    "crop_bbox": raw.get("crop_bbox"),
                }
            )
        # model_validate가 필수값, 허용 상태, 확률 범위, 좌표 길이를 마지막으로 검증한다.
        return cls.model_validate(
            {
                "analysis_type": internal.get("analysis_type"),
                "status": internal.get("status"),
                "detector_model": internal.get("detector_model"),
                "health_model": internal.get("health_model"),
                "thresholds": {
                    "detection": thresholds.get("detection"),
                    "min_detection_confidence": thresholds.get(
                        "min_detection_confidence"
                    ),
                    "health_uncertain": thresholds.get(
                        "health_uncertain"
                    ),
                },
                "results": public_results,
                "warnings": list(internal.get("warnings", [])),
            }
        )


# 서비스 오류도 predictor의 표준 빈 결과 형태를 거쳐 정상 응답과 같은 스키마로 만든다.
def safe_error_response(
    *,
    status: str,
    detection_threshold: float,
    min_detection_confidence: float,
    health_threshold: float,
    public_message: str,
) -> HealthCheckResponse:
    internal = predictor.invalid_response(
        status,
        detection_threshold=detection_threshold,
        min_detection_confidence=min_detection_confidence,
        health_threshold=health_threshold,
        detail=public_message,
    )
    return HealthCheckResponse.from_internal(internal)
