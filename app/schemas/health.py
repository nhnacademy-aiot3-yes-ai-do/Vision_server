"""Validated public DTOs for mushroom health-check responses."""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from scripts import predict_mushroom_health as predictor


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )


class HealthThresholds(CamelModel):
    detection: float = Field(ge=0.0, le=1.0)
    health_uncertain: float = Field(ge=0.0, le=1.0)


class MushroomHealthResult(CamelModel):
    species: str
    species_class_id: int = Field(ge=0, le=4)
    detected_count: int = Field(ge=1)
    detection_confidence: float = Field(ge=0.0, le=1.0)
    detection_confidence_min: float = Field(ge=0.0, le=1.0)
    health_status: Literal[
        "HEALTHY",
        "DISEASE_SUSPECTED",
        "UNCERTAIN",
    ]
    health_confidence: float = Field(ge=0.0, le=1.0)
    healthy_probability: float = Field(ge=0.0, le=1.0)
    disease_suspected_probability: float = Field(ge=0.0, le=1.0)
    bbox: list[int] = Field(min_length=4, max_length=4)
    crop_bbox: list[int] = Field(min_length=4, max_length=4)


class HealthCheckResponse(CamelModel):
    analysis_type: Literal["MUSHROOM_HEALTH_CHECK_V1"]
    status: str
    detector_model: str
    health_model: str
    thresholds: HealthThresholds
    results: list[MushroomHealthResult]
    warnings: list[str]

    @classmethod
    def from_internal(
        cls,
        internal: Mapping[str, Any],
    ) -> "HealthCheckResponse":
        predictor.assert_deidentified_response(internal)
        thresholds = internal.get("thresholds")
        if not isinstance(thresholds, Mapping):
            raise ValueError("내부 threshold 응답이 유효하지 않습니다")
        public_results: list[dict[str, Any]] = []
        for raw in internal.get("results", []):
            if not isinstance(raw, Mapping):
                raise ValueError("내부 result 응답이 유효하지 않습니다")
            public_results.append(
                {
                    "species": raw.get("species"),
                    "species_class_id": raw.get("class_id"),
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
        return cls.model_validate(
            {
                "analysis_type": internal.get("analysis_type"),
                "status": internal.get("status"),
                "detector_model": internal.get("detector_model"),
                "health_model": internal.get("health_model"),
                "thresholds": {
                    "detection": thresholds.get("detection"),
                    "health_uncertain": thresholds.get(
                        "health_uncertain"
                    ),
                },
                "results": public_results,
                "warnings": list(internal.get("warnings", [])),
            }
        )


def safe_error_response(
    *,
    status: str,
    detection_threshold: float,
    health_threshold: float,
    public_message: str,
) -> HealthCheckResponse:
    internal = predictor.invalid_response(
        status,
        detection_threshold=detection_threshold,
        health_threshold=health_threshold,
        detail=public_message,
    )
    return HealthCheckResponse.from_internal(internal)
