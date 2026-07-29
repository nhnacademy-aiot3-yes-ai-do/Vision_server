"""Validated environment-backed configuration for the health-check API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from scripts import predict_mushroom_health as predictor


DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
HARD_MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def _parse_float(
    environment: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _parse_int(
    environment: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _parse_bool(
    environment: Mapping[str, str],
    name: str,
    default: bool,
) -> bool:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class HealthAPISettings:
    detection_confidence: float = predictor.DEFAULT_DETECTION_CONFIDENCE
    health_uncertain_threshold: float = predictor.DEFAULT_HEALTH_THRESHOLD
    padding_ratio: float = predictor.DEFAULT_PADDING_RATIO
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    verify_model_sha256: bool = True
    device: str = "auto"

    def __post_init__(self) -> None:
        if not 0.0 <= self.detection_confidence <= 1.0:
            raise ValueError(
                "HEALTH_DETECTION_CONFIDENCE must be between 0 and 1"
            )
        if not 0.0 <= self.health_uncertain_threshold <= 1.0:
            raise ValueError(
                "HEALTH_UNCERTAIN_THRESHOLD must be between 0 and 1"
            )
        if not 0.0 <= self.padding_ratio <= 0.5:
            raise ValueError("HEALTH_PADDING_RATIO must be between 0 and 0.5")
        if not 0 < self.max_upload_bytes <= HARD_MAX_UPLOAD_BYTES:
            raise ValueError(
                "HEALTH_MAX_UPLOAD_BYTES must be between 1 and 100 MiB"
            )
        if not self.device.strip():
            raise ValueError("HEALTH_DEVICE must not be empty")

    @classmethod
    def from_env(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> "HealthAPISettings":
        source = os.environ if environment is None else environment
        return cls(
            detection_confidence=_parse_float(
                source,
                "HEALTH_DETECTION_CONFIDENCE",
                predictor.DEFAULT_DETECTION_CONFIDENCE,
            ),
            health_uncertain_threshold=_parse_float(
                source,
                "HEALTH_UNCERTAIN_THRESHOLD",
                predictor.DEFAULT_HEALTH_THRESHOLD,
            ),
            padding_ratio=_parse_float(
                source,
                "HEALTH_PADDING_RATIO",
                predictor.DEFAULT_PADDING_RATIO,
            ),
            max_upload_bytes=_parse_int(
                source,
                "HEALTH_MAX_UPLOAD_BYTES",
                DEFAULT_MAX_UPLOAD_BYTES,
            ),
            verify_model_sha256=_parse_bool(
                source,
                "HEALTH_VERIFY_MODEL_SHA256",
                True,
            ),
            device=source.get("HEALTH_DEVICE", "auto").strip() or "auto",
        )
