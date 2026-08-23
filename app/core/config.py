# 환경 변수 문자열을 타입이 보장된 설정 객체로 바꾸고 잘못된 값은 시작 단계에서 거부한다.
"""Validated environment-backed configuration for the health-check API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from scripts import predict_mushroom_health as predictor


# 일반 업로드 기본 한도와 설정으로도 넘을 수 없는 절대 상한을 분리한다.
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
HARD_MAX_UPLOAD_BYTES = 100 * 1024 * 1024


# 값이 없거나 공백이면 기본값을 쓰고, 값이 있으면 실수로 엄격하게 변환한다.
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


# 바이트 제한처럼 정수여야 하는 환경 변수를 변환한다.
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


# 운영 환경에서 자주 쓰는 여러 참·거짓 표기를 bool 값으로 정규화한다.
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


# 모델 경로 문자열의 앞뒤 공백과 사용자 홈 표기를 정리해 Path로 만든다.
def _parse_path(
    environment: Mapping[str, str],
    name: str,
    default: Path,
) -> Path:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return default
    return Path(raw.strip()).expanduser()


# 모델 위치, 추론 임계값, 업로드 정책을 한 번 검증한 뒤 변경 불가능하게 보관한다.
@dataclass(frozen=True)
class HealthAPISettings:
    """Validated process configuration for models, thresholds, and uploads."""

    detector_model_path: Path = predictor.DETECTOR_MODEL_PATH
    health_model_path: Path = predictor.HEALTH_MODEL_PATH
    detection_confidence: float = predictor.DEFAULT_DETECTION_CONFIDENCE
    min_detection_confidence: float = (
        predictor.DEFAULT_MIN_DETECTION_CONFIDENCE
    )
    health_uncertain_threshold: float = predictor.DEFAULT_HEALTH_THRESHOLD
    padding_ratio: float = predictor.DEFAULT_PADDING_RATIO
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    verify_model_sha256: bool = True
    device: str = "auto"

    # 객체 생성 경로와 관계없이 모든 숫자 범위 및 필수 문자열 제약을 동일하게 검사한다.
    def __post_init__(self) -> None:
        if not 0.0 <= self.detection_confidence <= 1.0:
            raise ValueError(
                "HEALTH_DETECTION_CONFIDENCE must be between 0 and 1"
            )
        if not 0.0 <= self.min_detection_confidence <= 1.0:
            raise ValueError(
                "HEALTH_MIN_DETECTION_CONFIDENCE must be between 0 and 1"
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

    # 주입된 매핑 또는 실제 프로세스 환경 변수를 읽어 완전한 설정 객체를 만든다.
    @classmethod
    def from_env(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> "HealthAPISettings":
        # 테스트에서는 별도 매핑을 주입하고 실제 실행에서는 os.environ을 사용한다.
        source = os.environ if environment is None else environment
        return cls(
            detector_model_path=_parse_path(
                source,
                "DETECTOR_MODEL_PATH",
                predictor.DETECTOR_MODEL_PATH,
            ),
            health_model_path=_parse_path(
                source,
                "HEALTH_MODEL_PATH",
                predictor.HEALTH_MODEL_PATH,
            ),
            detection_confidence=_parse_float(
                source,
                "HEALTH_DETECTION_CONFIDENCE",
                predictor.DEFAULT_DETECTION_CONFIDENCE,
            ),
            min_detection_confidence=_parse_float(
                source,
                "HEALTH_MIN_DETECTION_CONFIDENCE",
                predictor.DEFAULT_MIN_DETECTION_CONFIDENCE,
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
