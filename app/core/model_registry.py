"""Process-local lifecycle management for the two fixed inference models."""

from __future__ import annotations

import threading
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from app.core.config import HealthAPISettings
from scripts import predict_mushroom_health as predictor


ModelPair = tuple[Any, Any]
ModelLoader = Callable[..., ModelPair]


class RegistryState(str, Enum):
    UNLOADED = "UNLOADED"
    READY = "READY"
    FAILED = "FAILED"


class ModelRegistryError(RuntimeError):
    """Safe model lifecycle error without a local model path."""


def _class_names(wrapper: Any) -> dict[int, str]:
    source = getattr(wrapper, "model", wrapper)
    names = getattr(source, "names", None)
    if not isinstance(names, dict):
        raise ModelRegistryError("모델 class mapping을 확인할 수 없습니다")
    try:
        return {int(key): str(value) for key, value in names.items()}
    except (TypeError, ValueError) as exc:
        raise ModelRegistryError("모델 class mapping이 유효하지 않습니다") from exc


class ModelRegistry:
    """Loads one detector/classifier pair per application process."""

    def __init__(
        self,
        settings: HealthAPISettings,
        *,
        loader: ModelLoader | None = None,
        detector_path: Path | None = None,
        health_model_path: Path | None = None,
    ) -> None:
        self.settings = settings
        self._loader = loader or predictor.load_fixed_models
        self._detector_path = (
            detector_path
            if detector_path is not None
            else settings.detector_model_path
        )
        self._health_model_path = (
            health_model_path
            if health_model_path is not None
            else settings.health_model_path
        )
        self._models: ModelPair | None = None
        self._fingerprints: dict[Path, tuple[int, int, str]] | None = None
        self._state = RegistryState.UNLOADED
        self._load_attempts = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> RegistryState:
        return self._state

    @property
    def load_attempts(self) -> int:
        return self._load_attempts

    def _snapshot_models(self) -> dict[Path, tuple[int, int, str]]:
        paths = (self._detector_path, self._health_model_path)
        if any(not path.is_file() for path in paths):
            raise ModelRegistryError("고정 모델 파일을 찾을 수 없습니다")
        return {path: predictor.file_fingerprint(path) for path in paths}

    @staticmethod
    def _validate_class_mapping(detector: Any, classifier: Any) -> None:
        if _class_names(detector) != predictor.DETECTOR_MODEL_NAMES:
            raise ModelRegistryError("품종 모델 class mapping 불일치")
        if _class_names(classifier) != predictor.HEALTH_MODEL_NAMES:
            raise ModelRegistryError("건강 모델 class mapping 불일치")

    def load(self) -> ModelPair:
        with self._lock:
            if self._state == RegistryState.READY and self._models is not None:
                return self._models
            if self._state == RegistryState.FAILED:
                raise ModelRegistryError("모델 registry가 FAILED 상태입니다")
            self._load_attempts += 1
            try:
                before = self._snapshot_models()
                detector, classifier = self._loader(
                    device=self.settings.device,
                    verify_sha256=self.settings.verify_model_sha256,
                    detector_path=self._detector_path,
                    health_model_path=self._health_model_path,
                )
                self._validate_class_mapping(detector, classifier)
                after = self._snapshot_models()
                if after != before:
                    raise ModelRegistryError("모델 파일 변경이 감지되었습니다")
            except Exception as exc:
                self._models = None
                self._fingerprints = None
                self._state = RegistryState.FAILED
                raise ModelRegistryError(
                    "고정 모델을 로드하지 못했습니다"
                ) from exc
            self._models = (detector, classifier)
            self._fingerprints = before
            self._state = RegistryState.READY
            return self._models

    def get_models(self) -> ModelPair:
        with self._lock:
            if self._state != RegistryState.READY or self._models is None:
                raise ModelRegistryError("모델 registry가 준비되지 않았습니다")
            return self._models

    def assert_model_files_unchanged(self) -> None:
        with self._lock:
            if self._fingerprints is None:
                return
            if self._snapshot_models() != self._fingerprints:
                raise ModelRegistryError("모델 파일 변경이 감지되었습니다")

    def shutdown(self) -> None:
        with self._lock:
            if self._fingerprints is not None:
                if self._snapshot_models() != self._fingerprints:
                    self._state = RegistryState.FAILED
                    raise ModelRegistryError(
                        "모델 파일 변경이 감지되었습니다"
                    )
            self._models = None
            self._fingerprints = None
            if self._state != RegistryState.FAILED:
                self._state = RegistryState.UNLOADED

    def public_status(self) -> dict[str, Any]:
        return {
            "status": self._state.value,
            "loadAttempts": self._load_attempts,
            "detectorModel": predictor.DETECTOR_MODEL_NAME,
            "healthModel": predictor.HEALTH_MODEL_NAME,
        }
