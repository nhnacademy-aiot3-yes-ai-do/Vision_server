# 탐지·분류 모델을 프로세스당 한 번만 로드하고 파일 변경과 준비 상태를 관리한다.
"""Process-local lifecycle management for the two fixed inference models."""

from __future__ import annotations

import threading
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from app.core.config import HealthAPISettings
from scripts import predict_mushroom_health as predictor


# 레지스트리는 항상 품종 탐지기와 건강 분류기를 한 쌍으로 취급한다.
ModelPair = tuple[Any, Any]
ModelLoader = Callable[..., ModelPair]


# 모델 쌍의 생명주기를 아직 미로드, 사용 가능, 복구 없이 실패한 상태로 구분한다.
class RegistryState(str, Enum):
    UNLOADED = "UNLOADED"
    READY = "READY"
    FAILED = "FAILED"


# 내부 경로나 원래 로더 예외를 공개하지 않고 레지스트리 실패를 표현한다.
class ModelRegistryError(RuntimeError):
    """Safe model lifecycle error without a local model path."""


# Ultralytics wrapper 안쪽의 class 이름 매핑을 정수 키·문자열 값으로 정규화한다.
def _class_names(wrapper: Any) -> dict[int, str]:
    source = getattr(wrapper, "model", wrapper)
    names = getattr(source, "names", None)
    if not isinstance(names, dict):
        raise ModelRegistryError("모델 class mapping을 확인할 수 없습니다")
    try:
        return {int(key): str(value) for key, value in names.items()}
    except (TypeError, ValueError) as exc:
        raise ModelRegistryError("모델 class mapping이 유효하지 않습니다") from exc


# 하나의 애플리케이션 프로세스에서 고정 모델 쌍과 그 파일 지문을 보관한다.
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
        # 테스트에서는 loader와 경로를 주입할 수 있고, 운영에서는 검증된 기본 구현과 설정 경로를 쓴다.
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
        # 여러 요청이 동시에 시작되어도 로드·종료 상태 전이는 한 스레드씩만 수행한다.
        self._lock = threading.Lock()

    # readiness API가 모델 객체를 가져오지 않고도 읽을 수 있는 현재 생명주기 상태이다.
    @property
    def state(self) -> RegistryState:
        return self._state

    # 실제 로더 호출 시도 횟수로, READY 캐시 재사용은 증가시키지 않는다.
    @property
    def load_attempts(self) -> int:
        return self._load_attempts

    # 파일 크기·수정 시각·SHA-256을 함께 기록해 실행 중 교체 여부를 판별한다.
    def _snapshot_models(self) -> dict[Path, tuple[int, int, str]]:
        paths = (self._detector_path, self._health_model_path)
        if any(not path.is_file() for path in paths):
            raise ModelRegistryError("고정 모델 파일을 찾을 수 없습니다")
        return {path: predictor.file_fingerprint(path) for path in paths}

    # 실수로 다른 학습 가중치를 배포하지 않도록 두 모델의 클래스 순서를 고정 계약과 비교한다.
    @staticmethod
    def _validate_class_mapping(detector: Any, classifier: Any) -> None:
        if _class_names(detector) != predictor.DETECTOR_MODEL_NAMES:
            raise ModelRegistryError("품종 모델 class mapping 불일치")
        if _class_names(classifier) != predictor.HEALTH_MODEL_NAMES:
            raise ModelRegistryError("건강 모델 class mapping 불일치")

    # 모델 쌍을 최초 한 번 로드하고 검증에 모두 성공한 경우에만 READY 캐시에 게시한다.
    def load(self) -> ModelPair:
        with self._lock:
            # 이미 준비된 레지스트리는 같은 객체 쌍을 그대로 돌려준다.
            if self._state == RegistryState.READY and self._models is not None:
                return self._models
            # 실패 상태에서 임의 재시도하지 않아 부분 초기화나 서로 다른 상태가 섞이는 것을 막는다.
            if self._state == RegistryState.FAILED:
                raise ModelRegistryError("모델 registry가 FAILED 상태입니다")
            self._load_attempts += 1
            try:
                # 로드 전후 지문이 같아야 로딩 도중 파일 교체가 없었다고 확정할 수 있다.
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
                # 어느 검증 단계든 실패하면 부분적으로 만든 모델을 캐시하지 않고 FAILED로 고정한다.
                self._models = None
                self._fingerprints = None
                self._state = RegistryState.FAILED
                raise ModelRegistryError(
                    "고정 모델을 로드하지 못했습니다"
                ) from exc
            # 모든 검증 후에만 모델과 기준 지문을 원자적으로 공개한다.
            self._models = (detector, classifier)
            self._fingerprints = before
            self._state = RegistryState.READY
            return self._models

    # 추론 경로에서는 자동 로드하지 않고 READY 상태의 기존 모델만 반환한다.
    def get_models(self) -> ModelPair:
        with self._lock:
            if self._state != RegistryState.READY or self._models is None:
                raise ModelRegistryError("모델 registry가 준비되지 않았습니다")
            return self._models

    # 운영 중 모델 파일이 기준 지문과 같은지 명시적으로 확인하는 무결성 검사이다.
    def assert_model_files_unchanged(self) -> None:
        with self._lock:
            if self._fingerprints is None:
                return
            if self._snapshot_models() != self._fingerprints:
                raise ModelRegistryError("모델 파일 변경이 감지되었습니다")

    # 종료 시 마지막 무결성을 검사한 뒤 메모리 참조와 기준 지문을 제거한다.
    def shutdown(self) -> None:
        with self._lock:
            if self._fingerprints is not None:
                if self._snapshot_models() != self._fingerprints:
                    # 변경된 파일을 정상 종료로 감추지 않고 FAILED 상태와 예외로 남긴다.
                    self._state = RegistryState.FAILED
                    raise ModelRegistryError(
                        "모델 파일 변경이 감지되었습니다"
                    )
            self._models = None
            self._fingerprints = None
            if self._state != RegistryState.FAILED:
                self._state = RegistryState.UNLOADED

    # 모니터링에 필요한 논리 이름과 상태만 공개하고 실제 로컬 경로는 제외한다.
    def public_status(self) -> dict[str, Any]:
        return {
            "status": self._state.value,
            "loadAttempts": self._load_attempts,
            "detectorModel": predictor.DETECTOR_MODEL_NAME,
            "healthModel": predictor.HEALTH_MODEL_NAME,
        }
