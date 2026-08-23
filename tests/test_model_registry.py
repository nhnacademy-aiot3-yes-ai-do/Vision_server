# 실제 대형 모델 대신 임시 파일과 wrapper 대역으로 레지스트리의 상태 전이·무결성을 검증한다.
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest


# 어느 위치에서 pytest를 실행해도 프로젝트 모듈을 가져오도록 import 루트를 맞춘다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.core.config import HealthAPISettings
from app.core.model_registry import (
    ModelRegistry,
    ModelRegistryError,
    RegistryState,
)
from scripts import predict_mushroom_health as predictor


# 테스트 로더도 운영 로더와 같은 detector/classifier 쌍을 반환하도록 타입을 맞춘다.
ModelLoader = Callable[..., tuple[Any, Any]]


# 모델 파일 존재 여부와 지문 검사를 통과할 작은 임시 가중치 파일 두 개를 만든다.
def _fake_weights(tmp_path: Path) -> tuple[Path, Path]:
    detector = tmp_path / "detector.pt"
    classifier = tmp_path / "health.pt"
    detector.write_bytes(b"synthetic detector weights")
    classifier.write_bytes(b"synthetic health weights")
    return detector, classifier


# 운영 지문과 같은 크기·수정 시각·SHA-256 조합을 테스트 전후 비교용으로 계산한다.
def _fingerprint(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    return (
        stat.st_size,
        stat.st_mtime_ns,
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


# Ultralytics처럼 `wrapper.model.names`에 클래스 매핑이 있는 최소 대역을 만든다.
def _wrapper(names: dict[int, str]) -> SimpleNamespace:
    # Mirror the production Ultralytics wrappers: class names live on
    # ``wrapper.model.names``, rather than directly on the wrapper.
    return SimpleNamespace(model=SimpleNamespace(names=dict(names)))


# 운영 계약과 정확히 같은 클래스 매핑을 가진 유효한 탐지기·분류기 쌍이다.
def _valid_pair() -> tuple[SimpleNamespace, SimpleNamespace]:
    return (
        _wrapper(predictor.DETECTOR_MODEL_NAMES),
        _wrapper(predictor.HEALTH_MODEL_NAMES),
    )


# 테스트별 설정과 임시 모델 경로가 연결된 새 레지스트리를 만드는 공통 도우미이다.
def _registry(
    tmp_path: Path,
    loader: ModelLoader,
    *,
    verify_sha256: bool = True,
    device: str = "cpu",
) -> tuple[ModelRegistry, Path, Path]:
    detector_path, health_path = _fake_weights(tmp_path)
    settings = HealthAPISettings(
        device=device,
        verify_model_sha256=verify_sha256,
    )
    return (
        ModelRegistry(
            settings,
            loader=loader,
            detector_path=detector_path,
            health_model_path=health_path,
        ),
        detector_path,
        health_path,
    )


# 두 번 load해도 로더는 한 번만 실행되고 같은 모델 객체가 재사용되는지 확인한다.
def test_load_is_singleton_and_accepts_wrapper_model_names(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    pair = _valid_pair()

    def loader(**kwargs: object) -> tuple[Any, Any]:
        calls.append(dict(kwargs))
        return pair

    registry, _, _ = _registry(tmp_path, loader)

    first = registry.load()
    second = registry.load()

    assert first is second
    assert first[0] is pair[0]
    assert first[1] is pair[1]
    assert registry.get_models() is first
    assert registry.state is RegistryState.READY
    assert registry.load_attempts == 1
    detector_path, health_path = (
        registry._detector_path,
        registry._health_model_path,
    )
    assert calls == [
        {
            "device": "cpu",
            "verify_sha256": True,
            "detector_path": detector_path,
            "health_model_path": health_path,
        }
    ]


# 여러 스레드가 동시에 load해도 생명주기 잠금이 로더를 정확히 한 번만 실행하는지 확인한다.
def test_concurrent_load_calls_loader_exactly_once(tmp_path: Path) -> None:
    worker_count = 12
    start = threading.Barrier(worker_count)
    calls = 0
    calls_lock = threading.Lock()
    pair = _valid_pair()

    def loader(**_kwargs: object) -> tuple[Any, Any]:
        nonlocal calls
        with calls_lock:
            calls += 1
        # Keep the first caller in the loader briefly so the remaining
        # callers contend on ModelRegistry's lifecycle lock.
        time.sleep(0.03)
        return pair

    registry, _, _ = _registry(tmp_path, loader)

    def load_from_worker() -> tuple[Any, Any]:
        start.wait(timeout=5)
        return registry.load()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(
            executor.map(lambda _index: load_from_worker(), range(worker_count))
        )

    assert calls == 1
    assert registry.load_attempts == 1
    assert registry.state is RegistryState.READY
    assert all(result is results[0] for result in results)
    assert results[0][0] is pair[0]
    assert results[0][1] is pair[1]


# 탐지기나 분류기의 클래스 매핑이 다르면 캐시 없이 FAILED로 전환되는지 확인한다.
@pytest.mark.parametrize("invalid_model", ["detector", "classifier"])
def test_class_mapping_failure_sets_failed_without_cached_models(
    tmp_path: Path,
    invalid_model: str,
) -> None:
    detector, classifier = _valid_pair()
    if invalid_model == "detector":
        detector = _wrapper({0: "wrong-species"})
    else:
        classifier = _wrapper({0: "HEALTHY", 1: "wrong-health-class"})

    registry, _, _ = _registry(
        tmp_path,
        lambda **_kwargs: (detector, classifier),
    )

    with pytest.raises(ModelRegistryError, match="로드하지 못했습니다"):
        registry.load()

    assert registry.state is RegistryState.FAILED
    assert registry.load_attempts == 1
    assert registry._models is None
    assert registry._fingerprints is None
    with pytest.raises(ModelRegistryError, match="준비되지 않았습니다"):
        registry.get_models()


# SHA 검증 플래그와 장치 문자열이 변형 없이 실제 모델 로더에 전달되는지 확인한다.
@pytest.mark.parametrize("verify_sha256", [True, False])
def test_approved_sha_verification_flag_and_device_are_forwarded(
    tmp_path: Path,
    verify_sha256: bool,
) -> None:
    received: list[dict[str, object]] = []

    def loader(**kwargs: object) -> tuple[Any, Any]:
        received.append(dict(kwargs))
        return _valid_pair()

    registry, _, _ = _registry(
        tmp_path,
        loader,
        verify_sha256=verify_sha256,
        device="cuda:7",
    )

    registry.load()

    assert received == [
        {
            "device": "cuda:7",
            "verify_sha256": verify_sha256,
            "detector_path": registry._detector_path,
            "health_model_path": registry._health_model_path,
        }
    ]


# 설정 객체에 지정한 두 모델 경로가 기본값 대신 로더 인자로 전달되는지 확인한다.
def test_settings_model_paths_are_forwarded_to_loader(
    tmp_path: Path,
) -> None:
    detector_path, health_path = _fake_weights(tmp_path)
    settings = HealthAPISettings(
        detector_model_path=detector_path,
        health_model_path=health_path,
        verify_model_sha256=False,
        device="cpu",
    )
    received: list[dict[str, object]] = []

    def loader(**kwargs: object) -> tuple[Any, Any]:
        received.append(dict(kwargs))
        return _valid_pair()

    registry = ModelRegistry(settings, loader=loader)
    registry.load()

    assert received == [
        {
            "device": "cpu",
            "verify_sha256": False,
            "detector_path": detector_path,
            "health_model_path": health_path,
        }
    ]


# 환경 변수의 장치 값, 특히 공백 기본값이 정규화되어 로더까지 이어지는지 확인한다.
@pytest.mark.parametrize(
    ("raw_device", "expected_device"),
    [
        ("cpu", "cpu"),
        ("mps", "mps"),
        ("cuda:0", "cuda:0"),
        (" \t ", "auto"),
    ],
)
def test_health_device_environment_is_normalized_and_forwarded_to_loader(
    tmp_path: Path,
    raw_device: str,
    expected_device: str,
) -> None:
    detector_path, health_path = _fake_weights(tmp_path)
    settings = HealthAPISettings.from_env(
        {
            "DETECTOR_MODEL_PATH": str(detector_path),
            "HEALTH_MODEL_PATH": str(health_path),
            "HEALTH_VERIFY_MODEL_SHA256": "false",
            "HEALTH_DEVICE": raw_device,
        }
    )
    received: list[dict[str, object]] = []

    def loader(**kwargs: object) -> tuple[Any, Any]:
        received.append(dict(kwargs))
        return _valid_pair()

    registry = ModelRegistry(settings, loader=loader)
    registry.load()

    assert settings.device == expected_device
    assert received == [
        {
            "device": expected_device,
            "verify_sha256": False,
            "detector_path": detector_path,
            "health_model_path": health_path,
        }
    ]


# 첫 모델 생성 뒤 두 번째 모델 로드가 실패해도 부분 모델이 캐시에 남지 않는지 확인한다.
def test_partial_loader_failure_leaves_zero_cached_models(
    tmp_path: Path,
) -> None:
    created_detector = _wrapper(predictor.DETECTOR_MODEL_NAMES)

    def loader(**_kwargs: object) -> tuple[Any, Any]:
        # Simulate detector construction succeeding before classifier loading
        # fails. The partially created wrapper must never enter the registry.
        assert created_detector.model.names == predictor.DETECTOR_MODEL_NAMES
        raise RuntimeError("synthetic classifier load failure")

    registry, _, _ = _registry(tmp_path, loader)

    with pytest.raises(ModelRegistryError, match="로드하지 못했습니다") as error:
        registry.load()

    assert isinstance(error.value.__cause__, RuntimeError)
    assert registry.state is RegistryState.FAILED
    assert registry.load_attempts == 1
    assert registry._models is None
    assert registry._fingerprints is None
    with pytest.raises(ModelRegistryError, match="FAILED"):
        registry.load()
    assert registry.load_attempts == 1


# READY 이전 get_models가 암묵적 로드를 하지 않고 명확히 실패하는지 확인한다.
def test_get_models_before_ready_fails_without_loading(tmp_path: Path) -> None:
    loader_calls = 0

    def loader(**_kwargs: object) -> tuple[Any, Any]:
        nonlocal loader_calls
        loader_calls += 1
        return _valid_pair()

    registry, _, _ = _registry(tmp_path, loader)

    with pytest.raises(ModelRegistryError, match="준비되지 않았습니다"):
        registry.get_models()

    assert loader_calls == 0
    assert registry.load_attempts == 0
    assert registry.state is RegistryState.UNLOADED


# 정상 종료가 모델 파일을 바꾸지 않고 메모리 캐시와 지문만 해제하는지 확인한다.
def test_shutdown_clears_cache_without_changing_model_files(
    tmp_path: Path,
) -> None:
    registry, detector_path, health_path = _registry(
        tmp_path,
        lambda **_kwargs: _valid_pair(),
    )
    before = {
        detector_path: _fingerprint(detector_path),
        health_path: _fingerprint(health_path),
    }

    registry.load()
    registry.shutdown()

    assert {
        detector_path: _fingerprint(detector_path),
        health_path: _fingerprint(health_path),
    } == before
    assert registry.state is RegistryState.UNLOADED
    assert registry._models is None
    assert registry._fingerprints is None
    with pytest.raises(ModelRegistryError, match="준비되지 않았습니다"):
        registry.get_models()


# 공개 상태에는 논리적 모델 이름만 있고 실제 로컬 경로는 없는지 확인한다.
def test_public_status_never_exposes_model_or_local_paths() -> None:
    settings = HealthAPISettings(device="cpu")
    detector_path = Path("/mnt/d/private/models/detector.pt")
    health_path = Path("/home/kim75/private/models/health.pt")
    registry = ModelRegistry(
        settings,
        loader=lambda **_kwargs: _valid_pair(),
        detector_path=detector_path,
        health_model_path=health_path,
    )

    status = registry.public_status()
    serialized = json.dumps(status, ensure_ascii=False, allow_nan=False)

    assert status == {
        "status": "UNLOADED",
        "loadAttempts": 0,
        "detectorModel": predictor.DETECTOR_MODEL_NAME,
        "healthModel": predictor.HEALTH_MODEL_NAME,
    }
    assert str(detector_path) not in serialized
    assert str(health_path) not in serialized
    assert "/mnt/" not in serialized
    assert "/home/" not in serialized
    predictor.assert_deidentified_response(status)


# 모든 지원 환경 변수가 올바른 타입과 경로로 변환되어 설정 객체에 반영되는지 확인한다.
def test_settings_are_loaded_from_environment_and_validated() -> None:
    settings = HealthAPISettings.from_env(
        {
            "DETECTOR_MODEL_PATH": "runtime/models/detector/best.pt",
            "HEALTH_MODEL_PATH": "runtime/models/health/best.pt",
            "HEALTH_DETECTION_CONFIDENCE": "0.31",
            "HEALTH_MIN_DETECTION_CONFIDENCE": "0.55",
            "HEALTH_UNCERTAIN_THRESHOLD": "0.81",
            "HEALTH_PADDING_RATIO": "0.2",
            "HEALTH_MAX_UPLOAD_BYTES": "2048",
            "HEALTH_VERIFY_MODEL_SHA256": "false",
            "HEALTH_DEVICE": "cpu",
        }
    )

    assert settings == HealthAPISettings(
        detector_model_path=Path("runtime/models/detector/best.pt"),
        health_model_path=Path("runtime/models/health/best.pt"),
        detection_confidence=0.31,
        min_detection_confidence=0.55,
        health_uncertain_threshold=0.81,
        padding_ratio=0.2,
        max_upload_bytes=2048,
        verify_model_sha256=False,
        device="cpu",
    )


# 모델 경로 환경 변수가 비어 있으면 predictor의 로컬 기본 경로를 유지하는지 확인한다.
def test_blank_model_path_environment_uses_local_defaults() -> None:
    settings = HealthAPISettings.from_env(
        {
            "DETECTOR_MODEL_PATH": "  ",
            "HEALTH_MODEL_PATH": "",
        }
    )

    assert settings.detector_model_path == predictor.DETECTOR_MODEL_PATH
    assert settings.health_model_path == predictor.HEALTH_MODEL_PATH


# 허용 범위를 벗어난 숫자와 알 수 없는 bool 표기를 애플리케이션 시작 전에 거부하는지 확인한다.
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("HEALTH_DETECTION_CONFIDENCE", "1.1"),
        ("HEALTH_MIN_DETECTION_CONFIDENCE", "1.01"),
        ("HEALTH_UNCERTAIN_THRESHOLD", "-0.1"),
        ("HEALTH_PADDING_RATIO", "0.51"),
        ("HEALTH_MAX_UPLOAD_BYTES", "0"),
        ("HEALTH_VERIFY_MODEL_SHA256", "maybe"),
    ],
)
def test_invalid_environment_settings_fail_fast(
    name: str,
    value: str,
) -> None:
    with pytest.raises(ValueError):
        HealthAPISettings.from_env({name: value})


# 실제 고정 모델 로드는 명시적 환경 변수로 허용한 통합 테스트에서만 실행하고 파일 불변성을 확인한다.
@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_INTEGRATION_TESTS", "").strip().lower()
    != "true",
    reason=(
        "실제 고정 모델 로드는 RUN_MODEL_INTEGRATION_TESTS=true일 때만 허용"
    ),
)
def test_fixed_model_registry_integration_load_is_explicitly_opt_in() -> None:
    settings = HealthAPISettings(
        device=os.environ.get("HEALTH_DEVICE", "cpu"),
        verify_model_sha256=True,
    )
    registry = ModelRegistry(settings)
    paths = tuple(predictor.EXPECTED_MODEL_SHA256)
    before = {path: _fingerprint(path) for path in paths}

    try:
        detector, classifier = registry.load()
        assert registry.state is RegistryState.READY
        assert registry.get_models() == (detector, classifier)
        assert detector.model.names == predictor.DETECTOR_MODEL_NAMES
        assert classifier.model.names == predictor.HEALTH_MODEL_NAMES
        registry.assert_model_files_unchanged()
    finally:
        registry.shutdown()

    assert {path: _fingerprint(path) for path in paths} == before
