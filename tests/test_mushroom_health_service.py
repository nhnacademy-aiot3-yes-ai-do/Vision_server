# 업로드 검증부터 잠금 기반 추론까지 서비스 계층을 모델 대역으로 독립 검증한다.
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import io
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import pytest
from PIL import Image


# pytest를 어느 작업 디렉터리에서 실행해도 프로젝트 패키지를 import하도록 루트를 추가한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.core.config import HealthAPISettings
from app.core.model_registry import ModelRegistryError
from app.services import mushroom_health_service as service_module
from app.services.mushroom_health_service import (
    HealthServiceError,
    MushroomHealthService,
    decode_image_bytes,
)
from scripts import predict_mushroom_health as predictor


# autouse fixture가 함수를 바꾸기 전에 운영 executor 구현을 별도로 보관해 동작 자체도 테스트한다.
PRODUCTION_RUN_SYNC_IN_EXECUTOR = service_module.run_sync_in_executor


# 대부분의 서비스 단위 테스트에서는 스레드를 없애고 동기 작업을 즉시 실행해 결과에만 집중한다.
@pytest.fixture(autouse=True)
def run_service_thread_work_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep unit tests independent from worker-thread lifecycle behavior."""

    async def run_sync(
        _executor: Any,
        function: Any,
    ) -> Any:
        return function()

    monkeypatch.setattr(service_module, "run_sync_in_executor", run_sync)


# filename·MIME·파일 핸들·비동기 close를 가진 UploadFile 최소 대역이다.
class FakeUpload:
    def __init__(
        self,
        payload: bytes,
        *,
        filename: str = "mushroom.jpg",
        content_type: str = "image/jpeg",
    ) -> None:
        self.filename = filename
        self.content_type = content_type
        self._buffer = io.BytesIO(payload)
        self.file = self._buffer

    @property
    def closed(self) -> bool:
        return self._buffer.closed

    async def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    async def close(self) -> None:
        self._buffer.close()


# 지정된 탐지 목록을 반환하며 호출 횟수와 입력 이미지 크기를 기록한다.
class FakeDetector:
    model_name = "fake-detector"

    def __init__(
        self,
        detections: Sequence[predictor.Detection],
    ) -> None:
        self.detections = list(detections)
        self.calls = 0
        self.image_sizes: list[tuple[int, int]] = []

    def predict(
        self,
        image: Image.Image,
    ) -> list[predictor.Detection]:
        self.calls += 1
        self.image_sizes.append(image.size)
        return list(self.detections)


# 지정 확률을 반환하며 분류 호출 횟수와 crop 크기를 기록한다.
class FakeClassifier:
    model_name = "fake-classifier"

    def __init__(self, probabilities: tuple[float, float]) -> None:
        self.probabilities = probabilities
        self.calls = 0
        self.crop_sizes: list[tuple[int, int]] = []

    def predict(self, crop: Image.Image) -> tuple[float, float]:
        self.calls += 1
        self.crop_sizes.append(crop.size)
        return self.probabilities


# 서비스가 매 요청마다 준비된 모델 쌍만 조회하는지 관찰하는 레지스트리 대역이다.
class FakeRegistry:
    def __init__(self, detector: Any, classifier: Any) -> None:
        self.detector = detector
        self.classifier = classifier
        self.get_calls = 0

    def get_models(self) -> tuple[Any, Any]:
        self.get_calls += 1
        return self.detector, self.classifier


# 내부 경로가 포함된 레지스트리 오류를 의도적으로 발생시켜 비식별 오류 변환을 시험한다.
class FailingRegistry:
    def get_models(self) -> tuple[Any, Any]:
        raise ModelRegistryError(
            "secret model failure at /home/kim75/private/best.pt"
        )


# EXIF 방향 정보를 선택적으로 포함한 유효 JPEG를 메모리에서 생성한다.
def jpeg_bytes(
    *,
    size: tuple[int, int] = (100, 80),
    orientation: int | None = None,
) -> bytes:
    image = Image.new("RGB", size, (40, 90, 55))
    output = io.BytesIO()
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation
    image.save(output, format="JPEG", quality=95, exif=exif)
    return output.getvalue()


# 이미지 형식과 픽셀 제한 검증에 사용할 유효 PNG를 메모리에서 생성한다.
def png_bytes(*, size: tuple[int, int] = (20, 20)) -> bytes:
    image = Image.new("RGB", size, (20, 60, 100))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


# 느타리 class_id와 신뢰도를 고정하고 bbox만 바꾸기 쉬운 탐지 결과 도우미이다.
def oyster_detection(
    bbox: tuple[float, float, float, float],
) -> predictor.Detection:
    return predictor.Detection(
        class_id=0,
        species="느타리",
        bbox=bbox,
        confidence=0.95,
    )


# 테스트별로 업로드 한도와 건강 판정 임계값만 손쉽게 바꾼 설정을 만든다.
def settings(
    *,
    max_upload_bytes: int = 10 * 1024 * 1024,
    health_threshold: float = 0.70,
    max_inflight_requests: int = 1,
) -> HealthAPISettings:
    return HealthAPISettings(
        max_upload_bytes=max_upload_bytes,
        health_uncertain_threshold=health_threshold,
        max_inflight_requests=max_inflight_requests,
    )


# 비동기 analyze_upload을 동기 테스트에서 실행하고 서비스 executor를 항상 정리한다.
def run_analysis(
    service: MushroomHealthService,
    upload: FakeUpload,
) -> dict[str, Any]:
    try:
        return asyncio.run(
            service.analyze_upload(upload)  # type: ignore[arg-type]
        )
    finally:
        service.close()


# 운영 executor 도우미가 모델 작업을 이벤트 루프가 아닌 worker 스레드에서 순서대로 실행하는지 확인한다.
def test_production_executor_helper_runs_work_off_event_loop() -> None:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    main_thread = threading.get_ident()

    def worker(value: int) -> tuple[int, int]:
        time.sleep(0.005)
        return value, threading.get_ident()

    async def scenario() -> list[tuple[int, int]]:
        return [
            await PRODUCTION_RUN_SYNC_IN_EXECUTOR(
                executor,
                lambda value=value: worker(value),
            )
            for value in range(3)
        ]

    try:
        results = asyncio.run(scenario())
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert [value for value, _thread in results] == [0, 1, 2]
    assert all(thread != main_thread for _value, thread in results)


# 호출 코루틴이 취소되어도 worker가 끝난 뒤 CancelledError가 전달되는지 확인한다.
def test_executor_cancellation_waits_for_worker_completion() -> None:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    completed = threading.Event()

    def slow_worker() -> None:
        time.sleep(0.02)
        completed.set()

    async def scenario() -> None:
        task = asyncio.create_task(
            PRODUCTION_RUN_SYNC_IN_EXECUTOR(executor, slow_worker)
        )
        await asyncio.sleep(0.002)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(scenario())
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert completed.is_set()


# 같은 품종 bbox들을 합친 union 영역과 이미지 크기 기준 15% padding crop을 재사용하는지 확인한다.
def test_service_reuses_union_crop_and_image_relative_15_percent_padding() -> None:
    detector = FakeDetector(
        [
            oyster_detection((20, 20, 30, 30)),
            oyster_detection((40, 25, 60, 50)),
        ]
    )
    classifier = FakeClassifier((0.92, 0.08))
    registry = FakeRegistry(detector, classifier)
    service = MushroomHealthService(registry, settings())  # type: ignore[arg-type]
    upload = FakeUpload(jpeg_bytes())

    response = run_analysis(service, upload)

    assert response["results"][0]["bbox"] == [20, 20, 60, 50]
    assert response["results"][0]["crop_bbox"] == [5, 8, 75, 62]
    assert detector.image_sizes == [(100, 80)]
    assert classifier.crop_sizes == [(70, 54)]
    assert detector.calls == classifier.calls == registry.get_calls == 1
    assert upload.closed


# 이진 분류 확률의 최대값과 임계값에 따라 세 공개 건강 상태가 선택되는지 확인한다.
@pytest.mark.parametrize(
    ("probabilities", "expected_status"),
    [
        ((0.91, 0.09), "HEALTHY"),
        ((0.08, 0.92), "DISEASE_SUSPECTED"),
        ((0.55, 0.45), "UNCERTAIN"),
    ],
)
def test_service_maps_health_probabilities_to_public_states(
    probabilities: tuple[float, float],
    expected_status: str,
) -> None:
    detector = FakeDetector([oyster_detection((10, 10, 80, 70))])
    classifier = FakeClassifier(probabilities)
    registry = FakeRegistry(detector, classifier)
    service = MushroomHealthService(registry, settings())  # type: ignore[arg-type]

    response = run_analysis(service, FakeUpload(jpeg_bytes()))

    result = response["results"][0]
    assert result["health_status"] == expected_status
    assert result["healthy_probability"] == pytest.approx(probabilities[0])
    assert result["disease_suspected_probability"] == pytest.approx(
        probabilities[1]
    )
    assert detector.calls == classifier.calls == registry.get_calls == 1


# 여러 요청이 동시에 들어와도 inference_lock이 실제 모델 호출을 한 번씩 직렬화하는지 확인한다.
def test_inference_lock_serializes_concurrent_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    max_active = 0

    async def cooperative_run_sync(
        _executor: Any,
        function: Any,
    ) -> Any:
        nonlocal active, max_active
        is_prediction = (
            getattr(getattr(function, "func", None), "__name__", "")
            == "_predict_sync"
        )
        if not is_prediction:
            return function()
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.01)
            return function()
        finally:
            active -= 1

    monkeypatch.setattr(
        service_module,
        "run_sync_in_executor",
        cooperative_run_sync,
    )
    detector = FakeDetector([oyster_detection((10, 10, 80, 70))])
    classifier = FakeClassifier((0.9, 0.1))
    registry = FakeRegistry(detector, classifier)

    async def scenario() -> list[dict[str, Any]]:
        service = MushroomHealthService(
            registry,  # type: ignore[arg-type]
            settings(max_inflight_requests=4),
            inference_lock=asyncio.Lock(),
        )
        uploads = [FakeUpload(jpeg_bytes()) for _ in range(4)]
        try:
            return await asyncio.gather(
                *(
                    service.analyze_upload(upload)  # type: ignore[arg-type]
                    for upload in uploads
                )
            )
        finally:
            service.close()

    responses = asyncio.run(scenario())

    assert len(responses) == 4
    assert max_active == 1
    assert detector.calls == classifier.calls == registry.get_calls == 4


# 처리 한도를 넘은 요청은 본문을 읽지 않고 거부하며, 완료 후 permit을 반환하는지 확인한다.
def test_admission_limit_rejects_excess_before_read_and_releases_permit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_uploads: list[FakeUpload] = []
    original_read = service_module._read_upload_limited

    def tracking_read(
        upload: FakeUpload,
        max_upload_bytes: int,
    ) -> bytes:
        read_uploads.append(upload)
        return original_read(upload, max_upload_bytes)  # type: ignore[arg-type]

    monkeypatch.setattr(
        service_module,
        "_read_upload_limited",
        tracking_read,
    )
    detector = FakeDetector([oyster_detection((10, 10, 80, 70))])
    classifier = FakeClassifier((0.9, 0.1))
    registry = FakeRegistry(detector, classifier)
    first = FakeUpload(jpeg_bytes())
    second = FakeUpload(jpeg_bytes())
    third = FakeUpload(jpeg_bytes())

    async def scenario() -> tuple[
        dict[str, Any],
        HealthServiceError,
        dict[str, Any],
    ]:
        prediction_started = asyncio.Event()
        release_prediction = asyncio.Event()

        async def controlled_run_sync(
            _executor: Any,
            function: Any,
        ) -> Any:
            is_prediction = (
                getattr(getattr(function, "func", None), "__name__", "")
                == "_predict_sync"
            )
            if is_prediction:
                prediction_started.set()
                await release_prediction.wait()
            return function()

        monkeypatch.setattr(
            service_module,
            "run_sync_in_executor",
            controlled_run_sync,
        )
        service = MushroomHealthService(
            registry,  # type: ignore[arg-type]
            settings(max_inflight_requests=1),
            inference_lock=asyncio.Lock(),
        )
        first_task = asyncio.create_task(
            service.analyze_upload(first)  # type: ignore[arg-type]
        )
        try:
            await asyncio.wait_for(prediction_started.wait(), timeout=1)

            with pytest.raises(HealthServiceError) as captured:
                await service.analyze_upload(second)  # type: ignore[arg-type]

            assert captured.value.http_status == 429
            assert captured.value.status == "SERVICE_BUSY"
            assert captured.value.public_message == (
                "현재 분석 요청이 많습니다. 잠시 후 다시 시도해 주세요."
            )
            assert second not in read_uploads
            assert second.closed

            release_prediction.set()
            first_response = await first_task
            third_response = await service.analyze_upload(  # type: ignore[arg-type]
                third
            )
            return first_response, captured.value, third_response
        finally:
            release_prediction.set()
            if not first_task.done():
                await first_task
            service.close()

    first_response, _error, third_response = asyncio.run(scenario())

    assert first_response["status"] == "SUCCESS"
    assert third_response["status"] == "SUCCESS"
    assert read_uploads == [first, third]
    assert detector.calls == classifier.calls == registry.get_calls == 2


# 읽기·디코딩·추론 과정이 호출자가 제공한 원본 bytes 내용을 바꾸지 않는지 확인한다.
def test_upload_processing_does_not_mutate_source_bytes() -> None:
    payload = jpeg_bytes()
    before = hashlib.sha256(payload).hexdigest()
    detector = FakeDetector([oyster_detection((10, 10, 80, 70))])
    registry = FakeRegistry(detector, FakeClassifier((0.8, 0.2)))
    service = MushroomHealthService(registry, settings())  # type: ignore[arg-type]

    run_analysis(service, FakeUpload(payload))

    assert hashlib.sha256(payload).hexdigest() == before


# EXIF 회전은 적용하되 JPEG 형식 검증 정책과 RGB 변환은 그대로 유지하는지 확인한다.
def test_decode_applies_exif_orientation_without_changing_format_policy() -> None:
    payload = jpeg_bytes(size=(40, 20), orientation=6)

    decoded = decode_image_bytes(payload, "JPEG")

    assert decoded.mode == "RGB"
    assert decoded.size == (20, 40)


# JPEG로 선언한 PNG 바이트처럼 실제 형식이 다른 파일을 415로 거부하는지 확인한다.
def test_declared_format_must_match_image_bytes() -> None:
    with pytest.raises(HealthServiceError) as captured:
        decode_image_bytes(png_bytes(), "JPEG")

    assert captured.value.http_status == 415
    assert captured.value.status == "INVALID_IMAGE"


# 확장자와 MIME type 불일치 시 모델을 조회하지 않고 업로드 핸들을 닫는지 확인한다.
def test_extension_and_mime_type_must_match_and_upload_is_closed() -> None:
    upload = FakeUpload(
        jpeg_bytes(),
        filename="mushroom.png",
        content_type="image/jpeg",
    )
    registry = FakeRegistry(
        FakeDetector([]),
        FakeClassifier((0.5, 0.5)),
    )
    service = MushroomHealthService(registry, settings())  # type: ignore[arg-type]

    with pytest.raises(HealthServiceError) as captured:
        run_analysis(service, upload)

    assert captured.value.http_status == 415
    assert registry.get_calls == 0
    assert upload.closed


# 업로드 바이트 한도를 초과하면 이미지 디코딩과 모델 조회 전에 413으로 중단하는지 확인한다.
def test_upload_byte_limit_rejects_before_decode_and_model_call() -> None:
    payload = jpeg_bytes()
    upload = FakeUpload(payload)
    registry = FakeRegistry(
        FakeDetector([]),
        FakeClassifier((0.5, 0.5)),
    )
    service = MushroomHealthService(  # type: ignore[arg-type]
        registry,
        settings(max_upload_bytes=len(payload) - 1),
    )

    with pytest.raises(HealthServiceError) as captured:
        run_analysis(service, upload)

    assert captured.value.http_status == 413
    assert registry.get_calls == 0
    assert upload.closed


# 총 픽셀 수가 안전 한도를 넘으면 모델 호출 전 400 오류로 변환되는지 확인한다.
def test_pixel_limit_is_checked_before_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module.predictor, "MAX_IMAGE_PIXELS", 100)
    registry = FakeRegistry(
        FakeDetector([]),
        FakeClassifier((0.5, 0.5)),
    )
    service = MushroomHealthService(registry, settings())  # type: ignore[arg-type]

    with pytest.raises(HealthServiceError) as captured:
        run_analysis(
            service,
            FakeUpload(
                png_bytes(size=(20, 20)),
                filename="mushroom.png",
                content_type="image/png",
            ),
        )

    assert captured.value.http_status == 400
    assert registry.get_calls == 0


# Pillow 압축 폭탄 경고를 내부 정보 없는 안전한 INVALID_IMAGE 오류로 변환하는지 확인한다.
def test_pillow_decompression_bomb_is_converted_to_safe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = png_bytes(size=(8, 8))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)

    with pytest.raises(HealthServiceError) as captured:
        decode_image_bytes(payload, "PNG")

    assert captured.value.http_status == 400
    assert captured.value.status == "INVALID_IMAGE"
    assert "path" not in captured.value.public_message.lower()


# 손상된 이미지 바이트는 모델을 조회하지 않고 400으로 거부되는지 확인한다.
def test_corrupt_image_is_rejected_without_model_call() -> None:
    registry = FakeRegistry(
        FakeDetector([]),
        FakeClassifier((0.5, 0.5)),
    )
    service = MushroomHealthService(registry, settings())  # type: ignore[arg-type]

    with pytest.raises(HealthServiceError) as captured:
        run_analysis(service, FakeUpload(b"not a real jpeg"))

    assert captured.value.http_status == 400
    assert registry.get_calls == 0


# 레지스트리 내부 오류를 절대 경로가 없는 일반 추론 실패로 바꾸는지 확인한다.
def test_model_registry_error_is_converted_without_secret_or_path() -> None:
    service = MushroomHealthService(  # type: ignore[arg-type]
        FailingRegistry(),
        settings(),
    )

    with pytest.raises(HealthServiceError) as captured:
        run_analysis(service, FakeUpload(jpeg_bytes()))

    error = captured.value
    assert error.http_status == 500
    assert error.status == "INFERENCE_FAILED"
    assert error.public_message == "모델 추론을 완료하지 못했습니다."
    serialized = f"{error} {error.public_message}"
    assert "/home/" not in serialized
    assert "/mnt/" not in serialized
