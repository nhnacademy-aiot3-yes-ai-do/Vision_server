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


PRODUCTION_RUN_SYNC_IN_EXECUTOR = service_module.run_sync_in_executor


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


class FakeRegistry:
    def __init__(self, detector: Any, classifier: Any) -> None:
        self.detector = detector
        self.classifier = classifier
        self.get_calls = 0

    def get_models(self) -> tuple[Any, Any]:
        self.get_calls += 1
        return self.detector, self.classifier


class FailingRegistry:
    def get_models(self) -> tuple[Any, Any]:
        raise ModelRegistryError(
            "secret model failure at /home/kim75/private/best.pt"
        )


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


def png_bytes(*, size: tuple[int, int] = (20, 20)) -> bytes:
    image = Image.new("RGB", size, (20, 60, 100))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def oyster_detection(
    bbox: tuple[float, float, float, float],
) -> predictor.Detection:
    return predictor.Detection(
        class_id=0,
        species="느타리",
        bbox=bbox,
        confidence=0.95,
    )


def settings(
    *,
    max_upload_bytes: int = 10 * 1024 * 1024,
    health_threshold: float = 0.70,
) -> HealthAPISettings:
    return HealthAPISettings(
        max_upload_bytes=max_upload_bytes,
        health_uncertain_threshold=health_threshold,
    )


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
            settings(),
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


def test_upload_processing_does_not_mutate_source_bytes() -> None:
    payload = jpeg_bytes()
    before = hashlib.sha256(payload).hexdigest()
    detector = FakeDetector([oyster_detection((10, 10, 80, 70))])
    registry = FakeRegistry(detector, FakeClassifier((0.8, 0.2)))
    service = MushroomHealthService(registry, settings())  # type: ignore[arg-type]

    run_analysis(service, FakeUpload(payload))

    assert hashlib.sha256(payload).hexdigest() == before


def test_decode_applies_exif_orientation_without_changing_format_policy() -> None:
    payload = jpeg_bytes(size=(40, 20), orientation=6)

    decoded = decode_image_bytes(payload, "JPEG")

    assert decoded.mode == "RGB"
    assert decoded.size == (20, 40)


def test_declared_format_must_match_image_bytes() -> None:
    with pytest.raises(HealthServiceError) as captured:
        decode_image_bytes(png_bytes(), "JPEG")

    assert captured.value.http_status == 415
    assert captured.value.status == "INVALID_IMAGE"


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
