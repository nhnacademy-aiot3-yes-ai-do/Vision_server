from __future__ import annotations

import asyncio
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pytest
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import HealthAPISettings
from app import main as main_module
from app.main import create_app
from app.services import mushroom_health_service as service_module
from scripts import predict_mushroom_health as predictor


BOUNDARY = "mushroom-health-test-boundary"


@pytest.fixture(autouse=True)
def run_fake_work_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_sync(_executor: Any, function: Any) -> Any:
        return function()

    monkeypatch.setattr(main_module, "run_sync_in_executor", run_sync)
    monkeypatch.setattr(service_module, "run_sync_in_executor", run_sync)


def jpeg_bytes(
    *,
    size: tuple[int, int] = (100, 80),
    color: tuple[int, int, int] = (40, 80, 60),
) -> bytes:
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, subsampling=0)
    return buffer.getvalue()


class FakeDetector:
    model_name = "fake-species-detector"

    def __init__(self) -> None:
        self.detections: list[predictor.Detection] = [
            predictor.Detection(
                class_id=0,
                species="느타리",
                bbox=(10, 10, 70, 60),
                confidence=0.91,
            )
        ]
        self.calls: list[dict[str, Any]] = []
        self.failure: Exception | None = None

    def predict_with_threshold(
        self,
        image: Image.Image,
        threshold: float,
    ) -> list[predictor.Detection]:
        self.calls.append(
            {
                "size": image.size,
                "threshold": threshold,
            }
        )
        if self.failure is not None:
            raise self.failure
        return list(self.detections)


class FakeClassifier:
    model_name = "fake-health-classifier"

    def __init__(self) -> None:
        self.probabilities: list[tuple[float, float]] = [(0.95, 0.05)]
        self.calls: list[tuple[int, int]] = []
        self.failure: Exception | None = None

    def predict(self, crop: Image.Image) -> tuple[float, float]:
        self.calls.append(crop.size)
        if self.failure is not None:
            raise self.failure
        if not self.probabilities:
            raise AssertionError("unexpected fake classifier call")
        return self.probabilities.pop(0)


class FakeRegistry:
    def __init__(self, settings: HealthAPISettings) -> None:
        self.settings = settings
        self.detector = FakeDetector()
        self.classifier = FakeClassifier()
        self.load_calls = 0
        self.get_models_calls = 0
        self.shutdown_calls = 0
        self.loaded = False
        self.load_failure: Exception | None = None

    def load(self) -> tuple[FakeDetector, FakeClassifier]:
        self.load_calls += 1
        if self.load_failure is not None:
            raise self.load_failure
        self.loaded = True
        return self.detector, self.classifier

    def get_models(self) -> tuple[FakeDetector, FakeClassifier]:
        self.get_models_calls += 1
        if not self.loaded:
            raise RuntimeError("fake registry is not loaded")
        return self.detector, self.classifier

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.loaded = False


@dataclass(frozen=True)
class ASGIResponse:
    status_code: int
    headers: dict[str, str]
    content: bytes

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8"))


class DirectASGIClient:
    """Minimal synchronous ASGI client without the optional httpx package."""

    def __init__(self, application: Any) -> None:
        self.application = application
        self.loop: asyncio.AbstractEventLoop | None = None
        self.lifespan: Any = None

    def __enter__(self) -> "DirectASGIClient":
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.lifespan = self.application.router.lifespan_context(
            self.application
        )
        self.loop.run_until_complete(self.lifespan.__aenter__())
        return self

    def __exit__(self, *exc_info: object) -> None:
        assert self.loop is not None
        try:
            self.loop.run_until_complete(
                self.lifespan.__aexit__(*exc_info)
            )
        finally:
            self.loop.close()
            asyncio.set_event_loop(None)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes,
        headers: dict[str, str],
    ) -> ASGIResponse:
        request_messages = [
            {
                "type": "http.request",
                "body": body,
                "more_body": False,
            }
        ]
        response_status: int | None = None
        response_headers: dict[str, str] = {}
        response_parts: list[bytes] = []
        async def receive() -> dict[str, Any]:
            if request_messages:
                return request_messages.pop(0)
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            nonlocal response_status, response_headers
            if message["type"] == "http.response.start":
                response_status = int(message["status"])
                response_headers = {
                    key.decode("latin-1").lower(): value.decode("latin-1")
                    for key, value in message.get("headers", [])
                }
            elif message["type"] == "http.response.body":
                response_parts.append(message.get("body", b""))

        encoded_headers = [
            (key.lower().encode("latin-1"), value.encode("latin-1"))
            for key, value in headers.items()
        ]
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.5"},
            "http_version": "1.1",
            "method": method.upper(),
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "root_path": "",
            "headers": encoded_headers,
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }
        await self.application(scope, receive, send)
        assert response_status is not None
        return ASGIResponse(
            status_code=response_status,
            headers=response_headers,
            content=b"".join(response_parts),
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> ASGIResponse:
        assert self.loop is not None
        return self.loop.run_until_complete(
            self._request(
                method,
                path,
                body=body,
                headers=headers or {},
            )
        )


def multipart_body(
    payload: bytes,
    *,
    field_name: str = "image",
    filename: str = "mushroom.jpg",
    content_type: str = "image/jpeg",
) -> tuple[bytes, str]:
    boundary = BOUNDARY.encode("ascii")
    body = b"".join(
        (
            b"--",
            boundary,
            b"\r\n",
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("ascii"),
            payload,
            b"\r\n--",
            boundary,
            b"--\r\n",
        )
    )
    return body, f"multipart/form-data; boundary={BOUNDARY}"


def post_image(
    client: DirectASGIClient,
    payload: bytes,
    *,
    field_name: str = "image",
    filename: str = "mushroom.jpg",
    content_type: str = "image/jpeg",
) -> ASGIResponse:
    body, multipart_type = multipart_body(
        payload,
        field_name=field_name,
        filename=filename,
        content_type=content_type,
    )
    return client.request(
        "POST",
        "/api/v1/mushroom/health-check",
        body=body,
        headers={
            "content-type": multipart_type,
            "content-length": str(len(body)),
        },
    )


@pytest.fixture
def settings() -> HealthAPISettings:
    return HealthAPISettings(
        detection_confidence=0.25,
        health_uncertain_threshold=0.70,
        padding_ratio=0.15,
        max_upload_bytes=4096,
        verify_model_sha256=False,
        device="cpu",
    )


@pytest.fixture
def registry(settings: HealthAPISettings) -> FakeRegistry:
    return FakeRegistry(settings)


@pytest.fixture
def api_client(
    settings: HealthAPISettings,
    registry: FakeRegistry,
) -> Iterator[DirectASGIClient]:
    application = create_app(settings=settings, registry=registry)  # type: ignore[arg-type]
    with DirectASGIClient(application) as client:
        yield client


@pytest.mark.parametrize(
    ("probabilities", "expected_status"),
    [
        ((0.95, 0.05), "HEALTHY"),
        ((0.08, 0.92), "DISEASE_SUSPECTED"),
        ((0.60, 0.40), "UNCERTAIN"),
    ],
)
def test_valid_image_health_status_and_camel_case_contract(
    api_client: DirectASGIClient,
    registry: FakeRegistry,
    probabilities: tuple[float, float],
    expected_status: str,
) -> None:
    registry.classifier.probabilities = [probabilities]

    response = post_image(api_client, jpeg_bytes())

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "analysisType",
        "status",
        "detectorModel",
        "healthModel",
        "thresholds",
        "results",
        "warnings",
    }
    assert payload["analysisType"] == "MUSHROOM_HEALTH_CHECK_V1"
    assert payload["status"] == "SUCCESS"
    assert payload["thresholds"] == {
        "detection": pytest.approx(0.25),
        "healthUncertain": pytest.approx(0.70),
    }
    assert len(payload["results"]) == 1
    result = payload["results"][0]
    assert result["species"] == "느타리"
    assert result["speciesClassId"] == 0
    assert result["healthStatus"] == expected_status
    assert result["healthyProbability"] == pytest.approx(probabilities[0])
    assert result["diseaseSuspectedProbability"] == pytest.approx(
        probabilities[1]
    )
    assert result["healthConfidence"] == pytest.approx(max(probabilities))
    assert "analysis_type" not in payload
    assert "health_status" not in result
    assert registry.detector.calls[-1]["threshold"] == pytest.approx(0.25)


def test_no_detection_returns_successful_empty_result_without_classifier(
    api_client: DirectASGIClient,
    registry: FakeRegistry,
) -> None:
    registry.detector.detections = []

    response = post_image(api_client, jpeg_bytes())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "NO_MUSHROOM_DETECTED"
    assert payload["results"] == []
    assert registry.classifier.calls == []


def test_multiple_species_are_grouped_into_public_results(
    api_client: DirectASGIClient,
    registry: FakeRegistry,
) -> None:
    registry.detector.detections = [
        predictor.Detection(
            class_id=0,
            species="느타리",
            bbox=(5, 5, 35, 35),
            confidence=0.90,
        ),
        predictor.Detection(
            class_id=0,
            species="느타리",
            bbox=(30, 10, 55, 50),
            confidence=0.80,
        ),
        predictor.Detection(
            class_id=4,
            species="표고",
            bbox=(60, 20, 95, 70),
            confidence=0.93,
        ),
    ]
    registry.classifier.probabilities = [(0.90, 0.10), (0.10, 0.90)]

    response = post_image(api_client, jpeg_bytes())

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    by_species = {item["species"]: item for item in results}
    assert by_species["느타리"]["detectedCount"] == 2
    assert by_species["느타리"]["healthStatus"] == "HEALTHY"
    assert by_species["표고"]["detectedCount"] == 1
    assert by_species["표고"]["speciesClassId"] == 4
    assert by_species["표고"]["healthStatus"] == "DISEASE_SUSPECTED"
    assert len(registry.classifier.calls) == 2


def test_empty_and_corrupt_uploads_are_safe_400(
    api_client: DirectASGIClient,
    registry: FakeRegistry,
) -> None:
    for content in (b"", b"this is not a jpeg"):
        response = post_image(api_client, content)
        assert response.status_code == 400
        payload = response.json()
        assert payload["status"] == "INVALID_IMAGE"
        assert payload["results"] == []
    assert registry.get_models_calls == 0


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("mushroom.jpg", "image/png"),
        ("mushroom.txt", "text/plain"),
    ],
)
def test_mime_extension_mismatch_or_unsupported_type_is_415(
    api_client: DirectASGIClient,
    registry: FakeRegistry,
    filename: str,
    content_type: str,
) -> None:
    response = post_image(
        api_client,
        jpeg_bytes(),
        filename=filename,
        content_type=content_type,
    )

    assert response.status_code == 415
    assert response.json()["status"] == "INVALID_IMAGE"
    assert registry.get_models_calls == 0


def test_upload_larger_than_configured_limit_is_413(
    api_client: DirectASGIClient,
    registry: FakeRegistry,
) -> None:
    response = post_image(api_client, b"x" * 4097)

    assert response.status_code == 413
    assert response.json()["status"] == "INVALID_IMAGE"
    assert registry.get_models_calls == 0


def test_missing_image_field_is_framework_422(
    api_client: DirectASGIClient,
    registry: FakeRegistry,
) -> None:
    response = post_image(
        api_client,
        jpeg_bytes(),
        field_name="wrong_field",
    )

    assert response.status_code == 422
    payload = response.json()
    assert isinstance(payload["detail"], list)
    assert registry.get_models_calls == 0


def test_internal_failure_is_safe_500_without_absolute_path(
    api_client: DirectASGIClient,
    registry: FakeRegistry,
) -> None:
    registry.classifier.failure = RuntimeError(
        "private model failure at /home/kim75/secret/best.pt"
    )

    response = post_image(api_client, jpeg_bytes())

    assert response.status_code == 500
    payload = response.json()
    serialized = json.dumps(payload, ensure_ascii=False)
    assert payload["status"] == "INFERENCE_FAILED"
    assert payload["results"] == []
    assert "/home/" not in serialized
    assert "/mnt/" not in serialized
    assert "best.pt" not in serialized
    assert "private model failure" not in serialized


def test_success_response_contains_no_local_absolute_path(
    api_client: DirectASGIClient,
) -> None:
    response = post_image(api_client, jpeg_bytes())

    serialized = response.content.decode("utf-8")
    assert response.status_code == 200
    assert "/home/" not in serialized
    assert "/mnt/" not in serialized
    assert "C:\\\\" not in serialized


def test_registry_loads_once_for_multiple_requests_and_shuts_down_once(
    settings: HealthAPISettings,
) -> None:
    fake = FakeRegistry(settings)
    application = create_app(settings=settings, registry=fake)  # type: ignore[arg-type]

    with DirectASGIClient(application) as client:
        first = post_image(client, jpeg_bytes())
        fake.classifier.probabilities = [(0.10, 0.90)]
        second = post_image(client, jpeg_bytes(color=(80, 30, 20)))
        assert first.status_code == second.status_code == 200
        assert fake.load_calls == 1
        assert fake.get_models_calls == 2
        assert fake.shutdown_calls == 0

    assert fake.load_calls == 1
    assert fake.shutdown_calls == 1


def test_openapi_documents_multipart_contract_and_public_response(
    api_client: DirectASGIClient,
    registry: FakeRegistry,
) -> None:
    response = api_client.request("GET", "/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    operation = schema["paths"][
        "/api/v1/mushroom/health-check"
    ]["post"]
    assert "multipart/form-data" in operation["requestBody"]["content"]
    assert {
        "200",
        "400",
        "413",
        "415",
        "422",
        "500",
    } <= set(operation["responses"])
    success_schema = operation["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    assert success_schema["$ref"].endswith("/HealthCheckResponse")
    properties = schema["components"]["schemas"][
        "HealthCheckResponse"
    ]["properties"]
    assert "analysisType" in properties
    assert "detectorModel" in properties
    assert "analysis_type" not in properties
    assert registry.get_models_calls == 0
