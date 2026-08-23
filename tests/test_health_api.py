# 실제 모델 없이 ASGI 요청을 만들어 건강 판별 API의 공개 계약과 생명주기를 검증한다.
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


# 어느 작업 디렉터리에서 실행해도 프로젝트 패키지를 import할 수 있게 루트를 우선 경로에 둔다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import HealthAPISettings
from app import main as main_module
from app.main import create_app
from app.services import mushroom_health_service as service_module
from scripts import predict_mushroom_health as predictor


# multipart 본문을 직접 조립할 때 재현 가능한 고정 경계 문자열을 사용한다.
BOUNDARY = "mushroom-health-test-boundary"


# 단위 테스트에서는 스레드 생명주기를 배제하고 executor 작업을 호출 스레드에서 즉시 실행한다.
@pytest.fixture(autouse=True)
def run_fake_work_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_sync(_executor: Any, function: Any) -> Any:
        return function()

    monkeypatch.setattr(main_module, "run_sync_in_executor", run_sync)
    monkeypatch.setattr(service_module, "run_sync_in_executor", run_sync)


# 외부 이미지 파일 없이 API 요청에 넣을 유효한 JPEG 바이트를 메모리에서 생성한다.
def jpeg_bytes(
    *,
    size: tuple[int, int] = (100, 80),
    color: tuple[int, int, int] = (40, 80, 60),
) -> bytes:
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, subsampling=0)
    return buffer.getvalue()


# 호출 인자와 실패 여부를 관찰할 수 있는 품종 탐지기 대역이다.
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


# crop 크기와 호출 횟수를 기록하고 준비한 확률을 순서대로 반환하는 분류기 대역이다.
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


# 앱 lifespan의 load/get/shutdown 호출 횟수와 모델 재사용 여부를 검증하는 레지스트리 대역이다.
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


# 직접 수집한 ASGI 상태 코드·헤더·본문을 테스트에서 편하게 읽기 위한 값 객체이다.
@dataclass(frozen=True)
class ASGIResponse:
    status_code: int
    headers: dict[str, str]
    content: bytes

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8"))


# httpx 없이 lifespan과 ASGI receive/send 프로토콜을 직접 구동하는 동기 테스트 클라이언트이다.
class DirectASGIClient:
    """Minimal synchronous ASGI client without the optional httpx package."""

    def __init__(self, application: Any) -> None:
        self.application = application
        self.loop: asyncio.AbstractEventLoop | None = None
        self.lifespan: Any = None

    def __enter__(self) -> "DirectASGIClient":
        # 별도 이벤트 루프에서 FastAPI lifespan 시작 훅까지 완료한 뒤 요청을 허용한다.
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
            # 컨텍스트 종료 시 lifespan 종료 훅을 실행해 레지스트리 정리까지 검증한다.
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
        # 본문 전체를 담은 단일 http.request 메시지를 애플리케이션에 공급한다.
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
            # ASGI가 나누어 보내는 응답 시작 정보와 본문 조각을 하나의 응답 객체로 수집한다.
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
        # 실제 HTTP 서버가 만드는 것과 같은 최소 ASGI HTTP scope를 구성한다.
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
        # 테스트 본문은 동기 함수이므로 내부 비동기 요청을 클라이언트 이벤트 루프에서 끝까지 실행한다.
        return self.loop.run_until_complete(
            self._request(
                method,
                path,
                body=body,
                headers=headers or {},
            )
        )


# image 파일 파트 하나를 포함하는 multipart/form-data 본문과 Content-Type을 직접 만든다.
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


# 모든 API 테스트가 같은 엔드포인트와 multipart 헤더로 POST하도록 묶은 도우미이다.
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
        "/api/v1/internal/mushrooms/health-check",
        body=body,
        headers={
            "content-type": multipart_type,
            "content-length": str(len(body)),
        },
    )


# 경계값을 명확히 확인할 수 있는 테스트 전용 설정을 제공한다.
@pytest.fixture
def settings() -> HealthAPISettings:
    return HealthAPISettings(
        detection_confidence=0.25,
        min_detection_confidence=0.50,
        health_uncertain_threshold=0.70,
        padding_ratio=0.15,
        max_upload_bytes=4096,
        verify_model_sha256=False,
        device="cpu",
    )


# 각 테스트에 호출 기록이 비어 있는 새 가짜 레지스트리를 제공한다.
@pytest.fixture
def registry(settings: HealthAPISettings) -> FakeRegistry:
    return FakeRegistry(settings)


# 실제 create_app과 lifespan을 거쳐 요청할 수 있는 클라이언트를 준비하고 종료까지 수행한다.
@pytest.fixture
def api_client(
    settings: HealthAPISettings,
    registry: FakeRegistry,
) -> Iterator[DirectASGIClient]:
    application = create_app(settings=settings, registry=registry)  # type: ignore[arg-type]
    with DirectASGIClient(application) as client:
        yield client


# 세 가지 건강 확률 구간이 공개 상태로 바뀌고 모든 JSON 필드가 camelCase인지 확인한다.
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
        "minDetectionConfidence": pytest.approx(0.50),
        "healthUncertain": pytest.approx(0.70),
    }
    assert len(payload["results"]) == 1
    result = payload["results"][0]
    assert result["species"] == "느타리"
    assert result["speciesCode"] == "OYSTER"
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


# 최소 탐지 신뢰도 미만이면 분류기를 호출하지 않고 건강 관련 값을 null로 내보내는지 확인한다.
def test_low_detection_confidence_returns_null_health_values(
    api_client: DirectASGIClient,
    registry: FakeRegistry,
) -> None:
    registry.detector.detections[0] = predictor.Detection(
        class_id=0,
        species="느타리",
        bbox=(10, 10, 70, 60),
        confidence=0.49,
    )

    response = post_image(api_client, jpeg_bytes())

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["healthStatus"] == "UNCERTAIN"
    assert result["healthConfidence"] is None
    assert result["healthyProbability"] is None
    assert result["diseaseSuspectedProbability"] is None
    assert registry.classifier.calls == []
    assert predictor.LOW_DETECTION_CONFIDENCE_WARNING in response.json()[
        "warnings"
    ]
    serialized = response.content.decode("utf-8")
    assert '"healthConfidence":null' in serialized
    assert '"healthyProbability":null' in serialized
    assert '"diseaseSuspectedProbability":null' in serialized


# 최소 탐지 신뢰도와 정확히 같은 경계값은 분류 대상으로 포함되는지 확인한다.
def test_detection_confidence_equal_to_minimum_runs_health_classifier(
    api_client: DirectASGIClient,
    registry: FakeRegistry,
) -> None:
    registry.detector.detections[0] = predictor.Detection(
        class_id=0,
        species="느타리",
        bbox=(10, 10, 70, 60),
        confidence=0.50,
    )

    response = post_image(api_client, jpeg_bytes())

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["healthStatus"] == "HEALTHY"
    assert result["healthConfidence"] == pytest.approx(0.95)
    assert len(registry.classifier.calls) == 1
    assert predictor.LOW_DETECTION_CONFIDENCE_WARNING not in response.json()[
        "warnings"
    ]


# 여러 품종 중 신뢰도가 낮은 품종만 보류하고 충분한 품종은 정상 분류하는지 확인한다.
def test_low_detection_confidence_is_isolated_per_species(
    api_client: DirectASGIClient,
    registry: FakeRegistry,
) -> None:
    registry.detector.detections = [
        predictor.Detection(
            class_id=0,
            species="느타리",
            bbox=(5, 5, 35, 35),
            confidence=0.49,
        ),
        predictor.Detection(
            class_id=4,
            species="표고",
            bbox=(60, 20, 95, 70),
            confidence=0.93,
        ),
    ]
    registry.classifier.probabilities = [(0.10, 0.90)]

    response = post_image(api_client, jpeg_bytes())

    assert response.status_code == 200
    by_species = {
        result["species"]: result for result in response.json()["results"]
    }
    assert by_species["느타리"]["healthStatus"] == "UNCERTAIN"
    assert by_species["느타리"]["healthConfidence"] is None
    assert by_species["표고"]["healthStatus"] == "DISEASE_SUSPECTED"
    assert by_species["표고"]["healthConfidence"] == pytest.approx(0.90)
    assert len(registry.classifier.calls) == 1


# 버섯을 찾지 못한 경우도 오류가 아닌 빈 성공 결과이며 분류기는 호출하지 않는지 확인한다.
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


# 같은 품종의 여러 bbox는 하나로 묶고 서로 다른 품종은 별도 결과로 분류하는지 확인한다.
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
    assert by_species["표고"]["speciesCode"] == "SHIITAKE"
    assert by_species["표고"]["speciesClassId"] == 4
    assert by_species["표고"]["healthStatus"] == "DISEASE_SUSPECTED"
    assert len(registry.classifier.calls) == 2


# 빈 파일과 손상 파일을 모델 호출 전에 안전한 400 응답으로 거부하는지 확인한다.
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


# 지원하지 않는 형식 또는 확장자·MIME 불일치를 415로 거부하는지 확인한다.
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


# 설정된 업로드 바이트 상한을 넘으면 디코딩·모델 호출 전에 413을 반환하는지 확인한다.
def test_upload_larger_than_configured_limit_is_413(
    api_client: DirectASGIClient,
    registry: FakeRegistry,
) -> None:
    response = post_image(api_client, b"x" * 4097)

    assert response.status_code == 413
    assert response.json()["status"] == "INVALID_IMAGE"
    assert registry.get_models_calls == 0


# 동시 분석 한도 초과도 공통 공개 응답을 유지하며 429로 구분되는지 확인한다.
def test_inference_capacity_exceeded_is_safe_429(
    api_client: DirectASGIClient,
    registry: FakeRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def reject_busy(
        _service: Any,
        upload: Any,
    ) -> dict[str, Any]:
        await upload.close()
        raise service_module.HealthServiceError(
            http_status=429,
            status="SERVICE_BUSY",
            public_message=(
                "현재 분석 요청이 많습니다. 잠시 후 다시 시도해 주세요."
            ),
        )

    monkeypatch.setattr(
        service_module.MushroomHealthService,
        "analyze_upload",
        reject_busy,
    )

    response = post_image(api_client, jpeg_bytes())

    assert response.status_code == 429
    payload = response.json()
    assert payload["status"] == "SERVICE_BUSY"
    assert payload["results"] == []
    assert payload["warnings"] == [
        "AI 분석 참고 결과이며 확정 진단이 아닙니다.",
        "SERVICE_BUSY: 현재 분석 요청이 많습니다. 잠시 후 다시 시도해 주세요.",
    ]
    assert registry.get_models_calls == 0


# multipart 필드 이름이 계약과 다르면 FastAPI 검증이 422를 반환하는지 확인한다.
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


# 내부 모델 예외의 메시지와 절대 경로가 500 공개 응답으로 새지 않는지 확인한다.
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


# 정상 응답에도 서버의 Linux·Windows 로컬 절대 경로가 포함되지 않는지 확인한다.
def test_success_response_contains_no_local_absolute_path(
    api_client: DirectASGIClient,
) -> None:
    response = post_image(api_client, jpeg_bytes())

    serialized = response.content.decode("utf-8")
    assert response.status_code == 200
    assert "/home/" not in serialized
    assert "/mnt/" not in serialized
    assert "C:\\\\" not in serialized


# 실행 장치 설정은 추론에만 쓰이고 응답과 OpenAPI 문서에는 공개되지 않는지 확인한다.
def test_device_setting_is_not_exposed_in_response_or_openapi() -> None:
    settings = HealthAPISettings(
        detection_confidence=0.25,
        min_detection_confidence=0.50,
        health_uncertain_threshold=0.70,
        padding_ratio=0.15,
        max_upload_bytes=4096,
        verify_model_sha256=False,
        device="cuda:0",
    )
    registry = FakeRegistry(settings)
    application = create_app(
        settings=settings,
        registry=registry,  # type: ignore[arg-type]
    )

    with DirectASGIClient(application) as client:
        response = post_image(client, jpeg_bytes())
        openapi = client.request("GET", "/openapi.json")

    assert response.status_code == 200
    assert openapi.status_code == 200
    response_text = response.content.decode("utf-8")
    openapi_text = openapi.content.decode("utf-8")
    for serialized in (response_text, openapi_text):
        assert "cuda:0" not in serialized
        assert '"device"' not in serialized
        assert "HEALTH_DEVICE" not in serialized


# lifespan 전체에서 모델은 한 번 로드·종료되고 여러 요청이 같은 레지스트리를 쓰는지 확인한다.
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


# OpenAPI가 multipart 입력, 상태 코드, camelCase 응답 및 nullable 확률을 정확히 문서화하는지 확인한다.
def test_openapi_documents_multipart_contract_and_public_response(
    api_client: DirectASGIClient,
    registry: FakeRegistry,
) -> None:
    response = api_client.request("GET", "/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    operation = schema["paths"][
        "/api/v1/internal/mushrooms/health-check"
    ]["post"]
    assert "multipart/form-data" in operation["requestBody"]["content"]
    assert {
        "200",
        "400",
        "413",
        "415",
        "429",
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
    threshold_properties = schema["components"]["schemas"][
        "HealthThresholds"
    ]["properties"]
    assert "minDetectionConfidence" in threshold_properties
    assert "min_detection_confidence" not in threshold_properties
    result_properties = schema["components"]["schemas"][
        "MushroomHealthResult"
    ]["properties"]
    for name in (
        "healthConfidence",
        "healthyProbability",
        "diseaseSuspectedProbability",
    ):
        types = {
            option.get("type")
            for option in result_properties[name].get("anyOf", [])
        }
        assert "number" in types
        assert "null" in types
    assert registry.get_models_calls == 0
