from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from fastapi import FastAPI

from app.api.status import router as status_router
from app.core.config import HealthAPISettings
from app.core.model_registry import RegistryState
from app.main import create_app


@dataclass(frozen=True)
class ASGIResponse:
    status_code: int
    content: bytes

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8"))


async def _asgi_get(application: FastAPI, path: str) -> ASGIResponse:
    request_sent = False
    status_code: int | None = None
    response_parts: list[bytes] = []

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {
                "type": "http.request",
                "body": b"",
                "more_body": False,
            }
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = int(message["status"])
        elif message["type"] == "http.response.body":
            response_parts.append(message.get("body", b""))

    await application(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        },
        receive,
        send,
    )
    assert status_code is not None
    return ASGIResponse(status_code, b"".join(response_parts))


def get(application: FastAPI, path: str) -> ASGIResponse:
    return asyncio.run(_asgi_get(application, path))


class ProbeRegistry:
    def __init__(self, state: RegistryState) -> None:
        self._state = state
        self.state_reads = 0
        self.load_calls = 0
        self.get_models_calls = 0

    @property
    def state(self) -> RegistryState:
        self.state_reads += 1
        return self._state

    def load(self) -> None:
        self.load_calls += 1
        raise AssertionError("probe must not load models")

    def get_models(self) -> None:
        self.get_models_calls += 1
        raise AssertionError("probe must not fetch models")


def probe_app(registry: object | None) -> FastAPI:
    application = FastAPI()
    if registry is not None:
        application.state.model_registry = registry
    application.include_router(status_router)
    return application


def test_liveness_is_up_without_accessing_model_registry() -> None:
    class ExplodingRegistry:
        @property
        def state(self) -> RegistryState:
            raise AssertionError("liveness must not read registry state")

    response = get(probe_app(ExplodingRegistry()), "/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "UP"}


def test_readiness_is_ready_only_for_ready_registry_without_model_access() -> None:
    registry = ProbeRegistry(RegistryState.READY)

    response = get(probe_app(registry), "/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "READY"}
    assert registry.state_reads == 1
    assert registry.load_calls == 0
    assert registry.get_models_calls == 0


def test_readiness_is_not_ready_for_unloaded_failed_or_missing_registry() -> None:
    for state in (RegistryState.UNLOADED, RegistryState.FAILED):
        registry = ProbeRegistry(state)
        response = get(probe_app(registry), "/health/ready")
        assert response.status_code == 503
        assert response.json() == {"status": "NOT_READY"}
        assert registry.load_calls == 0
        assert registry.get_models_calls == 0

    missing = get(probe_app(None), "/health/ready")
    assert missing.status_code == 503
    assert missing.json() == {"status": "NOT_READY"}


def test_readiness_hides_registry_exception_and_local_path() -> None:
    class BrokenRegistry:
        @property
        def state(self) -> RegistryState:
            raise RuntimeError(
                "failed at /home/kim75/private/models/detector.pt"
            )

    response = get(probe_app(BrokenRegistry()), "/health/ready")
    serialized = response.content.decode("utf-8")

    assert response.status_code == 503
    assert response.json() == {"status": "NOT_READY"}
    assert "/home/" not in serialized
    assert "/mnt/" not in serialized
    assert "detector.pt" not in serialized
    assert "failed at" not in serialized


def test_main_openapi_documents_both_health_probes() -> None:
    registry = ProbeRegistry(RegistryState.UNLOADED)
    registry.settings = HealthAPISettings(verify_model_sha256=False)  # type: ignore[attr-defined]
    application = create_app(registry=registry)  # type: ignore[arg-type]

    schema = application.openapi()

    live = schema["paths"]["/health/live"]["get"]
    ready = schema["paths"]["/health/ready"]["get"]
    assert {"200"} <= set(live["responses"])
    assert {"200", "503"} <= set(ready["responses"])
    assert live["responses"]["200"]["content"]["application/json"][
        "schema"
    ]["$ref"].endswith("/LiveStatusResponse")
    assert ready["responses"]["200"]["content"]["application/json"][
        "schema"
    ]["$ref"].endswith("/ReadyStatusResponse")
    assert ready["responses"]["503"]["content"]["application/json"][
        "schema"
    ]["$ref"].endswith("/ReadyStatusResponse")
    assert "/api/v1/mushroom/health-check" in schema["paths"]
    assert registry.load_calls == 0
    assert registry.get_models_calls == 0
