"""FastAPI application factory for Mushroom Health Check API v1."""

from __future__ import annotations

import asyncio
import concurrent.futures
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.health import router as health_router
from app.core.config import HealthAPISettings
from app.core.model_registry import ModelRegistry
from app.services.mushroom_health_service import (
    MushroomHealthService,
    run_sync_in_executor,
)


def create_app(
    *,
    settings: HealthAPISettings | None = None,
    registry: ModelRegistry | None = None,
) -> FastAPI:
    resolved_settings = (
        settings
        or (getattr(registry, "settings", None) if registry is not None else None)
        or HealthAPISettings.from_env()
    )
    resolved_registry = registry or ModelRegistry(resolved_settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="mushroom-health-api",
        )
        loaded = False
        try:
            await run_sync_in_executor(executor, resolved_registry.load)
            loaded = True
            application.state.model_registry = resolved_registry
            application.state.health_service = MushroomHealthService(
                resolved_registry,
                resolved_settings,
                inference_lock=asyncio.Lock(),
                executor=executor,
            )
            yield
        finally:
            application.state.health_service = None
            try:
                if loaded:
                    await run_sync_in_executor(
                        executor,
                        resolved_registry.shutdown,
                    )
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

    application = FastAPI(
        title="Mushroom Health Check API",
        version="1.0.0",
        description=(
            "YOLO11n 품종 탐지와 이진 건강 분류를 결합한 내부 검증 API입니다. "
            "결과는 참고용이며 확정 진단이 아닙니다."
        ),
        lifespan=lifespan,
    )
    application.include_router(health_router)
    return application


app = create_app()
