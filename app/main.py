# 설정·모델·서비스의 생명주기를 연결하고 모든 HTTP 라우터를 등록하는 진입점이다.
"""FastAPI application factory for Mushroom Vision Service."""

from __future__ import annotations

import asyncio
import concurrent.futures
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.status import router as status_router
from app.core.config import HealthAPISettings
from app.core.model_registry import ModelRegistry
from app.services.mushroom_health_service import (
    MushroomHealthService,
    run_sync_in_executor,
)


# 테스트에서는 설정과 레지스트리를 주입하고, 운영에서는 환경 변수와 실제 모델 레지스트리를 만든다.
def create_app(
    *,
    settings: HealthAPISettings | None = None,
    registry: ModelRegistry | None = None,
) -> FastAPI:
    """Create one API process with a singleton model registry lifespan."""

    # 우선순위는 명시적 설정, 주입된 레지스트리의 설정, 환경 변수 설정 순서이다.
    resolved_settings = (
        settings
        or (getattr(registry, "settings", None) if registry is not None else None)
        or HealthAPISettings.from_env()
    )
    resolved_registry = registry or ModelRegistry(resolved_settings)

    # 서버 시작부터 종료까지 모델과 작업 스레드가 정확히 한 번 생성·해제되도록 묶는다.
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        # 파일 읽기·Pillow·모델 추론 같은 동기 작업을 이벤트 루프 밖에서 실행한다.
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="mushroom-health-api",
        )
        loaded = False
        try:
            # 요청 수신 전에 모델 쌍을 로드하고 성공한 경우에만 app.state에 서비스를 공개한다.
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
            # 종료가 시작되면 새 의존성 주입이 실패하도록 서비스 참조부터 제거한다.
            application.state.health_service = None
            try:
                if loaded:
                    # 로드가 완료된 레지스트리만 종료하여 모델 파일의 최종 무결성까지 검사한다.
                    await run_sync_in_executor(
                        executor,
                        resolved_registry.shutdown,
                    )
            finally:
                # 레지스트리 종료 성공 여부와 무관하게 작업 스레드는 반드시 회수한다.
                executor.shutdown(wait=True, cancel_futures=True)

    # OpenAPI 메타데이터와 lifespan을 가진 애플리케이션에 비즈니스·상태 라우터를 연결한다.
    application = FastAPI(
        title="Mushroom Vision Service API",
        version="1.0.0",
        description=(
            "YOLO11n 품종 탐지와 이진 건강 분류를 결합한 내부 검증 API입니다. "
            "결과는 참고용이며 확정 진단이 아닙니다."
        ),
        lifespan=lifespan,
    )
    application.include_router(health_router)
    application.include_router(status_router)
    return application


# ASGI 서버가 `app.main:app`으로 가져갈 기본 애플리케이션 인스턴스이다.
app = create_app()
