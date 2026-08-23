# 업로드를 안전하게 읽고 이미지 형식을 검증한 뒤 고정 모델 쌍의 추론을 직렬화한다.
"""Upload validation and reuse of the verified mushroom inference pipeline."""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
import warnings
from functools import partial
from pathlib import PurePath
from typing import Any, Callable, Mapping, TypeVar

from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import HealthAPISettings
from app.core.model_registry import ModelRegistry, ModelRegistryError
from scripts import predict_mushroom_health as predictor


# 대용량 요청을 한 번에 읽지 않고 1 MiB 단위로 제한을 확인한다.
READ_CHUNK_BYTES = 1024 * 1024
# 실제 이미지 형식별로 허용하는 파일 확장자와 MIME type의 조합을 명시한다.
FORMAT_RULES: Mapping[str, tuple[frozenset[str], frozenset[str]]] = {
    "JPEG": (
        frozenset({".jpg", ".jpeg"}),
        frozenset({"image/jpeg"}),
    ),
    "PNG": (
        frozenset({".png"}),
        frozenset({"image/png"}),
    ),
    "WEBP": (
        frozenset({".webp"}),
        frozenset({"image/webp"}),
    ),
}
ALLOWED_EXTENSIONS = frozenset(
    extension
    for extensions, _mime_types in FORMAT_RULES.values()
    for extension in extensions
)
ALLOWED_MIME_TYPES = frozenset(
    mime_type
    for _extensions, mime_types in FORMAT_RULES.values()
    for mime_type in mime_types
)
T = TypeVar("T")


# HTTP 상태, 공개 업무 상태, 사용자 메시지만 전달하는 서비스 계층의 안전한 예외이다.
class HealthServiceError(RuntimeError):
    def __init__(
        self,
        *,
        http_status: int,
        status: str,
        public_message: str,
    ) -> None:
        super().__init__(status)
        self.http_status = http_status
        self.status = status
        self.public_message = public_message


# 동기 함수를 전용 executor에 보내되 취소 시에도 실행 중인 작업이 끝날 때까지 기다린다.
async def run_sync_in_executor(
    executor: concurrent.futures.Executor,
    function: Callable[[], T],
) -> T:
    """Run sync work without releasing its caller early on cancellation."""
    future = executor.submit(function)
    try:
        # 짧게 양보하며 완료를 확인해 FastAPI 이벤트 루프가 다른 요청을 처리할 수 있게 한다.
        while not future.done():
            await asyncio.sleep(0.001)
        return future.result()
    except asyncio.CancelledError:
        # The model thread cannot be force-cancelled safely. Wait until it
        # finishes so an enclosing inference lock is not released too early.
        while not future.done():
            try:
                await asyncio.shield(asyncio.sleep(0.005))
            except asyncio.CancelledError:
                # 반복 취소가 들어와도 모델 스레드가 끝나기 전에 호출자가 빠져나가지 않게 한다.
                continue
        try:
            future.result()
        except Exception:
            pass
        raise


# 업로드 메타데이터의 확장자와 MIME type이 같은 허용 형식을 가리키는지 확인한다.
def _upload_rule(upload: UploadFile) -> tuple[str, str]:
    extension = PurePath(upload.filename or "").suffix.lower()
    mime_type = (upload.content_type or "").split(";", 1)[0].strip().lower()
    if extension not in ALLOWED_EXTENSIONS or mime_type not in ALLOWED_MIME_TYPES:
        raise HealthServiceError(
            http_status=415,
            status="INVALID_IMAGE",
            public_message="지원하지 않는 이미지 형식입니다.",
        )
    for image_format, (extensions, mime_types) in FORMAT_RULES.items():
        if extension in extensions and mime_type in mime_types:
            return image_format, mime_type
    # 각각은 허용 목록에 있어도 서로 다른 형식을 주장하면 위조 또는 실수로 보고 거부한다.
    raise HealthServiceError(
        http_status=415,
        status="INVALID_IMAGE",
        public_message="파일 확장자와 MIME type이 일치하지 않습니다.",
    )


# Starlette UploadFile의 내부 파일 핸들을 동기 작업 스레드에서 닫는다.
def _close_upload_sync(upload: UploadFile) -> None:
    upload.file.close()


# 설정된 최대 크기보다 한 바이트 더 읽어 초과 여부를 판별하고 항상 업로드를 닫는다.
def _read_upload_limited(
    upload: UploadFile,
    max_upload_bytes: int,
) -> bytes:
    payload = bytearray()
    try:
        while True:
            # 제한에 도달한 뒤에도 1바이트를 탐색해야 정확히 초과한 요청을 구분할 수 있다.
            remaining_with_probe = max_upload_bytes + 1 - len(payload)
            read_size = min(READ_CHUNK_BYTES, max(1, remaining_with_probe))
            chunk = upload.file.read(read_size)
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > max_upload_bytes:
                raise HealthServiceError(
                    http_status=413,
                    status="INVALID_IMAGE",
                    public_message="업로드 크기 제한을 초과했습니다.",
                )
    finally:
        # 정상 읽기, 크기 초과, 파일 읽기 오류 어느 경우에도 임시 파일 핸들을 남기지 않는다.
        _close_upload_sync(upload)
    if not payload:
        raise HealthServiceError(
            http_status=400,
            status="INVALID_IMAGE",
            public_message="빈 이미지 파일입니다.",
        )
    return bytes(payload)


# 파일 바이트의 실제 형식·프레임·픽셀 수를 확인하고 EXIF 방향이 반영된 RGB 이미지를 만든다.
def decode_image_bytes(
    payload: bytes,
    expected_format: str,
) -> Image.Image:
    try:
        with warnings.catch_warnings():
            # Pillow의 압축 폭탄 경고도 오류로 승격해 과도한 픽셀 이미지를 통과시키지 않는다.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as opened:
                actual_format = str(opened.format or "").upper()
                if actual_format != expected_format:
                    raise HealthServiceError(
                        http_status=415,
                        status="INVALID_IMAGE",
                        public_message=(
                            "선언된 형식과 실제 이미지 형식이 일치하지 않습니다."
                        ),
                    )
                if getattr(opened, "n_frames", 1) != 1:
                    raise HealthServiceError(
                        http_status=415,
                        status="INVALID_IMAGE",
                        public_message="애니메이션 이미지는 지원하지 않습니다.",
                    )
                width, height = opened.size
                if (
                    width <= 0
                    or height <= 0
                    or width * height > predictor.MAX_IMAGE_PIXELS
                ):
                    raise HealthServiceError(
                        http_status=400,
                        status="INVALID_IMAGE",
                        public_message="이미지 픽셀 수 제한을 초과했습니다.",
                    )
                opened.verify()
            # verify 이후 객체는 다시 사용할 수 없으므로 새로 열어 방향 보정·RGB 변환·전체 로드를 한다.
            with Image.open(io.BytesIO(payload)) as reopened:
                image = ImageOps.exif_transpose(reopened).convert("RGB")
                image.load()
    except HealthServiceError:
        # 이미 공개 메시지로 분류된 검증 실패는 상태를 보존한다.
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        # Pillow 세부 오류는 파일 구조나 경로를 숨긴 일반적인 400 오류로 바꾼다.
        raise HealthServiceError(
            http_status=400,
            status="INVALID_IMAGE",
            public_message="손상되었거나 유효하지 않은 이미지입니다.",
        ) from exc
    if image.width <= 0 or image.height <= 0:
        raise HealthServiceError(
            http_status=400,
            status="INVALID_IMAGE",
            public_message="빈 이미지입니다.",
        )
    return image


# 검증된 이미지와 모델 레지스트리를 연결하고 동시에 하나의 추론만 실행하도록 보호한다.
class MushroomHealthService:
    """Validate uploads and serialize access to the verified model pair."""

    def __init__(
        self,
        registry: ModelRegistry,
        settings: HealthAPISettings,
        *,
        inference_lock: asyncio.Lock | None = None,
        executor: concurrent.futures.Executor | None = None,
    ) -> None:
        self.registry = registry
        self.settings = settings
        self.inference_lock = inference_lock or asyncio.Lock()
        # app lifespan이 executor를 주입하면 소유하지 않고, 단독 사용 시에만 자체 생성·종료한다.
        self._owns_executor = executor is None
        self.executor = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="mushroom-health",
        )

    # 준비된 모델 쌍과 설정 임계값으로 검증된 predictor 파이프라인을 동기 실행한다.
    def _predict_sync(self, image: Image.Image) -> dict[str, Any]:
        detector, classifier = self.registry.get_models()
        response = predictor.predict_health(
            image,
            detector=detector,
            classifier=classifier,
            detection_threshold=self.settings.detection_confidence,
            min_detection_confidence=(
                self.settings.min_detection_confidence
            ),
            health_threshold=self.settings.health_uncertain_threshold,
            padding_ratio=self.settings.padding_ratio,
        )
        # 반환 직전에 로컬 경로 같은 비공개 정보가 섞이지 않았는지 확인한다.
        predictor.assert_deidentified_response(response)
        return response

    # 메타데이터 확인 → 제한 읽기 → 이미지 디코딩 → 잠금 추론 순서로 한 요청을 처리한다.
    async def analyze_upload(self, upload: UploadFile) -> dict[str, Any]:
        try:
            expected_format, _mime_type = _upload_rule(upload)
        except Exception:
            # 메타데이터 단계에서 거부되면 본문을 읽지 않았더라도 업로드 핸들은 반드시 닫는다.
            try:
                await run_sync_in_executor(
                    self.executor,
                    partial(_close_upload_sync, upload),
                )
            except Exception:
                pass
            raise
        # 파일 I/O와 Pillow 디코딩은 이벤트 루프를 막지 않도록 executor에서 수행한다.
        payload = await run_sync_in_executor(
            self.executor,
            partial(
                _read_upload_limited,
                upload,
                self.settings.max_upload_bytes,
            ),
        )
        image = await run_sync_in_executor(
            self.executor,
            partial(decode_image_bytes, payload, expected_format),
        )
        try:
            # 모델 wrapper가 동시 호출에 안전하다고 가정하지 않고 프로세스 내 추론을 직렬화한다.
            async with self.inference_lock:
                return await run_sync_in_executor(
                    self.executor,
                    partial(self._predict_sync, image),
                )
        except HealthServiceError:
            raise
        except ModelRegistryError as exc:
            # 레지스트리의 상세 원인이나 모델 경로를 외부 응답에 포함하지 않는다.
            raise HealthServiceError(
                http_status=500,
                status="INFERENCE_FAILED",
                public_message="모델 추론을 완료하지 못했습니다.",
            ) from exc
        except Exception as exc:
            # predictor 내부의 예상하지 못한 실패도 동일한 공개 추론 오류로 축약한다.
            raise HealthServiceError(
                http_status=500,
                status="INFERENCE_FAILED",
                public_message="모델 추론을 완료하지 못했습니다.",
            ) from exc

    # 서비스가 자체 생성한 executor만 종료하여 lifespan이 공유하는 executor를 중복 종료하지 않는다.
    def close(self) -> None:
        if self._owns_executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
