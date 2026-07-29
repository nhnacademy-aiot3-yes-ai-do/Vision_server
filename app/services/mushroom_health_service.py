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


READ_CHUNK_BYTES = 1024 * 1024
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


async def run_sync_in_executor(
    executor: concurrent.futures.Executor,
    function: Callable[[], T],
) -> T:
    """Run sync work without releasing its caller early on cancellation."""
    future = executor.submit(function)
    try:
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
                continue
        try:
            future.result()
        except Exception:
            pass
        raise


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
    raise HealthServiceError(
        http_status=415,
        status="INVALID_IMAGE",
        public_message="파일 확장자와 MIME type이 일치하지 않습니다.",
    )


def _close_upload_sync(upload: UploadFile) -> None:
    upload.file.close()


def _read_upload_limited(
    upload: UploadFile,
    max_upload_bytes: int,
) -> bytes:
    payload = bytearray()
    try:
        while True:
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
        _close_upload_sync(upload)
    if not payload:
        raise HealthServiceError(
            http_status=400,
            status="INVALID_IMAGE",
            public_message="빈 이미지 파일입니다.",
        )
    return bytes(payload)


def decode_image_bytes(
    payload: bytes,
    expected_format: str,
) -> Image.Image:
    try:
        with warnings.catch_warnings():
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
            with Image.open(io.BytesIO(payload)) as reopened:
                image = ImageOps.exif_transpose(reopened).convert("RGB")
                image.load()
    except HealthServiceError:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
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


class MushroomHealthService:
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
        self._owns_executor = executor is None
        self.executor = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="mushroom-health",
        )

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
        predictor.assert_deidentified_response(response)
        return response

    async def analyze_upload(self, upload: UploadFile) -> dict[str, Any]:
        try:
            expected_format, _mime_type = _upload_rule(upload)
        except Exception:
            try:
                await run_sync_in_executor(
                    self.executor,
                    partial(_close_upload_sync, upload),
                )
            except Exception:
                pass
            raise
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
            async with self.inference_lock:
                return await run_sync_in_executor(
                    self.executor,
                    partial(self._predict_sync, image),
                )
        except HealthServiceError:
            raise
        except ModelRegistryError as exc:
            raise HealthServiceError(
                http_status=500,
                status="INFERENCE_FAILED",
                public_message="모델 추론을 완료하지 못했습니다.",
            ) from exc
        except Exception as exc:
            raise HealthServiceError(
                http_status=500,
                status="INFERENCE_FAILED",
                public_message="모델 추론을 완료하지 못했습니다.",
            ) from exc

    def close(self) -> None:
        if self._owns_executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
