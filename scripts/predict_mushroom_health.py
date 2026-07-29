#!/usr/bin/env python3
"""Single-image mushroom species detection and health classification.

The two fixed ``best.pt`` files are loaded read-only.  A detector groups
objects by species, each species receives one union crop with image-relative
padding, and the health classifier returns HEALTHY, DISEASE_SUSPECTED, or
UNCERTAIN.  No path from the local machine is included in the JSON response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DETECTOR_MODEL_PATH = (
    ARTIFACTS_ROOT / "models" / "yolo11n_camera_holdout_v1" / "best.pt"
)
HEALTH_MODEL_PATH = (
    ARTIFACTS_ROOT
    / "models"
    / "yolo11n_health_date_camera_holdout_v1"
    / "best.pt"
)
DETECTOR_MODEL_NAME = "mushroom-yolo11n-camera-holdout-v1"
HEALTH_MODEL_NAME = "mushroom-health-yolo11n-date-camera-holdout-v1"
EXPECTED_MODEL_SHA256 = {
    DETECTOR_MODEL_PATH: (
        "8d17eb493f2eeccccd832c56da2f346dfc730e5605c460016386c6bd0be10d32"
    ),
    HEALTH_MODEL_PATH: (
        "720efb30093c2fbaf8866243a60870c7816f3e141c81e0fac015a358edce1f92"
    ),
}
DETECTOR_IMAGE_SIZE = 640
HEALTH_IMAGE_SIZE = 320
DEFAULT_DETECTION_CONFIDENCE = 0.25
DEFAULT_MIN_DETECTION_CONFIDENCE = 0.50
DEFAULT_HEALTH_THRESHOLD = 0.70
DEFAULT_PADDING_RATIO = 0.15
LOW_DETECTION_CONFIDENCE_WARNING = (
    "LOW_DETECTION_CONFIDENCE: 품종 탐지 신뢰도가 낮아 건강 상태를 "
    "판단하지 않았습니다."
)
MAX_IMAGE_PIXELS = 100_000_000
SPECIES_BY_CLASS_ID = {
    0: "느타리",
    1: "양송이",
    2: "큰느타리",
    3: "팽이",
    4: "표고",
}
DETECTOR_MODEL_NAMES = dict(SPECIES_BY_CLASS_ID)
HEALTH_MODEL_NAMES = {
    0: "0_healthy",
    1: "1_disease_suspected",
}
DISCLAIMER = "AI 분석 참고 결과이며 확정 진단이 아닙니다."
FORBIDDEN_PATH_PATTERNS = (
    re.compile(r"/mnt/[A-Za-z](?:/|$)", re.IGNORECASE),
    re.compile(r"/home/[^/\s]+", re.IGNORECASE),
    re.compile(r"[A-Za-z]:[\\/]", re.IGNORECASE),
    re.compile(r"\\\\Users\\\\", re.IGNORECASE),
)


@dataclass(frozen=True)
class Detection:
    class_id: int
    species: str
    bbox: tuple[float, float, float, float]
    confidence: float


class DetectorProtocol(Protocol):
    model_name: str

    def predict(self, image: Image.Image) -> Sequence[Detection]: ...


class ClassifierProtocol(Protocol):
    model_name: str

    def predict(self, crop: Image.Image) -> tuple[float, float]: ...


def probability(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("실수여야 합니다") from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("0~1 범위여야 합니다")
    return parsed


def padding_ratio(value: str) -> float:
    parsed = probability(value)
    if parsed > 0.5:
        raise argparse.ArgumentTypeError("padding ratio는 0~0.5여야 합니다")
    return parsed


def project_path(value: Path) -> Path:
    if value.is_absolute():
        return value.resolve()
    return (PROJECT_ROOT / value).resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "고정 YOLO11n 품종 탐지 모델과 건강 분류 모델로 단일 이미지를 "
            "분석합니다. FastAPI, 학습, 모델 변환은 수행하지 않습니다."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--detection-confidence",
        type=probability,
        default=DEFAULT_DETECTION_CONFIDENCE,
    )
    parser.add_argument(
        "--health-threshold",
        type=probability,
        default=DEFAULT_HEALTH_THRESHOLD,
    )
    parser.add_argument(
        "--padding-ratio",
        type=padding_ratio,
        default=DEFAULT_PADDING_RATIO,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--annotated-output",
        type=Path,
        help="선택 사항. 프로젝트 artifacts/health_predictions 아래 JPEG",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="선택 사항. 프로젝트 artifacts/health_predictions 아래 JSON",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="명시한 단일 이미지 출력이 이미 있을 때만 교체",
    )
    args = parser.parse_args(argv)
    args.image = project_path(args.image)
    for name in ("annotated_output", "json_output"):
        value = getattr(args, name)
        if value is not None:
            resolved = project_path(value)
            allowed = (ARTIFACTS_ROOT / "health_predictions").resolve()
            if not resolved.is_relative_to(allowed):
                parser.error(
                    f"--{name.replace('_', '-')}은 "
                    "artifacts/health_predictions 아래만 허용합니다"
                )
            setattr(args, name, resolved)
    outputs = [
        value
        for value in (args.annotated_output, args.json_output)
        if value is not None
    ]
    if any(path == args.image for path in outputs):
        parser.error("입력 이미지와 출력 경로는 같을 수 없습니다")
    if len(set(outputs)) != len(outputs):
        parser.error("annotated와 JSON 출력 경로는 서로 달라야 합니다")
    return args


def _finite(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 값이 숫자가 아닙니다") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} 값이 유한하지 않습니다")
    return parsed


def _validated_box(
    box: Sequence[float],
    image_size: tuple[int, int] | None = None,
) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError("bbox는 x1,y1,x2,y2 네 값이어야 합니다")
    x1, y1, x2, y2 = (
        _finite(value, "bbox") for value in box
    )
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox 폭과 높이는 양수여야 합니다")
    if image_size is not None:
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("이미지 크기는 양수여야 합니다")
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise ValueError("bbox가 이미지 범위를 벗어났습니다")
    return x1, y1, x2, y2


def group_detections_by_species(
    detections: Sequence[Detection],
) -> dict[str, list[Detection]]:
    grouped: dict[str, list[Detection]] = {}
    class_by_species: dict[str, int] = {}
    for detection in detections:
        if detection.class_id not in SPECIES_BY_CLASS_ID:
            raise ValueError(f"지원하지 않는 detector class: {detection.class_id}")
        expected = SPECIES_BY_CLASS_ID[detection.class_id]
        if detection.species != expected:
            raise ValueError("detector class_id와 species 매핑 불일치")
        _validated_box(detection.bbox)
        confidence = _finite(detection.confidence, "detection confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("detection confidence 범위 오류")
        if (
            detection.species in class_by_species
            and class_by_species[detection.species] != detection.class_id
        ):
            raise ValueError("동일 species에 서로 다른 class_id가 있습니다")
        class_by_species[detection.species] = detection.class_id
        grouped.setdefault(detection.species, []).append(detection)
    return {
        species: sorted(
            items,
            key=lambda item: (
                item.bbox[0],
                item.bbox[1],
                item.bbox[2],
                item.bbox[3],
                -item.confidence,
            ),
        )
        for species, items in sorted(
            grouped.items(),
            key=lambda item: next(
                value
                for value, name in SPECIES_BY_CLASS_ID.items()
                if name == item[0]
            ),
        )
    }


def union_bbox(
    boxes: Sequence[Sequence[float]],
) -> tuple[float, float, float, float]:
    if not boxes:
        raise ValueError("union bbox에 하나 이상의 bbox가 필요합니다")
    valid = [_validated_box(box) for box in boxes]
    return (
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    )


def outward_integer_box(
    box: Sequence[float],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = _validated_box(box)
    width, height = image_size
    result = (
        max(0, math.floor(x1)),
        max(0, math.floor(y1)),
        min(width, math.ceil(x2)),
        min(height, math.ceil(y2)),
    )
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("clip 후 bbox가 비었습니다")
    return result


def padded_union_crop_box(
    image_size: tuple[int, int],
    boxes: Sequence[Sequence[float]],
    *,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
) -> tuple[int, int, int, int]:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("이미지 크기는 양수여야 합니다")
    if not 0.0 <= padding_ratio <= 0.5:
        raise ValueError("padding ratio는 0~0.5여야 합니다")
    x1, y1, x2, y2 = union_bbox(boxes)
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    crop = (
        max(0, math.floor(x1 - pad_x)),
        max(0, math.floor(y1 - pad_y)),
        min(width, math.ceil(x2 + pad_x)),
        min(height, math.ceil(y2 + pad_y)),
    )
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        raise ValueError("padding 후 crop이 비었습니다")
    return crop


def health_status(
    healthy_probability: float,
    disease_probability: float,
    threshold: float,
) -> tuple[str, float]:
    healthy = _finite(healthy_probability, "HEALTHY probability")
    disease = _finite(disease_probability, "DISEASE probability")
    if not 0.0 <= healthy <= 1.0 or not 0.0 <= disease <= 1.0:
        raise ValueError("건강 확률 범위 오류")
    if abs((healthy + disease) - 1.0) > 1e-4:
        raise ValueError("건강 클래스 확률 합이 1이 아닙니다")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("health threshold 범위 오류")
    confidence = max(healthy, disease)
    if confidence < threshold:
        return "UNCERTAIN", confidence
    return (
        "HEALTHY" if healthy >= disease else "DISEASE_SUSPECTED",
        confidence,
    )


def _invoke_detector(
    detector: Any,
    image: Image.Image,
    detection_threshold: float,
) -> list[Detection]:
    if hasattr(detector, "predict_with_threshold"):
        raw = detector.predict_with_threshold(image, detection_threshold)
    else:
        raw = detector.predict(image)
    detections = list(raw)
    return [
        item
        for item in detections
        if _finite(item.confidence, "detection confidence")
        >= detection_threshold
    ]


def _invoke_classifier(
    classifier: Any,
    crop: Image.Image,
) -> tuple[float, float]:
    values = classifier.predict(crop)
    if not isinstance(values, Sequence) or len(values) != 2:
        raise ValueError("건강 분류기는 두 확률을 반환해야 합니다")
    healthy = _finite(values[0], "HEALTHY probability")
    disease = _finite(values[1], "DISEASE probability")
    health_status(healthy, disease, 0.0)
    return healthy, disease


def base_response(
    detector: Any,
    classifier: Any,
    detection_threshold: float,
    health_threshold: float,
) -> dict[str, Any]:
    return {
        "analysis_type": "MUSHROOM_HEALTH_CHECK_V1",
        "status": "SUCCESS",
        "detector_model": getattr(
            detector, "model_name", DETECTOR_MODEL_NAME
        ),
        "health_model": getattr(
            classifier, "model_name", HEALTH_MODEL_NAME
        ),
        "thresholds": {
            "detection": detection_threshold,
            "health_uncertain": health_threshold,
            "health_confidence": health_threshold,
        },
        "results": [],
        "warnings": [DISCLAIMER],
    }


def predict_health(
    image: Image.Image,
    *,
    detector: Any,
    classifier: Any,
    detection_threshold: float = DEFAULT_DETECTION_CONFIDENCE,
    min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE,
    health_threshold: float = DEFAULT_HEALTH_THRESHOLD,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
) -> dict[str, Any]:
    if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
        raise ValueError("유효한 PIL 이미지가 필요합니다")
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("이미지가 안전 픽셀 상한을 초과했습니다")
    if not 0.0 <= detection_threshold <= 1.0:
        raise ValueError("detection threshold 범위 오류")
    if not 0.0 <= min_detection_confidence <= 1.0:
        raise ValueError("minimum detection confidence 범위 오류")
    if not 0.0 <= health_threshold <= 1.0:
        raise ValueError("health threshold 범위 오류")
    if not 0.0 <= padding_ratio <= 0.5:
        raise ValueError("padding ratio는 0~0.5여야 합니다")
    response = base_response(
        detector,
        classifier,
        detection_threshold,
        health_threshold,
    )
    # Preserve the caller's image object and mode.
    inference_image = image.copy().convert("RGB")
    detections = _invoke_detector(
        detector,
        inference_image,
        detection_threshold,
    )
    if not detections:
        response["status"] = "NO_MUSHROOM_DETECTED"
        response["warnings"].append(
            "NO_MUSHROOM_DETECTED: 버섯 객체를 탐지하지 못했습니다."
        )
        return response
    grouped = group_detections_by_species(detections)
    if len(grouped) > 1:
        response["warnings"].append(
            "MULTIPLE_SPECIES_DETECTED: 서로 다른 품종을 각각 분석했습니다."
        )
    for species, items in grouped.items():
        boxes = [item.bbox for item in items]
        union = outward_integer_box(
            union_bbox(boxes),
            inference_image.size,
        )
        crop_box = padded_union_crop_box(
            inference_image.size,
            boxes,
            padding_ratio=padding_ratio,
        )
        detection_confidences = [float(item.confidence) for item in items]
        minimum_detection = min(detection_confidences)
        if minimum_detection < min_detection_confidence:
            healthy = None
            disease = None
            status = "UNCERTAIN"
            confidence = None
            if LOW_DETECTION_CONFIDENCE_WARNING not in response["warnings"]:
                response["warnings"].append(
                    LOW_DETECTION_CONFIDENCE_WARNING
                )
        else:
            crop = inference_image.crop(crop_box)
            if crop.width <= 0 or crop.height <= 0:
                raise ValueError("건강 분류 crop이 비었습니다")
            healthy, disease = _invoke_classifier(classifier, crop)
            status, confidence = health_status(
                healthy,
                disease,
                health_threshold,
            )
        if status == "UNCERTAIN" and confidence is not None:
            response["warnings"].append(
                f"UNCERTAIN: {species} 건강 confidence가 "
                f"{health_threshold:.2f} 미만입니다."
            )
        response["results"].append(
            {
                "species": species,
                "class_id": items[0].class_id,
                "detected_count": len(items),
                "detection_confidence": max(detection_confidences),
                "detection_confidence_min": minimum_detection,
                "health_status": status,
                "health_confidence": confidence,
                "healthy_probability": healthy,
                "disease_suspected_probability": disease,
                "bbox": list(union),
                "crop_bbox": list(crop_box),
            }
        )
    return response


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_annotated(
    image: Image.Image,
    response: Mapping[str, Any],
) -> Image.Image:
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = _font(max(14, min(28, annotated.width // 40)))
    colors = (
        (25, 210, 120),
        (60, 160, 255),
        (255, 180, 40),
        (230, 90, 210),
        (240, 90, 70),
    )
    for item in response.get("results", []):
        class_id = int(item["class_id"])
        bbox = tuple(int(value) for value in item["bbox"])
        color = colors[class_id]
        draw.rectangle(bbox, outline=color, width=max(2, annotated.width // 500))
        health_confidence = item.get("health_confidence")
        confidence_text = (
            f"{float(health_confidence):.3f}"
            if health_confidence is not None
            else "n/a"
        )
        text = (
            f"{item['species']} | {item['health_status']} "
            f"{confidence_text}"
        )
        text_box = draw.textbbox((bbox[0], bbox[1]), text, font=font)
        text_height = text_box[3] - text_box[1] + 8
        top = max(0, bbox[1] - text_height)
        draw.rectangle(
            (bbox[0], top, min(annotated.width, text_box[2] + 8), bbox[1]),
            fill=color,
        )
        draw.text((bbox[0] + 4, top + 2), text, fill=(0, 0, 0), font=font)
    return annotated


def load_valid_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise ValueError("입력 이미지 파일이 없습니다")
    try:
        with Image.open(path) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("손상되었거나 지원하지 않는 이미지입니다") from exc
    if image.width <= 0 or image.height <= 0:
        raise ValueError("빈 이미지입니다")
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("이미지가 안전 픽셀 상한을 초과했습니다")
    return image


def file_fingerprint(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return stat.st_size, stat.st_mtime_ns, digest.hexdigest()


@contextmanager
def isolated_ultralytics_runtime(
    runtime_parent: Path | None = None,
) -> Iterator[Path]:
    parent = str(runtime_parent) if runtime_parent is not None else None
    with tempfile.TemporaryDirectory(
        prefix="mushroom-health-runtime-",
        dir=parent,
    ) as temporary:
        runtime = Path(temporary)
        config = runtime / "ultralytics"
        matplotlib = runtime / "matplotlib"
        torch_home = runtime / "torch"
        xdg_cache = runtime / "cache"
        cuda_cache = runtime / "cuda"
        config.mkdir()
        matplotlib.mkdir()
        torch_home.mkdir()
        xdg_cache.mkdir()
        cuda_cache.mkdir()
        old_cwd = Path.cwd()
        old_values = {
            key: os.environ.get(key)
            for key in (
                "YOLO_CONFIG_DIR",
                "MPLCONFIGDIR",
                "TORCH_HOME",
                "XDG_CACHE_HOME",
                "CUDA_CACHE_PATH",
            )
        }
        try:
            os.environ["YOLO_CONFIG_DIR"] = str(config)
            os.environ["MPLCONFIGDIR"] = str(matplotlib)
            os.environ["TORCH_HOME"] = str(torch_home)
            os.environ["XDG_CACHE_HOME"] = str(xdg_cache)
            os.environ["CUDA_CACHE_PATH"] = str(cuda_cache)
            os.chdir(runtime)
            yield runtime
        finally:
            os.chdir(old_cwd)
            for key, value in old_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class UltralyticsDetector:
    model_name = DETECTOR_MODEL_NAME

    def __init__(self, model: Any, device: str = "auto") -> None:
        self.model = model
        self.device = device
        names = {int(key): str(value) for key, value in model.names.items()}
        if names != DETECTOR_MODEL_NAMES:
            raise RuntimeError("품종 detector class mapping 불일치")

    def predict_with_threshold(
        self,
        image: Image.Image,
        threshold: float,
    ) -> list[Detection]:
        device = None if self.device == "auto" else self.device
        results = self.model.predict(
            source=image,
            conf=threshold,
            imgsz=DETECTOR_IMAGE_SIZE,
            device=device,
            save=False,
            verbose=False,
            stream=False,
        )
        if len(results) != 1:
            raise RuntimeError("품종 detector 결과 수가 1이 아닙니다")
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        classes = boxes.cls.detach().cpu().tolist()
        if not (len(xyxy) == len(confidences) == len(classes)):
            raise RuntimeError("품종 detector tensor 길이 불일치")
        detections: list[Detection] = []
        for box, confidence, class_value in zip(
            xyxy, confidences, classes
        ):
            class_id = int(class_value)
            if class_id not in SPECIES_BY_CLASS_ID:
                raise RuntimeError("품종 detector class 범위 오류")
            # Clip harmless numerical spillover from model coordinates.
            x1, y1, x2, y2 = (_finite(value, "bbox") for value in box)
            clipped = (
                max(0.0, min(float(image.width), x1)),
                max(0.0, min(float(image.height), y1)),
                max(0.0, min(float(image.width), x2)),
                max(0.0, min(float(image.height), y2)),
            )
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            detections.append(
                Detection(
                    class_id=class_id,
                    species=SPECIES_BY_CLASS_ID[class_id],
                    bbox=clipped,
                    confidence=float(confidence),
                )
            )
        return detections

    def predict(self, image: Image.Image) -> list[Detection]:
        return self.predict_with_threshold(
            image, DEFAULT_DETECTION_CONFIDENCE
        )


class UltralyticsHealthClassifier:
    model_name = HEALTH_MODEL_NAME

    def __init__(self, model: Any, device: str = "auto") -> None:
        self.model = model
        self.device = device
        names = {int(key): str(value) for key, value in model.names.items()}
        if names != HEALTH_MODEL_NAMES:
            raise RuntimeError("건강 classifier class mapping 불일치")

    def predict(self, crop: Image.Image) -> tuple[float, float]:
        device = None if self.device == "auto" else self.device
        results = self.model.predict(
            source=crop,
            imgsz=HEALTH_IMAGE_SIZE,
            device=device,
            save=False,
            verbose=False,
            stream=False,
        )
        if len(results) != 1 or results[0].probs is None:
            raise RuntimeError("건강 classifier probability가 없습니다")
        values = results[0].probs.data.detach().cpu().tolist()
        if len(values) != 2:
            raise RuntimeError("건강 classifier class 수가 2가 아닙니다")
        top1 = 0 if values[0] >= values[1] else 1
        if int(results[0].probs.top1) != top1:
            raise RuntimeError("건강 classifier top1 mapping 불일치")
        return float(values[0]), float(values[1])


def load_fixed_models(
    *,
    device: str = "auto",
    verify_sha256: bool = True,
) -> tuple[UltralyticsDetector, UltralyticsHealthClassifier]:
    if not DETECTOR_MODEL_PATH.is_file() or not HEALTH_MODEL_PATH.is_file():
        raise FileNotFoundError("고정 detector 또는 health best.pt가 없습니다")
    if verify_sha256:
        for path, expected_hash in EXPECTED_MODEL_SHA256.items():
            actual_hash = file_fingerprint(path)[2]
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"승인된 고정 모델 SHA-256 불일치: {path.parent.name}"
                )
    from ultralytics import YOLO

    detector = YOLO(str(DETECTOR_MODEL_PATH.resolve()), task="detect")
    classifier = YOLO(str(HEALTH_MODEL_PATH.resolve()), task="classify")
    return (
        UltralyticsDetector(detector, device=device),
        UltralyticsHealthClassifier(classifier, device=device),
    )


def assert_deidentified_response(response: Mapping[str, Any]) -> None:
    serialized = json.dumps(
        response,
        ensure_ascii=False,
        allow_nan=False,
    )
    if any(pattern.search(serialized) for pattern in FORBIDDEN_PATH_PATTERNS):
        raise ValueError("JSON에 로컬 절대경로가 노출되었습니다")


def invalid_response(
    status: str,
    *,
    detection_threshold: float,
    health_threshold: float,
    detail: str,
) -> dict[str, Any]:
    return {
        "analysis_type": "MUSHROOM_HEALTH_CHECK_V1",
        "status": status,
        "detector_model": DETECTOR_MODEL_NAME,
        "health_model": HEALTH_MODEL_NAME,
        "thresholds": {
            "detection": detection_threshold,
            "health_uncertain": health_threshold,
            "health_confidence": health_threshold,
        },
        "results": [],
        "warnings": [DISCLAIMER, f"{status}: {detail}"],
    }


def write_optional_outputs(
    *,
    response: Mapping[str, Any],
    annotated_image: Image.Image | None,
    annotated_output: Path | None,
    json_output: Path | None,
    overwrite: bool,
) -> None:
    targets: list[tuple[Path, str]] = []
    if annotated_output is not None and annotated_image is not None:
        targets.append((annotated_output, "annotated"))
    if json_output is not None:
        targets.append((json_output, "json"))
    if not targets:
        return
    allowed_root = (ARTIFACTS_ROOT / "health_predictions").resolve()
    resolved_targets = [target.resolve() for target, _kind in targets]
    if len(set(resolved_targets)) != len(resolved_targets):
        raise ValueError("단일 이미지 출력 경로 충돌")
    for target in resolved_targets:
        if not target.is_relative_to(allowed_root):
            raise ValueError("단일 이미지 출력 allowlist 위반")
        if target.exists() and not overwrite:
            raise FileExistsError(f"출력이 이미 존재합니다: {target.name}")
    allowed_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".health_prediction.tmp-",
            dir=allowed_root,
        )
    )
    backup = Path(
        tempfile.mkdtemp(
            prefix=".health_prediction.backup-",
            dir=allowed_root,
        )
    )
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    activated: list[Path] = []
    preserve_backup = False
    try:
        for index, (target, kind) in enumerate(targets):
            suffix = ".jpg" if kind == "annotated" else ".json"
            staged_path = staging / f"{index:02d}{suffix}"
            if kind == "annotated":
                assert annotated_image is not None
                annotated_image.save(
                    staged_path,
                    format="JPEG",
                    quality=92,
                    subsampling=0,
                )
            else:
                staged_path.write_text(
                    json.dumps(
                        response,
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            staged[target] = staged_path
        for index, target in enumerate(staged):
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup_path = backup / f"{index:02d}_{target.name}"
                os.replace(target, backup_path)
                backups[target] = backup_path
        for target, source in staged.items():
            os.replace(source, target)
            activated.append(target)
    except Exception as activation_error:
        rollback_errors: list[Exception] = []
        for target in reversed(activated):
            if target.exists():
                try:
                    target.unlink()
                except Exception as rollback_error:
                    rollback_errors.append(rollback_error)
        for target, backup_path in reversed(tuple(backups.items())):
            if backup_path.exists():
                try:
                    os.replace(backup_path, target)
                except Exception as rollback_error:
                    rollback_errors.append(rollback_error)
        if rollback_errors:
            preserve_backup = True
            raise RuntimeError(
                "단일 이미지 출력 rollback 실패; 복구용 backup을 보존했습니다"
            ) from activation_error
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and not preserve_backup:
            shutil.rmtree(backup)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model_before = {
        path: file_fingerprint(path)
        for path in (DETECTOR_MODEL_PATH, HEALTH_MODEL_PATH)
    }
    image_before = (
        file_fingerprint(args.image) if args.image.is_file() else None
    )
    annotated_image: Image.Image | None = None
    try:
        image = load_valid_image(args.image)
    except ValueError as exc:
        response = invalid_response(
            "INVALID_IMAGE",
            detection_threshold=args.detection_confidence,
            health_threshold=args.health_threshold,
            detail=str(exc),
        )
    else:
        try:
            with isolated_ultralytics_runtime():
                detector, classifier = load_fixed_models(device=args.device)
                response = predict_health(
                    image,
                    detector=detector,
                    classifier=classifier,
                    detection_threshold=args.detection_confidence,
                    health_threshold=args.health_threshold,
                    padding_ratio=args.padding_ratio,
                )
        except Exception:
            response = invalid_response(
                "INFERENCE_FAILED",
                detection_threshold=args.detection_confidence,
                health_threshold=args.health_threshold,
                detail="모델 추론을 완료하지 못했습니다.",
            )
        if args.annotated_output is not None and response["results"]:
            annotated_image = draw_annotated(image, response)
    assert_deidentified_response(response)
    if {
        path: file_fingerprint(path)
        for path in (DETECTOR_MODEL_PATH, HEALTH_MODEL_PATH)
    } != model_before:
        raise RuntimeError("고정 모델 파일 변경 감지")
    if image_before is not None and file_fingerprint(args.image) != image_before:
        raise RuntimeError("입력 이미지 변경 감지")
    write_optional_outputs(
        response=response,
        annotated_image=annotated_image,
        annotated_output=args.annotated_output,
        json_output=args.json_output,
        overwrite=args.overwrite,
    )
    text = json.dumps(response, ensure_ascii=False, indent=2, allow_nan=False)
    print(text)
    return 0 if response["status"] not in {"INVALID_IMAGE", "INFERENCE_FAILED"} else 2


if __name__ == "__main__":
    sys.exit(main())
