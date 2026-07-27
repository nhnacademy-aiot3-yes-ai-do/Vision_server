#!/usr/bin/env python3
"""Create a camera-disjoint 6,000-image health classification pilot.

The final detection manifest is streamed and the canonical camera assignment
from ``audit_mushroom_health`` is reused.  Existing bounded hash sampling and
selected-member-only ZIP reading are reused from the YOLO pilot utilities.
Only Train/Validation candidates can be selected; Test images are never read.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from tqdm import tqdm

import audit_mushroom_health as health
import audit_mushroom_labels as label_audit
import create_yolo_camera_holdout_pilot as holdout
import create_yolo_pilot_dataset as pilot
import create_yolo_smoke_dataset as smoke
import inspect_aihub_archives as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "detection_dataset_manifest.csv"
DEFAULT_OUTPUT_DIR = ARTIFACTS_ROOT / "health_pilot"
DEFAULT_SEED = 20260726
DEFAULT_TRAIN_PER_STATUS = 500
DEFAULT_VALIDATION_PER_STATUS = 100
DEFAULT_PADDING_RATIO = 0.15
DEFAULT_MAX_TOTAL_IMAGES = 6_000
DEFAULT_MAX_REVIEW_IMAGES = 100
HARD_MAX_TOTAL_IMAGES = 6_000
HARD_MAX_REVIEW_IMAGES = 100
REVIEW_PER_GROUP = 5
MIN_CROP_DIMENSION = 16
TIMESTAMP_X_START_RATIO = 0.60
TIMESTAMP_Y_END_RATIO = 0.12
HEALTH_CLASSES = {
    0: ("HEALTHY", "0_healthy"),
    1: ("DISEASE_SUSPECTED", "1_disease_suspected"),
}
MANIFEST_COLUMNS = (
    "split",
    "health_class_id",
    "health_class_name",
    "species",
    "task",
    "normality",
    "disease_type",
    "camera_id",
    "capture_date",
    "capture_time",
    "image_archive_id",
    "image_member",
    "original_width",
    "original_height",
    "valid_bbox_count",
    "union_bbox",
    "padded_crop_bbox",
    "padding_ratio",
    "crop_width",
    "crop_height",
    "timestamp_region_overlap",
    "output_relative_path",
)
PROTECTED_ARTIFACT_DIRS = (
    ARTIFACTS_ROOT / "yolo_smoke",
    ARTIFACTS_ROOT / "yolo_pilot",
    ARTIFACTS_ROOT / "yolo_pilot_camera_holdout",
    ARTIFACTS_ROOT / "models",
)


@dataclass(frozen=True)
class CropGeometry:
    union_rect: tuple[float, float, float, float]
    crop_rect: tuple[int, int, int, int]
    timestamp_rect: tuple[int, int, int, int]
    timestamp_overlap: bool
    image_width: int
    image_height: int

    @property
    def crop_width(self) -> int:
        return self.crop_rect[2] - self.crop_rect[0]

    @property
    def crop_height(self) -> int:
        return self.crop_rect[3] - self.crop_rect[1]

    @property
    def crop_area_ratio(self) -> float:
        return (
            self.crop_width
            * self.crop_height
            / (self.image_width * self.image_height)
        )


@dataclass(frozen=True)
class CropPlan:
    source: Mapping[str, str]
    geometry: CropGeometry
    health_class_id: int
    health_class_name: str
    class_directory: str
    output_relative_path: str
    unique_key: str


@dataclass(frozen=True)
class HealthSelection:
    camera_stats: holdout.CameraStatistics
    health_assignment: health.CameraAssignment
    selection_assignment: holdout.CameraAssignment
    selected: list[dict[str, str]]


def bounded_ratio(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("실수여야 합니다") from exc
    if not 0.0 < number <= 0.50:
        raise argparse.ArgumentTypeError("padding ratio는 0 초과 0.5 이하여야 합니다")
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "species+camera_id를 Train/Validation 사이에 완전히 분리한 "
            "버섯 건강 체크 분류 파일럿 crop을 생성합니다."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/detection_dataset_manifest.csv"),
    )
    parser.add_argument(
        "--train-per-species-status",
        type=smoke.positive_int,
        default=DEFAULT_TRAIN_PER_STATUS,
    )
    parser.add_argument(
        "--validation-per-species-status",
        type=smoke.positive_int,
        default=DEFAULT_VALIDATION_PER_STATUS,
    )
    parser.add_argument(
        "--padding-ratio",
        type=bounded_ratio,
        default=DEFAULT_PADDING_RATIO,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-total-images",
        type=pilot.bounded_total,
        default=DEFAULT_MAX_TOTAL_IMAGES,
    )
    parser.add_argument(
        "--max-review-images",
        type=pilot.bounded_overlay,
        default=DEFAULT_MAX_REVIEW_IMAGES,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/health_pilot"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 health pilot 출력만 원자적으로 교체",
    )
    args = parser.parse_args(argv)
    args.manifest = pilot.project_path(args.manifest)
    args.output_dir = pilot.project_path(args.output_dir)
    validate_output_location(args.output_dir, parser)
    validate_requested_limits(
        args.train_per_species_status,
        args.validation_per_species_status,
        args.max_total_images,
        args.max_review_images,
    )
    return args


def validate_output_location(
    output_dir: Path,
    parser: argparse.ArgumentParser | None = None,
) -> None:
    output = output_dir.resolve()
    artifacts = ARTIFACTS_ROOT.resolve()

    def fail(message: str) -> None:
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    if output == artifacts or not output.is_relative_to(artifacts):
        fail("--output-dir은 프로젝트 artifacts의 전용 하위 경로여야 합니다")
    for protected in PROTECTED_ARTIFACT_DIRS:
        protected_path = protected.resolve()
        if (
            output == protected_path
            or output.is_relative_to(protected_path)
            or protected_path.is_relative_to(output)
        ):
            fail("--output-dir은 기존 yolo_* 및 models 경로와 분리해야 합니다")


def validate_requested_limits(
    train_per_species_status: int,
    validation_per_species_status: int,
    max_total_images: int,
    max_review_images: int,
) -> int:
    if max_total_images > HARD_MAX_TOTAL_IMAGES:
        raise ValueError(
            f"건강 파일럿 이미지 강제 상한은 {HARD_MAX_TOTAL_IMAGES}장입니다"
        )
    if max_review_images > HARD_MAX_REVIEW_IMAGES:
        raise ValueError(
            f"review 이미지 강제 상한은 {HARD_MAX_REVIEW_IMAGES}장입니다"
        )
    return pilot.validate_requested_limits(
        train_per_species_status,
        validation_per_species_status,
        max_total_images,
        max_review_images,
    )


def camera_health_maps(
    stats: holdout.CameraStatistics,
) -> tuple[
    dict[tuple[str, str], Counter[str]],
    dict[tuple[str, str], Counter[str]],
]:
    """Adapt streamed camera strata to the canonical health audit API."""
    health_by_camera: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    disease_by_camera: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for camera_key, strata in stats.strata_by_camera.items():
        for (task, normality, disease_type), count in strata.items():
            _state, issues = health.health_label_issues(
                task, normality, disease_type
            )
            if issues:
                raise ValueError(
                    "건강 라벨 모순: "
                    f"{camera_key}/{task}/{normality}/{disease_type}: "
                    + ",".join(issues)
                )
            health_by_camera[camera_key][normality] += count
            disease = health.normalized(disease_type)
            if normality == "abnormal" and disease:
                disease_by_camera[camera_key][disease] += count
    return dict(health_by_camera), dict(disease_by_camera)


def canonical_camera_assignment(
    stats: holdout.CameraStatistics,
    *,
    train_target: int,
    validation_target: int,
    seed: int,
) -> tuple[health.CameraAssignment, holdout.CameraAssignment]:
    health_by_camera, disease_by_camera = camera_health_maps(stats)
    canonical = health.choose_camera_holdout(
        health_by_camera,
        disease_by_camera,
        species_values=pilot.SPECIES,
        seed=seed,
        train_target=train_target,
        validation_target=validation_target,
    )
    if not canonical.feasible:
        adapted = holdout.CameraAssignment(
            feasible=False,
            train_cameras=canonical.train_cameras,
            validation_cameras=canonical.validation_cameras,
            pool_counts={},
            reason=canonical.reason,
            closest_shortfall=1,
        )
        return canonical, adapted
    pools = {
        (split, species): holdout.sum_camera_counts(stats, species, cameras)
        for species in pilot.SPECIES
        for split, cameras in (
            ("train", canonical.train_cameras[species]),
            ("validation", canonical.validation_cameras[species]),
        )
    }
    adapted = holdout.CameraAssignment(
        feasible=True,
        train_cameras=canonical.train_cameras,
        validation_cameras=canonical.validation_cameras,
        pool_counts=pools,
    )
    return canonical, adapted


def validate_selected_health_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    train_target: int,
    validation_target: int,
) -> None:
    if any(row["split"] == "test" for row in rows):
        raise ValueError("Test split은 건강 파일럿에 사용할 수 없습니다")
    identities = [smoke.candidate_unique_key(row) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("선택 목록에 중복 이미지가 있습니다")
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        _state, issues = health.health_label_issues(
            row["task"], row["normality"], row["disease_type"]
        )
        if issues:
            raise ValueError(
                f"선택 행 건강 라벨 모순: {smoke.candidate_unique_key(row)}"
            )
        counts[(row["split"], row["species"], row["normality"])] += 1
    mismatches = {}
    for split, target in (
        ("train", train_target),
        ("validation", validation_target),
    ):
        for species in pilot.SPECIES:
            for normality in ("normal", "abnormal"):
                key = (split, species, normality)
                if counts[key] != target:
                    mismatches[key] = (counts[key], target)
    if mismatches:
        raise ValueError(f"품종×상태 선택 수 불일치: {mismatches}")
    if holdout.camera_overlap(rows):
        raise ValueError("Train/Validation species+camera_id 중복")


def select_health_rows(
    manifest_path: Path,
    *,
    train_per_species_status: int,
    validation_per_species_status: int,
    seed: int,
    max_total_images: int,
) -> HealthSelection:
    """Stream the manifest and reuse canonical assignment plus bounded sampling."""
    stats = holdout.stream_camera_statistics(manifest_path)
    canonical, adapted = canonical_camera_assignment(
        stats,
        train_target=train_per_species_status,
        validation_target=validation_per_species_status,
        seed=seed,
    )
    if not canonical.feasible:
        raise ValueError(
            "정확한 건강 camera holdout 구성이 불가능합니다: "
            f"{canonical.limiting_species} {canonical.reason}"
        )
    selected = holdout.select_camera_holdout_rows(
        manifest_path,
        stats,
        adapted,
        train_per_species_task=train_per_species_status,
        validation_per_species_task=validation_per_species_status,
        seed=seed,
        max_total_images=max_total_images,
    )
    validate_selected_health_rows(
        selected,
        train_target=train_per_species_status,
        validation_target=validation_per_species_status,
    )
    return HealthSelection(stats, canonical, adapted, selected)


def numeric_bbox_value(bbox: Mapping[str, Any], name: str) -> float:
    try:
        value = float(bbox[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"bbox {name} 값이 유효하지 않습니다") from exc
    if not math.isfinite(value):
        raise ValueError(f"bbox {name} 값이 유한하지 않습니다")
    return value


def compact_number(value: float) -> int | float:
    if value.is_integer():
        return int(value)
    return round(value, 6)


def rect_as_bbox(rect: Sequence[float]) -> dict[str, int | float]:
    left, top, right, bottom = rect
    return {
        "x": compact_number(float(left)),
        "y": compact_number(float(top)),
        "width": compact_number(float(right - left)),
        "height": compact_number(float(bottom - top)),
    }


def timestamp_candidate_rect(
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    return (
        math.floor(image_width * TIMESTAMP_X_START_RATIO),
        0,
        image_width,
        math.ceil(image_height * TIMESTAMP_Y_END_RATIO),
    )


def rectangles_overlap(
    first: Sequence[float],
    second: Sequence[float],
) -> bool:
    return (
        max(first[0], second[0]) < min(first[2], second[2])
        and max(first[1], second[1]) < min(first[3], second[3])
    )


def crop_geometry(
    row: Mapping[str, str],
    *,
    padding_ratio: float,
) -> CropGeometry:
    width = smoke.positive_dimension(row["image_width"], "image_width")
    height = smoke.positive_dimension(row["image_height"], "image_height")
    try:
        boxes = json.loads(row["bbox_cleaned"])
    except json.JSONDecodeError as exc:
        raise ValueError("bbox_cleaned JSON 파싱 실패") from exc
    if not isinstance(boxes, list) or not boxes:
        raise ValueError("선택 이미지에 유효 bbox가 없습니다")
    if len(boxes) != int(row["valid_bbox_count"]):
        raise ValueError("valid_bbox_count와 bbox_cleaned 개수가 다릅니다")
    bounds: list[tuple[float, float, float, float]] = []
    for bbox in boxes:
        if not isinstance(bbox, Mapping):
            raise ValueError("bbox 항목이 객체가 아닙니다")
        x = numeric_bbox_value(bbox, "x")
        y = numeric_bbox_value(bbox, "y")
        box_width = numeric_bbox_value(bbox, "width")
        box_height = numeric_bbox_value(bbox, "height")
        if box_width <= 0 or box_height <= 0:
            raise ValueError("bbox width와 height는 양수여야 합니다")
        right = x + box_width
        bottom = y + box_height
        if x < 0 or y < 0 or right > width or bottom > height:
            raise ValueError("정제 bbox가 원본 이미지 범위를 벗어납니다")
        bounds.append((x, y, right, bottom))
    union = (
        min(item[0] for item in bounds),
        min(item[1] for item in bounds),
        max(item[2] for item in bounds),
        max(item[3] for item in bounds),
    )
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    crop_rect = (
        max(0, math.floor(union[0] - pad_x)),
        max(0, math.floor(union[1] - pad_y)),
        min(width, math.ceil(union[2] + pad_x)),
        min(height, math.ceil(union[3] + pad_y)),
    )
    crop_width = crop_rect[2] - crop_rect[0]
    crop_height = crop_rect[3] - crop_rect[1]
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("빈 crop입니다")
    if crop_width < MIN_CROP_DIMENSION or crop_height < MIN_CROP_DIMENSION:
        raise ValueError(
            f"지나치게 작은 crop입니다: {crop_width}x{crop_height}"
        )
    timestamp_rect = timestamp_candidate_rect(width, height)
    return CropGeometry(
        union_rect=union,
        crop_rect=crop_rect,
        timestamp_rect=timestamp_rect,
        timestamp_overlap=rectangles_overlap(crop_rect, timestamp_rect),
        image_width=width,
        image_height=height,
    )


def health_class_for_row(
    row: Mapping[str, str],
) -> tuple[int, str, str]:
    _state, issues = health.health_label_issues(
        row["task"], row["normality"], row["disease_type"]
    )
    if issues:
        raise ValueError("건강 class로 변환할 수 없는 모순 라벨")
    if row["normality"] == "normal":
        class_id = 0
    elif row["normality"] == "abnormal":
        class_id = 1
    else:
        raise ValueError(f"지원하지 않는 normality: {row['normality']}")
    name, directory = HEALTH_CLASSES[class_id]
    return class_id, name, directory


def build_crop_plans(
    selected: Sequence[Mapping[str, str]],
    *,
    padding_ratio: float,
) -> list[CropPlan]:
    plans: list[CropPlan] = []
    output_paths: set[str] = set()
    unique_keys: set[str] = set()
    for row in selected:
        if row["split"] == "test":
            raise ValueError("Test 이미지는 crop 계획에 사용할 수 없습니다")
        split_dir = "train" if row["split"] == "train" else "val"
        class_id, class_name, class_directory = health_class_for_row(row)
        basename = smoke.output_basename(row)
        relative = f"{split_dir}/{class_directory}/{basename}"
        unique_key = smoke.candidate_unique_key(row)
        if unique_key in unique_keys:
            raise ValueError("crop 계획에 중복 이미지가 있습니다")
        if relative in output_paths:
            raise ValueError("crop 출력 파일명 충돌")
        unique_keys.add(unique_key)
        output_paths.add(relative)
        plans.append(
            CropPlan(
                source=row,
                geometry=crop_geometry(row, padding_ratio=padding_ratio),
                health_class_id=class_id,
                health_class_name=class_name,
                class_directory=class_directory,
                output_relative_path=relative,
                unique_key=unique_key,
            )
        )
    return plans


def review_group(plan: CropPlan) -> tuple[str, str, int]:
    return (
        str(plan.source["split"]),
        str(plan.source["species"]),
        plan.health_class_id,
    )


def select_review_keys(
    plans: Sequence[CropPlan],
    *,
    seed: int,
    max_review_images: int,
    per_group: int = REVIEW_PER_GROUP,
) -> set[str]:
    if max_review_images > HARD_MAX_REVIEW_IMAGES:
        raise ValueError(
            f"review 이미지 강제 상한은 {HARD_MAX_REVIEW_IMAGES}장입니다"
        )
    buckets: dict[tuple[str, str, int], list[tuple[int, str, str]]] = (
        defaultdict(list)
    )
    for plan in plans:
        priority = 0 if plan.geometry.timestamp_overlap else 1
        rank = health.stable_hash(seed, "review", plan.unique_key)
        buckets[review_group(plan)].append((priority, rank, plan.unique_key))
    for group in buckets:
        buckets[group] = sorted(buckets[group])[:per_group]
    ordered_groups = sorted(
        buckets,
        key=lambda item: (
            0 if item[0] == "train" else 1,
            smoke.CLASS_NAMES_INV[item[1]],
            item[2],
        ),
    )
    selected: set[str] = set()
    depth = 0
    while len(selected) < max_review_images:
        added = False
        for group in ordered_groups:
            if depth >= len(buckets[group]):
                continue
            selected.add(buckets[group][depth][2])
            added = True
            if len(selected) == max_review_images:
                break
        if not added:
            break
        depth += 1
    if len(selected) > HARD_MAX_REVIEW_IMAGES:
        raise RuntimeError("review 안전 상한 초과")
    return selected


def save_rgb_image(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(
            destination,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=False,
        )
    elif suffix == ".png":
        image.save(destination, format="PNG")
    elif suffix == ".bmp":
        image.save(destination, format="BMP")
    elif suffix in {".tif", ".tiff"}:
        image.save(destination, format="TIFF")
    elif suffix == ".webp":
        image.save(destination, format="WEBP", quality=95)
    else:
        raise ValueError(f"지원하지 않는 crop 확장자: {suffix}")


def draw_review_image(
    image: Image.Image,
    plan: CropPlan,
    destination: Path,
) -> None:
    review = image.copy()
    draw = ImageDraw.Draw(review, "RGBA")
    line_width = max(3, round(min(review.size) / 250))
    crop_rect = plan.geometry.crop_rect
    timestamp_rect = plan.geometry.timestamp_rect
    draw.rectangle(crop_rect, outline=(255, 48, 48, 255), width=line_width)
    draw.rectangle(
        timestamp_rect,
        outline=(255, 210, 0, 255),
        width=max(2, line_width // 2),
    )
    font = label_audit.find_font(max(16, round(min(review.size) / 45)))
    status_ko = "정상" if plan.health_class_id == 0 else "병해 의심"
    text = "\n".join(
        (
            f"{status_ko} | {plan.source['species']}",
            f"camera {plan.source['camera_id']} | {plan.source['capture_date']}",
            f"crop {crop_rect[0]},{crop_rect[1]}-"
            f"{crop_rect[2]},{crop_rect[3]}",
            "timestamp ROI overlap: "
            f"{'yes' if plan.geometry.timestamp_overlap else 'no'}",
        )
    )
    text_box = draw.multiline_textbbox(
        (0, 0), text, font=font, spacing=4
    )
    panel_width = min(review.width, text_box[2] - text_box[0] + 24)
    panel_height = min(review.height, text_box[3] - text_box[1] + 20)
    draw.rectangle(
        (0, 0, panel_width, panel_height),
        fill=(0, 0, 0, 180),
    )
    draw.multiline_text(
        (12, 8),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        spacing=4,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    review.save(destination, format="JPEG", quality=90)


def manifest_row_for_plan(
    plan: CropPlan,
    *,
    padding_ratio: float,
) -> dict[str, Any]:
    source = plan.source
    geometry = plan.geometry
    return {
        "split": source["split"],
        "health_class_id": plan.health_class_id,
        "health_class_name": plan.health_class_name,
        "species": source["species"],
        "task": source["task"],
        "normality": source["normality"],
        "disease_type": health.normalized(source["disease_type"]),
        "camera_id": source["camera_id"],
        "capture_date": source["capture_date"],
        "capture_time": source["capture_time"],
        "image_archive_id": source["image_archive_id"],
        "image_member": base.normalize_member_name(source["image_member"]),
        "original_width": geometry.image_width,
        "original_height": geometry.image_height,
        "valid_bbox_count": source["valid_bbox_count"],
        "union_bbox": json.dumps(
            rect_as_bbox(geometry.union_rect),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "padded_crop_bbox": json.dumps(
            rect_as_bbox(geometry.crop_rect),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "padding_ratio": f"{padding_ratio:.6g}",
        "crop_width": geometry.crop_width,
        "crop_height": geometry.crop_height,
        "timestamp_region_overlap": geometry.timestamp_overlap,
        "output_relative_path": plan.output_relative_path,
    }


def extract_health_crops(
    plans: Sequence[CropPlan],
    archive_lookup: Mapping[str, base.ArchiveRef],
    output_root: Path,
    *,
    review_keys: set[str],
    padding_ratio: float,
    show_progress: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plan_by_key = {plan.unique_key: plan for plan in plans}
    rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    progress = tqdm(
        total=len(plans),
        desc="건강 파일럿 crop",
        unit="image",
        disable=not show_progress,
    )
    try:
        selected = [plan.source for plan in plans]
        for source, image_bytes in pilot.iter_selected_image_bytes(
            selected, archive_lookup
        ):
            unique_key = smoke.candidate_unique_key(source)
            plan = plan_by_key[unique_key]
            try:
                with Image.open(io.BytesIO(image_bytes)) as opened:
                    opened.load()
                    if opened.size != (
                        plan.geometry.image_width,
                        plan.geometry.image_height,
                    ):
                        raise ValueError(
                            "manifest/원본 이미지 크기 불일치: "
                            f"{opened.size} vs "
                            f"{plan.geometry.image_width}x"
                            f"{plan.geometry.image_height}"
                        )
                    image = opened.convert("RGB")
            except (UnidentifiedImageError, OSError) as exc:
                raise ValueError(
                    f"손상되거나 지원하지 않는 이미지: {unique_key}"
                ) from exc
            crop = image.crop(plan.geometry.crop_rect)
            if crop.size != (
                plan.geometry.crop_width,
                plan.geometry.crop_height,
            ):
                raise RuntimeError("crop 크기 검증 실패")
            if min(crop.size) < MIN_CROP_DIMENSION:
                raise RuntimeError("빈 이미지 또는 지나치게 작은 crop")
            destination = output_root / plan.output_relative_path
            save_rgb_image(crop, destination)
            rows.append(
                manifest_row_for_plan(
                    plan,
                    padding_ratio=padding_ratio,
                )
            )
            if unique_key in review_keys:
                split_dir = (
                    "train" if source["split"] == "train" else "val"
                )
                review_path = (
                    output_root
                    / "review"
                    / split_dir
                    / f"{Path(destination.name).stem}_review.jpg"
                )
                draw_review_image(image, plan, review_path)
                review_rows.append(
                    {
                        "split": source["split"],
                        "species": source["species"],
                        "health_class_id": plan.health_class_id,
                        "timestamp_region_overlap": (
                            plan.geometry.timestamp_overlap
                        ),
                        "review_relative_path": smoke.portable_relative(
                            review_path, output_root
                        ),
                        "unique_key": unique_key,
                    }
                )
            progress.update(1)
    finally:
        progress.close()
    rows.sort(
        key=lambda row: (
            0 if row["split"] == "train" else 1,
            smoke.CLASS_NAMES_INV[str(row["species"])],
            int(row["health_class_id"]),
            str(row["image_member"]),
        )
    )
    review_rows.sort(
        key=lambda row: (
            0 if row["split"] == "train" else 1,
            smoke.CLASS_NAMES_INV[str(row["species"])],
            int(row["health_class_id"]),
            str(row["review_relative_path"]),
        )
    )
    return rows, review_rows


def create_contact_sheets(
    output_root: Path,
    review_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, int], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in review_rows:
        grouped[
            (
                str(row["split"]),
                str(row["species"]),
                int(row["health_class_id"]),
            )
        ].append(row)
    results: list[dict[str, Any]] = []
    tile_size = (320, 240)
    for group in sorted(
        grouped,
        key=lambda item: (
            0 if item[0] == "train" else 1,
            smoke.CLASS_NAMES_INV[item[1]],
            item[2],
        ),
    ):
        split, species, class_id = group
        items = grouped[group]
        columns = min(5, len(items))
        rows_count = math.ceil(len(items) / columns)
        sheet = Image.new(
            "RGB",
            (columns * tile_size[0], rows_count * tile_size[1]),
            (28, 28, 28),
        )
        for index, item in enumerate(items):
            review_path = output_root / str(item["review_relative_path"])
            with Image.open(review_path) as opened:
                opened.load()
                tile = ImageOps.contain(opened.convert("RGB"), tile_size)
            left = (index % columns) * tile_size[0]
            top = (index // columns) * tile_size[1]
            left += (tile_size[0] - tile.width) // 2
            top += (tile_size[1] - tile.height) // 2
            sheet.paste(tile, (left, top))
        class_slug = HEALTH_CLASSES[class_id][1]
        split_slug = "train" if split == "train" else "val"
        destination = (
            output_root
            / "review"
            / "contact_sheets"
            / f"{split_slug}_{smoke.CLASS_NAMES_INV[species]}_"
            f"{class_slug}.jpg"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(destination, format="JPEG", quality=90)
        results.append(
            {
                "split": split,
                "species": species,
                "health_class_id": class_id,
                "tile_count": len(items),
                "relative_path": smoke.portable_relative(
                    destination, output_root
                ),
            }
        )
    return results


def snapshot_artifacts_excluding(
    artifacts_root: Path,
    output_dir: Path,
) -> dict[str, tuple[int, int]]:
    if not artifacts_root.exists():
        return {}
    root = artifacts_root.resolve()
    excluded = output_dir.resolve()
    transient_prefixes = (
        f".{excluded.name}.tmp-",
        f".{excluded.name}.backup-",
    )

    def is_allowed_output(path: Path) -> bool:
        if path == excluded or path.is_relative_to(excluded):
            return True
        try:
            relative_to_parent = path.relative_to(excluded.parent)
        except ValueError:
            return False
        return bool(relative_to_parent.parts) and relative_to_parent.parts[
            0
        ].startswith(transient_prefixes)

    snapshot: dict[str, tuple[int, int]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if is_allowed_output(resolved):
            continue
        stat = path.stat()
        snapshot[path.relative_to(root).as_posix()] = (
            stat.st_size,
            stat.st_mtime_ns,
        )
    return snapshot


def assert_artifacts_unchanged(
    artifacts_root: Path,
    output_dir: Path,
    snapshot: Mapping[str, tuple[int, int]],
) -> None:
    current = snapshot_artifacts_excluding(artifacts_root, output_dir)
    if current != dict(snapshot):
        raise RuntimeError("허용된 health_pilot 외 기존 artifacts 변경 감지")


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def crop_distribution_rows(
    manifest_rows: Sequence[Mapping[str, Any]],
) -> list[tuple[Any, ...]]:
    result: list[tuple[Any, ...]] = []
    scopes = (
        ("all", list(manifest_rows)),
        (
            "train",
            [row for row in manifest_rows if row["split"] == "train"],
        ),
        (
            "validation",
            [
                row
                for row in manifest_rows
                if row["split"] == "validation"
            ],
        ),
    )
    for scope, rows in scopes:
        widths = [float(row["crop_width"]) for row in rows]
        heights = [float(row["crop_height"]) for row in rows]
        areas = [
            float(row["crop_width"])
            * float(row["crop_height"])
            / (
                float(row["original_width"])
                * float(row["original_height"])
            )
            for row in rows
        ]
        result.append(
            (
                scope,
                len(rows),
                round(min(widths), 2),
                round(percentile(widths, 0.05), 2),
                round(percentile(widths, 0.50), 2),
                round(percentile(widths, 0.95), 2),
                round(max(widths), 2),
                round(min(heights), 2),
                round(percentile(heights, 0.05), 2),
                round(percentile(heights, 0.50), 2),
                round(percentile(heights, 0.95), 2),
                round(max(heights), 2),
                f"{percentile(areas, 0.05) * 100:.2f}%",
                f"{percentile(areas, 0.50) * 100:.2f}%",
                f"{percentile(areas, 0.95) * 100:.2f}%",
            )
        )
    return result


def markdown_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def validate_health_output(
    output_root: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    train_target: int,
    validation_target: int,
    max_review_images: int,
    review_rows: Sequence[Mapping[str, Any]],
    contact_sheets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_paths = {
        str(row["output_relative_path"]) for row in manifest_rows
    }
    actual_paths = {
        smoke.portable_relative(path, output_root)
        for split in ("train", "val")
        for class_directory in (
            "0_healthy",
            "1_disease_suspected",
        )
        for path in (output_root / split / class_directory).glob("*")
        if path.is_file() and path.suffix.lower() in base.IMAGE_EXTENSIONS
    }
    missing = expected_paths - actual_paths
    unexpected = actual_paths - expected_paths
    corrupt = 0
    empty = 0
    dimension_mismatches = 0
    for row in manifest_rows:
        path = output_root / str(row["output_relative_path"])
        if not path.exists() or path.stat().st_size <= 0:
            empty += 1
            continue
        try:
            with Image.open(path) as opened:
                opened.load()
                if opened.size != (
                    int(row["crop_width"]),
                    int(row["crop_height"]),
                ):
                    dimension_mismatches += 1
        except (UnidentifiedImageError, OSError):
            corrupt += 1
    group_counts = Counter(
        (
            str(row["split"]),
            str(row["species"]),
            str(row["normality"]),
        )
        for row in manifest_rows
    )
    group_mismatches = {}
    for split, target in (
        ("train", train_target),
        ("validation", validation_target),
    ):
        for species in pilot.SPECIES:
            for normality in ("normal", "abnormal"):
                key = (split, species, normality)
                if group_counts[key] != target:
                    group_mismatches[key] = (group_counts[key], target)
    train_groups = {
        (str(row["species"]), str(row["camera_id"]))
        for row in manifest_rows
        if row["split"] == "train"
    }
    validation_groups = {
        (str(row["species"]), str(row["camera_id"]))
        for row in manifest_rows
        if row["split"] == "validation"
    }
    source_keys = [
        (
            str(row["image_archive_id"]),
            base.normalize_member_name(str(row["image_member"])).casefold(),
        )
        for row in manifest_rows
    ]
    class_directory_mismatches = 0
    invalid_crop_rects = 0
    timestamp_count = 0
    for row in manifest_rows:
        expected_class_dir = HEALTH_CLASSES[int(row["health_class_id"])][1]
        parts = Path(str(row["output_relative_path"])).parts
        if len(parts) < 3 or parts[1] != expected_class_dir:
            class_directory_mismatches += 1
        bbox = json.loads(str(row["padded_crop_bbox"]))
        x = float(bbox["x"])
        y = float(bbox["y"])
        width = float(bbox["width"])
        height = float(bbox["height"])
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > float(row["original_width"])
            or y + height > float(row["original_height"])
        ):
            invalid_crop_rects += 1
        if str(row["timestamp_region_overlap"]).lower() == "true":
            timestamp_count += 1
    review_files = {
        smoke.portable_relative(path, output_root)
        for split in ("train", "val")
        for path in (output_root / "review" / split).glob("*.jpg")
        if path.is_file()
    }
    expected_review = {
        str(row["review_relative_path"]) for row in review_rows
    }
    validation = {
        "image_count": len(actual_paths),
        "train_images": sum(
            row["split"] == "train" for row in manifest_rows
        ),
        "validation_images": sum(
            row["split"] == "validation" for row in manifest_rows
        ),
        "test_images": sum(row["split"] == "test" for row in manifest_rows),
        "missing_images": len(missing),
        "unexpected_images": len(unexpected),
        "corrupt_images": corrupt,
        "empty_crops": empty,
        "dimension_mismatches": dimension_mismatches,
        "group_mismatches": group_mismatches,
        "camera_overlap": len(train_groups & validation_groups),
        "train_camera_groups": len(train_groups),
        "validation_camera_groups": len(validation_groups),
        "duplicate_images": len(source_keys) - len(set(source_keys)),
        "output_filename_collisions": (
            len(manifest_rows) - len(expected_paths)
        ),
        "class_directory_mismatches": class_directory_mismatches,
        "invalid_crop_rects": invalid_crop_rects,
        "timestamp_overlap_count": timestamp_count,
        "review_count": len(review_files),
        "missing_review": len(expected_review - review_files),
        "unexpected_review": len(review_files - expected_review),
        "contact_sheet_count": len(contact_sheets),
        "contact_sheet_tiles": sum(
            int(row["tile_count"]) for row in contact_sheets
        ),
    }
    failures = {
        key: value
        for key, value in validation.items()
        if key
        in {
            "test_images",
            "missing_images",
            "unexpected_images",
            "corrupt_images",
            "empty_crops",
            "dimension_mismatches",
            "camera_overlap",
            "duplicate_images",
            "output_filename_collisions",
            "class_directory_mismatches",
            "invalid_crop_rects",
            "missing_review",
            "unexpected_review",
        }
        and value
    }
    if group_mismatches:
        failures["group_mismatches"] = group_mismatches
    if len(actual_paths) != len(manifest_rows):
        failures["image_count"] = (len(actual_paths), len(manifest_rows))
    if validation["review_count"] > max_review_images:
        failures["review_count"] = validation["review_count"]
    if validation["contact_sheet_tiles"] != validation["review_count"]:
        failures["contact_sheet_tiles"] = (
            validation["contact_sheet_tiles"],
            validation["review_count"],
        )
    if failures:
        raise RuntimeError(f"건강 파일럿 출력 검증 실패: {failures}")
    return validation


def write_class_mapping(output_root: Path) -> None:
    payload = {
        "schema_version": 1,
        "task": "mushroom_health_binary_classification",
        "classes": {
            str(class_id): {
                "name": class_name,
                "directory": directory,
            }
            for class_id, (class_name, directory) in HEALTH_CLASSES.items()
        },
        "uncertain_policy": (
            "UNCERTAIN is an inference threshold state, not a training class."
        ),
    }
    smoke.write_text(
        output_root / "class_mapping.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def write_summary(
    output_root: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    selection: HealthSelection,
    validation: Mapping[str, Any],
    *,
    seed: int,
    train_target: int,
    validation_target: int,
    padding_ratio: float,
    max_total_images: int,
    max_review_images: int,
) -> None:
    species_status = Counter(
        (
            str(row["split"]),
            str(row["species"]),
            str(row["normality"]),
        )
        for row in manifest_rows
    )
    disease = Counter(
        (str(row["split"]), str(row["disease_type"]))
        for row in manifest_rows
        if row["normality"] == "abnormal"
    )
    timestamp = Counter(
        str(row["split"])
        for row in manifest_rows
        if str(row["timestamp_region_overlap"]).lower() == "true"
    )
    cameras = {
        split: {
            species: tuple(
                selection.health_assignment.train_cameras[species]
                if split == "train"
                else selection.health_assignment.validation_cameras[species]
            )
            for species in pilot.SPECIES
        }
        for split in ("train", "validation")
    }
    lines = [
        "# 버섯 건강 체크 v1 분류 파일럿 데이터셋",
        "",
        "## 실행 범위",
        "",
        f"- seed: `{seed}`",
        "- 그룹 키: `species + camera_id`",
        f"- Train 품종×상태: `{train_target}`장",
        f"- Validation 품종×상태: `{validation_target}`장",
        f"- 이미지 강제 상한: `{max_total_images}`장 "
        f"(코드 절대 상한 {HARD_MAX_TOTAL_IMAGES})",
        f"- review 강제 상한: `{max_review_images}`장 "
        f"(코드 절대 상한 {HARD_MAX_REVIEW_IMAGES})",
        f"- bbox union에 원본 이미지 폭·높이 기준 `{padding_ratio:.2f}` "
        "padding 후 경계 clip",
        "- crop 강제 resize 없음",
        "- Test 선택·이미지 읽기·추출: 0장",
        "- 선택된 이미지 ZIP 멤버만 읽었으며 전체 압축 해제 없음",
        "- 모델 학습 및 병해 종류 모델 생성 없음",
        "",
        "## 결과",
        "",
        f"- Train: **{validation['train_images']:,}장**",
        f"- Validation: **{validation['validation_images']:,}장**",
        f"- Train/Validation 카메라 그룹: "
        f"**{validation['train_camera_groups']:,}/"
        f"{validation['validation_camera_groups']:,}개**",
        f"- 카메라 그룹 중복: **{validation['camera_overlap']:,}개**",
        f"- 동일 원천 이미지 중복: **{validation['duplicate_images']:,}개**",
        f"- 누락/손상/빈 crop: **{validation['missing_images']:,}/"
        f"{validation['corrupt_images']:,}/{validation['empty_crops']:,}개**",
        f"- 출력 파일명 충돌: "
        f"**{validation['output_filename_collisions']:,}개**",
        f"- 범위 밖 crop: **{validation['invalid_crop_rects']:,}개**",
        f"- 원본 ZIP 변경: **없음**",
        f"- health_pilot 외 기존 artifacts 변경: **없음**",
        "",
        "## 품종×상태",
        "",
        markdown_table(
            ("split", "품종", "상태", "이미지"),
            (
                (
                    split,
                    species,
                    "HEALTHY" if normality == "normal" else "DISEASE_SUSPECTED",
                    species_status[(split, species, normality)],
                )
                for split in ("train", "validation")
                for species in pilot.SPECIES
                for normality in ("normal", "abnormal")
            ),
        ),
        "",
        "## 병해 종류 분포",
        "",
        markdown_table(
            ("split", "병해 종류", "이미지"),
            (
                (split, disease_type, count)
                for (split, disease_type), count in sorted(disease.items())
            ),
        ),
        "",
        "## 카메라 배정",
        "",
        markdown_table(
            ("split", "품종", "카메라 수", "camera_id"),
            (
                (
                    split,
                    species,
                    len(cameras[split][species]),
                    ", ".join(cameras[split][species]),
                )
                for split in ("train", "validation")
                for species in pilot.SPECIES
            ),
        ),
        "",
        "## crop 크기 분포",
        "",
        markdown_table(
            (
                "scope",
                "N",
                "W min",
                "W p5",
                "W p50",
                "W p95",
                "W max",
                "H min",
                "H p5",
                "H p50",
                "H p95",
                "H max",
                "면적 p5",
                "면적 p50",
                "면적 p95",
            ),
            crop_distribution_rows(manifest_rows),
        ),
        "",
        "면적은 원본 이미지 면적 대비 padded crop 비율이다.",
        "",
        "## 잠재 timestamp 영역",
        "",
        "- 정적 후보 영역: 원본 이미지의 우측 40% × 상단 12% "
        "(`x >= 0.60W`, `y < 0.12H`)",
        "- 이는 실제 timestamp 검출 결과가 아니며 OCR을 수행하지 않았다.",
        "- timestamp를 삭제하거나 마스킹하지 않았다.",
        f"- 전체 overlap: **{validation['timestamp_overlap_count']:,}/"
        f"{len(manifest_rows):,}장 "
        f"({validation['timestamp_overlap_count'] / len(manifest_rows):.2%})**",
        f"- Train overlap: **{timestamp['train']:,}/"
        f"{validation['train_images']:,}장 "
        f"({timestamp['train'] / validation['train_images']:.2%})**",
        f"- Validation overlap: **{timestamp['validation']:,}/"
        f"{validation['validation_images']:,}장 "
        f"({timestamp['validation'] / validation['validation_images']:.2%})**",
        f"- review/contact-sheet 표본: "
        f"**{validation['review_count']:,}장/"
        f"{validation['contact_sheet_count']:,}개**",
        "- 각 품종×상태×split에서 overlap 표본을 우선 선택했다.",
        "",
        "## 검증",
        "",
        f"- 품종×상태 수량 불일치: "
        f"{len(validation['group_mismatches']):,}개",
        f"- class directory 불일치: "
        f"{validation['class_directory_mismatches']:,}개",
        f"- crop 크기 불일치: "
        f"{validation['dimension_mismatches']:,}개",
        f"- review 누락/예상 밖: "
        f"{validation['missing_review']:,}/"
        f"{validation['unexpected_review']:,}개",
        "- 로컬 절대경로 노출: 0개",
        "",
    ]
    smoke.write_text(
        output_root / "health_pilot_summary.md",
        "\n".join(lines),
    )


def assert_deidentified_output(output_root: Path) -> None:
    text_paths = [
        path
        for path in output_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".csv", ".json", ".md", ".txt", ".yaml"}
    ]
    health.assert_deidentified(text_paths)


def create_health_dataset(
    selection: HealthSelection,
    archive_lookup: Mapping[str, base.ArchiveRef],
    output_dir: Path,
    *,
    train_per_species_status: int,
    validation_per_species_status: int,
    padding_ratio: float,
    seed: int,
    max_total_images: int,
    max_review_images: int,
    overwrite: bool,
    artifacts_root: Path = ARTIFACTS_ROOT,
    show_progress: bool = True,
) -> dict[str, Any]:
    expected_total = validate_requested_limits(
        train_per_species_status,
        validation_per_species_status,
        max_total_images,
        max_review_images,
    )
    selected = selection.selected
    if len(selected) != expected_total:
        raise ValueError(
            f"선택 이미지 {len(selected)}장(기대값 {expected_total}장)"
        )
    validate_selected_health_rows(
        selected,
        train_target=train_per_species_status,
        validation_target=validation_per_species_status,
    )
    plans = build_crop_plans(selected, padding_ratio=padding_ratio)
    review_keys = select_review_keys(
        plans,
        seed=seed,
        max_review_images=max_review_images,
    )
    source_snapshot = label_audit.source_zip_snapshot(
        ref.path for ref in archive_lookup.values()
    )
    artifacts_snapshot = snapshot_artifacts_excluding(
        artifacts_root, output_dir
    )
    staging, backup = smoke.prepare_staging(
        output_dir, overwrite=overwrite
    )
    try:
        for relative in (
            "train/0_healthy",
            "train/1_disease_suspected",
            "val/0_healthy",
            "val/1_disease_suspected",
            "review/train",
            "review/val",
            "review/contact_sheets",
        ):
            (staging / relative).mkdir(parents=True, exist_ok=True)
        write_class_mapping(staging)
        manifest_rows, review_rows = extract_health_crops(
            plans,
            archive_lookup,
            staging,
            review_keys=review_keys,
            padding_ratio=padding_ratio,
            show_progress=show_progress,
        )
        smoke.write_csv(
            staging / "health_pilot_manifest.csv",
            MANIFEST_COLUMNS,
            manifest_rows,
        )
        contact_sheets = create_contact_sheets(staging, review_rows)
        validation = validate_health_output(
            staging,
            manifest_rows,
            train_target=train_per_species_status,
            validation_target=validation_per_species_status,
            max_review_images=max_review_images,
            review_rows=review_rows,
            contact_sheets=contact_sheets,
        )
        write_summary(
            staging,
            manifest_rows,
            selection,
            validation,
            seed=seed,
            train_target=train_per_species_status,
            validation_target=validation_per_species_status,
            padding_ratio=padding_ratio,
            max_total_images=max_total_images,
            max_review_images=max_review_images,
        )
        assert_deidentified_output(staging)
        label_audit.assert_source_zips_unchanged(source_snapshot)
        assert_artifacts_unchanged(
            artifacts_root, output_dir, artifacts_snapshot
        )
        smoke.finalize_staging(staging, output_dir, backup)
        assert_artifacts_unchanged(
            artifacts_root, output_dir, artifacts_snapshot
        )
    except Exception:
        smoke.rollback_staging(staging, output_dir, backup)
        label_audit.assert_source_zips_unchanged(source_snapshot)
        assert_artifacts_unchanged(
            artifacts_root, output_dir, artifacts_snapshot
        )
        raise
    species_status = Counter(
        (
            str(row["split"]),
            str(row["species"]),
            str(row["normality"]),
        )
        for row in manifest_rows
    )
    disease = Counter(
        (str(row["split"]), str(row["disease_type"]))
        for row in manifest_rows
        if row["normality"] == "abnormal"
    )
    return {
        "train_images": validation["train_images"],
        "validation_images": validation["validation_images"],
        "total_images": validation["image_count"],
        "species_status_distribution": [
            {
                "split": split,
                "species": species,
                "status": (
                    "HEALTHY"
                    if normality == "normal"
                    else "DISEASE_SUSPECTED"
                ),
                "count": species_status[(split, species, normality)],
            }
            for split in ("train", "validation")
            for species in pilot.SPECIES
            for normality in ("normal", "abnormal")
        ],
        "disease_distribution": [
            {
                "split": split,
                "disease_type": disease_type,
                "count": count,
            }
            for (split, disease_type), count in sorted(disease.items())
        ],
        "train_camera_groups": validation["train_camera_groups"],
        "validation_camera_groups": validation["validation_camera_groups"],
        "camera_overlap": validation["camera_overlap"],
        "crop_distribution": [
            {
                "scope": row[0],
                "count": row[1],
                "width_min": row[2],
                "width_p50": row[4],
                "width_max": row[6],
                "height_min": row[7],
                "height_p50": row[9],
                "height_max": row[11],
                "area_p50": row[13],
            }
            for row in crop_distribution_rows(manifest_rows)
        ],
        "timestamp_overlap_count": validation["timestamp_overlap_count"],
        "timestamp_overlap_rate": (
            validation["timestamp_overlap_count"] / len(manifest_rows)
        ),
        "review_count": validation["review_count"],
        "contact_sheet_count": validation["contact_sheet_count"],
        "missing_images": validation["missing_images"],
        "corrupt_images": validation["corrupt_images"],
        "empty_crops": validation["empty_crops"],
        "test_images": validation["test_images"],
        "source_zip_modified": False,
        "protected_artifacts_modified": False,
    }


def run(
    args: argparse.Namespace,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    manifest_snapshot = health.snapshot_files((args.manifest,))
    try:
        selection = select_health_rows(
            args.manifest,
            train_per_species_status=args.train_per_species_status,
            validation_per_species_status=args.validation_per_species_status,
            seed=args.seed,
            max_total_images=args.max_total_images,
        )
        refs, issues, _directories = base.discover_archives(environ)
        if issues:
            raise RuntimeError(
                "아카이브 환경/구조 검증 실패:\n- " + "\n- ".join(issues)
            )
        lookup = smoke.image_archive_lookup(refs)
        result = create_health_dataset(
            selection,
            lookup,
            args.output_dir,
            train_per_species_status=args.train_per_species_status,
            validation_per_species_status=args.validation_per_species_status,
            padding_ratio=args.padding_ratio,
            seed=args.seed,
            max_total_images=args.max_total_images,
            max_review_images=args.max_review_images,
            overwrite=args.overwrite,
        )
    finally:
        health.assert_snapshot_unchanged(manifest_snapshot)
    result.update(
        {
            "output_dir": base.portable_output_path(args.output_dir),
            "hard_max_total_images": HARD_MAX_TOTAL_IMAGES,
            "hard_max_review_images": HARD_MAX_REVIEW_IMAGES,
            "full_archive_extraction": False,
            "test_split_used": False,
            "model_training": False,
            "disease_model_created": False,
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args, os.environ)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
