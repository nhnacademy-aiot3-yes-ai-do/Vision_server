#!/usr/bin/env python3
"""Create a timestamp-candidate-masked copy of the health pilot dataset.

The existing 6,000-row health pilot manifest is used one-to-one.  No sampling,
split changes, source ZIP access, model loading, or training occurs.  Every raw
crop is decoded to RGB and encoded with the same JPEG settings, whether or not
the candidate timestamp region intersects the crop.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageOps, JpegImagePlugin, UnidentifiedImageError
from tqdm import tqdm

import audit_mushroom_health as health
import create_health_pilot_dataset as raw_health
import create_yolo_smoke_dataset as smoke
import inspect_aihub_archives as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_SOURCE_DIR = ARTIFACTS_ROOT / "health_pilot"
DEFAULT_SOURCE_MANIFEST = (
    DEFAULT_SOURCE_DIR / "health_pilot_manifest.csv"
)
DEFAULT_OUTPUT_DIR = ARTIFACTS_ROOT / "health_pilot_timestamp_masked"
DEFAULT_MASK_COLOR = 127
DEFAULT_JPEG_QUALITY = 95
DEFAULT_MAX_REVIEW_PAIRS = 100
DEFAULT_SEED = 20260726
EXPECTED_IMAGES = 6_000
HARD_MAX_REVIEW_PAIRS = 100
REVIEW_PAIRS_PER_GROUP = 5
PROCESSING_VERSION = "health-timestamp-mask-v1"
ADDED_MANIFEST_COLUMNS = (
    "timestamp_candidate_overlap",
    "timestamp_mask_applied",
    "mask_rect_crop_coordinates",
    "mask_width",
    "mask_height",
    "mask_area_ratio",
    "source_raw_relative_path",
)
MASKED_MANIFEST_COLUMNS = (
    *(
        column
        for column in raw_health.MANIFEST_COLUMNS
        if column != "output_relative_path"
    ),
    *ADDED_MANIFEST_COLUMNS,
    "output_relative_path",
)


@dataclass(frozen=True)
class MaskGeometry:
    original_timestamp_rect: tuple[int, int, int, int]
    crop_rect_original: tuple[int, int, int, int]
    mask_rect_crop: tuple[int, int, int, int] | None
    crop_width: int
    crop_height: int

    @property
    def overlaps(self) -> bool:
        return self.mask_rect_crop is not None

    @property
    def mask_width(self) -> int:
        if self.mask_rect_crop is None:
            return 0
        return self.mask_rect_crop[2] - self.mask_rect_crop[0]

    @property
    def mask_height(self) -> int:
        if self.mask_rect_crop is None:
            return 0
        return self.mask_rect_crop[3] - self.mask_rect_crop[1]

    @property
    def mask_area_ratio(self) -> float:
        return (
            self.mask_width
            * self.mask_height
            / (self.crop_width * self.crop_height)
        )


@dataclass(frozen=True)
class SourceRecord:
    row: dict[str, str]
    source_path: Path
    unique_key: str
    geometry: MaskGeometry


def bounded_byte(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("0~255 정수여야 합니다") from exc
    if not 0 <= number <= 255:
        raise argparse.ArgumentTypeError("0~255 정수여야 합니다")
    return number


def bounded_quality(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("1~100 정수여야 합니다") from exc
    if not 1 <= number <= 100:
        raise argparse.ArgumentTypeError("1~100 정수여야 합니다")
    return number


def bounded_review_pairs(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("0~100 정수여야 합니다") from exc
    if not 0 <= number <= HARD_MAX_REVIEW_PAIRS:
        raise argparse.ArgumentTypeError(
            f"review pair는 0~{HARD_MAX_REVIEW_PAIRS}여야 합니다"
        )
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "기존 건강 파일럿 6,000장을 그대로 유지하면서 원본 기준 "
            "timestamp 후보 영역만 고정 회색으로 마스킹합니다."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("artifacts/health_pilot"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path(
            "artifacts/health_pilot/health_pilot_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/health_pilot_timestamp_masked"),
    )
    parser.add_argument(
        "--mask-color",
        type=bounded_byte,
        default=DEFAULT_MASK_COLOR,
    )
    parser.add_argument(
        "--jpeg-quality",
        type=bounded_quality,
        default=DEFAULT_JPEG_QUALITY,
    )
    parser.add_argument(
        "--max-review-pairs",
        type=bounded_review_pairs,
        default=DEFAULT_MAX_REVIEW_PAIRS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 masked 출력만 원자적으로 교체",
    )
    args = parser.parse_args(argv)
    args.source_dir = raw_health.pilot.project_path(args.source_dir)
    args.source_manifest = raw_health.pilot.project_path(
        args.source_manifest
    )
    args.output_dir = raw_health.pilot.project_path(args.output_dir)
    validate_locations(
        args.source_dir,
        args.source_manifest,
        args.output_dir,
        parser=parser,
    )
    return args


def validate_locations(
    source_dir: Path,
    source_manifest: Path,
    output_dir: Path,
    *,
    artifacts_root: Path = ARTIFACTS_ROOT,
    parser: argparse.ArgumentParser | None = None,
) -> None:
    source = source_dir.resolve()
    manifest = source_manifest.resolve()
    output = output_dir.resolve()
    artifacts = artifacts_root.resolve()

    def fail(message: str) -> None:
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    if not source.is_dir():
        fail("source-dir이 존재하는 디렉터리가 아닙니다")
    if not manifest.is_file() or not manifest.is_relative_to(source):
        fail("source-manifest는 source-dir 내부 파일이어야 합니다")
    if output == artifacts or not output.is_relative_to(artifacts):
        fail("output-dir은 프로젝트 artifacts의 전용 하위 경로여야 합니다")
    if (
        output == source
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        fail("output-dir과 raw source-dir은 완전히 분리되어야 합니다")
    for protected_name in (
        "yolo_smoke",
        "yolo_pilot",
        "yolo_pilot_camera_holdout",
        "models",
    ):
        protected = (artifacts / protected_name).resolve()
        if (
            output == protected
            or output.is_relative_to(protected)
            or protected.is_relative_to(output)
        ):
            fail("output-dir은 기존 yolo_* 및 models 경로와 분리해야 합니다")


def positive_int_value(value: Any, field_name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}가 정수가 아닙니다") from exc
    if number <= 0:
        raise ValueError(f"{field_name}가 양수가 아닙니다")
    return number


def integral_number(value: Any, field_name: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}가 숫자가 아닙니다") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{field_name}가 유한한 정수가 아닙니다")
    return int(number)


def parse_bbox_json(value: str, field_name: str) -> tuple[int, int, int, int]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} JSON 파싱 실패") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{field_name}가 객체가 아닙니다")
    x = integral_number(parsed.get("x"), f"{field_name}.x")
    y = integral_number(parsed.get("y"), f"{field_name}.y")
    width = integral_number(parsed.get("width"), f"{field_name}.width")
    height = integral_number(parsed.get("height"), f"{field_name}.height")
    if width <= 0 or height <= 0:
        raise ValueError(f"{field_name} 크기가 양수가 아닙니다")
    return x, y, x + width, y + height


def timestamp_mask_geometry(
    *,
    original_width: int,
    original_height: int,
    crop_rect_original: Sequence[int],
) -> MaskGeometry:
    if len(crop_rect_original) != 4:
        raise ValueError("crop rect는 4개 좌표여야 합니다")
    crop_left, crop_top, crop_right, crop_bottom = (
        int(value) for value in crop_rect_original
    )
    if (
        crop_left < 0
        or crop_top < 0
        or crop_right <= crop_left
        or crop_bottom <= crop_top
        or crop_right > original_width
        or crop_bottom > original_height
    ):
        raise ValueError("crop rect가 원본 이미지 범위를 벗어납니다")
    timestamp_rect = raw_health.timestamp_candidate_rect(
        original_width, original_height
    )
    intersection = (
        max(crop_left, timestamp_rect[0]),
        max(crop_top, timestamp_rect[1]),
        min(crop_right, timestamp_rect[2]),
        min(crop_bottom, timestamp_rect[3]),
    )
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    mask_rect: tuple[int, int, int, int] | None
    if (
        intersection[0] < intersection[2]
        and intersection[1] < intersection[3]
    ):
        mask_rect = (
            max(0, intersection[0] - crop_left),
            max(0, intersection[1] - crop_top),
            min(crop_width, intersection[2] - crop_left),
            min(crop_height, intersection[3] - crop_top),
        )
        if mask_rect[0] >= mask_rect[2] or mask_rect[1] >= mask_rect[3]:
            raise AssertionError("양의 원본 교차가 빈 crop mask가 됐습니다")
    else:
        mask_rect = None
    return MaskGeometry(
        original_timestamp_rect=timestamp_rect,
        crop_rect_original=(
            crop_left,
            crop_top,
            crop_right,
            crop_bottom,
        ),
        mask_rect_crop=mask_rect,
        crop_width=crop_width,
        crop_height=crop_height,
    )


def safe_relative_path(value: str, source_dir: Path) -> tuple[str, Path]:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"안전하지 않은 상대경로: {value}")
    normalized = relative.as_posix()
    resolved = (source_dir / relative).resolve()
    if not resolved.is_relative_to(source_dir.resolve()):
        raise ValueError(f"source-dir 밖으로 나가는 경로: {value}")
    return normalized, resolved


def raw_unique_key(row: Mapping[str, str]) -> str:
    return (
        f"{row['image_archive_id']}:"
        f"{base.normalize_member_name(row['image_member']).casefold()}"
    )


def boolean_text(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"boolean 값이 아닙니다: {value!r}")


def validate_raw_row(
    row: Mapping[str, str],
    source_dir: Path,
) -> tuple[Path, MaskGeometry]:
    split = row.get("split", "")
    if split == "test":
        raise ValueError("Test row가 raw health manifest에 존재합니다")
    if split not in {"train", "validation"}:
        raise ValueError(f"지원하지 않는 split: {split}")
    class_id = int(row["health_class_id"])
    if class_id not in raw_health.HEALTH_CLASSES:
        raise ValueError(f"지원하지 않는 health class: {class_id}")
    expected_name, expected_directory = raw_health.HEALTH_CLASSES[class_id]
    if row["health_class_name"] != expected_name:
        raise ValueError("health class id/name 불일치")
    _state, issues = health.health_label_issues(
        row["task"], row["normality"], row["disease_type"]
    )
    if issues:
        raise ValueError("raw health manifest에 라벨 모순이 있습니다")
    expected_class_id = 0 if row["normality"] == "normal" else 1
    if class_id != expected_class_id:
        raise ValueError("normality와 health class 불일치")
    output_relative, source_path = safe_relative_path(
        row["output_relative_path"], source_dir
    )
    expected_split_dir = "train" if split == "train" else "val"
    path_parts = Path(output_relative).parts
    if (
        len(path_parts) != 3
        or path_parts[0] != expected_split_dir
        or path_parts[1] != expected_directory
    ):
        raise ValueError("raw output_relative_path의 class 구조가 다릅니다")
    if Path(output_relative).suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError("모든 raw crop은 JPEG여야 합니다")
    original_width = positive_int_value(
        row["original_width"], "original_width"
    )
    original_height = positive_int_value(
        row["original_height"], "original_height"
    )
    crop_rect = parse_bbox_json(
        row["padded_crop_bbox"], "padded_crop_bbox"
    )
    geometry = timestamp_mask_geometry(
        original_width=original_width,
        original_height=original_height,
        crop_rect_original=crop_rect,
    )
    if geometry.crop_width != positive_int_value(
        row["crop_width"], "crop_width"
    ) or geometry.crop_height != positive_int_value(
        row["crop_height"], "crop_height"
    ):
        raise ValueError("padded crop bbox와 crop 크기가 다릅니다")
    if boolean_text(row["timestamp_region_overlap"]) != geometry.overlaps:
        raise ValueError("기존 timestamp overlap과 새 기하 계산이 다릅니다")
    return source_path, geometry


def load_source_records(
    source_manifest: Path,
    source_dir: Path,
    *,
    expected_images: int = EXPECTED_IMAGES,
) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    unique_keys: set[str] = set()
    output_paths: set[str] = set()
    with source_manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != tuple(raw_health.MANIFEST_COLUMNS):
            missing = set(raw_health.MANIFEST_COLUMNS) - set(fieldnames)
            extra = set(fieldnames) - set(raw_health.MANIFEST_COLUMNS)
            raise ValueError(
                "raw manifest schema가 canonical 22개 컬럼과 다릅니다"
                f" (누락={sorted(missing)}, 추가={sorted(extra)})"
            )
        for row in reader:
            if None in row:
                raise ValueError("raw manifest 행에 header 밖 값이 있습니다")
            source_path, geometry = validate_raw_row(row, source_dir)
            key = raw_unique_key(row)
            relative = row["output_relative_path"]
            portable_relative_key = Path(relative).as_posix().casefold()
            if key in unique_keys:
                raise ValueError(f"동일 원천 이미지 중복: {key}")
            if portable_relative_key in output_paths:
                raise ValueError(f"raw 출력 경로 중복: {relative}")
            unique_keys.add(key)
            output_paths.add(portable_relative_key)
            records.append(
                SourceRecord(
                    row=dict(row),
                    source_path=source_path,
                    unique_key=key,
                    geometry=geometry,
                )
            )
    if len(records) != expected_images:
        raise ValueError(
            f"raw manifest {len(records)}행(기대 {expected_images}행)"
        )
    return records


def apply_timestamp_mask(
    image: Image.Image,
    geometry: MaskGeometry,
    *,
    mask_color: int,
) -> Image.Image:
    if image.size != (geometry.crop_width, geometry.crop_height):
        raise ValueError("raw crop 실제 크기와 manifest 크기가 다릅니다")
    result = image.convert("RGB")
    if geometry.mask_rect_crop is not None:
        color = (mask_color, mask_color, mask_color)
        result.paste(color, geometry.mask_rect_crop)
    return result


def save_uniform_jpeg(
    image: Image.Image,
    destination: Path,
    *,
    jpeg_quality: int,
) -> None:
    if destination.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError("masked 출력은 JPEG여야 합니다")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(
        destination,
        format="JPEG",
        quality=jpeg_quality,
        subsampling=0,
        optimize=False,
        progressive=False,
    )


def mask_rect_json(geometry: MaskGeometry) -> str:
    if geometry.mask_rect_crop is None:
        return "null"
    left, top, right, bottom = geometry.mask_rect_crop
    return json.dumps(
        {
            "x": left,
            "y": top,
            "width": right - left,
            "height": bottom - top,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def masked_manifest_row(record: SourceRecord) -> dict[str, Any]:
    row: dict[str, Any] = dict(record.row)
    relative = record.row["output_relative_path"]
    row.update(
        {
            "timestamp_candidate_overlap": record.geometry.overlaps,
            "timestamp_mask_applied": record.geometry.overlaps,
            "mask_rect_crop_coordinates": mask_rect_json(
                record.geometry
            ),
            "mask_width": record.geometry.mask_width,
            "mask_height": record.geometry.mask_height,
            "mask_area_ratio": (
                f"{record.geometry.mask_area_ratio:.10f}"
            ),
            "source_raw_relative_path": relative,
            "output_relative_path": relative,
        }
    )
    return row


def process_images(
    records: Sequence[SourceRecord],
    output_root: Path,
    *,
    mask_color: int,
    jpeg_quality: int,
    show_progress: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    progress = tqdm(
        records,
        total=len(records),
        desc="timestamp masked crop",
        unit="image",
        disable=not show_progress,
    )
    for record in progress:
        try:
            with Image.open(record.source_path) as opened:
                opened.load()
                masked = apply_timestamp_mask(
                    opened,
                    record.geometry,
                    mask_color=mask_color,
                )
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(
                "누락되거나 손상된 raw crop: "
                f"{record.row['output_relative_path']}"
            ) from exc
        destination = (
            output_root / record.row["output_relative_path"]
        )
        save_uniform_jpeg(
            masked,
            destination,
            jpeg_quality=jpeg_quality,
        )
        rows.append(masked_manifest_row(record))
    return rows


def review_group(record: SourceRecord) -> tuple[str, str, int]:
    return (
        record.row["split"],
        record.row["species"],
        int(record.row["health_class_id"]),
    )


def select_review_records(
    records: Sequence[SourceRecord],
    *,
    max_review_pairs: int,
    seed: int,
    per_group: int = REVIEW_PAIRS_PER_GROUP,
) -> list[SourceRecord]:
    if max_review_pairs > HARD_MAX_REVIEW_PAIRS:
        raise ValueError(
            f"review pair는 {HARD_MAX_REVIEW_PAIRS}개 이하여야 합니다"
        )
    buckets: dict[
        tuple[str, str, int], tuple[list[SourceRecord], list[SourceRecord]]
    ] = {}
    grouped: dict[
        tuple[str, str, int], list[SourceRecord]
    ] = defaultdict(list)
    for record in records:
        grouped[review_group(record)].append(record)
    expected_groups = {
        (split, species, class_id)
        for split in ("train", "validation")
        for species in raw_health.pilot.SPECIES
        for class_id in raw_health.HEALTH_CLASSES
    }
    if set(grouped) != expected_groups:
        raise ValueError("review에 필요한 split×species×class 그룹이 부족합니다")
    for group, group_records in grouped.items():
        overlap = sorted(
            (record for record in group_records if record.geometry.overlaps),
            key=lambda record: health.stable_hash(
                seed, "masked-review", record.unique_key
            ),
        )
        non_overlap = sorted(
            (
                record
                for record in group_records
                if not record.geometry.overlaps
            ),
            key=lambda record: health.stable_hash(
                seed, "masked-review", record.unique_key
            ),
        )
        buckets[group] = (overlap, non_overlap)
    candidates: dict[tuple[str, str, int], list[SourceRecord]] = {}
    for group, (overlap, non_overlap) in buckets.items():
        chosen = overlap[:per_group]
        if len(chosen) < per_group:
            chosen.extend(non_overlap[: per_group - len(chosen)])
        candidates[group] = chosen
    ordered_groups = sorted(
        candidates,
        key=lambda group: (
            0 if group[0] == "train" else 1,
            smoke.CLASS_NAMES_INV[group[1]],
            group[2],
        ),
    )
    result: list[SourceRecord] = []
    depth = 0
    while len(result) < max_review_pairs:
        added = False
        for group in ordered_groups:
            if depth >= len(candidates[group]):
                continue
            result.append(candidates[group][depth])
            added = True
            if len(result) == max_review_pairs:
                break
        if not added:
            break
        depth += 1
    return result


def make_pair_tile(
    before: Image.Image,
    after: Image.Image,
    record: SourceRecord,
) -> Image.Image:
    panel_size = (360, 270)
    header_height = 54
    tile = Image.new(
        "RGB",
        (panel_size[0] * 2, panel_size[1] + header_height),
        (30, 30, 30),
    )
    before_panel = ImageOps.contain(before.convert("RGB"), panel_size)
    after_panel = ImageOps.contain(after.convert("RGB"), panel_size)
    before_left = (panel_size[0] - before_panel.width) // 2
    after_left = panel_size[0] + (
        panel_size[0] - after_panel.width
    ) // 2
    top = header_height + (
        panel_size[1] - before_panel.height
    ) // 2
    tile.paste(before_panel, (before_left, top))
    top_after = header_height + (
        panel_size[1] - after_panel.height
    ) // 2
    tile.paste(after_panel, (after_left, top_after))
    draw = ImageDraw.Draw(tile)
    font = raw_health.label_audit.find_font(18)
    status = (
        "HEALTHY"
        if record.row["health_class_id"] == "0"
        else "DISEASE_SUSPECTED"
    )
    title = (
        f"{record.row['split']} | {record.row['species']} | {status} | "
        f"camera {record.row['camera_id']} | "
        f"{record.row['capture_date']} | "
        f"mask {'yes' if record.geometry.overlaps else 'no'}"
    )
    draw.text((8, 5), title, font=font, fill=(255, 255, 255))
    draw.text((8, 29), "BEFORE", font=font, fill=(255, 220, 80))
    draw.text(
        (panel_size[0] + 8, 29),
        "AFTER",
        font=font,
        fill=(255, 220, 80),
    )
    return tile


def create_review_contact_sheets(
    source_dir: Path,
    output_root: Path,
    selected: Sequence[SourceRecord],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, int], list[SourceRecord]
    ] = defaultdict(list)
    for record in selected:
        grouped[review_group(record)].append(record)
    sheets: list[dict[str, Any]] = []
    for group in sorted(
        grouped,
        key=lambda item: (
            0 if item[0] == "train" else 1,
            smoke.CLASS_NAMES_INV[item[1]],
            item[2],
        ),
    ):
        records = grouped[group]
        tiles: list[Image.Image] = []
        for record in records:
            before_path = (
                source_dir / record.row["output_relative_path"]
            )
            after_path = (
                output_root / record.row["output_relative_path"]
            )
            with Image.open(before_path) as before_opened:
                before_opened.load()
                before = before_opened.convert("RGB")
            with Image.open(after_path) as after_opened:
                after_opened.load()
                after = after_opened.convert("RGB")
            if before.size != after.size:
                raise RuntimeError("review before/after crop 크기 불일치")
            tiles.append(make_pair_tile(before, after, record))
        if not tiles:
            continue
        sheet = Image.new(
            "RGB",
            (max(tile.width for tile in tiles), sum(tile.height for tile in tiles)),
            (20, 20, 20),
        )
        y = 0
        for tile in tiles:
            sheet.paste(tile, (0, y))
            y += tile.height
        split, species, class_id = group
        split_slug = "train" if split == "train" else "val"
        destination = (
            output_root
            / "review"
            / "contact_sheets"
            / f"{split_slug}_{smoke.CLASS_NAMES_INV[species]}_"
            f"{raw_health.HEALTH_CLASSES[class_id][1]}.jpg"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(
            destination,
            format="JPEG",
            quality=90,
            subsampling=0,
            optimize=False,
            progressive=False,
        )
        sheets.append(
            {
                "split": split,
                "species": species,
                "health_class_id": class_id,
                "pair_count": len(records),
                "masked_pair_count": sum(
                    record.geometry.overlaps for record in records
                ),
                "relative_path": smoke.portable_relative(
                    destination, output_root
                ),
            }
        )
    return sheets


def jpeg_quantization_signature(image: Image.Image) -> tuple[Any, ...]:
    quantization = getattr(image, "quantization", None)
    if not isinstance(quantization, Mapping):
        return ()
    return tuple(
        (int(table), tuple(int(value) for value in values))
        for table, values in sorted(quantization.items())
    )


def image_set(
    root: Path,
) -> set[str]:
    return {
        smoke.portable_relative(path, root)
        for split in ("train", "val")
        for class_directory in (
            "0_healthy",
            "1_disease_suspected",
        )
        for path in (root / split / class_directory).glob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    }


def validate_output(
    records: Sequence[SourceRecord],
    manifest_rows: Sequence[Mapping[str, Any]],
    output_root: Path,
    *,
    review_sheets: Sequence[Mapping[str, Any]],
    max_review_pairs: int,
) -> dict[str, Any]:
    if len(records) != len(manifest_rows):
        raise RuntimeError("raw/masked manifest 행 수가 다릅니다")
    preserved_mismatches = 0
    identity_mismatches = 0
    split_changes = 0
    class_changes = 0
    camera_changes = 0
    crop_size_changes = 0
    mask_bounds_errors = 0
    overlap_flag_mismatches = 0
    for record, output in zip(records, manifest_rows):
        if raw_unique_key(record.row) != raw_unique_key(output):
            identity_mismatches += 1
        for column in raw_health.MANIFEST_COLUMNS:
            if str(record.row.get(column, "")) != str(
                output.get(column, "")
            ):
                preserved_mismatches += 1
                break
        if output["split"] != record.row["split"]:
            split_changes += 1
        if (
            str(output["health_class_id"])
            != record.row["health_class_id"]
            or output["health_class_name"]
            != record.row["health_class_name"]
        ):
            class_changes += 1
        if (
            output["species"] != record.row["species"]
            or str(output["camera_id"]) != record.row["camera_id"]
        ):
            camera_changes += 1
        if (
            str(output["crop_width"]) != record.row["crop_width"]
            or str(output["crop_height"]) != record.row["crop_height"]
        ):
            crop_size_changes += 1
        applied = boolean_text(output["timestamp_mask_applied"])
        overlap = boolean_text(output["timestamp_candidate_overlap"])
        if applied != record.geometry.overlaps or overlap != applied:
            overlap_flag_mismatches += 1
        if record.geometry.mask_rect_crop is not None:
            left, top, right, bottom = record.geometry.mask_rect_crop
            if (
                left < 0
                or top < 0
                or right > record.geometry.crop_width
                or bottom > record.geometry.crop_height
                or left >= right
                or top >= bottom
            ):
                mask_bounds_errors += 1
    expected_paths = {
        record.row["output_relative_path"] for record in records
    }
    actual_paths = image_set(output_root)
    missing_images = expected_paths - actual_paths
    unexpected_images = actual_paths - expected_paths
    corrupt_images = 0
    image_size_changes = 0
    non_jpeg_images = 0
    non_rgb_images = 0
    progressive_images = 0
    sampling_errors = 0
    qtable_signatures: set[tuple[Any, ...]] = set()
    for record in records:
        path = output_root / record.row["output_relative_path"]
        if not path.is_file():
            continue
        try:
            with Image.open(path) as opened:
                opened.load()
                if opened.format != "JPEG":
                    non_jpeg_images += 1
                if opened.mode != "RGB":
                    non_rgb_images += 1
                if opened.size != (
                    record.geometry.crop_width,
                    record.geometry.crop_height,
                ):
                    image_size_changes += 1
                if opened.info.get("progressive") or opened.info.get(
                    "progression"
                ):
                    progressive_images += 1
                if JpegImagePlugin.get_sampling(opened) != 0:
                    sampling_errors += 1
                qtable_signatures.add(
                    jpeg_quantization_signature(opened)
                )
        except (UnidentifiedImageError, OSError):
            corrupt_images += 1
    source_keys = [record.unique_key for record in records]
    output_paths = [
        str(row["output_relative_path"]) for row in manifest_rows
    ]
    train_groups = {
        (record.row["species"], record.row["camera_id"])
        for record in records
        if record.row["split"] == "train"
    }
    val_groups = {
        (record.row["species"], record.row["camera_id"])
        for record in records
        if record.row["split"] == "validation"
    }
    split_counts = Counter(record.row["split"] for record in records)
    class_counts = Counter(
        (
            record.row["split"],
            record.row["species"],
            record.row["health_class_name"],
        )
        for record in records
    )
    output_class_counts = Counter(
        (
            str(row["split"]),
            str(row["species"]),
            str(row["health_class_name"]),
        )
        for row in manifest_rows
    )
    masked_count = sum(record.geometry.overlaps for record in records)
    applied_count = sum(
        boolean_text(row["timestamp_mask_applied"])
        for row in manifest_rows
    )
    review_pairs = sum(
        int(sheet["pair_count"]) for sheet in review_sheets
    )
    validation = {
        "total_images": len(records),
        "train_images": split_counts["train"],
        "validation_images": split_counts["validation"],
        "test_images": split_counts["test"],
        "masked_images": masked_count,
        "mask_applied_images": applied_count,
        "preserved_manifest_mismatches": preserved_mismatches,
        "identity_mismatches": identity_mismatches,
        "split_changes": split_changes,
        "class_changes": class_changes,
        "camera_changes": camera_changes,
        "crop_size_changes": crop_size_changes,
        "image_size_changes": image_size_changes,
        "mask_bounds_errors": mask_bounds_errors,
        "overlap_flag_mismatches": overlap_flag_mismatches,
        "missing_images": len(missing_images),
        "unexpected_images": len(unexpected_images),
        "corrupt_images": corrupt_images,
        "non_jpeg_images": non_jpeg_images,
        "non_rgb_images": non_rgb_images,
        "progressive_images": progressive_images,
        "sampling_errors": sampling_errors,
        "jpeg_qtable_signatures": len(qtable_signatures),
        "duplicate_source_images": (
            len(source_keys) - len(set(source_keys))
        ),
        "duplicate_output_paths": (
            len(output_paths) - len(set(output_paths))
        ),
        "camera_overlap": len(train_groups & val_groups),
        "class_distribution_mismatches": (
            0 if class_counts == output_class_counts else 1
        ),
        "review_pairs": review_pairs,
        "review_contact_sheets": len(review_sheets),
    }
    zero_required = (
        "test_images",
        "preserved_manifest_mismatches",
        "identity_mismatches",
        "split_changes",
        "class_changes",
        "camera_changes",
        "crop_size_changes",
        "image_size_changes",
        "mask_bounds_errors",
        "overlap_flag_mismatches",
        "missing_images",
        "unexpected_images",
        "corrupt_images",
        "non_jpeg_images",
        "non_rgb_images",
        "progressive_images",
        "sampling_errors",
        "duplicate_source_images",
        "duplicate_output_paths",
        "camera_overlap",
        "class_distribution_mismatches",
    )
    failures = {
        name: validation[name]
        for name in zero_required
        if validation[name]
    }
    if masked_count != applied_count:
        failures["mask_applied_images"] = (
            applied_count,
            masked_count,
        )
    if validation["jpeg_qtable_signatures"] != 1:
        failures["jpeg_qtable_signatures"] = validation[
            "jpeg_qtable_signatures"
        ]
    if review_pairs > max_review_pairs:
        failures["review_pairs"] = review_pairs
    if failures:
        raise RuntimeError(f"masked dataset 검증 실패: {failures}")
    return validation


def class_species_mask_rows(
    records: Sequence[SourceRecord],
    dimension: str,
) -> list[tuple[Any, ...]]:
    if dimension == "class":
        groups = Counter(
            record.row["health_class_name"] for record in records
        )
        masks = Counter(
            record.row["health_class_name"]
            for record in records
            if record.geometry.overlaps
        )
    elif dimension == "species":
        groups = Counter(record.row["species"] for record in records)
        masks = Counter(
            record.row["species"]
            for record in records
            if record.geometry.overlaps
        )
    else:
        raise ValueError("지원하지 않는 dimension")
    return [
        (
            value,
            groups[value],
            masks[value],
            f"{masks[value] / groups[value]:.2%}",
        )
        for value in sorted(
            groups,
            key=lambda item: (
                smoke.CLASS_NAMES_INV.get(item, 999),
                item,
            ),
        )
    ]


def write_class_mapping(
    source_dir: Path,
    output_root: Path,
) -> None:
    source = source_dir / "class_mapping.json"
    if not source.is_file():
        raise FileNotFoundError("raw class_mapping.json이 없습니다")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("classes", {}).get("0", {}).get("name") != "HEALTHY":
        raise ValueError("raw class 0 mapping이 다릅니다")
    if (
        payload.get("classes", {}).get("1", {}).get("name")
        != "DISEASE_SUSPECTED"
    ):
        raise ValueError("raw class 1 mapping이 다릅니다")
    smoke.write_text(
        output_root / "class_mapping.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


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


def write_summary(
    output_root: Path,
    records: Sequence[SourceRecord],
    validation: Mapping[str, Any],
    *,
    mask_color: int,
    jpeg_quality: int,
    max_review_pairs: int,
    seed: int,
) -> None:
    split_totals = Counter(record.row["split"] for record in records)
    split_masks = Counter(
        record.row["split"]
        for record in records
        if record.geometry.overlaps
    )
    lines = [
        "# 버섯 건강 체크 timestamp-masked 파일럿",
        "",
        "## 처리 범위",
        "",
        f"- source manifest 행: `{len(records):,}`",
        "- 새로운 샘플링 및 split 변경 없음",
        "- Test 선택·사용: 0장",
        "- 원본 기준 timestamp 후보 ROI: "
        "`x=[floor(0.60W), W), y=[0, ceil(0.12H))`",
        f"- 마스크 색: `RGB({mask_color},{mask_color},{mask_color})`",
        f"- JPEG: quality `{jpeg_quality}`, subsampling `0`, "
        "optimize `False`, progressive `False`",
        "- overlap 여부와 무관하게 모든 6,000장을 동일 경로로 재인코딩",
        "- OCR, 모델 로드, 모델 학습 없음",
        "",
        "## 결과",
        "",
        f"- 전체/Train/Validation: **{validation['total_images']:,}/"
        f"{validation['train_images']:,}/"
        f"{validation['validation_images']:,}장**",
        f"- 마스킹: **{validation['masked_images']:,}/"
        f"{validation['total_images']:,}장 "
        f"({validation['masked_images'] / validation['total_images']:.2%})**",
        f"- raw manifest 불일치: "
        f"**{validation['preserved_manifest_mismatches']:,}건**",
        f"- crop 크기 변경: **{validation['crop_size_changes']:,}건**",
        f"- 누락/손상: **{validation['missing_images']:,}/"
        f"{validation['corrupt_images']:,}장**",
        f"- Train/Validation camera group 중복: "
        f"**{validation['camera_overlap']:,}개**",
        f"- raw health_pilot 및 기존 artifacts 변경: **없음**",
        "",
        "## split별 마스킹",
        "",
        markdown_table(
            ("split", "전체", "마스킹", "비율"),
            (
                (
                    split,
                    split_totals[split],
                    split_masks[split],
                    f"{split_masks[split] / split_totals[split]:.2%}",
                )
                for split in ("train", "validation")
            ),
        ),
        "",
        "## 클래스별 마스킹",
        "",
        markdown_table(
            ("클래스", "전체", "마스킹", "비율"),
            class_species_mask_rows(records, "class"),
        ),
        "",
        "## 품종별 마스킹",
        "",
        markdown_table(
            ("품종", "전체", "마스킹", "비율"),
            class_species_mask_rows(records, "species"),
        ),
        "",
        "## 검증",
        "",
        f"- source identity 불일치: "
        f"{validation['identity_mismatches']:,}건",
        f"- split/class/camera 변경: "
        f"{validation['split_changes']:,}/"
        f"{validation['class_changes']:,}/"
        f"{validation['camera_changes']:,}건",
        f"- 이미지 크기 변경: "
        f"{validation['image_size_changes']:,}건",
        f"- mask 범위 오류: "
        f"{validation['mask_bounds_errors']:,}건",
        f"- raw overlap flag 불일치: "
        f"{validation['overlap_flag_mismatches']:,}건",
        f"- JPEG/RGB/subsampling/progressive 오류: "
        f"{validation['non_jpeg_images']:,}/"
        f"{validation['non_rgb_images']:,}/"
        f"{validation['sampling_errors']:,}/"
        f"{validation['progressive_images']:,}건",
        f"- JPEG quantization signature 종류: "
        f"{validation['jpeg_qtable_signatures']:,}개",
        f"- 중복 source/output: "
        f"{validation['duplicate_source_images']:,}/"
        f"{validation['duplicate_output_paths']:,}건",
        f"- review pair/contact sheet: "
        f"{validation['review_pairs']:,}/"
        f"{validation['review_contact_sheets']:,}개 "
        f"(상한 {max_review_pairs})",
        "- 로컬 절대경로 노출: 0건",
        "",
        "## 해석 주의",
        "",
        "이 데이터셋은 timestamp 후보 영역 마스킹 ablation용이다. "
        "실제 timestamp 존재를 OCR로 확인한 결과가 아니며, "
        "배경·촬영 날짜·task 수집 편향은 여전히 남아 있다.",
        f"- 처리 버전: `{PROCESSING_VERSION}`",
        f"- seed: `{seed}`",
        "",
    ]
    smoke.write_text(
        output_root / "health_timestamp_masked_summary.md",
        "\n".join(lines),
    )


def write_manifest(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    smoke.write_csv(path, MASKED_MANIFEST_COLUMNS, rows)


def validate_serialized_manifest(
    path: Path,
    records: Sequence[SourceRecord],
) -> int:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(records):
        return abs(len(rows) - len(records)) or 1
    mismatches = 0
    for source, output in zip(records, rows):
        for column in raw_health.MANIFEST_COLUMNS:
            if source.row[column] != output[column]:
                mismatches += 1
                break
    return mismatches


def prepare_output_transaction(
    output_dir: Path,
    *,
    overwrite: bool,
) -> tuple[Path, bool]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_existed = output_dir.exists()
    if output_existed and not output_dir.is_dir():
        raise ValueError("output-dir 경로에 디렉터리가 아닌 파일이 있습니다")
    if output_existed and not overwrite:
        raise FileExistsError(
            "출력 디렉터리가 이미 존재합니다: "
            f"{base.portable_output_path(output_dir)}"
        )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    return staging, output_existed


def activate_output_transaction(
    staging: Path,
    output_dir: Path,
    *,
    output_existed: bool,
) -> Path | None:
    backup: Path | None = None
    if output_existed:
        backup = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.backup-",
                dir=output_dir.parent,
            )
        )
        backup.rmdir()
        os.replace(output_dir, backup)
    try:
        os.replace(staging, output_dir)
    except Exception:
        if backup is not None and backup.exists() and not output_dir.exists():
            os.replace(backup, output_dir)
        raise
    return backup


def rollback_output_transaction(
    staging: Path,
    output_dir: Path,
    backup: Path | None,
    *,
    activated: bool,
    output_existed: bool,
) -> None:
    if staging.exists():
        shutil.rmtree(staging)
    if not activated:
        return
    if output_dir.exists():
        shutil.rmtree(output_dir)
    if backup is not None and backup.exists():
        os.replace(backup, output_dir)
    elif output_existed:
        raise RuntimeError("기존 masked output backup을 복원할 수 없습니다")


def create_masked_dataset(
    records: Sequence[SourceRecord],
    source_dir: Path,
    source_manifest: Path,
    output_dir: Path,
    *,
    mask_color: int,
    jpeg_quality: int,
    max_review_pairs: int,
    seed: int,
    overwrite: bool,
    artifacts_root: Path = ARTIFACTS_ROOT,
    show_progress: bool = True,
    expected_images: int = EXPECTED_IMAGES,
) -> dict[str, Any]:
    if len(records) != expected_images:
        raise ValueError(
            f"masked 생성 입력 {len(records)}행(기대 {expected_images}행)"
        )
    validate_locations(
        source_dir,
        source_manifest,
        output_dir,
        artifacts_root=artifacts_root,
    )
    source_manifest_snapshot = health.snapshot_files((source_manifest,))
    artifacts_snapshot = raw_health.snapshot_artifacts_excluding(
        artifacts_root, output_dir
    )
    selected_review = select_review_records(
        records,
        max_review_pairs=max_review_pairs,
        seed=seed,
    )
    staging, output_existed = prepare_output_transaction(
        output_dir,
        overwrite=overwrite,
    )
    backup: Path | None = None
    activated = False
    try:
        for relative in (
            "train/0_healthy",
            "train/1_disease_suspected",
            "val/0_healthy",
            "val/1_disease_suspected",
            "review/contact_sheets",
        ):
            (staging / relative).mkdir(parents=True, exist_ok=True)
        write_class_mapping(source_dir, staging)
        manifest_rows = process_images(
            records,
            staging,
            mask_color=mask_color,
            jpeg_quality=jpeg_quality,
            show_progress=show_progress,
        )
        review_sheets = create_review_contact_sheets(
            source_dir,
            staging,
            selected_review,
        )
        validation = validate_output(
            records,
            manifest_rows,
            staging,
            review_sheets=review_sheets,
            max_review_pairs=max_review_pairs,
        )
        manifest_path = (
            staging / "health_timestamp_masked_manifest.csv"
        )
        write_manifest(manifest_path, manifest_rows)
        serialized_mismatches = validate_serialized_manifest(
            manifest_path, records
        )
        if serialized_mismatches:
            raise RuntimeError(
                "직렬화된 masked manifest와 raw manifest 불일치: "
                f"{serialized_mismatches}건"
            )
        write_summary(
            staging,
            records,
            validation,
            mask_color=mask_color,
            jpeg_quality=jpeg_quality,
            max_review_pairs=max_review_pairs,
            seed=seed,
        )
        raw_health.assert_deidentified_output(staging)
        health.assert_snapshot_unchanged(source_manifest_snapshot)
        raw_health.assert_artifacts_unchanged(
            artifacts_root, output_dir, artifacts_snapshot
        )
        backup = activate_output_transaction(
            staging,
            output_dir,
            output_existed=output_existed,
        )
        activated = True
        raw_health.assert_artifacts_unchanged(
            artifacts_root, output_dir, artifacts_snapshot
        )
    except Exception:
        rollback_output_transaction(
            staging,
            output_dir,
            backup,
            activated=activated,
            output_existed=output_existed,
        )
        health.assert_snapshot_unchanged(source_manifest_snapshot)
        raw_health.assert_artifacts_unchanged(
            artifacts_root, output_dir, artifacts_snapshot
        )
        raise
    if backup is not None:
        # At this point the new output passed every post-activation check.
        # A cleanup failure must not replace it with a partially deleted backup.
        shutil.rmtree(backup)
    class_masks = Counter(
        record.row["health_class_name"]
        for record in records
        if record.geometry.overlaps
    )
    class_totals = Counter(
        record.row["health_class_name"] for record in records
    )
    species_masks = Counter(
        record.row["species"]
        for record in records
        if record.geometry.overlaps
    )
    species_totals = Counter(record.row["species"] for record in records)
    return {
        "total_images": validation["total_images"],
        "train_images": validation["train_images"],
        "validation_images": validation["validation_images"],
        "masked_images": validation["masked_images"],
        "masked_rate": (
            validation["masked_images"] / validation["total_images"]
        ),
        "class_mask_distribution": [
            {
                "class_name": class_name,
                "total": class_totals[class_name],
                "masked": class_masks[class_name],
                "rate": class_masks[class_name] / class_totals[class_name],
            }
            for class_name in sorted(class_totals)
        ],
        "species_mask_distribution": [
            {
                "species": species,
                "total": species_totals[species],
                "masked": species_masks[species],
                "rate": species_masks[species] / species_totals[species],
            }
            for species in raw_health.pilot.SPECIES
        ],
        "raw_manifest_mismatches": validation[
            "preserved_manifest_mismatches"
        ],
        "crop_size_changes": validation["crop_size_changes"]
        + validation["image_size_changes"],
        "test_images": validation["test_images"],
        "missing_images": validation["missing_images"],
        "corrupt_images": validation["corrupt_images"],
        "review_pairs": validation["review_pairs"],
        "review_contact_sheets": validation[
            "review_contact_sheets"
        ],
        "source_raw_modified": False,
        "protected_artifacts_modified": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest_snapshot = health.snapshot_files(
        (args.source_manifest,)
    )
    try:
        records = load_source_records(
            args.source_manifest,
            args.source_dir,
            expected_images=EXPECTED_IMAGES,
        )
        result = create_masked_dataset(
            records,
            args.source_dir,
            args.source_manifest,
            args.output_dir,
            mask_color=args.mask_color,
            jpeg_quality=args.jpeg_quality,
            max_review_pairs=args.max_review_pairs,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    finally:
        health.assert_snapshot_unchanged(source_manifest_snapshot)
    result.update(
        {
            "output_dir": base.portable_output_path(args.output_dir),
            "processing_version": PROCESSING_VERSION,
            "new_sampling": False,
            "split_changed": False,
            "model_loaded": False,
            "model_training": False,
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
