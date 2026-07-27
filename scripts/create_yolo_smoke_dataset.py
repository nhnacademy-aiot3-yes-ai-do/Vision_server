#!/usr/bin/env python3
"""Create a bounded YOLO smoke dataset from the final detection manifest.

The manifest is streamed.  At most 200 selected image members are read from
read-only source ZIPs.  Test records are never selected, and the script never
extracts an archive wholesale or performs model training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw

import audit_mushroom_labels as audit
import build_detection_manifest as detection
import inspect_aihub_archives as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "detection_dataset_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "yolo_smoke"
DEFAULT_SPLITS = ("train", "validation")
DEFAULT_SEED = 20260726
DEFAULT_SAMPLES_PER_GROUP = 10
DEFAULT_MAX_TOTAL_IMAGES = 200
HARD_MAX_TOTAL_IMAGES = 200
ALLOWED_SPLITS = frozenset(DEFAULT_SPLITS)
TASKS = frozenset(("생육", "병해"))
CLASS_NAMES = {
    0: "느타리",
    1: "양송이",
    2: "큰느타리",
    3: "팽이",
    4: "표고",
}
REQUIRED_MANIFEST_COLUMNS = frozenset(
    (
        "split",
        "species",
        "class_id",
        "task",
        "normality",
        "disease_type",
        "image_archive_id",
        "image_member",
        "image_width",
        "image_height",
        "valid_bbox_count",
        "bbox_cleaned",
    )
)
SMOKE_MANIFEST_COLUMNS = (
    "split",
    "yolo_split",
    "species",
    "class_id",
    "task",
    "normality",
    "disease_type",
    "image_archive_id",
    "image_member",
    "image_width",
    "image_height",
    "bbox_count",
    "image_path",
    "label_path",
    "overlay_path",
)
FORBIDDEN_PATH_FRAGMENTS = ("/mnt/d", "/home/kim75", "\\mnt\\d", "\\home\\kim75")


@dataclass(frozen=True)
class Candidate:
    score: str
    unique_key: str
    row: dict[str, str]


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def line(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.15f} {self.y_center:.15f} "
            f"{self.width:.15f} {self.height:.15f}"
        )


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("양의 정수여야 합니다") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("양의 정수여야 합니다")
    return number


def bounded_max_images(value: str) -> int:
    number = positive_int(value)
    if number > HARD_MAX_TOTAL_IMAGES:
        raise argparse.ArgumentTypeError(
            f"안전 제한상 {HARD_MAX_TOTAL_IMAGES} 이하여야 합니다"
        )
    return number


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "최종 manifest에서 Train/Validation 이미지만 최대 200장 선택해 "
            "YOLO smoke dataset과 bbox overlay를 생성합니다."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/detection_dataset_manifest.csv"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="허용값: train validation (test는 항상 거부)",
    )
    parser.add_argument(
        "--samples-per-species-task",
        type=positive_int,
        default=DEFAULT_SAMPLES_PER_GROUP,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/yolo_smoke"),
    )
    parser.add_argument(
        "--max-total-images",
        type=bounded_max_images,
        default=DEFAULT_MAX_TOTAL_IMAGES,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 smoke 출력 디렉터리를 안전하게 교체",
    )
    args = parser.parse_args(argv)
    args.splits = normalize_requested_splits(args.splits)
    args.manifest = project_path(args.manifest)
    args.output_dir = project_path(args.output_dir)
    artifacts_root = (PROJECT_ROOT / "artifacts").resolve()
    if not args.output_dir.is_relative_to(artifacts_root):
        parser.error("--output-dir은 프로젝트의 artifacts 아래여야 합니다")
    return args


def normalize_requested_splits(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(value.strip().lower() for value in values))
    if not normalized:
        raise ValueError("split을 하나 이상 지정해야 합니다")
    if "test" in normalized:
        raise ValueError("Test split은 smoke dataset에 사용할 수 없습니다")
    unsupported = sorted(set(normalized) - ALLOWED_SPLITS)
    if unsupported:
        raise ValueError("지원하지 않는 split: " + ", ".join(unsupported))
    return tuple(split for split in DEFAULT_SPLITS if split in normalized)


def stable_score(seed: int, *parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(f"{seed}\x1e{payload}".encode("utf-8")).hexdigest()


def candidate_group(row: Mapping[str, str]) -> tuple[str, str, str]:
    return row["split"], row["species"], row["task"]


def candidate_unique_key(row: Mapping[str, str]) -> str:
    return (
        f"{row['image_archive_id']}:"
        f"{base.normalize_member_name(row['image_member']).casefold()}"
    )


def select_manifest_rows(
    manifest_path: Path,
    *,
    splits: Sequence[str],
    samples_per_species_task: int,
    seed: int,
    max_total_images: int,
) -> list[dict[str, str]]:
    """Stream CSV and retain only the best N hashes per split/species/task."""
    requested = normalize_requested_splits(splits)
    if max_total_images > HARD_MAX_TOTAL_IMAGES:
        raise ValueError(
            f"max_total_images는 {HARD_MAX_TOTAL_IMAGES} 이하여야 합니다"
        )
    if samples_per_species_task <= 0 or max_total_images <= 0:
        raise ValueError("샘플 수 제한은 양수여야 합니다")
    buckets: dict[tuple[str, str, str], list[Candidate]] = defaultdict(list)
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "manifest 필수 컬럼 누락: " + ", ".join(sorted(missing))
            )
        for row in reader:
            if row["split"] not in requested:
                continue
            if row["task"] not in TASKS:
                continue
            class_id = parse_class_id(row["class_id"])
            if CLASS_NAMES[class_id] != row["species"]:
                raise ValueError(
                    f"class_id/품종 불일치: {row['class_id']} {row['species']}"
                )
            group = candidate_group(row)
            unique_key = candidate_unique_key(row)
            score = stable_score(seed, unique_key)
            candidate = Candidate(score, unique_key, dict(row))
            bucket = buckets[group]
            bucket.append(candidate)
            bucket.sort(key=lambda item: (item.score, item.unique_key))
            if len(bucket) > samples_per_species_task:
                bucket.pop()

    ordered_groups = sorted(
        buckets,
        key=lambda group: (
            CLASS_NAMES_INV.get(group[1], 999),
            0 if group[2] == "생육" else 1,
            0 if group[0] == "train" else 1,
        ),
    )
    selected: list[dict[str, str]] = []
    selected_keys: set[str] = set()
    depth = 0
    while len(selected) < max_total_images:
        added = False
        for group in ordered_groups:
            bucket = buckets[group]
            if depth >= len(bucket):
                continue
            candidate = bucket[depth]
            if candidate.unique_key in selected_keys:
                continue
            selected.append(candidate.row)
            selected_keys.add(candidate.unique_key)
            added = True
            if len(selected) == max_total_images:
                break
        if not added:
            break
        depth += 1
    selected.sort(
        key=lambda row: (
            0 if row["split"] == "train" else 1,
            int(row["class_id"]),
            0 if row["task"] == "생육" else 1,
            stable_score(seed, candidate_unique_key(row)),
        )
    )
    if len(selected) > min(max_total_images, HARD_MAX_TOTAL_IMAGES):
        raise RuntimeError("smoke 이미지 안전 상한을 초과했습니다")
    if any(row["split"] == "test" for row in selected):
        raise RuntimeError("Test 레코드가 선택되었습니다")
    return selected


CLASS_NAMES_INV = {name: class_id for class_id, name in CLASS_NAMES.items()}


def parse_class_id(value: Any) -> int:
    try:
        class_id = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"유효하지 않은 class_id: {value!r}") from exc
    if class_id not in CLASS_NAMES:
        raise ValueError(f"class_id 범위 오류: {class_id}")
    return class_id


def positive_dimension(value: Any, field_name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}가 정수가 아닙니다: {value!r}") from exc
    if result <= 0:
        raise ValueError(f"{field_name}가 양수가 아닙니다: {result}")
    return result


def convert_bbox_to_yolo(
    bbox: Mapping[str, Any],
    *,
    class_id: int,
    image_width: int,
    image_height: int,
) -> YoloBox:
    parse_class_id(class_id)
    width_px = float(bbox["width"])
    height_px = float(bbox["height"])
    x_px = float(bbox["x"])
    y_px = float(bbox["y"])
    if width_px <= 0 or height_px <= 0:
        raise ValueError("bbox width와 height는 양수여야 합니다")
    values = YoloBox(
        class_id=class_id,
        x_center=(x_px + width_px / 2) / image_width,
        y_center=(y_px + height_px / 2) / image_height,
        width=width_px / image_width,
        height=height_px / image_height,
    )
    validate_yolo_box(values)
    return values


def validate_yolo_box(box: YoloBox) -> None:
    if box.class_id not in CLASS_NAMES:
        raise ValueError(f"class_id 범위 오류: {box.class_id}")
    coordinates = (box.x_center, box.y_center, box.width, box.height)
    if not all(0.0 <= value <= 1.0 for value in coordinates):
        raise ValueError(f"YOLO 정규화 좌표 범위 오류: {coordinates}")
    if box.width <= 0 or box.height <= 0:
        raise ValueError("YOLO width와 height는 0보다 커야 합니다")
    if (
        box.x_center - box.width / 2 < -1e-12
        or box.y_center - box.height / 2 < -1e-12
        or box.x_center + box.width / 2 > 1 + 1e-12
        or box.y_center + box.height / 2 > 1 + 1e-12
    ):
        raise ValueError("YOLO bbox 가장자리가 이미지 범위를 벗어납니다")


def yolo_boxes_for_row(row: Mapping[str, str]) -> list[YoloBox]:
    class_id = parse_class_id(row["class_id"])
    image_width = positive_dimension(row["image_width"], "image_width")
    image_height = positive_dimension(row["image_height"], "image_height")
    try:
        cleaned = json.loads(row["bbox_cleaned"])
    except json.JSONDecodeError as exc:
        raise ValueError("bbox_cleaned JSON 파싱 실패") from exc
    if not isinstance(cleaned, list) or not cleaned:
        raise ValueError("선택된 이미지에 유효 bbox가 없습니다")
    boxes = [
        convert_bbox_to_yolo(
            bbox,
            class_id=class_id,
            image_width=image_width,
            image_height=image_height,
        )
        for bbox in cleaned
    ]
    if len(boxes) != int(row["valid_bbox_count"]):
        raise ValueError("valid_bbox_count와 bbox_cleaned 개수가 다릅니다")
    return boxes


def yolo_split(split: str) -> str:
    if split == "train":
        return "train"
    if split == "validation":
        return "val"
    raise ValueError(f"Test 또는 미지원 split: {split}")


def portable_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def output_basename(row: Mapping[str, str]) -> str:
    member = base.normalize_member_name(row["image_member"])
    suffix = Path(member).suffix.lower()
    if suffix not in base.IMAGE_EXTENSIONS:
        raise ValueError(f"지원하지 않는 이미지 확장자: {suffix}")
    digest = hashlib.sha256(
        f"{row['image_archive_id']}\x1f{member.casefold()}".encode("utf-8")
    ).hexdigest()[:20]
    return f"{row['image_archive_id'].lower()}_{digest}{suffix}"


def prepare_staging(output_dir: Path, *, overwrite: bool) -> tuple[Path, Path | None]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"출력 디렉터리가 이미 존재합니다: {base.portable_output_path(output_dir)}"
        )
    staging = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    backup = output_dir.with_name(f".{output_dir.name}.backup-{os.getpid()}")
    for path in (staging, backup):
        if path.exists():
            shutil.rmtree(path)
    if output_dir.exists():
        os.replace(output_dir, backup)
    staging.mkdir(parents=True)
    return staging, backup if backup.exists() else None


def finalize_staging(
    staging: Path,
    output_dir: Path,
    backup: Path | None,
) -> None:
    try:
        os.replace(staging, output_dir)
    except Exception:
        if backup and backup.exists() and not output_dir.exists():
            os.replace(backup, output_dir)
        raise
    if backup and backup.exists():
        shutil.rmtree(backup)


def rollback_staging(
    staging: Path,
    output_dir: Path,
    backup: Path | None,
) -> None:
    if staging.exists():
        shutil.rmtree(staging)
    if backup and backup.exists() and not output_dir.exists():
        os.replace(backup, output_dir)


def image_archive_lookup(
    refs: Sequence[base.ArchiveRef],
) -> dict[str, base.ArchiveRef]:
    lookup: dict[str, base.ArchiveRef] = {}
    for ref in refs:
        if ref.kind != "image" or ref.split not in ALLOWED_SPLITS:
            continue
        if ref.archive_id in lookup:
            raise ValueError(f"이미지 archive ID 중복: {ref.archive_id}")
        lookup[ref.archive_id] = ref
    return lookup


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_static_metadata(output_root: Path) -> None:
    data_yaml = "\n".join(
        (
            "train: images/train",
            "val: images/val",
            "nc: 5",
            "names:",
            "  0: 느타리",
            "  1: 양송이",
            "  2: 큰느타리",
            "  3: 팽이",
            "  4: 표고",
            "",
        )
    )
    write_text(output_root / "data.yaml", data_yaml)
    write_text(
        output_root / "classes.txt",
        "\n".join(CLASS_NAMES[index] for index in sorted(CLASS_NAMES)) + "\n",
    )


def draw_overlay(
    image_bytes: bytes,
    boxes: Sequence[YoloBox],
    *,
    expected_width: int,
    expected_height: int,
    destination: Path,
) -> None:
    with Image.open(io.BytesIO(image_bytes)) as source:
        source.load()
        if source.size != (expected_width, expected_height):
            raise ValueError(
                "manifest/이미지 크기 불일치: "
                f"manifest={expected_width}x{expected_height}, image={source.size}"
            )
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(expected_width, expected_height) / 300))
    for box in boxes:
        x1 = (box.x_center - box.width / 2) * expected_width
        y1 = (box.y_center - box.height / 2) * expected_height
        x2 = (box.x_center + box.width / 2) * expected_width
        y2 = (box.y_center + box.height / 2) * expected_height
        draw.rectangle(
            (x1, y1, x2, y2),
            outline=(255, 48, 48),
            width=line_width,
        )
        draw.text(
            (max(0, x1 + line_width), max(0, y1 + line_width)),
            str(box.class_id),
            fill=(255, 255, 0),
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="JPEG", quality=92)


def extract_selected_images(
    selected: Sequence[Mapping[str, str]],
    archive_lookup: Mapping[str, base.ArchiveRef],
    output_root: Path,
) -> list[dict[str, Any]]:
    """Read exactly the selected members and create images/labels/overlays."""
    by_archive: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in selected:
        if row["split"] == "test":
            raise RuntimeError("Test 이미지는 추출할 수 없습니다")
        by_archive[row["image_archive_id"]].append(row)
    manifest_rows: list[dict[str, Any]] = []
    for archive_id in sorted(by_archive):
        ref = archive_lookup.get(archive_id)
        if ref is None:
            raise ValueError(f"이미지 ZIP을 찾을 수 없음: {archive_id}")
        with zipfile.ZipFile(ref.path, "r") as archive:
            available = {
                base.normalize_member_name(info.filename): info
                for info in archive.infolist()
                if not info.is_dir()
            }
            for row in by_archive[archive_id]:
                member = base.normalize_member_name(row["image_member"])
                info = available.get(member)
                if info is None:
                    raise FileNotFoundError(
                        f"{archive_id} 내부 이미지 누락: {member}"
                    )
                boxes = yolo_boxes_for_row(row)
                image_bytes = archive.read(info)
                split_dir = yolo_split(row["split"])
                basename = output_basename(row)
                image_path = output_root / "images" / split_dir / basename
                label_path = (
                    output_root
                    / "labels"
                    / split_dir
                    / f"{Path(basename).stem}.txt"
                )
                overlay_path = (
                    output_root
                    / "overlays"
                    / split_dir
                    / f"{Path(basename).stem}.jpg"
                )
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(image_bytes)
                write_text(
                    label_path,
                    "\n".join(box.line() for box in boxes) + "\n",
                )
                draw_overlay(
                    image_bytes,
                    boxes,
                    expected_width=positive_dimension(
                        row["image_width"], "image_width"
                    ),
                    expected_height=positive_dimension(
                        row["image_height"], "image_height"
                    ),
                    destination=overlay_path,
                )
                manifest_rows.append(
                    {
                        "split": row["split"],
                        "yolo_split": split_dir,
                        "species": row["species"],
                        "class_id": row["class_id"],
                        "task": row["task"],
                        "normality": row["normality"],
                        "disease_type": row["disease_type"],
                        "image_archive_id": archive_id,
                        "image_member": member,
                        "image_width": row["image_width"],
                        "image_height": row["image_height"],
                        "bbox_count": len(boxes),
                        "image_path": portable_relative(
                            image_path, output_root
                        ),
                        "label_path": portable_relative(
                            label_path, output_root
                        ),
                        "overlay_path": portable_relative(
                            overlay_path, output_root
                        ),
                    }
                )
    manifest_rows.sort(
        key=lambda row: (
            0 if row["split"] == "train" else 1,
            int(row["class_id"]),
            row["task"],
            row["image_member"],
        )
    )
    return manifest_rows


def parse_label_file(path: Path) -> tuple[int, int]:
    bbox_count = 0
    errors = 0
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return 0, 1
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            errors += 1
            continue
        try:
            box = YoloBox(
                class_id=int(parts[0]),
                x_center=float(parts[1]),
                y_center=float(parts[2]),
                width=float(parts[3]),
                height=float(parts[4]),
            )
            validate_yolo_box(box)
        except (ValueError, TypeError):
            errors += 1
        else:
            bbox_count += 1
    return bbox_count, errors


def text_path_leaks(output_root: Path) -> list[str]:
    leaks: list[str] = []
    text_suffixes = {".txt", ".csv", ".yaml", ".md"}
    for path in output_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if any(fragment in text for fragment in FORBIDDEN_PATH_FRAGMENTS):
            leaks.append(portable_relative(path, output_root))
    return sorted(leaks)


def validate_output(
    output_root: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_images = {
        str(row["image_path"]) for row in manifest_rows
    }
    expected_labels = {
        str(row["label_path"]) for row in manifest_rows
    }
    actual_images = {
        portable_relative(path, output_root)
        for split in ("train", "val")
        for path in (output_root / "images" / split).glob("*")
        if path.is_file() and path.suffix.lower() in base.IMAGE_EXTENSIONS
    }
    actual_labels = {
        portable_relative(path, output_root)
        for split in ("train", "val")
        for path in (output_root / "labels" / split).glob("*.txt")
        if path.is_file()
    }
    missing_images = sorted(expected_images - actual_images)
    missing_labels = sorted(expected_labels - actual_labels)
    unexpected_images = sorted(actual_images - expected_images)
    unexpected_labels = sorted(actual_labels - expected_labels)
    coordinate_errors = 0
    bbox_total = 0
    empty_labels = 0
    for relative in sorted(expected_labels & actual_labels):
        count, errors = parse_label_file(output_root / relative)
        bbox_total += count
        coordinate_errors += errors
        if count == 0:
            empty_labels += 1
    train_sources = {
        (str(row["image_archive_id"]), str(row["image_member"]).casefold())
        for row in manifest_rows
        if row["split"] == "train"
    }
    val_sources = {
        (str(row["image_archive_id"]), str(row["image_member"]).casefold())
        for row in manifest_rows
        if row["split"] == "validation"
    }
    test_count = sum(row["split"] == "test" for row in manifest_rows)
    duplicate_output_images = len(manifest_rows) - len(expected_images)
    result = {
        "image_count": len(actual_images),
        "label_count": len(actual_labels),
        "bbox_count": bbox_total,
        "missing_images": len(missing_images),
        "missing_labels": len(missing_labels),
        "unexpected_images": len(unexpected_images),
        "unexpected_labels": len(unexpected_labels),
        "empty_labels": empty_labels,
        "coordinate_errors": coordinate_errors,
        "train_val_source_overlap": len(train_sources & val_sources),
        "duplicate_output_images": duplicate_output_images,
        "test_images": test_count,
        "path_leaks": text_path_leaks(output_root),
    }
    failure_keys = (
        "missing_images",
        "missing_labels",
        "unexpected_images",
        "unexpected_labels",
        "empty_labels",
        "coordinate_errors",
        "train_val_source_overlap",
        "duplicate_output_images",
        "test_images",
    )
    if len(actual_images) != len(actual_labels):
        raise RuntimeError(f"이미지/라벨 수 불일치: {result}")
    if any(result[key] for key in failure_keys) or result["path_leaks"]:
        raise RuntimeError(f"smoke dataset 검증 실패: {result}")
    return result


def distribution_rows(
    manifest_rows: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, str, int]]:
    counts: Counter[tuple[str, str, str]] = Counter(
        (
            str(row["split"]),
            str(row["species"]),
            str(row["task"]),
        )
        for row in manifest_rows
    )
    return [
        (split, species, task, count)
        for (split, species, task), count in sorted(
            counts.items(),
            key=lambda item: (
                0 if item[0][0] == "train" else 1,
                CLASS_NAMES_INV[item[0][1]],
                0 if item[0][2] == "생육" else 1,
            ),
        )
    ]


def markdown_table(
    headers: Sequence[str], rows: Sequence[Sequence[Any]]
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
    manifest_rows: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    *,
    seed: int,
    samples_per_species_task: int,
    max_total_images: int,
    source_zip_modified: bool,
) -> None:
    split_counts = Counter(str(row["split"]) for row in manifest_rows)
    lines = [
        "# YOLO smoke dataset 요약",
        "",
        "## 실행 범위",
        "",
        f"- seed: `{seed}`",
        f"- 품종×작업×split별 상한: `{samples_per_species_task}`",
        f"- 전체 이미지 강제 상한: `{max_total_images}` "
        f"(코드 절대 상한 {HARD_MAX_TOTAL_IMAGES})",
        "- 사용 split: Train, Validation",
        "- Test 이미지: 0개",
        "- 선택된 이미지 ZIP 멤버만 읽고 나머지 이미지 바이트는 읽지 않음",
        "- 전체 ZIP 압축 해제 및 모델 학습을 수행하지 않음",
        "",
        "## 결과",
        "",
        f"- Train 이미지: **{split_counts['train']:,}개**",
        f"- Validation 이미지: **{split_counts['validation']:,}개**",
        f"- 이미지/라벨: **{validation['image_count']:,}/"
        f"{validation['label_count']:,}개**",
        f"- bbox: **{validation['bbox_count']:,}개**",
        f"- 누락 이미지/라벨: **{validation['missing_images']:,}/"
        f"{validation['missing_labels']:,}개**",
        f"- 좌표 검증 오류: **{validation['coordinate_errors']:,}개**",
        f"- Train/Validation 동일 원천 이미지: "
        f"**{validation['train_val_source_overlap']:,}개**",
        f"- 원본 ZIP 변경: **{'있음' if source_zip_modified else '없음'}**",
        "",
        "## 품종×작업×split 샘플 수",
        "",
        markdown_table(
            ("split", "품종", "작업", "이미지 수"),
            distribution_rows(manifest_rows),
        ),
        "",
        "## 검증",
        "",
        f"- 빈 라벨: {validation['empty_labels']:,}개",
        f"- 예상 밖 이미지/라벨: {validation['unexpected_images']:,}/"
        f"{validation['unexpected_labels']:,}개",
        f"- 중복 출력 이미지: {validation['duplicate_output_images']:,}개",
        f"- Test 이미지: {validation['test_images']:,}개",
        f"- 로컬 절대경로 노출 파일: {len(validation['path_leaks']):,}개",
        "",
    ]
    write_text(output_root / "smoke_dataset_summary.md", "\n".join(lines))


def create_smoke_dataset(
    selected: Sequence[Mapping[str, str]],
    archive_lookup: Mapping[str, base.ArchiveRef],
    output_dir: Path,
    *,
    seed: int,
    samples_per_species_task: int,
    max_total_images: int,
    overwrite: bool,
) -> dict[str, Any]:
    if len(selected) > min(max_total_images, HARD_MAX_TOTAL_IMAGES):
        raise ValueError("선택 이미지 수가 안전 상한을 초과했습니다")
    if any(row["split"] == "test" for row in selected):
        raise ValueError("Test split은 추출할 수 없습니다")
    selected_keys = [candidate_unique_key(row) for row in selected]
    if len(selected_keys) != len(set(selected_keys)):
        raise ValueError("선택 목록에 중복 이미지가 있습니다")
    source_snapshot = audit.source_zip_snapshot(
        ref.path for ref in archive_lookup.values()
    )
    staging, backup = prepare_staging(output_dir, overwrite=overwrite)
    try:
        for relative in (
            "images/train",
            "images/val",
            "labels/train",
            "labels/val",
            "overlays/train",
            "overlays/val",
        ):
            (staging / relative).mkdir(parents=True, exist_ok=True)
        write_static_metadata(staging)
        manifest_rows = extract_selected_images(
            selected, archive_lookup, staging
        )
        write_csv(
            staging / "smoke_manifest.csv",
            SMOKE_MANIFEST_COLUMNS,
            manifest_rows,
        )
        validation = validate_output(staging, manifest_rows)
        write_summary(
            staging,
            manifest_rows,
            validation,
            seed=seed,
            samples_per_species_task=samples_per_species_task,
            max_total_images=max_total_images,
            source_zip_modified=False,
        )
        leaks = text_path_leaks(staging)
        if leaks:
            raise RuntimeError(
                "결과 파일에서 로컬 절대경로 발견: " + ", ".join(leaks)
            )
        audit.assert_source_zips_unchanged(source_snapshot)
        finalize_staging(staging, output_dir, backup)
    except Exception:
        rollback_staging(staging, output_dir, backup)
        audit.assert_source_zips_unchanged(source_snapshot)
        raise
    split_counts = Counter(str(row["split"]) for row in manifest_rows)
    return {
        "train_images": split_counts["train"],
        "validation_images": split_counts["validation"],
        "total_images": len(manifest_rows),
        "bbox_count": validation["bbox_count"],
        "missing_images": validation["missing_images"],
        "missing_labels": validation["missing_labels"],
        "coordinate_errors": validation["coordinate_errors"],
        "test_images": validation["test_images"],
        "train_val_source_overlap": validation["train_val_source_overlap"],
        "source_zip_modified": False,
        "distribution": [
            {
                "split": split,
                "species": species,
                "task": task,
                "count": count,
            }
            for split, species, task, count in distribution_rows(manifest_rows)
        ],
    }


def run(
    args: argparse.Namespace,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    selected = select_manifest_rows(
        args.manifest,
        splits=args.splits,
        samples_per_species_task=args.samples_per_species_task,
        seed=args.seed,
        max_total_images=args.max_total_images,
    )
    refs, issues, _directories = base.discover_archives(environ)
    if issues:
        raise RuntimeError(
            "아카이브 환경/구조 검증 실패:\n- " + "\n- ".join(issues)
        )
    lookup = image_archive_lookup(refs)
    result = create_smoke_dataset(
        selected,
        lookup,
        args.output_dir,
        seed=args.seed,
        samples_per_species_task=args.samples_per_species_task,
        max_total_images=args.max_total_images,
        overwrite=args.overwrite,
    )
    result.update(
        {
            "output_dir": base.portable_output_path(args.output_dir),
            "full_archive_extraction": False,
            "full_dataset_conversion": False,
            "test_split_used": False,
            "model_training": False,
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
