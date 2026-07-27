#!/usr/bin/env python3
"""Create a deterministic 6,000-image YOLO pilot dataset.

The final detection manifest is streamed twice: once for stratum counts and
once for bounded hash sampling.  Only selected image members are read from
source ZIPs.  Test data, whole-archive extraction, full conversion, and model
training are forbidden.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tqdm import tqdm

import audit_mushroom_labels as audit
import inspect_aihub_archives as base
import create_yolo_smoke_dataset as smoke


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "detection_dataset_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "yolo_pilot"
SMOKE_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "yolo_smoke"
DEFAULT_TRAIN_PER_GROUP = 500
DEFAULT_VALIDATION_PER_GROUP = 100
DEFAULT_SEED = 20260726
HARD_MAX_TOTAL_IMAGES = 6_000
HARD_MAX_OVERLAY_IMAGES = 100
DEFAULT_MAX_OVERLAY_IMAGES = 100
OVERLAY_PER_GROUP = 5
SPECIES = tuple(smoke.CLASS_NAMES[index] for index in sorted(smoke.CLASS_NAMES))
TASKS = ("생육", "병해")
SPLITS = ("train", "validation")
PILOT_SOURCE_COLUMNS = (
    "split",
    "group_key",
    "species",
    "class_id",
    "task",
    "normality",
    "disease_type",
    "camera_id",
    "capture_date",
    "capture_time",
    "label_archive_id",
    "image_archive_id",
    "json_member",
    "image_member",
    "image_width",
    "image_height",
    "valid_bbox_count",
    "bbox_cleaned",
    "official_split",
)
PILOT_MANIFEST_COLUMNS = (
    "split",
    "yolo_split",
    "group_key",
    "species",
    "class_id",
    "task",
    "normality",
    "disease_type",
    "camera_id",
    "capture_date",
    "capture_time",
    "label_archive_id",
    "image_archive_id",
    "json_member",
    "image_member",
    "image_width",
    "image_height",
    "bbox_count",
    "image_path",
    "label_path",
    "overlay_path",
    "official_split",
)

# Re-export the tested smoke conversion primitives as the pilot's shared API.
convert_bbox_to_yolo = smoke.convert_bbox_to_yolo
yolo_boxes_for_row = smoke.yolo_boxes_for_row


@dataclass(frozen=True)
class PilotCandidate:
    rank: int
    unique_key: str
    row: dict[str, str]


def bounded_total(value: str) -> int:
    number = smoke.positive_int(value)
    if number > HARD_MAX_TOTAL_IMAGES:
        raise argparse.ArgumentTypeError(
            f"안전 제한상 {HARD_MAX_TOTAL_IMAGES} 이하여야 합니다"
        )
    return number


def bounded_overlay(value: str) -> int:
    number = smoke.positive_int(value)
    if number > HARD_MAX_OVERLAY_IMAGES:
        raise argparse.ArgumentTypeError(
            f"overlay는 {HARD_MAX_OVERLAY_IMAGES}장 이하여야 합니다"
        )
    return number


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train 5,000장과 Validation 1,000장의 균형 잡힌 YOLO "
            "파일럿 데이터셋을 생성합니다. Test와 모델 학습은 사용하지 않습니다."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/detection_dataset_manifest.csv"),
    )
    parser.add_argument(
        "--train-per-species-task",
        type=smoke.positive_int,
        default=DEFAULT_TRAIN_PER_GROUP,
    )
    parser.add_argument(
        "--validation-per-species-task",
        type=smoke.positive_int,
        default=DEFAULT_VALIDATION_PER_GROUP,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-total-images",
        type=bounded_total,
        default=HARD_MAX_TOTAL_IMAGES,
    )
    parser.add_argument(
        "--max-overlay-images",
        type=bounded_overlay,
        default=DEFAULT_MAX_OVERLAY_IMAGES,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/yolo_pilot"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 파일럿 출력 디렉터리를 안전하게 교체",
    )
    args = parser.parse_args(argv)
    args.manifest = project_path(args.manifest)
    args.output_dir = project_path(args.output_dir)
    artifacts_root = (PROJECT_ROOT / "artifacts").resolve()
    if not args.output_dir.is_relative_to(artifacts_root):
        parser.error("--output-dir은 프로젝트의 artifacts 아래여야 합니다")
    smoke_root = SMOKE_OUTPUT_DIR.resolve()
    if (
        args.output_dir == smoke_root
        or args.output_dir.is_relative_to(smoke_root)
        or smoke_root.is_relative_to(args.output_dir)
    ):
        parser.error("--output-dir은 기존 artifacts/yolo_smoke와 분리해야 합니다")
    validate_requested_limits(
        args.train_per_species_task,
        args.validation_per_species_task,
        args.max_total_images,
        args.max_overlay_images,
    )
    return args


def requested_group_targets(
    train_per_species_task: int,
    validation_per_species_task: int,
) -> dict[tuple[str, str, str], int]:
    return {
        (split, species, task): (
            train_per_species_task
            if split == "train"
            else validation_per_species_task
        )
        for split in SPLITS
        for species in SPECIES
        for task in TASKS
    }


def validate_requested_limits(
    train_per_species_task: int,
    validation_per_species_task: int,
    max_total_images: int,
    max_overlay_images: int,
) -> int:
    if min(
        train_per_species_task,
        validation_per_species_task,
        max_total_images,
        max_overlay_images,
    ) <= 0:
        raise ValueError("요청 수는 모두 양수여야 합니다")
    if max_total_images > HARD_MAX_TOTAL_IMAGES:
        raise ValueError(
            f"파일럿 이미지 강제 상한은 {HARD_MAX_TOTAL_IMAGES}장입니다"
        )
    if max_overlay_images > HARD_MAX_OVERLAY_IMAGES:
        raise ValueError(
            f"overlay 강제 상한은 {HARD_MAX_OVERLAY_IMAGES}장입니다"
        )
    requested = 10 * (
        train_per_species_task + validation_per_species_task
    )
    if requested > max_total_images:
        raise ValueError(
            f"요청 이미지 {requested}장이 max_total_images "
            f"{max_total_images}장을 초과합니다"
        )
    return requested


def source_group(row: Mapping[str, str]) -> tuple[str, str, str]:
    return row["split"], row["species"], row["task"]


def source_stratum(row: Mapping[str, str]) -> tuple[str, str]:
    normality = row.get("normality") or "<missing>"
    disease_type = row.get("disease_type") or "<missing>"
    return normality, disease_type


def validate_source_row(row: Mapping[str, str]) -> None:
    if row["split"] == "test":
        raise ValueError("Test split 레코드는 선택 대상으로 사용할 수 없습니다")
    if row["split"] not in SPLITS:
        raise ValueError(f"지원하지 않는 split: {row['split']}")
    class_id = smoke.parse_class_id(row["class_id"])
    if smoke.CLASS_NAMES[class_id] != row["species"]:
        raise ValueError(
            f"class_id/품종 불일치: {row['class_id']} {row['species']}"
        )
    if row["task"] not in TASKS:
        raise ValueError(f"지원하지 않는 작업: {row['task']}")


def manifest_stratum_counts(
    manifest_path: Path,
) -> dict[tuple[str, str, str], Counter[tuple[str, str]]]:
    counts: dict[
        tuple[str, str, str], Counter[tuple[str, str]]
    ] = defaultdict(Counter)
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = smoke.REQUIRED_MANIFEST_COLUMNS - set(
            reader.fieldnames or ()
        )
        if missing:
            raise ValueError(
                "manifest 필수 컬럼 누락: " + ", ".join(sorted(missing))
            )
        for row in reader:
            if row["split"] == "test":
                continue
            validate_source_row(row)
            counts[source_group(row)][source_stratum(row)] += 1
    return counts


def proportional_quotas(
    counts: Mapping[tuple[str, str], int],
    target: int,
    *,
    seed: int,
    group: tuple[str, str, str],
) -> dict[tuple[str, str], int]:
    total = sum(counts.values())
    if total < target:
        raise ValueError(
            f"{group} 후보 {total}개가 요청 {target}개보다 적습니다"
        )
    quotas: dict[tuple[str, str], int] = {}
    remainders: list[tuple[float, str, tuple[str, str]]] = []
    for stratum, count in sorted(counts.items()):
        exact = target * count / total
        base_quota = min(count, int(exact))
        quotas[stratum] = base_quota
        remainders.append(
            (
                exact - base_quota,
                smoke.stable_score(seed, group, stratum),
                stratum,
            )
        )
    remaining = target - sum(quotas.values())
    for _fraction, _tie, stratum in sorted(
        remainders, key=lambda item: (-item[0], item[1], item[2])
    ):
        if remaining == 0:
            break
        if quotas[stratum] < counts[stratum]:
            quotas[stratum] += 1
            remaining -= 1
    if remaining:
        for stratum in sorted(
            counts,
            key=lambda value: smoke.stable_score(seed, group, value),
        ):
            while remaining and quotas[stratum] < counts[stratum]:
                quotas[stratum] += 1
                remaining -= 1
    if remaining or sum(quotas.values()) != target:
        raise RuntimeError(f"{group} 층화 할당량 계산 실패")
    return quotas


def build_stratum_quotas(
    counts: Mapping[
        tuple[str, str, str], Mapping[tuple[str, str], int]
    ],
    group_targets: Mapping[tuple[str, str, str], int],
    *,
    seed: int,
) -> dict[tuple[str, str, str, str, str], int]:
    quotas: dict[tuple[str, str, str, str, str], int] = {}
    missing_groups = sorted(set(group_targets) - set(counts))
    if missing_groups:
        raise ValueError(f"manifest 후보가 없는 그룹: {missing_groups}")
    for group, target in sorted(group_targets.items()):
        allocation = proportional_quotas(
            counts[group], target, seed=seed, group=group
        )
        for (normality, disease_type), quota in allocation.items():
            if quota:
                quotas[(*group, normality, disease_type)] = quota
    return quotas


def select_pilot_rows(
    manifest_path: Path,
    *,
    train_per_species_task: int,
    validation_per_species_task: int,
    seed: int,
    max_total_images: int,
) -> list[dict[str, str]]:
    expected_total = validate_requested_limits(
        train_per_species_task,
        validation_per_species_task,
        max_total_images,
        1,
    )
    group_targets = requested_group_targets(
        train_per_species_task, validation_per_species_task
    )
    counts = manifest_stratum_counts(manifest_path)
    quotas = build_stratum_quotas(
        counts, group_targets, seed=seed
    )
    heaps: dict[
        tuple[str, str, str, str, str],
        list[tuple[int, int, PilotCandidate]],
    ] = defaultdict(list)
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            if row["split"] == "test":
                continue
            validate_source_row(row)
            normality, disease_type = source_stratum(row)
            stratum_key = (*source_group(row), normality, disease_type)
            limit = quotas.get(stratum_key, 0)
            if limit <= 0:
                continue
            unique_key = smoke.candidate_unique_key(row)
            rank = int(smoke.stable_score(seed, unique_key), 16)
            retained_row = {
                column: row.get(column, "")
                for column in PILOT_SOURCE_COLUMNS
            }
            candidate = PilotCandidate(rank, unique_key, retained_row)
            heap = heaps[stratum_key]
            item = (-rank, line_number, candidate)
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, item)
    selected: list[dict[str, str]] = []
    for stratum_key, quota in sorted(quotas.items()):
        heap = heaps.get(stratum_key, [])
        if len(heap) != quota:
            raise RuntimeError(
                f"{stratum_key} 선택 수 {len(heap)}개(요청 {quota}개)"
            )
        selected.extend(item[2].row for item in heap)
    selected.sort(
        key=lambda row: (
            0 if row["split"] == "train" else 1,
            smoke.CLASS_NAMES_INV[row["species"]],
            0 if row["task"] == "생육" else 1,
            smoke.stable_score(seed, smoke.candidate_unique_key(row)),
        )
    )
    unique_keys = [smoke.candidate_unique_key(row) for row in selected]
    if len(unique_keys) != len(set(unique_keys)):
        raise RuntimeError("파일럿 선택 결과에 중복 이미지가 있습니다")
    if len(selected) != expected_total:
        raise RuntimeError(
            f"파일럿 선택 수 {len(selected)}개(기대값 {expected_total}개)"
        )
    actual_groups = Counter(source_group(row) for row in selected)
    mismatches = {
        group: (actual_groups[group], target)
        for group, target in group_targets.items()
        if actual_groups[group] != target
    }
    if mismatches:
        raise RuntimeError(f"품종×작업 요청 수 불일치: {mismatches}")
    if any(row["split"] == "test" for row in selected):
        raise RuntimeError("Test 레코드가 선택되었습니다")
    return selected


def select_overlay_keys(
    selected: Sequence[Mapping[str, str]],
    *,
    seed: int,
    max_overlay_images: int,
    per_group: int = OVERLAY_PER_GROUP,
) -> set[str]:
    if max_overlay_images > HARD_MAX_OVERLAY_IMAGES:
        raise ValueError(
            f"overlay 강제 상한은 {HARD_MAX_OVERLAY_IMAGES}장입니다"
        )
    buckets: dict[
        tuple[str, str, str], list[tuple[str, str]]
    ] = defaultdict(list)
    for row in selected:
        if row["split"] == "test":
            raise ValueError("Test overlay는 생성할 수 없습니다")
        unique_key = smoke.candidate_unique_key(row)
        score = smoke.stable_score(seed, "overlay", unique_key)
        bucket = buckets[source_group(row)]
        bucket.append((score, unique_key))
        bucket.sort()
        if len(bucket) > per_group:
            bucket.pop()
    ordered_groups = sorted(
        buckets,
        key=lambda group: (
            0 if group[0] == "train" else 1,
            smoke.CLASS_NAMES_INV[group[1]],
            0 if group[2] == "생육" else 1,
        ),
    )
    result: set[str] = set()
    depth = 0
    while len(result) < max_overlay_images:
        added = False
        for group in ordered_groups:
            if depth >= len(buckets[group]):
                continue
            result.add(buckets[group][depth][1])
            added = True
            if len(result) == max_overlay_images:
                break
        if not added:
            break
        depth += 1
    if len(result) > min(max_overlay_images, HARD_MAX_OVERLAY_IMAGES):
        raise RuntimeError("overlay 안전 상한을 초과했습니다")
    return result


def directory_snapshot(
    root: Path,
) -> dict[str, tuple[int, int]]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def assert_directory_unchanged(
    root: Path,
    before: Mapping[str, tuple[int, int]],
) -> None:
    after = directory_snapshot(root)
    if dict(before) != after:
        raise RuntimeError(
            f"기존 디렉터리 변경 감지: {base.portable_output_path(root)}"
        )


def extract_pilot_images(
    selected: Sequence[Mapping[str, str]],
    archive_lookup: Mapping[str, base.ArchiveRef],
    output_root: Path,
    *,
    overlay_keys: set[str],
    show_progress: bool,
) -> list[dict[str, Any]]:
    by_archive: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in selected:
        if row["split"] == "test":
            raise RuntimeError("Test 이미지는 추출할 수 없습니다")
        by_archive[row["image_archive_id"]].append(row)
    rows: list[dict[str, Any]] = []
    progress = tqdm(
        total=len(selected),
        desc="YOLO 파일럿 이미지",
        unit="image",
        disable=not show_progress,
    )
    try:
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
                    boxes = smoke.yolo_boxes_for_row(row)
                    image_bytes = archive.read(info)
                    split_dir = smoke.yolo_split(row["split"])
                    basename = smoke.output_basename(row)
                    image_path = (
                        output_root / "images" / split_dir / basename
                    )
                    label_path = (
                        output_root
                        / "labels"
                        / split_dir
                        / f"{Path(basename).stem}.txt"
                    )
                    unique_key = smoke.candidate_unique_key(row)
                    overlay_path: Path | None = None
                    if unique_key in overlay_keys:
                        overlay_path = (
                            output_root
                            / "overlays"
                            / split_dir
                            / f"{Path(basename).stem}.jpg"
                        )
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image_path.write_bytes(image_bytes)
                    smoke.write_text(
                        label_path,
                        "\n".join(box.line() for box in boxes) + "\n",
                    )
                    if overlay_path is not None:
                        smoke.draw_overlay(
                            image_bytes,
                            boxes,
                            expected_width=smoke.positive_dimension(
                                row["image_width"], "image_width"
                            ),
                            expected_height=smoke.positive_dimension(
                                row["image_height"], "image_height"
                            ),
                            destination=overlay_path,
                        )
                    rows.append(
                        {
                            "split": row["split"],
                            "yolo_split": split_dir,
                            "group_key": row.get("group_key", ""),
                            "species": row["species"],
                            "class_id": row["class_id"],
                            "task": row["task"],
                            "normality": row["normality"],
                            "disease_type": row["disease_type"],
                            "camera_id": row.get("camera_id", ""),
                            "capture_date": row.get("capture_date", ""),
                            "capture_time": row.get("capture_time", ""),
                            "label_archive_id": row.get(
                                "label_archive_id", ""
                            ),
                            "image_archive_id": archive_id,
                            "json_member": row.get("json_member", ""),
                            "image_member": member,
                            "image_width": row["image_width"],
                            "image_height": row["image_height"],
                            "bbox_count": len(boxes),
                            "image_path": smoke.portable_relative(
                                image_path, output_root
                            ),
                            "label_path": smoke.portable_relative(
                                label_path, output_root
                            ),
                            "overlay_path": (
                                smoke.portable_relative(
                                    overlay_path, output_root
                                )
                                if overlay_path is not None
                                else ""
                            ),
                            "official_split": row.get(
                                "official_split", ""
                            ),
                        }
                    )
                    progress.update(1)
    finally:
        progress.close()
    rows.sort(
        key=lambda row: (
            0 if row["split"] == "train" else 1,
            int(row["class_id"]),
            0 if row["task"] == "생육" else 1,
            row["image_member"],
        )
    )
    return rows


def validate_pilot_output(
    output_root: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    train_per_species_task: int,
    validation_per_species_task: int,
    max_overlay_images: int,
) -> dict[str, Any]:
    validation = smoke.validate_output(output_root, manifest_rows)
    expected_targets = requested_group_targets(
        train_per_species_task, validation_per_species_task
    )
    actual_groups = Counter(
        (
            str(row["split"]),
            str(row["species"]),
            str(row["task"]),
        )
        for row in manifest_rows
    )
    group_mismatches = {
        group: {
            "actual": actual_groups[group],
            "expected": target,
        }
        for group, target in expected_targets.items()
        if actual_groups[group] != target
    }
    train_expected = 10 * train_per_species_task
    val_expected = 10 * validation_per_species_task
    train_actual = sum(row["split"] == "train" for row in manifest_rows)
    val_actual = sum(
        row["split"] == "validation" for row in manifest_rows
    )
    overlay_paths = {
        str(row["overlay_path"])
        for row in manifest_rows
        if row.get("overlay_path")
    }
    actual_overlays = {
        smoke.portable_relative(path, output_root)
        for split in ("train", "val")
        for path in (output_root / "overlays" / split).glob("*.jpg")
        if path.is_file()
    }
    output_names = [str(row["image_path"]) for row in manifest_rows]
    validation.update(
        {
            "train_images": train_actual,
            "validation_images": val_actual,
            "expected_train_images": train_expected,
            "expected_validation_images": val_expected,
            "group_mismatches": group_mismatches,
            "overlay_count": len(actual_overlays),
            "missing_overlays": len(overlay_paths - actual_overlays),
            "unexpected_overlays": len(actual_overlays - overlay_paths),
            "output_filename_collisions": (
                len(output_names) - len(set(output_names))
            ),
        }
    )
    if train_actual != train_expected or val_actual != val_expected:
        raise RuntimeError(f"파일럿 split 수량 검증 실패: {validation}")
    if group_mismatches:
        raise RuntimeError(f"파일럿 그룹 수량 검증 실패: {validation}")
    if (
        validation["overlay_count"] > max_overlay_images
        or validation["overlay_count"] > HARD_MAX_OVERLAY_IMAGES
        or validation["missing_overlays"]
        or validation["unexpected_overlays"]
    ):
        raise RuntimeError(f"overlay 검증 실패: {validation}")
    if validation["output_filename_collisions"]:
        raise RuntimeError(f"출력 파일명 충돌: {validation}")
    return validation


def pilot_distribution(
    manifest_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Counter[Any]]:
    return {
        "species_task": Counter(
            (
                str(row["split"]),
                str(row["species"]),
                str(row["task"]),
            )
            for row in manifest_rows
        ),
        "normality": Counter(
            (str(row["split"]), str(row["normality"]))
            for row in manifest_rows
        ),
        "disease": Counter(
            (str(row["split"]), str(row["disease_type"]))
            for row in manifest_rows
            if row["task"] == "병해"
        ),
    }


def ordered_species_task_rows(
    counts: Mapping[tuple[str, str, str], int],
) -> list[tuple[str, str, str, int]]:
    return [
        (split, species, task, counts.get((split, species, task), 0))
        for split in SPLITS
        for species in SPECIES
        for task in TASKS
    ]


def write_pilot_summary(
    output_root: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    *,
    seed: int,
    train_per_species_task: int,
    validation_per_species_task: int,
    max_total_images: int,
    max_overlay_images: int,
) -> None:
    distribution = pilot_distribution(manifest_rows)
    normality_rows = [
        (split, normality, count)
        for (split, normality), count in sorted(
            distribution["normality"].items(),
            key=lambda item: (
                0 if item[0][0] == "train" else 1,
                item[0][1],
            ),
        )
    ]
    disease_rows = [
        (split, disease, count)
        for (split, disease), count in sorted(
            distribution["disease"].items(),
            key=lambda item: (
                0 if item[0][0] == "train" else 1,
                item[0][1],
            ),
        )
    ]
    lines = [
        "# YOLO 파일럿 데이터셋 요약",
        "",
        "## 실행 범위",
        "",
        f"- seed: `{seed}`",
        f"- Train 품종×작업별 요청: `{train_per_species_task}`장",
        f"- Validation 품종×작업별 요청: "
        f"`{validation_per_species_task}`장",
        f"- 전체 이미지 강제 상한: `{max_total_images}` "
        f"(코드 절대 상한 {HARD_MAX_TOTAL_IMAGES})",
        f"- overlay 상한: `{max_overlay_images}` "
        f"(코드 절대 상한 {HARD_MAX_OVERLAY_IMAGES})",
        "- 사용 split: Train, Validation",
        "- Test 이미지: 0장",
        "- 병해 데이터는 split×품종×병해 종류 비율을 기준으로 층화 샘플링",
        "- 선택된 이미지 ZIP 멤버만 읽고 모델 학습은 수행하지 않음",
        "",
        "## 결과",
        "",
        f"- Train 이미지: **{validation['train_images']:,}장**",
        f"- Validation 이미지: "
        f"**{validation['validation_images']:,}장**",
        f"- 이미지/라벨: **{validation['image_count']:,}/"
        f"{validation['label_count']:,}개**",
        f"- bbox: **{validation['bbox_count']:,}개**",
        f"- overlay: **{validation['overlay_count']:,}장**",
        f"- 누락 이미지/라벨: **{validation['missing_images']:,}/"
        f"{validation['missing_labels']:,}개**",
        f"- 좌표 검증 오류: **{validation['coordinate_errors']:,}개**",
        f"- Train/Validation 동일 이미지: "
        f"**{validation['train_val_source_overlap']:,}개**",
        f"- Test 이미지: **{validation['test_images']:,}개**",
        "- 원본 ZIP 변경: **없음**",
        "- 기존 `artifacts/yolo_smoke` 변경: **없음**",
        "",
        "## 품종×작업×split",
        "",
        smoke.markdown_table(
            ("split", "품종", "작업", "이미지 수"),
            ordered_species_task_rows(distribution["species_task"]),
        ),
        "",
        "## 정상 여부",
        "",
        smoke.markdown_table(
            ("split", "정상 여부", "이미지 수"), normality_rows
        ),
        "",
        "## 병해 종류",
        "",
        smoke.markdown_table(
            ("split", "병해 종류", "이미지 수"), disease_rows
        ),
        "",
        "## 검증",
        "",
        f"- 빈 라벨: {validation['empty_labels']:,}개",
        f"- 출력 파일명 충돌: "
        f"{validation['output_filename_collisions']:,}개",
        f"- 예상 밖 이미지/라벨: "
        f"{validation['unexpected_images']:,}/"
        f"{validation['unexpected_labels']:,}개",
        f"- overlay 누락/예상 밖: "
        f"{validation['missing_overlays']:,}/"
        f"{validation['unexpected_overlays']:,}개",
        f"- 로컬 절대경로 노출 파일: "
        f"{len(validation['path_leaks']):,}개",
        "",
    ]
    smoke.write_text(
        output_root / "pilot_dataset_summary.md", "\n".join(lines)
    )


def create_pilot_dataset(
    selected: Sequence[Mapping[str, str]],
    archive_lookup: Mapping[str, base.ArchiveRef],
    output_dir: Path,
    *,
    train_per_species_task: int,
    validation_per_species_task: int,
    seed: int,
    max_total_images: int,
    max_overlay_images: int,
    overwrite: bool,
    smoke_dir: Path = SMOKE_OUTPUT_DIR,
    show_progress: bool = True,
) -> dict[str, Any]:
    expected_total = validate_requested_limits(
        train_per_species_task,
        validation_per_species_task,
        max_total_images,
        max_overlay_images,
    )
    if len(selected) != expected_total:
        raise ValueError(
            f"선택 이미지 {len(selected)}장(기대값 {expected_total}장)"
        )
    if any(row["split"] == "test" for row in selected):
        raise ValueError("Test split은 추출할 수 없습니다")
    source_keys = [smoke.candidate_unique_key(row) for row in selected]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("선택 목록에 중복 이미지가 있습니다")
    source_snapshot = audit.source_zip_snapshot(
        ref.path for ref in archive_lookup.values()
    )
    smoke_snapshot = directory_snapshot(smoke_dir)
    overlay_keys = select_overlay_keys(
        selected,
        seed=seed,
        max_overlay_images=max_overlay_images,
    )
    staging, backup = smoke.prepare_staging(
        output_dir, overwrite=overwrite
    )
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
        smoke.write_static_metadata(staging)
        rows = extract_pilot_images(
            selected,
            archive_lookup,
            staging,
            overlay_keys=overlay_keys,
            show_progress=show_progress,
        )
        smoke.write_csv(
            staging / "pilot_manifest.csv",
            PILOT_MANIFEST_COLUMNS,
            rows,
        )
        validation = validate_pilot_output(
            staging,
            rows,
            train_per_species_task=train_per_species_task,
            validation_per_species_task=validation_per_species_task,
            max_overlay_images=max_overlay_images,
        )
        write_pilot_summary(
            staging,
            rows,
            validation,
            seed=seed,
            train_per_species_task=train_per_species_task,
            validation_per_species_task=validation_per_species_task,
            max_total_images=max_total_images,
            max_overlay_images=max_overlay_images,
        )
        leaks = smoke.text_path_leaks(staging)
        if leaks:
            raise RuntimeError(
                "결과 파일에서 로컬 절대경로 발견: "
                + ", ".join(leaks)
            )
        audit.assert_source_zips_unchanged(source_snapshot)
        assert_directory_unchanged(smoke_dir, smoke_snapshot)
        smoke.finalize_staging(staging, output_dir, backup)
        assert_directory_unchanged(smoke_dir, smoke_snapshot)
    except Exception:
        smoke.rollback_staging(staging, output_dir, backup)
        audit.assert_source_zips_unchanged(source_snapshot)
        assert_directory_unchanged(smoke_dir, smoke_snapshot)
        raise
    distribution = pilot_distribution(rows)
    return {
        "train_images": validation["train_images"],
        "validation_images": validation["validation_images"],
        "total_images": validation["image_count"],
        "bbox_count": validation["bbox_count"],
        "overlay_count": validation["overlay_count"],
        "missing_images": validation["missing_images"],
        "missing_labels": validation["missing_labels"],
        "coordinate_errors": validation["coordinate_errors"],
        "train_val_source_overlap": validation[
            "train_val_source_overlap"
        ],
        "test_images": validation["test_images"],
        "source_zip_modified": False,
        "smoke_directory_modified": False,
        "species_task_distribution": [
            {
                "split": split,
                "species": species,
                "task": task,
                "count": count,
            }
            for split, species, task, count in ordered_species_task_rows(
                distribution["species_task"]
            )
        ],
        "disease_distribution": [
            {
                "split": split,
                "disease_type": disease,
                "count": count,
            }
            for (split, disease), count in sorted(
                distribution["disease"].items()
            )
        ],
    }


def run(
    args: argparse.Namespace,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    selected = select_pilot_rows(
        args.manifest,
        train_per_species_task=args.train_per_species_task,
        validation_per_species_task=args.validation_per_species_task,
        seed=args.seed,
        max_total_images=args.max_total_images,
    )
    refs, issues, _directories = base.discover_archives(environ)
    if issues:
        raise RuntimeError(
            "아카이브 환경/구조 검증 실패:\n- " + "\n- ".join(issues)
        )
    lookup = smoke.image_archive_lookup(refs)
    result = create_pilot_dataset(
        selected,
        lookup,
        args.output_dir,
        train_per_species_task=args.train_per_species_task,
        validation_per_species_task=args.validation_per_species_task,
        seed=args.seed,
        max_total_images=args.max_total_images,
        max_overlay_images=args.max_overlay_images,
        overwrite=args.overwrite,
    )
    result.update(
        {
            "output_dir": base.portable_output_path(args.output_dir),
            "hard_max_total_images": HARD_MAX_TOTAL_IMAGES,
            "hard_max_overlay_images": HARD_MAX_OVERLAY_IMAGES,
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
