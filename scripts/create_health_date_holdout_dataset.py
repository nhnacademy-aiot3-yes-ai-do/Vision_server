#!/usr/bin/env python3
"""Create the fixed date+global-camera holdout health pilot dataset.

The detection manifest is streamed twice.  Test rows are skipped before any
eligibility decision, and the old detection Train/Validation assignment is not
used for the new split.  Only these two intersections are eligible:

* non-Validation date + fixed Train global camera
* fixed Validation date + fixed Validation global camera

The crop, selected-ZIP-member reader, review rendering, and source/archive
invariance checks are shared with ``create_health_pilot_dataset``.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import audit_health_date_holdout as date_audit
import audit_mushroom_health as health
import audit_mushroom_labels as label_audit
import create_health_pilot_dataset as health_pilot
import create_yolo_pilot_dataset as pilot
import create_yolo_smoke_dataset as smoke
import inspect_aihub_archives as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "detection_dataset_manifest.csv"
DEFAULT_OUTPUT_DIR = ARTIFACTS_ROOT / "health_pilot_date_holdout"
DEFAULT_VALIDATION_DATES = (
    "2021-11-04",
    "2021-11-26",
    "2021-12-04",
)
FIXED_VALIDATION_CAMERA_IDS = (
    "1",
    "2",
    "6",
    "10",
    "13",
    "14",
    "16",
    "18",
    "20",
)
FIXED_TRAIN_CAMERA_IDS = (
    "3",
    "4",
    "5",
    "7",
    "8",
    "9",
    "11",
    "12",
    "15",
    "17",
    "19",
)
DEFAULT_SEED = 20260726
DEFAULT_TRAIN_PER_STATUS = 500
DEFAULT_VALIDATION_PER_STATUS = 100
DEFAULT_PADDING_RATIO = 0.15
DEFAULT_MAX_TOTAL_IMAGES = 6_000
DEFAULT_MAX_REVIEW_IMAGES = 100
HARD_MAX_TOTAL_IMAGES = 6_000
HARD_MAX_REVIEW_IMAGES = 100
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
    "output_relative_path",
)
PROTECTED_ARTIFACT_NAMES = (
    "health_pilot",
    "health_pilot_timestamp_masked",
    "yolo_smoke",
    "yolo_pilot",
    "yolo_pilot_camera_holdout",
    "models",
)

# Re-export the shared, already-tested geometry primitive.
crop_geometry = health_pilot.crop_geometry


@dataclass(frozen=True)
class DateCandidate:
    rank: int
    unique_key: str
    source_manifest_split: str
    row: dict[str, str]


@dataclass
class DateHoldoutStatistics:
    strata: dict[
        tuple[str, str, str], Counter[tuple[str, str]]
    ] = field(default_factory=lambda: defaultdict(Counter))
    eligible_counts: Counter[tuple[str, str, str]] = field(
        default_factory=Counter
    )
    eligible_rows: int = 0
    skipped_test_rows: int = 0
    excluded_cross_date_camera_rows: int = 0
    excluded_unknown_camera_rows: int = 0
    source_split_counts: Counter[tuple[str, str]] = field(
        default_factory=Counter
    )
    candidate_dates: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class DateHoldoutSelection:
    statistics: DateHoldoutStatistics
    selected: list[dict[str, str]]
    validation_dates: tuple[str, ...]
    train_camera_ids: tuple[str, ...]
    validation_camera_ids: tuple[str, ...]


def normalize_validation_dates(
    values: Sequence[str],
    *,
    require_fixed: bool = False,
) -> tuple[str, ...]:
    if not values:
        raise ValueError("Validation 날짜를 하나 이상 지정해야 합니다")
    canonical = tuple(
        date_audit.canonical_capture_date(value) for value in values
    )
    if len(set(canonical)) != len(canonical):
        raise ValueError("Validation 날짜가 중복되었습니다")
    ordered = tuple(sorted(canonical))
    if require_fixed and set(ordered) != set(DEFAULT_VALIDATION_DATES):
        raise ValueError(
            "Validation 날짜는 감사에서 확정한 3개와 정확히 같아야 합니다: "
            + ", ".join(DEFAULT_VALIDATION_DATES)
        )
    return ordered


def validate_requested_limits(
    train_per_species_status: int,
    validation_per_species_status: int,
    max_total_images: int,
    max_review_images: int,
) -> int:
    if max_total_images > HARD_MAX_TOTAL_IMAGES:
        raise ValueError(
            f"date-holdout 이미지 강제 상한은 {HARD_MAX_TOTAL_IMAGES}장입니다"
        )
    if max_review_images > HARD_MAX_REVIEW_IMAGES:
        raise ValueError(
            f"review 이미지 강제 상한은 {HARD_MAX_REVIEW_IMAGES}장입니다"
        )
    return health_pilot.validate_requested_limits(
        train_per_species_status,
        validation_per_species_status,
        max_total_images,
        max_review_images,
    )


def validate_output_location(
    output_dir: Path,
    *,
    artifacts_root: Path = ARTIFACTS_ROOT,
    parser: argparse.ArgumentParser | None = None,
) -> None:
    output = output_dir.resolve()
    artifacts = artifacts_root.resolve()

    def fail(message: str) -> None:
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    if output == artifacts or not output.is_relative_to(artifacts):
        fail("--output-dir은 프로젝트 artifacts의 전용 하위 경로여야 합니다")
    for name in PROTECTED_ARTIFACT_NAMES:
        protected = (artifacts / name).resolve()
        if (
            output == protected
            or output.is_relative_to(protected)
            or protected.is_relative_to(output)
        ):
            fail(
                "--output-dir은 기존 raw/masked/yolo/models artifacts와 "
                "완전히 분리해야 합니다"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "감사에서 고정한 날짜와 global camera_id를 동시에 분리한 "
            "6,000장 버섯 건강 이진 분류 파일럿을 생성합니다."
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
        "--validation-dates",
        nargs="+",
        default=list(DEFAULT_VALIDATION_DATES),
    )
    parser.add_argument(
        "--padding-ratio",
        type=health_pilot.bounded_ratio,
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
        default=Path("artifacts/health_pilot_date_holdout"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 date-holdout 출력만 안전하게 교체",
    )
    args = parser.parse_args(argv)
    args.manifest = pilot.project_path(args.manifest)
    args.output_dir = pilot.project_path(args.output_dir)
    try:
        args.validation_dates = normalize_validation_dates(
            args.validation_dates,
            require_fixed=True,
        )
        if args.seed != DEFAULT_SEED:
            raise ValueError(
                f"감사에서 확정한 seed {DEFAULT_SEED}만 사용할 수 있습니다"
            )
        validate_requested_limits(
            args.train_per_species_status,
            args.validation_per_species_status,
            args.max_total_images,
            args.max_review_images,
        )
    except ValueError as exc:
        parser.error(str(exc))
    validate_output_location(args.output_dir, parser=parser)
    return args


def _camera_tuple(values: Iterable[str]) -> tuple[str, ...]:
    normalized = {health.normalized(value) for value in values}
    if "" in normalized:
        raise ValueError("camera_id 분할에 빈 값이 있습니다")
    return tuple(sorted(normalized, key=health.natural_text_key))


def _validate_camera_partition(
    train_camera_ids: Sequence[str],
    validation_camera_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    train = _camera_tuple(train_camera_ids)
    validation = _camera_tuple(validation_camera_ids)
    overlap = set(train) & set(validation)
    if overlap:
        raise ValueError(
            "Train/Validation global camera_id 분할 중복: "
            + ", ".join(sorted(overlap, key=health.natural_text_key))
        )
    return train, validation


def _validate_candidate_row(row: Mapping[str, str]) -> None:
    if "class_id" in row:
        pilot.validate_source_row(row)
    else:
        if row["split"] not in {"train", "validation"}:
            raise ValueError(f"지원하지 않는 split: {row['split']}")
        if row["species"] not in pilot.SPECIES:
            raise ValueError(f"지원하지 않는 품종: {row['species']}")
        if row["task"] not in pilot.TASKS:
            raise ValueError(f"지원하지 않는 작업: {row['task']}")
    _state, issues = health.health_label_issues(
        row["task"],
        row["normality"],
        row["disease_type"],
    )
    if issues:
        raise ValueError(
            "건강 라벨 모순: "
            f"{smoke.candidate_unique_key(row)} "
            + ",".join(issues)
        )
    if "health_class_id" in row:
        class_id, class_name, _directory = health_pilot.health_class_for_row(
            row
        )
        if int(row["health_class_id"]) != class_id or str(
            row["health_class_name"]
        ) != class_name:
            raise ValueError("건강 class ID/name과 라벨이 일치하지 않습니다")


def assigned_split(
    row: Mapping[str, str],
    *,
    validation_dates: Sequence[str],
    train_camera_ids: Sequence[str] = FIXED_TRAIN_CAMERA_IDS,
    validation_camera_ids: Sequence[str] = FIXED_VALIDATION_CAMERA_IDS,
) -> str | None:
    """Return the strict date∩camera split, or None for a crossed row."""
    if row["split"] == "test":
        raise ValueError("Test 행은 split 배정 전에 제외해야 합니다")
    capture_date = date_audit.canonical_capture_date(row["capture_date"])
    camera_id = health.normalized(row["camera_id"])
    validation_date_set = set(validation_dates)
    train_camera_set = set(train_camera_ids)
    validation_camera_set = set(validation_camera_ids)
    if capture_date in validation_date_set:
        return "validation" if camera_id in validation_camera_set else None
    return "train" if camera_id in train_camera_set else None


def stream_eligible_statistics(
    manifest_path: Path,
    *,
    validation_dates: Sequence[str],
    train_camera_ids: Sequence[str] = FIXED_TRAIN_CAMERA_IDS,
    validation_camera_ids: Sequence[str] = FIXED_VALIDATION_CAMERA_IDS,
) -> DateHoldoutStatistics:
    dates = normalize_validation_dates(validation_dates)
    train_cameras, validation_cameras = _validate_camera_partition(
        train_camera_ids,
        validation_camera_ids,
    )
    known_cameras = set(train_cameras) | set(validation_cameras)
    stats = DateHoldoutStatistics()
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = (
            set(pilot.PILOT_SOURCE_COLUMNS)
            | smoke.REQUIRED_MANIFEST_COLUMNS
            | {"camera_id", "capture_date", "normality", "disease_type"}
        )
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "manifest 필수 컬럼 누락: " + ", ".join(sorted(missing))
            )
        for row in reader:
            if row["split"] == "test":
                stats.skipped_test_rows += 1
                continue
            _validate_candidate_row(row)
            capture_date = date_audit.canonical_capture_date(
                row["capture_date"]
            )
            camera_id = health.normalized(row["camera_id"])
            stats.candidate_dates.add(capture_date)
            new_split = assigned_split(
                row,
                validation_dates=dates,
                train_camera_ids=train_cameras,
                validation_camera_ids=validation_cameras,
            )
            if new_split is None:
                if camera_id not in known_cameras:
                    stats.excluded_unknown_camera_rows += 1
                else:
                    stats.excluded_cross_date_camera_rows += 1
                continue
            normality = health.normalized(row["normality"])
            disease = health.normalized(row["disease_type"]) or "<missing>"
            group = (new_split, row["species"], normality)
            stratum = (disease, capture_date)
            stats.strata[group][stratum] += 1
            stats.eligible_counts[group] += 1
            stats.source_split_counts[(new_split, row["split"])] += 1
            stats.eligible_rows += 1
    return stats


def requested_targets(
    train_per_species_status: int,
    validation_per_species_status: int,
) -> dict[tuple[str, str, str], int]:
    return {
        (split, species, normality): (
            train_per_species_status
            if split == "train"
            else validation_per_species_status
        )
        for split in ("train", "validation")
        for species in pilot.SPECIES
        for normality in ("normal", "abnormal")
    }


def build_stratum_quotas(
    stats: DateHoldoutStatistics,
    targets: Mapping[tuple[str, str, str], int],
    *,
    seed: int,
) -> dict[tuple[str, str, str, str, str], int]:
    quotas: dict[tuple[str, str, str, str, str], int] = {}
    for group, target in sorted(targets.items()):
        counts = stats.strata.get(group)
        if not counts or sum(counts.values()) < target:
            available = sum(counts.values()) if counts else 0
            raise ValueError(
                f"{group} strict date+camera 후보 수량 부족: "
                f"{available}/{target}"
            )
        allocation = pilot.proportional_quotas(
            counts,
            target,
            seed=seed,
            group=group,
        )
        for (disease, capture_date), quota in allocation.items():
            if quota:
                quotas[(*group, disease, capture_date)] = quota
    return quotas


def validate_selected_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    train_target: int,
    validation_target: int,
    validation_dates: Sequence[str],
    train_camera_ids: Sequence[str] = FIXED_TRAIN_CAMERA_IDS,
    validation_camera_ids: Sequence[str] = FIXED_VALIDATION_CAMERA_IDS,
) -> dict[str, Any]:
    dates = set(normalize_validation_dates(validation_dates))
    allowed_train_cameras, allowed_validation_cameras = (
        _validate_camera_partition(
            train_camera_ids,
            validation_camera_ids,
        )
    )
    identities = [smoke.candidate_unique_key(row) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("선택 목록에 중복 이미지가 있습니다")
    if any(row["split"] == "test" for row in rows):
        raise ValueError("Test 이미지는 date-holdout에 사용할 수 없습니다")

    counts: Counter[tuple[str, str, str]] = Counter()
    dates_by_split: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
    }
    cameras_by_split: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
    }
    for row in rows:
        split = row["split"]
        if split not in {"train", "validation"}:
            raise ValueError(f"지원하지 않는 split: {split}")
        _validate_candidate_row(row)
        capture_date = date_audit.canonical_capture_date(
            row["capture_date"]
        )
        camera_id = health.normalized(row["camera_id"])
        dates_by_split[split].add(capture_date)
        cameras_by_split[split].add(camera_id)
        counts[(split, row["species"], row["normality"])] += 1

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
        raise ValueError(f"품종×상태 수량 불일치: {mismatches}")

    date_overlap = dates_by_split["train"] & dates_by_split["validation"]
    if date_overlap:
        raise ValueError(
            "Train/Validation 날짜 중복: " + ", ".join(sorted(date_overlap))
        )
    if dates_by_split["validation"] != dates:
        raise ValueError(
            "Validation 날짜 목록 불일치: "
            f"{sorted(dates_by_split['validation'])} != {sorted(dates)}"
        )
    if dates_by_split["train"] & dates:
        raise ValueError("Train에 고정 Validation 날짜가 포함되었습니다")

    camera_overlap = (
        cameras_by_split["train"] & cameras_by_split["validation"]
    )
    if camera_overlap:
        raise ValueError(
            "Train/Validation global camera_id 중복: "
            + ", ".join(
                sorted(camera_overlap, key=health.natural_text_key)
            )
        )
    if not cameras_by_split["train"] <= set(allowed_train_cameras):
        raise ValueError("Train에 허용되지 않은 global camera_id가 있습니다")
    if not cameras_by_split["validation"] <= set(
        allowed_validation_cameras
    ):
        raise ValueError(
            "Validation에 허용되지 않은 global camera_id가 있습니다"
        )
    return {
        "counts": counts,
        "train_dates": tuple(sorted(dates_by_split["train"])),
        "validation_dates": tuple(sorted(dates_by_split["validation"])),
        "date_overlap": len(date_overlap),
        "train_camera_ids": tuple(
            sorted(
                cameras_by_split["train"],
                key=health.natural_text_key,
            )
        ),
        "validation_camera_ids": tuple(
            sorted(
                cameras_by_split["validation"],
                key=health.natural_text_key,
            )
        ),
        "camera_overlap": len(camera_overlap),
    }


def select_date_holdout_rows(
    manifest_path: Path,
    *,
    train_per_species_status: int,
    validation_per_species_status: int,
    validation_dates: Sequence[str],
    seed: int,
    max_total_images: int,
    train_camera_ids: Sequence[str] = FIXED_TRAIN_CAMERA_IDS,
    validation_camera_ids: Sequence[str] = FIXED_VALIDATION_CAMERA_IDS,
) -> DateHoldoutSelection:
    expected_total = validate_requested_limits(
        train_per_species_status,
        validation_per_species_status,
        max_total_images,
        1,
    )
    dates = normalize_validation_dates(validation_dates)
    train_cameras, validation_cameras = _validate_camera_partition(
        train_camera_ids,
        validation_camera_ids,
    )
    stats = stream_eligible_statistics(
        manifest_path,
        validation_dates=dates,
        train_camera_ids=train_cameras,
        validation_camera_ids=validation_cameras,
    )
    targets = requested_targets(
        train_per_species_status,
        validation_per_species_status,
    )
    quotas = build_stratum_quotas(stats, targets, seed=seed)
    heaps: dict[
        tuple[str, str, str, str, str],
        list[tuple[int, int, DateCandidate]],
    ] = defaultdict(list)
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            if row["split"] == "test":
                continue
            _validate_candidate_row(row)
            new_split = assigned_split(
                row,
                validation_dates=dates,
                train_camera_ids=train_cameras,
                validation_camera_ids=validation_cameras,
            )
            if new_split is None:
                continue
            disease = health.normalized(row["disease_type"]) or "<missing>"
            capture_date = date_audit.canonical_capture_date(
                row["capture_date"]
            )
            stratum = (
                new_split,
                row["species"],
                health.normalized(row["normality"]),
                disease,
                capture_date,
            )
            limit = quotas.get(stratum, 0)
            if limit <= 0:
                continue
            unique_key = smoke.candidate_unique_key(row)
            rank = int(smoke.stable_score(seed, unique_key), 16)
            retained = {
                column: row.get(column, "")
                for column in pilot.PILOT_SOURCE_COLUMNS
            }
            source_split = retained["split"]
            retained["split"] = new_split
            retained["_source_manifest_split"] = source_split
            candidate = DateCandidate(
                rank=rank,
                unique_key=unique_key,
                source_manifest_split=source_split,
                row=retained,
            )
            item = (-rank, line_number, candidate)
            heap = heaps[stratum]
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, item)

    selected: list[dict[str, str]] = []
    for stratum, quota in sorted(quotas.items()):
        heap = heaps.get(stratum, [])
        if len(heap) != quota:
            raise RuntimeError(
                f"{stratum} 선택 수량 {len(heap)}/{quota}"
            )
        selected.extend(item[2].row for item in heap)
    selected.sort(
        key=lambda row: (
            0 if row["split"] == "train" else 1,
            smoke.CLASS_NAMES_INV[row["species"]],
            0 if row["normality"] == "normal" else 1,
            smoke.stable_score(seed, smoke.candidate_unique_key(row)),
        )
    )
    if len(selected) != expected_total:
        raise RuntimeError(
            f"선택 수량 {len(selected)}/{expected_total}"
        )
    validate_selected_rows(
        selected,
        train_target=train_per_species_status,
        validation_target=validation_per_species_status,
        validation_dates=dates,
        train_camera_ids=train_cameras,
        validation_camera_ids=validation_cameras,
    )
    return DateHoldoutSelection(
        statistics=stats,
        selected=selected,
        validation_dates=dates,
        train_camera_ids=train_cameras,
        validation_camera_ids=validation_cameras,
    )


def verify_fixed_audit_plan(
    manifest_path: Path,
    *,
    validation_dates: Sequence[str],
    train_target: int,
    validation_target: int,
    seed: int,
) -> None:
    """Fail closed if the fixed production plan drifts from the audit code."""
    dates = normalize_validation_dates(validation_dates, require_fixed=True)
    stats = date_audit.stream_date_statistics(manifest_path)
    capacity = date_audit.date_capacity(
        stats,
        set(dates),
        train_target=train_target,
        validation_target=validation_target,
    )
    if not capacity.exact_6000:
        raise RuntimeError("고정 날짜 분할이 더 이상 목표 수량을 충족하지 않습니다")
    plan = date_audit.choose_global_camera_partition(
        stats,
        dates,
        train_target=train_target,
        validation_target=validation_target,
        seed=seed,
    )
    if not plan.feasible:
        raise RuntimeError("감사 global camera_id 분할을 재현할 수 없습니다")
    if set(plan.train_camera_ids) != set(FIXED_TRAIN_CAMERA_IDS) or set(
        plan.validation_camera_ids
    ) != set(FIXED_VALIDATION_CAMERA_IDS):
        raise RuntimeError(
            "감사 global camera_id 분할 결과 drift: "
            f"train={plan.train_camera_ids}, "
            f"validation={plan.validation_camera_ids}"
        )


def _prepare_output_transaction(
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


def _activate_output_transaction(
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


def _rollback_output_transaction(
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
        raise RuntimeError("기존 date-holdout output을 복원할 수 없습니다")


def _project_manifest_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {column: row.get(column, "") for column in MANIFEST_COLUMNS}
        for row in rows
    ]


def _validate_serialized_manifest(
    path: Path,
    *,
    expected_rows: int,
    train_target: int,
    validation_target: int,
    validation_dates: Sequence[str],
) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise RuntimeError("직렬화된 manifest 컬럼 불일치")
        rows = list(reader)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"직렬화된 manifest 행 수 {len(rows)}/{expected_rows}"
        )
    validate_selected_rows(
        rows,
        train_target=train_target,
        validation_target=validation_target,
        validation_dates=validation_dates,
    )
    return rows


def validate_date_holdout_output(
    output_root: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    train_target: int,
    validation_target: int,
    validation_dates: Sequence[str],
    max_review_images: int,
    review_rows: Sequence[Mapping[str, Any]],
    contact_sheets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validation = health_pilot.validate_health_output(
        output_root,
        manifest_rows,
        train_target=train_target,
        validation_target=validation_target,
        max_review_images=max_review_images,
        review_rows=review_rows,
        contact_sheets=contact_sheets,
    )
    split = validate_selected_rows(
        manifest_rows,
        train_target=train_target,
        validation_target=validation_target,
        validation_dates=validation_dates,
    )
    validation.update(split)
    if validation["image_count"] != (
        10 * (train_target + validation_target)
    ):
        raise RuntimeError("전체 이미지 수 불일치")
    return validation


def _markdown_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> str:
    return health_pilot.markdown_table(headers, rows)


def write_summary(
    output_root: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    selection: DateHoldoutSelection,
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
        (str(row["split"]), health.normalized(row["disease_type"]))
        for row in manifest_rows
        if row["normality"] == "abnormal"
    )
    source_changes = Counter(
        (
            row["split"],
            row.get("_source_manifest_split", ""),
        )
        for row in selection.selected
    )
    pool_rows = (
        (
            split,
            species,
            "HEALTHY" if normality == "normal" else "DISEASE_SUSPECTED",
            selection.statistics.eligible_counts[
                (split, species, normality)
            ],
            train_target if split == "train" else validation_target,
        )
        for split in ("train", "validation")
        for species in pilot.SPECIES
        for normality in ("normal", "abnormal")
    )
    lines = [
        "# 버섯 건강 체크 date+camera holdout 파일럿",
        "",
        "## 실행 범위",
        "",
        f"- seed: `{seed}`",
        "- split 제약: 전역 `capture_date` + 전역 `camera_id`",
        "- Validation 날짜: "
        + ", ".join(validation["validation_dates"]),
        "- Train global camera_id: "
        + ", ".join(validation["train_camera_ids"]),
        "- Validation global camera_id: "
        + ", ".join(validation["validation_camera_ids"]),
        f"- Train 품종×상태: `{train_target}`장",
        f"- Validation 품종×상태: `{validation_target}`장",
        f"- 이미지 강제 상한: `{max_total_images}`장",
        f"- review 강제 상한: `{max_review_images}`장",
        f"- 원본 이미지 폭·높이 기준 `{padding_ratio:.2f}` padding",
        "- timestamp 마스킹: 적용하지 않음(raw baseline)",
        "- Test 선택·이미지 읽기·추출: 0장",
        "- 전체 ZIP 압축 해제 없음; 선택된 이미지 멤버만 읽음",
        "- 모델 학습 없음",
        "",
        "## 결과",
        "",
        f"- Train: **{validation['train_images']:,}장**",
        f"- Validation: **{validation['validation_images']:,}장**",
        f"- 날짜 중복: **{validation['date_overlap']:,}개**",
        f"- global camera_id 중복: "
        f"**{validation['camera_overlap']:,}개**",
        f"- Test: **{validation['test_images']:,}장**",
        f"- 누락/손상/빈 crop: **{validation['missing_images']:,}/"
        f"{validation['corrupt_images']:,}/"
        f"{validation['empty_crops']:,}개**",
        f"- 범위 밖 crop: **{validation['invalid_crop_rects']:,}개**",
        f"- 출력 파일명 충돌: "
        f"**{validation['output_filename_collisions']:,}개**",
        "- 원본 ZIP 변경: **없음**",
        "- date-holdout 외 기존 artifacts 변경: **없음**",
        "",
        "## 날짜",
        "",
        "- Train: " + ", ".join(validation["train_dates"]),
        "- Validation: " + ", ".join(validation["validation_dates"]),
        "",
        "## global camera_id",
        "",
        "- Train: " + ", ".join(validation["train_camera_ids"]),
        "- Validation: " + ", ".join(
            validation["validation_camera_ids"]
        ),
        "",
        "## strict pool과 선택 수량",
        "",
        _markdown_table(
            ("split", "품종", "상태", "가용", "선택"),
            pool_rows,
        ),
        "",
        "## 품종×상태",
        "",
        _markdown_table(
            ("split", "품종", "상태", "이미지"),
            (
                (
                    split,
                    species,
                    (
                        "HEALTHY"
                        if normality == "normal"
                        else "DISEASE_SUSPECTED"
                    ),
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
        _markdown_table(
            ("split", "병해 종류", "이미지"),
            (
                (split, disease_type, count)
                for (split, disease_type), count in sorted(disease.items())
            ),
        ),
        "",
        "## 기존 detection split 변경",
        "",
        _markdown_table(
            ("새 split", "기존 split", "이미지"),
            (
                (new_split, old_split, count)
                for (new_split, old_split), count in sorted(
                    source_changes.items()
                )
            ),
        ),
        "",
        "## 스트리밍 선택",
        "",
        f"- Test 조기 제외: "
        f"{selection.statistics.skipped_test_rows:,}행",
        f"- strict date∩camera 후보: "
        f"{selection.statistics.eligible_rows:,}행",
        f"- 반대 date×camera 교차 조합 제외: "
        f"{selection.statistics.excluded_cross_date_camera_rows:,}행",
        f"- 미할당 camera 제외: "
        f"{selection.statistics.excluded_unknown_camera_rows:,}행",
        "",
        "## crop 크기 분포",
        "",
        _markdown_table(
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
            health_pilot.crop_distribution_rows(manifest_rows),
        ),
        "",
        f"- review 표본: {validation['review_count']:,}장",
        f"- contact sheet: {validation['contact_sheet_count']:,}개",
        "",
        "## 제한 사항",
        "",
        "- 이 데이터셋은 이진 HEALTHY/DISEASE_SUSPECTED 평가용이다.",
        "- 일부 품종×병해 종류가 Validation에 없을 수 있다.",
        "- 병해 종류 모델 평가용 데이터셋으로 해석하지 않는다.",
        "- 기존 고정 Test와 날짜가 겹치므로 최종 3-way 독립 평가가 아니다.",
        "- 외부 스마트폰 Test 전에는 서비스 일반화 성능으로 인정하지 않는다.",
        "- timestamp 마스킹을 적용하지 않은 date-holdout raw baseline이다.",
        "",
        "## 안전 검증",
        "",
        "- 선택 ZIP 멤버 외 이미지 바이트 읽기: 0장",
        "- 전체 ZIP 압축 해제: 없음",
        "- 로컬 절대경로 노출: 0개",
        "- 원본 ZIP 크기/mtime 변경: 0개",
        "- 기존 artifacts 변경: 0개",
        "- 모델 학습: 실행하지 않음",
        "",
    ]
    smoke.write_text(
        output_root / "health_date_holdout_summary.md",
        "\n".join(lines),
    )


def assert_deidentified_output(output_root: Path) -> None:
    health_pilot.assert_deidentified_output(output_root)


def create_date_holdout_dataset(
    selection: DateHoldoutSelection,
    archive_lookup: Mapping[str, base.ArchiveRef],
    output_dir: Path,
    *,
    train_per_species_status: int,
    validation_per_species_status: int,
    validation_dates: Sequence[str],
    padding_ratio: float,
    seed: int,
    max_total_images: int,
    max_review_images: int,
    overwrite: bool,
    artifacts_root: Path = ARTIFACTS_ROOT,
    show_progress: bool = True,
) -> dict[str, Any]:
    validate_output_location(
        output_dir,
        artifacts_root=artifacts_root,
    )
    expected_total = validate_requested_limits(
        train_per_species_status,
        validation_per_species_status,
        max_total_images,
        max_review_images,
    )
    dates = normalize_validation_dates(validation_dates)
    if tuple(selection.validation_dates) != dates:
        raise ValueError("selection과 요청 Validation 날짜가 다릅니다")
    if len(selection.selected) != expected_total:
        raise ValueError(
            f"선택 이미지 {len(selection.selected)}/{expected_total}"
        )
    validate_selected_rows(
        selection.selected,
        train_target=train_per_species_status,
        validation_target=validation_per_species_status,
        validation_dates=dates,
        train_camera_ids=selection.train_camera_ids,
        validation_camera_ids=selection.validation_camera_ids,
    )
    plans = health_pilot.build_crop_plans(
        selection.selected,
        padding_ratio=padding_ratio,
    )
    review_keys = health_pilot.select_review_keys(
        plans,
        seed=seed,
        max_review_images=max_review_images,
    )
    source_snapshot = label_audit.source_zip_snapshot(
        ref.path for ref in archive_lookup.values()
    )
    artifacts_snapshot = health_pilot.snapshot_artifacts_excluding(
        artifacts_root,
        output_dir,
    )
    staging, output_existed = _prepare_output_transaction(
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
            "review/train",
            "review/val",
            "review/contact_sheets",
        ):
            (staging / relative).mkdir(parents=True, exist_ok=True)
        health_pilot.write_class_mapping(staging)
        internal_rows, review_rows = health_pilot.extract_health_crops(
            plans,
            archive_lookup,
            staging,
            review_keys=review_keys,
            padding_ratio=padding_ratio,
            show_progress=show_progress,
        )
        contact_sheets = health_pilot.create_contact_sheets(
            staging,
            review_rows,
        )
        validation = validate_date_holdout_output(
            staging,
            internal_rows,
            train_target=train_per_species_status,
            validation_target=validation_per_species_status,
            validation_dates=dates,
            max_review_images=max_review_images,
            review_rows=review_rows,
            contact_sheets=contact_sheets,
        )
        manifest_rows = _project_manifest_rows(internal_rows)
        manifest_path = staging / "health_date_holdout_manifest.csv"
        smoke.write_csv(
            manifest_path,
            MANIFEST_COLUMNS,
            manifest_rows,
        )
        _validate_serialized_manifest(
            manifest_path,
            expected_rows=expected_total,
            train_target=train_per_species_status,
            validation_target=validation_per_species_status,
            validation_dates=dates,
        )
        # Individual review renders are temporary contact-sheet tiles.
        shutil.rmtree(staging / "review" / "train")
        shutil.rmtree(staging / "review" / "val")
        write_summary(
            staging,
            internal_rows,
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
        health_pilot.assert_artifacts_unchanged(
            artifacts_root,
            output_dir,
            artifacts_snapshot,
        )
        backup = _activate_output_transaction(
            staging,
            output_dir,
            output_existed=output_existed,
        )
        activated = True
        health_pilot.assert_artifacts_unchanged(
            artifacts_root,
            output_dir,
            artifacts_snapshot,
        )
    except Exception:
        _rollback_output_transaction(
            staging,
            output_dir,
            backup,
            activated=activated,
            output_existed=output_existed,
        )
        label_audit.assert_source_zips_unchanged(source_snapshot)
        health_pilot.assert_artifacts_unchanged(
            artifacts_root,
            output_dir,
            artifacts_snapshot,
        )
        raise
    if backup is not None:
        shutil.rmtree(backup)

    species_status = Counter(
        (
            str(row["split"]),
            str(row["species"]),
            str(row["normality"]),
        )
        for row in internal_rows
    )
    disease = Counter(
        (str(row["split"]), health.normalized(row["disease_type"]))
        for row in internal_rows
        if row["normality"] == "abnormal"
    )
    return {
        "total_images": validation["image_count"],
        "train_images": validation["train_images"],
        "validation_images": validation["validation_images"],
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
        "train_dates": list(validation["train_dates"]),
        "validation_dates": list(validation["validation_dates"]),
        "date_overlap": validation["date_overlap"],
        "train_camera_ids": list(validation["train_camera_ids"]),
        "validation_camera_ids": list(
            validation["validation_camera_ids"]
        ),
        "camera_overlap": validation["camera_overlap"],
        "review_count": validation["review_count"],
        "contact_sheet_count": validation["contact_sheet_count"],
        "missing_images": validation["missing_images"],
        "corrupt_images": validation["corrupt_images"],
        "empty_crops": validation["empty_crops"],
        "invalid_crop_rects": validation["invalid_crop_rects"],
        "output_filename_collisions": validation[
            "output_filename_collisions"
        ],
        "test_images": validation["test_images"],
        "source_zip_modified": False,
        "protected_artifacts_modified": False,
    }


def run(
    args: argparse.Namespace,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    if args.seed != DEFAULT_SEED:
        raise ValueError(
            f"감사에서 확정한 seed {DEFAULT_SEED}만 사용할 수 있습니다"
        )
    manifest_snapshot = health.snapshot_files((args.manifest,))
    try:
        verify_fixed_audit_plan(
            args.manifest,
            validation_dates=args.validation_dates,
            train_target=args.train_per_species_status,
            validation_target=args.validation_per_species_status,
            seed=args.seed,
        )
        selection = select_date_holdout_rows(
            args.manifest,
            train_per_species_status=args.train_per_species_status,
            validation_per_species_status=(
                args.validation_per_species_status
            ),
            validation_dates=args.validation_dates,
            seed=args.seed,
            max_total_images=args.max_total_images,
        )
        refs, issues, _directories = base.discover_archives(environ)
        if issues:
            raise RuntimeError(
                "아카이브 환경/구조 검증 실패:\n- "
                + "\n- ".join(issues)
            )
        lookup = smoke.image_archive_lookup(refs)
        result = create_date_holdout_dataset(
            selection,
            lookup,
            args.output_dir,
            train_per_species_status=args.train_per_species_status,
            validation_per_species_status=(
                args.validation_per_species_status
            ),
            validation_dates=args.validation_dates,
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
            "test_split_used": False,
            "full_archive_extraction": False,
            "timestamp_masking": False,
            "model_training": False,
            "input_manifest_modified": False,
            "existing_artifacts_modified": False,
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
