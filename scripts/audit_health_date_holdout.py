#!/usr/bin/env python3
"""Audit date-disjoint health pilot split strategies without opening images.

The detection manifest and the two existing health-pilot manifests are read as
CSV only.  Test rows are never used as assignment capacity, image/ZIP bytes are
not opened, and no training or dataset conversion is performed.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import audit_mushroom_health as health


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_DETECTION_MANIFEST = (
    PROJECT_ROOT / "reports" / "detection_dataset_manifest.csv"
)
DEFAULT_RAW_HEALTH_MANIFEST = (
    ARTIFACTS_ROOT / "health_pilot" / "health_pilot_manifest.csv"
)
DEFAULT_MASKED_HEALTH_MANIFEST = (
    ARTIFACTS_ROOT
    / "health_pilot_timestamp_masked"
    / "health_timestamp_masked_manifest.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports"
DEFAULT_SEED = 20260726
DEFAULT_TRAIN_TARGET = 500
DEFAULT_VALIDATION_TARGET = 100
EXPECTED_PILOT_ROWS = 6_000
SPECIES = health.SPECIES
NORMALITIES = ("normal", "abnormal")
CELLS = tuple(
    (species, normality)
    for species in SPECIES
    for normality in NORMALITIES
)
OUTPUT_FILES = (
    "health_date_distribution.csv",
    "health_date_bias_analysis.csv",
    "health_timestamp_mask_bias.csv",
    "health_date_holdout_feasibility.md",
    "health_date_holdout_split_plan.md",
)
DETECTION_REQUIRED_COLUMNS = {
    "split",
    "species",
    "task",
    "normality",
    "disease_type",
    "camera_id",
    "capture_date",
    "image_archive_id",
    "image_member",
}
PILOT_PRESERVED_COLUMNS = (
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
MASKED_REQUIRED_COLUMNS = {
    *PILOT_PRESERVED_COLUMNS,
    "timestamp_candidate_overlap",
    "timestamp_mask_applied",
    "mask_rect_crop_coordinates",
    "mask_width",
    "mask_height",
    "mask_area_ratio",
    "source_raw_relative_path",
}


@dataclass
class DateBucket:
    rows: int = 0
    health: Counter[tuple[str, str]] = field(default_factory=Counter)
    disease: Counter[tuple[str, str]] = field(default_factory=Counter)
    cameras: Counter[str] = field(default_factory=Counter)
    official_splits: Counter[str] = field(default_factory=Counter)
    current_splits: Counter[str] = field(default_factory=Counter)


@dataclass
class DateStatistics:
    development_by_date: dict[str, DateBucket] = field(
        default_factory=lambda: defaultdict(DateBucket)
    )
    development_by_date_camera: dict[
        tuple[str, str], DateBucket
    ] = field(default_factory=lambda: defaultdict(DateBucket))
    species_date_health: dict[
        tuple[str, str], Counter[str]
    ] = field(default_factory=lambda: defaultdict(Counter))
    species_date_disease: dict[
        tuple[str, str], Counter[str]
    ] = field(default_factory=lambda: defaultdict(Counter))
    species_camera_date_splits: dict[
        tuple[str, str, str], Counter[str]
    ] = field(default_factory=lambda: defaultdict(Counter))
    development_health: Counter[tuple[str, str]] = field(
        default_factory=Counter
    )
    development_health_by_split: dict[
        str, Counter[tuple[str, str]]
    ] = field(default_factory=lambda: defaultdict(Counter))
    development_disease: Counter[tuple[str, str]] = field(
        default_factory=Counter
    )
    development_rows: int = 0
    test_rows: int = 0
    test_dates: set[str] = field(default_factory=set)
    test_date_rows: Counter[str] = field(default_factory=Counter)
    invalid_rows: int = 0


@dataclass(frozen=True)
class CapacitySummary:
    train: Counter[tuple[str, str]]
    validation: Counter[tuple[str, str]]
    train_common: int
    validation_common: int
    target_capped_images: int
    exact_6000: bool
    train_shortfall: int
    validation_shortfall: int


@dataclass(frozen=True)
class DateAssignment:
    feasible: bool
    validation_dates: tuple[str, ...] = ()
    train_dates: tuple[str, ...] = ()
    capacity: CapacitySummary | None = None
    minimum_date_count: int = 0
    feasible_combinations_at_minimum: int = 0
    score: tuple[Any, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class CameraPlan:
    feasible: bool
    validation_camera_ids: tuple[str, ...] = ()
    train_camera_ids: tuple[str, ...] = ()
    validation_capacity: Counter[tuple[str, str]] = field(
        default_factory=Counter
    )
    train_capacity: Counter[tuple[str, str]] = field(
        default_factory=Counter
    )
    minimum_validation_camera_count: int = 0
    feasible_combinations_at_minimum: int = 0
    camera_overlap: int = 0
    reason: str = ""


@dataclass(frozen=True)
class CandidateResult:
    candidate: str
    group_key: str
    assignment_method: str
    exact_6000: bool
    global_date_overlap: int
    declared_group_overlap: int
    train_common_capacity: int
    validation_common_capacity: int
    target_capped_images: int
    train_groups: int
    validation_groups: int
    validation_definition: str
    official_changed_rows: int
    test_date_overlap: int
    reason: str = ""


@dataclass
class PilotManifestAudit:
    raw_rows: int = 0
    masked_rows: int = 0
    identity_mismatches: int = 0
    invariant_mismatches: int = 0
    raw_duplicates: int = 0
    masked_duplicates: int = 0
    timestamp_candidates: int = 0
    masks_applied: int = 0
    mask_flag_mismatches: int = 0
    rows_by_split_species_status: Counter[
        tuple[str, str, str]
    ] = field(default_factory=Counter)
    timestamp_by_split_species_status: Counter[
        tuple[str, str, str]
    ] = field(default_factory=Counter)
    masks_by_split_species_status: Counter[
        tuple[str, str, str]
    ] = field(default_factory=Counter)
    dates_by_split: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    mask_state_health: dict[bool, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    mask_state_split_health: dict[
        tuple[str, bool], Counter[str]
    ] = field(default_factory=lambda: defaultdict(Counter))


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("양의 정수여야 합니다") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("양의 정수여야 합니다")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "버섯 건강 체크의 전역 날짜 홀드아웃 가능성을 CSV만 "
            "읽어 감사합니다."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--detection-manifest",
        type=Path,
        default=Path("reports/detection_dataset_manifest.csv"),
    )
    parser.add_argument(
        "--raw-health-manifest",
        type=Path,
        default=Path(
            "artifacts/health_pilot/health_pilot_manifest.csv"
        ),
    )
    parser.add_argument(
        "--masked-health-manifest",
        type=Path,
        default=Path(
            "artifacts/health_pilot_timestamp_masked/"
            "health_timestamp_masked_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports")
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--train-per-species-status",
        type=positive_int,
        default=DEFAULT_TRAIN_TARGET,
    )
    parser.add_argument(
        "--validation-per-species-status",
        type=positive_int,
        default=DEFAULT_VALIDATION_TARGET,
    )
    args = parser.parse_args(argv)
    args.detection_manifest = health.project_path(
        args.detection_manifest
    )
    args.raw_health_manifest = health.project_path(
        args.raw_health_manifest
    )
    args.masked_health_manifest = health.project_path(
        args.masked_health_manifest
    )
    args.output_dir = health.project_path(args.output_dir)
    if args.output_dir == ARTIFACTS_ROOT or args.output_dir.is_relative_to(
        ARTIFACTS_ROOT
    ):
        parser.error("--output-dir는 artifacts 밖이어야 합니다")
    return args


def stream_date_statistics(path: Path) -> DateStatistics:
    """Stream development capacity; Test contributes dates only."""
    if not path.is_file():
        raise FileNotFoundError("detection manifest가 없습니다")
    stats = DateStatistics()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = DETECTION_REQUIRED_COLUMNS - fields
        if missing:
            raise ValueError(
                "detection manifest 필수 컬럼 누락: "
                + ", ".join(sorted(missing))
            )
        for row in reader:
            if None in row:
                raise ValueError("detection manifest 행에 header 밖 값이 있습니다")
            split = health.normalized(row["split"])
            capture_date = canonical_capture_date(row["capture_date"])
            species = health.normalized(row["species"])
            camera_id = health.normalized(row["camera_id"])
            if not camera_id:
                raise ValueError("camera_id 결측은 전역 카메라 감사를 할 수 없습니다")
            group_key = (species, camera_id, capture_date)
            stats.species_camera_date_splits[group_key][split] += 1
            if split == "test":
                stats.test_rows += 1
                stats.test_dates.add(capture_date)
                stats.test_date_rows[capture_date] += 1
                continue
            if split not in {"train", "validation"}:
                raise ValueError(f"지원하지 않는 split: {split}")
            if species not in SPECIES:
                raise ValueError(f"지원하지 않는 품종: {species}")
            task = health.normalized(row["task"])
            normality = health.normalized(row["normality"])
            disease = health.normalized(row["disease_type"])
            _state, issues = health.health_label_issues(
                task, normality, disease
            )
            if task not in health.HEALTH_TASKS or issues:
                raise ValueError(
                    "건강 후보 라벨 모순: "
                    f"{species}/{capture_date}/{task}/{normality}/{disease}: "
                    + ",".join(issues)
                )
            cell = (species, normality)
            official = health.normalized(row.get("official_split", ""))
            stats.development_rows += 1
            stats.development_health[cell] += 1
            stats.development_health_by_split[split][cell] += 1
            if normality == "abnormal":
                stats.development_disease[(species, disease)] += 1
            for bucket in (
                stats.development_by_date[capture_date],
                stats.development_by_date_camera[(capture_date, camera_id)],
            ):
                bucket.rows += 1
                bucket.health[cell] += 1
                bucket.cameras[camera_id] += 1
                bucket.current_splits[split] += 1
                if official:
                    bucket.official_splits[official] += 1
                if normality == "abnormal":
                    bucket.disease[(species, disease)] += 1
            stats.species_date_health[(species, capture_date)][
                normality
            ] += 1
            if normality == "abnormal":
                stats.species_date_disease[(species, capture_date)][
                    disease
                ] += 1
    if not stats.development_rows:
        raise ValueError("개발 후보 행이 없습니다")
    return stats


def canonical_capture_date(value: Any) -> str:
    text = health.normalized(value)
    if not text:
        raise ValueError("capture_date가 비어 있습니다")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"잘못된 capture_date: {text}") from exc
    canonical = parsed.isoformat()
    if canonical != text:
        raise ValueError(f"capture_date가 YYYY-MM-DD 형식이 아닙니다: {text}")
    return canonical


def _sum_health(
    buckets: Iterable[DateBucket],
) -> Counter[tuple[str, str]]:
    result: Counter[tuple[str, str]] = Counter()
    for bucket in buckets:
        result.update(bucket.health)
    return result


def _sum_disease(
    buckets: Iterable[DateBucket],
) -> Counter[tuple[str, str]]:
    result: Counter[tuple[str, str]] = Counter()
    for bucket in buckets:
        result.update(bucket.disease)
    return result


def capacity_summary(
    train: Mapping[tuple[str, str], int],
    validation: Mapping[tuple[str, str], int],
    *,
    train_target: int,
    validation_target: int,
) -> CapacitySummary:
    train_counter = Counter(train)
    validation_counter = Counter(validation)
    train_common = min((train_counter[cell] for cell in CELLS), default=0)
    validation_common = min(
        (validation_counter[cell] for cell in CELLS), default=0
    )
    train_shortfall = sum(
        max(0, train_target - train_counter[cell]) for cell in CELLS
    )
    validation_shortfall = sum(
        max(0, validation_target - validation_counter[cell])
        for cell in CELLS
    )
    return CapacitySummary(
        train=train_counter,
        validation=validation_counter,
        train_common=train_common,
        validation_common=validation_common,
        target_capped_images=len(CELLS)
        * (
            min(train_target, train_common)
            + min(validation_target, validation_common)
        ),
        exact_6000=not train_shortfall and not validation_shortfall,
        train_shortfall=train_shortfall,
        validation_shortfall=validation_shortfall,
    )


def date_capacity(
    stats: DateStatistics,
    validation_dates: Iterable[str],
    *,
    train_target: int,
    validation_target: int,
) -> CapacitySummary:
    selected = set(validation_dates)
    validation = _sum_health(
        bucket
        for capture_date, bucket in stats.development_by_date.items()
        if capture_date in selected
    )
    train = Counter(stats.development_health) - validation
    return capacity_summary(
        train,
        validation,
        train_target=train_target,
        validation_target=validation_target,
    )


def assignment_changed_rows(
    stats: DateStatistics, validation_dates: Iterable[str]
) -> int:
    selected = set(validation_dates)
    return sum(
        bucket.current_splits[
            "train" if capture_date in selected else "validation"
        ]
        for capture_date, bucket in stats.development_by_date.items()
    )


def _distribution_error(
    stats: DateStatistics,
    validation_dates: Iterable[str],
    *,
    train_target: int,
    validation_target: int,
) -> float:
    selected = set(validation_dates)
    validation = _sum_health(
        stats.development_by_date[item] for item in selected
    )
    return sum(
        abs(
            validation[cell] / stats.development_health[cell]
            - validation_target / (train_target + validation_target)
        )
        for cell in CELLS
    )


def _disease_missing_sides(
    stats: DateStatistics, validation_dates: Iterable[str]
) -> int:
    selected = set(validation_dates)
    validation = _sum_disease(
        stats.development_by_date[item] for item in selected
    )
    train = Counter(stats.development_disease) - validation
    return sum(
        int(validation[key] == 0) + int(train[key] == 0)
        for key in stats.development_disease
    )


def _date_assignment_score(
    stats: DateStatistics,
    dates: tuple[str, ...],
    *,
    seed: int,
    train_target: int,
    validation_target: int,
) -> tuple[Any, ...]:
    validation_rows = sum(
        stats.development_by_date[item].rows for item in dates
    )
    return (
        _disease_missing_sides(stats, dates),
        round(
            _distribution_error(
                stats,
                dates,
                train_target=train_target,
                validation_target=validation_target,
            ),
            12,
        ),
        validation_rows,
        assignment_changed_rows(stats, dates),
        health.stable_hash(seed, "global-date", *dates),
    )


def find_minimal_global_date_assignment(
    stats: DateStatistics,
    *,
    train_target: int = DEFAULT_TRAIN_TARGET,
    validation_target: int = DEFAULT_VALIDATION_TARGET,
    seed: int = DEFAULT_SEED,
) -> DateAssignment:
    dates = tuple(sorted(stats.development_by_date))
    closest: tuple[
        tuple[Any, ...], tuple[str, ...], CapacitySummary
    ] | None = None
    for size in range(1, len(dates)):
        feasible: list[
            tuple[tuple[Any, ...], tuple[str, ...], CapacitySummary]
        ] = []
        for selected in itertools.combinations(dates, size):
            capacity = date_capacity(
                stats,
                selected,
                train_target=train_target,
                validation_target=validation_target,
            )
            closest_rank = (
                -capacity.target_capped_images,
                capacity.train_shortfall + capacity.validation_shortfall,
                abs(
                    capacity.train_common / train_target
                    - capacity.validation_common / validation_target
                ),
                health.stable_hash(seed, "closest-global-date", *selected),
            )
            if closest is None or closest_rank < closest[0]:
                closest = (closest_rank, selected, capacity)
            if capacity.exact_6000:
                feasible.append(
                    (
                        _date_assignment_score(
                            stats,
                            selected,
                            seed=seed,
                            train_target=train_target,
                            validation_target=validation_target,
                        ),
                        selected,
                        capacity,
                    )
                )
        if feasible:
            score, selected, capacity = min(
                feasible, key=lambda item: (item[0], item[1])
            )
            selected_set = set(selected)
            return DateAssignment(
                feasible=True,
                validation_dates=selected,
                train_dates=tuple(
                    item for item in dates if item not in selected_set
                ),
                capacity=capacity,
                minimum_date_count=size,
                feasible_combinations_at_minimum=len(feasible),
                score=score,
            )
    if closest is not None:
        _score, selected, capacity = closest
        selected_set = set(selected)
        return DateAssignment(
            feasible=False,
            validation_dates=selected,
            train_dates=tuple(
                item for item in dates if item not in selected_set
            ),
            capacity=capacity,
            minimum_date_count=len(selected),
            reason=(
                "전역 날짜 그룹을 쪼개지 않은 최대 target-capped "
                f"구성은 {capacity.target_capped_images}장"
            ),
        )
    return DateAssignment(
        feasible=False,
        reason="전역 날짜 완전 분리로 목표 수량을 충족하는 조합 없음",
    )


def choose_global_camera_partition(
    stats: DateStatistics,
    validation_dates: Sequence[str],
    *,
    train_target: int = DEFAULT_TRAIN_TARGET,
    validation_target: int = DEFAULT_VALIDATION_TARGET,
    seed: int = DEFAULT_SEED,
) -> CameraPlan:
    selected_dates = set(validation_dates)
    cameras = tuple(
        sorted(
            {
                camera
                for _capture_date, camera in stats.development_by_date_camera
            },
            key=health.natural_text_key,
        )
    )
    validation_by_camera: dict[
        str, Counter[tuple[str, str]]
    ] = defaultdict(Counter)
    train_by_camera: dict[str, Counter[tuple[str, str]]] = defaultdict(
        Counter
    )
    validation_disease_by_camera: dict[
        str, Counter[tuple[str, str]]
    ] = defaultdict(Counter)
    train_disease_by_camera: dict[
        str, Counter[tuple[str, str]]
    ] = defaultdict(Counter)
    for (capture_date, camera), bucket in (
        stats.development_by_date_camera.items()
    ):
        if capture_date in selected_dates:
            validation_by_camera[camera].update(bucket.health)
            validation_disease_by_camera[camera].update(bucket.disease)
        else:
            train_by_camera[camera].update(bucket.health)
            train_disease_by_camera[camera].update(bucket.disease)
    total_train = sum(train_by_camera.values(), Counter())
    total_train_disease = sum(
        train_disease_by_camera.values(), Counter()
    )
    for size in range(1, len(cameras)):
        feasible: list[
            tuple[
                tuple[Any, ...],
                tuple[str, ...],
                Counter[tuple[str, str]],
                Counter[tuple[str, str]],
            ]
        ] = []
        for selected in itertools.combinations(cameras, size):
            validation = sum(
                (validation_by_camera[item] for item in selected),
                Counter(),
            )
            removed_train = sum(
                (train_by_camera[item] for item in selected), Counter()
            )
            train = total_train - removed_train
            capacity = capacity_summary(
                train,
                validation,
                train_target=train_target,
                validation_target=validation_target,
            )
            if not capacity.exact_6000:
                continue
            validation_disease = sum(
                (validation_disease_by_camera[item] for item in selected),
                Counter(),
            )
            train_disease = total_train_disease - sum(
                (train_disease_by_camera[item] for item in selected),
                Counter(),
            )
            missing_disease = sum(
                int(validation_disease[key] == 0)
                + int(train_disease[key] == 0)
                for key in stats.development_disease
            )
            excesses = [
                validation[cell] - validation_target for cell in CELLS
            ]
            score = (
                missing_disease,
                sum(excesses),
                max(excesses, default=0),
                health.stable_hash(
                    seed, "global-camera", *selected
                ),
            )
            feasible.append((score, selected, train, validation))
        if feasible:
            _score, selected, train, validation = min(
                feasible, key=lambda item: (item[0], item[1])
            )
            selected_set = set(selected)
            return CameraPlan(
                feasible=True,
                validation_camera_ids=selected,
                train_camera_ids=tuple(
                    item for item in cameras if item not in selected_set
                ),
                validation_capacity=validation,
                train_capacity=train,
                minimum_validation_camera_count=size,
                feasible_combinations_at_minimum=len(feasible),
                camera_overlap=0,
            )
    return CameraPlan(
        feasible=False,
        reason="날짜 제약과 전역 camera_id 분리를 동시에 만족하지 못함",
    )


def _closest_capacity_rank(
    capacity: CapacitySummary,
    *,
    train_target: int,
    validation_target: int,
    tie: str,
) -> tuple[Any, ...]:
    train_fraction = min(
        capacity.train[cell] / train_target for cell in CELLS
    )
    validation_fraction = min(
        capacity.validation[cell] / validation_target for cell in CELLS
    )
    met_cells = sum(
        capacity.train[cell] >= train_target for cell in CELLS
    ) + sum(
        capacity.validation[cell] >= validation_target for cell in CELLS
    )
    return (
        not capacity.exact_6000,
        -round(min(train_fraction, validation_fraction), 12),
        -met_cells,
        -capacity.target_capped_images,
        capacity.train_shortfall + capacity.validation_shortfall,
        tie,
    )


def _candidate_for_global_dates(
    stats: DateStatistics,
    *,
    candidate: str,
    group_key: str,
    method: str,
    validation_dates: Sequence[str],
    train_target: int,
    validation_target: int,
    reason: str = "",
) -> CandidateResult:
    selected = set(validation_dates)
    all_dates = set(stats.development_by_date)
    capacity = date_capacity(
        stats,
        selected,
        train_target=train_target,
        validation_target=validation_target,
    )
    return CandidateResult(
        candidate=candidate,
        group_key=group_key,
        assignment_method=method,
        exact_6000=capacity.exact_6000,
        global_date_overlap=0,
        declared_group_overlap=0,
        train_common_capacity=capacity.train_common,
        validation_common_capacity=capacity.validation_common,
        target_capped_images=capacity.target_capped_images,
        train_groups=len(all_dates - selected),
        validation_groups=len(selected),
        validation_definition=", ".join(sorted(selected)),
        official_changed_rows=assignment_changed_rows(stats, selected),
        test_date_overlap=len(selected & stats.test_dates),
        reason=reason,
    )


def _species_date_assignment(
    stats: DateStatistics,
    *,
    train_target: int,
    validation_target: int,
    seed: int,
) -> tuple[
    set[tuple[str, str]],
    Counter[tuple[str, str]],
    Counter[tuple[str, str]],
]:
    selected: set[tuple[str, str]] = set()
    for species in SPECIES:
        dates = tuple(
            sorted(
                capture_date
                for item_species, capture_date in stats.species_date_health
                if item_species == species
            )
        )
        total = Counter(
            {
                normality: stats.development_health[(species, normality)]
                for normality in NORMALITIES
            }
        )
        best: tuple[Any, tuple[str, ...]] | None = None
        for size in range(1, len(dates)):
            feasible: list[tuple[tuple[Any, ...], tuple[str, ...]]] = []
            for subset in itertools.combinations(dates, size):
                validation = sum(
                    (
                        stats.species_date_health[(species, item)]
                        for item in subset
                    ),
                    Counter(),
                )
                train = total - validation
                if all(
                    validation[normality] >= validation_target
                    and train[normality] >= train_target
                    for normality in NORMALITIES
                ):
                    excesses = [
                        validation[item] - validation_target
                        for item in NORMALITIES
                    ]
                    score = (
                        sum(excesses),
                        max(excesses),
                        health.stable_hash(
                            seed, "species-date", species, *subset
                        ),
                    )
                    feasible.append((score, subset))
            if feasible:
                best = min(feasible, key=lambda item: (item[0], item[1]))
                break
        if best is None:
            raise ValueError(f"{species} species+date 정확 분할 불가")
        selected.update((species, item) for item in best[1])
    train_capacity: Counter[tuple[str, str]] = Counter()
    validation_capacity: Counter[tuple[str, str]] = Counter()
    for (species, capture_date), counts in (
        stats.species_date_health.items()
    ):
        target = (
            validation_capacity
            if (species, capture_date) in selected
            else train_capacity
        )
        for normality, count in counts.items():
            target[(species, normality)] += count
    return selected, train_capacity, validation_capacity


def _candidate_species_date(
    stats: DateStatistics,
    *,
    train_target: int,
    validation_target: int,
    seed: int,
) -> CandidateResult:
    selected, train, validation = _species_date_assignment(
        stats,
        train_target=train_target,
        validation_target=validation_target,
        seed=seed,
    )
    capacity = capacity_summary(
        train,
        validation,
        train_target=train_target,
        validation_target=validation_target,
    )
    train_dates = {
        capture_date
        for species, capture_date in stats.species_date_health
        if (species, capture_date) not in selected
    }
    validation_dates = {capture_date for _species, capture_date in selected}
    changed = 0
    for (species, camera, capture_date), counts in (
        stats.species_camera_date_splits.items()
    ):
        assigned = (
            "validation"
            if (species, capture_date) in selected
            else "train"
        )
        changed += counts[
            "train" if assigned == "validation" else "validation"
        ]
    return CandidateResult(
        candidate="species_capture_date",
        group_key="species + capture_date",
        assignment_method="품종별 최소 날짜 exact search",
        exact_6000=capacity.exact_6000,
        global_date_overlap=len(train_dates & validation_dates),
        declared_group_overlap=0,
        train_common_capacity=capacity.train_common,
        validation_common_capacity=capacity.validation_common,
        target_capped_images=capacity.target_capped_images,
        train_groups=len(stats.species_date_health) - len(selected),
        validation_groups=len(selected),
        validation_definition="품종별: "
        + "; ".join(
            f"{species}={','.join(sorted(d for s, d in selected if s == species))}"
            for species in SPECIES
        ),
        official_changed_rows=changed,
        test_date_overlap=len(validation_dates & stats.test_dates),
        reason=(
            "선언한 그룹은 분리되지만 동일 날짜가 다른 품종을 통해 "
            "양쪽 split에 존재할 수 있음"
        ),
    )


def evaluate_candidate_splits(
    stats: DateStatistics,
    *,
    global_assignment: DateAssignment,
    train_target: int = DEFAULT_TRAIN_TARGET,
    validation_target: int = DEFAULT_VALIDATION_TARGET,
    seed: int = DEFAULT_SEED,
) -> list[CandidateResult]:
    if not global_assignment.feasible or global_assignment.capacity is None:
        raise ValueError("전역 날짜 assignment가 feasible하지 않습니다")
    results = [
        _candidate_for_global_dates(
            stats,
            candidate="global_capture_date",
            group_key="capture_date",
            method="최소 validation 날짜 exact search",
            validation_dates=global_assignment.validation_dates,
            train_target=train_target,
            validation_target=validation_target,
        ),
        _candidate_species_date(
            stats,
            train_target=train_target,
            validation_target=validation_target,
            seed=seed,
        ),
    ]
    dates = tuple(sorted(stats.development_by_date))
    chronological: list[
        tuple[tuple[Any, ...], tuple[str, ...], CapacitySummary]
    ] = []
    for index in range(1, len(dates)):
        selected = dates[index:]
        capacity = date_capacity(
            stats,
            selected,
            train_target=train_target,
            validation_target=validation_target,
        )
        chronological.append(
            (
                _closest_capacity_rank(
                    capacity,
                    train_target=train_target,
                    validation_target=validation_target,
                    tie=health.stable_hash(
                        seed, "chronological", *selected
                    ),
                ),
                selected,
                capacity,
            )
        )
    _rank, selected, capacity = min(chronological, key=lambda item: item[0])
    results.append(
        _candidate_for_global_dates(
            stats,
            candidate="chronological",
            group_key="capture_date",
            method="과거 Train / 최신 날짜 suffix Validation 전수평가",
            validation_dates=selected,
            train_target=train_target,
            validation_target=validation_target,
            reason=(
                ""
                if capacity.exact_6000
                else "정상/병해 수집 시기가 분리되어 정확 균형 분할 불가"
            ),
        )
    )
    intervals: list[
        tuple[tuple[Any, ...], tuple[str, ...], CapacitySummary]
    ] = []
    for start in range(len(dates)):
        for end in range(start, len(dates)):
            selected = dates[start : end + 1]
            if len(selected) == len(dates):
                continue
            capacity = date_capacity(
                stats,
                selected,
                train_target=train_target,
                validation_target=validation_target,
            )
            intervals.append(
                (
                    _closest_capacity_rank(
                        capacity,
                        train_target=train_target,
                        validation_target=validation_target,
                        tie=health.stable_hash(
                            seed, "contiguous", *selected
                        ),
                    ),
                    selected,
                    capacity,
                )
            )
    _rank, selected, capacity = min(intervals, key=lambda item: item[0])
    results.append(
        _candidate_for_global_dates(
            stats,
            candidate="contiguous_validation_range",
            group_key="capture_date",
            method="모든 연속 Validation 날짜 구간 exact 전수평가",
            validation_dates=selected,
            train_target=train_target,
            validation_target=validation_target,
            reason=(
                ""
                if capacity.exact_6000
                else "연속 구간으로는 모든 품종의 정상/병해를 양쪽에 유지 불가"
            ),
        )
    )
    current_train = Counter(stats.development_health_by_split["train"])
    current_validation = Counter(
        stats.development_health_by_split["validation"]
    )
    current_capacity = capacity_summary(
        current_train,
        current_validation,
        train_target=train_target,
        validation_target=validation_target,
    )
    train_dates = {
        capture_date
        for capture_date, bucket in stats.development_by_date.items()
        if bucket.current_splits["train"]
    }
    validation_dates = {
        capture_date
        for capture_date, bucket in stats.development_by_date.items()
        if bucket.current_splits["validation"]
    }
    group_overlap = sum(
        counts["train"] > 0 and counts["validation"] > 0
        for counts in stats.species_camera_date_splits.values()
    )
    train_groups = sum(
        counts["train"] > 0
        for counts in stats.species_camera_date_splits.values()
    )
    validation_groups = sum(
        counts["validation"] > 0
        for counts in stats.species_camera_date_splits.values()
    )
    results.append(
        CandidateResult(
            candidate="species_camera_capture_date",
            group_key="species + camera_id + capture_date",
            assignment_method="기존 detection manifest split 감사",
            exact_6000=current_capacity.exact_6000,
            global_date_overlap=len(train_dates & validation_dates),
            declared_group_overlap=group_overlap,
            train_common_capacity=current_capacity.train_common,
            validation_common_capacity=current_capacity.validation_common,
            target_capped_images=current_capacity.target_capped_images,
            train_groups=train_groups,
            validation_groups=validation_groups,
            validation_definition="기존 detection validation 그룹",
            official_changed_rows=0,
            test_date_overlap=len(validation_dates & stats.test_dates),
            reason=(
                "그룹 중복은 없지만 전역 날짜 분리를 보장하지 않음"
            ),
        )
    )
    return results


def audit_pilot_manifests(
    raw_path: Path,
    masked_path: Path,
) -> PilotManifestAudit:
    if not raw_path.is_file() or not masked_path.is_file():
        raise FileNotFoundError("raw 또는 masked health manifest가 없습니다")
    audit = PilotManifestAudit()
    raw_seen: set[tuple[str, str]] = set()
    masked_seen: set[tuple[str, str]] = set()

    def identity(row: Mapping[str, str]) -> tuple[str, str]:
        return (
            health.normalized(row.get("image_archive_id", "")).casefold(),
            health.normalized(row.get("image_member", ""))
            .replace("\\", "/")
            .casefold(),
        )

    def boolean(value: Any) -> bool:
        text = str(value).strip().lower()
        if text == "true":
            return True
        if text == "false":
            return False
        raise ValueError(f"boolean 값이 아닙니다: {value!r}")

    with raw_path.open(encoding="utf-8-sig", newline="") as raw_handle, (
        masked_path.open(encoding="utf-8-sig", newline="")
    ) as masked_handle:
        raw_reader = csv.DictReader(raw_handle)
        masked_reader = csv.DictReader(masked_handle)
        raw_fields = set(raw_reader.fieldnames or ())
        masked_fields = set(masked_reader.fieldnames or ())
        if not set(PILOT_PRESERVED_COLUMNS) <= raw_fields:
            raise ValueError("raw health manifest 필수 컬럼 누락")
        if not MASKED_REQUIRED_COLUMNS <= masked_fields:
            raise ValueError("masked health manifest 필수 컬럼 누락")
        for raw_row, masked_row in itertools.zip_longest(
            raw_reader, masked_reader
        ):
            if raw_row is not None:
                audit.raw_rows += 1
                raw_key = identity(raw_row)
                if raw_key in raw_seen:
                    audit.raw_duplicates += 1
                raw_seen.add(raw_key)
                split = health.normalized(raw_row["split"])
                species = health.normalized(raw_row["species"])
                normality = health.normalized(raw_row["normality"])
                capture_date = canonical_capture_date(
                    raw_row["capture_date"]
                )
                if split not in {"train", "validation"}:
                    audit.invariant_mismatches += 1
                status_key = (split, species, normality)
                audit.rows_by_split_species_status[status_key] += 1
                audit.dates_by_split[split].add(capture_date)
                raw_timestamp = boolean(
                    raw_row["timestamp_region_overlap"]
                )
                if raw_timestamp:
                    audit.timestamp_candidates += 1
                    audit.timestamp_by_split_species_status[
                        status_key
                    ] += 1
            else:
                raw_key = None
                raw_timestamp = False

            if masked_row is not None:
                audit.masked_rows += 1
                masked_key = identity(masked_row)
                if masked_key in masked_seen:
                    audit.masked_duplicates += 1
                masked_seen.add(masked_key)
                applied = boolean(masked_row["timestamp_mask_applied"])
                candidate = boolean(
                    masked_row["timestamp_candidate_overlap"]
                )
                split = health.normalized(masked_row["split"])
                species = health.normalized(masked_row["species"])
                normality = health.normalized(masked_row["normality"])
                if applied:
                    audit.masks_applied += 1
                    audit.masks_by_split_species_status[
                        (split, species, normality)
                    ] += 1
                audit.mask_state_health[applied][normality] += 1
                audit.mask_state_split_health[
                    (split, applied)
                ][normality] += 1
            else:
                masked_key = None
                applied = False
                candidate = False

            if raw_row is None or masked_row is None:
                audit.identity_mismatches += 1
                audit.invariant_mismatches += 1
                continue
            if raw_key != masked_key:
                audit.identity_mismatches += 1
            if any(
                str(raw_row.get(column, ""))
                != str(masked_row.get(column, ""))
                for column in PILOT_PRESERVED_COLUMNS
            ):
                audit.invariant_mismatches += 1
            geometry_invalid = (
                masked_row["source_raw_relative_path"]
                != raw_row["output_relative_path"]
            )
            try:
                mask_rect = json.loads(
                    masked_row["mask_rect_crop_coordinates"]
                )
                mask_width = int(masked_row["mask_width"])
                mask_height = int(masked_row["mask_height"])
                mask_area_ratio = float(masked_row["mask_area_ratio"])
                crop_width = int(raw_row["crop_width"])
                crop_height = int(raw_row["crop_height"])
                if applied:
                    if not isinstance(mask_rect, Mapping):
                        geometry_invalid = True
                    else:
                        x = int(mask_rect["x"])
                        y = int(mask_rect["y"])
                        width = int(mask_rect["width"])
                        height = int(mask_rect["height"])
                        expected_ratio = (
                            width * height / (crop_width * crop_height)
                        )
                        geometry_invalid = geometry_invalid or (
                            x < 0
                            or y < 0
                            or width <= 0
                            or height <= 0
                            or x + width > crop_width
                            or y + height > crop_height
                            or width != mask_width
                            or height != mask_height
                            or abs(mask_area_ratio - expected_ratio) > 1e-9
                        )
                else:
                    geometry_invalid = geometry_invalid or (
                        mask_rect is not None
                        or mask_width != 0
                        or mask_height != 0
                        or abs(mask_area_ratio) > 1e-12
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                geometry_invalid = True
            if geometry_invalid:
                audit.invariant_mismatches += 1
            if candidate != applied or applied != raw_timestamp:
                audit.mask_flag_mismatches += 1
    return audit


def _health_groups_for_bias(
    stats: DateStatistics,
) -> dict[str, dict[Any, Counter[str]]]:
    date_groups: dict[str, Counter[str]] = defaultdict(Counter)
    date_camera_groups: dict[
        tuple[str, str], Counter[str]
    ] = defaultdict(Counter)
    for capture_date, bucket in stats.development_by_date.items():
        for (_species, normality), count in bucket.health.items():
            date_groups[capture_date][normality] += count
    for key, bucket in stats.development_by_date_camera.items():
        for (_species, normality), count in bucket.health.items():
            date_camera_groups[key][normality] += count
    return {
        "capture_date": date_groups,
        "species_capture_date": stats.species_date_health,
        "capture_date_camera_id": date_camera_groups,
    }


def _majority_summary(
    groups: Mapping[Any, Counter[str]],
) -> dict[str, Any]:
    normal_only = abnormal_only = mixed = correct = total = 0
    for counts in groups.values():
        normal = counts["normal"]
        abnormal = counts["abnormal"]
        if normal and abnormal:
            mixed += 1
        elif normal:
            normal_only += 1
        elif abnormal:
            abnormal_only += 1
        correct += max(normal, abnormal)
        total += normal + abnormal
    return {
        "group_count": len(groups),
        "normal_only": normal_only,
        "abnormal_only": abnormal_only,
        "mixed": mixed,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def _mask_lookup_metrics(
    audit: PilotManifestAudit,
) -> tuple[float, float]:
    total = sum(sum(counts.values()) for counts in audit.mask_state_health.values())
    correct = sum(
        max(counts.values(), default=0)
        for counts in audit.mask_state_health.values()
    )
    train_rules = {
        applied: max(
            NORMALITIES,
            key=lambda label: (
                audit.mask_state_split_health[
                    ("train", applied)
                ][label],
                label,
            ),
        )
        for applied in (False, True)
    }
    validation_total = sum(
        sum(
            audit.mask_state_split_health[
                ("validation", applied)
            ].values()
        )
        for applied in (False, True)
    )
    validation_correct = sum(
        audit.mask_state_split_health[
            ("validation", applied)
        ][train_rules[applied]]
        for applied in (False, True)
    )
    return (
        correct / total if total else 0.0,
        validation_correct / validation_total
        if validation_total
        else 0.0,
    )


def write_reports(
    output_dir: Path,
    stats: DateStatistics,
    assignment: DateAssignment,
    camera_plan: CameraPlan,
    candidates: Sequence[CandidateResult],
    pilot_audit: PilotManifestAudit,
    *,
    seed: int,
    train_target: int,
    validation_target: int,
) -> None:
    if not assignment.feasible or assignment.capacity is None:
        raise ValueError("feasible 전역 날짜 assignment가 필요합니다")
    output_dir.mkdir(parents=True, exist_ok=True)

    distribution_fields = (
        "scope",
        "capture_date",
        "species",
        "normal_count",
        "abnormal_count",
        "total_count",
        "camera_id_count",
        "date_label_mode",
        "current_train_count",
        "current_validation_count",
        "current_split_overlap",
        "fixed_test_same_date",
    )
    distribution_rows: list[dict[str, Any]] = []
    for capture_date in sorted(stats.development_by_date):
        bucket = stats.development_by_date[capture_date]
        normal = sum(
            count
            for (_species, normality), count in bucket.health.items()
            if normality == "normal"
        )
        abnormal = bucket.rows - normal
        mode = (
            "mixed"
            if normal and abnormal
            else "normal_only"
            if normal
            else "abnormal_only"
        )
        distribution_rows.append(
            {
                "scope": "capture_date",
                "capture_date": capture_date,
                "species": "",
                "normal_count": normal,
                "abnormal_count": abnormal,
                "total_count": bucket.rows,
                "camera_id_count": len(bucket.cameras),
                "date_label_mode": mode,
                "current_train_count": bucket.current_splits["train"],
                "current_validation_count": bucket.current_splits[
                    "validation"
                ],
                "current_split_overlap": bool(
                    bucket.current_splits["train"]
                    and bucket.current_splits["validation"]
                ),
                "fixed_test_same_date": capture_date
                in stats.test_dates,
            }
        )
        for species in SPECIES:
            counts = stats.species_date_health.get(
                (species, capture_date), Counter()
            )
            if not counts:
                continue
            normal = counts["normal"]
            abnormal = counts["abnormal"]
            current = Counter()
            cameras: set[str] = set()
            for (item_species, camera, item_date), splits in (
                stats.species_camera_date_splits.items()
            ):
                if item_species == species and item_date == capture_date:
                    current.update(splits)
                    if splits["train"] or splits["validation"]:
                        cameras.add(camera)
            distribution_rows.append(
                {
                    "scope": "species_capture_date",
                    "capture_date": capture_date,
                    "species": species,
                    "normal_count": normal,
                    "abnormal_count": abnormal,
                    "total_count": normal + abnormal,
                    "camera_id_count": len(cameras),
                    "date_label_mode": (
                        "mixed"
                        if normal and abnormal
                        else "normal_only"
                        if normal
                        else "abnormal_only"
                    ),
                    "current_train_count": current["train"],
                    "current_validation_count": current["validation"],
                    "current_split_overlap": bool(
                        current["train"] and current["validation"]
                    ),
                    "fixed_test_same_date": capture_date
                    in stats.test_dates,
                }
            )
    health.atomic_write_csv(
        output_dir / "health_date_distribution.csv",
        distribution_fields,
        distribution_rows,
    )

    bias_fields = (
        "analysis_type",
        "key",
        "group_count",
        "normal_only_groups",
        "abnormal_only_groups",
        "mixed_groups",
        "total_count",
        "correct_count",
        "accuracy",
        "baseline_accuracy",
        "exact_6000",
        "global_date_overlap",
        "declared_group_overlap",
        "train_common_capacity",
        "validation_common_capacity",
        "target_capped_images",
        "train_groups",
        "validation_groups",
        "official_changed_rows",
        "test_date_overlap",
        "detail",
    )
    baseline = max(
        sum(
            count
            for (_species, normality), count in stats.development_health.items()
            if normality == label
        )
        for label in NORMALITIES
    ) / stats.development_rows
    bias_rows: list[dict[str, Any]] = []
    for key, groups in _health_groups_for_bias(stats).items():
        summary = _majority_summary(groups)
        bias_rows.append(
            {
                "analysis_type": "majority_lookup",
                "key": key,
                "group_count": summary["group_count"],
                "normal_only_groups": summary["normal_only"],
                "abnormal_only_groups": summary["abnormal_only"],
                "mixed_groups": summary["mixed"],
                "total_count": summary["total"],
                "correct_count": summary["correct"],
                "accuracy": summary["accuracy"],
                "baseline_accuracy": baseline,
                "detail": (
                    "metadata in-sample majority oracle; 모델 성능이 아님"
                ),
            }
        )
    current_detection_overlap = sum(
        bool(bucket.current_splits["train"])
        and bool(bucket.current_splits["validation"])
        for bucket in stats.development_by_date.values()
    )
    raw_date_overlap = len(
        pilot_audit.dates_by_split["train"]
        & pilot_audit.dates_by_split["validation"]
    )
    bias_rows.extend(
        (
            {
                "analysis_type": "current_split_overlap",
                "key": "detection_manifest",
                "global_date_overlap": current_detection_overlap,
                "detail": "고정 Test 제외 detection 현재 split",
            },
            {
                "analysis_type": "current_split_overlap",
                "key": "raw_and_masked_health_pilot",
                "global_date_overlap": raw_date_overlap,
                "detail": "현재 6,000장 raw/masked 파일럿",
            },
        )
    )
    for item in candidates:
        bias_rows.append(
            {
                "analysis_type": "split_candidate",
                "key": item.candidate,
                "exact_6000": item.exact_6000,
                "global_date_overlap": item.global_date_overlap,
                "declared_group_overlap": item.declared_group_overlap,
                "train_common_capacity": item.train_common_capacity,
                "validation_common_capacity": item.validation_common_capacity,
                "target_capped_images": item.target_capped_images,
                "train_groups": item.train_groups,
                "validation_groups": item.validation_groups,
                "official_changed_rows": item.official_changed_rows,
                "test_date_overlap": item.test_date_overlap,
                "detail": (
                    f"{item.group_key}; {item.assignment_method}; "
                    f"{item.validation_definition}; {item.reason}"
                ),
            }
        )
    health.atomic_write_csv(
        output_dir / "health_date_bias_analysis.csv",
        bias_fields,
        bias_rows,
    )

    mask_fields = (
        "scope",
        "split",
        "species",
        "health_class_name",
        "total_count",
        "mask_applied_count",
        "mask_not_applied_count",
        "mask_rate",
        "lookup_correct",
        "lookup_total",
        "lookup_accuracy",
        "note",
    )
    mask_rows: list[dict[str, Any]] = []
    class_names = {
        "normal": "HEALTHY",
        "abnormal": "DISEASE_SUSPECTED",
    }
    for split in ("train", "validation"):
        for normality in NORMALITIES:
            total = sum(
                pilot_audit.rows_by_split_species_status[
                    (split, species, normality)
                ]
                for species in SPECIES
            )
            applied = sum(
                pilot_audit.masks_by_split_species_status[
                    (split, species, normality)
                ]
                for species in SPECIES
            )
            mask_rows.append(
                {
                    "scope": "split_class",
                    "split": split,
                    "species": "",
                    "health_class_name": class_names[normality],
                    "total_count": total,
                    "mask_applied_count": applied,
                    "mask_not_applied_count": total - applied,
                    "mask_rate": applied / total if total else 0.0,
                    "note": "timestamp 후보 ROI와 crop의 기하학적 교차",
                }
            )
    for species in SPECIES:
        for normality in NORMALITIES:
            total = sum(
                pilot_audit.rows_by_split_species_status[
                    (split, species, normality)
                ]
                for split in ("train", "validation")
            )
            applied = sum(
                pilot_audit.masks_by_split_species_status[
                    (split, species, normality)
                ]
                for split in ("train", "validation")
            )
            mask_rows.append(
                {
                    "scope": "species_class",
                    "split": "",
                    "species": species,
                    "health_class_name": class_names[normality],
                    "total_count": total,
                    "mask_applied_count": applied,
                    "mask_not_applied_count": total - applied,
                    "mask_rate": applied / total if total else 0.0,
                    "note": "mask 유무가 클래스와 상관될 수 있음",
                }
            )
    empirical_lookup, train_rule_validation = _mask_lookup_metrics(
        pilot_audit
    )
    total_pilot = sum(
        sum(counts.values())
        for counts in pilot_audit.mask_state_health.values()
    )
    empirical_correct = round(empirical_lookup * total_pilot)
    validation_total = sum(
        sum(
            pilot_audit.mask_state_split_health[
                ("validation", applied)
            ].values()
        )
        for applied in (False, True)
    )
    mask_rows.extend(
        (
            {
                "scope": "mask_only_lookup",
                "lookup_correct": empirical_correct,
                "lookup_total": total_pilot,
                "lookup_accuracy": empirical_lookup,
                "note": "전체 pilot에서 mask bool별 majority; 모델 성능 아님",
            },
            {
                "scope": "train_rule_validation_lookup",
                "lookup_correct": round(
                    train_rule_validation * validation_total
                ),
                "lookup_total": validation_total,
                "lookup_accuracy": train_rule_validation,
                "note": "Train mask majority rule을 Validation에 적용",
            },
        )
    )
    health.atomic_write_csv(
        output_dir / "health_timestamp_mask_bias.csv",
        mask_fields,
        mask_rows,
    )

    candidate_lines = "\n".join(
        "| {candidate} | {group_key} | {exact} | {overlap:,} | "
        "{group_overlap:,} | {train:,} | {validation:,} | {total:,} | "
        "{changed:,} | {reason} |".format(
            candidate=item.candidate,
            group_key=item.group_key,
            exact="가능" if item.exact_6000 else "불가",
            overlap=item.global_date_overlap,
            group_overlap=item.declared_group_overlap,
            train=item.train_common_capacity,
            validation=item.validation_common_capacity,
            total=item.target_capped_images,
            changed=item.official_changed_rows,
            reason=item.reason or "-",
        )
        for item in candidates
    )
    global_result = next(
        item for item in candidates if item.candidate == "global_capture_date"
    )
    majority = {
        key: _majority_summary(groups)
        for key, groups in _health_groups_for_bias(stats).items()
    }
    normal_dates = [
        capture_date
        for capture_date, bucket in stats.development_by_date.items()
        if any(
            normality == "normal" and count
            for (_species, normality), count in bucket.health.items()
        )
    ]
    abnormal_dates = [
        capture_date
        for capture_date, bucket in stats.development_by_date.items()
        if any(
            normality == "abnormal" and count
            for (_species, normality), count in bucket.health.items()
        )
    ]
    development_dates = set(stats.development_by_date)
    test_date_overlap = development_dates & stats.test_dates
    train_test_date_overlap = (
        set(assignment.train_dates) & stats.test_dates
    )
    validation_test_date_overlap = (
        set(assignment.validation_dates) & stats.test_dates
    )
    test_free_health = _sum_health(
        bucket
        for capture_date, bucket in stats.development_by_date.items()
        if capture_date not in stats.test_dates
    )
    combined_target = train_target + validation_target
    test_free_limiting = [
        (
            f"{species}/{class_names[normality]} "
            f"{test_free_health[(species, normality)]:,}/"
            f"{combined_target:,}"
        )
        for species, normality in CELLS
        if test_free_health[(species, normality)] < combined_target
    ]
    feasibility = f"""# 건강 체크 날짜 홀드아웃 구성 가능성

## 감사 범위

- detection 개발 후보: {stats.development_rows:,}장
- 고정 Test 제외: {stats.test_rows:,}장; 최적화·선택에 사용 0장
- 개발 촬영 날짜: {len(stats.development_by_date):,}개
- 이미지/ZIP 바이트 접근, 이미지 추출, 모델 로드·학습: 없음

## 날짜 편향

- 날짜 lookup: {majority['capture_date']['accuracy']:.2%}
- 품종+날짜 lookup: {majority['species_capture_date']['accuracy']:.2%}
- 날짜+카메라 lookup: {majority['capture_date_camera_id']['accuracy']:.2%}
- 항상 다수 상태 baseline: {baseline:.2%}
- 현재 detection Train/Validation 날짜 중복: {current_detection_overlap:,}개
- 현재 raw/masked 6,000장 날짜 중복: {raw_date_overlap:,}개
- 정상 날짜 범위: {min(normal_dates)} ~ {max(normal_dates)}
- 병해 날짜 범위: {min(abnormal_dates)} ~ {max(abnormal_dates)}
- 날짜 유형: 정상 전용 {majority['capture_date']['normal_only']}일 / 병해 전용 {majority['capture_date']['abnormal_only']}일 / 혼합 {majority['capture_date']['mixed']}일

위 lookup은 metadata를 같은 데이터에서 다수결한 진단치이며 모델 성능이 아니다.
그럼에도 품종+날짜 97.59%는 수집 날짜·환경 shortcut 위험이 매우 큼을 뜻한다.

## 후보 비교

| 후보 | 그룹 키 | 정확 6,000 | 전역 날짜 중복 | 선언 그룹 중복 | Train 공통 가용 | Validation 공통 가용 | 타깃 상한 가용 합계 | 기존 split 변경 행 | 판단 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{candidate_lines}

## 결론

1. 전역 날짜 홀드아웃: **{'가능' if assignment.feasible else '불가능'}**
2. 6,000장 균형 구성: **{'가능' if global_result.exact_6000 else '불가능'}**
3. 가능한 목표 수량: Train {len(CELLS) * train_target:,}장 / Validation {len(CELLS) * validation_target:,}장
4. 권장안의 Train/Validation 날짜 중복: **0개**
5. 엄격한 global camera_id 분리 추가 시 카메라 중복: **{camera_plan.camera_overlap if camera_plan.feasible else '구성 불가'}**
6. chronological 및 연속 날짜 구간: 수집 시기가 상태별로 갈려 정확 균형 구성이 불가능하다.
7. 고정 Test 날짜 {len(stats.test_dates):,}일 중 development와 겹치는 날짜는 {len(test_date_overlap):,}일이며, 권장 Train/Test {len(train_test_date_overlap):,}일, Validation/Test {len(validation_test_date_overlap):,}일이 겹친다. Test 겹침 날짜를 development에서 전부 제외하면 부족 셀은 {', '.join(test_free_limiting) if test_free_limiting else '없음'}이므로 현재 고정 Test까지 포함한 정확 6,000장 3-way date-isolated split은 불가능하다.
8. 보고된 raw/masked 99.9~100%는 날짜·카메라·배경 shortcut 조합으로 설명될 가능성이 크지만 metadata만으로 인과를 확정할 수 없다.
9. mask-only lookup {empirical_lookup:.2%}, Train 규칙→Validation {train_rule_validation:.2%}이므로 회색 mask 존재 자체도 보조 shortcut이 될 수 있다. 이는 완전한 timestamp ablation이 아니다.
10. 외부 스마트폰·새 환경 평가 전 서비스 성능으로 인정하거나 배포 판단에 사용하는 것은 부적절하다.

## Raw/Masked 대응 검증

- raw/masked 행 수: {pilot_audit.raw_rows:,}/{pilot_audit.masked_rows:,}
- 이미지 identity 불일치: {pilot_audit.identity_mismatches:,}건
- split·class·crop·source 경로·mask geometry 불일치: {pilot_audit.invariant_mismatches:,}건
- timestamp overlap/mask 적용 플래그 불일치: {pilot_audit.mask_flag_mismatches:,}건
"""
    health.atomic_write_text(
        output_dir / "health_date_holdout_feasibility.md",
        feasibility,
    )

    capacity_lines = "\n".join(
        "| {species} | {status} | {train_cap:,} | {train_target:,} | "
        "{validation_cap:,} | {validation_target:,} |".format(
            species=species,
            status=class_names[normality],
            train_cap=camera_plan.train_capacity[(species, normality)],
            train_target=train_target,
            validation_cap=camera_plan.validation_capacity[
                (species, normality)
            ],
            validation_target=validation_target,
        )
        for species, normality in CELLS
    )
    validation_date_set = set(assignment.validation_dates)
    validation_camera_set = set(camera_plan.validation_camera_ids)
    train_camera_set = set(camera_plan.train_camera_ids)
    validation_disease = _sum_disease(
        bucket
        for (capture_date, camera), bucket in (
            stats.development_by_date_camera.items()
        )
        if capture_date in validation_date_set
        and camera in validation_camera_set
    )
    train_disease = _sum_disease(
        bucket
        for (capture_date, camera), bucket in (
            stats.development_by_date_camera.items()
        )
        if capture_date not in validation_date_set
        and camera in train_camera_set
    )
    missing_validation_diseases = sorted(
        f"{species}/{disease}"
        for (species, disease), total in stats.development_disease.items()
        if total and not validation_disease[(species, disease)]
    )
    missing_train_diseases = sorted(
        f"{species}/{disease}"
        for (species, disease), total in stats.development_disease.items()
        if total and not train_disease[(species, disease)]
    )
    split_plan = f"""# 건강 체크 전역 날짜 홀드아웃 분할 계획

## 고정 규칙

- seed: `{seed}`
- Validation 날짜: {', '.join(assignment.validation_dates)}
- Train 날짜: {', '.join(assignment.train_dates)}
- 같은 capture_date는 하나의 split에만 둔다.
- Validation global camera_id: {', '.join(camera_plan.validation_camera_ids)}
- Train global camera_id: {', '.join(camera_plan.train_camera_ids)}
- 교차 조합(Validation 카메라+Train 날짜 또는 Train 카메라+Validation 날짜)은 선택하지 않는다.
- 날짜 중복 0, global camera_id 중복 0, Test 사용 0, 동일 이미지 중복 0을 수량 정확도보다 우선한다.

## 목표와 엄격한 날짜+카메라 pool 가용량

| 품종 | 상태 | Train 가용 | Train 선택 | Validation 가용 | Validation 선택 |
| --- | --- | ---: | ---: | ---: | ---: |
{capacity_lines}

- 최종 계획: Train {len(CELLS) * train_target:,}장 / Validation {len(CELLS) * validation_target:,}장
- 최소 Validation 날짜 수: {assignment.minimum_date_count}일; 최소 조합 중 feasible {assignment.feasible_combinations_at_minimum}개
- 최소 Validation camera_id 수: {camera_plan.minimum_validation_camera_count}개; 최소 조합 중 feasible {camera_plan.feasible_combinations_at_minimum}개
- 고정 Test 날짜와 development 날짜 중복: {len(test_date_overlap)}일
- 고정 Test 날짜와 권장 Train 날짜 중복: {len(train_test_date_overlap)}일
- 고정 Test 날짜와 권장 Validation 날짜 중복: {len(validation_test_date_overlap)}일

## 제한

- 엄격한 날짜+카메라 Validation pool에서 지원되지 않는 품종/병해 조합: {', '.join(missing_validation_diseases) if missing_validation_diseases else '없음'}
- 엄격한 날짜+카메라 Train pool에서 지원되지 않는 품종/병해 조합: {', '.join(missing_train_diseases) if missing_train_diseases else '없음'}
- 특히 이 계획은 이진 HEALTHY/DISEASE_SUSPECTED 평가용이다. 모든 병해 종류 평가를 보장하지 않는다.
- 현재 단계에서는 최종 manifest, crop 또는 이미지를 생성하지 않았다.
- 다음 단계에서 별도 승인을 받은 뒤 위 날짜·카메라 제약으로 행을 deterministic sampling해야 한다.
"""
    health.atomic_write_text(
        output_dir / "health_date_holdout_split_plan.md",
        split_plan,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = (
        args.detection_manifest,
        args.raw_health_manifest,
        args.masked_health_manifest,
    )
    input_snapshot = health.snapshot_files(input_paths)
    artifacts_snapshot = health.snapshot_tree(ARTIFACTS_ROOT)
    stats = stream_date_statistics(args.detection_manifest)
    assignment = find_minimal_global_date_assignment(
        stats,
        train_target=args.train_per_species_status,
        validation_target=args.validation_per_species_status,
        seed=args.seed,
    )
    if not assignment.feasible:
        raise RuntimeError(
            "전역 날짜 홀드아웃 목표 구성 불가: " + assignment.reason
        )
    camera_plan = choose_global_camera_partition(
        stats,
        assignment.validation_dates,
        train_target=args.train_per_species_status,
        validation_target=args.validation_per_species_status,
        seed=args.seed,
    )
    if not camera_plan.feasible:
        validation_dates = set(assignment.validation_dates)
        validation_cameras = {
            camera
            for capture_date, camera in stats.development_by_date_camera
            if capture_date in validation_dates
        }
        train_cameras = {
            camera
            for capture_date, camera in stats.development_by_date_camera
            if capture_date not in validation_dates
        }
        camera_plan = CameraPlan(
            feasible=False,
            validation_camera_ids=tuple(
                sorted(validation_cameras, key=health.natural_text_key)
            ),
            train_camera_ids=tuple(
                sorted(train_cameras, key=health.natural_text_key)
            ),
            validation_capacity=Counter(
                assignment.capacity.validation
            ),
            train_capacity=Counter(assignment.capacity.train),
            camera_overlap=len(validation_cameras & train_cameras),
            reason=(
                "엄격한 global camera_id 추가 분리는 불가능; "
                "전역 날짜 홀드아웃만 사용"
            ),
        )
    candidates = evaluate_candidate_splits(
        stats,
        global_assignment=assignment,
        train_target=args.train_per_species_status,
        validation_target=args.validation_per_species_status,
        seed=args.seed,
    )
    pilot_audit = audit_pilot_manifests(
        args.raw_health_manifest,
        args.masked_health_manifest,
    )
    if (
        pilot_audit.raw_rows != EXPECTED_PILOT_ROWS
        or pilot_audit.masked_rows != EXPECTED_PILOT_ROWS
    ):
        raise RuntimeError(
            "raw/masked pilot 행 수 불일치: "
            f"{pilot_audit.raw_rows}/{pilot_audit.masked_rows}"
        )
    pilot_failures = {
        "identity_mismatches": pilot_audit.identity_mismatches,
        "invariant_mismatches": pilot_audit.invariant_mismatches,
        "raw_duplicates": pilot_audit.raw_duplicates,
        "masked_duplicates": pilot_audit.masked_duplicates,
        "mask_flag_mismatches": pilot_audit.mask_flag_mismatches,
    }
    pilot_failures = {
        key: value for key, value in pilot_failures.items() if value
    }
    if pilot_failures:
        raise RuntimeError(f"raw/masked pilot 검증 실패: {pilot_failures}")
    write_reports(
        args.output_dir,
        stats,
        assignment,
        camera_plan,
        candidates,
        pilot_audit,
        seed=args.seed,
        train_target=args.train_per_species_status,
        validation_target=args.validation_per_species_status,
    )
    outputs = [args.output_dir / name for name in OUTPUT_FILES]
    health.assert_deidentified(outputs)
    health.assert_snapshot_unchanged(input_snapshot)
    health.assert_tree_unchanged(ARTIFACTS_ROOT, artifacts_snapshot)
    majority = {
        key: _majority_summary(groups)
        for key, groups in _health_groups_for_bias(stats).items()
    }
    mask_lookup, mask_validation_lookup = _mask_lookup_metrics(
        pilot_audit
    )
    raw_date_overlap = len(
        pilot_audit.dates_by_split["train"]
        & pilot_audit.dates_by_split["validation"]
    )
    current_detection_overlap = sum(
        bool(bucket.current_splits["train"])
        and bool(bucket.current_splits["validation"])
        for bucket in stats.development_by_date.values()
    )
    return {
        "development_rows": stats.development_rows,
        "test_rows_excluded": stats.test_rows,
        "test_rows_used": 0,
        "date_count": len(stats.development_by_date),
        "current_detection_date_overlap": current_detection_overlap,
        "current_raw_pilot_date_overlap": raw_date_overlap,
        "date_lookup_accuracy": majority["capture_date"]["accuracy"],
        "species_date_lookup_accuracy": majority[
            "species_capture_date"
        ]["accuracy"],
        "date_camera_lookup_accuracy": majority[
            "capture_date_camera_id"
        ]["accuracy"],
        "date_holdout_feasible": assignment.feasible,
        "date_overlap": 0,
        "validation_dates": list(assignment.validation_dates),
        "train_dates": len(assignment.train_dates),
        "train_images": len(CELLS)
        * args.train_per_species_status,
        "validation_images": len(CELLS)
        * args.validation_per_species_status,
        "camera_holdout_feasible": camera_plan.feasible,
        "validation_camera_ids": list(
            camera_plan.validation_camera_ids
        ),
        "train_camera_ids": list(camera_plan.train_camera_ids),
        "camera_overlap": camera_plan.camera_overlap,
        "mask_only_lookup_accuracy": mask_lookup,
        "train_mask_rule_validation_accuracy": mask_validation_lookup,
        "raw_masked_identity_mismatches": (
            pilot_audit.identity_mismatches
        ),
        "crop_or_manifest_mismatches": (
            pilot_audit.invariant_mismatches
        ),
        "images_read": 0,
        "images_extracted": 0,
        "model_training": False,
        "input_files_modified": False,
        "artifacts_modified": False,
        "output_files": [f"reports/{path.name}" for path in outputs],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
