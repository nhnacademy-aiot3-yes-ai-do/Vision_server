#!/usr/bin/env python3
"""Read-only data audit for the Mushroom Health Check v1 scope.

The completed 332,100-row label-quality database is consumed with SQLite's
immutable read-only mode.  The detection manifest is streamed with csv.DictReader.
No image bytes are opened, no archive member is extracted, and no model is loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import inspect_aihub_archives as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUALITY_DB = PROJECT_ROOT / "artifacts" / "quality_analysis.sqlite3"
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "detection_dataset_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports"
DEFAULT_SEED = 20260726
EXPECTED_FULL_JSON = 332_100
SPECIES = ("느타리", "양송이", "큰느타리", "팽이", "표고")
HEALTH_TASKS = ("생육", "병해")
EXPECTED_TASK_NORMALITY = {
    "배양": "normal",
    "생육": "normal",
    "병해": "abnormal",
}
GROUP_CANDIDATES = {
    "species_camera": ("species", "camera_id"),
    "species_camera_date": ("species", "camera_id", "capture_date"),
    "camera": ("camera_id",),
    "species_date": ("species", "capture_date"),
}
LOCAL_PATH_MARKERS = ("/mnt/", "/home/", "\\Users\\")
OUTPUT_FILES = (
    "health_label_distribution.csv",
    "health_disease_species_mapping.csv",
    "health_camera_bias_analysis.csv",
    "health_input_strategy.md",
    "health_model_scope_decision.md",
    "health_split_strategy.md",
    "health_pilot_feasibility.md",
)


@dataclass
class HealthStats:
    total: int = 0
    normality: Counter[str] = field(default_factory=Counter)
    task_normality: Counter[tuple[str, str]] = field(default_factory=Counter)
    species_normality: Counter[tuple[str, str]] = field(default_factory=Counter)
    disease: Counter[str] = field(default_factory=Counter)
    species_disease: Counter[tuple[str, str]] = field(default_factory=Counter)
    issue_counts: Counter[str] = field(default_factory=Counter)
    contradictory_records: int = 0
    candidate_total: int = 0
    candidate_normality: Counter[str] = field(default_factory=Counter)
    candidate_species_normality: Counter[tuple[str, str]] = field(
        default_factory=Counter
    )
    candidate_health_camera: dict[tuple[str, str], Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    candidate_health_date: dict[tuple[str, str], Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    disease_camera: dict[tuple[str, str], Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    disease_date: dict[tuple[str, str], Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )


@dataclass
class ManifestStats:
    total_rows: int = 0
    test_rows: int = 0
    development_rows: int = 0
    development_normality: Counter[str] = field(default_factory=Counter)
    development_species_normality: Counter[tuple[str, str]] = field(
        default_factory=Counter
    )
    development_disease: Counter[str] = field(default_factory=Counter)
    development_species_disease: Counter[tuple[str, str]] = field(
        default_factory=Counter
    )
    development_health_camera: dict[tuple[str, str], Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    development_health_date: dict[tuple[str, str], Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    development_disease_camera: dict[
        tuple[str, str], Counter[str]
    ] = field(default_factory=lambda: defaultdict(Counter))
    development_disease_date: dict[
        tuple[str, str], Counter[str]
    ] = field(default_factory=lambda: defaultdict(Counter))
    disease_by_species: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    disease_by_species_camera: dict[tuple[str, str], Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    disease_by_species_camera_date: dict[
        tuple[str, str, str], Counter[str]
    ] = field(default_factory=lambda: defaultdict(Counter))
    camera_disease: dict[tuple[str, str], Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    group_sizes: dict[str, Counter[tuple[str, ...]]] = field(
        default_factory=lambda: {
            name: Counter() for name in GROUP_CANDIDATES
        }
    )
    group_split_counts: dict[
        str, dict[tuple[str, ...], Counter[str]]
    ] = field(
        default_factory=lambda: {
            name: defaultdict(Counter) for name in GROUP_CANDIDATES
        }
    )
    development_group_sizes: dict[str, Counter[tuple[str, ...]]] = field(
        default_factory=lambda: {
            name: Counter() for name in GROUP_CANDIDATES
        }
    )


@dataclass
class CameraAssignment:
    feasible: bool
    train_cameras: dict[str, tuple[str, ...]] = field(default_factory=dict)
    validation_cameras: dict[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    reason: str = ""
    limiting_species: str = ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "완료된 전체 라벨 감사 DB와 detection manifest를 읽기 전용으로 "
            "스트리밍하여 건강 체크 v1 데이터 편향과 분할 가능성을 분석합니다."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--quality-db",
        type=Path,
        default=Path("artifacts/quality_analysis.sqlite3"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/detection_dataset_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--train-per-species-status",
        type=positive_int,
        default=500,
    )
    parser.add_argument(
        "--validation-per-species-status",
        type=positive_int,
        default=100,
    )
    args = parser.parse_args(argv)
    args.quality_db = project_path(args.quality_db)
    args.manifest = project_path(args.manifest)
    args.output_dir = project_path(args.output_dir)
    if args.output_dir == PROJECT_ROOT / "artifacts" or (
        PROJECT_ROOT / "artifacts"
    ) in args.output_dir.parents:
        parser.error("--output-dir는 artifacts 밖이어야 합니다")
    return args


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("정수여야 합니다") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("양수여야 합니다")
    return parsed


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def stable_hash(seed: int, *parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(
        f"{seed}\x1e{payload}".encode("utf-8")
    ).hexdigest()


def natural_text_key(value: str) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, value)


def normalized(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text in {"", "<missing>", "None", "null"}:
        return ""
    return text


def health_label_issues(
    task: str,
    normality: str,
    disease_type: str | None,
) -> tuple[str, tuple[str, ...]]:
    """Return API health state and all label consistency issues."""
    disease = normalized(disease_type)
    issues: list[str] = []
    if normality == "normal":
        state = "HEALTHY"
    elif normality == "abnormal":
        state = "DISEASE_SUSPECTED"
    else:
        state = "UNCERTAIN"
        issues.append("invalid_normality")
    expected = EXPECTED_TASK_NORMALITY.get(task)
    if expected and normality != expected:
        issues.append("task_normality_mismatch")
    if normality == "abnormal" and not disease:
        issues.append("abnormal_missing_disease")
    if normality == "normal" and disease:
        issues.append("normal_has_disease")
    return state, tuple(issues)


def open_complete_quality_db(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError("완료된 품질 감사 DB가 없습니다")
    wal = Path(str(path) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError(
            "품질 감사 DB에 미반영 WAL이 있어 immutable 읽기를 거부합니다"
        )
    connection = sqlite3.connect(
        f"file:{path.resolve()}?mode=ro&immutable=1",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    required = {
        "species",
        "task",
        "normality",
        "disease_type",
        "camera_id",
        "capture_date",
    }
    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(quality_records)")
    }
    if not required <= columns:
        connection.close()
        raise ValueError("품질 감사 DB에 건강 감사 필수 필드가 없습니다")
    metadata = dict(
        connection.execute("SELECT key, value FROM quality_metadata")
    )
    row_count = connection.execute(
        "SELECT COUNT(*) FROM quality_records"
    ).fetchone()[0]
    if (
        metadata.get("scan_complete") != "true"
        or int(metadata.get("processed_json", "0")) != row_count
        or int(metadata.get("failed_json", "-1")) != 0
        or row_count != EXPECTED_FULL_JSON
    ):
        connection.close()
        raise ValueError("전체 라벨 품질 감사 DB가 완전하지 않습니다")
    return connection


def update_health_stats(
    stats: HealthStats,
    row: Mapping[str, Any],
) -> None:
    species = normalized(row["species"])
    task = normalized(row["task"])
    normality = normalized(row["normality"])
    disease = normalized(row["disease_type"])
    camera = normalized(row["camera_id"]) or "<missing>"
    capture_date = normalized(row["capture_date"]) or "<missing>"
    stats.total += 1
    stats.normality[normality] += 1
    stats.task_normality[(task, normality)] += 1
    stats.species_normality[(species, normality)] += 1
    if disease:
        stats.disease[disease] += 1
        stats.species_disease[(species, disease)] += 1
    _state, issues = health_label_issues(task, normality, disease)
    if issues:
        stats.contradictory_records += 1
        stats.issue_counts.update(issues)
    if task not in HEALTH_TASKS:
        return
    stats.candidate_total += 1
    stats.candidate_normality[normality] += 1
    stats.candidate_species_normality[(species, normality)] += 1
    stats.candidate_health_camera[(species, camera)][normality] += 1
    stats.candidate_health_date[(species, capture_date)][normality] += 1
    if normality == "abnormal" and disease:
        stats.disease_camera[(species, disease)][camera] += 1
        stats.disease_date[(species, disease)][capture_date] += 1


def stream_health_statistics(
    connection: sqlite3.Connection,
    *,
    batch_size: int = 5_000,
) -> HealthStats:
    stats = HealthStats()
    cursor = connection.execute(
        """
        SELECT species, task, normality, disease_type, camera_id, capture_date
        FROM quality_records
        ORDER BY official_split, archive_id, json_member
        """
    )
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        for row in rows:
            update_health_stats(stats, row)
    return stats


def group_key(row: Mapping[str, str], columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(normalized(row[column]) or "<missing>" for column in columns)


def analyze_detection_manifest(path: Path) -> ManifestStats:
    if not path.is_file():
        raise FileNotFoundError("detection manifest가 없습니다")
    stats = ManifestStats()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "split",
            "species",
            "task",
            "normality",
            "disease_type",
            "camera_id",
            "capture_date",
        }
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError("detection manifest 필수 컬럼이 없습니다")
        for row in reader:
            stats.total_rows += 1
            split = normalized(row["split"])
            species = normalized(row["species"])
            normality = normalized(row["normality"])
            disease = normalized(row["disease_type"])
            camera = normalized(row["camera_id"]) or "<missing>"
            capture_date = normalized(row["capture_date"]) or "<missing>"
            for name, columns in GROUP_CANDIDATES.items():
                key = group_key(row, columns)
                stats.group_sizes[name][key] += 1
                stats.group_split_counts[name][key][split] += 1
            if split == "test":
                stats.test_rows += 1
                continue
            if split not in {"train", "validation"}:
                raise ValueError(f"지원하지 않는 split: {split}")
            stats.development_rows += 1
            stats.development_normality[normality] += 1
            stats.development_species_normality[(species, normality)] += 1
            if disease:
                stats.development_disease[disease] += 1
                stats.development_species_disease[(species, disease)] += 1
            stats.development_health_camera[(species, camera)][normality] += 1
            stats.development_health_date[
                (species, capture_date)
            ][normality] += 1
            for name, columns in GROUP_CANDIDATES.items():
                stats.development_group_sizes[name][
                    group_key(row, columns)
                ] += 1
            if normality == "abnormal" and disease:
                stats.development_disease_camera[
                    (species, disease)
                ][camera] += 1
                stats.development_disease_date[
                    (species, disease)
                ][capture_date] += 1
                stats.disease_by_species[species][disease] += 1
                stats.disease_by_species_camera[
                    (species, camera)
                ][disease] += 1
                stats.disease_by_species_camera_date[
                    (species, camera, capture_date)
                ][disease] += 1
                stats.camera_disease[(species, camera)][disease] += 1
    return stats


def group_purity(
    groups: Mapping[Any, Counter[str]],
    labels: Sequence[str] = ("normal", "abnormal"),
) -> dict[str, Any]:
    only = Counter()
    correct = 0
    total = 0
    for counts in groups.values():
        present = [label for label in labels if counts[label] > 0]
        if len(present) == 1:
            only[present[0]] += 1
        elif len(present) > 1:
            only["mixed"] += 1
        group_total = sum(counts[label] for label in labels)
        total += group_total
        correct += max((counts[label] for label in labels), default=0)
    return {
        "groups": len(groups),
        "normal_only": only["normal"],
        "abnormal_only": only["abnormal"],
        "mixed": only["mixed"],
        "majority_accuracy": correct / total if total else 0.0,
    }


def majority_lookup_accuracy(groups: Mapping[Any, Counter[str]]) -> float:
    total = sum(sum(counts.values()) for counts in groups.values())
    correct = sum(max(counts.values(), default=0) for counts in groups.values())
    return correct / total if total else 0.0


def concentration(values: Counter[str]) -> dict[str, Any]:
    total = sum(values.values())
    if not total:
        return {
            "total": 0,
            "support": 0,
            "top_key": "",
            "top_count": 0,
            "top_share": 0.0,
            "hhi": 0.0,
        }
    top_key, top_count = sorted(
        values.items(),
        key=lambda item: (-item[1], natural_text_key(item[0])),
    )[0]
    return {
        "total": total,
        "support": len(values),
        "top_key": top_key,
        "top_count": top_count,
        "top_share": top_count / total,
        "hhi": sum((count / total) ** 2 for count in values.values()),
    }


def camera_overlap(
    train_cameras: Mapping[str, Sequence[str]],
    validation_cameras: Mapping[str, Sequence[str]],
) -> set[tuple[str, str]]:
    train = {
        (species, camera)
        for species, cameras in train_cameras.items()
        for camera in cameras
    }
    validation = {
        (species, camera)
        for species, cameras in validation_cameras.items()
        for camera in cameras
    }
    return train & validation


def choose_camera_holdout(
    health_by_camera: Mapping[tuple[str, str], Counter[str]],
    disease_by_camera: Mapping[tuple[str, str], Counter[str]],
    *,
    species_values: Sequence[str] = SPECIES,
    seed: int = DEFAULT_SEED,
    train_target: int = 500,
    validation_target: int = 100,
    validation_camera_fraction: float = 0.20,
) -> CameraAssignment:
    train_result: dict[str, tuple[str, ...]] = {}
    validation_result: dict[str, tuple[str, ...]] = {}
    for species in species_values:
        cameras = sorted(
            (
                camera
                for item_species, camera in health_by_camera
                if item_species == species
            ),
            key=natural_text_key,
        )
        if len(cameras) < 2:
            return CameraAssignment(
                feasible=False,
                reason="품종별 Train/Validation 카메라가 2개 미만",
                limiting_species=species,
            )
        validation_count = max(
            1, min(len(cameras) - 1, round(len(cameras) * validation_camera_fraction))
        )
        total_health = Counter()
        total_disease = Counter()
        for camera in cameras:
            total_health.update(health_by_camera[(species, camera)])
            total_disease.update(disease_by_camera.get((species, camera), Counter()))
        best: tuple[Any, tuple[str, ...]] | None = None
        for validation in itertools.combinations(cameras, validation_count):
            validation_set = set(validation)
            validation_health = Counter()
            validation_disease = Counter()
            for camera in validation:
                validation_health.update(health_by_camera[(species, camera)])
                validation_disease.update(
                    disease_by_camera.get((species, camera), Counter())
                )
            train_health = total_health - validation_health
            train_disease = total_disease - validation_disease
            if any(
                validation_health[label] < validation_target
                or train_health[label] < train_target
                for label in ("normal", "abnormal")
            ):
                continue
            disease_names = set(total_disease)
            missing_sides = sum(
                int(validation_disease[name] == 0)
                + int(train_disease[name] == 0)
                for name in disease_names
            )
            validation_disease_total = sum(validation_disease.values())
            disease_divergence = sum(
                abs(
                    (
                        validation_disease[name] / validation_disease_total
                        if validation_disease_total
                        else 0.0
                    )
                    - total_disease[name] / sum(total_disease.values())
                )
                for name in disease_names
            )
            health_fraction_error = sum(
                abs(
                    validation_health[label] / total_health[label]
                    - validation_camera_fraction
                )
                for label in ("normal", "abnormal")
                if total_health[label]
            )
            rank = (
                missing_sides,
                round(disease_divergence, 12),
                round(health_fraction_error, 12),
                stable_hash(seed, species, *validation),
            )
            if best is None or rank < best[0]:
                best = (rank, tuple(validation))
        if best is None:
            return CameraAssignment(
                feasible=False,
                reason=(
                    f"정상/병해 Train {train_target}, Validation "
                    f"{validation_target} 수량을 카메라 분리로 충족할 수 없음"
                ),
                limiting_species=species,
            )
        validation = best[1]
        validation_set = set(validation)
        train = tuple(camera for camera in cameras if camera not in validation_set)
        train_result[species] = train
        validation_result[species] = validation
    if camera_overlap(train_result, validation_result):
        raise AssertionError("카메라 그룹 할당 중복")
    return CameraAssignment(
        feasible=True,
        train_cameras=train_result,
        validation_cameras=validation_result,
    )


def filtered_camera_maps_without_fixed_test(
    stats: ManifestStats,
) -> tuple[
    dict[tuple[str, str], Counter[str]],
    dict[tuple[str, str], Counter[str]],
]:
    health: dict[tuple[str, str], Counter[str]] = {}
    disease: dict[tuple[str, str], Counter[str]] = {}
    split_counts = stats.group_split_counts["species_camera"]
    for key, counts in stats.development_health_camera.items():
        if "test" in split_counts[key]:
            continue
        health[key] = Counter(counts)
        disease[key] = Counter(stats.camera_disease.get(key, Counter()))
    return health, disease


def group_candidate_rows(stats: ManifestStats) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, columns in GROUP_CANDIDATES.items():
        all_groups = stats.group_sizes[name]
        split_counts = stats.group_split_counts[name]
        multi = sum(1 for counts in split_counts.values() if len(counts) > 1)
        test_overlap = sum(
            1
            for counts in split_counts.values()
            if "test" in counts and any(
                split in counts for split in ("train", "validation")
            )
        )
        rows_sharing_test = sum(
            counts["train"] + counts["validation"]
            for counts in split_counts.values()
            if "test" in counts
        )
        rows.append(
            {
                "candidate": name,
                "columns": " + ".join(columns),
                "group_count": len(all_groups),
                "largest_group": max(all_groups.values(), default=0),
                "development_group_count": len(
                    stats.development_group_sizes[name]
                ),
                "development_largest_group": max(
                    stats.development_group_sizes[name].values(),
                    default=0,
                ),
                "current_multi_split_groups": multi,
                "fixed_test_overlap_groups": test_overlap,
                "development_rows_sharing_fixed_test_group": rows_sharing_test,
                "development_rows_sharing_rate": (
                    rows_sharing_test / stats.development_rows
                    if stats.development_rows
                    else 0.0
                ),
            }
        )
    return rows


def proportional_allocation(
    available: Counter[str],
    total: int,
    *,
    seed: int,
    parts: Sequence[Any],
) -> Counter[str]:
    available_total = sum(available.values())
    if total < 0 or total > available_total:
        raise ValueError("요청 수량이 가용 수량을 벗어났습니다")
    if total == 0:
        return Counter()
    result = Counter()
    fractions: list[tuple[float, str, str]] = []
    for label, count in available.items():
        exact = total * count / available_total
        floor = math.floor(exact)
        result[label] = floor
        fractions.append(
            (
                exact - floor,
                stable_hash(seed, *parts, label),
                label,
            )
        )
    remaining = total - sum(result.values())
    for _fraction, _digest, label in sorted(
        fractions, key=lambda item: (-item[0], item[1])
    )[:remaining]:
        result[label] += 1
    return result


def pilot_distribution(
    stats: ManifestStats,
    assignment: CameraAssignment,
    *,
    train_target: int,
    validation_target: int,
    seed: int,
) -> tuple[
    Counter[tuple[str, str, str]],
    Counter[tuple[str, str]],
    dict[str, Counter[tuple[str, str]]],
]:
    health_rows: Counter[tuple[str, str, str]] = Counter()
    disease_rows: Counter[tuple[str, str]] = Counter()
    capacities: dict[str, Counter[tuple[str, str]]] = {
        "train": Counter(),
        "validation": Counter(),
    }
    for split, cameras_by_species, target in (
        ("train", assignment.train_cameras, train_target),
        ("validation", assignment.validation_cameras, validation_target),
    ):
        for species in SPECIES:
            normal = 0
            abnormal = 0
            disease_available = Counter()
            for camera in cameras_by_species[species]:
                counts = stats.development_health_camera[(species, camera)]
                normal += counts["normal"]
                abnormal += counts["abnormal"]
                disease_available.update(
                    stats.camera_disease.get((species, camera), Counter())
                )
            capacities[split][(species, "normal")] = normal
            capacities[split][(species, "abnormal")] = abnormal
            if normal < target or abnormal < target:
                raise AssertionError("선택된 카메라의 파일럿 가용량 부족")
            health_rows[(split, species, "normal")] = target
            health_rows[(split, species, "abnormal")] = target
            allocation = proportional_allocation(
                disease_available,
                target,
                seed=seed,
                parts=(split, species),
            )
            for disease, count in allocation.items():
                disease_rows[(split, disease)] += count
    return health_rows, disease_rows, capacities


def distribution_rows(
    stats: HealthStats,
    manifest_stats: ManifestStats,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        scope: str,
        dimension: str,
        count: int,
        total: int,
        *,
        species: str = "",
        task: str = "",
        normality: str = "",
        disease_type: str = "",
        value: str = "",
        note: str = "",
    ) -> None:
        rows.append(
            {
                "scope": scope,
                "dimension": dimension,
                "species": species,
                "task": task,
                "normality": normality,
                "disease_type": disease_type,
                "value": value,
                "count": count,
                "scope_total": total,
                "rate": count / total if total else 0.0,
                "note": note,
            }
        )

    for value, count in sorted(stats.normality.items()):
        add(
            "full_json",
            "normality",
            count,
            stats.total,
            value=value,
            note="META.DBYHS_NORMALITY_ALTERNATIVE boolean normalized",
        )
    for (task, normality), count in sorted(stats.task_normality.items()):
        add(
            "full_json",
            "task_normality",
            count,
            stats.total,
            task=task,
            normality=normality,
        )
    for (species, normality), count in sorted(
        stats.species_normality.items()
    ):
        add(
            "full_json",
            "species_normality",
            count,
            sum(
                item
                for (item_species, _), item in stats.species_normality.items()
                if item_species == species
            ),
            species=species,
            normality=normality,
        )
    for disease, count in stats.disease.most_common():
        add(
            "full_json",
            "disease_type",
            count,
            sum(stats.disease.values()),
            disease_type=disease,
        )
    for (species, disease), count in sorted(
        stats.species_disease.items()
    ):
        add(
            "full_json",
            "species_disease",
            count,
            stats.disease[disease],
            species=species,
            disease_type=disease,
        )
    for value, count in sorted(stats.candidate_normality.items()):
        add(
            "growth_disease_json",
            "normality",
            count,
            stats.candidate_total,
            value=value,
            note="배양 제외; bbox 정제 전",
        )
    for (species, normality), count in sorted(
        stats.candidate_species_normality.items()
    ):
        add(
            "growth_disease_json",
            "species_normality",
            count,
            sum(
                item
                for (
                    item_species,
                    _,
                ), item in stats.candidate_species_normality.items()
                if item_species == species
            ),
            species=species,
            normality=normality,
            note="배양 제외; bbox 정제 전",
        )
    for (species, normality), count in sorted(
        manifest_stats.development_species_normality.items()
    ):
        add(
            "development_manifest",
            "species_normality",
            count,
            sum(
                item
                for (
                    item_species,
                    _,
                ), item in (
                    manifest_stats.development_species_normality.items()
                )
                if item_species == species
            ),
            species=species,
            normality=normality,
            note="정제 detection manifest; 고정 Test 제외",
        )
    for issue in (
        "invalid_normality",
        "task_normality_mismatch",
        "abnormal_missing_disease",
        "normal_has_disease",
    ):
        add(
            "full_json",
            "label_issue",
            stats.issue_counts[issue],
            stats.total,
            value=issue,
        )
    return rows


def disease_mapping_rows(stats: HealthStats) -> list[dict[str, Any]]:
    support: dict[str, list[str]] = defaultdict(list)
    for species, disease in stats.species_disease:
        support[disease].append(species)
    rows: list[dict[str, Any]] = []
    for disease in stats.disease:
        species_values = sorted(support[disease])
        total = stats.disease[disease]
        for species in species_values:
            count = stats.species_disease[(species, disease)]
            species_disease_total = sum(
                item
                for (item_species, _), item in stats.species_disease.items()
                if item_species == species
            )
            rows.append(
                {
                    "disease_type": disease,
                    "species": species,
                    "count": count,
                    "disease_total": total,
                    "share_within_disease": count / total if total else 0.0,
                    "species_disease_total": species_disease_total,
                    "share_within_species_disease": (
                        count / species_disease_total
                        if species_disease_total
                        else 0.0
                    ),
                    "supported_species_count": len(species_values),
                    "supported_species": "|".join(species_values),
                    "exclusive_to_species": len(species_values) == 1,
                    "shortcut_risk": (
                        "complete_species_shortcut"
                        if len(species_values) == 1
                        else "conditional_species_bias"
                    ),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            -int(row["disease_total"]),
            str(row["disease_type"]),
            str(row["species"]),
        ),
    )


def camera_bias_rows(
    full_stats: HealthStats,
    manifest_stats: ManifestStats,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    camera_purity = group_purity(
        manifest_stats.development_health_camera
    )
    date_purity = group_purity(manifest_stats.development_health_date)
    for analysis_type, result in (
        ("development_health_species_camera", camera_purity),
        ("development_health_species_date", date_purity),
    ):
        rows.append(
            {
                "analysis_type": analysis_type,
                "species": "",
                "disease_type": "",
                "group_count": result["groups"],
                "normal_only_groups": result["normal_only"],
                "abnormal_only_groups": result["abnormal_only"],
                "mixed_groups": result["mixed"],
                "total_count": manifest_stats.development_rows,
                "support_count": result["groups"],
                "top_group": "",
                "top_group_count": "",
                "top_group_share": "",
                "hhi": "",
                "majority_lookup_accuracy": result["majority_accuracy"],
                "note": "진단용 metadata lookup; 모델 성능이 아님",
            }
        )
    for species in SPECIES:
        species_camera_groups = {
            key: counts
            for key, counts in manifest_stats.development_health_camera.items()
            if key[0] == species
        }
        species_date_groups = {
            key: counts
            for key, counts in manifest_stats.development_health_date.items()
            if key[0] == species
        }
        for analysis_type, groups in (
            ("development_health_species_camera_by_species", species_camera_groups),
            ("development_health_species_date_by_species", species_date_groups),
        ):
            result = group_purity(groups)
            rows.append(
                {
                    "analysis_type": analysis_type,
                    "species": species,
                    "disease_type": "",
                    "group_count": result["groups"],
                    "normal_only_groups": result["normal_only"],
                    "abnormal_only_groups": result["abnormal_only"],
                    "mixed_groups": result["mixed"],
                    "total_count": sum(
                        sum(counts.values()) for counts in groups.values()
                    ),
                    "support_count": result["groups"],
                    "top_group": "",
                    "top_group_count": "",
                    "top_group_share": "",
                    "hhi": "",
                    "majority_lookup_accuracy": result["majority_accuracy"],
                    "note": "Test 제외; 품종별 배경 shortcut 진단",
                }
            )
    keys = sorted(manifest_stats.development_disease_camera)
    for species, disease in keys:
        camera = concentration(
            manifest_stats.development_disease_camera[(species, disease)]
        )
        date = concentration(
            manifest_stats.development_disease_date[(species, disease)]
        )
        for analysis_type, result in (
            ("development_disease_camera", camera),
            ("development_disease_date", date),
        ):
            rows.append(
                {
                    "analysis_type": analysis_type,
                    "species": species,
                    "disease_type": disease,
                    "group_count": "",
                    "normal_only_groups": "",
                    "abnormal_only_groups": "",
                    "mixed_groups": "",
                    "total_count": result["total"],
                    "support_count": result["support"],
                    "top_group": result["top_key"],
                    "top_group_count": result["top_count"],
                    "top_group_share": result["top_share"],
                    "hhi": result["hhi"],
                    "majority_lookup_accuracy": "",
                    "note": (
                        "Test 제외; 카메라/날짜 집중도. 높은 값은 shortcut 위험"
                    ),
                }
            )
    full_camera = group_purity(full_stats.candidate_health_camera)
    full_date = group_purity(full_stats.candidate_health_date)
    for analysis_type, result in (
        ("all_health_candidates_species_camera", full_camera),
        ("all_health_candidates_species_date", full_date),
    ):
        rows.append(
            {
                "analysis_type": analysis_type,
                "species": "",
                "disease_type": "",
                "group_count": result["groups"],
                "normal_only_groups": result["normal_only"],
                "abnormal_only_groups": result["abnormal_only"],
                "mixed_groups": result["mixed"],
                "total_count": full_stats.candidate_total,
                "support_count": result["groups"],
                "top_group": "",
                "top_group_count": "",
                "top_group_share": "",
                "hhi": "",
                "majority_lookup_accuracy": result["majority_accuracy"],
                "note": "전체 생육+병해 JSON; Test 포함",
            }
        )
    return rows


def atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def format_count(value: int) -> str:
    return f"{value:,}"


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def health_input_strategy_markdown(
    health_stats: HealthStats,
    manifest_stats: ManifestStats,
) -> str:
    camera = group_purity(manifest_stats.development_health_camera)
    date = group_purity(manifest_stats.development_health_date)
    return f"""# 버섯 건강 체크 v1 입력 이미지 전략

## 전제

- 현재 bbox는 병반이 아니라 버섯 객체/군집 bbox다.
- 건강 라벨은 JSON 이미지 단위이며 병반 위치 annotation은 없다.
- 생육 폴더는 전부 정상, 병해 폴더는 전부 비정상이라 촬영 프로토콜이 라벨과 완전히 결합돼 있다.
- Test 제외 개발 풀에서 품종+카메라 {camera['groups']}개 중 정상 전용 {camera['normal_only']}개, 혼합 {camera['mixed']}개다.
- 품종+날짜 다수 상태 lookup은 {format_percent(date['majority_accuracy'])}를 재현한다. 이는 모델 성능이 아니라 날짜 shortcut 위험의 진단치다.

## 입력 후보 비교

| 입력 | 병해 문맥 | 배경 shortcut | 주요 위험 | 판단 |
| --- | --- | --- | --- | --- |
| 전체 원본 이미지 | 가장 많음 | 가장 큼 | 재배실·카메라·작업 폴더 배경을 암기 | 기준선 ablation만 |
| 모든 bbox union crop | 버섯 전체 보존 | 낮음 | 배지·경계 문맥이 부족할 수 있음 | padding과 함께 사용 |
| union crop + 이미지 크기 10~20% padding | 버섯·배지 문맥 균형 | 전체 이미지보다 낮음 | padding이 너무 크면 배경 재유입 | **권장** |
| 가장 큰 bbox crop | 일부 개체만 | 낮음 | 다른 개체·군집·주변 증상을 누락 | 비권장 |

## 권장안

모든 유효 버섯 bbox의 union을 계산하고 이미지 폭·높이 기준 **15% context padding**을 각 방향으로 적용한 뒤 영상 경계로 clip한다.

- Train과 추론에서 동일한 crop 정책을 사용한다.
- 다중 bbox를 모두 포함해 JSON 단위 건강 라벨과 입력 범위를 맞춘다.
- 배지와 주변 군집 문맥은 보존하되 재배실 전체 배경은 줄인다.
- 10%, 15%, 20% padding 및 전체 이미지 기준선을 카메라 홀드아웃 Validation에서 비교한다.
- 검출 bbox가 없거나 crop 신뢰성이 낮으면 `UNCERTAIN`을 반환하거나 전체 이미지 보조 경로를 별도 검증한다.

## 병반 위치 출력 제외

병해명은 이미지 수준 메타데이터이고 bbox는 버섯 객체/군집을 표시한다. 병반 bbox·mask·point가 없으므로 모델이 병반 위치를 학습하거나 API에서 위치를 반환할 감독 신호가 없다. 건강 v1은 이미지/crop 수준 상태만 반환한다.
"""


def health_model_scope_markdown(
    health_stats: HealthStats,
    manifest_stats: ManifestStats,
) -> str:
    global_disease = (
        max(manifest_stats.development_disease.values())
        / sum(manifest_stats.development_disease.values())
    )
    species_disease = majority_lookup_accuracy(
        manifest_stats.disease_by_species
    )
    return f"""# 버섯 건강 체크 v1 모델 범위 결정

## 확정 기능

1. 기존 YOLO11n: 5품종, bbox, 개수, confidence
2. 건강 1단계: `HEALTHY` / `DISEASE_SUSPECTED` / `UNCERTAIN`
3. 건강 2단계: `DISEASE_SUSPECTED`일 때만 품종 조건부 병해 종류 후보와 confidence

수확 적기, 생육 단계, 예상 수확일, 병반 위치, 해충, 실제 갓·대 크기는 v1에서 제외한다.

## 모델 후보 비교

| 후보 | 장점 | 핵심 위험 | v1 판단 |
| --- | --- | --- | --- |
| 전체 이미지 이진 분류 | 문맥 최대 | 작업·카메라·재배실 암기 | ablation만 |
| context crop 이진 분류 | 객체와 배지 문맥 균형 | crop 정책 의존 | **권장 입력** |
| 품종별 이진 모델 5개 | 품종별 외형 최적화 | 데이터·운영 분절 | 보조 실험 |
| 공통 이진 모델 1개 | 데이터 공유·배포 단순 | 품종별 성능 차이 | **권장 1단계** |
| 건강+병해 단일 분류 | 단일 호출 | 불균형·품종 shortcut·오류 전파 불투명 | 비권장 |
| 이진 후 병해 2단계 | 건강과 병해 후보를 분리 | 1단계 오류 전파 | **권장 구조** |
| species+disease 복합 클래스 | 허용 조합 표현 쉬움 | 종 외형만으로 병해를 맞히는 shortcut | 단일 flat head는 비권장 |

## 병해 종류 편향

- 전역 다수 병해만 고르면 {format_percent(global_disease)}다.
- 품종별 다수 병해 lookup은 {format_percent(species_disease)}다.
- 세균성검은썩음병은 팽이에만, 솜털곰팡이병은 양송이에만 존재한다.
- 따라서 병해 종류 모델은 품종 외형을 지름길로 사용할 수 있다.

## 권장 구조

### 1단계

하나의 공통 이진 classifier를 15% context union crop으로 학습한다. 품종 균형 sampling과 품종별 recall/F1을 보고한다. Validation에서 calibration한 두 threshold 사이를 `UNCERTAIN`으로 둔다.

### 2단계

공유 backbone과 **품종별 disease head 또는 species-conditioned class mask**를 사용한다. 알려진 품종에서 가능한 병해만 후보로 반환하고 품종별 macro F1·recall·혼동행렬을 평가한다. 단일 전역 flat class 정확도만 보고하지 않는다.

## 개발 가능성

- 건강 이진 모델: 수량상 개발 가능. 생육 정상 {format_count(health_stats.candidate_normality['normal'])}개, 병해 {format_count(health_stats.candidate_normality['abnormal'])}개다.
- 병해 종류 모델: 5종 모두 수천 건으로 후보 모델 개발은 가능하지만 최대/최소 불균형과 품종·카메라·날짜 편향 때문에 제한적 후보 출력으로 시작해야 한다.
- 병반 위치 모델: 위치 annotation 부재로 개발 근거가 없다.

## API 상태

```json
{{
  "health_status": "HEALTHY | DISEASE_SUSPECTED | UNCERTAIN",
  "health_confidence": 0.0,
  "disease_candidates": [],
  "unsupported": [
    "harvest_readiness",
    "growth_stage",
    "expected_harvest_date",
    "physical_cap_stipe_size",
    "pest_detection"
  ]
}}
```

병해 후보는 `DISEASE_SUSPECTED`일 때만 채우며 threshold는 Test가 아닌 Validation에서 결정한다.
"""


def health_split_markdown(
    manifest_stats: ManifestStats,
    assignment: CameraAssignment,
    fixed_test_assignment: CameraAssignment,
    *,
    seed: int,
) -> str:
    candidate_rows = group_candidate_rows(manifest_stats)
    table = "\n".join(
        "| {columns} | {group_count:,} | {largest_group:,} | "
        "{current_multi_split_groups:,} | {fixed_test_overlap_groups:,} | "
        "{development_rows_sharing_rate:.2%} |".format(**row)
        for row in candidate_rows
    )
    free_health, _free_disease = filtered_camera_maps_without_fixed_test(
        manifest_stats
    )
    fixed_test_capacity_table = "\n".join(
        f"| {species} | "
        f"{sum(1 for item_species, _camera in free_health if item_species == species)} | "
        f"{sum(counts['normal'] for (item_species, _camera), counts in free_health.items() if item_species == species):,} | "
        f"{sum(counts['abnormal'] for (item_species, _camera), counts in free_health.items() if item_species == species):,} |"
        for species in SPECIES
    )
    return f"""# 버섯 건강 체크 v1 누수 없는 분할 전략

## 원칙

- seed: `{seed}`
- 동일 그룹은 하나의 split에만 둔다.
- 기존 고정 Test {format_count(manifest_stats.test_rows)}장은 재배정·선정·학습·모델 선택에 사용하지 않는다.
- 아래 Test 수치는 그룹 중복 진단에만 사용했다.

## 후보 비교

| 그룹 키 | 전체 그룹 | 최대 그룹 | 현재 다중 split 그룹 | 고정 Test와 겹친 그룹 | 개발 행 중 Test 그룹 공유 |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

## 판단

1. `species + camera_id`
   - 카메라 배경 누수를 가장 직접적으로 차단하므로 **건강 모델 Train/Validation 파일럿 권장 키**다.
   - 현재 고정 Test와 70개 그룹이 겹쳐 기존 3-way split 전체에는 소급 적용할 수 없다.
2. `species + camera_id + capture_date`
   - 기존 split에서 그룹 중복 0이지만 같은 카메라가 다른 날짜로 양쪽에 남아 배경 암기를 막지 못한다.
3. `camera_id`
   - ID 1~20이 품종 사이에서 반복돼 동일 물리 카메라인지 불명확하고 모든 그룹이 고정 Test와 겹친다.
4. `species + capture_date`
   - 최대 그룹이 크고 카메라가 섞이며 날짜가 건강 라벨과 강하게 결합돼 있다.

## Test 제약

고정 Test 카메라와 겹치는 개발 그룹을 모두 제외한 뒤 `species + camera_id` 3-way 분리를 시도하면 결과는 **{'가능' if fixed_test_assignment.feasible else '불가능'}**이다.

사유: `{fixed_test_assignment.reason or '-'}`

| 품종 | Test와 겹치지 않는 카메라 | 정상 가용 | 병해 가용 |
| --- | ---: | ---: | ---: |
{fixed_test_capacity_table}

특히 표고는 고정 Test와 겹치지 않는 카메라가 정상 전용 1개뿐이어서 병해 학습/검증 수량을 만들 수 없다. 따라서 기존 Test를 카메라 홀드아웃 Test라고 주장하지 않는다.

## 권장 운영

- 이번 6,000장 파일럿: 기존 train+validation 후보만 사용해 `species + camera_id`로 Train/Validation 완전 분리
- 모델 선택: 위 Validation만 사용
- 기존 Test: 봉인 유지; 현재 카메라 홀드아웃 성능 주장에는 사용하지 않음
- 최종 배포 평가: 이후 별도 외부 스마트폰 또는 새 카메라 기반 Test를 승인받아 구성
- Train/Validation 카메라 중복: `{'0' if assignment.feasible else '구성 불가'}`
"""


def health_pilot_markdown(
    manifest_stats: ManifestStats,
    assignment: CameraAssignment,
    health_rows: Counter[tuple[str, str, str]],
    disease_rows: Counter[tuple[str, str]],
    capacities: Mapping[str, Counter[tuple[str, str]]],
    *,
    seed: int,
    train_target: int,
    validation_target: int,
) -> str:
    health_table = "\n".join(
        f"| {split} | {species} | {status} | {count:,} | "
        f"{capacities[split][(species, status)]:,} |"
        for (split, species, status), count in sorted(health_rows.items())
    )
    disease_table = "\n".join(
        f"| {split} | {disease} | {count:,} |"
        for (split, disease), count in sorted(disease_rows.items())
    )
    camera_table = "\n".join(
        f"| {split} | {species} | {', '.join(cameras)} | {len(cameras)} |"
        for species in SPECIES
        for split, cameras in (
            ("train", assignment.train_cameras.get(species, ())),
            ("validation", assignment.validation_cameras.get(species, ())),
        )
    )
    return f"""# 버섯 건강 체크 v1 6,000장 파일럿 가능성

## 결론

정확한 구성이 **{'가능' if assignment.feasible else '불가능'}**하다.

- Test 사용: 0장
- Test 제외 개발 후보: {format_count(manifest_stats.development_rows)}장
- 제외한 고정 Test: {format_count(manifest_stats.test_rows)}장
- seed: `{seed}`
- 그룹 키: `species + camera_id`
- Train/Validation 그룹 중복: {len(camera_overlap(assignment.train_cameras, assignment.validation_cameras))}개
- 이미지는 추출하지 않았고 manifest도 생성하지 않았다.

## 목표와 가용량

| split | 품종 | 상태 | 계획 수량 | 선택 카메라 가용량 |
| --- | --- | --- | ---: | ---: |
{health_table}

Train은 품종별 정상/병해 각 {train_target}장, Validation은 각 {validation_target}장으로 총 5,000/1,000장이다.

## 결정적 카메라 배정

| split | 품종 | camera_id | 카메라 수 |
| --- | --- | --- | ---: |
{camera_table}

## 병해 종류 계획 분포

카메라별 가용 분포를 비례 배분한 feasibility quota다. 아직 실제 이미지 선택 manifest가 아니다.

| split | 병해 종류 | 계획 이미지 |
| --- | --- | ---: |
{disease_table}

모든 지원 병해가 Train과 Validation에 남도록 카메라 조합을 우선했다.

## 제한

이 파일럿은 Train/Validation 카메라 홀드아웃 가능성을 증명한다. 기존 고정 Test와는 같은 `species+camera_id`가 존재하므로 최종 3-way 카메라 독립 평가로 해석하지 않는다.
"""


def write_reports(
    output_dir: Path,
    health_stats: HealthStats,
    manifest_stats: ManifestStats,
    assignment: CameraAssignment,
    fixed_test_assignment: CameraAssignment,
    *,
    seed: int,
    train_target: int,
    validation_target: int,
) -> None:
    atomic_write_csv(
        output_dir / "health_label_distribution.csv",
        (
            "scope",
            "dimension",
            "species",
            "task",
            "normality",
            "disease_type",
            "value",
            "count",
            "scope_total",
            "rate",
            "note",
        ),
        distribution_rows(health_stats, manifest_stats),
    )
    atomic_write_csv(
        output_dir / "health_disease_species_mapping.csv",
        (
            "disease_type",
            "species",
            "count",
            "disease_total",
            "share_within_disease",
            "species_disease_total",
            "share_within_species_disease",
            "supported_species_count",
            "supported_species",
            "exclusive_to_species",
            "shortcut_risk",
        ),
        disease_mapping_rows(health_stats),
    )
    atomic_write_csv(
        output_dir / "health_camera_bias_analysis.csv",
        (
            "analysis_type",
            "species",
            "disease_type",
            "group_count",
            "normal_only_groups",
            "abnormal_only_groups",
            "mixed_groups",
            "total_count",
            "support_count",
            "top_group",
            "top_group_count",
            "top_group_share",
            "hhi",
            "majority_lookup_accuracy",
            "note",
        ),
        camera_bias_rows(health_stats, manifest_stats),
    )
    health_rows, disease_rows, capacities = pilot_distribution(
        manifest_stats,
        assignment,
        train_target=train_target,
        validation_target=validation_target,
        seed=seed,
    )
    atomic_write_text(
        output_dir / "health_input_strategy.md",
        health_input_strategy_markdown(health_stats, manifest_stats),
    )
    atomic_write_text(
        output_dir / "health_model_scope_decision.md",
        health_model_scope_markdown(health_stats, manifest_stats),
    )
    atomic_write_text(
        output_dir / "health_split_strategy.md",
        health_split_markdown(
            manifest_stats,
            assignment,
            fixed_test_assignment,
            seed=seed,
        ),
    )
    atomic_write_text(
        output_dir / "health_pilot_feasibility.md",
        health_pilot_markdown(
            manifest_stats,
            assignment,
            health_rows,
            disease_rows,
            capacities,
            seed=seed,
            train_target=train_target,
            validation_target=validation_target,
        ),
    )


def snapshot_files(paths: Iterable[Path]) -> dict[Path, tuple[int, int]]:
    return {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in paths
        if path.is_file()
    }


def snapshot_tree(root: Path) -> dict[str, tuple[int, int]]:
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


def assert_snapshot_unchanged(
    snapshot: Mapping[Path, tuple[int, int]],
) -> None:
    changed = [
        path.name
        for path, expected in snapshot.items()
        if not path.is_file()
        or (path.stat().st_size, path.stat().st_mtime_ns) != expected
    ]
    if changed:
        raise RuntimeError("읽기 전용 원본 변경 감지: " + ", ".join(changed))


def assert_tree_unchanged(
    root: Path,
    snapshot: Mapping[str, tuple[int, int]],
) -> None:
    if snapshot_tree(root) != dict(snapshot):
        raise RuntimeError("기존 artifacts 변경 감지")


def assert_deidentified(paths: Iterable[Path]) -> None:
    hits: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8-sig")
        if any(marker in text for marker in LOCAL_PATH_MARKERS):
            hits.append(path.name)
    if hits:
        raise RuntimeError("로컬 절대경로 노출: " + ", ".join(sorted(hits)))


def discover_source_label_archives() -> list[Path]:
    refs, issues, _directories = base.discover_archives(os.environ)
    labels = [ref.path for ref in refs if ref.kind == "label"]
    if issues or len(labels) != 10:
        raise RuntimeError(
            "환경변수에서 원본 라벨 ZIP 10개를 안전하게 확인하지 못했습니다"
        )
    return labels


def run(args: argparse.Namespace) -> dict[str, Any]:
    label_archives = discover_source_label_archives()
    source_snapshot = snapshot_files(label_archives)
    artifacts_snapshot = snapshot_tree(PROJECT_ROOT / "artifacts")
    input_snapshot = snapshot_files((args.quality_db, args.manifest))
    connection = open_complete_quality_db(args.quality_db)
    try:
        health_stats = stream_health_statistics(connection)
    finally:
        connection.close()
    manifest_stats = analyze_detection_manifest(args.manifest)
    assignment = choose_camera_holdout(
        manifest_stats.development_health_camera,
        manifest_stats.camera_disease,
        seed=args.seed,
        train_target=args.train_per_species_status,
        validation_target=args.validation_per_species_status,
    )
    if not assignment.feasible:
        raise RuntimeError("6,000장 파일럿 구성 불가: " + assignment.reason)
    free_health, free_disease = filtered_camera_maps_without_fixed_test(
        manifest_stats
    )
    fixed_test_assignment = choose_camera_holdout(
        free_health,
        free_disease,
        seed=args.seed,
        train_target=args.train_per_species_status,
        validation_target=args.validation_per_species_status,
    )
    write_reports(
        args.output_dir,
        health_stats,
        manifest_stats,
        assignment,
        fixed_test_assignment,
        seed=args.seed,
        train_target=args.train_per_species_status,
        validation_target=args.validation_per_species_status,
    )
    outputs = [args.output_dir / name for name in OUTPUT_FILES]
    assert_deidentified(outputs)
    assert_snapshot_unchanged(source_snapshot)
    assert_snapshot_unchanged(input_snapshot)
    assert_tree_unchanged(PROJECT_ROOT / "artifacts", artifacts_snapshot)
    camera_bias = group_purity(manifest_stats.development_health_camera)
    date_bias = group_purity(manifest_stats.development_health_date)
    return {
        "full_json": health_stats.total,
        "full_normal": health_stats.normality["normal"],
        "full_abnormal": health_stats.normality["abnormal"],
        "candidate_normal": health_stats.candidate_normality["normal"],
        "candidate_abnormal": health_stats.candidate_normality["abnormal"],
        "contradictory_records": health_stats.contradictory_records,
        "development_rows": manifest_stats.development_rows,
        "test_rows_excluded": manifest_stats.test_rows,
        "camera_majority_lookup": camera_bias["majority_accuracy"],
        "date_majority_lookup": date_bias["majority_accuracy"],
        "pilot_feasible": assignment.feasible,
        "pilot_camera_overlap": len(
            camera_overlap(
                assignment.train_cameras,
                assignment.validation_cameras,
            )
        ),
        "fixed_test_three_way_camera_feasible": fixed_test_assignment.feasible,
        "image_bytes_read": False,
        "images_extracted": 0,
        "model_training": False,
        "source_zip_modified": False,
        "artifacts_modified": False,
        "output_files": [f"reports/{path.name}" for path in outputs],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
