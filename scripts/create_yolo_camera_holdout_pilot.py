#!/usr/bin/env python3
"""Create a camera-disjoint 6,000-image YOLO pilot dataset.

Only detection-manifest Train/Validation candidates are considered.  Camera
groups are species + camera_id and are assigned wholly to one new split.
Selected image members alone are read from read-only ZIPs; Test, full archive
extraction, full conversion, and model training are forbidden.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import itertools
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import audit_mushroom_labels as audit
import inspect_aihub_archives as base
import create_yolo_pilot_dataset as pilot
import create_yolo_smoke_dataset as smoke


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "detection_dataset_manifest.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "yolo_pilot_camera_holdout"
)
SMOKE_DIR = PROJECT_ROOT / "artifacts" / "yolo_smoke"
PILOT_DIR = PROJECT_ROOT / "artifacts" / "yolo_pilot"
DEFAULT_SEED = 20260726
DEFAULT_TRAIN_PER_GROUP = 500
DEFAULT_VALIDATION_PER_GROUP = 100
DEFAULT_MAX_TOTAL_IMAGES = 6_000
DEFAULT_MAX_OVERLAY_IMAGES = 100
VALIDATION_CAMERAS_PER_SPECIES = 4
CAMERA_MANIFEST_COLUMNS = (
    *pilot.PILOT_MANIFEST_COLUMNS,
    "camera_holdout_group",
    "source_manifest_split",
)


@dataclass
class CameraStatistics:
    by_camera: dict[tuple[str, str], Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    strata_by_camera: dict[
        tuple[str, str], Counter[tuple[str, str, str]]
    ] = field(default_factory=lambda: defaultdict(Counter))
    candidate_rows: int = 0
    skipped_test_rows: int = 0


@dataclass
class CameraAssignment:
    feasible: bool
    validation_cameras: dict[str, tuple[str, ...]]
    train_cameras: dict[str, tuple[str, ...]]
    pool_counts: dict[tuple[str, str], Counter[str]]
    reason: str = ""
    closest_shortfall: int = 0


@dataclass(frozen=True)
class HoldoutCandidate:
    rank: int
    unique_key: str
    source_manifest_split: str
    row: dict[str, str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "species+camera_id를 Train/Validation 사이에 완전히 분리한 "
            "6,000장 YOLO 파일럿 데이터셋을 생성합니다."
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
        type=pilot.bounded_total,
        default=DEFAULT_MAX_TOTAL_IMAGES,
    )
    parser.add_argument(
        "--max-overlay-images",
        type=pilot.bounded_overlay,
        default=DEFAULT_MAX_OVERLAY_IMAGES,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/yolo_pilot_camera_holdout"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.manifest = pilot.project_path(args.manifest)
    args.output_dir = pilot.project_path(args.output_dir)
    artifacts_root = (PROJECT_ROOT / "artifacts").resolve()
    if not args.output_dir.is_relative_to(artifacts_root):
        parser.error("--output-dir은 프로젝트의 artifacts 아래여야 합니다")
    for protected in (SMOKE_DIR.resolve(), PILOT_DIR.resolve()):
        if (
            args.output_dir == protected
            or args.output_dir.is_relative_to(protected)
            or protected.is_relative_to(args.output_dir)
        ):
            parser.error(
                "--output-dir은 기존 yolo_smoke/yolo_pilot과 분리해야 합니다"
            )
    pilot.validate_requested_limits(
        args.train_per_species_task,
        args.validation_per_species_task,
        args.max_total_images,
        args.max_overlay_images,
    )
    return args


def camera_sort_key(camera_id: str) -> tuple[int, int | str]:
    try:
        return 0, int(camera_id)
    except ValueError:
        return 1, camera_id


def stream_camera_statistics(manifest_path: Path) -> CameraStatistics:
    stats = CameraStatistics()
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = smoke.REQUIRED_MANIFEST_COLUMNS - set(
            reader.fieldnames or ()
        )
        if "camera_id" not in set(reader.fieldnames or ()):
            missing = set(missing) | {"camera_id"}
        if missing:
            raise ValueError(
                "manifest 필수 컬럼 누락: " + ", ".join(sorted(missing))
            )
        for row in reader:
            if row["split"] == "test":
                stats.skipped_test_rows += 1
                continue
            pilot.validate_source_row(row)
            species = row["species"]
            camera_id = row["camera_id"] or "<missing>"
            task = row["task"]
            disease = row["disease_type"] or "<missing>"
            normality = row["normality"] or "<missing>"
            camera_key = (species, camera_id)
            stats.by_camera[camera_key]["total"] += 1
            stats.by_camera[camera_key][f"task:{task}"] += 1
            if task == "병해":
                stats.by_camera[camera_key][f"disease:{disease}"] += 1
            stats.strata_by_camera[camera_key][
                (task, normality, disease)
            ] += 1
            stats.candidate_rows += 1
    return stats


def validation_camera_count(camera_count: int) -> int:
    if camera_count < 2:
        return 0
    return min(VALIDATION_CAMERAS_PER_SPECIES, camera_count - 1)


def sum_camera_counts(
    stats: CameraStatistics,
    species: str,
    cameras: Sequence[str],
) -> Counter[str]:
    result: Counter[str] = Counter()
    for camera_id in cameras:
        result.update(stats.by_camera[(species, camera_id)])
    return result


def assignment_score(
    total: Mapping[str, int],
    validation: Mapping[str, int],
    *,
    seed: int,
    species: str,
    subset: tuple[str, ...],
    validation_fraction: float,
) -> tuple[int, float, str]:
    train = Counter(total) - Counter(validation)
    missing_disease_sides = 0
    score = 0.0
    for feature, count in total.items():
        if feature == "total":
            continue
        if feature.startswith("disease:"):
            weight = 10.0
            if validation.get(feature, 0) == 0 or train.get(feature, 0) == 0:
                missing_disease_sides += 1
        elif feature.startswith("task:"):
            weight = 5.0
        else:
            continue
        target = count * validation_fraction
        score += weight * (
            (validation.get(feature, 0) - target) / max(target, 1.0)
        ) ** 2
    tie = smoke.stable_score(seed, species, *subset)
    return missing_disease_sides, score, tie


def choose_camera_assignment(
    stats: CameraStatistics,
    *,
    train_per_species_task: int,
    validation_per_species_task: int,
    seed: int,
) -> CameraAssignment:
    validation_cameras: dict[str, tuple[str, ...]] = {}
    train_cameras: dict[str, tuple[str, ...]] = {}
    pool_counts: dict[tuple[str, str], Counter[str]] = {}
    reasons: list[str] = []
    total_shortfall = 0
    validation_fraction = validation_per_species_task / (
        train_per_species_task + validation_per_species_task
    )
    for species in pilot.SPECIES:
        cameras = sorted(
            (
                camera_id
                for candidate_species, camera_id in stats.by_camera
                if candidate_species == species
            ),
            key=camera_sort_key,
        )
        choose_count = validation_camera_count(len(cameras))
        if choose_count == 0:
            reasons.append(f"{species}: 카메라가 2개 미만")
            continue
        total = sum_camera_counts(stats, species, cameras)
        best_feasible: tuple[
            tuple[int, float, str], tuple[str, ...], Counter[str]
        ] | None = None
        best_closest: tuple[
            int, tuple[int, float, str], tuple[str, ...], Counter[str]
        ] | None = None
        for subset in itertools.combinations(cameras, choose_count):
            validation = sum_camera_counts(stats, species, subset)
            train = total - validation
            shortfall = sum(
                (
                    max(
                        0,
                        validation_per_species_task
                        - validation[f"task:{task}"],
                    )
                    + max(
                        0,
                        train_per_species_task - train[f"task:{task}"],
                    )
                )
                for task in pilot.TASKS
            )
            score = assignment_score(
                total,
                validation,
                seed=seed,
                species=species,
                subset=subset,
                validation_fraction=validation_fraction,
            )
            closest_rank = (shortfall, score, subset, validation)
            if best_closest is None or closest_rank[:3] < best_closest[:3]:
                best_closest = closest_rank
            if shortfall == 0:
                feasible_rank = (score, subset, validation)
                if (
                    best_feasible is None
                    or feasible_rank[:2] < best_feasible[:2]
                ):
                    best_feasible = feasible_rank
        if best_feasible is None:
            assert best_closest is not None
            shortfall, _score, subset, validation = best_closest
            total_shortfall += shortfall
            reasons.append(
                f"{species}: 정확 수량 불가, 최소 부족 {shortfall}장"
            )
        else:
            _score, subset, validation = best_feasible
        validation_set = set(subset)
        train_subset = tuple(
            camera for camera in cameras if camera not in validation_set
        )
        validation_cameras[species] = tuple(subset)
        train_cameras[species] = train_subset
        pool_counts[("validation", species)] = Counter(validation)
        pool_counts[("train", species)] = total - validation
    feasible = not reasons
    return CameraAssignment(
        feasible=feasible,
        validation_cameras=validation_cameras,
        train_cameras=train_cameras,
        pool_counts=pool_counts,
        reason="; ".join(reasons),
        closest_shortfall=total_shortfall,
    )


def assigned_split(
    assignment: CameraAssignment,
    species: str,
    camera_id: str,
) -> str:
    if camera_id in assignment.validation_cameras.get(species, ()):
        return "validation"
    if camera_id in assignment.train_cameras.get(species, ()):
        return "train"
    raise ValueError(f"할당되지 않은 카메라: {species}/{camera_id}")


def camera_holdout_stratum_quotas(
    stats: CameraStatistics,
    assignment: CameraAssignment,
    *,
    train_per_species_task: int,
    validation_per_species_task: int,
    seed: int,
) -> dict[tuple[str, str, str, str, str], int]:
    group_strata: dict[
        tuple[str, str, str], Counter[tuple[str, str]]
    ] = defaultdict(Counter)
    for (species, camera_id), strata in stats.strata_by_camera.items():
        split = assigned_split(assignment, species, camera_id)
        for (task, normality, disease), count in strata.items():
            group_strata[(split, species, task)][
                (normality, disease)
            ] += count
    targets = pilot.requested_group_targets(
        train_per_species_task, validation_per_species_task
    )
    quotas: dict[tuple[str, str, str, str, str], int] = {}
    for group, target in sorted(targets.items()):
        if group not in group_strata:
            raise ValueError(f"후보가 없는 할당 그룹: {group}")
        allocation = pilot.proportional_quotas(
            group_strata[group],
            target,
            seed=seed,
            group=group,
        )
        for (normality, disease), quota in allocation.items():
            if quota:
                quotas[(*group, normality, disease)] = quota
    return quotas


def select_camera_holdout_rows(
    manifest_path: Path,
    stats: CameraStatistics,
    assignment: CameraAssignment,
    *,
    train_per_species_task: int,
    validation_per_species_task: int,
    seed: int,
    max_total_images: int,
) -> list[dict[str, str]]:
    if not assignment.feasible:
        raise ValueError(
            "정확한 camera holdout 구성이 불가능합니다: "
            + assignment.reason
        )
    expected_total = pilot.validate_requested_limits(
        train_per_species_task,
        validation_per_species_task,
        max_total_images,
        1,
    )
    quotas = camera_holdout_stratum_quotas(
        stats,
        assignment,
        train_per_species_task=train_per_species_task,
        validation_per_species_task=validation_per_species_task,
        seed=seed,
    )
    heaps: dict[
        tuple[str, str, str, str, str],
        list[tuple[int, int, HoldoutCandidate]],
    ] = defaultdict(list)
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            if row["split"] == "test":
                continue
            pilot.validate_source_row(row)
            source_split = row["split"]
            species = row["species"]
            camera_id = row["camera_id"] or "<missing>"
            new_split = assigned_split(
                assignment, species, camera_id
            )
            normality, disease = pilot.source_stratum(row)
            stratum_key = (
                new_split,
                species,
                row["task"],
                normality,
                disease,
            )
            limit = quotas.get(stratum_key, 0)
            if limit <= 0:
                continue
            unique_key = smoke.candidate_unique_key(row)
            rank = int(smoke.stable_score(seed, unique_key), 16)
            retained = {
                column: row.get(column, "")
                for column in pilot.PILOT_SOURCE_COLUMNS
            }
            retained["split"] = new_split
            candidate = HoldoutCandidate(
                rank=rank,
                unique_key=unique_key,
                source_manifest_split=source_split,
                row=retained,
            )
            heap = heaps[stratum_key]
            item = (-rank, line_number, candidate)
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, item)
    selected: list[dict[str, str]] = []
    source_split_by_key: dict[str, str] = {}
    for stratum_key, quota in sorted(quotas.items()):
        heap = heaps.get(stratum_key, [])
        if len(heap) != quota:
            raise RuntimeError(
                f"{stratum_key} 선택 수 {len(heap)}개(요청 {quota}개)"
            )
        for _negative, _line, candidate in heap:
            selected.append(candidate.row)
            source_split_by_key[candidate.unique_key] = (
                candidate.source_manifest_split
            )
    selected.sort(
        key=lambda row: (
            0 if row["split"] == "train" else 1,
            smoke.CLASS_NAMES_INV[row["species"]],
            0 if row["task"] == "생육" else 1,
            smoke.stable_score(seed, smoke.candidate_unique_key(row)),
        )
    )
    for row in selected:
        row["_source_manifest_split"] = source_split_by_key[
            smoke.candidate_unique_key(row)
        ]
    keys = [smoke.candidate_unique_key(row) for row in selected]
    if len(keys) != len(set(keys)):
        raise RuntimeError("camera holdout 선택 결과에 중복 이미지가 있습니다")
    if len(selected) != expected_total:
        raise RuntimeError(
            f"선택 수 {len(selected)}개(기대값 {expected_total}개)"
        )
    target_counts = pilot.requested_group_targets(
        train_per_species_task, validation_per_species_task
    )
    actual_counts = Counter(pilot.source_group(row) for row in selected)
    mismatches = {
        group: (actual_counts[group], target)
        for group, target in target_counts.items()
        if actual_counts[group] != target
    }
    if mismatches:
        raise RuntimeError(f"품종×작업 요청 수 불일치: {mismatches}")
    overlaps = camera_overlap(selected)
    if overlaps:
        raise RuntimeError(f"선택 결과 카메라 누수: {sorted(overlaps)}")
    return selected


def camera_overlap(
    rows: Sequence[Mapping[str, Any]],
) -> set[tuple[str, str]]:
    train = {
        (str(row["species"]), str(row["camera_id"]))
        for row in rows
        if row["split"] == "train"
    }
    validation = {
        (str(row["species"]), str(row["camera_id"]))
        for row in rows
        if row["split"] == "validation"
    }
    return train & validation


def enrich_manifest_rows(
    extracted: Sequence[dict[str, Any]],
    selected: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    selected_by_key = {
        smoke.candidate_unique_key(row): row for row in selected
    }
    result: list[dict[str, Any]] = []
    for row in extracted:
        key = smoke.candidate_unique_key(row)
        source = selected_by_key[key]
        enriched = dict(row)
        enriched["camera_holdout_group"] = json.dumps(
            [row["species"], row["camera_id"]],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        enriched["source_manifest_split"] = source.get(
            "_source_manifest_split", ""
        )
        result.append(enriched)
    return result


def validate_camera_holdout_output(
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    train_per_species_task: int,
    validation_per_species_task: int,
    max_overlay_images: int,
) -> dict[str, Any]:
    validation = pilot.validate_pilot_output(
        output_root,
        rows,
        train_per_species_task=train_per_species_task,
        validation_per_species_task=validation_per_species_task,
        max_overlay_images=max_overlay_images,
    )
    overlaps = camera_overlap(rows)
    train_cameras = {
        (str(row["species"]), str(row["camera_id"]))
        for row in rows
        if row["split"] == "train"
    }
    val_cameras = {
        (str(row["species"]), str(row["camera_id"]))
        for row in rows
        if row["split"] == "validation"
    }
    validation.update(
        {
            "camera_overlap": len(overlaps),
            "train_camera_groups": len(train_cameras),
            "validation_camera_groups": len(val_cameras),
        }
    )
    if overlaps:
        raise RuntimeError(
            f"Train/Validation species+camera_id 누수: {sorted(overlaps)}"
        )
    return validation


def camera_list_rows(
    assignment: CameraAssignment,
) -> list[tuple[str, str, str, int]]:
    rows: list[tuple[str, str, str, int]] = []
    for species in pilot.SPECIES:
        for split, cameras in (
            ("train", assignment.train_cameras[species]),
            ("validation", assignment.validation_cameras[species]),
        ):
            rows.append(
                (
                    split,
                    species,
                    ", ".join(cameras),
                    len(cameras),
                )
            )
    return rows


def per_camera_rows(
    stats: CameraStatistics,
    assignment: CameraAssignment,
) -> list[tuple[str, str, str, int, int, int]]:
    rows: list[tuple[str, str, str, int, int, int]] = []
    for species in pilot.SPECIES:
        cameras = sorted(
            (
                camera_id
                for candidate_species, camera_id in stats.by_camera
                if candidate_species == species
            ),
            key=camera_sort_key,
        )
        for camera_id in cameras:
            counts = stats.by_camera[(species, camera_id)]
            rows.append(
                (
                    species,
                    camera_id,
                    assigned_split(assignment, species, camera_id),
                    counts["total"],
                    counts["task:생육"],
                    counts["task:병해"],
                )
            )
    return rows


def write_camera_holdout_summary(
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    stats: CameraStatistics,
    assignment: CameraAssignment,
    validation: Mapping[str, Any],
    *,
    seed: int,
    train_per_species_task: int,
    validation_per_species_task: int,
    max_total_images: int,
    max_overlay_images: int,
) -> None:
    distribution = pilot.pilot_distribution(rows)
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
        "# YOLO 카메라 홀드아웃 파일럿 요약",
        "",
        "## 구성",
        "",
        "- 그룹 키: `species + camera_id`",
        f"- seed: `{seed}`",
        f"- Train 품종×작업별: `{train_per_species_task}`장",
        f"- Validation 품종×작업별: "
        f"`{validation_per_species_task}`장",
        f"- 이미지 상한: `{max_total_images}`장",
        f"- overlay 상한: `{max_overlay_images}`장",
        "- detection manifest의 Train/Validation 후보만 사용",
        f"- 건너뛴 최종 Test 행: `{stats.skipped_test_rows:,}`개",
        "- 모델 학습 및 전체 데이터 변환을 수행하지 않음",
        "",
        "## 결과",
        "",
        f"- Train 이미지: **{validation['train_images']:,}장**",
        f"- Validation 이미지: "
        f"**{validation['validation_images']:,}장**",
        f"- bbox: **{validation['bbox_count']:,}개**",
        f"- overlay: **{validation['overlay_count']:,}장**",
        f"- Train 카메라 그룹: "
        f"**{validation['train_camera_groups']:,}개**",
        f"- Validation 카메라 그룹: "
        f"**{validation['validation_camera_groups']:,}개**",
        f"- 카메라 중복: **{validation['camera_overlap']:,}개**",
        f"- 이미지/라벨 누락: **{validation['missing_images']:,}/"
        f"{validation['missing_labels']:,}개**",
        f"- 좌표 오류: **{validation['coordinate_errors']:,}개**",
        f"- Test 이미지: **{validation['test_images']:,}개**",
        "- 원본 ZIP 변경: **없음**",
        "- 기존 `yolo_smoke`, `yolo_pilot` 변경: **없음**",
        "",
        "## 품종×작업",
        "",
        smoke.markdown_table(
            ("split", "품종", "작업", "이미지 수"),
            pilot.ordered_species_task_rows(
                distribution["species_task"]
            ),
        ),
        "",
        "## 병해 종류",
        "",
        smoke.markdown_table(
            ("split", "병해 종류", "이미지 수"), disease_rows
        ),
        "",
        "## Train/Validation 카메라 목록",
        "",
        smoke.markdown_table(
            ("split", "품종", "camera_id", "카메라 수"),
            camera_list_rows(assignment),
        ),
        "",
        "## 전체 후보 카메라별 이미지 수",
        "",
        smoke.markdown_table(
            (
                "품종",
                "camera_id",
                "할당 split",
                "전체",
                "생육",
                "병해",
            ),
            per_camera_rows(stats, assignment),
        ),
        "",
        "## 검증",
        "",
        f"- 빈 라벨: {validation['empty_labels']:,}개",
        f"- 이미지 중복/파일명 충돌: "
        f"{validation['duplicate_output_images']:,}/"
        f"{validation['output_filename_collisions']:,}개",
        f"- overlay 누락/예상 밖: "
        f"{validation['missing_overlays']:,}/"
        f"{validation['unexpected_overlays']:,}개",
        f"- 로컬 절대경로 노출 파일: "
        f"{len(validation['path_leaks']):,}개",
        "",
    ]
    smoke.write_text(
        output_root / "camera_holdout_summary.md", "\n".join(lines)
    )


def create_camera_holdout_dataset(
    selected: Sequence[Mapping[str, str]],
    stats: CameraStatistics,
    assignment: CameraAssignment,
    archive_lookup: Mapping[str, base.ArchiveRef],
    output_dir: Path,
    *,
    train_per_species_task: int,
    validation_per_species_task: int,
    seed: int,
    max_total_images: int,
    max_overlay_images: int,
    overwrite: bool,
    smoke_dir: Path = SMOKE_DIR,
    pilot_dir: Path = PILOT_DIR,
    show_progress: bool = True,
) -> dict[str, Any]:
    expected_total = pilot.validate_requested_limits(
        train_per_species_task,
        validation_per_species_task,
        max_total_images,
        max_overlay_images,
    )
    if not assignment.feasible:
        raise ValueError("불가능한 카메라 할당으로 생성할 수 없습니다")
    if len(selected) != expected_total:
        raise ValueError(
            f"선택 이미지 {len(selected)}장(기대값 {expected_total}장)"
        )
    if camera_overlap(selected):
        raise ValueError("선택 목록에 카메라 중복이 있습니다")
    if any(row["split"] == "test" for row in selected):
        raise ValueError("Test split은 추출할 수 없습니다")
    source_keys = [smoke.candidate_unique_key(row) for row in selected]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("선택 목록에 중복 이미지가 있습니다")
    source_snapshot = audit.source_zip_snapshot(
        ref.path for ref in archive_lookup.values()
    )
    protected_snapshots = {
        smoke_dir: pilot.directory_snapshot(smoke_dir),
        pilot_dir: pilot.directory_snapshot(pilot_dir),
    }
    overlay_keys = pilot.select_overlay_keys(
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
        extracted = pilot.extract_pilot_images(
            selected,
            archive_lookup,
            staging,
            overlay_keys=overlay_keys,
            show_progress=show_progress,
        )
        rows = enrich_manifest_rows(extracted, selected)
        smoke.write_csv(
            staging / "camera_holdout_manifest.csv",
            CAMERA_MANIFEST_COLUMNS,
            rows,
        )
        validation = validate_camera_holdout_output(
            staging,
            rows,
            train_per_species_task=train_per_species_task,
            validation_per_species_task=validation_per_species_task,
            max_overlay_images=max_overlay_images,
        )
        write_camera_holdout_summary(
            staging,
            rows,
            stats,
            assignment,
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
        for path, snapshot in protected_snapshots.items():
            pilot.assert_directory_unchanged(path, snapshot)
        smoke.finalize_staging(staging, output_dir, backup)
        for path, snapshot in protected_snapshots.items():
            pilot.assert_directory_unchanged(path, snapshot)
    except Exception:
        smoke.rollback_staging(staging, output_dir, backup)
        audit.assert_source_zips_unchanged(source_snapshot)
        for path, snapshot in protected_snapshots.items():
            pilot.assert_directory_unchanged(path, snapshot)
        raise
    distribution = pilot.pilot_distribution(rows)
    return {
        "train_images": validation["train_images"],
        "validation_images": validation["validation_images"],
        "bbox_count": validation["bbox_count"],
        "overlay_count": validation["overlay_count"],
        "train_camera_groups": validation["train_camera_groups"],
        "validation_camera_groups": validation[
            "validation_camera_groups"
        ],
        "camera_overlap": validation["camera_overlap"],
        "missing_images": validation["missing_images"],
        "missing_labels": validation["missing_labels"],
        "coordinate_errors": validation["coordinate_errors"],
        "test_images": validation["test_images"],
        "source_zip_modified": False,
        "protected_directories_modified": False,
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


def assignment_report(
    stats: CameraStatistics,
    assignment: CameraAssignment,
) -> dict[str, Any]:
    return {
        "feasible": assignment.feasible,
        "reason": assignment.reason,
        "closest_shortfall": assignment.closest_shortfall,
        "candidate_rows": stats.candidate_rows,
        "skipped_test_rows": stats.skipped_test_rows,
        "species_camera_counts": {
            species: sum(
                candidate_species == species
                for candidate_species, _camera in stats.by_camera
            )
            for species in pilot.SPECIES
        },
        "validation_cameras": {
            species: list(assignment.validation_cameras.get(species, ()))
            for species in pilot.SPECIES
        },
        "train_cameras": {
            species: list(assignment.train_cameras.get(species, ()))
            for species in pilot.SPECIES
        },
    }


def run(
    args: argparse.Namespace,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    stats = stream_camera_statistics(args.manifest)
    assignment = choose_camera_assignment(
        stats,
        train_per_species_task=args.train_per_species_task,
        validation_per_species_task=args.validation_per_species_task,
        seed=args.seed,
    )
    feasibility = assignment_report(stats, assignment)
    if not assignment.feasible:
        return {
            "generated": False,
            "feasibility": feasibility,
            "model_training": False,
            "test_split_used": False,
        }
    selected = select_camera_holdout_rows(
        args.manifest,
        stats,
        assignment,
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
    result = create_camera_holdout_dataset(
        selected,
        stats,
        assignment,
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
            "generated": True,
            "feasibility": feasibility,
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
