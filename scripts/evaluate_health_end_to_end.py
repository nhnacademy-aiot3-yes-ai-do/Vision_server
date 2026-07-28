#!/usr/bin/env python3
"""Evaluate the fixed detector -> union crop -> health classifier pipeline.

Exactly 100 non-Test source images are selected from the health date+camera
holdout Validation manifest (10 per species and health state).  Images seen by
the fixed camera-holdout detector pilot's Train or Validation split are audited
and non-overlapping images are preferred.  Only selected ZIP members are read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_mushroom_labels as label_audit
import create_health_pilot_dataset as health_pilot
import create_yolo_pilot_dataset as yolo_pilot
import create_yolo_smoke_dataset as smoke
import inspect_aihub_archives as base
import predict_mushroom_health as health_predictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
REPORTS_ROOT = PROJECT_ROOT / "reports"
HEALTH_MANIFEST = (
    ARTIFACTS_ROOT
    / "health_pilot_date_holdout"
    / "health_date_holdout_manifest.csv"
)
DETECTOR_TRAIN_MANIFEST = (
    ARTIFACTS_ROOT
    / "yolo_pilot_camera_holdout"
    / "camera_holdout_manifest.csv"
)
DETECTION_MANIFEST = REPORTS_ROOT / "detection_dataset_manifest.csv"
REVIEW_ROOT = ARTIFACTS_ROOT / "health_end_to_end_review"
REPORT_NAMES = (
    "health_end_to_end_predictions.csv",
    "health_end_to_end_metrics.json",
    "health_end_to_end_errors.csv",
    "health_end_to_end_evaluation.md",
)
REPORT_PATHS = tuple(REPORTS_ROOT / name for name in REPORT_NAMES)
SEED = 20260726
PER_SPECIES_STATUS = 10
HARD_MAX_IMAGES = 100
DETECTION_THRESHOLD = 0.25
HEALTH_THRESHOLD = 0.70
PADDING_RATIO = 0.15
SPECIES = tuple(health_predictor.SPECIES_BY_CLASS_ID.values())
HEALTH_STATUSES = ("HEALTHY", "DISEASE_SUSPECTED")
FORBIDDEN_PATH_PATTERNS = health_predictor.FORBIDDEN_PATH_PATTERNS
PREDICTION_COLUMNS = (
    "sample_index",
    "split",
    "actual_species",
    "actual_health_status",
    "disease_type",
    "capture_date",
    "camera_id",
    "image_archive_id",
    "image_member",
    "detector_training_overlap",
    "detector_validation_overlap",
    "detector_found_any",
    "species_detection_success",
    "exact_species_match",
    "predicted_species",
    "unexpected_species",
    "detected_count",
    "detection_confidence",
    "detector_union_bbox",
    "detector_crop_bbox",
    "gt_union_bbox",
    "detector_union_iou",
    "detector_gt_crop_iou",
    "gt_crop_coverage",
    "detector_crop_coverage",
    "detector_gt_crop_area_ratio",
    "detector_health_status",
    "detector_health_confidence",
    "detector_healthy_probability",
    "detector_disease_suspected_probability",
    "gt_crop_bbox",
    "gt_crop_health_status",
    "gt_crop_health_confidence",
    "gt_crop_healthy_probability",
    "gt_crop_disease_suspected_probability",
    "detector_gt_status_agreement",
    "disease_probability_delta",
    "pipeline_status",
    "warnings",
    "review_relative_path",
)
ERROR_COLUMNS = (*PREDICTION_COLUMNS, "error_reasons")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "고정 detector와 health classifier의 Validation 원본 100장 "
            "end-to-end 평가를 수행합니다. Test, 학습, 변환은 사용하지 않습니다."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def normalized(value: Any) -> str:
    return "" if value is None else str(value).strip().replace("\ufeff", "")


def normalized_member(value: Any) -> str:
    member = base.normalize_member_name(normalized(value))
    pure = PurePosixPath(member)
    if (
        not member
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in member
    ):
        raise ValueError("image_member 상대경로가 유효하지 않습니다")
    return member


def image_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    archive_id = normalized(row.get("image_archive_id")).casefold()
    member = normalized_member(row.get("image_member")).casefold()
    return archive_id, member


def _safe_relative(value: Any, label: str) -> str:
    text = normalized(value).replace("\\", "/")
    pure = PurePosixPath(text)
    if (
        not text
        or pure.is_absolute()
        or ".." in pure.parts
        or re.match(r"^[A-Za-z]:/", text)
    ):
        raise ValueError(f"{label} 상대경로 오류")
    if label == "output_relative_path" and (
        not pure.parts or pure.parts[0] != "val"
    ):
        raise ValueError("output_relative_path는 val/ 아래여야 합니다")
    return pure.as_posix()


def _health_status_from_source(row: Mapping[str, str]) -> str:
    explicit = normalized(row.get("actual_health_status"))
    if explicit:
        if explicit not in HEALTH_STATUSES:
            raise ValueError("지원하지 않는 actual_health_status")
        return explicit
    class_name = normalized(row.get("health_class_name"))
    class_id = normalized(row.get("health_class_id"))
    if class_name in HEALTH_STATUSES:
        expected_id = "0" if class_name == "HEALTHY" else "1"
        if class_id and class_id != expected_id:
            raise ValueError("health class id/name 불일치")
        return class_name
    raise ValueError("건강 상태 필드가 없습니다")


def load_evaluation_candidates(
    manifest_path: Path,
) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError("health evaluation manifest가 없습니다")
    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required_common = {
            "split",
            "capture_date",
            "camera_id",
            "image_archive_id",
            "image_member",
            "output_relative_path",
        }
        if missing := required_common - fields:
            raise ValueError(
                "evaluation manifest 필수 컬럼 누락: "
                + ", ".join(sorted(missing))
            )
        if not (
            {"actual_species", "actual_health_status"} <= fields
            or {"species", "health_class_id", "health_class_name"} <= fields
        ):
            raise ValueError("evaluation manifest species/health 컬럼 누락")
        for line_number, source in enumerate(reader, start=2):
            split = normalized(source.get("split")).lower()
            if split == "test":
                raise ValueError(
                    f"Test 행은 평가 후보로 사용할 수 없습니다: {line_number}"
                )
            if split == "train":
                continue
            if split != "validation":
                raise ValueError(f"지원하지 않는 split: {split}")
            species = normalized(
                source.get("actual_species") or source.get("species")
            )
            if species not in SPECIES:
                raise ValueError(f"지원하지 않는 품종: {species}")
            status = _health_status_from_source(source)
            member = normalized_member(source.get("image_member"))
            output_relative = _safe_relative(
                source.get("output_relative_path"),
                "output_relative_path",
            )
            archive_id = normalized(source.get("image_archive_id"))
            if not archive_id:
                raise ValueError("image_archive_id 누락")
            canonical = dict(source)
            canonical.update(
                {
                    "split": "validation",
                    "actual_species": species,
                    "actual_health_status": status,
                    "species": species,
                    "image_archive_id": archive_id,
                    "image_member": member,
                    "output_relative_path": output_relative,
                    "capture_date": normalized(source.get("capture_date")),
                    "camera_id": normalized(source.get("camera_id")),
                    "disease_type": normalized(source.get("disease_type")),
                }
            )
            identity = image_identity(canonical)
            if identity in seen:
                raise ValueError("evaluation manifest 원본 이미지 중복")
            seen.add(identity)
            canonical["group_key"] = normalized(
                source.get("group_key")
            ) or json.dumps(
                [archive_id, member],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            candidates.append(canonical)
    if not candidates:
        raise ValueError("Validation 평가 후보가 없습니다")
    return candidates


def load_detector_provenance_rows(
    manifest_path: Path,
) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError("detector pilot manifest가 없습니다")
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        needed = {
            "split",
            "species",
            "camera_id",
            "capture_date",
            "image_archive_id",
            "image_member",
        }
        if missing := needed - set(reader.fieldnames or ()):
            raise ValueError(
                "detector manifest 필수 컬럼 누락: "
                + ", ".join(sorted(missing))
            )
        for source in reader:
            split = normalized(source["split"]).lower()
            if split == "test":
                raise ValueError("detector provenance manifest에 Test 행이 있습니다")
            if split not in {"train", "validation"}:
                raise ValueError("detector provenance split 오류")
            row = {
                "split": split,
                "actual_species": normalized(source["species"]),
                "capture_date": normalized(source["capture_date"]),
                "camera_id": normalized(source["camera_id"]),
                "image_archive_id": normalized(source["image_archive_id"]),
                "image_member": normalized_member(source["image_member"]),
            }
            identity = image_identity(row)
            if identity in seen:
                raise ValueError("detector provenance 원본 이미지 중복")
            seen.add(identity)
            rows.append(row)
    if not rows:
        raise ValueError("detector provenance 행이 없습니다")
    return rows


def load_detector_training_rows(
    manifest_path: Path,
) -> list[dict[str, str]]:
    return [
        row
        for row in load_detector_provenance_rows(manifest_path)
        if row["split"] == "train"
    ]


def verify_selected_detection_sources(
    selected: Sequence[Mapping[str, Any]],
    detection_manifest: Path,
) -> dict[tuple[str, str], str]:
    """Verify selected originals against the final detection split metadata."""
    if not detection_manifest.is_file():
        raise FileNotFoundError("final detection manifest가 없습니다")
    wanted = {image_identity(row): row for row in selected}
    if len(wanted) != len(selected):
        raise ValueError("selected source identity 중복")
    matched: dict[tuple[str, str], str] = {}
    with detection_manifest.open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        required = {
            "split",
            "species",
            "image_archive_id",
            "image_member",
        }
        if missing := required - set(reader.fieldnames or ()):
            raise ValueError(
                "detection manifest 필수 컬럼 누락: "
                + ", ".join(sorted(missing))
            )
        for source in reader:
            identity = image_identity(source)
            if identity not in wanted:
                continue
            if identity in matched:
                raise ValueError("detection manifest selected source 중복")
            split = normalized(source["split"]).lower()
            if split == "test":
                raise ValueError("최종 detection Test 이미지가 선택되었습니다")
            if split not in {"train", "validation"}:
                raise ValueError("detection manifest split 오류")
            if normalized(source["species"]) != normalized(
                wanted[identity]["actual_species"]
            ):
                raise ValueError("health/detection species 불일치")
            matched[identity] = split
    if set(matched) != set(wanted):
        raise ValueError(
            f"selected/detection manifest 1:1 불일치: "
            f"{len(matched)}/{len(wanted)}"
        )
    return matched


def audit_evaluation_overlap(
    evaluation_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    eval_images = {image_identity(row) for row in evaluation_rows}
    train_images = {image_identity(row) for row in training_rows}
    eval_dates = {normalized(row.get("capture_date")) for row in evaluation_rows}
    train_dates = {normalized(row.get("capture_date")) for row in training_rows}
    eval_cameras = {normalized(row.get("camera_id")) for row in evaluation_rows}
    train_cameras = {normalized(row.get("camera_id")) for row in training_rows}

    def species_of(row: Mapping[str, Any]) -> str:
        return normalized(row.get("actual_species") or row.get("species"))

    eval_species_camera = {
        (species_of(row), normalized(row.get("camera_id")))
        for row in evaluation_rows
    }
    train_species_camera = {
        (species_of(row), normalized(row.get("camera_id")))
        for row in training_rows
    }
    image_affected = sum(
        image_identity(row) in train_images for row in evaluation_rows
    )
    date_affected = sum(
        normalized(row.get("capture_date")) in train_dates
        for row in evaluation_rows
    )
    camera_affected = sum(
        normalized(row.get("camera_id")) in train_cameras
        for row in evaluation_rows
    )
    species_camera_affected = sum(
        (
            species_of(row),
            normalized(row.get("camera_id")),
        )
        in train_species_camera
        for row in evaluation_rows
    )
    return {
        "image_overlap_count": len(eval_images & train_images),
        "capture_date_overlap_count": len(eval_dates & train_dates),
        "camera_id_overlap_count": len(eval_cameras & train_cameras),
        "species_camera_overlap_count": len(
            eval_species_camera & train_species_camera
        ),
        "image_overlap_affected_rows": image_affected,
        "capture_date_overlap_affected_rows": date_affected,
        "camera_id_overlap_affected_rows": camera_affected,
        "species_camera_overlap_affected_rows": species_camera_affected,
    }


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return normalized(value).lower() in {"1", "true", "yes"}


def _selection_identity(row: Mapping[str, Any]) -> str:
    if row.get("image_archive_id") and row.get("image_member"):
        archive, member = image_identity(row)
        return f"{archive}:{member}"
    return normalized(
        row.get("output_relative_path") or row.get("image_member")
    )


def deterministic_stratified_sample(
    candidates: Sequence[Mapping[str, Any]],
    *,
    per_species_status: int,
    seed: int,
) -> list[dict[str, Any]]:
    if per_species_status <= 0:
        raise ValueError("per_species_status는 양수여야 합니다")
    groups_by_species: dict[
        str, dict[str, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    identities: set[str] = set()
    for source in candidates:
        if normalized(source.get("split")).lower() == "test":
            raise ValueError("Test 후보는 선택할 수 없습니다")
        row = dict(source)
        species = normalized(
            row.get("actual_species") or row.get("species")
        )
        status = normalized(row.get("actual_health_status"))
        if status not in HEALTH_STATUSES:
            raise ValueError("후보 건강 상태 오류")
        identity = _selection_identity(row)
        if not identity or identity in identities:
            raise ValueError("후보 이미지 식별자 누락 또는 중복")
        identities.add(identity)
        group_key = normalized(row.get("group_key")) or identity
        groups_by_species[species][group_key].append(row)

    selected: list[dict[str, Any]] = []
    ordered_species = [
        item for item in SPECIES if item in groups_by_species
    ]
    for species in ordered_species:
        groups = groups_by_species[species]
        for status in HEALTH_STATUSES:
            if not any(
                row["actual_health_status"] == status
                for rows in groups.values()
                for row in rows
            ):
                raise ValueError(f"표본 strata 누락: {species}/{status}")
        ordered_groups = sorted(
            groups.items(),
            key=lambda item: (
                any(
                    _truthy(row.get("detector_training_overlap"))
                    or _truthy(row.get("detector_validation_overlap"))
                    for row in item[1]
                ),
                any(
                    _truthy(row.get("detector_training_overlap"))
                    for row in item[1]
                ),
                smoke.stable_score(seed, species, item[0]),
            ),
        )
        # Two-dimensional subset-sum over whole groups. A camera/group that
        # contains both states is selected or rejected as one unit.
        solutions: dict[tuple[int, int], tuple[int, ...]] = {
            (0, 0): ()
        }
        target = (per_species_status, per_species_status)
        for index, (_group, rows) in enumerate(ordered_groups):
            vector = (
                sum(
                    row["actual_health_status"] == "HEALTHY"
                    for row in rows
                ),
                sum(
                    row["actual_health_status"] == "DISEASE_SUSPECTED"
                    for row in rows
                ),
            )
            additions: dict[tuple[int, int], tuple[int, ...]] = {}
            for totals, chosen in tuple(solutions.items()):
                new_totals = (
                    totals[0] + vector[0],
                    totals[1] + vector[1],
                )
                if (
                    new_totals[0] <= target[0]
                    and new_totals[1] <= target[1]
                    and new_totals not in solutions
                    and new_totals not in additions
                ):
                    additions[new_totals] = (*chosen, index)
            solutions.update(additions)
            if target in solutions:
                break
        if target not in solutions:
            raise ValueError(
                f"그룹을 쪼개지 않고 {species}의 상태별 "
                f"{per_species_status}장을 선택할 수 없습니다"
            )
        for index in solutions[target]:
            selected.extend(dict(row) for row in ordered_groups[index][1])
    selected.sort(
        key=lambda row: (
            SPECIES.index(
                normalized(row.get("actual_species") or row.get("species"))
            ),
            HEALTH_STATUSES.index(normalized(row["actual_health_status"])),
            smoke.stable_score(seed, _selection_identity(row)),
        )
    )
    expected = len(ordered_species) * len(HEALTH_STATUSES) * per_species_status
    if len(selected) != expected:
        raise RuntimeError(f"선택 수량 불일치: {len(selected)}/{expected}")
    return selected


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_end_to_end_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    health_field: str = "detector_health_status",
    require_detection: bool = True,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("end-to-end metric 대상이 없습니다")
    total = len(rows)
    detection_success = sum(
        _truthy(row.get("species_detection_success")) for row in rows
    )
    exact_species_match = sum(
        _truthy(row.get("exact_species_match")) for row in rows
    )
    unexpected_species = sum(
        bool(normalized(row.get("unexpected_species"))) for row in rows
    )
    outcomes: list[tuple[str, str]] = []
    per_class: dict[str, dict[str, Any]] = {}
    for row in rows:
        actual = normalized(row.get("actual_health_status"))
        if actual not in HEALTH_STATUSES:
            raise ValueError("actual health status 오류")
        predicted = normalized(row.get(health_field))
        if predicted not in {"", *HEALTH_STATUSES, "UNCERTAIN"}:
            raise ValueError("predicted health status 오류")
        detected = _truthy(row.get("species_detection_success"))
        if require_detection and not detected and predicted:
            raise ValueError(
                "품종 탐지 실패 행에 detector health status가 있습니다"
            )
        outcome = (
            predicted
            if predicted and (detected or not require_detection)
            else "UNAVAILABLE"
        )
        outcomes.append((actual, outcome))
    analyzable = sum(outcome != "UNAVAILABLE" for _actual, outcome in outcomes)
    automatic = sum(outcome in HEALTH_STATUSES for _actual, outcome in outcomes)
    correct = sum(actual == outcome for actual, outcome in outcomes)
    automatic_correct = sum(
        actual == outcome and outcome in HEALTH_STATUSES
        for actual, outcome in outcomes
    )
    exact_pipeline_correct = sum(
        actual == outcome
        and _truthy(row.get("exact_species_match"))
        for row, (actual, outcome) in zip(rows, outcomes)
    )
    uncertain = sum(outcome == "UNCERTAIN" for _actual, outcome in outcomes)
    unavailable = sum(
        outcome == "UNAVAILABLE" for _actual, outcome in outcomes
    )
    disease_as_healthy = sum(
        actual == "DISEASE_SUSPECTED" and outcome == "HEALTHY"
        for actual, outcome in outcomes
    )
    for class_name in HEALTH_STATUSES:
        support = sum(actual == class_name for actual, _outcome in outcomes)
        tp = sum(
            actual == class_name and outcome == class_name
            for actual, outcome in outcomes
        )
        fp = sum(
            actual != class_name and outcome == class_name
            for actual, outcome in outcomes
        )
        fn = support - tp
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, support)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class[class_name] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    confusion = {
        actual: {
            predicted: sum(
                observed_actual == actual and outcome == predicted
                for observed_actual, outcome in outcomes
            )
            for predicted in (
                "HEALTHY",
                "DISEASE_SUSPECTED",
                "UNCERTAIN",
                "UNAVAILABLE",
            )
        }
        for actual in HEALTH_STATUSES
    }
    return {
        "total": total,
        "detection_success_count": (
            detection_success if require_detection else None
        ),
        "detection_success_rate": (
            _safe_divide(detection_success, total)
            if require_detection
            else None
        ),
        "exact_species_match_count": (
            exact_species_match if require_detection else None
        ),
        "exact_species_match_rate": (
            _safe_divide(exact_species_match, total)
            if require_detection
            else None
        ),
        "unexpected_species_image_count": (
            unexpected_species if require_detection else None
        ),
        "unexpected_species_image_rate": (
            _safe_divide(unexpected_species, total)
            if require_detection
            else None
        ),
        "health_analyzable_count": analyzable,
        "health_analyzable_rate": _safe_divide(analyzable, total),
        "automatic_decision_count": automatic,
        "automatic_decision_rate": _safe_divide(automatic, total),
        "automatic_decision_accuracy": _safe_divide(
            automatic_correct, automatic
        ),
        "accuracy": _safe_divide(correct, total),
        "exact_species_and_health_accuracy": (
            _safe_divide(exact_pipeline_correct, total)
            if require_detection
            else None
        ),
        "macro_f1": mean(
            per_class[class_name]["f1"] for class_name in HEALTH_STATUSES
        ),
        "healthy_recall": per_class["HEALTHY"]["recall"],
        "disease_suspected_recall": per_class[
            "DISEASE_SUSPECTED"
        ]["recall"],
        "uncertain_count": uncertain,
        "uncertain_rate": _safe_divide(uncertain, total),
        "unavailable_count": unavailable,
        "disease_as_healthy_count": disease_as_healthy,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def evaluate_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    predictor: Any,
) -> dict[str, Any]:
    predictions = [dict(predictor(dict(candidate))) for candidate in candidates]
    return {
        "predictions": predictions,
        "detector_metrics": compute_end_to_end_metrics(
            predictions,
            health_field="detector_health_status",
            require_detection=True,
        ),
        "gt_crop_metrics": compute_end_to_end_metrics(
            predictions,
            health_field="gt_crop_health_status",
            require_detection=False,
        ),
    }


def assert_deidentified_json(paths: Iterable[Path]) -> None:
    leaks: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if any(pattern.search(text) for pattern in FORBIDDEN_PATH_PATTERNS):
            leaks.append(path.name)
    if leaks:
        raise ValueError(
            "local absolute path 노출: " + ", ".join(sorted(leaks))
        )


def file_fingerprint(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return stat.st_size, stat.st_mtime_ns, digest.hexdigest()


def reports_snapshot() -> dict[str, tuple[int, int]]:
    if not REPORTS_ROOT.exists():
        return {}
    allowed = set(REPORT_NAMES)
    return {
        path.relative_to(REPORTS_ROOT).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in REPORTS_ROOT.rglob("*")
        if path.is_file()
        and path.relative_to(REPORTS_ROOT).as_posix() not in allowed
        and not path.relative_to(REPORTS_ROOT).parts[0].startswith(
            ".health_end_to_end_reports.tmp-"
        )
    }


def _project_snapshot() -> dict[str, tuple[int, int]]:
    root = PROJECT_ROOT.resolve()
    review = REVIEW_ROOT.resolve()
    report_targets = {path.resolve() for path in REPORT_PATHS}
    prefixes = (
        ".health_end_to_end_review.tmp-",
        ".health_end_to_end_review.backup-",
        ".health_end_to_end_reports.tmp-",
    )
    snapshot: dict[str, tuple[int, int]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved == review or resolved.is_relative_to(review):
            continue
        if resolved in report_targets:
            continue
        relative = path.relative_to(root)
        if any(part.startswith(prefixes) for part in relative.parts):
            continue
        stat = path.stat()
        snapshot[relative.as_posix()] = (
            stat.st_size,
            stat.st_mtime_ns,
        )
    return snapshot


def _parse_xywh_rect(
    raw_value: Any,
    image_size: tuple[int, int],
    label: str,
) -> tuple[int, int, int, int]:
    try:
        value = json.loads(normalized(raw_value))
        x = float(value["x"])
        y = float(value["y"])
        width = float(value["width"])
        height = float(value["height"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} 파싱 실패") from exc
    box = (
        math.floor(x),
        math.floor(y),
        math.ceil(x + width),
        math.ceil(y + height),
    )
    image_width, image_height = image_size
    if (
        box[0] < 0
        or box[1] < 0
        or box[2] > image_width
        or box[3] > image_height
        or box[2] <= box[0]
        or box[3] <= box[1]
    ):
        raise ValueError(f"{label}이 원본 범위를 벗어납니다")
    return box


def _intersection_area(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _rect_area(rect: Sequence[float]) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def _geometry_comparison(
    detector_rect: Sequence[float],
    ground_truth_rect: Sequence[float],
) -> dict[str, float]:
    intersection = _intersection_area(detector_rect, ground_truth_rect)
    detector_area = _rect_area(detector_rect)
    ground_truth_area = _rect_area(ground_truth_rect)
    union = detector_area + ground_truth_area - intersection
    return {
        "iou": _safe_divide(intersection, union),
        "gt_coverage": _safe_divide(intersection, ground_truth_area),
        "detector_coverage": _safe_divide(intersection, detector_area),
        "area_ratio": _safe_divide(detector_area, ground_truth_area),
    }


def _review_image(
    image: Image.Image,
    response: Mapping[str, Any],
    row: Mapping[str, Any],
    gt_crop_box: tuple[int, int, int, int],
    detector_crop_box: tuple[int, int, int, int] | None,
) -> Image.Image:
    annotated = health_predictor.draw_annotated(image, response)
    draw = ImageDraw.Draw(annotated)
    draw.rectangle(gt_crop_box, outline=(40, 220, 255), width=max(2, image.width // 500))
    font = health_predictor._font(max(14, min(26, image.width // 42)))
    label = (
        f"GT crop | {row['actual_species']} | "
        f"{row['actual_health_status']}"
    )
    draw.text((gt_crop_box[0] + 4, gt_crop_box[1] + 4), label, fill=(40, 220, 255), font=font)
    if detector_crop_box is not None:
        draw.rectangle(
            detector_crop_box,
            outline=(255, 165, 40),
            width=max(2, image.width // 500),
        )
        draw.text(
            (detector_crop_box[0] + 4, detector_crop_box[1] + 28),
            "Detector classifier crop",
            fill=(255, 165, 40),
            font=font,
        )
    return annotated


def _contact_sheet(image_paths: Sequence[Path], destination: Path) -> None:
    if not image_paths:
        raise ValueError("contact sheet 이미지가 없습니다")
    tile_size = (320, 260)
    columns = 5
    rows = math.ceil(len(image_paths) / columns)
    sheet = Image.new(
        "RGB",
        (columns * tile_size[0], rows * tile_size[1]),
        (25, 25, 25),
    )
    for index, path in enumerate(image_paths):
        with Image.open(path) as opened:
            opened.load()
            tile = ImageOps.contain(opened.convert("RGB"), tile_size)
        left = (index % columns) * tile_size[0]
        top = (index // columns) * tile_size[1]
        left += (tile_size[0] - tile.width) // 2
        top += (tile_size[1] - tile.height) // 2
        sheet.paste(tile, (left, top))
    sheet.save(destination, format="JPEG", quality=90, subsampling=0)


def _run_selected_images(
    selected: Sequence[Mapping[str, Any]],
    archive_lookup: Mapping[str, base.ArchiveRef],
    staged_review: Path,
    *,
    device: str,
) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    selected_by_identity = {image_identity(row): dict(row) for row in selected}
    if len(selected_by_identity) != len(selected):
        raise ValueError("선택 이미지 원본 식별자 중복")
    predictions: list[dict[str, Any]] = []
    read_identities: set[tuple[str, str]] = set()
    review_paths: list[Path] = []
    images_dir = staged_review / "images"
    images_dir.mkdir(parents=True)
    progress = tqdm(total=len(selected), desc="건강 E2E", unit="image")
    with health_predictor.isolated_ultralytics_runtime(staged_review):
        detector, classifier = health_predictor.load_fixed_models(device=device)
        try:
            for source, image_bytes in yolo_pilot.iter_selected_image_bytes(
                selected, archive_lookup
            ):
                identity = image_identity(source)
                if identity not in selected_by_identity:
                    raise RuntimeError("선택되지 않은 ZIP member가 읽혔습니다")
                if identity in read_identities:
                    raise RuntimeError("동일 ZIP member 중복 읽기")
                read_identities.add(identity)
                row = selected_by_identity[identity]
                try:
                    with Image.open(io.BytesIO(image_bytes)) as opened:
                        opened.load()
                        image = ImageOps.exif_transpose(opened).convert("RGB")
                except (OSError, UnidentifiedImageError) as exc:
                    raise ValueError("선택 원본 이미지가 손상되었습니다") from exc
                if row.get("original_width") and row.get("original_height"):
                    expected_size = (
                        int(row["original_width"]),
                        int(row["original_height"]),
                    )
                    if image.size != expected_size:
                        raise ValueError("manifest/원본 이미지 크기 불일치")
                response = health_predictor.predict_health(
                    image,
                    detector=detector,
                    classifier=classifier,
                    detection_threshold=DETECTION_THRESHOLD,
                    health_threshold=HEALTH_THRESHOLD,
                    padding_ratio=PADDING_RATIO,
                )
                matches = [
                    item
                    for item in response["results"]
                    if item["species"] == row["actual_species"]
                ]
                if len(matches) > 1:
                    raise RuntimeError("동일 실제 품종 결과 중복")
                matching = matches[0] if matches else None
                predicted_species_values = [
                    str(item["species"]) for item in response["results"]
                ]
                predicted_species_set = set(predicted_species_values)
                unexpected_species = sorted(
                    predicted_species_set - {str(row["actual_species"])},
                    key=lambda value: SPECIES.index(value),
                )
                exact_species_match = predicted_species_set == {
                    str(row["actual_species"])
                }
                gt_union_box = _parse_xywh_rect(
                    row.get("union_bbox"),
                    image.size,
                    "ground-truth union_bbox",
                )
                gt_crop_box = _parse_xywh_rect(
                    row.get("padded_crop_bbox"),
                    image.size,
                    "ground-truth padded_crop_bbox",
                )
                detector_union_box = (
                    tuple(int(value) for value in matching["bbox"])
                    if matching
                    else None
                )
                detector_crop_box = (
                    tuple(int(value) for value in matching["crop_bbox"])
                    if matching
                    else None
                )
                union_geometry = (
                    _geometry_comparison(detector_union_box, gt_union_box)
                    if detector_union_box is not None
                    else None
                )
                crop_geometry = (
                    _geometry_comparison(detector_crop_box, gt_crop_box)
                    if detector_crop_box is not None
                    else None
                )
                gt_crop = image.crop(gt_crop_box)
                gt_healthy, gt_disease = classifier.predict(gt_crop)
                gt_status, gt_confidence = health_predictor.health_status(
                    gt_healthy,
                    gt_disease,
                    HEALTH_THRESHOLD,
                )
                basename = smoke.output_basename(row)
                review_path = images_dir / basename
                review = _review_image(
                    image,
                    response,
                    row,
                    gt_crop_box,
                    detector_crop_box,
                )
                review.save(
                    review_path,
                    format="JPEG",
                    quality=92,
                    subsampling=0,
                )
                review_paths.append(review_path)
                predicted_species = "|".join(predicted_species_values)
                prediction = {
                    "split": "validation",
                    "actual_species": row["actual_species"],
                    "actual_health_status": row["actual_health_status"],
                    "disease_type": row.get("disease_type", ""),
                    "capture_date": row.get("capture_date", ""),
                    "camera_id": row.get("camera_id", ""),
                    "image_archive_id": row["image_archive_id"],
                    "image_member": row["image_member"],
                    "detector_training_overlap": _truthy(
                        row.get("detector_training_overlap")
                    ),
                    "detector_validation_overlap": _truthy(
                        row.get("detector_validation_overlap")
                    ),
                    "detector_found_any": bool(response["results"]),
                    "species_detection_success": matching is not None,
                    "exact_species_match": exact_species_match,
                    "predicted_species": predicted_species,
                    "unexpected_species": "|".join(unexpected_species),
                    "detected_count": (
                        matching["detected_count"] if matching else 0
                    ),
                    "detection_confidence": (
                        matching["detection_confidence"] if matching else ""
                    ),
                    "detector_union_bbox": (
                        json.dumps(matching["bbox"], separators=(",", ":"))
                        if matching
                        else ""
                    ),
                    "detector_crop_bbox": (
                        json.dumps(
                            matching["crop_bbox"], separators=(",", ":")
                        )
                        if matching
                        else ""
                    ),
                    "gt_union_bbox": json.dumps(
                        list(gt_union_box), separators=(",", ":")
                    ),
                    "detector_union_iou": (
                        union_geometry["iou"] if union_geometry else ""
                    ),
                    "detector_gt_crop_iou": (
                        crop_geometry["iou"] if crop_geometry else ""
                    ),
                    "gt_crop_coverage": (
                        crop_geometry["gt_coverage"] if crop_geometry else ""
                    ),
                    "detector_crop_coverage": (
                        crop_geometry["detector_coverage"]
                        if crop_geometry
                        else ""
                    ),
                    "detector_gt_crop_area_ratio": (
                        crop_geometry["area_ratio"] if crop_geometry else ""
                    ),
                    "detector_health_status": (
                        matching["health_status"] if matching else ""
                    ),
                    "detector_health_confidence": (
                        matching["health_confidence"] if matching else ""
                    ),
                    "detector_healthy_probability": (
                        matching["healthy_probability"] if matching else ""
                    ),
                    "detector_disease_suspected_probability": (
                        matching["disease_suspected_probability"]
                        if matching
                        else ""
                    ),
                    "gt_crop_bbox": json.dumps(
                        list(gt_crop_box), separators=(",", ":")
                    ),
                    "gt_crop_health_status": gt_status,
                    "gt_crop_health_confidence": gt_confidence,
                    "gt_crop_healthy_probability": gt_healthy,
                    "gt_crop_disease_suspected_probability": gt_disease,
                    "detector_gt_status_agreement": (
                        matching["health_status"] == gt_status
                        if matching
                        else ""
                    ),
                    "disease_probability_delta": (
                        float(matching["disease_suspected_probability"])
                        - gt_disease
                        if matching
                        else ""
                    ),
                    "pipeline_status": response["status"],
                    "warnings": " | ".join(response["warnings"]),
                    "review_relative_path": (
                        "artifacts/health_end_to_end_review/images/"
                        + basename
                    ),
                    "_service_response": response,
                }
                predictions.append(prediction)
                progress.update(1)
        finally:
            progress.close()
    if read_identities != set(selected_by_identity):
        raise RuntimeError("선택/실제 ZIP read 식별자 집합 불일치")
    predictions.sort(
        key=lambda row: (
            SPECIES.index(str(row["actual_species"])),
            HEALTH_STATUSES.index(str(row["actual_health_status"])),
            smoke.stable_score(
                SEED,
                row["image_archive_id"],
                row["image_member"],
            ),
        )
    )
    for index, row in enumerate(predictions, start=1):
        row["sample_index"] = index
    _contact_sheet(review_paths, staged_review / "contact_sheet.jpg")
    return predictions, read_identities


def _error_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not _truthy(row.get("species_detection_success")):
        reasons.append("SPECIES_DETECTION_FAILED")
    if normalized(row.get("unexpected_species")):
        reasons.append("UNEXPECTED_SPECIES_DETECTED")
    elif (
        row.get("exact_species_match") not in {None, ""}
        and not _truthy(row.get("exact_species_match"))
    ):
        reasons.append("SPECIES_SET_MISMATCH")
    status = normalized(row.get("detector_health_status"))
    if status == "UNCERTAIN":
        reasons.append("DETECTOR_CROP_UNCERTAIN")
    elif status and status != normalized(row.get("actual_health_status")):
        reasons.append("DETECTOR_CROP_HEALTH_MISMATCH")
    if (
        normalized(row.get("gt_crop_health_status"))
        != normalized(row.get("actual_health_status"))
    ):
        reasons.append("GT_CROP_HEALTH_MISMATCH")
    return reasons


def _species_metrics(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    require_detection: bool,
) -> dict[str, dict[str, Any]]:
    return {
        species: compute_end_to_end_metrics(
            [row for row in rows if row["actual_species"] == species],
            health_field=field,
            require_detection=require_detection,
        )
        for species in SPECIES
    }


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _metric_delta(
    minuend: Mapping[str, Any],
    subtrahend: Mapping[str, Any],
) -> dict[str, float]:
    return {
        "accuracy": float(minuend["accuracy"]) - float(subtrahend["accuracy"]),
        "macro_f1": float(minuend["macro_f1"]) - float(subtrahend["macro_f1"]),
        "healthy_recall": (
            float(minuend["healthy_recall"])
            - float(subtrahend["healthy_recall"])
        ),
        "disease_suspected_recall": (
            float(minuend["disease_suspected_recall"])
            - float(subtrahend["disease_suspected_recall"])
        ),
    }


def _numeric_summary(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    values = [
        float(row[field])
        for row in rows
        if normalized(row.get(field))
    ]
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "mean": mean(values) if values else None,
        "max": max(values) if values else None,
    }


def _write_outputs(
    report_staging: Path,
    staged_review: Path,
    predictions: Sequence[Mapping[str, Any]],
    *,
    candidate_train_overlap: Mapping[str, int],
    candidate_validation_overlap: Mapping[str, int],
    selected_train_overlap: Mapping[str, int],
    selected_validation_overlap: Mapping[str, int],
    source_read_count: int,
) -> dict[str, Any]:
    detector_metrics = compute_end_to_end_metrics(
        predictions,
        health_field="detector_health_status",
        require_detection=True,
    )
    gt_metrics = compute_end_to_end_metrics(
        predictions,
        health_field="gt_crop_health_status",
        require_detection=False,
    )
    detector_species = _species_metrics(
        predictions, "detector_health_status", True
    )
    gt_species = _species_metrics(
        predictions, "gt_crop_health_status", False
    )
    paired = [
        row
        for row in predictions
        if _truthy(row.get("species_detection_success"))
    ]
    paired_detector_metrics = (
        compute_end_to_end_metrics(
            paired,
            health_field="detector_health_status",
            require_detection=True,
        )
        if paired
        else None
    )
    paired_gt_metrics = (
        compute_end_to_end_metrics(
            paired,
            health_field="gt_crop_health_status",
            require_detection=False,
        )
        if paired
        else None
    )
    paired_status_agreement_count = sum(
        _truthy(row.get("detector_gt_status_agreement")) for row in paired
    )
    geometry = {
        field: _numeric_summary(predictions, field)
        for field in (
            "detector_union_iou",
            "detector_gt_crop_iou",
            "gt_crop_coverage",
            "detector_crop_coverage",
            "detector_gt_crop_area_ratio",
            "disease_probability_delta",
        )
    }
    errors: list[dict[str, Any]] = []
    for row in predictions:
        reasons = _error_reasons(row)
        if reasons:
            errors.append({**row, "error_reasons": "|".join(reasons)})
    smoke.write_csv(
        report_staging / REPORT_NAMES[0],
        PREDICTION_COLUMNS,
        predictions,
    )
    smoke.write_csv(
        report_staging / REPORT_NAMES[2],
        ERROR_COLUMNS,
        errors,
    )
    example_source = next(
        (
            row
            for row in predictions
            if _truthy(row.get("species_detection_success"))
        ),
        None,
    )
    single_image_example: dict[str, Any] | None = None
    if example_source is not None:
        single_image_example = json.loads(
            json.dumps(
                example_source["_service_response"],
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    payload = {
        "schema_version": 1,
        "scope": {
            "analysis_type": "MUSHROOM_HEALTH_CHECK_END_TO_END_V1",
            "evaluation_images": len(predictions),
            "per_species_status": PER_SPECIES_STATUS,
            "seed": SEED,
            "source_split": "health date+camera holdout Validation",
            "test_images_used": 0,
            "selected_zip_members_read": source_read_count,
            "full_archive_extraction": False,
            "model_training": False,
            "model_conversion": False,
        },
        "models": {
            "detector": health_predictor.DETECTOR_MODEL_NAME,
            "health": health_predictor.HEALTH_MODEL_NAME,
        },
        "single_image_json_example": single_image_example,
        "thresholds": {
            "detection": DETECTION_THRESHOLD,
            "health_uncertain": HEALTH_THRESHOLD,
            "padding_ratio": PADDING_RATIO,
        },
        "detector_provenance_overlap_audit": {
            "candidate_validation_images": 1000,
            "detector_train": {
                "candidate": dict(candidate_train_overlap),
                "selected": dict(selected_train_overlap),
            },
            "detector_validation": {
                "candidate": dict(candidate_validation_overlap),
                "selected": dict(selected_validation_overlap),
            },
        },
        "detector_crop_metrics": detector_metrics,
        "ground_truth_crop_metrics": gt_metrics,
        "overall_ground_truth_minus_end_to_end": {
            "definition": (
                "전체 100장 기준 차이로, detector 품종 누락과 crop 차이를 "
                "모두 포함합니다."
            ),
            **_metric_delta(gt_metrics, detector_metrics),
        },
        "paired_crop_comparison": {
            "definition": (
                "정답 품종이 탐지된 동일 이미지에서 detector crop과 "
                "ground-truth crop만 비교합니다."
            ),
            "paired_images": len(paired),
            "detector_crop_metrics": paired_detector_metrics,
            "ground_truth_crop_metrics": paired_gt_metrics,
            "ground_truth_minus_detector": (
                _metric_delta(paired_gt_metrics, paired_detector_metrics)
                if paired_gt_metrics is not None
                and paired_detector_metrics is not None
                else None
            ),
            "status_agreement_count": paired_status_agreement_count,
            "status_agreement_rate": _safe_divide(
                paired_status_agreement_count, len(paired)
            ),
        },
        "detector_ground_truth_geometry": geometry,
        "per_species_detector_crop": detector_species,
        "per_species_ground_truth_crop": gt_species,
        "error_rows": len(errors),
        "safety": {
            "test_images_used": 0,
            "source_zip_modified": False,
            "model_files_modified": False,
            "source_manifests_modified": False,
            "existing_artifacts_modified": False,
            "outputs_outside_allowlist": 0,
            "local_absolute_paths_exposed": 0,
        },
        "limitations": [
            "정상은 2021-11-04, 병해는 2021-11-26/12-04로 날짜와 상태가 결합되어 있습니다.",
            "선택 표본은 detector 원본 Train/Validation과 동일 이미지를 제외하지만 촬영 날짜·카메라 환경 중복은 별도로 남아 있습니다.",
            "ground-truth crop은 원본 ZIP 이미지를 메모리에서 직접 crop한 것으로 기존 JPEG95 health Validation 파일과 bit-identical하지 않습니다.",
            "내부 AIHub 환경 평가이며 외부 스마트폰 일반화 성능이 아닙니다.",
        ],
    }
    (report_staging / REPORT_NAMES[1]).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 버섯 건강 체크 end-to-end 평가",
        "",
        "## 범위",
        "",
        f"- 평가 이미지: **{len(predictions):,}장**",
        "- 품종×상태별 10장, Test 0장",
        f"- 선택 ZIP member read: **{source_read_count:,}개**",
        "- 전체 ZIP 압축 해제·학습·모델 변환: 수행하지 않음",
        "",
        "## Detector provenance 중복 감사",
        "",
        f"- detector Train 동일 원본: 후보 **{candidate_train_overlap['image_overlap_count']:,}장**, "
        f"최종 **{selected_train_overlap['image_overlap_count']:,}장**",
        f"- detector Validation 동일 원본: 후보 **{candidate_validation_overlap['image_overlap_count']:,}장**, "
        f"최종 **{selected_validation_overlap['image_overlap_count']:,}장**",
        f"- 최종 표본 중 detector Train과 동일 날짜 영향을 받는 행: "
        f"**{selected_train_overlap['capture_date_overlap_affected_rows']:,}/{len(predictions):,}장**",
        f"- 최종 표본 중 detector Train과 동일 camera_id 영향을 받는 행: "
        f"**{selected_train_overlap['camera_id_overlap_affected_rows']:,}/{len(predictions):,}장**",
        f"- 최종 표본 중 detector Train과 동일 species+camera 영향을 받는 행: "
        f"**{selected_train_overlap['species_camera_overlap_affected_rows']:,}/{len(predictions):,}장**",
        "",
        "## Detector crop end-to-end 지표",
        "",
        "- 품종 탐지 성공은 정답 품종이 결과에 하나 존재함을 뜻하며, bbox 품질은 별도 IoU로 본다.",
        f"- 정답 품종 탐지 성공률: **{detector_metrics['detection_success_rate']:.6f}**",
        f"- 예측 품종 집합 완전 일치율: **{detector_metrics['exact_species_match_rate']:.6f}**",
        f"- 예상 밖 추가 품종 탐지 이미지: **{detector_metrics['unexpected_species_image_count']:,}장**",
        f"- 건강 분석 가능률: **{detector_metrics['health_analyzable_rate']:.6f}**",
        "- 아래 target-species health Accuracy/Macro F1/Recall은 전체 100장을 분모로 하며 UNCERTAIN과 정답 품종 탐지 실패는 오답/FN으로 집계한다. 추가 품종 오탐은 품종 집합 완전 일치율에서 별도 평가한다.",
        f"- target-species health E2E Accuracy: **{detector_metrics['accuracy']:.6f}**",
        f"- target-species health E2E Macro F1: **{detector_metrics['macro_f1']:.6f}**",
        f"- 정확한 품종 집합 + 건강 상태 Accuracy: **{detector_metrics['exact_species_and_health_accuracy']:.6f}**",
        f"- 자동 판정 대상 Accuracy: **{detector_metrics['automatic_decision_accuracy']:.6f}**",
        f"- HEALTHY recall: **{detector_metrics['healthy_recall']:.6f}**",
        f"- DISEASE_SUSPECTED recall: **{detector_metrics['disease_suspected_recall']:.6f}**",
        f"- UNCERTAIN: **{detector_metrics['uncertain_count']:,}장 "
        f"({detector_metrics['uncertain_rate']:.2%})**",
        f"- 병해→HEALTHY: **{detector_metrics['disease_as_healthy_count']:,}장**",
        "",
        "## Ground-truth crop 비교",
        "",
        f"- GT crop accuracy: **{gt_metrics['accuracy']:.6f}**",
        f"- 전체 E2E detector crop accuracy: **{detector_metrics['accuracy']:.6f}**",
        f"- 전체 GT - E2E accuracy: **{gt_metrics['accuracy'] - detector_metrics['accuracy']:+.6f}** "
        "(탐지 누락과 crop 차이를 함께 포함)",
        f"- 탐지 성공 paired 이미지: **{len(paired):,}장**",
        (
            f"- paired GT - detector accuracy: "
            f"**{paired_gt_metrics['accuracy'] - paired_detector_metrics['accuracy']:+.6f}**"
            if paired_gt_metrics is not None
            and paired_detector_metrics is not None
            else "- paired 비교 불가"
        ),
        f"- detector/GT crop 평균 IoU: **{geometry['detector_gt_crop_iou']['mean']}**",
        "",
        "## 품종별 detector crop",
        "",
        _markdown_table(
            ("품종", "N", "탐지 성공", "accuracy", "macro F1", "병해 recall"),
            (
                (
                    species,
                    detector_species[species]["total"],
                    f"{detector_species[species]['detection_success_rate']:.3f}",
                    f"{detector_species[species]['accuracy']:.3f}",
                    f"{detector_species[species]['macro_f1']:.3f}",
                    f"{detector_species[species]['disease_suspected_recall']:.3f}",
                )
                for species in SPECIES
            ),
        ),
        "",
        "## 제한",
        "",
        "- 정상은 2021-11-04, 병해는 2021-11-26/12-04로 날짜와 상태가 결합되어 있다.",
        "- 원본 직접 GT crop과 기존 JPEG95 health Validation 입력은 bit-identical하지 않다.",
        "- 동일 원본은 배제했지만 detector Train과 같은 촬영 날짜·카메라 환경은 남아 있다.",
        "- 내부 AIHub 환경 평가이며 외부 스마트폰 일반화 성능으로 해석하지 않는다.",
        "",
        "## 안전 검증",
        "",
        "- Test 이미지 사용: 0장",
        "- 원본 ZIP·모델·기존 데이터셋·기존 artifacts 변경: 0",
        "- 허용 출력 외 생성 및 로컬 절대경로 노출: 0",
        "",
    ]
    (report_staging / REPORT_NAMES[3]).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return payload


def _prepare_staging(overwrite: bool) -> tuple[Path, Path]:
    targets = [REVIEW_ROOT, *REPORT_PATHS]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "end-to-end 출력이 이미 존재합니다: "
            + ", ".join(path.name for path in existing)
        )
    REVIEW_ROOT.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    review_staging: Path | None = None
    report_staging: Path | None = None
    try:
        review_staging = Path(
            tempfile.mkdtemp(
                prefix=".health_end_to_end_review.tmp-",
                dir=REVIEW_ROOT.parent,
            )
        )
        report_staging = Path(
            tempfile.mkdtemp(
                prefix=".health_end_to_end_reports.tmp-",
                dir=REPORTS_ROOT,
            )
        )
    except Exception:
        if review_staging is not None:
            _remove_checked(review_staging)
        if report_staging is not None:
            _remove_checked(report_staging)
        raise
    return review_staging, report_staging


def _remove_checked(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    if path.exists():
        raise RuntimeError(f"임시 경로 정리 실패: {path.name}")


def _activate(
    review_staging: Path,
    report_staging: Path,
) -> tuple[Path, dict[Path, Path], list[Path]]:
    backup_root = Path(
        tempfile.mkdtemp(
            prefix=".health_end_to_end_review.backup-",
            dir=REVIEW_ROOT.parent,
        )
    )
    targets = [REVIEW_ROOT, *REPORT_PATHS]
    sources = [
        review_staging,
        *(report_staging / name for name in REPORT_NAMES),
    ]
    backups: dict[Path, Path] = {}
    activated: list[Path] = []
    try:
        for index, target in enumerate(targets):
            if target.exists():
                backup = backup_root / f"{index:02d}_{target.name}"
                os.replace(target, backup)
                backups[target] = backup
        for target, source in zip(targets, sources):
            os.replace(source, target)
            activated.append(target)
    except Exception:
        for target in reversed(activated):
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        for target, backup in reversed(tuple(backups.items())):
            if backup.exists():
                os.replace(backup, target)
        _remove_checked(backup_root)
        raise
    return backup_root, backups, activated


def _rollback(
    backup_root: Path,
    backups: Mapping[Path, Path],
    activated: Sequence[Path],
) -> None:
    for target in reversed(activated):
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    for target, backup in reversed(tuple(backups.items())):
        if backup.exists():
            os.replace(backup, target)
    _remove_checked(backup_root)


def run_evaluation(*, device: str, overwrite: bool) -> dict[str, Any]:
    candidates = load_evaluation_candidates(HEALTH_MANIFEST)
    if len(candidates) != 1000:
        raise ValueError(f"Validation 후보 수 불일치: {len(candidates)}/1000")
    detector_provenance = load_detector_provenance_rows(
        DETECTOR_TRAIN_MANIFEST
    )
    detector_train = [
        row for row in detector_provenance if row["split"] == "train"
    ]
    detector_validation = [
        row for row in detector_provenance if row["split"] == "validation"
    ]
    train_identities = {image_identity(row) for row in detector_train}
    validation_identities = {
        image_identity(row) for row in detector_validation
    }
    candidate_train_overlap = audit_evaluation_overlap(
        candidates, detector_train
    )
    candidate_validation_overlap = audit_evaluation_overlap(
        candidates, detector_validation
    )
    adapted: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        row["detector_training_overlap"] = (
            image_identity(candidate) in train_identities
        )
        row["detector_validation_overlap"] = (
            image_identity(candidate) in validation_identities
        )
        # E2E selection is image-level; no date/camera group is split here.
        row["group_key"] = _selection_identity(row)
        adapted.append(row)
    selected = deterministic_stratified_sample(
        adapted,
        per_species_status=PER_SPECIES_STATUS,
        seed=SEED,
    )
    if len(selected) != HARD_MAX_IMAGES:
        raise RuntimeError(f"E2E 선택 수량 불일치: {len(selected)}/100")
    if any(
        _truthy(row["detector_training_overlap"])
        or _truthy(row["detector_validation_overlap"])
        for row in selected
    ):
        raise RuntimeError(
            "비중복 후보가 충분하지만 detector Train/Validation 중복이 선택됨"
        )
    selected_train_overlap = audit_evaluation_overlap(
        selected, detector_train
    )
    selected_validation_overlap = audit_evaluation_overlap(
        selected, detector_validation
    )
    if selected_train_overlap["image_overlap_count"]:
        raise RuntimeError("최종 E2E 평가에 detector Train 원본 중복")
    if selected_validation_overlap["image_overlap_count"]:
        raise RuntimeError("최종 E2E 평가에 detector Validation 원본 중복")
    counts = Counter(
        (row["actual_species"], row["actual_health_status"])
        for row in selected
    )
    expected_counts = {
        (species, status): PER_SPECIES_STATUS
        for species in SPECIES
        for status in HEALTH_STATUSES
    }
    if counts != expected_counts:
        raise RuntimeError("품종×상태 선택 수량 불일치")
    selected_detection_splits = verify_selected_detection_sources(
        selected,
        DETECTION_MANIFEST,
    )
    if any(split == "test" for split in selected_detection_splits.values()):
        raise RuntimeError("최종 detection Test 이미지가 포함되었습니다")

    refs, issues, _directories = base.discover_archives(os.environ)
    if issues:
        raise RuntimeError(
            "아카이브 환경/구조 검증 실패:\n- " + "\n- ".join(issues)
        )
    archive_lookup = smoke.image_archive_lookup(refs)
    selected_archive_ids = {row["image_archive_id"] for row in selected}
    if not selected_archive_ids <= set(archive_lookup):
        raise ValueError("선택 이미지 archive ID를 찾을 수 없습니다")

    protected_files = (
        health_predictor.DETECTOR_MODEL_PATH,
        health_predictor.HEALTH_MODEL_PATH,
        HEALTH_MANIFEST,
        DETECTOR_TRAIN_MANIFEST,
        DETECTION_MANIFEST,
    )
    protected_before = {
        path: file_fingerprint(path) for path in protected_files
    }
    source_zip_before = label_audit.source_zip_snapshot(
        ref.path for ref in archive_lookup.values()
    )
    artifacts_before = health_pilot.snapshot_artifacts_excluding(
        ARTIFACTS_ROOT, REVIEW_ROOT
    )
    reports_before = reports_snapshot()
    project_before = _project_snapshot()
    review_staging, report_staging = _prepare_staging(overwrite)
    backup_root: Path | None = None
    backups: dict[Path, Path] = {}
    activated: list[Path] = []
    try:
        predictions, read_identities = _run_selected_images(
            selected,
            archive_lookup,
            review_staging,
            device=device,
        )
        if len(read_identities) != HARD_MAX_IMAGES:
            raise RuntimeError("선택 ZIP member read 수 불일치")
        payload = _write_outputs(
            report_staging,
            review_staging,
            predictions,
            candidate_train_overlap=candidate_train_overlap,
            candidate_validation_overlap=candidate_validation_overlap,
            selected_train_overlap=selected_train_overlap,
            selected_validation_overlap=selected_validation_overlap,
            source_read_count=len(read_identities),
        )
        report_files = {
            path.name for path in report_staging.iterdir() if path.is_file()
        }
        if report_files != set(REPORT_NAMES):
            raise RuntimeError("staged report 파일 집합 불일치")
        review_files = {
            path.relative_to(review_staging).as_posix()
            for path in review_staging.rglob("*")
            if path.is_file()
        }
        expected_review = {
            "contact_sheet.jpg",
            *{
                f"images/{smoke.output_basename(row)}"
                for row in selected
            },
        }
        if review_files != expected_review:
            raise RuntimeError("staged review 파일 집합 불일치")
        assert_deidentified_json(report_staging.iterdir())
        with (report_staging / REPORT_NAMES[0]).open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            if sum(1 for _ in csv.DictReader(handle)) != HARD_MAX_IMAGES:
                raise RuntimeError("prediction CSV 행 수 불일치")
        if {
            path: file_fingerprint(path) for path in protected_files
        } != protected_before:
            raise RuntimeError("모델 또는 source manifest 변경 감지")
        label_audit.assert_source_zips_unchanged(source_zip_before)
        health_pilot.assert_artifacts_unchanged(
            ARTIFACTS_ROOT, REVIEW_ROOT, artifacts_before
        )
        if reports_snapshot() != reports_before:
            raise RuntimeError("허용 보고서 외 reports 변경 감지")
        if _project_snapshot() != project_before:
            raise RuntimeError("허용 출력 외 프로젝트 파일 변경 감지")

        backup_root, backups, activated = _activate(
            review_staging, report_staging
        )
        if {
            path: file_fingerprint(path) for path in protected_files
        } != protected_before:
            raise RuntimeError("활성화 후 모델/source manifest 변경 감지")
        label_audit.assert_source_zips_unchanged(source_zip_before)
        health_pilot.assert_artifacts_unchanged(
            ARTIFACTS_ROOT, REVIEW_ROOT, artifacts_before
        )
        if reports_snapshot() != reports_before:
            raise RuntimeError("활성화 후 허용 외 reports 변경 감지")
        if _project_snapshot() != project_before:
            raise RuntimeError("활성화 후 허용 출력 외 프로젝트 변경 감지")
        assert_deidentified_json(REPORT_PATHS)
        _remove_checked(report_staging)
        _remove_checked(backup_root)
    except Exception:
        if backup_root is not None and backup_root.exists():
            _rollback(backup_root, backups, activated)
        _remove_checked(review_staging)
        _remove_checked(report_staging)
        if {
            path: file_fingerprint(path) for path in protected_files
        } != protected_before:
            raise RuntimeError("실패 처리 중 모델/source manifest 변경 감지")
        label_audit.assert_source_zips_unchanged(source_zip_before)
        health_pilot.assert_artifacts_unchanged(
            ARTIFACTS_ROOT, REVIEW_ROOT, artifacts_before
        )
        if reports_snapshot() != reports_before:
            raise RuntimeError("실패 처리 중 허용 외 reports 변경 감지")
        if _project_snapshot() != project_before:
            raise RuntimeError("실패 처리 중 허용 출력 외 프로젝트 변경 감지")
        raise

    detector = payload["detector_crop_metrics"]
    ground_truth = payload["ground_truth_crop_metrics"]
    return {
        "evaluation_images": detector["total"],
        "detector_training_image_overlap": selected_train_overlap[
            "image_overlap_count"
        ],
        "detector_validation_image_overlap": selected_validation_overlap[
            "image_overlap_count"
        ],
        "detection_success_rate": detector["detection_success_rate"],
        "health_analyzable_rate": detector["health_analyzable_rate"],
        "accuracy": detector["accuracy"],
        "macro_f1": detector["macro_f1"],
        "healthy_recall": detector["healthy_recall"],
        "disease_suspected_recall": detector[
            "disease_suspected_recall"
        ],
        "uncertain_count": detector["uncertain_count"],
        "uncertain_rate": detector["uncertain_rate"],
        "disease_as_healthy_count": detector[
            "disease_as_healthy_count"
        ],
        "ground_truth_crop_accuracy": ground_truth["accuracy"],
        "ground_truth_minus_detector_accuracy": (
            ground_truth["accuracy"] - detector["accuracy"]
        ),
        "test_images_used": 0,
        "selected_zip_members_read": HARD_MAX_IMAGES,
        "model_files_modified": False,
        "source_zips_modified": False,
        "existing_artifacts_modified": False,
        "reports": [f"reports/{name}" for name in REPORT_NAMES],
        "review": "artifacts/health_end_to_end_review/",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_evaluation(device=args.device, overwrite=args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
