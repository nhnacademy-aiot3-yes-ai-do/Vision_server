#!/usr/bin/env python3
"""Evaluate the fixed health date+camera holdout Validation split.

Only explicit Validation image paths verified against the health manifest are
sent to the local ``best.pt`` classifier.  Train and Test images are never
enumerated by Ultralytics.  Reports and error-review artifacts are built in a
staging directory and activated only after metric, path, and input-invariance
checks pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_mushroom_health as health
import audit_mushroom_labels as label_audit
import create_health_pilot_dataset as health_pilot
import create_yolo_smoke_dataset as smoke
import inspect_aihub_archives as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DATASET_ROOT = ARTIFACTS_ROOT / "health_pilot_date_holdout"
DEFAULT_MODEL = (
    DATASET_ROOT
    / "training_runs"
    / "yolo11n_health_binary_date_camera_holdout_30ep"
    / "weights"
    / "best.pt"
)
DEFAULT_MANIFEST = DATASET_ROOT / "health_date_holdout_manifest.csv"
DEFAULT_VALIDATION_DIR = DATASET_ROOT / "val"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_EVALUATION_DIR = DATASET_ROOT / "evaluation"
DEFAULT_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE = 320
DEFAULT_MAX_ERROR_REVIEW = 100
EXPECTED_VALIDATION_IMAGES = 1_000
CLASS_NAMES = {
    0: "HEALTHY",
    1: "DISEASE_SUSPECTED",
}
MODEL_CLASS_NAMES = {
    0: "0_healthy",
    1: "1_disease_suspected",
}
CLASS_DIRECTORIES = {
    0: "0_healthy",
    1: "1_disease_suspected",
}
THRESHOLDS = (0.60, 0.70, 0.80, 0.90, 0.95)
REPORT_FILENAMES = (
    "health_classifier_overall_metrics.json",
    "health_classifier_species_metrics.csv",
    "health_classifier_date_metrics.csv",
    "health_classifier_camera_metrics.csv",
    "health_classifier_predictions.csv",
    "health_classifier_errors.csv",
    "health_classifier_threshold_analysis.csv",
    "health_classifier_evaluation.md",
)
REQUIRED_MANIFEST_COLUMNS = {
    "split",
    "health_class_id",
    "health_class_name",
    "species",
    "task",
    "normality",
    "disease_type",
    "camera_id",
    "capture_date",
    "image_archive_id",
    "image_member",
    "output_relative_path",
}
PREDICTION_COLUMNS = (
    "prediction_index",
    "split",
    "actual_class_id",
    "actual_class_name",
    "predicted_class_id",
    "predicted_class_name",
    "is_correct",
    "healthy_probability",
    "disease_suspected_probability",
    "top1_confidence",
    "species",
    "capture_date",
    "camera_id",
    "disease_type",
    "image_archive_id",
    "image_member",
    "output_relative_path",
)
ERROR_COLUMNS = (
    *PREDICTION_COLUMNS,
    "error_type",
    "confidence_rank",
    "review_relative_path",
)
SPECIES_METRIC_COLUMNS = (
    "species",
    "total",
    "healthy_support",
    "disease_suspected_support",
    "healthy_predicted",
    "disease_suspected_predicted",
    "correct",
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "healthy_precision",
    "healthy_recall",
    "healthy_f1",
    "disease_suspected_precision",
    "disease_suspected_recall",
    "disease_suspected_f1",
    "tn",
    "fp",
    "fn",
    "tp",
)
GROUP_METRIC_COLUMNS = (
    "group_value",
    "total",
    "healthy_support",
    "disease_suspected_support",
    "correct",
    "accuracy",
    "disease_suspected_tp",
    "disease_suspected_fn",
    "disease_suspected_recall",
    "mean_top1_confidence",
    "min_top1_confidence",
    "max_top1_confidence",
)
THRESHOLD_COLUMNS = (
    "threshold",
    "total_count",
    "auto_decision_count",
    "auto_decision_rate",
    "uncertain_count",
    "uncertain_rate",
    "auto_decision_correct_count",
    "auto_decision_accuracy",
    "healthy_support",
    "healthy_auto_correct_count",
    "healthy_recall",
    "disease_suspected_support",
    "disease_suspected_auto_correct_count",
    "disease_suspected_recall",
    "disease_as_healthy_count",
    "disease_uncertain_count",
)
FORBIDDEN_PATH_PATTERNS = (
    re.compile(r"/mnt/[A-Za-z](?:/|$)", re.IGNORECASE),
    re.compile(r"/home/[^/\s]+", re.IGNORECASE),
    re.compile(r"[A-Za-z]:[\\/]", re.IGNORECASE),
    re.compile(r"\\\\Users\\\\", re.IGNORECASE),
)


@dataclass(frozen=True)
class ValidationRecord:
    actual_class_id: int
    actual_class_name: str
    species: str
    capture_date: str
    camera_id: str
    disease_type: str
    image_archive_id: str
    image_member: str
    output_relative_path: str
    image_path: Path


@dataclass(frozen=True)
class Prediction:
    actual_class_id: int
    predicted_class_id: int
    healthy_probability: float
    disease_probability: float
    species: str
    capture_date: str
    camera_id: str
    disease_type: str
    image_member: str
    output_relative_path: str
    image_archive_id: str = ""

    @property
    def actual_class_name(self) -> str:
        return CLASS_NAMES[self.actual_class_id]

    @property
    def predicted_class_name(self) -> str:
        return CLASS_NAMES[self.predicted_class_id]

    @property
    def disease_suspected_probability(self) -> float:
        return self.disease_probability

    @property
    def top1_confidence(self) -> float:
        return max(self.healthy_probability, self.disease_probability)

    @property
    def is_correct(self) -> bool:
        return self.actual_class_id == self.predicted_class_id


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "건강 체크 date+camera holdout best.pt를 Validation 1,000장에 "
            "한해 상세 평가합니다. Train/Test 및 재학습은 사용하지 않습니다."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=DEFAULT_VALIDATION_DIR,
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=DEFAULT_EVALUATION_DIR,
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--imgsz",
        type=positive_int,
        default=DEFAULT_IMAGE_SIZE,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Ultralytics device; auto는 현재 실행 환경에서 자동 선택",
    )
    parser.add_argument(
        "--max-error-review",
        type=positive_int,
        default=DEFAULT_MAX_ERROR_REVIEW,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 평가 출력만 staging/rollback 방식으로 교체",
    )
    args = parser.parse_args(argv)
    for field in (
        "model",
        "manifest",
        "validation_dir",
        "reports_dir",
        "evaluation_dir",
    ):
        setattr(args, field, project_path(getattr(args, field)))
    if args.model.name != "best.pt":
        parser.error("--model은 best.pt만 허용합니다")
    if args.model != DEFAULT_MODEL.resolve():
        parser.error("--model은 확정된 date+camera holdout best.pt로 고정됩니다")
    if args.manifest != DEFAULT_MANIFEST.resolve():
        parser.error("--manifest는 확정된 date+camera holdout manifest로 고정됩니다")
    if args.validation_dir != DEFAULT_VALIDATION_DIR.resolve():
        parser.error("--validation-dir은 확정된 Validation 1,000장으로 고정됩니다")
    if args.reports_dir != DEFAULT_REPORTS_DIR.resolve():
        parser.error("--reports-dir은 프로젝트 reports로 고정됩니다")
    if args.evaluation_dir != DEFAULT_EVALUATION_DIR.resolve():
        parser.error("--evaluation-dir은 고정 date-holdout evaluation 경로입니다")
    if args.max_error_review > DEFAULT_MAX_ERROR_REVIEW:
        parser.error(
            f"오분류 review 상한은 {DEFAULT_MAX_ERROR_REVIEW}장입니다"
        )
    return args


def _parse_class_id(value: Any) -> int:
    try:
        class_id = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"class id가 정수가 아닙니다: {value!r}") from exc
    if class_id not in CLASS_NAMES:
        raise ValueError(f"class id 범위 오류: {class_id}")
    return class_id


def _safe_validation_reference(
    value: str,
    validation_dir: Path,
) -> tuple[str, Path]:
    text = str(value).strip().replace("\ufeff", "")
    if not text or "\\" in text or re.match(r"^[A-Za-z]:", text):
        raise ValueError(f"unsafe relative path: {value!r}")
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"relative path escape: {value!r}")
    if len(pure.parts) != 3 or pure.parts[0] != "val":
        raise ValueError(f"Validation relative path 형식 오류: {value!r}")
    root = validation_dir.resolve(strict=True)
    candidate = root.joinpath(*pure.parts[1:])
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"manifest image file missing: {text}") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"Validation path escape 또는 비파일: {text}")
    return pure.as_posix(), resolved


def _actual_validation_images(validation_dir: Path) -> set[str]:
    if not validation_dir.is_dir():
        raise ValueError("Validation image path directory가 없습니다")
    result: set[str] = set()
    root = validation_dir.resolve(strict=True)
    for path in validation_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in base.IMAGE_EXTENSIONS:
            continue
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError("Validation path symlink escape 감지")
        relative = resolved.relative_to(root).as_posix()
        result.add(f"val/{relative}")
    return result


def load_validation_records(
    manifest_path: Path,
    validation_dir: Path,
    *,
    expected_count: int | None = EXPECTED_VALIDATION_IMAGES,
) -> list[ValidationRecord]:
    if not manifest_path.is_file():
        raise FileNotFoundError("health manifest가 없습니다")
    if not validation_dir.is_dir():
        raise ValueError("Validation image path directory가 없습니다")
    records: list[ValidationRecord] = []
    seen_paths: set[str] = set()
    seen_members: set[tuple[str, str]] = set()
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "manifest 필수 컬럼 누락: " + ", ".join(sorted(missing))
            )
        for line_number, row in enumerate(reader, start=2):
            split = health.normalized(row["split"]).lower()
            if split == "test":
                raise ValueError(
                    f"Test row is forbidden in evaluation manifest: {line_number}"
                )
            if split == "train":
                continue
            if split != "validation":
                raise ValueError(f"지원하지 않는 split: {split}")
            class_id = _parse_class_id(row["health_class_id"])
            class_name = health.normalized(row["health_class_name"])
            if class_name != CLASS_NAMES[class_id]:
                raise ValueError("manifest class id/name 불일치")
            relative, image_path = _safe_validation_reference(
                row["output_relative_path"],
                validation_dir,
            )
            parts = PurePosixPath(relative).parts
            if parts[1] != CLASS_DIRECTORIES[class_id]:
                raise ValueError("manifest class directory 불일치")
            normality = health.normalized(row["normality"])
            expected_normality = "normal" if class_id == 0 else "abnormal"
            if normality != expected_normality:
                raise ValueError("manifest class/normality 불일치")
            task = health.normalized(row["task"])
            expected_task = "생육" if class_id == 0 else "병해"
            if task != expected_task:
                raise ValueError("manifest class/task 불일치")
            disease_type = health.normalized(row["disease_type"])
            if class_id == 0 and disease_type:
                raise ValueError("HEALTHY 행에 병해 종류가 있습니다")
            if class_id == 1 and not disease_type:
                raise ValueError("DISEASE_SUSPECTED 행의 병해 종류가 없습니다")
            member = base.normalize_member_name(row["image_member"])
            if not member or member.startswith("/") or ".." in PurePosixPath(
                member
            ).parts:
                raise ValueError("image_member relative path 오류")
            identity = (
                health.normalized(row["image_archive_id"]).casefold(),
                member.casefold(),
            )
            if relative in seen_paths or identity in seen_members:
                raise ValueError("manifest Validation 이미지 중복")
            seen_paths.add(relative)
            seen_members.add(identity)
            records.append(
                ValidationRecord(
                    actual_class_id=class_id,
                    actual_class_name=class_name,
                    species=health.normalized(row["species"]),
                    capture_date=health.normalized(row["capture_date"]),
                    camera_id=health.normalized(row["camera_id"]),
                    disease_type=disease_type,
                    image_archive_id=health.normalized(
                        row["image_archive_id"]
                    ),
                    image_member=member,
                    output_relative_path=relative,
                    image_path=image_path,
                )
            )
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(
            f"Validation manifest image count 불일치: "
            f"{len(records)}/{expected_count}"
        )
    expected_paths = {record.output_relative_path for record in records}
    actual_paths = _actual_validation_images(validation_dir)
    missing_images = expected_paths - actual_paths
    unexpected_images = actual_paths - expected_paths
    if missing_images or unexpected_images:
        raise ValueError(
            "manifest/image 1:1 mismatch: "
            f"missing={len(missing_images)}, extra={len(unexpected_images)}"
        )
    return records


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_binary_metrics(
    predictions: Sequence[Prediction],
) -> dict[str, Any]:
    if not predictions:
        raise ValueError("지표를 계산할 prediction이 없습니다")
    matrix = [[0, 0], [0, 0]]
    for item in predictions:
        if item.actual_class_id not in CLASS_NAMES or (
            item.predicted_class_id not in CLASS_NAMES
        ):
            raise ValueError("prediction class id 범위 오류")
        matrix[item.actual_class_id][item.predicted_class_id] += 1
    total = len(predictions)
    correct = matrix[0][0] + matrix[1][1]
    per_class: dict[str, dict[str, Any]] = {}
    for class_id in (0, 1):
        tp = matrix[class_id][class_id]
        fp = sum(matrix[row][class_id] for row in (0, 1)) - tp
        fn = sum(matrix[class_id]) - tp
        support = sum(matrix[class_id])
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class[CLASS_NAMES[class_id]] = {
            "class_id": class_id,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "total": total,
        "correct": correct,
        "accuracy": _safe_divide(correct, total),
        "balanced_accuracy": mean(
            per_class[name]["recall"] for name in CLASS_NAMES.values()
        ),
        "macro_precision": mean(
            per_class[name]["precision"] for name in CLASS_NAMES.values()
        ),
        "macro_recall": mean(
            per_class[name]["recall"] for name in CLASS_NAMES.values()
        ),
        "macro_f1": mean(
            per_class[name]["f1"] for name in CLASS_NAMES.values()
        ),
        "confusion_matrix": {
            "labels": [CLASS_NAMES[0], CLASS_NAMES[1]],
            "axis": "rows=actual, columns=predicted",
            "matrix": matrix,
        },
        "per_class": per_class,
    }


def aggregate_group_metrics(
    predictions: Sequence[Prediction],
    field_name: str,
) -> dict[str, dict[str, Any]]:
    allowed = {"species", "capture_date", "camera_id"}
    if field_name not in allowed:
        raise ValueError(f"지원하지 않는 group field: {field_name}")
    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for item in predictions:
        grouped[str(getattr(item, field_name))].append(item)
    return {
        group: compute_binary_metrics(items)
        for group, items in sorted(grouped.items())
    }


def analyze_thresholds(
    predictions: Sequence[Prediction],
    *,
    thresholds: Sequence[float] = THRESHOLDS,
) -> list[dict[str, Any]]:
    total = len(predictions)
    if not total:
        raise ValueError("threshold 분석 대상이 없습니다")
    healthy_support = sum(item.actual_class_id == 0 for item in predictions)
    disease_support = total - healthy_support
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        if not 0.5 <= threshold <= 1.0:
            raise ValueError(f"threshold 범위 오류: {threshold}")
        automatic = [
            item
            for item in predictions
            if item.top1_confidence >= threshold
        ]
        uncertain_count = total - len(automatic)
        automatic_correct = sum(item.is_correct for item in automatic)
        healthy_correct = sum(
            item.actual_class_id == 0
            and item.predicted_class_id == 0
            for item in automatic
        )
        disease_correct = sum(
            item.actual_class_id == 1
            and item.predicted_class_id == 1
            for item in automatic
        )
        disease_as_healthy = sum(
            item.actual_class_id == 1
            and item.predicted_class_id == 0
            for item in automatic
        )
        disease_uncertain = sum(
            item.actual_class_id == 1
            and item.top1_confidence < threshold
            for item in predictions
        )
        rows.append(
            {
                "threshold": float(threshold),
                "total_count": total,
                "auto_decision_count": len(automatic),
                "auto_decision_rate": _safe_divide(
                    len(automatic), total
                ),
                "uncertain_count": uncertain_count,
                "uncertain_rate": _safe_divide(uncertain_count, total),
                "auto_decision_correct_count": automatic_correct,
                "auto_decision_accuracy": _safe_divide(
                    automatic_correct, len(automatic)
                ),
                "healthy_support": healthy_support,
                "healthy_auto_correct_count": healthy_correct,
                "healthy_recall": _safe_divide(
                    healthy_correct, healthy_support
                ),
                "disease_suspected_support": disease_support,
                "disease_suspected_auto_correct_count": disease_correct,
                "disease_suspected_recall": _safe_divide(
                    disease_correct, disease_support
                ),
                "disease_as_healthy_count": disease_as_healthy,
                "disease_uncertain_count": disease_uncertain,
            }
        )
    return rows


def confidence_distribution(
    predictions: Sequence[Prediction],
) -> list[dict[str, Any]]:
    bins = (
        (0.50, 0.60, False),
        (0.60, 0.70, False),
        (0.70, 0.80, False),
        (0.80, 0.90, False),
        (0.90, 0.95, False),
        (0.95, 1.00, True),
    )
    rows: list[dict[str, Any]] = []
    total = len(predictions)
    for lower, upper, include_upper in bins:
        selected = [
            item
            for item in predictions
            if item.top1_confidence >= lower
            and (
                item.top1_confidence <= upper
                if include_upper
                else item.top1_confidence < upper
            )
        ]
        correct = sum(item.is_correct for item in selected)
        rows.append(
            {
                "bin": (
                    f"[{lower:.2f}, {upper:.2f}]"
                    if include_upper
                    else f"[{lower:.2f}, {upper:.2f})"
                ),
                "lower": lower,
                "upper": upper,
                "count": len(selected),
                "rate": _safe_divide(len(selected), total),
                "correct": correct,
                "errors": len(selected) - correct,
                "accuracy": (
                    _safe_divide(correct, len(selected))
                    if selected
                    else None
                ),
            }
        )
    return rows


def select_error_reviews(
    predictions: Sequence[Prediction],
    *,
    max_errors: int = DEFAULT_MAX_ERROR_REVIEW,
) -> list[Prediction]:
    if not 0 <= max_errors <= DEFAULT_MAX_ERROR_REVIEW:
        raise ValueError("오분류 review 상한은 0~100입니다")
    errors = [item for item in predictions if not item.is_correct]
    errors.sort(
        key=lambda item: (
            -item.top1_confidence,
            item.output_relative_path,
        )
    )
    return errors[:max_errors]


def predictions_from_probabilities(
    records: Sequence[ValidationRecord],
    probabilities: Sequence[Sequence[float]],
) -> list[Prediction]:
    if len(records) != len(probabilities):
        raise ValueError("record/probability 수가 다릅니다")
    predictions: list[Prediction] = []
    for record, values in zip(records, probabilities):
        if len(values) != 2:
            raise ValueError("분류 probability tensor는 2개여야 합니다")
        healthy_probability = float(values[0])
        disease_probability = float(values[1])
        if (
            not math.isfinite(healthy_probability)
            or not math.isfinite(disease_probability)
            or healthy_probability < 0
            or disease_probability < 0
            or healthy_probability > 1
            or disease_probability > 1
            or abs(healthy_probability + disease_probability - 1.0) > 1e-4
        ):
            raise ValueError("유효하지 않은 class probability")
        predicted = 0 if healthy_probability >= disease_probability else 1
        predictions.append(
            Prediction(
                actual_class_id=record.actual_class_id,
                predicted_class_id=predicted,
                healthy_probability=healthy_probability,
                disease_probability=disease_probability,
                species=record.species,
                capture_date=record.capture_date,
                camera_id=record.camera_id,
                disease_type=record.disease_type,
                image_archive_id=record.image_archive_id,
                image_member=record.image_member,
                output_relative_path=record.output_relative_path,
            )
        )
    return predictions


def run_ultralytics_inference(
    model_path: Path,
    records: Sequence[ValidationRecord],
    *,
    batch_size: int,
    image_size: int,
    device: str,
    runtime_dir: Path,
    show_progress: bool = True,
) -> tuple[list[Prediction], dict[int, str]]:
    if model_path.name != "best.pt":
        raise ValueError("평가에는 best.pt만 사용할 수 있습니다")
    if not model_path.is_file():
        raise FileNotFoundError("best.pt가 없습니다")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir = runtime_dir / "matplotlib"
    yolo_dir = runtime_dir / "ultralytics"
    mpl_dir.mkdir()
    yolo_dir.mkdir()
    old_cwd = Path.cwd()
    old_environment = {
        name: os.environ.get(name)
        for name in ("MPLCONFIGDIR", "YOLO_CONFIG_DIR")
    }
    probabilities: list[list[float]] = []
    names: dict[int, str]
    progress = tqdm(
        total=len(records),
        desc="건강 분류 Validation",
        unit="image",
        disable=not show_progress,
    )
    try:
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)
        os.environ["YOLO_CONFIG_DIR"] = str(yolo_dir)
        os.chdir(runtime_dir)
        from ultralytics import YOLO

        model = YOLO(str(model_path.resolve()), task="classify")
        names = {
            int(index): str(name)
            for index, name in dict(model.names).items()
        }
        if names != MODEL_CLASS_NAMES:
            raise RuntimeError(
                f"model class mapping 불일치: {names} != "
                f"{MODEL_CLASS_NAMES}"
            )
        predict_device: str | None = None if device == "auto" else device
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            sources = [str(record.image_path.resolve()) for record in chunk]
            source_list = runtime_dir / f"validated_batch_{start:04d}.txt"
            source_list.write_text(
                "\n".join(sources) + "\n",
                encoding="utf-8",
            )
            results = model.predict(
                source=str(source_list),
                imgsz=image_size,
                batch=min(batch_size, len(chunk)),
                device=predict_device,
                save=False,
                verbose=False,
                stream=False,
            )
            if len(results) != len(chunk):
                raise RuntimeError("Ultralytics 결과 수가 입력과 다릅니다")
            expected_paths = {
                record.image_path.resolve() for record in chunk
            }
            probabilities_by_path: dict[Path, list[float]] = {}
            for result in results:
                result_path = Path(result.path).resolve()
                if result_path not in expected_paths:
                    raise RuntimeError("Ultralytics 결과에 미승인 입력 경로가 있습니다")
                if result_path in probabilities_by_path:
                    raise RuntimeError("Ultralytics 결과 입력 경로 중복")
                if result.probs is None:
                    raise RuntimeError("classification probability가 없습니다")
                values = result.probs.data.detach().cpu().tolist()
                if len(values) != 2:
                    raise RuntimeError("모델 출력 class 수가 2가 아닙니다")
                computed_top1 = 0 if values[0] >= values[1] else 1
                if int(result.probs.top1) != computed_top1:
                    raise RuntimeError("Ultralytics top1/class probability 불일치")
                if (
                    abs(
                        float(result.probs.top1conf.detach().cpu())
                        - max(float(value) for value in values)
                    )
                    > 1e-5
                ):
                    raise RuntimeError(
                        "Ultralytics top1 confidence/probability 불일치"
                    )
                probabilities_by_path[result_path] = [
                    float(value) for value in values
                ]
            if set(probabilities_by_path) != expected_paths:
                raise RuntimeError("Ultralytics 결과와 승인 입력 경로 집합 불일치")
            probabilities.extend(
                probabilities_by_path[record.image_path.resolve()]
                for record in chunk
            )
            source_list.unlink()
            if source_list.exists():
                raise RuntimeError("임시 Validation 경로 목록 정리 실패")
            for _record in chunk:
                progress.update(1)
    finally:
        progress.close()
        os.chdir(old_cwd)
        for name, value in old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return predictions_from_probabilities(records, probabilities), names


def _prediction_row(
    item: Prediction,
    prediction_index: int,
) -> dict[str, Any]:
    return {
        "prediction_index": prediction_index,
        "split": "validation",
        "actual_class_id": item.actual_class_id,
        "actual_class_name": item.actual_class_name,
        "predicted_class_id": item.predicted_class_id,
        "predicted_class_name": item.predicted_class_name,
        "is_correct": item.is_correct,
        "healthy_probability": f"{item.healthy_probability:.10f}",
        "disease_suspected_probability": (
            f"{item.disease_probability:.10f}"
        ),
        "top1_confidence": f"{item.top1_confidence:.10f}",
        "species": item.species,
        "capture_date": item.capture_date,
        "camera_id": item.camera_id,
        "disease_type": item.disease_type,
        "image_archive_id": item.image_archive_id,
        "image_member": item.image_member,
        "output_relative_path": item.output_relative_path,
    }


def _error_type(item: Prediction) -> str:
    if item.actual_class_id == 1 and item.predicted_class_id == 0:
        return "FALSE_NEGATIVE_DISEASE"
    return "FALSE_POSITIVE_DISEASE"


def create_error_review(
    predictions: Sequence[Prediction],
    records_by_output: Mapping[str, ValidationRecord],
    output_dir: Path,
    *,
    max_errors: int = DEFAULT_MAX_ERROR_REVIEW,
) -> tuple[dict[str, str], Path, int]:
    review_dir = output_dir / "error_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    selected = select_error_reviews(
        predictions,
        max_errors=max_errors,
    )
    relative_by_output: dict[str, str] = {}
    review_paths: list[Path] = []
    for rank, item in enumerate(selected, start=1):
        record = records_by_output.get(item.output_relative_path)
        if record is None:
            raise ValueError("오분류 review 원본 record가 없습니다")
        try:
            with Image.open(record.image_path) as opened:
                opened.load()
                source = opened.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError("오분류 review 원본 이미지 손상") from exc
        display = ImageOps.contain(source, (1280, 1100))
        panel_height = 150
        canvas = Image.new(
            "RGB",
            (display.width, display.height + panel_height),
            (20, 20, 20),
        )
        canvas.paste(display, (0, panel_height))
        draw = ImageDraw.Draw(canvas)
        font = label_audit.find_font(max(16, min(28, display.width // 35)))
        disease = item.disease_type or "-"
        text = "\n".join(
            (
                f"actual={item.actual_class_name} | "
                f"pred={item.predicted_class_name}",
                f"P(HEALTHY)={item.healthy_probability:.4f} | "
                f"P(DISEASE)={item.disease_probability:.4f}",
                f"{item.species} | camera={item.camera_id} | "
                f"date={item.capture_date} | disease={disease}",
            )
        )
        draw.multiline_text(
            (12, 10),
            text,
            fill=(255, 255, 255),
            font=font,
            spacing=5,
        )
        digest = hashlib.sha256(
            item.output_relative_path.encode("utf-8")
        ).hexdigest()[:16]
        destination = review_dir / f"error_{rank:03d}_{digest}.jpg"
        canvas.save(
            destination,
            format="JPEG",
            quality=92,
            subsampling=0,
            optimize=False,
        )
        relative = destination.relative_to(output_dir).as_posix()
        relative_by_output[item.output_relative_path] = relative
        review_paths.append(destination)

    contact_path = output_dir / "error_contact_sheet.jpg"
    if review_paths:
        tile_size = (320, 300)
        columns = min(5, len(review_paths))
        rows = math.ceil(len(review_paths) / columns)
        sheet = Image.new(
            "RGB",
            (columns * tile_size[0], rows * tile_size[1]),
            (25, 25, 25),
        )
        for index, path in enumerate(review_paths):
            with Image.open(path) as opened:
                opened.load()
                tile = ImageOps.contain(opened.convert("RGB"), tile_size)
            left = (index % columns) * tile_size[0]
            top = (index // columns) * tile_size[1]
            left += (tile_size[0] - tile.width) // 2
            top += (tile_size[1] - tile.height) // 2
            sheet.paste(tile, (left, top))
    else:
        sheet = Image.new("RGB", (800, 180), (25, 25, 25))
        draw = ImageDraw.Draw(sheet)
        draw.text((30, 70), "No misclassified Validation images.", fill="white")
    sheet.save(
        contact_path,
        format="JPEG",
        quality=90,
        subsampling=0,
        optimize=False,
    )
    return relative_by_output, contact_path, len(selected)


def _group_metric_rows(
    predictions: Sequence[Prediction],
    field_name: str,
) -> list[dict[str, Any]]:
    grouped_predictions: dict[str, list[Prediction]] = defaultdict(list)
    for item in predictions:
        grouped_predictions[str(getattr(item, field_name))].append(item)
    rows: list[dict[str, Any]] = []
    for group, items in sorted(grouped_predictions.items()):
        metrics = compute_binary_metrics(items)
        disease = metrics["per_class"]["DISEASE_SUSPECTED"]
        confidences = [item.top1_confidence for item in items]
        rows.append(
            {
                "group_value": group,
                "total": metrics["total"],
                "healthy_support": metrics["per_class"]["HEALTHY"][
                    "support"
                ],
                "disease_suspected_support": disease["support"],
                "correct": metrics["correct"],
                "accuracy": metrics["accuracy"],
                "disease_suspected_tp": disease["tp"],
                "disease_suspected_fn": disease["fn"],
                "disease_suspected_recall": (
                    disease["recall"] if disease["support"] else ""
                ),
                "mean_top1_confidence": mean(confidences),
                "min_top1_confidence": min(confidences),
                "max_top1_confidence": max(confidences),
            }
        )
    return rows


def _species_metric_rows(
    predictions: Sequence[Prediction],
) -> list[dict[str, Any]]:
    grouped = aggregate_group_metrics(predictions, "species")
    rows: list[dict[str, Any]] = []
    for species, metrics in grouped.items():
        healthy_metrics = metrics["per_class"]["HEALTHY"]
        disease_metrics = metrics["per_class"]["DISEASE_SUSPECTED"]
        matrix = metrics["confusion_matrix"]["matrix"]
        rows.append(
            {
                "species": species,
                "total": metrics["total"],
                "healthy_support": healthy_metrics["support"],
                "disease_suspected_support": disease_metrics["support"],
                "healthy_predicted": matrix[0][0] + matrix[1][0],
                "disease_suspected_predicted": matrix[0][1]
                + matrix[1][1],
                "correct": metrics["correct"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "healthy_precision": healthy_metrics["precision"],
                "healthy_recall": healthy_metrics["recall"],
                "healthy_f1": healthy_metrics["f1"],
                "disease_suspected_precision": disease_metrics["precision"],
                "disease_suspected_recall": disease_metrics["recall"],
                "disease_suspected_f1": disease_metrics["f1"],
                "tn": matrix[0][0],
                "fp": matrix[0][1],
                "fn": matrix[1][0],
                "tp": matrix[1][1],
            }
        )
    return rows


def _write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    smoke.write_csv(path, columns, rows)


def _markdown_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> str:
    return health_pilot.markdown_table(headers, rows)


def _format_metric(value: Any) -> str:
    if value == "" or value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_evaluation_reports(
    report_dir: Path,
    predictions: Sequence[Prediction],
    records: Sequence[ValidationRecord],
    *,
    model_reference: str,
    manifest_reference: str,
    validation_reference: str,
    model_class_names: Mapping[int, str],
    review_relative_by_output: Mapping[str, str],
    review_count: int,
    review_limit: int,
    contact_sheet_reference: str,
    image_size: int,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    overall = compute_binary_metrics(predictions)
    species_rows = _species_metric_rows(predictions)
    date_rows = _group_metric_rows(predictions, "capture_date")
    camera_rows = _group_metric_rows(predictions, "camera_id")
    threshold_rows = analyze_thresholds(predictions)
    confidence_rows = confidence_distribution(predictions)
    supported_disease_types = sorted(
        {
            item.disease_type or "<missing>"
            for item in predictions
            if item.actual_class_id == 1
        }
    )
    disease_false_negatives: Counter[str] = Counter(
        {disease_type: 0 for disease_type in supported_disease_types}
    )
    disease_false_negatives.update(
        item.disease_type or "<missing>"
        for item in predictions
        if item.actual_class_id == 1 and item.predicted_class_id == 0
    )
    prediction_index = {
        item.output_relative_path: index
        for index, item in enumerate(predictions, start=1)
    }
    prediction_rows = [
        _prediction_row(item, prediction_index[item.output_relative_path])
        for item in predictions
    ]
    all_errors = sorted(
        (item for item in predictions if not item.is_correct),
        key=lambda item: (
            -item.top1_confidence,
            item.output_relative_path,
        ),
    )
    confidence_rank = {
        item.output_relative_path: rank
        for rank, item in enumerate(all_errors, start=1)
    }
    error_rows: list[dict[str, Any]] = []
    for item in all_errors:
        row = _prediction_row(
            item,
            prediction_index[item.output_relative_path],
        )
        row.update(
            {
                "error_type": _error_type(item),
                "confidence_rank": confidence_rank[
                    item.output_relative_path
                ],
                "review_relative_path": (
                    review_relative_by_output.get(
                        item.output_relative_path, ""
                    )
                ),
            }
        )
        error_rows.append(row)

    overall_payload = {
        "schema_version": 1,
        "scope": {
            "dataset": "health_pilot_date_holdout",
            "split": "validation",
            "validation_images": len(records),
            "train_images_evaluated": 0,
            "test_images_accessed": 0,
            "model": model_reference,
            "manifest": manifest_reference,
            "validation_directory": validation_reference,
            "image_size": image_size,
        },
        "class_mapping": {
            str(class_id): {
                "api_name": CLASS_NAMES[class_id],
                "model_name": model_class_names[class_id],
            }
            for class_id in (0, 1)
        },
        "overall_metrics": overall,
        "disease_as_healthy_by_disease_type": dict(
            sorted(disease_false_negatives.items())
        ),
        "confidence_distribution": confidence_rows,
        "uncertain_threshold_policy": {
            "rule": "top1_confidence < threshold => UNCERTAIN",
            "thresholds_are_final": False,
            "class_recall_denominator": (
                "all actual class rows; UNCERTAIN counts as not recalled"
            ),
            "rows": threshold_rows,
        },
        "error_review": {
            "total_errors": len(all_errors),
            "review_images": review_count,
            "review_limit": review_limit,
            "selection": (
                f"all errors when <={review_limit}, otherwise "
                "highest-confidence errors first"
            ),
            "contact_sheet": contact_sheet_reference,
        },
        "safety": {
            "model_file_modified": False,
            "validation_images_modified": False,
            "test_images_accessed": 0,
            "existing_artifacts_modified": False,
            "outputs_outside_allowed_paths": 0,
            "local_absolute_paths_exposed": 0,
            "model_retrained": False,
        },
    }
    smoke.write_text(
        report_dir / REPORT_FILENAMES[0],
        json.dumps(
            overall_payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    _write_csv(
        report_dir / REPORT_FILENAMES[1],
        SPECIES_METRIC_COLUMNS,
        species_rows,
    )
    _write_csv(
        report_dir / REPORT_FILENAMES[2],
        ("capture_date", *GROUP_METRIC_COLUMNS[1:]),
        (
            {
                "capture_date": row["group_value"],
                **{
                    key: value
                    for key, value in row.items()
                    if key != "group_value"
                },
            }
            for row in date_rows
        ),
    )
    _write_csv(
        report_dir / REPORT_FILENAMES[3],
        ("camera_id", *GROUP_METRIC_COLUMNS[1:]),
        (
            {
                "camera_id": row["group_value"],
                **{
                    key: value
                    for key, value in row.items()
                    if key != "group_value"
                },
            }
            for row in camera_rows
        ),
    )
    _write_csv(
        report_dir / REPORT_FILENAMES[4],
        PREDICTION_COLUMNS,
        prediction_rows,
    )
    _write_csv(
        report_dir / REPORT_FILENAMES[5],
        ERROR_COLUMNS,
        error_rows,
    )
    _write_csv(
        report_dir / REPORT_FILENAMES[6],
        THRESHOLD_COLUMNS,
        threshold_rows,
    )

    lowest_date = min(
        date_rows,
        key=lambda row: (row["accuracy"], row["group_value"]),
    )
    lowest_camera = min(
        camera_rows,
        key=lambda row: (row["accuracy"], row["group_value"]),
    )
    lines = [
        "# 건강 체크 분류기 Validation 상세 평가",
        "",
        "## 범위",
        "",
        f"- 모델: `{model_reference}` (`best.pt`만 사용)",
        f"- manifest: `{manifest_reference}`",
        f"- Validation: **{len(records):,}장**",
        "- Train 평가: 0장",
        "- Test 접근·평가: 0장",
        "- 재학습: 수행하지 않음",
        "",
        "## 전체 지표",
        "",
        f"- Accuracy: **{overall['accuracy']:.6f}**",
        f"- Balanced accuracy: **{overall['balanced_accuracy']:.6f}**",
        f"- Macro precision: **{overall['macro_precision']:.6f}**",
        f"- Macro recall: **{overall['macro_recall']:.6f}**",
        f"- Macro F1: **{overall['macro_f1']:.6f}**",
        "",
        "Confusion matrix는 행이 실제, 열이 예측이며 순서는 "
        "`HEALTHY`, `DISEASE_SUSPECTED`이다.",
        "",
        _markdown_table(
            ("실제\\예측", "HEALTHY", "DISEASE_SUSPECTED"),
            (
                (
                    CLASS_NAMES[index],
                    overall["confusion_matrix"]["matrix"][index][0],
                    overall["confusion_matrix"]["matrix"][index][1],
                )
                for index in (0, 1)
            ),
        ),
        "",
        "## 클래스별 지표",
        "",
        _markdown_table(
            ("클래스", "support", "precision", "recall", "F1"),
            (
                (
                    name,
                    overall["per_class"][name]["support"],
                    f"{overall['per_class'][name]['precision']:.6f}",
                    f"{overall['per_class'][name]['recall']:.6f}",
                    f"{overall['per_class'][name]['f1']:.6f}",
                )
                for name in CLASS_NAMES.values()
            ),
        ),
        "",
        "## 품종별",
        "",
        _markdown_table(
            ("품종", "N", "accuracy", "balanced", "macro F1", "병해 recall"),
            (
                (
                    row["species"],
                    row["total"],
                    f"{row['accuracy']:.6f}",
                    f"{row['balanced_accuracy']:.6f}",
                    f"{row['macro_f1']:.6f}",
                    f"{row['disease_suspected_recall']:.6f}",
                )
                for row in species_rows
            ),
        ),
        "",
        "## 날짜·카메라 최저 성능",
        "",
        f"- 날짜 최저: `{lowest_date['group_value']}` "
        f"accuracy={lowest_date['accuracy']:.6f}, "
        "병해 recall="
        f"{_format_metric(lowest_date['disease_suspected_recall'])}",
        f"- camera_id 최저: `{lowest_camera['group_value']}` "
        f"accuracy={lowest_camera['accuracy']:.6f}, "
        "병해 recall="
        f"{_format_metric(lowest_camera['disease_suspected_recall'])}",
        "- 2021-11-04는 HEALTHY만, 2021-11-26/12-04는 "
        "DISEASE_SUSPECTED만 포함하므로 날짜별 클래스 비교에 주의해야 한다.",
        "",
        "## 병해를 HEALTHY로 반환한 오류",
        "",
        (
            _markdown_table(
                ("병해 종류", "오류"),
                sorted(disease_false_negatives.items()),
            )
            if disease_false_negatives
            else "없음"
        ),
        "",
        "## Confidence 구간",
        "",
        _markdown_table(
            ("구간", "N", "비율", "정답", "오류", "accuracy"),
            (
                (
                    row["bin"],
                    row["count"],
                    f"{row['rate']:.2%}",
                    row["correct"],
                    row["errors"],
                    _format_metric(row["accuracy"]),
                )
                for row in confidence_rows
            ),
        ),
        "",
        "## UNCERTAIN 후보",
        "",
        "규칙은 `top1_confidence < threshold`이다. 클래스 recall은 "
        "전체 실제 클래스가 분모이며 UNCERTAIN은 미회수로 처리한다.",
        "",
        _markdown_table(
            (
                "threshold",
                "자동 비율",
                "UNCERTAIN",
                "자동 accuracy",
                "HEALTHY recall",
                "DISEASE recall",
                "병해→정상",
            ),
            (
                (
                    f"{row['threshold']:.2f}",
                    f"{row['auto_decision_rate']:.2%}",
                    f"{row['uncertain_rate']:.2%}",
                    f"{row['auto_decision_accuracy']:.6f}",
                    f"{row['healthy_recall']:.6f}",
                    f"{row['disease_suspected_recall']:.6f}",
                    row["disease_as_healthy_count"],
                )
                for row in threshold_rows
            ),
        ),
        "",
        "이 표는 현재 Validation 결과를 설명하는 후보 분석이며 서비스 "
        "최종 threshold를 확정하지 않는다.",
        "",
        "## 오분류 review",
        "",
        f"- 전체 오분류: **{len(all_errors):,}장**",
        f"- review 저장: **{review_count:,}장**",
        f"- 오분류가 {review_limit:,}장을 넘으면 top1 confidence가 "
        f"높은 순으로 최대 {review_limit:,}장만 저장한다.",
        f"- contact sheet: `{contact_sheet_reference}`",
        "",
        "## 안전 검증",
        "",
        "- 모델 파일 변경: 0",
        "- Validation 이미지 변경: 0",
        "- Test 이미지 접근: 0",
        "- 기존 artifacts 변경: 0",
        "- 허용 출력 외 파일 생성: 0",
        "- 로컬 절대경로 노출: 0",
        "",
    ]
    smoke.write_text(
        report_dir / REPORT_FILENAMES[7],
        "\n".join(lines),
    )
    return {
        "overall": overall,
        "species_rows": species_rows,
        "date_rows": date_rows,
        "camera_rows": camera_rows,
        "threshold_rows": threshold_rows,
        "confidence_rows": confidence_rows,
        "errors": all_errors,
        "disease_false_negatives": disease_false_negatives,
    }


def assert_deidentified_outputs(paths: Iterable[Path]) -> None:
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


def _file_fingerprint(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return stat.st_size, stat.st_mtime_ns, digest.hexdigest()


def _tree_snapshot(root: Path) -> dict[str, tuple[int, int]]:
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


def _reports_snapshot(
    reports_dir: Path,
    allowed_names: set[str],
) -> dict[str, tuple[int, int]]:
    if not reports_dir.exists():
        return {}
    return {
        path.relative_to(reports_dir).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in reports_dir.rglob("*")
        if path.is_file()
        and path.relative_to(reports_dir).as_posix() not in allowed_names
    }


def _prepare_staging(
    evaluation_dir: Path,
    report_paths: Sequence[Path],
    *,
    overwrite: bool,
) -> Path:
    targets = [evaluation_dir, *report_paths]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "평가 출력이 이미 존재합니다: "
            + ", ".join(path.name for path in existing)
        )
    evaluation_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{evaluation_dir.name}.tmp-",
            dir=evaluation_dir.parent,
        )
    )


def _activate_outputs(
    staging: Path,
    staged_evaluation: Path,
    evaluation_dir: Path,
    staged_reports: Mapping[Path, Path],
) -> tuple[dict[Path, Path], list[Path]]:
    backup_root = staging / "_backups"
    backup_root.mkdir()
    backups: dict[Path, Path] = {}
    targets = [evaluation_dir, *staged_reports]
    activated: list[Path] = []
    try:
        for index, target in enumerate(targets):
            if target.exists():
                backup = backup_root / f"{index:02d}_{target.name}"
                os.replace(target, backup)
                backups[target] = backup
        os.replace(staged_evaluation, evaluation_dir)
        activated.append(evaluation_dir)
        for target, source in staged_reports.items():
            target.parent.mkdir(parents=True, exist_ok=True)
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
        raise
    return backups, activated


def _rollback_activated(
    backups: Mapping[Path, Path],
    activated: Sequence[Path],
) -> None:
    for target in reversed(activated):
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    for target, backup in backups.items():
        if backup.exists():
            os.replace(backup, target)


def _portable_project_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _assert_fixed_evaluation_scope(
    model_path: Path,
    manifest_path: Path,
    validation_dir: Path,
    reports_dir: Path,
    evaluation_dir: Path,
) -> None:
    expected = {
        "model": DEFAULT_MODEL.resolve(),
        "manifest": DEFAULT_MANIFEST.resolve(),
        "validation": DEFAULT_VALIDATION_DIR.resolve(),
        "reports": DEFAULT_REPORTS_DIR.resolve(),
        "evaluation": DEFAULT_EVALUATION_DIR.resolve(),
    }
    actual = {
        "model": model_path.resolve(),
        "manifest": manifest_path.resolve(),
        "validation": validation_dir.resolve(),
        "reports": reports_dir.resolve(),
        "evaluation": evaluation_dir.resolve(),
    }
    mismatches = [
        name for name in expected if actual[name] != expected[name]
    ]
    if mismatches:
        raise ValueError(
            "고정된 date+camera holdout 평가 범위와 다른 경로: "
            + ", ".join(mismatches)
        )


def _project_snapshot_excluding_outputs(
    evaluation_dir: Path,
    report_paths: Sequence[Path],
) -> dict[str, tuple[int, int]]:
    root = PROJECT_ROOT.resolve()
    excluded_evaluation = evaluation_dir.resolve()
    excluded_reports = {path.resolve() for path in report_paths}
    transient_prefixes = (
        f".{excluded_evaluation.name}.tmp-",
        f".{excluded_evaluation.name}.backup-",
    )

    def excluded(path: Path) -> bool:
        if path == excluded_evaluation or path.is_relative_to(
            excluded_evaluation
        ):
            return True
        if path in excluded_reports:
            return True
        try:
            relative = path.relative_to(excluded_evaluation.parent)
        except ValueError:
            return False
        return bool(relative.parts) and relative.parts[0].startswith(
            transient_prefixes
        )

    snapshot: dict[str, tuple[int, int]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if excluded(resolved):
            continue
        stat = path.stat()
        snapshot[path.relative_to(root).as_posix()] = (
            stat.st_size,
            stat.st_mtime_ns,
        )
    return snapshot


def _remove_tree_checked(path: Path) -> None:
    if not path.exists():
        return
    shutil.rmtree(path)
    if path.exists():
        raise RuntimeError(f"임시 출력 정리 실패: {path.name}")


def evaluate(
    *,
    model_path: Path,
    manifest_path: Path,
    validation_dir: Path,
    reports_dir: Path,
    evaluation_dir: Path,
    batch_size: int,
    image_size: int,
    device: str,
    max_error_review: int,
    overwrite: bool,
    show_progress: bool = True,
    expected_count: int = EXPECTED_VALIDATION_IMAGES,
) -> dict[str, Any]:
    _assert_fixed_evaluation_scope(
        model_path,
        manifest_path,
        validation_dir,
        reports_dir,
        evaluation_dir,
    )
    if model_path.name != "best.pt":
        raise ValueError("best.pt만 평가할 수 있습니다")
    if not model_path.is_file():
        raise FileNotFoundError("best.pt가 없습니다")
    if not manifest_path.is_file():
        raise FileNotFoundError("health manifest가 없습니다")
    if not validation_dir.is_dir():
        raise FileNotFoundError("Validation image directory가 없습니다")
    if expected_count != EXPECTED_VALIDATION_IMAGES:
        raise ValueError("평가 범위는 Validation 1,000장으로 고정됩니다")
    if batch_size <= 0 or image_size <= 0:
        raise ValueError("batch_size와 image_size는 양수여야 합니다")
    if not 0 <= max_error_review <= DEFAULT_MAX_ERROR_REVIEW:
        raise ValueError("오분류 review 상한은 0~100장입니다")

    records = load_validation_records(
        manifest_path,
        validation_dir,
        expected_count=expected_count,
    )
    report_paths = [reports_dir / name for name in REPORT_FILENAMES]
    model_before = _file_fingerprint(model_path)
    manifest_before = _file_fingerprint(manifest_path)
    validation_before = _tree_snapshot(validation_dir)
    artifacts_before = health_pilot.snapshot_artifacts_excluding(
        ARTIFACTS_ROOT,
        evaluation_dir,
    )
    reports_before = _reports_snapshot(
        reports_dir,
        set(REPORT_FILENAMES),
    )
    project_before = _project_snapshot_excluding_outputs(
        evaluation_dir,
        report_paths,
    )

    staging = _prepare_staging(
        evaluation_dir,
        report_paths,
        overwrite=overwrite,
    )
    staged_evaluation = staging / "evaluation"
    staged_reports_dir = staging / "reports"
    staged_evaluation.mkdir()
    staged_reports_dir.mkdir()
    staged_report_paths = {
        target: staged_reports_dir / target.name for target in report_paths
    }

    backups: dict[Path, Path] = {}
    activated: list[Path] = []
    try:
        runtime_dir = staged_evaluation / "_runtime"
        predictions, model_names = run_ultralytics_inference(
            model_path,
            records,
            batch_size=batch_size,
            image_size=image_size,
            device=device,
            runtime_dir=runtime_dir,
            show_progress=show_progress,
        )
        _remove_tree_checked(runtime_dir)
        if len(predictions) != expected_count:
            raise RuntimeError(
                f"prediction 수 불일치: {len(predictions)}/{expected_count}"
            )
        records_by_output = {
            record.output_relative_path: record for record in records
        }
        review_map, contact_path, review_count = create_error_review(
            predictions,
            records_by_output,
            staged_evaluation,
            max_errors=max_error_review,
        )
        evaluation_reference = _portable_project_path(evaluation_dir)
        contact_reference = (
            f"{evaluation_reference}/error_contact_sheet.jpg"
        )
        report_data = _write_evaluation_reports(
            staged_reports_dir,
            predictions,
            records,
            model_reference=_portable_project_path(model_path),
            manifest_reference=_portable_project_path(manifest_path),
            validation_reference=_portable_project_path(validation_dir),
            model_class_names=model_names,
            review_relative_by_output={
                key: f"{evaluation_reference}/{value}"
                for key, value in review_map.items()
            },
            review_count=review_count,
            review_limit=max_error_review,
            contact_sheet_reference=contact_reference,
            image_size=image_size,
        )
        if not contact_path.is_file():
            raise RuntimeError("오분류 contact sheet가 생성되지 않았습니다")
        text_outputs = list(staged_reports_dir.iterdir())
        assert_deidentified_outputs(text_outputs)
        if set(path.name for path in text_outputs) != set(REPORT_FILENAMES):
            raise RuntimeError("staged report 파일 집합 불일치")
        with (staged_reports_dir / REPORT_FILENAMES[4]).open(
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            if sum(1 for _row in csv.DictReader(handle)) != expected_count:
                raise RuntimeError("predictions CSV 행 수 불일치")
        error_review_files = list(
            (staged_evaluation / "error_review").glob("*.jpg")
        )
        if len(error_review_files) != review_count:
            raise RuntimeError("오분류 review 파일 수 불일치")
        if not (staged_evaluation / "error_review").is_dir():
            raise RuntimeError("오분류 review 디렉터리가 없습니다")
        expected_evaluation_files = {
            "error_contact_sheet.jpg",
            *{
                f"error_review/{path.name}"
                for path in error_review_files
            },
        }
        actual_evaluation_files = {
            path.relative_to(staged_evaluation).as_posix()
            for path in staged_evaluation.rglob("*")
            if path.is_file()
        }
        if actual_evaluation_files != expected_evaluation_files:
            raise RuntimeError("staged evaluation 파일 집합 불일치")

        if _file_fingerprint(model_path) != model_before:
            raise RuntimeError("모델 파일 변경 감지")
        if _file_fingerprint(manifest_path) != manifest_before:
            raise RuntimeError("manifest 변경 감지")
        if _tree_snapshot(validation_dir) != validation_before:
            raise RuntimeError("Validation 이미지 변경 감지")
        health_pilot.assert_artifacts_unchanged(
            ARTIFACTS_ROOT,
            evaluation_dir,
            artifacts_before,
        )
        if _reports_snapshot(
            reports_dir,
            set(REPORT_FILENAMES),
        ) != reports_before:
            raise RuntimeError("허용 보고서 외 reports 변경 감지")
        if (
            _project_snapshot_excluding_outputs(
                evaluation_dir,
                report_paths,
            )
            != project_before
        ):
            raise RuntimeError("허용 출력 외 프로젝트 파일 변경 감지")

        backups, activated = _activate_outputs(
            staging,
            staged_evaluation,
            evaluation_dir,
            staged_report_paths,
        )
        if _file_fingerprint(model_path) != model_before:
            raise RuntimeError("활성화 후 모델 파일 변경 감지")
        if _file_fingerprint(manifest_path) != manifest_before:
            raise RuntimeError("활성화 후 manifest 변경 감지")
        if _tree_snapshot(validation_dir) != validation_before:
            raise RuntimeError("활성화 후 Validation 이미지 변경 감지")
        health_pilot.assert_artifacts_unchanged(
            ARTIFACTS_ROOT,
            evaluation_dir,
            artifacts_before,
        )
        if _reports_snapshot(
            reports_dir,
            set(REPORT_FILENAMES),
        ) != reports_before:
            raise RuntimeError("활성화 후 허용 외 reports 변경 감지")
        if (
            _project_snapshot_excluding_outputs(
                evaluation_dir,
                report_paths,
            )
            != project_before
        ):
            raise RuntimeError("활성화 후 허용 출력 외 프로젝트 파일 변경 감지")
        assert_deidentified_outputs(report_paths)
        _remove_tree_checked(staging)
    except Exception:
        if activated:
            _rollback_activated(backups, activated)
        _remove_tree_checked(staging)
        if _file_fingerprint(model_path) != model_before:
            raise RuntimeError("실패 처리 중 모델 변경 감지")
        if _file_fingerprint(manifest_path) != manifest_before:
            raise RuntimeError("실패 처리 중 manifest 변경 감지")
        if _tree_snapshot(validation_dir) != validation_before:
            raise RuntimeError("실패 처리 중 Validation 변경 감지")
        health_pilot.assert_artifacts_unchanged(
            ARTIFACTS_ROOT,
            evaluation_dir,
            artifacts_before,
        )
        if _reports_snapshot(
            reports_dir,
            set(REPORT_FILENAMES),
        ) != reports_before:
            raise RuntimeError("실패 처리 중 허용 외 reports 변경 감지")
        if (
            _project_snapshot_excluding_outputs(
                evaluation_dir,
                report_paths,
            )
            != project_before
        ):
            raise RuntimeError("실패 처리 중 허용 출력 외 프로젝트 변경 감지")
        raise

    overall = report_data["overall"]
    disease_as_healthy = overall["confusion_matrix"]["matrix"][1][0]
    return {
        "validation_images": expected_count,
        "accuracy": overall["accuracy"],
        "balanced_accuracy": overall["balanced_accuracy"],
        "macro_f1": overall["macro_f1"],
        "class_metrics": overall["per_class"],
        "species_metrics": report_data["species_rows"],
        "date_metrics": report_data["date_rows"],
        "camera_metrics": report_data["camera_rows"],
        "total_errors": len(report_data["errors"]),
        "disease_as_healthy": disease_as_healthy,
        "disease_false_negatives": dict(
            sorted(report_data["disease_false_negatives"].items())
        ),
        "confidence_distribution": report_data["confidence_rows"],
        "threshold_analysis": report_data["threshold_rows"],
        "error_review_images": review_count,
        "test_images_accessed": 0,
        "model_modified": False,
        "validation_images_modified": False,
        "existing_artifacts_modified": False,
        "outputs_outside_allowed_paths": 0,
        "model_retrained": False,
        "reports": [f"reports/{name}" for name in REPORT_FILENAMES],
        "evaluation_outputs": [
            "artifacts/health_pilot_date_holdout/evaluation/error_review/",
            "artifacts/health_pilot_date_holdout/evaluation/"
            "error_contact_sheet.jpg",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = evaluate(
        model_path=args.model,
        manifest_path=args.manifest,
        validation_dir=args.validation_dir,
        reports_dir=args.reports_dir,
        evaluation_dir=args.evaluation_dir,
        batch_size=args.batch_size,
        image_size=args.imgsz,
        device=args.device,
        max_error_review=args.max_error_review,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
