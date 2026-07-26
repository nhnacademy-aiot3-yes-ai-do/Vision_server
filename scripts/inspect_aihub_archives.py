#!/usr/bin/env python3
"""Read-only inventory and sampled schema inspection for AIHub mushroom ZIPs.

The program never extracts an archive wholesale and never decodes images.  Normal
inspection reads ZIP central directories and a bounded number of JSON/image
members.  ``--check-zip-integrity`` is deliberately opt-in because ZipFile.testzip
reads every member in every archive.
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
import sqlite3
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

from tqdm import tqdm


ENVIRONMENT_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("TRAIN_LABEL_ARCHIVES", "train", "label", "TL"),
    ("TRAIN_IMAGE_ARCHIVES", "train", "image", "TS"),
    ("VAL_LABEL_ARCHIVES", "validation", "label", "VL"),
    ("VAL_IMAGE_ARCHIVES", "validation", "image", "VS"),
)
EXPECTED_INDICES = tuple(range(1, 6))
TASKS = ("배양", "생육", "병해")
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}
DECODING_ORDER = ("utf-8-sig", "utf-8", "cp949")
ARCHIVE_RE = re.compile(r"^(TL|TS|VL|VS)([1-5])_(.+)\.zip$", re.IGNORECASE)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports"
DEFAULT_SAMPLE_EXTRACT_DIR = PROJECT_ROOT / "artifacts" / "sample_extract"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
DEFAULT_SCAN_DATABASE = PROJECT_ROOT / "artifacts" / "full_json_scan.sqlite3"
FULL_SCAN_CHECKPOINT_VERSION = 1
FULL_SCAN_SETTINGS_VERSION = 1
DEFAULT_CHECKPOINT_INTERVAL = 5_000
MISSING_VALUE = "<missing>"

NUMERIC_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("temperature", "온도", "TEMPERATURE"),
    ("humidity", "습도", "HUMIDITY"),
    ("co2", "CO2", "CARBON_DIOXIDE"),
    ("illumination", "조도", "ILLUMINATION_INTENSITY"),
    ("pileus_diameter", "갓 직경", "PILEUS_DIAMETER"),
    ("pileus_thickness", "갓 두께", "PILEUS_THICKNESS"),
    ("stipe_length", "대 길이", "STIPE_LENGTH"),
    ("stipe_thickness", "대 두께", "STIPE_THICKNESS"),
    ("gross_weight", "총중량", "GROSS_WEIGHT"),
)

GROWTH_STAGE_KEYWORDS = (
    "STAGE",
    "GROWTH_STAGE",
    "GROWTH_PHASE",
    "생육단계",
    "성장단계",
)
FARM_FACILITY_KEYWORDS = (
    "FARM",
    "FARMER",
    "GROWER",
    "FACILITY",
    "COMPANY",
    "CONTRIBUTOR",
    "농가",
    "농장",
    "재배사",
    "재배시설",
)


@dataclass(frozen=True)
class ArchiveRef:
    env_name: str
    split: str
    kind: str
    prefix: str
    index: int
    species: str
    path: Path

    @property
    def archive_id(self) -> str:
        return f"{self.prefix}{self.index}"


@dataclass
class ArchiveInspection:
    ref: ArchiveRef
    size_bytes: int
    file_count: int = 0
    directory_entry_count: int = 0
    extension_counts: Counter[str] = field(default_factory=Counter)
    top_level_folders: set[str] = field(default_factory=set)
    json_count: int = 0
    json_compressed_bytes: int = 0
    json_uncompressed_bytes: int = 0
    json_manifest_sha256: str = ""
    image_count: int = 0
    zip_open_status: str = "not_checked"
    integrity_status: str = "not_requested"
    first_bad_member: str = ""
    error: str = ""
    member_names: list[str] = field(default_factory=list)

    @property
    def corrupt_status(self) -> str:
        if self.zip_open_status != "central_directory_ok":
            return "yes"
        if self.integrity_status == "failed":
            return "yes"
        if self.integrity_status == "passed":
            return "no"
        return "unknown_full_check_not_requested"


@dataclass
class JsonSample:
    archive: ArchiveRef
    member_name: str
    task: str
    encoding: str = ""
    data: Any = None
    path_values: dict[str, list[Any]] = field(default_factory=dict)
    error: str = ""


@dataclass
class MatchResult:
    sample: JsonSample
    image_filename: str
    matched_member: str
    method: str
    status: str
    candidate_count: int
    extracted_to: str = ""


@dataclass
class PathStat:
    documents_present: int = 0
    documents_non_missing: int = 0
    occurrences: int = 0
    types: Counter[str] = field(default_factory=Counter)
    examples: list[Any] = field(default_factory=list)


@dataclass
class NumericSummary:
    total: int = 0
    present: int = 0
    missing: int = 0
    invalid: int = 0
    value_sum: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def update(self, value: Any) -> None:
        self.total += 1
        if value is None or (isinstance(value, str) and not value.strip()):
            self.missing += 1
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.invalid += 1
            return
        numeric = float(value)
        if not math.isfinite(numeric):
            self.invalid += 1
            return
        self.present += 1
        self.value_sum += numeric
        self.minimum = (
            numeric if self.minimum is None else min(self.minimum, numeric)
        )
        self.maximum = (
            numeric if self.maximum is None else max(self.maximum, numeric)
        )

    @property
    def mean(self) -> float | None:
        return self.value_sum / self.present if self.present else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "present": self.present,
            "missing": self.missing,
            "invalid": self.invalid,
            "value_sum": self.value_sum,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NumericSummary":
        return cls(
            total=int(payload.get("total", 0)),
            present=int(payload.get("present", 0)),
            missing=int(payload.get("missing", 0)),
            invalid=int(payload.get("invalid", 0)),
            value_sum=float(payload.get("value_sum", 0.0)),
            minimum=payload.get("minimum"),
            maximum=payload.get("maximum"),
        )


@dataclass
class CandidateFieldStat:
    occurrences: int = 0
    non_missing: int = 0
    types: Counter[str] = field(default_factory=Counter)
    examples: list[Any] = field(default_factory=list)

    def update(self, value: Any) -> None:
        self.occurrences += 1
        self.types[json_type(value)] += 1
        if is_meaningful(value):
            self.non_missing += 1
        example = compact_example(value)
        if example not in self.examples and len(self.examples) < 5:
            self.examples.append(example)

    def to_dict(self) -> dict[str, Any]:
        return {
            "occurrences": self.occurrences,
            "non_missing": self.non_missing,
            "types": dict(self.types),
            "examples": self.examples,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateFieldStat":
        return cls(
            occurrences=int(payload.get("occurrences", 0)),
            non_missing=int(payload.get("non_missing", 0)),
            types=Counter(payload.get("types", {})),
            examples=list(payload.get("examples", [])),
        )


@dataclass
class FullScanStats:
    attempted_json: int = 0
    processed_json: int = 0
    failed_json: int = 0
    species_counts: Counter[str] = field(default_factory=Counter)
    split_counts: Counter[str] = field(default_factory=Counter)
    task_counts: Counter[str] = field(default_factory=Counter)
    normality_counts: Counter[str] = field(default_factory=Counter)
    disease_counts: Counter[str] = field(default_factory=Counter)
    segmentation_document_counts: Counter[str] = field(default_factory=Counter)
    segmentation_annotation_counts: Counter[str] = field(default_factory=Counter)
    bbox_document_counts: Counter[str] = field(default_factory=Counter)
    bbox_annotation_counts: Counter[str] = field(default_factory=Counter)
    annotation_count_distribution: Counter[int] = field(default_factory=Counter)
    numeric: dict[str, NumericSummary] = field(
        default_factory=lambda: {
            key: NumericSummary() for key, _label, _meta_key in NUMERIC_FIELDS
        }
    )
    camera_counts: Counter[str] = field(default_factory=Counter)
    date_counts: Counter[str] = field(default_factory=Counter)
    date_minimum: str | None = None
    date_maximum: str | None = None
    invalid_date_count: int = 0
    field_presence: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    encoding_counts: Counter[str] = field(default_factory=Counter)
    candidate_fields: dict[str, dict[str, CandidateFieldStat]] = field(
        default_factory=lambda: {
            "growth_stage": {},
            "farm_or_facility_id": {},
        }
    )
    failure_examples: list[dict[str, str]] = field(default_factory=list)

    def add_failure(
        self, archive_id: str, member_name: str, error: Exception
    ) -> None:
        self.attempted_json += 1
        self.failed_json += 1
        if len(self.failure_examples) < 50:
            self.failure_examples.append(
                {
                    "archive": archive_id,
                    "member": normalize_member_name(member_name),
                    "error": redact_local_absolute_paths(
                        f"{type(error).__name__}: {error}"
                    ),
                }
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted_json": self.attempted_json,
            "processed_json": self.processed_json,
            "failed_json": self.failed_json,
            "species_counts": dict(self.species_counts),
            "split_counts": dict(self.split_counts),
            "task_counts": dict(self.task_counts),
            "normality_counts": dict(self.normality_counts),
            "disease_counts": dict(self.disease_counts),
            "segmentation_document_counts": dict(
                self.segmentation_document_counts
            ),
            "segmentation_annotation_counts": dict(
                self.segmentation_annotation_counts
            ),
            "bbox_document_counts": dict(self.bbox_document_counts),
            "bbox_annotation_counts": dict(self.bbox_annotation_counts),
            "annotation_count_distribution": {
                str(key): value
                for key, value in self.annotation_count_distribution.items()
            },
            "numeric": {
                key: value.to_dict() for key, value in self.numeric.items()
            },
            "camera_counts": dict(self.camera_counts),
            "date_counts": dict(self.date_counts),
            "date_minimum": self.date_minimum,
            "date_maximum": self.date_maximum,
            "invalid_date_count": self.invalid_date_count,
            "field_presence": {
                key: dict(value) for key, value in self.field_presence.items()
            },
            "encoding_counts": dict(self.encoding_counts),
            "candidate_fields": {
                kind: {
                    path: stat.to_dict() for path, stat in paths.items()
                }
                for kind, paths in self.candidate_fields.items()
            },
            "failure_examples": self.failure_examples,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FullScanStats":
        stats = cls()
        stats.attempted_json = int(payload.get("attempted_json", 0))
        stats.processed_json = int(payload.get("processed_json", 0))
        stats.failed_json = int(payload.get("failed_json", 0))
        for name in (
            "species_counts",
            "split_counts",
            "task_counts",
            "normality_counts",
            "disease_counts",
            "segmentation_document_counts",
            "segmentation_annotation_counts",
            "bbox_document_counts",
            "bbox_annotation_counts",
            "camera_counts",
            "date_counts",
            "encoding_counts",
        ):
            setattr(stats, name, Counter(payload.get(name, {})))
        stats.annotation_count_distribution = Counter(
            {
                int(key): int(value)
                for key, value in payload.get(
                    "annotation_count_distribution", {}
                ).items()
            }
        )
        stats.numeric = {
            key: NumericSummary.from_dict(
                payload.get("numeric", {}).get(key, {})
            )
            for key, _label, _meta_key in NUMERIC_FIELDS
        }
        stats.date_minimum = payload.get("date_minimum")
        stats.date_maximum = payload.get("date_maximum")
        stats.invalid_date_count = int(payload.get("invalid_date_count", 0))
        stats.field_presence = defaultdict(
            Counter,
            {
                key: Counter(value)
                for key, value in payload.get("field_presence", {}).items()
            },
        )
        stats.candidate_fields = {
            kind: {
                path: CandidateFieldStat.from_dict(stat)
                for path, stat in paths.items()
            }
            for kind, paths in payload.get("candidate_fields", {}).items()
        }
        stats.candidate_fields.setdefault("growth_stage", {})
        stats.candidate_fields.setdefault("farm_or_facility_id", {})
        stats.failure_examples = list(payload.get("failure_examples", []))
        return stats


@dataclass
class FullScanResult:
    stats: FullScanStats
    complete: bool
    resumed: bool
    resume_count: int
    expected_json: int
    json_compressed_bytes: int
    json_uncompressed_bytes: int
    checkpoint_path: Path
    database_path: Path


@dataclass(frozen=True)
class SemanticField:
    key: str
    korean_name: str
    paths: tuple[str, ...]
    note: str = ""
    derived: bool = False


SEMANTIC_FIELDS: tuple[SemanticField, ...] = (
    SemanticField("species", "버섯 품종", ("$.INFO.CATEGORY_NAME",)),
    SemanticField(
        "work_type",
        "배양·생육·병해 구분",
        ("$.INFO.DATASET_NAME",),
        "전용 필드는 없으며 ZIP 멤버의 최상위 폴더에서 확정하고 DATASET_NAME으로 보조 확인",
        derived=True,
    ),
    SemanticField(
        "growth_stage",
        "생육 단계",
        (),
        "샘플 스키마에서 전용 JSON 필드를 찾지 못함; 일부 DATASET_NAME 문자열에 생육/생육2가 포함됨",
    ),
    SemanticField(
        "normality",
        "정상 여부",
        ("$.META.DBYHS_NORMALITY_ALTERNATIVE",),
    ),
    SemanticField("disease_type", "병해 종류", ("$.META.DBYHS_SPCHCKN",)),
    SemanticField(
        "bounding_box",
        "bounding box",
        (
            "$.ANNOTATION_INFO[].BOUNDING_BOX_X_COORDINATE",
            "$.ANNOTATION_INFO[].BOUNDING_BOX_Y_COORDINATE",
            "$.ANNOTATION_INFO[].BOUNDING_BOX_WIDTH",
            "$.ANNOTATION_INFO[].BOUNDING_BOX_HEIGHT",
        ),
        "x, y, width, height의 네 필드로 구성",
    ),
    SemanticField(
        "segmentation",
        "segmentation 또는 polygon",
        ("$.ANNOTATION_INFO[].SEGMENTATION",),
    ),
    SemanticField("temperature", "온도", ("$.META.TEMPERATURE",)),
    SemanticField("humidity", "습도", ("$.META.HUMIDITY",)),
    SemanticField("co2", "CO2", ("$.META.CARBON_DIOXIDE",)),
    SemanticField("illumination", "조도", ("$.META.ILLUMINATION_INTENSITY",)),
    SemanticField("pileus_diameter", "갓 직경", ("$.META.PILEUS_DIAMETER",)),
    SemanticField("pileus_thickness", "갓 두께", ("$.META.PILEUS_THICKNESS",)),
    SemanticField("stipe_length", "대 길이", ("$.META.STIPE_LENGTH",)),
    SemanticField("stipe_thickness", "대 두께", ("$.META.STIPE_THICKNESS",)),
    SemanticField("gross_weight", "총중량", ("$.META.GROSS_WEIGHT",)),
    SemanticField("capture_date", "촬영 날짜", ("$.META.IMAGE_CREATE_DATE",)),
    SemanticField("camera_id", "카메라 ID", ("$.META.IP_CAMERA_ID",)),
    SemanticField(
        "farm_or_facility_id",
        "농가 또는 재배사 식별값",
        (),
        "샘플 스키마에서 전용 식별 필드를 찾지 못함; INFO.CONTRIBUTOR는 존재하지만 값이 비어 있음",
    ),
    SemanticField("image_filename", "이미지 파일명", ("$.IMAGE.IMAGE_FILE_NAME",)),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIHub 버섯 ZIP을 압축 해제 없이 인벤토리/샘플/전체 라벨 "
            "메타데이터 분석합니다."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample-json-per-archive",
        type=bounded_sample_count,
        default=1,
        metavar="N",
        help="라벨 ZIP의 작업 폴더별 JSON 샘플 수(0~20)",
    )
    parser.add_argument(
        "--sample-image-pairs",
        type=bounded_sample_count,
        default=0,
        metavar="N",
        help="라벨/원천 ZIP 쌍별 이미지 매칭 샘플 수(0~20)",
    )
    parser.add_argument(
        "--check-zip-integrity",
        action="store_true",
        help="고비용: 모든 ZIP 멤버를 읽어 CRC를 검사",
    )
    parser.add_argument(
        "--extract-samples",
        action="store_true",
        help="매칭된 제한 이미지 샘플만 artifacts/sample_extract에 추출",
    )
    parser.add_argument(
        "--scan-all-json",
        action="store_true",
        help="라벨 ZIP의 모든 JSON만 스트리밍 집계하고 체크포인트를 저장",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="CSV/JSON/Markdown 보고서 출력 디렉터리",
    )
    args = parser.parse_args(argv)
    if args.extract_samples and args.sample_image_pairs == 0:
        parser.error("--extract-samples에는 1 이상의 --sample-image-pairs가 필요합니다")
    if args.scan_all_json and args.check_zip_integrity:
        parser.error("--scan-all-json은 --check-zip-integrity와 함께 사용할 수 없습니다")
    if args.scan_all_json and args.extract_samples:
        parser.error("--scan-all-json은 이미지 샘플을 추출하지 않습니다")
    if args.scan_all_json and args.sample_image_pairs:
        parser.error("--scan-all-json은 --sample-image-pairs와 함께 사용할 수 없습니다")
    if args.output_dir == Path("reports"):
        args.output_dir = DEFAULT_OUTPUT_DIR
    return args


def bounded_sample_count(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("정수여야 합니다") from exc
    if not 0 <= parsed <= 20:
        raise argparse.ArgumentTypeError("0 이상 20 이하여야 합니다")
    return parsed


def normalize_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def extension_of(name: str) -> str:
    suffix = PurePosixPath(normalize_member_name(name)).suffix.lower()
    return suffix or "(none)"


def classify_task(member_name: str) -> str:
    normalized = normalize_member_name(member_name)
    parts = PurePosixPath(normalized).parts
    for task in TASKS:
        if task in parts:
            return task
    for task in TASKS:
        if task in normalized:
            return task
    return "기타"


def archive_sort_key(path: Path) -> tuple[str, int, str]:
    match = ARCHIVE_RE.match(path.name)
    if not match:
        return ("ZZ", 999, path.name)
    return (match.group(1).upper(), int(match.group(2)), path.name)


def discover_archives(
    environ: Mapping[str, str],
) -> tuple[list[ArchiveRef], list[str], dict[str, Path]]:
    refs: list[ArchiveRef] = []
    issues: list[str] = []
    directories: dict[str, Path] = {}
    dataset_raw = environ.get("MUSHROOM_DATASET", "")
    if dataset_raw:
        dataset_root = Path(dataset_raw).expanduser()
        directories["MUSHROOM_DATASET"] = dataset_root
        if not dataset_root.is_dir():
            issues.append(
                f"MUSHROOM_DATASET 디렉터리가 존재하지 않음: {dataset_root}"
            )
    else:
        issues.append("MUSHROOM_DATASET 환경변수가 설정되지 않음")
    for env_name, split, kind, prefix in ENVIRONMENT_SPECS:
        raw = environ.get(env_name, "")
        if not raw:
            issues.append(f"{env_name} 환경변수가 설정되지 않음")
            continue
        directory = Path(raw).expanduser()
        directories[env_name] = directory
        if not directory.is_dir():
            issues.append(f"{env_name} 디렉터리가 존재하지 않음: {directory}")
            continue
        zip_paths = sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() == ".zip"
            ),
            key=archive_sort_key,
        )
        if len(zip_paths) != 5:
            issues.append(
                f"{env_name}: ZIP {len(zip_paths)}개 발견(기대값 5개)"
            )
        seen_indices: set[int] = set()
        for path in zip_paths:
            match = ARCHIVE_RE.match(path.name)
            if not match:
                issues.append(f"{env_name}: 예상 형식이 아닌 ZIP 이름: {path.name}")
                continue
            actual_prefix, index_text, species = match.groups()
            actual_prefix = actual_prefix.upper()
            index = int(index_text)
            if actual_prefix != prefix:
                issues.append(
                    f"{env_name}: 접두어 {prefix}가 아닌 파일: {path.name}"
                )
                continue
            if index in seen_indices:
                issues.append(f"{env_name}: {prefix}{index} 인덱스 중복")
            seen_indices.add(index)
            refs.append(
                ArchiveRef(
                    env_name=env_name,
                    split=split,
                    kind=kind,
                    prefix=prefix,
                    index=index,
                    species=species,
                    path=path,
                )
            )
        missing = sorted(set(EXPECTED_INDICES) - seen_indices)
        if missing:
            issues.append(
                f"{env_name}: 누락 인덱스 "
                + ", ".join(f"{prefix}{index}" for index in missing)
            )
    refs.sort(key=lambda ref: (ref.split, ref.kind, ref.index))
    return refs, issues, directories


def path_is_within(path: Path, parent: Path) -> bool:
    resolved_path = path.resolve()
    resolved_parent = parent.resolve()
    return (
        resolved_path == resolved_parent
        or resolved_parent in resolved_path.parents
    )


def validate_output_location(
    output_dir: Path, source_directories: Mapping[str, Path]
) -> None:
    for env_name, source in source_directories.items():
        if source.exists() and path_is_within(output_dir, source):
            raise ValueError(
                f"--output-dir는 원본 데이터 경로 밖이어야 합니다: "
                f"{output_dir} (inside {env_name}={source})"
            )


def portable_dataset_path(path: Path, dataset_root: Path) -> str:
    """Render a dataset path without recording the machine's absolute path."""
    try:
        relative = path.resolve().relative_to(dataset_root.resolve())
    except (OSError, ValueError):
        return path.name or "(dataset-root-relative-path-unavailable)"
    if relative == Path("."):
        return "${MUSHROOM_DATASET}"
    return "${MUSHROOM_DATASET}/" + relative.as_posix()


def portable_output_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except (OSError, ValueError):
        return path.name or "(external-output)"


def sanitize_report_text(text: str, dataset_root: Path) -> str:
    sanitized = text.replace(
        str(dataset_root.resolve()), "${MUSHROOM_DATASET}"
    )
    sanitized = sanitized.replace(str(PROJECT_ROOT), "${PROJECT_ROOT}")
    return redact_local_absolute_paths(sanitized)


def redact_local_absolute_paths(text: str) -> str:
    sanitized = re.sub(
        r"(?<![:/$}])/(?:[^\s`\"'|,)]+)",
        "<local-absolute-path>",
        text,
    )
    sanitized = re.sub(
        r"\b[A-Za-z]:[\\/][^\s`\"'|,)]+",
        "<local-absolute-path>",
        sanitized,
    )
    return sanitized


def inspect_archive(
    ref: ArchiveRef, *, check_integrity: bool = False
) -> ArchiveInspection:
    try:
        size = ref.path.stat().st_size
    except OSError as exc:
        return ArchiveInspection(
            ref=ref,
            size_bytes=0,
            zip_open_status="file_error",
            integrity_status="not_run",
            error=f"{type(exc).__name__}: {exc}",
        )
    inspection = ArchiveInspection(ref=ref, size_bytes=size)
    json_manifest = hashlib.sha256()
    try:
        with zipfile.ZipFile(ref.path, mode="r") as archive:
            infos = archive.infolist()
            for info in infos:
                normalized = normalize_member_name(info.filename)
                if info.is_dir() or normalized.endswith("/"):
                    inspection.directory_entry_count += 1
                    continue
                inspection.file_count += 1
                inspection.member_names.append(normalized)
                extension = extension_of(normalized)
                inspection.extension_counts[extension] += 1
                parts = PurePosixPath(normalized).parts
                if len(parts) > 1:
                    inspection.top_level_folders.add(parts[0])
                else:
                    inspection.top_level_folders.add("(root)")
                if extension == ".json":
                    inspection.json_count += 1
                    inspection.json_compressed_bytes += info.compress_size
                    inspection.json_uncompressed_bytes += info.file_size
                    json_manifest.update(
                        normalized.encode("utf-8", errors="surrogatepass")
                    )
                    json_manifest.update(b"\0")
                    json_manifest.update(
                        f"{info.CRC}:{info.compress_size}:{info.file_size}".encode(
                            "ascii"
                        )
                    )
                    json_manifest.update(b"\0")
                if extension in IMAGE_EXTENSIONS:
                    inspection.image_count += 1
            inspection.zip_open_status = "central_directory_ok"
            inspection.json_manifest_sha256 = json_manifest.hexdigest()
            if check_integrity:
                bad_member = archive.testzip()
                if bad_member is None:
                    inspection.integrity_status = "passed"
                else:
                    inspection.integrity_status = "failed"
                    inspection.first_bad_member = normalize_member_name(bad_member)
            else:
                inspection.integrity_status = "not_requested"
    except (OSError, zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        inspection.zip_open_status = "open_failed"
        inspection.integrity_status = "not_run"
        inspection.error = f"{type(exc).__name__}: {exc}"
    return inspection


def build_pairing_rows(
    refs: Sequence[ArchiveRef],
    inspections: Mapping[Path, ArchiveInspection],
) -> list[dict[str, Any]]:
    lookup = {(ref.split, ref.kind, ref.index): ref for ref in refs}
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        label_prefix = "TL" if split == "train" else "VL"
        image_prefix = "TS" if split == "train" else "VS"
        for index in EXPECTED_INDICES:
            label = lookup.get((split, "label", index))
            image = lookup.get((split, "image", index))
            species_match = bool(
                label and image and label.species.casefold() == image.species.casefold()
            )
            open_ok = bool(
                label
                and image
                and inspections[label.path].zip_open_status
                == "central_directory_ok"
                and inspections[image.path].zip_open_status
                == "central_directory_ok"
            )
            status = (
                "ok"
                if label and image and species_match and open_ok
                else "failed"
            )
            rows.append(
                {
                    "split": split,
                    "expected_pair": f"{label_prefix}{index} ↔ {image_prefix}{index}",
                    "label_archive": label.path.name if label else "",
                    "image_archive": image.path.name if image else "",
                    "label_species": label.species if label else "",
                    "image_species": image.species if image else "",
                    "species_match": species_match,
                    "central_directory_open": open_ok,
                    "status": status,
                }
            )
    return rows


def deterministic_spread(items: Sequence[str], count: int) -> list[str]:
    """Select deterministic, reasonably distributed samples from a sorted list."""
    if count <= 0 or not items:
        return []
    ordered = sorted(items)
    if len(ordered) <= count:
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = [
        round(position * (len(ordered) - 1) / (count - 1))
        for position in range(count)
    ]
    return [ordered[index] for index in indices]


def select_json_members(
    inspection: ArchiveInspection, per_task: int
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in inspection.member_names:
        if extension_of(name) == ".json":
            grouped[classify_task(name)].append(name)
    task_order = (*TASKS, "기타")
    return {
        task: deterministic_spread(grouped.get(task, []), per_task)
        for task in task_order
        if grouped.get(task)
    }


def decode_json_bytes(raw: bytes) -> tuple[str, str]:
    errors: list[str] = []
    for encoding in DECODING_ORDER:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeDecodeError(
        "utf-8-sig/utf-8/cp949",
        raw,
        0,
        min(1, len(raw)),
        "; ".join(errors),
    )


def json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def iter_json_nodes(value: Any, path: str = "$") -> Iterator[tuple[str, Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from iter_json_nodes(child, f"{path}.{key}")
    elif isinstance(value, list):
        child_path = f"{path}[]"
        if not value:
            yield child_path, []
        else:
            for child in value:
                yield from iter_json_nodes(child, child_path)


def collect_path_values(data: Any) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = defaultdict(list)
    for path, value in iter_json_nodes(data):
        result[path].append(value)
    return dict(result)


def is_meaningful(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return True


def compact_example(value: Any) -> Any:
    if isinstance(value, dict):
        return {"keys": list(value)[:12], "size": len(value)}
    if isinstance(value, list):
        preview = value[:8]
        return {"preview": preview, "size": len(value)}
    if isinstance(value, str) and len(value) > 160:
        return value[:157] + "..."
    return value


def read_json_samples(
    label_inspections: Sequence[ArchiveInspection], per_task: int
) -> list[JsonSample]:
    samples: list[JsonSample] = []
    for inspection in label_inspections:
        selected = select_json_members(inspection, per_task)
        if not selected:
            continue
        try:
            with zipfile.ZipFile(inspection.ref.path, mode="r") as archive:
                for task, member_names in selected.items():
                    for member_name in member_names:
                        sample = JsonSample(
                            archive=inspection.ref,
                            member_name=member_name,
                            task=task,
                        )
                        try:
                            raw = archive.read(member_name)
                            text, encoding = decode_json_bytes(raw)
                            sample.encoding = encoding
                            sample.data = json.loads(text)
                            sample.path_values = collect_path_values(sample.data)
                        except (
                            KeyError,
                            OSError,
                            RuntimeError,
                            NotImplementedError,
                            UnicodeError,
                            json.JSONDecodeError,
                            zipfile.BadZipFile,
                        ) as exc:
                            sample.error = redact_local_absolute_paths(
                                f"{type(exc).__name__}: {exc}"
                            )
                        samples.append(sample)
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            samples.append(
                JsonSample(
                    archive=inspection.ref,
                    member_name="",
                    task="",
                    error=redact_local_absolute_paths(
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )
    return samples


def aggregate_path_stats(samples: Sequence[JsonSample]) -> dict[str, PathStat]:
    stats: dict[str, PathStat] = defaultdict(PathStat)
    for sample in samples:
        if sample.error:
            continue
        for path, values in sample.path_values.items():
            stat = stats[path]
            stat.documents_present += 1
            stat.occurrences += len(values)
            if any(is_meaningful(value) for value in values):
                stat.documents_non_missing += 1
            for value in values:
                stat.types[json_type(value)] += 1
                example = compact_example(value)
                if example not in stat.examples and len(stat.examples) < 5:
                    stat.examples.append(example)
    return dict(stats)


def semantic_values(sample: JsonSample, field: SemanticField) -> list[Any]:
    if field.key == "work_type":
        return [sample.task] if sample.task else []
    if not field.paths:
        return []
    if field.key == "bounding_box":
        component_values = [
            sample.path_values.get(path, []) for path in field.paths
        ]
        if not all(component_values):
            return []
        item_count = min(len(values) for values in component_values)
        return [
            tuple(component_values[position][index] for position in range(4))
            for index in range(item_count)
        ]
    values: list[Any] = []
    for path in field.paths:
        values.extend(sample.path_values.get(path, []))
    return values


def canonical_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return serialized if len(serialized) <= 240 else serialized[:237] + "..."
    if value is None:
        return "null"
    return str(value)


def build_label_distribution(
    samples: Sequence[JsonSample],
) -> list[dict[str, Any]]:
    valid = [sample for sample in samples if not sample.error]
    grouped: dict[tuple[str, str, str], list[JsonSample]] = defaultdict(list)
    for sample in valid:
        grouped[
            (sample.archive.split, sample.archive.species, sample.task)
        ].append(sample)
    rows: list[dict[str, Any]] = []
    for (split, species, task), group in sorted(grouped.items()):
        total = len(group)
        for field in SEMANTIC_FIELDS:
            present_count = 0
            value_counter: Counter[str] = Counter()
            observed_types: Counter[str] = Counter()
            for sample in group:
                values = semantic_values(sample, field)
                meaningful = [value for value in values if is_meaningful(value)]
                if meaningful:
                    present_count += 1
                for value in meaningful:
                    observed_types[json_type(value)] += 1
                    value_counter[canonical_value(value)] += 1
            missing_count = total - present_count
            top_values = dict(value_counter.most_common(20))
            rows.append(
                {
                    "split": split,
                    "species": species,
                    "task": task,
                    "semantic_key": field.key,
                    "label": field.korean_name,
                    "json_paths": " | ".join(field.paths)
                    if field.paths
                    else "(not found)",
                    "sample_count": total,
                    "non_missing_count": present_count,
                    "missing_count": missing_count,
                    "missing_rate": f"{missing_count / total:.6f}" if total else "",
                    "observed_types": json.dumps(
                        dict(observed_types), ensure_ascii=False, sort_keys=True
                    ),
                    "top_value_counts": json.dumps(
                        top_values, ensure_ascii=False, sort_keys=True
                    ),
                    "note": field.note,
                }
            )
    return rows


def mapping_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None


def canonical_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (bool, int, float)):
        return str(value)
    return None


def candidate_field_kinds(key: str) -> list[str]:
    normalized = re.sub(r"[^A-Z0-9가-힣]+", "_", key.upper())
    kinds: list[str] = []
    if any(keyword in normalized for keyword in GROWTH_STAGE_KEYWORDS):
        kinds.append("growth_stage")
    if any(keyword in normalized for keyword in FARM_FACILITY_KEYWORDS):
        kinds.append("farm_or_facility_id")
    return kinds


def collect_candidate_fields(
    value: Any,
    stats: FullScanStats,
    path: str = "$",
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            for kind in candidate_field_kinds(str(key)):
                candidate = stats.candidate_fields[kind].setdefault(
                    child_path, CandidateFieldStat()
                )
                candidate.update(child)
            collect_candidate_fields(child, stats, child_path)
    elif isinstance(value, list):
        child_path = f"{path}[]"
        for child in value:
            collect_candidate_fields(child, stats, child_path)


def presence_status(value: Any) -> str:
    return "non_missing" if is_meaningful(value) else "missing"


def scalar_presence_status(value: Any) -> str:
    if canonical_scalar(value) is not None:
        return "non_missing"
    return "missing" if not is_meaningful(value) else "invalid"


def valid_bbox(annotation: Any) -> bool:
    if not isinstance(annotation, dict):
        return False
    keys = (
        "BOUNDING_BOX_X_COORDINATE",
        "BOUNDING_BOX_Y_COORDINATE",
        "BOUNDING_BOX_WIDTH",
        "BOUNDING_BOX_HEIGHT",
    )
    return all(key in annotation and annotation[key] is not None for key in keys)


def process_full_json_document(
    stats: FullScanStats,
    archive_ref: ArchiveRef,
    member_name: str,
    data: Any,
    encoding: str,
) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    info = mapping_value(data, "INFO")
    image = mapping_value(data, "IMAGE")
    meta = mapping_value(data, "META")
    annotations_value = mapping_value(data, "ANNOTATION_INFO")
    annotations = annotations_value if isinstance(annotations_value, list) else []

    species_value = mapping_value(info, "CATEGORY_NAME")
    species = canonical_scalar(species_value) or MISSING_VALUE
    task = classify_task(member_name)
    normality = mapping_value(meta, "DBYHS_NORMALITY_ALTERNATIVE")
    disease_value = mapping_value(meta, "DBYHS_SPCHCKN")
    disease = canonical_scalar(disease_value) or MISSING_VALUE
    image_filename_value = mapping_value(image, "IMAGE_FILE_NAME")
    image_filename = canonical_scalar(image_filename_value)
    camera_value = mapping_value(meta, "IP_CAMERA_ID")
    camera = canonical_scalar(camera_value)
    capture_date_value = mapping_value(meta, "IMAGE_CREATE_DATE")
    capture_date = canonical_scalar(capture_date_value)
    capture_time = canonical_scalar(
        mapping_value(meta, "IMAGE_CREATE_TIME")
    )

    stats.attempted_json += 1
    stats.processed_json += 1
    stats.encoding_counts[encoding] += 1
    stats.species_counts[species] += 1
    stats.split_counts[archive_ref.split] += 1
    stats.task_counts[task] += 1
    stats.field_presence["species"][
        scalar_presence_status(species_value)
    ] += 1
    stats.field_presence["image_filename"][
        scalar_presence_status(image_filename_value)
    ] += 1
    stats.field_presence["camera_id"][
        scalar_presence_status(camera_value)
    ] += 1

    if normality is True:
        stats.normality_counts["normal"] += 1
        stats.field_presence["normality"]["non_missing"] += 1
    elif normality is False:
        stats.normality_counts["abnormal"] += 1
        stats.field_presence["normality"]["non_missing"] += 1
    elif normality is None or normality == "":
        stats.normality_counts[MISSING_VALUE] += 1
        stats.field_presence["normality"]["missing"] += 1
    else:
        stats.normality_counts["invalid_or_other"] += 1
        stats.field_presence["normality"]["invalid"] += 1

    stats.disease_counts[disease] += 1
    stats.field_presence["disease_type"][
        presence_status(disease_value)
    ] += 1
    stats.annotation_count_distribution[len(annotations)] += 1

    bbox_present = False
    segmentation_values: list[Any] = []
    for annotation in annotations:
        annotation_bbox = valid_bbox(annotation)
        stats.bbox_annotation_counts[
            "present" if annotation_bbox else "missing"
        ] += 1
        bbox_present = bbox_present or annotation_bbox
        if not isinstance(annotation, dict) or "SEGMENTATION" not in annotation:
            stats.segmentation_annotation_counts["missing"] += 1
            continue
        segmentation = annotation["SEGMENTATION"]
        segmentation_values.append(segmentation)
        stats.segmentation_annotation_counts[
            "null" if segmentation is None else "non_null"
        ] += 1
    stats.bbox_document_counts[
        "present" if bbox_present else "missing"
    ] += 1
    if not segmentation_values:
        segmentation_document_status = "missing"
    elif any(value is not None for value in segmentation_values):
        segmentation_document_status = "non_null"
    else:
        segmentation_document_status = "null"
    stats.segmentation_document_counts[segmentation_document_status] += 1

    for key, _label, meta_key in NUMERIC_FIELDS:
        stats.numeric[key].update(mapping_value(meta, meta_key))

    stats.camera_counts[camera or MISSING_VALUE] += 1
    leakage_date: str | None = None
    if not capture_date:
        stats.field_presence["capture_date"]["missing"] += 1
    else:
        try:
            normalized_date = date.fromisoformat(capture_date).isoformat()
        except ValueError:
            stats.invalid_date_count += 1
            stats.field_presence["capture_date"]["invalid"] += 1
        else:
            stats.field_presence["capture_date"]["non_missing"] += 1
            leakage_date = normalized_date
            stats.date_counts[normalized_date] += 1
            stats.date_minimum = (
                normalized_date
                if stats.date_minimum is None
                else min(stats.date_minimum, normalized_date)
            )
            stats.date_maximum = (
                normalized_date
                if stats.date_maximum is None
                else max(stats.date_maximum, normalized_date)
            )

    collect_candidate_fields(data, stats)

    normalized_filename = (
        normalize_member_name(image_filename).casefold()
        if image_filename
        else None
    )
    image_stem = (
        PurePosixPath(normalized_filename).stem if normalized_filename else None
    )
    return {
        "split": archive_ref.split,
        "archive_id": archive_ref.archive_id,
        "member_name": normalize_member_name(member_name),
        "image_filename": normalized_filename,
        "image_stem": image_stem,
        "species": None if species == MISSING_VALUE else species.casefold(),
        "camera_id": camera.casefold() if camera else None,
        "capture_date": leakage_date,
        "capture_time": capture_time,
    }


def archive_checkpoint_key(ref: ArchiveRef) -> str:
    return f"{ref.split}:{ref.archive_id}:{ref.path.name}"


def full_scan_settings(
    label_inspections: Sequence[ArchiveInspection],
) -> dict[str, Any]:
    archives: list[dict[str, Any]] = []
    for inspection in sorted(
        label_inspections,
        key=lambda item: (item.ref.split, item.ref.index),
    ):
        archives.append(
            {
                "key": archive_checkpoint_key(inspection.ref),
                "archive_name": inspection.ref.path.name,
                "split": inspection.ref.split,
                "species": inspection.ref.species,
                "size_bytes": inspection.size_bytes,
                "json_count": inspection.json_count,
                "json_manifest_sha256": inspection.json_manifest_sha256,
            }
        )
    return {
        "settings_version": FULL_SCAN_SETTINGS_VERSION,
        "encoding_order": list(DECODING_ORDER),
        "numeric_fields": [
            {"key": key, "meta_key": meta_key}
            for key, _label, meta_key in NUMERIC_FIELDS
        ],
        "archives": archives,
    }


def settings_digest(settings: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        settings, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_full_scan_checkpoint(
    path: Path,
    settings: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("checkpoint_version") != FULL_SCAN_CHECKPOINT_VERSION:
        raise ValueError(
            "지원하지 않는 전체 JSON 체크포인트 버전입니다: "
            f"{payload.get('checkpoint_version')}"
        )
    expected_digest = settings_digest(settings)
    if payload.get("settings_digest") != expected_digest:
        raise ValueError(
            "체크포인트의 데이터/분석 설정이 현재 아카이브와 다릅니다. "
            "기존 체크포인트를 보존한 뒤 새 artifacts 경로에서 시작하십시오."
        )
    return payload


def checkpoint_payload(
    *,
    settings: Mapping[str, Any],
    stats: FullScanStats,
    archive_positions: Mapping[str, int],
    completed_archives: set[str],
    created_at: str,
    resumed_count: int,
    status: str,
    database_name: str,
    checkpoint_interval: int,
) -> dict[str, Any]:
    return {
        "checkpoint_version": FULL_SCAN_CHECKPOINT_VERSION,
        "settings_digest": settings_digest(settings),
        "settings": settings,
        "status": status,
        "created_at": created_at,
        "updated_at": datetime.now().astimezone().isoformat(),
        "resumed_count": resumed_count,
        "database_name": database_name,
        "operational_settings": {
            "checkpoint_interval_json": checkpoint_interval,
            "sqlite_journal_mode": "WAL",
            "sqlite_insert_mode": "INSERT OR REPLACE",
        },
        "archive_positions": dict(archive_positions),
        "completed_archives": sorted(completed_archives),
        "statistics": stats.to_dict(),
    }


def initialize_scan_database(
    path: Path,
    current_settings_digest: str,
) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS scan_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    existing = connection.execute(
        "SELECT value FROM scan_metadata WHERE key = 'settings_digest'"
    ).fetchone()
    if existing and existing[0] != current_settings_digest:
        connection.close()
        raise ValueError(
            "SQLite 작업 DB의 분석 설정이 현재 아카이브와 다릅니다. "
            "기존 DB를 보존한 뒤 새 artifacts 경로에서 시작하십시오."
        )
    connection.execute(
        "INSERT OR REPLACE INTO scan_metadata(key, value) VALUES(?, ?)",
        ("settings_digest", current_settings_digest),
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS label_records (
            split TEXT NOT NULL,
            archive_id TEXT NOT NULL,
            member_name TEXT NOT NULL,
            image_filename TEXT,
            image_stem TEXT,
            species TEXT,
            camera_id TEXT,
            capture_date TEXT,
            capture_time TEXT,
            PRIMARY KEY (split, archive_id, member_name)
        )
        """
    )
    connection.commit()
    return connection


def insert_leakage_record(
    connection: sqlite3.Connection, record: Mapping[str, Any]
) -> None:
    connection.execute(
        """
        INSERT OR REPLACE INTO label_records(
            split, archive_id, member_name, image_filename, image_stem,
            species, camera_id, capture_date, capture_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record["split"],
            record["archive_id"],
            record["member_name"],
            record["image_filename"],
            record["image_stem"],
            record["species"],
            record["camera_id"],
            record["capture_date"],
            record["capture_time"],
        ),
    )


def scan_all_label_json(
    label_inspections: Sequence[ArchiveInspection],
    *,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    database_path: Path = DEFAULT_SCAN_DATABASE,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    show_progress: bool = True,
    stop_after: int | None = None,
) -> FullScanResult:
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if any(item.ref.kind != "label" for item in label_inspections):
        raise ValueError("전체 JSON 스캔에는 라벨 ZIP만 전달해야 합니다")
    ordered = sorted(
        label_inspections,
        key=lambda item: (item.ref.split, item.ref.index),
    )
    settings = full_scan_settings(ordered)
    digest = settings_digest(settings)
    checkpoint_path = checkpoint_dir / "full_json_scan_checkpoint.json"
    checkpoint = load_full_scan_checkpoint(checkpoint_path, settings)
    resumed = checkpoint is not None
    if checkpoint:
        stats = FullScanStats.from_dict(checkpoint.get("statistics", {}))
        positions = {
            str(key): int(value)
            for key, value in checkpoint.get("archive_positions", {}).items()
        }
        completed = set(checkpoint.get("completed_archives", []))
        created_at = str(checkpoint.get("created_at", ""))
        resume_count = int(checkpoint.get("resumed_count", 0)) + 1
    else:
        stats = FullScanStats()
        positions: dict[str, int] = {}
        completed: set[str] = set()
        created_at = datetime.now().astimezone().isoformat()
        resume_count = 0

    expected_json = sum(item.json_count for item in ordered)
    compressed_bytes = sum(item.json_compressed_bytes for item in ordered)
    uncompressed_bytes = sum(item.json_uncompressed_bytes for item in ordered)
    if checkpoint and stats.processed_json and not database_path.exists():
        raise ValueError(
            "체크포인트는 존재하지만 SQLite 작업 DB가 없습니다. "
            "체크포인트와 DB를 함께 보존해야 재개할 수 있습니다."
        )
    connection = initialize_scan_database(database_path, digest)
    database_record_count = int(
        connection.execute("SELECT COUNT(*) FROM label_records").fetchone()[0]
    )
    if checkpoint and database_record_count < stats.processed_json:
        connection.close()
        raise ValueError(
            "SQLite 작업 DB의 레코드 수가 체크포인트보다 적어 안전하게 "
            "재개할 수 없습니다."
        )
    outer = tqdm(
        total=len(ordered),
        initial=len(completed),
        desc="전체 라벨 ZIP",
        unit="ZIP",
        position=0,
        disable=not show_progress,
    )
    total_bar = tqdm(
        total=expected_json,
        initial=stats.attempted_json,
        desc="전체 JSON",
        unit="JSON",
        position=1,
        disable=not show_progress,
    )
    attempts_this_run = 0
    attempts_since_checkpoint = 0

    def save(status: str) -> None:
        connection.commit()
        atomic_write_json(
            checkpoint_path,
            checkpoint_payload(
                settings=settings,
                stats=stats,
                archive_positions=positions,
                completed_archives=completed,
                created_at=created_at,
                resumed_count=resume_count,
                status=status,
                database_name=database_path.name,
                checkpoint_interval=checkpoint_interval,
            ),
        )

    try:
        for inspection in ordered:
            key = archive_checkpoint_key(inspection.ref)
            if key in completed:
                continue
            with zipfile.ZipFile(inspection.ref.path, mode="r") as archive:
                json_infos = [
                    info
                    for info in archive.infolist()
                    if not info.is_dir()
                    and extension_of(info.filename) == ".json"
                ]
                start = positions.get(key, 0)
                if start < 0 or start > len(json_infos):
                    raise ValueError(
                        f"체크포인트 JSON 위치가 유효하지 않음: "
                        f"{inspection.ref.archive_id}={start}"
                    )
                current = tqdm(
                    total=len(json_infos),
                    initial=start,
                    desc=f"현재 ZIP {inspection.ref.path.name}",
                    unit="JSON",
                    position=2,
                    leave=False,
                    disable=not show_progress,
                )
                try:
                    for index in range(start, len(json_infos)):
                        info = json_infos[index]
                        member_name = normalize_member_name(info.filename)
                        try:
                            raw = archive.read(info)
                            text, encoding = decode_json_bytes(raw)
                            data = json.loads(text)
                            if not isinstance(data, dict):
                                raise ValueError("JSON root must be an object")
                        except (
                            OSError,
                            RuntimeError,
                            UnicodeError,
                            json.JSONDecodeError,
                            zipfile.BadZipFile,
                            ValueError,
                        ) as exc:
                            stats.add_failure(
                                inspection.ref.archive_id, member_name, exc
                            )
                        else:
                            record = process_full_json_document(
                                stats,
                                inspection.ref,
                                member_name,
                                data,
                                encoding,
                            )
                            insert_leakage_record(connection, record)
                            del data, text, raw
                        positions[key] = index + 1
                        attempts_this_run += 1
                        attempts_since_checkpoint += 1
                        current.update(1)
                        total_bar.update(1)
                        if attempts_since_checkpoint >= checkpoint_interval:
                            save("in_progress")
                            attempts_since_checkpoint = 0
                        if (
                            stop_after is not None
                            and attempts_this_run >= stop_after
                        ):
                            save("in_progress")
                            return FullScanResult(
                                stats=stats,
                                complete=False,
                                resumed=resumed,
                                resume_count=resume_count,
                                expected_json=expected_json,
                                json_compressed_bytes=compressed_bytes,
                                json_uncompressed_bytes=uncompressed_bytes,
                                checkpoint_path=checkpoint_path,
                                database_path=database_path,
                            )
                finally:
                    current.close()
            completed.add(key)
            positions[key] = len(json_infos)
            save("in_progress")
            attempts_since_checkpoint = 0
            outer.update(1)
        save("complete")
        return FullScanResult(
            stats=stats,
            complete=True,
            resumed=resumed,
            resume_count=resume_count,
            expected_json=expected_json,
            json_compressed_bytes=compressed_bytes,
            json_uncompressed_bytes=uncompressed_bytes,
            checkpoint_path=checkpoint_path,
            database_path=database_path,
        )
    except KeyboardInterrupt:
        save("interrupted")
        raise
    finally:
        total_bar.close()
        outer.close()
        connection.close()


def create_leakage_indexes(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_leak_filename "
        "ON label_records(image_filename, split)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_leak_stem "
        "ON label_records(image_stem, split)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_leak_session "
        "ON label_records(species, camera_id, capture_date, capture_time, split)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_leak_camera_date "
        "ON label_records(camera_id, capture_date, split)"
    )
    connection.commit()


def leakage_metric(
    connection: sqlite3.Connection,
    columns: Sequence[str],
    *,
    example_limit: int = 20,
) -> dict[str, Any]:
    column_sql = ", ".join(columns)
    present_sql = " AND ".join(
        f"{column} IS NOT NULL AND {column} <> ''" for column in columns
    )
    grouped = f"""
        SELECT {column_sql},
               SUM(CASE WHEN split = 'train' THEN 1 ELSE 0 END) AS train_count,
               SUM(CASE WHEN split = 'validation' THEN 1 ELSE 0 END) AS validation_count
        FROM label_records
        WHERE {present_sql}
        GROUP BY {column_sql}
        HAVING SUM(CASE WHEN split = 'train' THEN 1 ELSE 0 END) > 0
           AND SUM(CASE WHEN split = 'validation' THEN 1 ELSE 0 END) > 0
    """
    summary = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(train_count), 0), "
        "COALESCE(SUM(validation_count), 0) FROM (" + grouped + ")"
    ).fetchone()
    example_rows = connection.execute(
        "SELECT * FROM (" + grouped + ") "
        "ORDER BY train_count + validation_count DESC, "
        + column_sql
        + " LIMIT ?",
        (example_limit,),
    ).fetchall()
    examples = []
    for row in example_rows:
        examples.append(
            {
                **{
                    column: row[index]
                    for index, column in enumerate(columns)
                },
                "train_count": row[len(columns)],
                "validation_count": row[len(columns) + 1],
            }
        )
    return {
        "duplicate_key_count": int(summary[0]),
        "train_record_count": int(summary[1]),
        "validation_record_count": int(summary[2]),
        "examples": examples,
    }


def analyze_data_leakage(database_path: Path) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(database_path)
    try:
        create_leakage_indexes(connection)
        return {
            "image_filename": leakage_metric(
                connection, ("image_filename",)
            ),
            "image_stem": leakage_metric(connection, ("image_stem",)),
            "capture_session": leakage_metric(
                connection,
                (
                    "species",
                    "camera_id",
                    "capture_date",
                    "capture_time",
                ),
            ),
            "camera_date": leakage_metric(
                connection, ("camera_id", "capture_date")
            ),
        }
    finally:
        connection.close()


def counter_distribution_rows(
    dimension: str,
    counts: Mapping[Any, int],
    total: int,
    note: str = "",
) -> list[dict[str, Any]]:
    rows = []
    for value, count in sorted(counts.items(), key=lambda item: str(item[0])):
        rows.append(
            {
                "dimension": dimension,
                "value": value,
                "count": count,
                "rate": f"{count / total:.8f}" if total else "",
                "note": note,
            }
        )
    return rows


def full_label_distribution_rows(
    stats: FullScanStats,
) -> list[dict[str, Any]]:
    total = stats.processed_json
    rows: list[dict[str, Any]] = []
    for dimension, counter, note in (
        ("species", stats.species_counts, "INFO.CATEGORY_NAME"),
        ("split", stats.split_counts, "라벨 ZIP 구분"),
        ("task", stats.task_counts, "ZIP 내부 최상위 폴더"),
        (
            "normality",
            stats.normality_counts,
            "META.DBYHS_NORMALITY_ALTERNATIVE",
        ),
        ("disease_type", stats.disease_counts, "META.DBYHS_SPCHCKN"),
        (
            "segmentation_document",
            stats.segmentation_document_counts,
            "JSON 단위; annotation 중 하나라도 non-null이면 non_null",
        ),
        (
            "bbox_document",
            stats.bbox_document_counts,
            "JSON 단위; 완전한 x/y/width/height bbox가 하나 이상이면 present",
        ),
        (
            "annotation_count",
            stats.annotation_count_distribution,
            "len(ANNOTATION_INFO)",
        ),
    ):
        rows.extend(counter_distribution_rows(dimension, counter, total, note))
    annotation_total = sum(stats.segmentation_annotation_counts.values())
    rows.extend(
        counter_distribution_rows(
            "segmentation_annotation",
            stats.segmentation_annotation_counts,
            annotation_total,
            "annotation 단위 SEGMENTATION null/non-null/필드 부재",
        )
    )
    bbox_annotation_total = sum(stats.bbox_annotation_counts.values())
    rows.extend(
        counter_distribution_rows(
            "bbox_annotation",
            stats.bbox_annotation_counts,
            bbox_annotation_total,
            "annotation 단위 완전한 x/y/width/height bbox 존재 여부",
        )
    )
    return rows


def completeness_row(
    *,
    key: str,
    label: str,
    path: str,
    total: int,
    non_missing: int,
    missing: int,
    invalid: int = 0,
    note: str = "",
) -> dict[str, Any]:
    return {
        "field_key": key,
        "label": label,
        "json_path": path,
        "total_json": total,
        "non_missing_count": non_missing,
        "missing_count": missing,
        "missing_rate": f"{missing / total:.8f}" if total else "",
        "invalid_count": invalid,
        "note": note,
    }


def full_field_completeness_rows(
    stats: FullScanStats,
) -> list[dict[str, Any]]:
    total = stats.processed_json
    rows: list[dict[str, Any]] = []
    basic = (
        ("species", "버섯 품종", "$.INFO.CATEGORY_NAME"),
        (
            "normality",
            "정상 여부",
            "$.META.DBYHS_NORMALITY_ALTERNATIVE",
        ),
        ("disease_type", "병해 종류", "$.META.DBYHS_SPCHCKN"),
        ("camera_id", "카메라 ID", "$.META.IP_CAMERA_ID"),
        ("capture_date", "촬영 날짜", "$.META.IMAGE_CREATE_DATE"),
        ("image_filename", "이미지 파일명", "$.IMAGE.IMAGE_FILE_NAME"),
    )
    for key, label, path in basic:
        counter = stats.field_presence.get(key, Counter())
        non_missing = int(counter.get("non_missing", 0))
        missing = int(counter.get("missing", 0))
        invalid = int(counter.get("invalid", 0))
        rows.append(
            completeness_row(
                key=key,
                label=label,
                path=path,
                total=total,
                non_missing=non_missing,
                missing=missing,
                invalid=invalid,
            )
        )
    segmentation_non_null = int(
        stats.segmentation_document_counts.get("non_null", 0)
    )
    rows.append(
        completeness_row(
            key="segmentation",
            label="segmentation 또는 polygon",
            path="$.ANNOTATION_INFO[].SEGMENTATION",
            total=total,
            non_missing=segmentation_non_null,
            missing=total - segmentation_non_null,
            note="null과 필드 부재를 결측으로 집계",
        )
    )
    bbox_present = int(stats.bbox_document_counts.get("present", 0))
    rows.append(
        completeness_row(
            key="bounding_box",
            label="bounding box",
            path="$.ANNOTATION_INFO[].BOUNDING_BOX_*",
            total=total,
            non_missing=bbox_present,
            missing=total - bbox_present,
            note="완전한 x/y/width/height bbox가 하나 이상인 JSON",
        )
    )
    for key, label, meta_key in NUMERIC_FIELDS:
        summary = stats.numeric[key]
        rows.append(
            completeness_row(
                key=key,
                label=label,
                path=f"$.META.{meta_key}",
                total=summary.total,
                non_missing=summary.present,
                missing=summary.missing,
                invalid=summary.invalid,
                note="0은 유효값이며 null/빈 문자열과 구분",
            )
        )
    for key, label in (
        ("growth_stage", "별도의 생육 단계 필드"),
        ("farm_or_facility_id", "농가 또는 재배사 식별값"),
    ):
        candidates = stats.candidate_fields.get(key, {})
        rows.append(
            completeness_row(
                key=key,
                label=label,
                path="(전용 필드 확정 안 됨)",
                total=total,
                non_missing=0,
                missing=total,
                note=(
                    f"유사 이름 후보 {len(candidates)}개는 의미를 확정하지 않고 "
                    "full_dataset_analysis.md에 별도 보고"
                ),
            )
        )
    return rows


def full_environment_rows(stats: FullScanStats) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, label, meta_key in NUMERIC_FIELDS:
        summary = stats.numeric[key]
        rows.append(
            {
                "field_key": key,
                "label": label,
                "json_path": f"$.META.{meta_key}",
                "total_json": summary.total,
                "valid_numeric_count": summary.present,
                "missing_count": summary.missing,
                "missing_rate": (
                    f"{summary.missing / summary.total:.8f}"
                    if summary.total
                    else ""
                ),
                "invalid_count": summary.invalid,
                "minimum": summary.minimum,
                "mean": summary.mean,
                "maximum": summary.maximum,
                "zero_is_valid": True,
            }
        )
    return rows


def full_camera_date_rows(stats: FullScanStats) -> list[dict[str, Any]]:
    total = stats.processed_json
    rows: list[dict[str, Any]] = []
    for camera_id, count in sorted(stats.camera_counts.items()):
        rows.append(
            {
                "distribution_type": "camera_id",
                "camera_id": camera_id,
                "capture_date": "",
                "count": count,
                "rate": f"{count / total:.8f}" if total else "",
            }
        )
    for capture_date, count in sorted(stats.date_counts.items()):
        rows.append(
            {
                "distribution_type": "capture_date",
                "camera_id": "",
                "capture_date": capture_date,
                "count": count,
                "rate": f"{count / total:.8f}" if total else "",
            }
        )
    return rows


def write_data_leakage_report(
    path: Path,
    leakage: Mapping[str, Mapping[str, Any]],
    result: FullScanResult,
) -> None:
    labels = {
        "image_filename": "동일 IMAGE_FILE_NAME",
        "image_stem": "동일 이미지 stem",
        "capture_session": "품종+카메라+날짜+시간 촬영 세션",
        "camera_date": "동일 카메라+동일 촬영 날짜",
    }
    lines = [
        "# Train/Validation 데이터 누수 가능성 검사",
        "",
        "분석 범위: 전체 라벨 JSON 스트리밍 스캔",
        "",
        f"- 처리 성공 JSON: {result.stats.processed_json:,}개",
        f"- 실패 JSON: {result.stats.failed_json:,}개",
        f"- 체크포인트에서 재개: {'예' if result.resumed else '아니요'}",
        "- 이미지 바이트 읽기/디코딩: 수행하지 않음",
        "- ZIP 전체 CRC 검사: 수행하지 않음",
        "- 중복 키는 프로젝트 내부 SQLite에서 집계했으며 전체 값을 메모리에 보관하지 않았습니다.",
        "- 이미지명과 stem은 경로 구분자를 정규화하고 대소문자를 구분하지 않아 보수적으로 비교했습니다.",
        "- 이 결과는 중복 가능성 신호이며 곧바로 데이터 누수의 원인을 확정하지 않습니다.",
        "",
        "| 검사 | 양쪽에 존재하는 키 | train 레코드 | validation 레코드 |",
        "|---|---:|---:|---:|",
    ]
    for key, label in labels.items():
        metric = leakage[key]
        lines.append(
            f"| {label} | {metric['duplicate_key_count']:,} | "
            f"{metric['train_record_count']:,} | "
            f"{metric['validation_record_count']:,} |"
        )
    for key, label in labels.items():
        metric = leakage[key]
        lines.extend(["", f"## {label} 예시", ""])
        examples = metric.get("examples", [])
        if not examples:
            lines.append("- 발견되지 않음")
            continue
        lines.append("```json")
        lines.append(json.dumps(examples, ensure_ascii=False, indent=2))
        lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_full_dataset_report(
    path: Path,
    result: FullScanResult,
    leakage: Mapping[str, Mapping[str, Any]],
) -> None:
    stats = result.stats
    resumed_text = (
        f"예 ({result.resume_count}회 재개 기록)"
        if result.resumed
        else "아니요"
    )
    lines = [
        "# AIHub 버섯 전체 라벨 JSON 분석",
        "",
        f"생성 시각: {datetime.now().astimezone().isoformat()}",
        "",
        "## 분석 범위",
        "",
        "- 분석 방식: 전체 라벨 JSON 스트리밍 스캔",
        f"- 전체 예상 JSON: {result.expected_json:,}개",
        f"- 처리 성공 JSON: {stats.processed_json:,}개",
        f"- 실패 JSON: {stats.failed_json:,}개",
        f"- 체크포인트에서 재개: {resumed_text}",
        f"- JSON 압축 멤버 크기 합계: {human_size(result.json_compressed_bytes)}",
        f"- JSON 압축 해제 후 크기 합계: {human_size(result.json_uncompressed_bytes)}",
        "- 이미지 ZIP은 중앙 디렉터리의 파일명만 확인했고 이미지 바이트는 읽거나 디코딩하지 않았습니다.",
        "- ZIP 전체 CRC 검사는 수행하지 않았습니다.",
        "- ZIP 전체 압축 해제, YOLO 변환, 모델 학습을 수행하지 않았습니다.",
        f"- 체크포인트: `{portable_output_path(result.checkpoint_path)}`",
        f"- SQLite 작업 DB: `{portable_output_path(result.database_path)}`",
        "",
        "## 핵심 분포",
        "",
        f"- 품종: `{json.dumps(dict(stats.species_counts), ensure_ascii=False)}`",
        f"- split: `{json.dumps(dict(stats.split_counts), ensure_ascii=False)}`",
        f"- 작업: `{json.dumps(dict(stats.task_counts), ensure_ascii=False)}`",
        f"- 정상 여부: `{json.dumps(dict(stats.normality_counts), ensure_ascii=False)}`",
        f"- segmentation(JSON 단위): "
        f"`{json.dumps(dict(stats.segmentation_document_counts), ensure_ascii=False)}`",
        f"- bbox(JSON 단위): "
        f"`{json.dumps(dict(stats.bbox_document_counts), ensure_ascii=False)}`",
        f"- 촬영 날짜 범위: {stats.date_minimum or '(없음)'} ~ "
        f"{stats.date_maximum or '(없음)'}",
        "",
        "## 환경·형태 수치",
        "",
        "| 필드 | 결측률 | 최소 | 평균 | 최대 | 유효값 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in full_environment_rows(stats):
        missing_rate = (
            float(row["missing_rate"]) if row["missing_rate"] else 0.0
        )
        lines.append(
            f"| {row['label']} | {missing_rate:.2%} | "
            f"{row['minimum']} | {row['mean']} | {row['maximum']} | "
            f"{row['valid_numeric_count']:,} |"
        )
    lines.extend(
        [
            "",
            "## 별도 생육 단계 및 농가/재배사 식별 필드 재검증",
            "",
            "아래 항목은 키 이름이 유사한 후보일 뿐 의미를 임의로 확정하지 않았습니다.",
        ]
    )
    for kind, label in (
        ("growth_stage", "생육 단계 유사 필드"),
        ("farm_or_facility_id", "농가/재배사 유사 필드"),
    ):
        lines.extend(["", f"### {label}", ""])
        candidates = stats.candidate_fields.get(kind, {})
        if not candidates:
            lines.append("- 유사 이름의 JSON 필드도 발견되지 않음")
            continue
        for candidate_path, candidate in sorted(candidates.items()):
            lines.append(
                f"- `{candidate_path}`: 타입 "
                f"`{json.dumps(dict(candidate.types), ensure_ascii=False)}`, "
                f"비결측 {candidate.non_missing:,}/{candidate.occurrences:,}, "
                f"샘플 `{json.dumps(candidate.examples, ensure_ascii=False)}`"
            )
    lines.extend(
        [
            "",
            "## 데이터 누수 가능성",
            "",
        ]
    )
    for key, label in (
        ("image_filename", "동일 IMAGE_FILE_NAME"),
        ("image_stem", "동일 stem"),
        ("capture_session", "동일 촬영 세션"),
        ("camera_date", "동일 카메라·날짜"),
    ):
        lines.append(
            f"- {label}: {leakage[key]['duplicate_key_count']:,}개 중복 키"
        )
    if stats.failure_examples:
        lines.extend(["", "## 실패 JSON 예시", "", "```json"])
        lines.append(
            json.dumps(stats.failure_examples, ensure_ascii=False, indent=2)
        )
        lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_full_scan_reports(
    output_dir: Path,
    result: FullScanResult,
    leakage: Mapping[str, Mapping[str, Any]],
) -> None:
    stats = result.stats
    scope = {
        "analysis_scope": "full_label_json",
        "processed_json": stats.processed_json,
        "failed_json": stats.failed_json,
        "resumed": result.resumed,
        "image_bytes_read": False,
        "full_zip_crc_checked": False,
    }

    def scoped(
        rows: Iterable[Mapping[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        for row in rows:
            yield {**scope, **row}

    scope_fields = (
        "analysis_scope",
        "processed_json",
        "failed_json",
        "resumed",
        "image_bytes_read",
        "full_zip_crc_checked",
    )
    write_csv(
        output_dir / "full_label_distribution.csv",
        scope_fields + ("dimension", "value", "count", "rate", "note"),
        scoped(full_label_distribution_rows(stats)),
    )
    write_csv(
        output_dir / "full_field_completeness.csv",
        scope_fields
        + (
            "field_key",
            "label",
            "json_path",
            "total_json",
            "non_missing_count",
            "missing_count",
            "missing_rate",
            "invalid_count",
            "note",
        ),
        scoped(full_field_completeness_rows(stats)),
    )
    write_csv(
        output_dir / "full_environment_statistics.csv",
        scope_fields
        + (
            "field_key",
            "label",
            "json_path",
            "total_json",
            "valid_numeric_count",
            "missing_count",
            "missing_rate",
            "invalid_count",
            "minimum",
            "mean",
            "maximum",
            "zero_is_valid",
        ),
        scoped(full_environment_rows(stats)),
    )
    write_csv(
        output_dir / "full_camera_date_distribution.csv",
        scope_fields
        + (
            "distribution_type",
            "camera_id",
            "capture_date",
            "count",
            "rate",
        ),
        scoped(full_camera_date_rows(stats)),
    )
    write_data_leakage_report(
        output_dir / "data_leakage_analysis.md", leakage, result
    )
    write_full_dataset_report(
        output_dir / "full_dataset_analysis.md", result, leakage
    )


def round_robin_samples(samples: Sequence[JsonSample], count: int) -> list[JsonSample]:
    groups: dict[str, list[JsonSample]] = defaultdict(list)
    for sample in samples:
        if not sample.error:
            groups[sample.task].append(sample)
    for task_samples in groups.values():
        task_samples.sort(key=lambda item: item.member_name)
    ordered_tasks = [task for task in (*TASKS, "기타") if groups.get(task)]
    selected: list[JsonSample] = []
    position = 0
    while len(selected) < count:
        added = False
        for task in ordered_tasks:
            if position < len(groups[task]):
                selected.append(groups[task][position])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        position += 1
    return selected


def image_filename_for(sample: JsonSample) -> str:
    values = sample.path_values.get("$.IMAGE.IMAGE_FILE_NAME", [])
    for value in values:
        if isinstance(value, str) and value.strip():
            return normalize_member_name(value.strip())
    return ""


def build_image_indexes(
    inspection: ArchiveInspection,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    full: dict[str, list[str]] = defaultdict(list)
    basename: dict[str, list[str]] = defaultdict(list)
    stem: dict[str, list[str]] = defaultdict(list)
    for member in inspection.member_names:
        if extension_of(member) not in IMAGE_EXTENSIONS:
            continue
        normalized = normalize_member_name(member)
        pure = PurePosixPath(normalized)
        full[normalized.casefold()].append(normalized)
        basename[pure.name.casefold()].append(normalized)
        stem[pure.stem.casefold()].append(normalized)
    return dict(full), dict(basename), dict(stem)


def choose_candidate(
    candidates: Sequence[str], method: str
) -> tuple[str, str, str, int]:
    unique = sorted(set(candidates))
    if len(unique) == 1:
        return unique[0], method, "matched", 1
    if len(unique) > 1:
        return "", f"{method}_ambiguous", "ambiguous", len(unique)
    return "", "", "unmatched", 0


def match_image(
    sample: JsonSample,
    indexes: tuple[
        Mapping[str, list[str]],
        Mapping[str, list[str]],
        Mapping[str, list[str]],
    ],
) -> MatchResult:
    image_filename = image_filename_for(sample)
    if not image_filename:
        return MatchResult(
            sample=sample,
            image_filename="",
            matched_member="",
            method="missing_json_image_filename",
            status="unmatched",
            candidate_count=0,
        )
    full, basename, stem = indexes
    label_parent = PurePosixPath(sample.member_name).parent
    image_name = PurePosixPath(image_filename).name
    full_candidates: list[str] = []
    attempted: set[str] = set()
    for candidate_path in (
        image_filename,
        str(label_parent / image_name),
    ):
        normalized = normalize_member_name(candidate_path)
        folded = normalized.casefold()
        if folded in attempted:
            continue
        attempted.add(folded)
        full_candidates.extend(full.get(folded, []))
    member, method, status, candidate_count = choose_candidate(
        full_candidates, "full_path"
    )
    if status != "unmatched":
        return MatchResult(
            sample,
            image_filename,
            member,
            method,
            status,
            candidate_count,
        )
    member, method, status, candidate_count = choose_candidate(
        basename.get(image_name.casefold(), []), "filename"
    )
    if status != "unmatched":
        return MatchResult(
            sample,
            image_filename,
            member,
            method,
            status,
            candidate_count,
        )
    image_stem = PurePosixPath(image_name).stem.casefold()
    member, method, status, candidate_count = choose_candidate(
        stem.get(image_stem, []), "stem"
    )
    return MatchResult(
        sample,
        image_filename,
        member,
        method or "not_found",
        status,
        candidate_count,
    )


def build_match_results(
    refs: Sequence[ArchiveRef],
    inspections: Mapping[Path, ArchiveInspection],
    samples: Sequence[JsonSample],
    pairs_per_archive: int,
) -> list[MatchResult]:
    if pairs_per_archive <= 0:
        return []
    ref_lookup = {(ref.split, ref.kind, ref.index): ref for ref in refs}
    samples_by_archive: dict[Path, list[JsonSample]] = defaultdict(list)
    for sample in samples:
        samples_by_archive[sample.archive.path].append(sample)
    results: list[MatchResult] = []
    label_refs = sorted(
        (ref for ref in refs if ref.kind == "label"),
        key=lambda ref: (ref.split, ref.index),
    )
    for label_ref in label_refs:
        image_ref = ref_lookup.get((label_ref.split, "image", label_ref.index))
        selected = round_robin_samples(
            samples_by_archive.get(label_ref.path, []), pairs_per_archive
        )
        if not image_ref:
            for sample in selected:
                results.append(
                    MatchResult(
                        sample=sample,
                        image_filename=image_filename_for(sample),
                        matched_member="",
                        method="paired_image_archive_missing",
                        status="unmatched",
                        candidate_count=0,
                    )
                )
            continue
        image_inspection = inspections[image_ref.path]
        indexes = build_image_indexes(image_inspection)
        results.extend(match_image(sample, indexes) for sample in selected)
    return results


def safe_destination(root: Path, relative_member: str) -> Path:
    normalized = normalize_member_name(relative_member)
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ValueError(f"안전하지 않은 ZIP 멤버 경로: {relative_member!r}")
    if pure.parts and ":" in pure.parts[0]:
        raise ValueError(f"드라이브 경로로 보이는 ZIP 멤버: {relative_member!r}")
    destination = root.joinpath(*pure.parts)
    resolved_root = root.resolve()
    resolved_destination = destination.resolve()
    if resolved_destination != resolved_root and resolved_root not in resolved_destination.parents:
        raise ValueError(f"추출 루트를 벗어나는 경로: {relative_member!r}")
    return destination


def extract_matched_samples(
    results: Sequence[MatchResult],
    refs: Sequence[ArchiveRef],
    root: Path = DEFAULT_SAMPLE_EXTRACT_DIR,
) -> list[str]:
    errors: list[str] = []
    ref_lookup = {(ref.split, ref.kind, ref.index): ref for ref in refs}
    grouped: dict[Path, list[MatchResult]] = defaultdict(list)
    for result in results:
        if result.status != "matched" or not result.matched_member:
            continue
        image_ref = ref_lookup.get(
            (result.sample.archive.split, "image", result.sample.archive.index)
        )
        if image_ref:
            grouped[image_ref.path].append(result)
    for archive_path, archive_results in grouped.items():
        image_ref = next(ref for ref in refs if ref.path == archive_path)
        archive_root = (
            root
            / image_ref.split
            / f"{image_ref.prefix}{image_ref.index}_{image_ref.species}"
        )
        try:
            with zipfile.ZipFile(archive_path, mode="r") as archive:
                info_lookup = {
                    normalize_member_name(info.filename): info
                    for info in archive.infolist()
                    if not info.is_dir()
                }
                for result in archive_results:
                    try:
                        destination = safe_destination(
                            archive_root, result.matched_member
                        )
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        info = info_lookup[result.matched_member]
                        with archive.open(info, mode="r") as source:
                            with destination.open("wb") as target:
                                shutil.copyfileobj(source, target, length=1024 * 1024)
                        try:
                            result.extracted_to = str(
                                destination.relative_to(PROJECT_ROOT)
                            )
                        except ValueError:
                            result.extracted_to = (
                                "(external-sample-root)/"
                                + result.matched_member
                            )
                    except (KeyError, OSError, RuntimeError, ValueError) as exc:
                        errors.append(
                            f"{archive_path.name}:{result.matched_member}: "
                            f"{type(exc).__name__}: {exc}"
                        )
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            errors.append(f"{archive_path.name}: {type(exc).__name__}: {exc}")
    return errors


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def inventory_rows(
    inspections: Sequence[ArchiveInspection],
    dataset_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in inspections:
        rows.append(
            {
                "split": item.ref.split,
                "kind": item.ref.kind,
                "archive_id": item.ref.archive_id,
                "species": item.ref.species,
                "archive_name": item.ref.path.name,
                "archive_path": portable_dataset_path(
                    item.ref.path, dataset_root
                ),
                "size_bytes": item.size_bytes,
                "size_human": human_size(item.size_bytes),
                "file_count": item.file_count,
                "directory_entry_count": item.directory_entry_count,
                "extension_counts": json.dumps(
                    dict(sorted(item.extension_counts.items())),
                    ensure_ascii=False,
                ),
                "top_level_folders": " | ".join(
                    sorted(item.top_level_folders)
                ),
                "json_count": item.json_count,
                "json_compressed_bytes": item.json_compressed_bytes,
                "json_uncompressed_bytes": item.json_uncompressed_bytes,
                "image_count": item.image_count,
                "zip_open_status": item.zip_open_status,
                "integrity_status": item.integrity_status,
                "corrupt_status": item.corrupt_status,
                "first_bad_member": item.first_bad_member,
                "error": sanitize_report_text(item.error, dataset_root),
            }
        )
    return rows


def schema_payload(
    samples: Sequence[JsonSample],
    path_stats: Mapping[str, PathStat],
    sample_per_task: int,
) -> dict[str, Any]:
    valid_count = sum(not sample.error for sample in samples)
    error_count = sum(bool(sample.error) for sample in samples)
    all_paths = []
    for path, stat in sorted(path_stats.items()):
        all_paths.append(
            {
                "path": path,
                "types": dict(sorted(stat.types.items())),
                "documents_present": stat.documents_present,
                "documents_non_missing": stat.documents_non_missing,
                "occurrences": stat.occurrences,
                "examples": stat.examples,
            }
        )
    semantic: list[dict[str, Any]] = []
    for field in SEMANTIC_FIELDS:
        candidates = []
        for path in field.paths:
            stat = path_stats.get(path)
            candidates.append(
                {
                    "path": path,
                    "found_in_samples": stat is not None,
                    "types": dict(sorted(stat.types.items())) if stat else {},
                    "documents_present": stat.documents_present if stat else 0,
                    "documents_non_missing": (
                        stat.documents_non_missing if stat else 0
                    ),
                    "non_missing_rate": (
                        stat.documents_non_missing / valid_count
                        if stat and valid_count
                        else 0.0
                    ),
                }
            )
        semantic.append(
            {
                "key": field.key,
                "label": field.korean_name,
                "status": (
                    "derived"
                    if field.derived
                    else "path_found_all_values_missing"
                    if candidates
                    and any(item["found_in_samples"] for item in candidates)
                    and not any(
                        item["documents_non_missing"] for item in candidates
                    )
                    else "found"
                    if any(item["found_in_samples"] for item in candidates)
                    else "not_found_in_samples"
                ),
                "candidates": candidates,
                "note": field.note,
            }
        )
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "analysis_scope": {
            "sample_json_per_task_per_archive": sample_per_task,
            "valid_json_samples": valid_count,
            "json_sample_errors": error_count,
            "encoding_attempt_order": list(DECODING_ORDER),
            "sample_based": True,
        },
        "root_structure": {
            "description": "all observed normalized JSON paths, including containers",
            "array_path_notation": "[]",
        },
        "semantic_fields": semantic,
        "all_field_paths": all_paths,
        "sample_errors": [
            {
                "archive": sample.archive.path.name,
                "member": sample.member_name,
                "error": sample.error,
            }
            for sample in samples
            if sample.error
        ],
    }


def write_pairing_report(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    discovery_issues: Sequence[str],
) -> None:
    lines = [
        "# 아카이브 대응 검사",
        "",
        "대응은 인덱스, 품종명, ZIP 중앙 디렉터리 열기 성공 여부를 기준으로 판정했습니다.",
        "",
        "| 구분 | 기대 대응 | 라벨 ZIP | 원천 ZIP | 품종 일치 | 열기 | 결과 |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {split} | {expected_pair} | {label_archive} | {image_archive} | "
            "{species_match} | {central_directory_open} | {status} |".format(
                **row
            )
        )
    lines.extend(["", "## 발견 사항", ""])
    if discovery_issues:
        lines.extend(f"- {issue}" for issue in discovery_issues)
    else:
        lines.append("- 네 디렉터리에서 각각 ZIP 5개를 확인했습니다.")
        lines.append("- 기대한 10개 라벨↔원천 대응이 모두 확인되었습니다.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sample_report(
    path: Path,
    samples: Sequence[JsonSample],
    matches: Sequence[MatchResult],
    extraction_errors: Sequence[str],
    sample_per_task: int,
    pairs_per_archive: int,
) -> None:
    valid = [sample for sample in samples if not sample.error]
    encoding_counts = Counter(sample.encoding for sample in valid)
    task_counts = Counter(
        (sample.archive.split, sample.archive.species, sample.task)
        for sample in valid
    )
    match_counts = Counter(result.status for result in matches)
    method_counts = Counter(result.method for result in matches)
    lines = [
        "# 제한 샘플 검사",
        "",
        f"- 작업별·라벨 ZIP별 JSON 상한: {sample_per_task}",
        f"- 라벨/원천 ZIP 쌍별 이미지 매칭 상한: {pairs_per_archive}",
        f"- JSON 샘플 성공/실패: {len(valid)}/{len(samples) - len(valid)}",
        f"- 인코딩: `{json.dumps(dict(encoding_counts), ensure_ascii=False)}`",
        f"- 이미지 매칭 상태: `{json.dumps(dict(match_counts), ensure_ascii=False)}`",
        f"- 이미지 매칭 방법: `{json.dumps(dict(method_counts), ensure_ascii=False)}`",
        "",
        "## JSON 샘플 분포",
        "",
        "| 구분 | 품종 | 작업 | 수 |",
        "|---|---|---|---:|",
    ]
    for (split, species, task), count in sorted(task_counts.items()):
        lines.append(f"| {split} | {species} | {task} | {count} |")
    lines.extend(
        [
            "",
            "## 이미지 샘플 매칭",
            "",
            "| 구분 | 품종 | JSON | 이미지 필드 | 방법 | 결과 | 추출 위치 |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for result in matches:
        lines.append(
            f"| {result.sample.archive.split} | {result.sample.archive.species} | "
            f"`{result.sample.member_name}` | `{result.image_filename}` | "
            f"{result.method} | {result.status} | "
            f"`{result.extracted_to}` |"
        )
    if not matches:
        lines.append("| - | - | - | - | 비활성 | - | - |")
    if extraction_errors:
        lines.extend(["", "## 샘플 추출 오류", ""])
        lines.extend(f"- {error}" for error in extraction_errors)
    lines.extend(
        [
            "",
            "이미지 파일은 ZIP 멤버를 바이트 단위로 복사했으며 디코딩하지 않았습니다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_report(
    path: Path,
    directories: Mapping[str, Path],
    inspections: Sequence[ArchiveInspection],
    pairing_rows: Sequence[Mapping[str, Any]],
    samples: Sequence[JsonSample],
    distribution_rows: Sequence[Mapping[str, Any]],
    matches: Sequence[MatchResult],
    check_integrity: bool,
    extract_samples: bool,
) -> None:
    dataset_root = directories["MUSHROOM_DATASET"]
    total_size = sum(item.size_bytes for item in inspections)
    total_files = sum(item.file_count for item in inspections)
    total_json = sum(item.json_count for item in inspections)
    total_images = sum(item.image_count for item in inspections)
    pair_ok = sum(row["status"] == "ok" for row in pairing_rows)
    valid_samples = sum(not sample.error for sample in samples)
    match_counts = Counter(result.status for result in matches)
    missing_by_field: dict[str, list[float]] = defaultdict(list)
    for row in distribution_rows:
        missing_by_field[str(row["label"])].append(float(row["missing_rate"]))
    lines = [
        "# AIHub 지능형 스마트팜 통합 데이터(버섯) 인벤토리",
        "",
        f"생성 시각: {datetime.now().astimezone().isoformat()}",
        "",
        "## 데이터 위치",
        "",
    ]
    for env_name, directory in directories.items():
        lines.append(
            f"- `{env_name}`: "
            f"`{portable_dataset_path(directory, dataset_root)}`"
        )
    lines.extend(
        [
            "",
            "## 전체 요약",
            "",
            f"- ZIP: {len(inspections)}개, 합계 {human_size(total_size)}",
            f"- 중앙 디렉터리 기준 내부 파일: {total_files:,}개",
            f"- JSON: {total_json:,}개",
            f"- 이미지: {total_images:,}개",
            f"- 정상 대응: {pair_ok}/{len(pairing_rows)}쌍",
            f"- 읽기 성공 JSON 샘플: {valid_samples}개",
            f"- 이미지 매칭: `{json.dumps(dict(match_counts), ensure_ascii=False)}`",
            "",
            "## 안전 범위",
            "",
            "- 원본 ZIP은 읽기 전용으로 열었고 이름 변경·이동·삭제·수정하지 않았습니다.",
            "- 전체 압축 해제 및 이미지 디코딩을 수행하지 않았습니다.",
            (
                "- 전체 ZIP CRC 무결성 검사를 수행했습니다."
                if check_integrity
                else "- 전체 ZIP CRC 무결성 검사는 수행하지 않았습니다. "
                "인벤토리의 손상 여부는 중앙 디렉터리 판독 범위만 확인되었습니다."
            ),
            (
                "- 제한된 매칭 이미지 샘플만 artifacts/sample_extract 아래에 추출했습니다."
                if extract_samples
                else "- 이미지 샘플 추출은 수행하지 않았습니다."
            ),
            "",
            "## 샘플 기준 평균 결측률",
            "",
            "| 라벨 | 평균 결측률 |",
            "|---|---:|",
        ]
    )
    for label, rates in missing_by_field.items():
        average = sum(rates) / len(rates) if rates else 0.0
        lines.append(f"| {label} | {average:.2%} |")
    lines.extend(
        [
            "",
            "세부 경로·타입은 `label_schema.json`, 품종/작업별 결측률과 값 분포는 "
            "`label_distribution.csv`를 참조하십시오.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def unmatched_rows(matches: Sequence[MatchResult]) -> list[dict[str, Any]]:
    rows = []
    for result in matches:
        if result.status == "matched":
            continue
        rows.append(
            {
                "split": result.sample.archive.split,
                "species": result.sample.archive.species,
                "label_archive": result.sample.archive.path.name,
                "json_member": result.sample.member_name,
                "image_filename": result.image_filename,
                "status": result.status,
                "match_method": result.method,
                "candidate_count": result.candidate_count,
            }
        )
    return rows


def run_analysis(args: argparse.Namespace, environ: Mapping[str, str]) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    refs, discovery_issues, directories = discover_archives(environ)
    validate_output_location(output_dir, directories)
    if "MUSHROOM_DATASET" not in directories:
        raise ValueError("MUSHROOM_DATASET 환경변수가 필요합니다")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = directories["MUSHROOM_DATASET"]
    inspections_list = [
        inspect_archive(ref, check_integrity=args.check_zip_integrity)
        for ref in refs
    ]
    inspections = {item.ref.path: item for item in inspections_list}
    pairing_rows = build_pairing_rows(refs, inspections)
    write_csv(
        output_dir / "archive_inventory.csv",
        (
            "split",
            "kind",
            "archive_id",
            "species",
            "archive_name",
            "archive_path",
            "size_bytes",
            "size_human",
            "file_count",
            "directory_entry_count",
            "extension_counts",
            "top_level_folders",
            "json_count",
            "json_compressed_bytes",
            "json_uncompressed_bytes",
            "image_count",
            "zip_open_status",
            "integrity_status",
            "corrupt_status",
            "first_bad_member",
            "error",
        ),
        inventory_rows(inspections_list, dataset_root),
    )
    write_pairing_report(
        output_dir / "archive_pairing.md",
        pairing_rows,
        [
            sanitize_report_text(issue, dataset_root)
            for issue in discovery_issues
        ],
    )
    label_inspections = [
        item
        for item in inspections_list
        if item.ref.kind == "label"
        and item.zip_open_status == "central_directory_ok"
    ]
    failed_archives = [
        item
        for item in inspections_list
        if item.zip_open_status != "central_directory_ok"
    ]
    failed_pairs = [row for row in pairing_rows if row["status"] != "ok"]

    if getattr(args, "scan_all_json", False):
        result = scan_all_label_json(label_inspections)
        if result.complete:
            leakage = analyze_data_leakage(result.database_path)
            write_full_scan_reports(output_dir, result, leakage)
        print(
            json.dumps(
                {
                    "analysis_scope": "full_label_json",
                    "complete": result.complete,
                    "expected_json": result.expected_json,
                    "processed_json": result.stats.processed_json,
                    "failed_json": result.stats.failed_json,
                    "resumed": result.resumed,
                    "resume_count": result.resume_count,
                    "image_bytes_read": False,
                    "full_zip_crc_checked": False,
                    "checkpoint": portable_output_path(
                        result.checkpoint_path
                    ),
                    "database": portable_output_path(result.database_path),
                    "output_dir": portable_output_path(output_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return (
            1
            if discovery_issues
            or failed_archives
            or failed_pairs
            or result.stats.failed_json
            or not result.complete
            else 0
        )

    samples = read_json_samples(
        label_inspections, args.sample_json_per_archive
    )
    if args.sample_image_pairs > args.sample_json_per_archive:
        pair_samples = read_json_samples(
            label_inspections, args.sample_image_pairs
        )
    else:
        pair_samples = samples
    path_stats = aggregate_path_stats(samples)
    distribution = build_label_distribution(samples)
    matches = build_match_results(
        refs,
        inspections,
        pair_samples,
        args.sample_image_pairs,
    )
    extraction_errors: list[str] = []
    if args.extract_samples:
        extraction_errors = extract_matched_samples(matches, refs)

    (output_dir / "label_schema.json").write_text(
        json.dumps(
            schema_payload(
                samples, path_stats, args.sample_json_per_archive
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(
        output_dir / "label_distribution.csv",
        (
            "split",
            "species",
            "task",
            "semantic_key",
            "label",
            "json_paths",
            "sample_count",
            "non_missing_count",
            "missing_count",
            "missing_rate",
            "observed_types",
            "top_value_counts",
            "note",
        ),
        distribution,
    )
    write_csv(
        output_dir / "unmatched_sample_files.csv",
        (
            "split",
            "species",
            "label_archive",
            "json_member",
            "image_filename",
            "status",
            "match_method",
            "candidate_count",
        ),
        unmatched_rows(matches),
    )
    write_sample_report(
        output_dir / "sample_inspection.md",
        samples,
        matches,
        extraction_errors,
        args.sample_json_per_archive,
        args.sample_image_pairs,
    )
    write_dataset_report(
        output_dir / "dataset_inventory.md",
        directories,
        inspections_list,
        pairing_rows,
        samples,
        distribution,
        matches,
        args.check_zip_integrity,
        args.extract_samples,
    )

    sample_errors = [sample for sample in samples if sample.error]
    pair_sample_errors = [
        sample for sample in pair_samples if sample.error
    ] if pair_samples is not samples else []
    print(
        json.dumps(
            {
                "archives": len(inspections_list),
                "discovery_issues": len(discovery_issues),
                "failed_archives": len(failed_archives),
                "pairing_ok": len(pairing_rows) - len(failed_pairs),
                "pairing_total": len(pairing_rows),
                "valid_json_samples": len(samples) - len(sample_errors),
                "json_sample_errors": len(sample_errors),
                "image_pair_json_sample_errors": len(pair_sample_errors),
                "image_match_status": dict(Counter(r.status for r in matches)),
                "extraction_errors": len(extraction_errors),
                "output_dir": portable_output_path(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return (
        1
        if discovery_issues
        or failed_archives
        or failed_pairs
        or sample_errors
        or pair_sample_errors
        or extraction_errors
        else 0
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_analysis(args, os.environ)


if __name__ == "__main__":
    sys.exit(main())
