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
import json
import os
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence


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
        description="AIHub 버섯 ZIP을 압축 해제 없이 인벤토리/샘플 분석합니다.",
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="CSV/JSON/Markdown 보고서 출력 디렉터리",
    )
    args = parser.parse_args(argv)
    if args.extract_samples and args.sample_image_pairs == 0:
        parser.error("--extract-samples에는 1 이상의 --sample-image-pairs가 필요합니다")
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
                if extension in IMAGE_EXTENSIONS:
                    inspection.image_count += 1
            inspection.zip_open_status = "central_directory_ok"
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
                            UnicodeError,
                            json.JSONDecodeError,
                            zipfile.BadZipFile,
                        ) as exc:
                            sample.error = f"{type(exc).__name__}: {exc}"
                        samples.append(sample)
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            samples.append(
                JsonSample(
                    archive=inspection.ref,
                    member_name="",
                    task="",
                    error=f"{type(exc).__name__}: {exc}",
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
                            result.extracted_to = str(destination)
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
                "archive_path": str(item.ref.path),
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
                "image_count": item.image_count,
                "zip_open_status": item.zip_open_status,
                "integrity_status": item.integrity_status,
                "corrupt_status": item.corrupt_status,
                "first_bad_member": item.first_bad_member,
                "error": item.error,
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
        lines.append(f"- `{env_name}`: `{directory}`")
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
    output_dir.mkdir(parents=True, exist_ok=True)
    inspections_list = [
        inspect_archive(ref, check_integrity=args.check_zip_integrity)
        for ref in refs
    ]
    inspections = {item.ref.path: item for item in inspections_list}
    pairing_rows = build_pairing_rows(refs, inspections)
    label_inspections = [
        item
        for item in inspections_list
        if item.ref.kind == "label"
        and item.zip_open_status == "central_directory_ok"
    ]
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
            "image_count",
            "zip_open_status",
            "integrity_status",
            "corrupt_status",
            "first_bad_member",
            "error",
        ),
        inventory_rows(inspections_list),
    )
    write_pairing_report(
        output_dir / "archive_pairing.md", pairing_rows, discovery_issues
    )
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

    failed_archives = [
        item for item in inspections_list if item.zip_open_status != "central_directory_ok"
    ]
    failed_pairs = [row for row in pairing_rows if row["status"] != "ok"]
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
                "output_dir": str(output_dir),
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
