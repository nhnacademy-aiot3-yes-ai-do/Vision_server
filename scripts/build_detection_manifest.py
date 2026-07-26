#!/usr/bin/env python3
"""Build a leakage-safe detection manifest without reading image bytes.

Only label JSON members are decompressed.  Image ZIPs are opened to read their
central directories, and source ZIPs are checked before/after the run for size
and mtime changes.  No image extraction, decoding, YOLO conversion, or training
is performed by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tqdm import tqdm

import audit_mushroom_labels as audit
import inspect_aihub_archives as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / "artifacts" / "detection_manifest.sqlite3"
DEFAULT_QUALITY_DATABASE = PROJECT_ROOT / "artifacts" / "quality_analysis.sqlite3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports"
SCHEMA_VERSION = 1
DEFAULT_SEED = 20260726
TARGET_TASKS = frozenset(("생육", "병해"))
SPECIES_CLASS_IDS = {
    "느타리": 0,
    "양송이": 1,
    "큰느타리": 2,
    "팽이": 3,
    "표고": 4,
}
SPLITS = ("train", "validation", "test")
MANIFEST_COLUMNS = (
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
    "original_annotation_count",
    "valid_bbox_count",
    "bbox_original",
    "bbox_cleaned",
    "bbox_cleanup_flags",
    "has_segmentation",
    "official_split",
)


@dataclass
class BboxCleanupResult:
    original: list[dict[str, Any]] = field(default_factory=list)
    cleaned: list[dict[str, Any]] = field(default_factory=list)
    flags: list[dict[str, Any]] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "생육·병해 라벨 JSON만 읽어 bbox를 정제하고 누수 없는 "
            "Train/Validation/Test manifest를 생성합니다. 이미지 바이트는 "
            "읽지 않습니다."
        )
    )
    parser.add_argument(
        "--build-manifest",
        action="store_true",
        help="실제 라벨 JSON을 순차 스캔하여 manifest와 검증 보고서를 생성",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("artifacts/detection_manifest.sqlite3"),
        help="프로젝트 기준 작업용 SQLite 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="프로젝트 기준 보고서 출력 경로",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="tqdm 진행률을 표시하지 않음",
    )
    args = parser.parse_args(argv)
    if not args.build_manifest:
        parser.error("--build-manifest가 필요합니다")
    args.database = project_path(args.database)
    args.output_dir = project_path(args.output_dir)
    return args


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def compact_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def json_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def normalized(value: Any, default: str = "") -> str:
    result = audit.normalized_scalar(value)
    return result if result is not None else default


def bbox_values_with_raw(
    annotation: Any,
) -> tuple[dict[str, Any], tuple[float, float, float, float] | None]:
    keys = (
        "BOUNDING_BOX_X_COORDINATE",
        "BOUNDING_BOX_Y_COORDINATE",
        "BOUNDING_BOX_WIDTH",
        "BOUNDING_BOX_HEIGHT",
    )
    if not isinstance(annotation, dict):
        return {"value": annotation}, None
    raw = {
        "x": annotation.get(keys[0]),
        "y": annotation.get(keys[1]),
        "width": annotation.get(keys[2]),
        "height": annotation.get(keys[3]),
    }
    numbers = tuple(audit.numeric(annotation.get(key)) for key in keys)
    if any(value is None for value in numbers):
        return raw, None
    return raw, (numbers[0], numbers[1], numbers[2], numbers[3])  # type: ignore[arg-type]


def clean_annotations(
    annotations: Any,
    image_width: int | None,
    image_height: int | None,
) -> BboxCleanupResult:
    """Clean boxes while retaining an annotation-index audit trail."""
    result = BboxCleanupResult()
    if not isinstance(annotations, list):
        result.counts["invalid_annotation_container"] += 1
        return result
    dimensions_valid = bool(
        image_width is not None
        and image_height is not None
        and image_width > 0
        and image_height > 0
    )
    for index, annotation in enumerate(annotations):
        raw, values = bbox_values_with_raw(annotation)
        original_entry = {"annotation_index": index, **raw}
        result.original.append(original_entry)
        annotation_flags: list[str] = []
        if values is None:
            annotation_flags.append("removed_incomplete_bbox")
            result.counts["removed_incomplete_bbox"] += 1
        elif not dimensions_valid:
            annotation_flags.append("removed_invalid_image_dimensions")
            result.counts["removed_invalid_image_dimensions"] += 1
        else:
            x, y, width, height = values
            if width <= 0 or height <= 0:
                annotation_flags.append("removed_nonpositive_bbox")
                result.counts["removed_nonpositive_bbox"] += 1
            else:
                x1 = max(0.0, min(float(image_width), x))
                y1 = max(0.0, min(float(image_height), y))
                x2 = max(0.0, min(float(image_width), x + width))
                y2 = max(0.0, min(float(image_height), y + height))
                cleaned_width = x2 - x1
                cleaned_height = y2 - y1
                if (x1, y1, x2, y2) != (
                    x,
                    y,
                    x + width,
                    y + height,
                ):
                    annotation_flags.append("clipped_to_image_bounds")
                    result.counts["clipped_to_image_bounds"] += 1
                if cleaned_width < 1 or cleaned_height < 1:
                    annotation_flags.append("removed_after_clip_lt_1px")
                    result.counts["removed_after_clip_lt_1px"] += 1
                else:
                    area_fraction = (
                        cleaned_width
                        * cleaned_height
                        / (float(image_width) * float(image_height))
                    )
                    if area_fraction < audit.TINY_BBOX_AREA_FRACTION:
                        annotation_flags.append("retained_small_bbox")
                        result.counts["retained_small_bbox"] += 1
                    result.cleaned.append(
                        {
                            "annotation_index": index,
                            "x": json_number(x1),
                            "y": json_number(y1),
                            "width": json_number(cleaned_width),
                            "height": json_number(cleaned_height),
                        }
                    )
        if annotation_flags:
            result.flags.append(
                {"annotation_index": index, "flags": annotation_flags}
            )
    result.counts["original_bbox_candidates"] = len(result.original)
    result.counts["valid_bbox"] = len(result.cleaned)
    return result


def initialize_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE detection_records (
            record_key TEXT PRIMARY KEY,
            new_split TEXT,
            group_key TEXT NOT NULL,
            species TEXT NOT NULL,
            class_id INTEGER NOT NULL,
            task TEXT NOT NULL,
            normality TEXT NOT NULL,
            disease_type TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            capture_date TEXT NOT NULL,
            capture_time TEXT NOT NULL,
            label_archive_id TEXT NOT NULL,
            image_archive_id TEXT NOT NULL,
            json_member TEXT NOT NULL,
            image_member TEXT NOT NULL,
            image_member_key TEXT NOT NULL,
            image_width INTEGER NOT NULL,
            image_height INTEGER NOT NULL,
            original_annotation_count INTEGER NOT NULL,
            valid_bbox_count INTEGER NOT NULL,
            bbox_original TEXT NOT NULL,
            bbox_cleaned TEXT NOT NULL,
            bbox_cleanup_flags TEXT NOT NULL,
            has_segmentation INTEGER NOT NULL,
            official_split TEXT NOT NULL,
            removed_nonpositive_count INTEGER NOT NULL,
            removed_after_clip_count INTEGER NOT NULL,
            removed_incomplete_count INTEGER NOT NULL,
            clipped_count INTEGER NOT NULL,
            retained_small_count INTEGER NOT NULL
        );
        CREATE TABLE excluded_records (
            record_key TEXT PRIMARY KEY,
            exclusion_reason TEXT NOT NULL,
            detail_flags TEXT NOT NULL,
            official_split TEXT NOT NULL,
            label_archive_id TEXT NOT NULL,
            image_archive_id TEXT NOT NULL,
            species TEXT NOT NULL,
            task TEXT NOT NULL,
            json_member TEXT NOT NULL,
            image_member TEXT NOT NULL,
            original_annotation_count INTEGER,
            removed_bbox_count INTEGER NOT NULL
        );
        """
    )
    connection.execute(
        "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    connection.commit()
    return connection


def insert_excluded(
    connection: sqlite3.Connection,
    *,
    ref: base.ArchiveRef,
    image_archive_id: str,
    member_name: str,
    task: str,
    reason: str,
    details: Iterable[str] = (),
    image_member: str = "",
    annotation_count: int | None = None,
    removed_bbox_count: int = 0,
    species: str | None = None,
) -> None:
    record_key = f"{ref.split}:{ref.archive_id}:{base.normalize_member_name(member_name)}"
    connection.execute(
        """
        INSERT OR REPLACE INTO excluded_records VALUES(
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            record_key,
            reason,
            compact_json(sorted(set(details))),
            ref.split,
            ref.archive_id,
            image_archive_id,
            species or ref.species,
            task,
            base.normalize_member_name(member_name),
            base.normalize_member_name(image_member),
            annotation_count,
            removed_bbox_count,
        ),
    )


def pair_archives(
    refs: Sequence[base.ArchiveRef],
) -> list[tuple[base.ArchiveRef, base.ArchiveRef]]:
    lookup = {(ref.split, ref.kind, ref.index): ref for ref in refs}
    pairs: list[tuple[base.ArchiveRef, base.ArchiveRef]] = []
    for label in sorted(
        (ref for ref in refs if ref.kind == "label"),
        key=lambda ref: (ref.split, ref.index),
    ):
        image = lookup.get((label.split, "image", label.index))
        if image is None:
            raise ValueError(f"{label.archive_id} 대응 이미지 ZIP이 없습니다")
        if image.species != label.species:
            raise ValueError(
                f"{label.archive_id}와 {image.archive_id}의 품종이 다릅니다"
            )
        pairs.append((label, image))
    if len(pairs) != 10:
        raise ValueError(f"라벨-이미지 ZIP 쌍 {len(pairs)}개(기대값 10개)")
    return pairs


def scan_detection_records(
    pairs: Sequence[tuple[base.ArchiveRef, base.ArchiveRef]],
    database_path: Path,
    *,
    show_progress: bool = True,
    commit_interval: int = 5_000,
) -> dict[str, int]:
    """Read target label JSON and stage cleaned records in SQLite."""
    source_paths = [
        ref.path for pair in pairs for ref in pair
    ]
    snapshot = audit.source_zip_snapshot(source_paths)
    temporary = database_path.with_name(
        f".{database_path.name}.tmp-{os.getpid()}"
    )
    if temporary.exists():
        temporary.unlink()
    connection = initialize_database(temporary)
    counters: Counter[str] = Counter()
    try:
        for label_ref, image_ref in tqdm(
            pairs,
            desc="탐지 manifest 라벨 ZIP",
            unit="ZIP",
            disable=not show_progress,
        ):
            image_inspection = base.inspect_archive(image_ref)
            if image_inspection.zip_open_status != "central_directory_ok":
                raise zipfile.BadZipFile(
                    f"{image_ref.archive_id}: 이미지 ZIP 중앙 디렉터리 오류"
                )
            indexes = base.build_image_indexes(image_inspection)
            with zipfile.ZipFile(label_ref.path, "r") as label_zip:
                infos = [
                    info
                    for info in label_zip.infolist()
                    if not info.is_dir()
                    and base.extension_of(info.filename) == ".json"
                ]
                for info in tqdm(
                    infos,
                    desc=f"현재 ZIP {label_ref.archive_id}",
                    unit="JSON",
                    leave=False,
                    disable=not show_progress,
                ):
                    task = base.classify_task(info.filename)
                    counters["encountered_json"] += 1
                    if task not in TARGET_TASKS:
                        insert_excluded(
                            connection,
                            ref=label_ref,
                            image_archive_id=image_ref.archive_id,
                            member_name=info.filename,
                            task=task,
                            reason="task_excluded_culture",
                        )
                        counters["excluded_task"] += 1
                        continue
                    try:
                        raw = label_zip.read(info)
                        text, _encoding = base.decode_json_bytes(raw)
                        data = json.loads(text)
                        if not isinstance(data, dict):
                            raise ValueError("JSON root is not an object")
                        info_data = (
                            data.get("INFO")
                            if isinstance(data.get("INFO"), dict)
                            else {}
                        )
                        image_data = (
                            data.get("IMAGE")
                            if isinstance(data.get("IMAGE"), dict)
                            else {}
                        )
                        meta = (
                            data.get("META")
                            if isinstance(data.get("META"), dict)
                            else {}
                        )
                        annotations = data.get("ANNOTATION_INFO")
                        annotation_list = (
                            annotations if isinstance(annotations, list) else []
                        )
                        species = normalized(
                            info_data.get("CATEGORY_NAME"), label_ref.species
                        )
                        if species not in SPECIES_CLASS_IDS:
                            raise ValueError(f"unsupported species: {species}")
                        width = audit.integer(image_data.get("WIDTH"))
                        height = audit.integer(image_data.get("HEIGHT"))
                        cleanup = clean_annotations(
                            annotation_list, width, height
                        )
                        sample = base.JsonSample(
                            archive=label_ref,
                            member_name=info.filename,
                            task=task,
                            data=data,
                            path_values={
                                "$.IMAGE.IMAGE_FILE_NAME": [
                                    image_data.get("IMAGE_FILE_NAME")
                                ]
                            },
                        )
                        match = base.match_image(sample, indexes)
                        if match.status != "matched":
                            insert_excluded(
                                connection,
                                ref=label_ref,
                                image_archive_id=image_ref.archive_id,
                                member_name=info.filename,
                                task=task,
                                reason="image_match_failed",
                                details=(match.method, match.status),
                                annotation_count=len(annotation_list),
                                removed_bbox_count=(
                                    len(annotation_list)
                                    - len(cleanup.cleaned)
                                ),
                                species=species,
                            )
                            counters["excluded_image_match"] += 1
                            continue
                        if not cleanup.cleaned:
                            if not annotation_list:
                                reason = "no_annotations"
                            else:
                                reason = "all_bboxes_removed"
                            detail_flags = [
                                flag
                                for item in cleanup.flags
                                for flag in item["flags"]
                            ]
                            insert_excluded(
                                connection,
                                ref=label_ref,
                                image_archive_id=image_ref.archive_id,
                                member_name=info.filename,
                                task=task,
                                reason=reason,
                                details=detail_flags,
                                image_member=match.matched_member,
                                annotation_count=len(annotation_list),
                                removed_bbox_count=len(annotation_list),
                                species=species,
                            )
                            counters[f"excluded_{reason}"] += 1
                            continue
                        normality_value = meta.get(
                            "DBYHS_NORMALITY_ALTERNATIVE"
                        )
                        if normality_value is True:
                            normality = "normal"
                        elif normality_value is False:
                            normality = "abnormal"
                        else:
                            normality = "missing_or_invalid"
                        camera_id = normalized(meta.get("IP_CAMERA_ID"), "<missing>")
                        capture_date = normalized(
                            meta.get("IMAGE_CREATE_DATE"), "<missing>"
                        )
                        capture_time = normalized(
                            meta.get("IMAGE_CREATE_TIME"), "<missing>"
                        )
                        disease_type = normalized(
                            meta.get("DBYHS_SPCHCKN"), "<missing>"
                        )
                        group_tuple = (species, camera_id, capture_date)
                        group_key = compact_json(group_tuple)
                        has_segmentation = any(
                            isinstance(annotation, dict)
                            and annotation.get("SEGMENTATION") is not None
                            for annotation in annotation_list
                        )
                        record_key = (
                            f"{label_ref.split}:{label_ref.archive_id}:"
                            f"{base.normalize_member_name(info.filename)}"
                        )
                        connection.execute(
                            """
                            INSERT INTO detection_records VALUES(
                                ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                            )
                            """,
                            (
                                record_key,
                                group_key,
                                species,
                                SPECIES_CLASS_IDS[species],
                                task,
                                normality,
                                disease_type,
                                camera_id,
                                capture_date,
                                capture_time,
                                label_ref.archive_id,
                                image_ref.archive_id,
                                base.normalize_member_name(info.filename),
                                base.normalize_member_name(
                                    match.matched_member
                                ),
                                base.normalize_member_name(
                                    match.matched_member
                                ).casefold(),
                                width,
                                height,
                                len(annotation_list),
                                len(cleanup.cleaned),
                                compact_json(cleanup.original),
                                compact_json(cleanup.cleaned),
                                compact_json(cleanup.flags),
                                int(has_segmentation),
                                label_ref.split,
                                cleanup.counts["removed_nonpositive_bbox"],
                                cleanup.counts["removed_after_clip_lt_1px"],
                                cleanup.counts["removed_incomplete_bbox"],
                                cleanup.counts["clipped_to_image_bounds"],
                                cleanup.counts["retained_small_bbox"],
                            ),
                        )
                        counters["included"] += 1
                        counters.update(cleanup.counts)
                    except (
                        OSError,
                        RuntimeError,
                        UnicodeError,
                        json.JSONDecodeError,
                        ValueError,
                        zipfile.BadZipFile,
                    ) as exc:
                        insert_excluded(
                            connection,
                            ref=label_ref,
                            image_archive_id=image_ref.archive_id,
                            member_name=info.filename,
                            task=task,
                            reason="json_or_schema_error",
                            details=(type(exc).__name__,),
                        )
                        counters["excluded_json_or_schema_error"] += 1
                    if counters["encountered_json"] % commit_interval == 0:
                        connection.commit()
            connection.commit()
        connection.executescript(
            """
            CREATE INDEX idx_detection_group ON detection_records(group_key);
            CREATE INDEX idx_detection_image ON detection_records(image_member_key);
            CREATE INDEX idx_detection_distribution
                ON detection_records(species, task, normality, disease_type);
            """
        )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES('scan_complete', 'true')"
        )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES('source_zip_modified', 'false')"
        )
        connection.commit()
    finally:
        connection.close()
        audit.assert_source_zips_unchanged(snapshot)
    os.replace(temporary, database_path)
    return dict(counters)


def enrich_known_empty_annotations(
    database_path: Path,
    quality_database_path: Path = DEFAULT_QUALITY_DATABASE,
) -> int:
    """Mark the four known empty culture JSONs without rereading their bytes."""
    if not quality_database_path.is_file():
        return 0
    quality = audit.sql_connection(quality_database_path, read_only=True)
    target = audit.sql_connection(database_path)
    updated = 0
    try:
        rows = quality.execute(
            """
            SELECT official_split, archive_id, json_member
            FROM quality_records
            WHERE annotation_count = 0
            """
        )
        for row in rows:
            record_key = (
                f"{row['official_split']}:{row['archive_id']}:"
                f"{base.normalize_member_name(row['json_member'])}"
            )
            cursor = target.execute(
                """
                UPDATE excluded_records
                SET detail_flags=?,
                    original_annotation_count=0
                WHERE record_key=? AND exclusion_reason='task_excluded_culture'
                """,
                (
                    compact_json(
                        ("no_annotations", "task_excluded_culture")
                    ),
                    record_key,
                ),
            )
            updated += cursor.rowcount
        target.execute(
            "INSERT OR REPLACE INTO metadata(key, value) "
            "VALUES('known_empty_annotation_json', ?)",
            (str(updated),),
        )
        target.commit()
    finally:
        quality.close()
        target.close()
    return updated


def load_split_groups(
    connection: sqlite3.Connection,
) -> dict[tuple[str, ...], audit.SplitGroup]:
    groups: dict[tuple[str, ...], audit.SplitGroup] = {}
    rows = connection.execute(
        """
        SELECT group_key, species, task, normality, disease_type,
               official_split, COUNT(*) AS count
        FROM detection_records
        GROUP BY group_key, species, task, normality, disease_type,
                 official_split
        ORDER BY group_key
        """
    )
    for row in rows:
        decoded = json.loads(row["group_key"])
        key = tuple(str(value) for value in decoded)
        group = groups.setdefault(key, audit.SplitGroup(key=key))
        count = int(row["count"])
        group.total += count
        group.official_counts[row["official_split"]] += count
        group.features[f"species:{row['species']}"] += count
        group.features[f"task:{row['task']}"] += count
        group.features[f"normality:{row['normality']}"] += count
        if row["disease_type"] != "<missing>":
            group.features[f"disease:{row['disease_type']}"] += count
    return groups


def assign_splits(
    connection: sqlite3.Connection,
    *,
    seed: int,
) -> dict[str, Counter[str]]:
    groups = load_split_groups(connection)
    assignment, assigned_counts = audit.deterministic_group_assignment(
        groups,
        seed=seed,
        ratios=audit.SPLIT_RATIOS,
        stratify_by_first_key=True,
    )
    connection.executemany(
        "UPDATE detection_records SET new_split=? WHERE group_key=?",
        (
            (split, compact_json(key))
            for key, split in sorted(assignment.items())
        ),
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('seed', ?)",
        (str(seed),),
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) "
        "VALUES('group_key_fields', 'species,camera_id,capture_date')"
    )
    connection.commit()
    return assigned_counts


def count_query(
    connection: sqlite3.Connection, sql: str, parameters: Sequence[Any] = ()
) -> int:
    return int(connection.execute(sql, parameters).fetchone()[0])


def validate_database(connection: sqlite3.Connection) -> dict[str, int]:
    result = {
        "records_without_split": count_query(
            connection,
            "SELECT COUNT(*) FROM detection_records WHERE new_split IS NULL",
        ),
        "group_split_leakage": count_query(
            connection,
            """
            SELECT COUNT(*) FROM (
                SELECT group_key
                FROM detection_records
                GROUP BY group_key
                HAVING COUNT(DISTINCT new_split) > 1
            )
            """,
        ),
        "image_split_leakage": count_query(
            connection,
            """
            SELECT COUNT(*) FROM (
                SELECT image_member_key
                FROM detection_records
                GROUP BY image_member_key
                HAVING COUNT(DISTINCT new_split) > 1
            )
            """,
        ),
    }
    invalid_bounds = 0
    invalid_size = 0
    valid_boxes = 0
    for row in connection.execute(
        """
        SELECT image_width, image_height, valid_bbox_count, bbox_cleaned
        FROM detection_records
        """
    ):
        boxes = json.loads(row["bbox_cleaned"])
        if len(boxes) != row["valid_bbox_count"]:
            raise RuntimeError("valid_bbox_count와 bbox_cleaned 길이가 다릅니다")
        for box in boxes:
            valid_boxes += 1
            x = float(box["x"])
            y = float(box["y"])
            width = float(box["width"])
            height = float(box["height"])
            if width <= 0 or height <= 0:
                invalid_size += 1
            if (
                x < 0
                or y < 0
                or x + width > row["image_width"]
                or y + height > row["image_height"]
            ):
                invalid_bounds += 1
    result["valid_bbox"] = valid_boxes
    result["invalid_bbox_size"] = invalid_size
    result["invalid_bbox_bounds"] = invalid_bounds
    if any(
        result[key]
        for key in (
            "records_without_split",
            "group_split_leakage",
            "image_split_leakage",
            "invalid_bbox_size",
            "invalid_bbox_bounds",
        )
    ):
        raise RuntimeError(f"manifest 검증 실패: {result}")
    return result


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    os.replace(temporary, path)


def manifest_rows(
    connection: sqlite3.Connection,
) -> Iterable[dict[str, Any]]:
    columns_sql = ", ".join(
        (
            "new_split AS split",
            *MANIFEST_COLUMNS[1:],
        )
    )
    for row in connection.execute(
        f"""
        SELECT {columns_sql}
        FROM detection_records
        ORDER BY
            CASE new_split
                WHEN 'train' THEN 0
                WHEN 'validation' THEN 1
                ELSE 2
            END,
            species, label_archive_id, json_member
        """
    ):
        output = dict(row)
        output["has_segmentation"] = bool(output["has_segmentation"])
        yield output


def excluded_rows(
    connection: sqlite3.Connection,
) -> Iterable[dict[str, Any]]:
    columns = (
        "exclusion_reason",
        "detail_flags",
        "official_split",
        "label_archive_id",
        "image_archive_id",
        "species",
        "task",
        "json_member",
        "image_member",
        "original_annotation_count",
        "removed_bbox_count",
    )
    for row in connection.execute(
        f"""
        SELECT {", ".join(columns)}
        FROM excluded_records
        ORDER BY exclusion_reason, official_split, label_archive_id, json_member
        """
    ):
        yield dict(row)


def distribution_rows(
    connection: sqlite3.Connection,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes = (
        ("species", "species", ""),
        ("task", "task", ""),
        ("normality", "normality", ""),
        (
            "disease_type",
            "disease_type",
            "WHERE task='병해' AND disease_type <> '<missing>'",
        ),
    )
    for scope, column, where in scopes:
        total = count_query(
            connection,
            f"SELECT COUNT(*) FROM detection_records {where}",
        )
        split_totals = {
            row["new_split"]: int(row["count"])
            for row in connection.execute(
                f"""
                SELECT new_split, COUNT(*) AS count
                FROM detection_records {where}
                GROUP BY new_split
                """
            )
        }
        overall = {
            row["value"]: int(row["count"])
            for row in connection.execute(
                f"""
                SELECT {column} AS value, COUNT(*) AS count
                FROM detection_records {where} GROUP BY {column}
                """
            )
        }
        by_split = {
            (row["new_split"], row["value"]): int(row["count"])
            for row in connection.execute(
                f"""
                SELECT new_split, {column} AS value, COUNT(*) AS count
                FROM detection_records {where}
                GROUP BY new_split, {column}
                """
            )
        }
        for value in sorted(overall):
            overall_count = overall[value]
            overall_rate = overall_count / total if total else 0.0
            for split in SPLITS:
                count = by_split.get((split, value), 0)
                split_total = split_totals.get(split, 0)
                split_rate = count / split_total if split_total else 0.0
                rows.append(
                    {
                        "scope": scope,
                        "value": value,
                        "split": split,
                        "count": count,
                        "split_total": split_total,
                        "rate_percent": f"{split_rate * 100:.6f}",
                        "overall_count": overall_count,
                        "overall_rate_percent": f"{overall_rate * 100:.6f}",
                        "absolute_deviation_pp": (
                            f"{abs(split_rate - overall_rate) * 100:.6f}"
                        ),
                    }
                )
    return rows


def cleanup_summary_rows(
    connection: sqlite3.Connection,
) -> list[dict[str, Any]]:
    included = connection.execute(
        """
        SELECT
            SUM(original_annotation_count) AS original_count,
            SUM(valid_bbox_count) AS valid_count,
            SUM(removed_nonpositive_count) AS nonpositive,
            SUM(removed_after_clip_count) AS after_clip,
            SUM(removed_incomplete_count) AS incomplete,
            SUM(clipped_count) AS clipped,
            SUM(retained_small_count) AS small
        FROM detection_records
        """
    ).fetchone()
    excluded_all_removed = connection.execute(
        """
        SELECT COALESCE(SUM(removed_bbox_count), 0)
        FROM excluded_records
        WHERE exclusion_reason='all_bboxes_removed'
        """
    ).fetchone()[0]
    excluded_empty = connection.execute(
        """
        SELECT COUNT(*)
        FROM excluded_records
        WHERE detail_flags LIKE '%no_annotations%'
        """
    ).fetchone()[0]
    rows = [
        (
            "bbox",
            "original_bbox_in_included_json",
            "bbox",
            included["original_count"] or 0,
            "최종 포함 JSON에 원래 기록된 annotation",
        ),
        (
            "bbox",
            "valid_bbox",
            "bbox",
            included["valid_count"] or 0,
            "정제 후 최종 유효 bbox",
        ),
        (
            "cleanup",
            "removed_nonpositive_bbox",
            "bbox",
            included["nonpositive"] or 0,
            "width 또는 height가 0 이하",
        ),
        (
            "cleanup",
            "removed_after_clip_lt_1px",
            "bbox",
            included["after_clip"] or 0,
            "clip 후 width 또는 height가 1px 미만",
        ),
        (
            "cleanup",
            "removed_incomplete_bbox",
            "bbox",
            included["incomplete"] or 0,
            "bbox 좌표 필드 누락 또는 비수치",
        ),
        (
            "cleanup",
            "all_bboxes_removed_excluded_json_bbox",
            "bbox",
            excluded_all_removed,
            "모든 bbox가 제거되어 제외된 JSON의 bbox",
        ),
        (
            "exclusion",
            "no_annotations_json",
            "json",
            excluded_empty,
            "배양 제외 28,710개에 포함된 annotation 없는 JSON",
        ),
        (
            "cleanup",
            "clipped_to_image_bounds",
            "bbox",
            included["clipped"] or 0,
            "이미지 경계로 clip된 뒤 유지",
        ),
        (
            "policy",
            "retained_small_bbox",
            "bbox",
            included["small"] or 0,
            "면적 비율 0.1% 미만이나 일괄 제거하지 않음",
        ),
    ]
    return [
        {
            "category": category,
            "reason": reason,
            "unit": unit,
            "count": count,
            "notes": notes,
        }
        for category, reason, unit, count, notes in rows
    ]


def maximum_deviations(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    result: dict[str, float] = defaultdict(float)
    for row in rows:
        result[str(row["scope"])] = max(
            result[str(row["scope"])],
            float(row["absolute_deviation_pp"]),
        )
    return dict(result)


def markdown_table(
    headers: Sequence[str], rows: Sequence[Sequence[Any]]
) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_summary(
    connection: sqlite3.Connection,
    output_path: Path,
    *,
    seed: int,
    validation: Mapping[str, int],
    distribution: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total = count_query(connection, "SELECT COUNT(*) FROM detection_records")
    valid_bbox = count_query(
        connection,
        "SELECT COALESCE(SUM(valid_bbox_count), 0) FROM detection_records",
    )
    excluded = count_query(connection, "SELECT COUNT(*) FROM excluded_records")
    splits = [
        dict(row)
        for row in connection.execute(
            """
            SELECT new_split AS split, COUNT(*) AS count
            FROM detection_records
            GROUP BY new_split
            ORDER BY CASE new_split
                WHEN 'train' THEN 0 WHEN 'validation' THEN 1 ELSE 2 END
            """
        )
    ]
    excluded_reasons = [
        dict(row)
        for row in connection.execute(
            """
            SELECT exclusion_reason, COUNT(*) AS count,
                   COALESCE(SUM(removed_bbox_count), 0) AS removed_bbox_count
            FROM excluded_records
            GROUP BY exclusion_reason
            ORDER BY exclusion_reason
            """
        )
    ]
    empty_annotation_json = count_query(
        connection,
        "SELECT COUNT(*) FROM excluded_records "
        "WHERE detail_flags LIKE '%no_annotations%'",
    )
    official_changes = count_query(
        connection,
        "SELECT COUNT(*) FROM detection_records "
        "WHERE new_split <> official_split",
    )
    official_matrix = [
        dict(row)
        for row in connection.execute(
            """
            SELECT official_split, new_split, COUNT(*) AS count
            FROM detection_records
            GROUP BY official_split, new_split
            ORDER BY official_split, new_split
            """
        )
    ]
    deviations = maximum_deviations(distribution)
    split_rows = [
        (
            row["split"],
            f"{row['count']:,}",
            f"{row['count'] / total * 100:.3f}%",
        )
        for row in splits
    ]
    reason_rows = [
        (
            row["exclusion_reason"],
            f"{row['count']:,}",
            f"{row['removed_bbox_count']:,}",
        )
        for row in excluded_reasons
    ]
    official_rows = [
        (
            row["official_split"],
            row["new_split"],
            f"{row['count']:,}",
        )
        for row in official_matrix
    ]
    lines = [
        "# 탐지 데이터 manifest 검증 요약",
        "",
        "## 범위와 고정 정책",
        "",
        "- 대상: 생육·병해 JSON, 5품종(느타리·양송이·큰느타리·팽이·표고)",
        "- 제외: 배양 JSON 및 정제 후 유효 bbox가 없는 JSON",
        "- 클래스: 품종 5개 객체 탐지(class_id 0~4)",
        "- 그룹 키: `species + camera_id + capture_date`",
        f"- 고정 seed: `{seed}`",
        "- 목표 비율: Train 70% / Validation 15% / Test 15%; 그룹 누수 0을 우선",
        "- Test는 고정된 최종 평가 세트이며 학습·모델 선택에 사용하지 않음",
        "",
        "## 최종 규모",
        "",
        f"- 포함 이미지(JSON): **{total:,}개**",
        f"- 제외 JSON: **{excluded:,}개**",
        f"- 최종 유효 bbox: **{valid_bbox:,}개**",
        "",
        markdown_table(("split", "이미지 수", "비율"), split_rows),
        "",
        "## 제외 사유",
        "",
        markdown_table(
            ("사유", "JSON 수", "제거 bbox 수"),
            reason_rows,
        ),
        "",
        f"- annotation이 없는 JSON **{empty_annotation_json:,}개**는 모두 "
        "`task_excluded_culture`에 포함된 배양 데이터로, 첫 탐지 후보에서 제외됨",
        "",
        "작은 bbox는 자동 제거하지 않았고, annotation이 개체인지 군집인지 "
        "자동으로 해석하지 않았다. 원본 annotation 수와 유효 bbox 수를 "
        "manifest에 함께 기록했다.",
        "",
        "## 검증 결과",
        "",
        f"- 둘 이상의 split에 걸친 group_key: **{validation['group_split_leakage']:,}개**",
        f"- 둘 이상의 split에 걸친 image_member: **{validation['image_split_leakage']:,}개**",
        f"- 범위 밖 정제 bbox: **{validation['invalid_bbox_bounds']:,}개**",
        f"- width/height가 양수가 아닌 정제 bbox: **{validation['invalid_bbox_size']:,}개**",
        f"- split 미할당 레코드: **{validation['records_without_split']:,}개**",
        "",
        "전체 분포 대비 split별 최대 절대 편차:",
        "",
        markdown_table(
            ("분포", "최대 절대 편차(pp)"),
            [
                ("품종", f"{deviations.get('species', 0.0):.4f}"),
                ("작업", f"{deviations.get('task', 0.0):.4f}"),
                ("정상 여부", f"{deviations.get('normality', 0.0):.4f}"),
                ("병해 종류", f"{deviations.get('disease_type', 0.0):.4f}"),
            ],
        ),
        "",
        "## 공식 split 변경",
        "",
        f"- 최종 포함 레코드 중 공식 Training/Validation과 새 split이 다른 "
        f"레코드: **{official_changes:,}개 ({official_changes / total * 100:.3f}%)**",
        "",
        markdown_table(
            ("공식 split", "새 split", "이미지 수"),
            official_rows,
        ),
        "",
        "공식 Validation을 최종 평가셋으로 확정하지 않고, 동일 그룹을 한 "
        "split에만 배치해 새 Test를 별도로 고정했다.",
        "",
        "## 안전 범위",
        "",
        "- 라벨 ZIP의 대상 JSON만 순차적으로 읽음",
        "- 이미지 ZIP은 중앙 디렉터리의 파일명만 조회하고 이미지 바이트를 읽지 않음",
        "- 이미지 디코딩·추출 및 전체 ZIP 압축 해제를 하지 않음",
        "- ZIP 전체 CRC 검사를 하지 않음",
        "- 원본 ZIP을 수정·이동·삭제·이름 변경하지 않음",
        "- YOLO 변환 및 모델 학습을 하지 않음",
        "- 출력에는 ZIP 내부 상대경로와 archive ID만 기록하며 로컬 절대경로를 기록하지 않음",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.tmp-{os.getpid()}"
    )
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, output_path)
    return {
        "included": total,
        "excluded": excluded,
        "valid_bbox": valid_bbox,
        "splits": {row["split"]: row["count"] for row in splits},
        "deviations": deviations,
        "official_changes": official_changes,
        "excluded_reasons": {
            row["exclusion_reason"]: row["count"]
            for row in excluded_reasons
        },
    }


def write_reports(
    database_path: Path,
    output_dir: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    connection = audit.sql_connection(database_path)
    try:
        assigned_counts = assign_splits(connection, seed=seed)
        validation = validate_database(connection)
        distribution = distribution_rows(connection)
        write_csv(
            output_dir / "detection_dataset_manifest.csv",
            MANIFEST_COLUMNS,
            manifest_rows(connection),
        )
        write_csv(
            output_dir / "detection_split_distribution.csv",
            (
                "scope",
                "value",
                "split",
                "count",
                "split_total",
                "rate_percent",
                "overall_count",
                "overall_rate_percent",
                "absolute_deviation_pp",
            ),
            distribution,
        )
        write_csv(
            output_dir / "detection_excluded_records.csv",
            (
                "exclusion_reason",
                "detail_flags",
                "official_split",
                "label_archive_id",
                "image_archive_id",
                "species",
                "task",
                "json_member",
                "image_member",
                "original_annotation_count",
                "removed_bbox_count",
            ),
            excluded_rows(connection),
        )
        write_csv(
            output_dir / "detection_bbox_cleanup_summary.csv",
            ("category", "reason", "unit", "count", "notes"),
            cleanup_summary_rows(connection),
        )
        summary = write_summary(
            connection,
            output_dir / "detection_manifest_summary.md",
            seed=seed,
            validation=validation,
            distribution=distribution,
        )
        summary["validation"] = validation
        summary["assigned_counts"] = {
            split: dict(counts)
            for split, counts in assigned_counts.items()
        }
        return summary
    finally:
        connection.close()


def run(
    args: argparse.Namespace,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    refs, issues, _directories = base.discover_archives(environ)
    if issues:
        raise RuntimeError(
            "아카이브 환경/구조 검증 실패:\n- " + "\n- ".join(issues)
        )
    pairs = pair_archives(refs)
    scan_result = scan_detection_records(
        pairs,
        args.database,
        show_progress=not args.no_progress,
    )
    scan_result["known_empty_annotations"] = enrich_known_empty_annotations(
        args.database
    )
    report_result = write_reports(
        args.database,
        args.output_dir,
        seed=args.seed,
    )
    return {
        "scan": scan_result,
        "reports": report_result,
        "database": base.portable_output_path(args.database),
        "output_dir": base.portable_output_path(args.output_dir),
        "image_bytes_read": False,
        "image_decoded": False,
        "source_zip_modified": False,
        "yolo_conversion": False,
        "model_training": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args, os.environ)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
