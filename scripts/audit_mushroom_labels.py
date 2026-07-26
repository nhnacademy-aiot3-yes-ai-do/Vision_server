#!/usr/bin/env python3
"""Full label-quality, limited visual-audit, and split-strategy analysis.

This script never modifies source ZIPs, never extracts archives wholesale, and
never trains or converts a model.  It reads all label JSON only when explicitly
requested and reads image bytes only for the bounded visual-audit manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sqlite3
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

import inspect_aihub_archives as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / "artifacts" / "quality_analysis.sqlite3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports"
DEFAULT_VISUAL_DIR = PROJECT_ROOT / "artifacts" / "visual_audit"
DEFAULT_MANUAL_REVIEW = (
    PROJECT_ROOT / "configs" / "visual_audit_review.json"
)
QUALITY_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260726
MAX_VISUAL_PER_COMBINATION = 20
TINY_BBOX_AREA_FRACTION = 0.001
NEAR_FULL_BBOX_AREA_FRACTION = 0.90
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SPLIT_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "species_camera_date_time",
        ("species", "camera_id", "capture_date", "capture_time"),
    ),
    ("species_camera_date", ("species", "camera_id", "capture_date")),
    ("species_camera", ("species", "camera_id")),
    ("species_date", ("species", "capture_date")),
)


@dataclass
class QualityRecord:
    official_split: str
    archive_id: str
    archive_index: int
    species: str
    task: str
    normality: str
    disease_type: str | None
    camera_id: str | None
    capture_date: str | None
    capture_time: str | None
    image_filename: str
    json_member: str
    image_width: int | None
    image_height: int | None
    annotation_count: int
    bbox_count: int
    segmentation_count: int
    bbox_missing_reason: str
    bbox_nonpositive_count: int
    bbox_out_of_bounds_count: int
    bbox_tiny_count: int
    bbox_near_full_count: int
    segmentation_out_of_bounds_count: int
    segmentation_invalid_count: int

    def as_sql_tuple(self) -> tuple[Any, ...]:
        return tuple(getattr(self, field) for field in QUALITY_COLUMNS)


@dataclass
class SplitGroup:
    key: tuple[str, ...]
    total: int = 0
    official_counts: Counter[str] = field(default_factory=Counter)
    features: Counter[str] = field(default_factory=Counter)


QUALITY_COLUMNS = (
    "official_split",
    "archive_id",
    "archive_index",
    "species",
    "task",
    "normality",
    "disease_type",
    "camera_id",
    "capture_date",
    "capture_time",
    "image_filename",
    "json_member",
    "image_width",
    "image_height",
    "annotation_count",
    "bbox_count",
    "segmentation_count",
    "bbox_missing_reason",
    "bbox_nonpositive_count",
    "bbox_out_of_bounds_count",
    "bbox_tiny_count",
    "bbox_near_full_count",
    "segmentation_out_of_bounds_count",
    "segmentation_invalid_count",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIHub 버섯 전체 라벨 품질, 제한 시각 감사, 그룹 분할 후보를 "
            "분석합니다."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scan-metadata",
        action="store_true",
        help="라벨 ZIP의 전체 JSON 메타데이터를 품질 감사 DB에 집계",
    )
    parser.add_argument(
        "--visual-audit",
        action="store_true",
        help="선정된 제한 샘플 이미지만 추출하고 overlay 생성",
    )
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="완료된 품질 DB와 기존 manifest로 보고서만 재생성",
    )
    parser.add_argument(
        "--max-visual-per-combination",
        type=bounded_visual_count,
        default=MAX_VISUAL_PER_COMBINATION,
        metavar="N",
        help="품종×작업 조합별 시각 감사 최대 이미지 수(1~20)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("artifacts/quality_analysis.sqlite3"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
    )
    parser.add_argument(
        "--visual-dir",
        type=Path,
        default=Path("artifacts/visual_audit"),
    )
    args = parser.parse_args(argv)
    if not (args.scan_metadata or args.visual_audit or args.reports_only):
        parser.error(
            "--scan-metadata, --visual-audit, --reports-only 중 하나 이상이 필요합니다"
        )
    args.database = project_path(args.database)
    args.output_dir = project_path(args.output_dir)
    args.visual_dir = project_path(args.visual_dir)
    return args


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def bounded_visual_count(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("정수여야 합니다") from exc
    if not 1 <= parsed <= MAX_VISUAL_PER_COMBINATION:
        raise argparse.ArgumentTypeError("1 이상 20 이하여야 합니다")
    return parsed


def sql_connection(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def stable_hash(seed: int, *parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(f"{seed}\x1e{payload}".encode("utf-8")).hexdigest()


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def integer(value: Any) -> int | None:
    number = numeric(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def normalized_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    return None


def segmentation_polygons(value: Any) -> list[list[tuple[float, float]]]:
    """Normalize flat or nested segmentation arrays into polygon point lists."""
    if value is None:
        return []
    if isinstance(value, list) and value and all(
        numeric(item) is not None for item in value
    ):
        if len(value) < 6 or len(value) % 2:
            return []
        points = [
            (float(value[index]), float(value[index + 1]))
            for index in range(0, len(value), 2)
        ]
        return [points]
    polygons: list[list[tuple[float, float]]] = []
    if isinstance(value, list):
        for child in value:
            polygons.extend(segmentation_polygons(child))
    return polygons


def bbox_values(annotation: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(annotation, dict):
        return None
    values = (
        numeric(annotation.get("BOUNDING_BOX_X_COORDINATE")),
        numeric(annotation.get("BOUNDING_BOX_Y_COORDINATE")),
        numeric(annotation.get("BOUNDING_BOX_WIDTH")),
        numeric(annotation.get("BOUNDING_BOX_HEIGHT")),
    )
    if any(value is None for value in values):
        return None
    return values  # type: ignore[return-value]


def validate_bbox(
    bbox: tuple[float, float, float, float],
    width: int | None,
    height: int | None,
) -> set[str]:
    x, y, box_width, box_height = bbox
    issues: set[str] = set()
    if box_width <= 0 or box_height <= 0:
        issues.add("bbox_nonpositive")
    if width and height:
        if x < 0 or y < 0 or x + box_width > width or y + box_height > height:
            issues.add("bbox_out_of_bounds")
        if box_width > 0 and box_height > 0:
            area_fraction = (box_width * box_height) / (width * height)
            if area_fraction < TINY_BBOX_AREA_FRACTION:
                issues.add("bbox_tiny")
            if area_fraction >= NEAR_FULL_BBOX_AREA_FRACTION:
                issues.add("bbox_near_full")
    return issues


def validate_segmentation(
    segmentation: Any,
    width: int | None,
    height: int | None,
) -> tuple[int, int]:
    if segmentation is None:
        return 0, 0
    polygons = segmentation_polygons(segmentation)
    if not polygons:
        return 0, 1
    out_of_bounds = 0
    if width and height:
        for polygon in polygons:
            if any(
                x < 0 or y < 0 or x > width or y > height
                for x, y in polygon
            ):
                out_of_bounds += 1
    return out_of_bounds, 0


def quality_record(
    ref: base.ArchiveRef,
    member_name: str,
    data: Any,
) -> QualityRecord:
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    info = data.get("INFO") if isinstance(data.get("INFO"), dict) else {}
    image = data.get("IMAGE") if isinstance(data.get("IMAGE"), dict) else {}
    meta = data.get("META") if isinstance(data.get("META"), dict) else {}
    annotations_value = data.get("ANNOTATION_INFO")
    annotations = annotations_value if isinstance(annotations_value, list) else []
    width = integer(image.get("WIDTH"))
    height = integer(image.get("HEIGHT"))
    bbox_count = 0
    segmentation_count = 0
    bbox_issues: Counter[str] = Counter()
    segmentation_out = 0
    segmentation_invalid = 0
    for annotation in annotations:
        bbox = bbox_values(annotation)
        if bbox is not None:
            bbox_count += 1
            for issue in validate_bbox(bbox, width, height):
                bbox_issues[issue] += 1
        segmentation = (
            annotation.get("SEGMENTATION")
            if isinstance(annotation, dict)
            else None
        )
        if segmentation is not None:
            segmentation_count += 1
        out_count, invalid_count = validate_segmentation(
            segmentation, width, height
        )
        segmentation_out += out_count
        segmentation_invalid += invalid_count
    if not annotations:
        bbox_missing_reason = "no_annotations"
    elif bbox_count == 0:
        bbox_missing_reason = "no_complete_bbox"
    else:
        bbox_missing_reason = ""
    normality_value = meta.get("DBYHS_NORMALITY_ALTERNATIVE")
    if normality_value is True:
        normality = "normal"
    elif normality_value is False:
        normality = "abnormal"
    else:
        normality = "missing_or_invalid"
    return QualityRecord(
        official_split=ref.split,
        archive_id=ref.archive_id,
        archive_index=ref.index,
        species=normalized_scalar(info.get("CATEGORY_NAME")) or ref.species,
        task=base.classify_task(member_name),
        normality=normality,
        disease_type=normalized_scalar(meta.get("DBYHS_SPCHCKN")),
        camera_id=normalized_scalar(meta.get("IP_CAMERA_ID")),
        capture_date=normalized_scalar(meta.get("IMAGE_CREATE_DATE")),
        capture_time=normalized_scalar(meta.get("IMAGE_CREATE_TIME")),
        image_filename=normalized_scalar(image.get("IMAGE_FILE_NAME")) or "",
        json_member=base.normalize_member_name(member_name),
        image_width=width,
        image_height=height,
        annotation_count=len(annotations),
        bbox_count=bbox_count,
        segmentation_count=segmentation_count,
        bbox_missing_reason=bbox_missing_reason,
        bbox_nonpositive_count=bbox_issues["bbox_nonpositive"],
        bbox_out_of_bounds_count=bbox_issues["bbox_out_of_bounds"],
        bbox_tiny_count=bbox_issues["bbox_tiny"],
        bbox_near_full_count=bbox_issues["bbox_near_full"],
        segmentation_out_of_bounds_count=segmentation_out,
        segmentation_invalid_count=segmentation_invalid,
    )


def source_zip_snapshot(paths: Iterable[Path]) -> dict[Path, tuple[int, int]]:
    return {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in paths
    }


def assert_source_zips_unchanged(
    snapshot: Mapping[Path, tuple[int, int]]
) -> None:
    changed = [
        path.name
        for path, before in snapshot.items()
        if not path.exists()
        or (path.stat().st_size, path.stat().st_mtime_ns) != before
    ]
    if changed:
        raise RuntimeError(
            "원본 ZIP 변경 감지: " + ", ".join(sorted(changed))
        )


def label_scan_digest(label_refs: Sequence[base.ArchiveRef]) -> str:
    digest = hashlib.sha256(f"quality-schema:{QUALITY_SCHEMA_VERSION}".encode())
    for ref in sorted(label_refs, key=lambda item: (item.split, item.index)):
        stat = ref.path.stat()
        digest.update(
            f"{ref.split}:{ref.archive_id}:{ref.path.name}:"
            f"{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
        )
    return digest.hexdigest()


def initialize_quality_database(
    path: Path,
    settings_digest: str,
) -> sqlite3.Connection:
    connection = sql_connection(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS quality_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    existing = connection.execute(
        "SELECT value FROM quality_metadata WHERE key='settings_digest'"
    ).fetchone()
    if existing and existing["value"] != settings_digest:
        connection.close()
        raise ValueError(
            "기존 품질 감사 DB의 데이터 설정이 현재 라벨 ZIP과 다릅니다"
        )
    connection.execute(
        "INSERT OR REPLACE INTO quality_metadata(key, value) VALUES(?, ?)",
        ("settings_digest", settings_digest),
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS quality_records (
            official_split TEXT NOT NULL,
            archive_id TEXT NOT NULL,
            archive_index INTEGER NOT NULL,
            species TEXT NOT NULL,
            task TEXT NOT NULL,
            normality TEXT NOT NULL,
            disease_type TEXT,
            camera_id TEXT,
            capture_date TEXT,
            capture_time TEXT,
            image_filename TEXT NOT NULL,
            json_member TEXT NOT NULL,
            image_width INTEGER,
            image_height INTEGER,
            annotation_count INTEGER NOT NULL,
            bbox_count INTEGER NOT NULL,
            segmentation_count INTEGER NOT NULL,
            bbox_missing_reason TEXT NOT NULL,
            bbox_nonpositive_count INTEGER NOT NULL,
            bbox_out_of_bounds_count INTEGER NOT NULL,
            bbox_tiny_count INTEGER NOT NULL,
            bbox_near_full_count INTEGER NOT NULL,
            segmentation_out_of_bounds_count INTEGER NOT NULL,
            segmentation_invalid_count INTEGER NOT NULL,
            PRIMARY KEY(official_split, archive_id, json_member)
        )
        """
    )
    connection.commit()
    return connection


def quality_scan_complete(
    connection: sqlite3.Connection,
    expected_json: int,
) -> bool:
    row = connection.execute(
        "SELECT value FROM quality_metadata WHERE key='scan_complete'"
    ).fetchone()
    count = connection.execute(
        "SELECT COUNT(*) AS count FROM quality_records"
    ).fetchone()["count"]
    return bool(row and row["value"] == "true" and count == expected_json)


def scan_label_quality_metadata(
    label_refs: Sequence[base.ArchiveRef],
    database_path: Path,
    *,
    show_progress: bool = True,
    commit_interval: int = 5_000,
) -> dict[str, int]:
    if any(ref.kind != "label" for ref in label_refs):
        raise ValueError("라벨 ZIP만 품질 메타데이터 스캔에 사용할 수 있습니다")
    ordered = sorted(label_refs, key=lambda item: (item.split, item.index))
    snapshot = source_zip_snapshot(ref.path for ref in ordered)
    settings_digest = label_scan_digest(ordered)
    total_json = 0
    json_counts: dict[Path, int] = {}
    for ref in ordered:
        with zipfile.ZipFile(ref.path, "r") as archive:
            count = sum(
                1
                for info in archive.infolist()
                if not info.is_dir()
                and base.extension_of(info.filename) == ".json"
            )
        json_counts[ref.path] = count
        total_json += count
    connection = initialize_quality_database(database_path, settings_digest)
    if quality_scan_complete(connection, total_json):
        connection.close()
        assert_source_zips_unchanged(snapshot)
        return {"processed": total_json, "failed": 0, "skipped_complete": 1}

    placeholders = ", ".join("?" for _ in QUALITY_COLUMNS)
    columns_sql = ", ".join(QUALITY_COLUMNS)
    statement = (
        f"INSERT OR REPLACE INTO quality_records({columns_sql}) "
        f"VALUES({placeholders})"
    )
    processed = 0
    failed = 0
    total_bar = tqdm(
        total=total_json,
        desc="품질 메타데이터 전체 JSON",
        unit="JSON",
        disable=not show_progress,
    )
    try:
        for ref in tqdm(
            ordered,
            desc="품질 메타데이터 라벨 ZIP",
            unit="ZIP",
            disable=not show_progress,
        ):
            with zipfile.ZipFile(ref.path, "r") as archive:
                infos = [
                    info
                    for info in archive.infolist()
                    if not info.is_dir()
                    and base.extension_of(info.filename) == ".json"
                ]
                for info in tqdm(
                    infos,
                    desc=f"현재 ZIP {ref.path.name}",
                    unit="JSON",
                    leave=False,
                    disable=not show_progress,
                ):
                    try:
                        raw = archive.read(info)
                        text, _encoding = base.decode_json_bytes(raw)
                        data = json.loads(text)
                        record = quality_record(ref, info.filename, data)
                    except (
                        OSError,
                        RuntimeError,
                        UnicodeError,
                        json.JSONDecodeError,
                        ValueError,
                        zipfile.BadZipFile,
                    ):
                        failed += 1
                    else:
                        connection.execute(statement, record.as_sql_tuple())
                        processed += 1
                    total_bar.update(1)
                    if (processed + failed) % commit_interval == 0:
                        connection.commit()
        connection.execute(
            "INSERT OR REPLACE INTO quality_metadata(key, value) VALUES(?, ?)",
            ("scan_complete", "true" if failed == 0 else "false"),
        )
        connection.execute(
            "INSERT OR REPLACE INTO quality_metadata(key, value) VALUES(?, ?)",
            ("processed_json", str(processed)),
        )
        connection.execute(
            "INSERT OR REPLACE INTO quality_metadata(key, value) VALUES(?, ?)",
            ("failed_json", str(failed)),
        )
        connection.commit()
        create_quality_indexes(connection)
    finally:
        total_bar.close()
        connection.close()
        assert_source_zips_unchanged(snapshot)
    return {"processed": processed, "failed": failed, "skipped_complete": 0}


def create_quality_indexes(connection: sqlite3.Connection) -> None:
    statements = (
        "CREATE INDEX IF NOT EXISTS idx_quality_species_task "
        "ON quality_records(species, task)",
        "CREATE INDEX IF NOT EXISTS idx_quality_normality "
        "ON quality_records(species, normality)",
        "CREATE INDEX IF NOT EXISTS idx_quality_disease "
        "ON quality_records(species, disease_type)",
        "CREATE INDEX IF NOT EXISTS idx_quality_segmentation "
        "ON quality_records(species, task, segmentation_count)",
        "CREATE INDEX IF NOT EXISTS idx_quality_group_session "
        "ON quality_records(species, camera_id, capture_date, capture_time)",
    )
    for statement in statements:
        connection.execute(statement)
    connection.commit()


def query_counts(
    connection: sqlite3.Connection,
    columns: Sequence[str],
    *,
    where: str = "",
) -> list[dict[str, Any]]:
    columns_sql = ", ".join(columns)
    sql = f"SELECT {columns_sql}, COUNT(*) AS count FROM quality_records"
    if where:
        sql += " WHERE " + where
    sql += " GROUP BY " + columns_sql + " ORDER BY " + columns_sql
    return [dict(row) for row in connection.execute(sql)]


def quality_statistics(
    database_path: Path,
) -> dict[str, Any]:
    connection = sql_connection(database_path, read_only=True)
    try:
        total = connection.execute(
            "SELECT COUNT(*) AS count FROM quality_records"
        ).fetchone()["count"]
        species = query_counts(connection, ("species",))
        species_task = query_counts(connection, ("species", "task"))
        species_normality = query_counts(
            connection, ("species", "normality")
        )
        disease = query_counts(
            connection,
            ("disease_type",),
            where="disease_type IS NOT NULL AND disease_type <> ''",
        )
        species_disease = query_counts(
            connection,
            ("species", "disease_type"),
            where="disease_type IS NOT NULL AND disease_type <> ''",
        )
        annotation_distribution = query_counts(
            connection, ("annotation_count",)
        )
        segmentation_species_task = [
            dict(row)
            for row in connection.execute(
                """
                SELECT species, task,
                       SUM(CASE WHEN segmentation_count > 0 THEN 1 ELSE 0 END)
                           AS non_null_count,
                       COUNT(*) AS total_count
                FROM quality_records
                GROUP BY species, task
                ORDER BY species, task
                """
            )
        ]
        bbox_missing = [
            dict(row)
            for row in connection.execute(
                """
                SELECT official_split, archive_id, species, task, json_member,
                       image_filename, annotation_count, bbox_missing_reason
                FROM quality_records
                WHERE bbox_count = 0
                ORDER BY official_split, archive_id, json_member
                """
            )
        ]
        task_normality = query_counts(
            connection, ("task", "normality")
        )
        abnormal_missing_disease = connection.execute(
            """
            SELECT COUNT(*) AS count FROM quality_records
            WHERE normality='abnormal'
              AND (disease_type IS NULL OR disease_type='')
            """
        ).fetchone()["count"]
        automatic_issue_totals = dict(
            connection.execute(
                """
                SELECT
                    SUM(bbox_nonpositive_count) AS bbox_nonpositive,
                    SUM(bbox_out_of_bounds_count) AS bbox_out_of_bounds,
                    SUM(bbox_tiny_count) AS bbox_tiny,
                    SUM(bbox_near_full_count) AS bbox_near_full,
                    SUM(segmentation_out_of_bounds_count) AS segmentation_out,
                    SUM(segmentation_invalid_count) AS segmentation_invalid
                FROM quality_records
                """
            ).fetchone()
        )
        annotation_totals = dict(
            connection.execute(
                """
                SELECT
                    SUM(annotation_count) AS annotation_count,
                    SUM(bbox_count) AS bbox_count,
                    SUM(segmentation_count) AS segmentation_count
                FROM quality_records
                """
            ).fetchone()
        )
        return {
            "total": total,
            "species": species,
            "species_task": species_task,
            "species_normality": species_normality,
            "disease": disease,
            "species_disease": species_disease,
            "annotation_distribution": annotation_distribution,
            "segmentation_species_task": segmentation_species_task,
            "bbox_missing": bbox_missing,
            "task_normality": task_normality,
            "abnormal_missing_disease": abnormal_missing_disease,
            "automatic_issue_totals": automatic_issue_totals,
            "annotation_totals": annotation_totals,
        }
    finally:
        connection.close()


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def disease_distribution_rows(stats: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_disease = sum(int(item["count"]) for item in stats["disease"])
    overall_sorted = sorted(
        stats["disease"], key=lambda item: int(item["count"]), reverse=True
    )
    for rank, item in enumerate(overall_sorted, start=1):
        count = int(item["count"])
        rows.append(
            {
                "scope": "overall",
                "species": "",
                "disease_type": item["disease_type"],
                "count": count,
                "rate_within_scope": count / total_disease,
                "class_rank": rank,
            }
        )
    totals_by_species: Counter[str] = Counter()
    for item in stats["species_disease"]:
        totals_by_species[item["species"]] += int(item["count"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in stats["species_disease"]:
        grouped[item["species"]].append(item)
    for species, items in sorted(grouped.items()):
        ordered = sorted(
            items, key=lambda item: int(item["count"]), reverse=True
        )
        for rank, item in enumerate(ordered, start=1):
            count = int(item["count"])
            rows.append(
                {
                    "scope": "species",
                    "species": species,
                    "disease_type": item["disease_type"],
                    "count": count,
                    "rate_within_scope": count / totals_by_species[species],
                    "class_rank": rank,
                }
            )
    return rows


def annotation_quality_rows(stats: Mapping[str, Any]) -> list[dict[str, Any]]:
    total = int(stats["total"])
    rows: list[dict[str, Any]] = []
    for item in stats["annotation_distribution"]:
        count = int(item["count"])
        rows.append(
            {
                "section": "annotation_count",
                "species": "",
                "task": "",
                "metric": "json_annotation_count",
                "value": item["annotation_count"],
                "count": count,
                "rate": count / total,
                "archive_id": "",
                "json_member": "",
                "reason": "",
            }
        )
    for item in stats["segmentation_species_task"]:
        count = int(item["non_null_count"])
        group_total = int(item["total_count"])
        rows.append(
            {
                "section": "segmentation_species_task",
                "species": item["species"],
                "task": item["task"],
                "metric": "segmentation_non_null_json",
                "value": "non_null",
                "count": count,
                "rate": count / group_total if group_total else 0,
                "archive_id": "",
                "json_member": "",
                "reason": "",
            }
        )
    for item in stats["bbox_missing"]:
        rows.append(
            {
                "section": "bbox_missing_json",
                "species": item["species"],
                "task": item["task"],
                "metric": "bbox_missing",
                "value": item["official_split"],
                "count": 1,
                "rate": 1 / total,
                "archive_id": item["archive_id"],
                "json_member": item["json_member"],
                "reason": item["bbox_missing_reason"],
            }
        )
    for metric, count in stats["automatic_issue_totals"].items():
        rows.append(
            {
                "section": "automatic_coordinate_validation",
                "species": "",
                "task": "",
                "metric": metric,
                "value": "annotation_or_polygon_count",
                "count": int(count or 0),
                "rate": "",
                "archive_id": "",
                "json_member": "",
                "reason": "",
            }
        )
    return rows


def markdown_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def write_full_data_quality_report(
    path: Path,
    stats: Mapping[str, Any],
) -> None:
    disease_counts = [int(item["count"]) for item in stats["disease"]]
    imbalance_ratio = (
        max(disease_counts) / min(disease_counts)
        if disease_counts and min(disease_counts)
        else 0.0
    )
    disease_species: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for item in stats["species_disease"]:
        disease_species[str(item["disease_type"])].append(
            (str(item["species"]), int(item["count"]))
        )
    exclusive_diseases = {
        disease: values[0][0]
        for disease, values in disease_species.items()
        if len(values) == 1
    }
    lines = [
        "# AIHub 버섯 전체 라벨 품질 검토",
        "",
        "## 분석 범위",
        "",
        f"- 전체 라벨 JSON: {stats['total']:,}개",
        "- 원본 라벨 ZIP을 읽기 전용으로 순차 스캔",
        "- 전체 이미지 추출·디코딩 없음",
        "- 모델 학습·YOLO 변환 없음",
        "",
        "## 품종별 개수",
        "",
    ]
    lines.extend(
        markdown_table(
            ("품종", "JSON 수"),
            ((item["species"], f"{item['count']:,}") for item in stats["species"]),
        )
    )
    lines.extend(["", "## 품종×작업 개수", ""])
    lines.extend(
        markdown_table(
            ("품종", "작업", "JSON 수"),
            (
                (item["species"], item["task"], f"{item['count']:,}")
                for item in stats["species_task"]
            ),
        )
    )
    lines.extend(["", "## 품종별 정상/비정상", ""])
    lines.extend(
        markdown_table(
            ("품종", "정상 여부", "JSON 수"),
            (
                (item["species"], item["normality"], f"{item['count']:,}")
                for item in stats["species_normality"]
            ),
        )
    )
    lines.extend(["", "## 병해 종류별 개수", ""])
    lines.extend(
        markdown_table(
            ("병해 종류", "JSON 수"),
            (
                (item["disease_type"], f"{item['count']:,}")
                for item in sorted(
                    stats["disease"],
                    key=lambda value: int(value["count"]),
                    reverse=True,
                )
            ),
        )
    )
    lines.extend(["", "## 품종×병해 종류", ""])
    lines.extend(
        markdown_table(
            ("품종", "병해 종류", "JSON 수"),
            (
                (
                    item["species"],
                    item["disease_type"],
                    f"{item['count']:,}",
                )
                for item in stats["species_disease"]
            ),
        )
    )
    lines.extend(
        [
            "",
            "## 병해 클래스 불균형",
            "",
            f"- 최대/최소 클래스 비율: {imbalance_ratio:.2f}:1",
            "- 40.11%인 세균갈색무늬병과 6.50%인 솜털곰팡이병 사이의 차이가 커 병해 클래스는 불균형합니다.",
            "- 클래스별 데이터 수 차이가 커서 단순 정확도보다 macro F1, 클래스별 재현율, 혼동행렬이 필요합니다.",
            "- 품종별 병해 종류의 지원 집합이 다르므로 모델이 병변 대신 품종을 지름길로 사용할 위험이 큽니다.",
            "",
            "품종 하나에만 존재하는 병해: "
            + (
                ", ".join(
                    f"{disease}→{species}"
                    for disease, species in sorted(exclusive_diseases.items())
                )
                if exclusive_diseases
                else "없음"
            ),
            "",
            "## annotation 개수 분포",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ("annotation 수", "JSON 수", "비율"),
            (
                (
                    item["annotation_count"],
                    f"{item['count']:,}",
                    f"{int(item['count']) / int(stats['total']):.4%}",
                )
                for item in stats["annotation_distribution"]
            ),
        )
    )
    lines.extend(["", "## bbox 결측 JSON", ""])
    lines.extend(
        markdown_table(
            (
                "공식 split",
                "archive ID",
                "품종",
                "작업",
                "JSON 멤버",
                "annotation 수",
                "원인",
            ),
            (
                (
                    item["official_split"],
                    item["archive_id"],
                    item["species"],
                    item["task"],
                    f"`{item['json_member']}`",
                    item["annotation_count"],
                    item["bbox_missing_reason"],
                )
                for item in stats["bbox_missing"]
            ),
        )
    )
    lines.extend(["", "## segmentation non-null 품종×작업 분포", ""])
    lines.extend(
        markdown_table(
            ("품종", "작업", "non-null JSON", "조합 전체", "비율"),
            (
                (
                    item["species"],
                    item["task"],
                    f"{item['non_null_count']:,}",
                    f"{item['total_count']:,}",
                    f"{int(item['non_null_count']) / int(item['total_count']):.2%}",
                )
                for item in stats["segmentation_species_task"]
            ),
        )
    )
    lines.extend(
        [
            "",
            "## 정상/비정상과 작업 폴더의 대응",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ("작업", "정상 여부", "JSON 수"),
            (
                (item["task"], item["normality"], f"{item['count']:,}")
                for item in stats["task_normality"]
            ),
        )
    )
    lines.extend(
        [
            "",
            f"- 비정상이지만 병해 종류가 비어 있는 JSON: "
            f"{stats['abnormal_missing_disease']:,}개",
            "- 대응은 완전합니다: 병해 폴더는 모두 비정상이고 배양·생육 폴더는 모두 정상입니다.",
            "- 따라서 정상/비정상 모델은 병변보다 작업별 촬영 장소·구도·조명 차이를 학습할 위험이 매우 큽니다.",
            "",
            "## 자동 좌표 검사",
            "",
            "아래 수치는 JSON 수가 아니라 annotation 또는 polygon 수입니다. "
            "작은 bbox는 양송이 유묘처럼 실제로 작은 개체도 포함하므로 "
            "곧바로 오류로 확정하지 않습니다.",
            "",
        ]
    )
    annotation_total = int(
        stats["annotation_totals"]["annotation_count"] or 0
    )
    segmentation_total = int(
        stats["annotation_totals"]["segmentation_count"] or 0
    )
    denominators = {
        "bbox_nonpositive": annotation_total,
        "bbox_out_of_bounds": annotation_total,
        "bbox_tiny": annotation_total,
        "bbox_near_full": annotation_total,
        "segmentation_out": segmentation_total,
        "segmentation_invalid": segmentation_total,
    }
    lines.extend(
        markdown_table(
            ("검사", "발견 수", "해당 annotation 기준 비율"),
            (
                (
                    key,
                    f"{int(value or 0):,}",
                    (
                        f"{int(value or 0) / denominators[key]:.4%}"
                        if denominators[key]
                        else "N/A"
                    ),
                )
                for key, value in stats["automatic_issue_totals"].items()
            ),
        )
    )
    lines.extend(
        [
            "",
            "- 0 이하 bbox 8개는 대표 JSON 확인 결과 모두 width=0, height=0인 점 annotation이므로 학습 전 제거 대상입니다.",
            "- 범위 밖 bbox 9개는 대표 확인에서 영상 경계를 조금 넘겨 컨테이너 또는 버섯 군집을 감싼 사례였으며, 좌표 클리핑 후 의미 보존 여부를 확인해야 합니다.",
            "- 작은 bbox는 양송이 유묘를 실제로 표시하는 경우가 많아 면적 임계값만으로 삭제하면 안 됩니다.",
            "- 범위 밖 segmentation은 제한 표본에서 영상 경계에 걸쳐 잘린 양송이 개체였으므로 오류와 정상 truncated instance를 구분해야 합니다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_groups(
    connection: sqlite3.Connection,
    columns: Sequence[str],
) -> dict[tuple[str, ...], SplitGroup]:
    key_sql = ", ".join(
        f"COALESCE({column}, '<missing>') AS key_{index}"
        for index, column in enumerate(columns)
    )
    group_sql = ", ".join(columns)
    rows = connection.execute(
        f"""
        SELECT {key_sql}, species, task, normality,
               COALESCE(disease_type, '<missing>') AS disease_type,
               official_split, COUNT(*) AS count
        FROM quality_records
        GROUP BY {group_sql}, species, task, normality,
                 COALESCE(disease_type, '<missing>'), official_split
        """
    )
    groups: dict[tuple[str, ...], SplitGroup] = {}
    for row in rows:
        key = tuple(
            str(row[f"key_{index}"]) for index in range(len(columns))
        )
        group = groups.setdefault(key, SplitGroup(key=key))
        count = int(row["count"])
        group.total += count
        group.official_counts[row["official_split"]] += count
        group.features[f"species:{row['species']}"] += count
        group.features[f"task:{row['task']}"] += count
        group.features[f"normality:{row['normality']}"] += count
        if row["disease_type"] != "<missing>":
            group.features[f"disease:{row['disease_type']}"] += count
    return groups


def feature_weight(feature: str) -> float:
    if feature == "__total__":
        return 10.0
    if feature.startswith("disease:"):
        return 15.0
    if feature.startswith("task:"):
        return 30.0
    if feature.startswith("species:"):
        return 1.0
    if feature.startswith("normality:"):
        # 정상/비정상은 현재 작업과 완전히 대응하므로 중복 가중하지 않습니다.
        return 0.0
    return 1.0


def deterministic_group_assignment(
    groups: Mapping[tuple[str, ...], SplitGroup],
    *,
    seed: int,
    ratios: Mapping[str, float] = SPLIT_RATIOS,
    stratify_by_first_key: bool = False,
) -> tuple[dict[tuple[str, ...], str], dict[str, Counter[str]]]:
    if stratify_by_first_key:
        partitions: dict[str, dict[tuple[str, ...], SplitGroup]] = defaultdict(
            dict
        )
        for key, group in groups.items():
            partitions[key[0]][key] = group
        combined_assignment: dict[tuple[str, ...], str] = {}
        combined_counts = {split: Counter() for split in ratios}
        for partition_name, partition in sorted(partitions.items()):
            partition_assignment, partition_counts = (
                deterministic_group_assignment(
                    partition,
                    seed=int(
                        stable_hash(seed, partition_name)[:16], 16
                    ),
                    ratios=ratios,
                    stratify_by_first_key=False,
                )
            )
            combined_assignment.update(partition_assignment)
            for split in ratios:
                combined_counts[split].update(partition_counts[split])
        return combined_assignment, combined_counts

    overall = Counter({"__total__": sum(group.total for group in groups.values())})
    for group in groups.values():
        overall.update(group.features)
    targets = {
        split: Counter(
            {feature: value * ratio for feature, value in overall.items()}
        )
        for split, ratio in ratios.items()
    }
    assigned_counts = {split: Counter() for split in ratios}
    assignment: dict[tuple[str, ...], str] = {}
    ordered = sorted(
        groups.values(),
        key=lambda group: (
            -group.total,
            stable_hash(seed, *group.key),
        ),
    )
    for group in ordered:
        additions = Counter(group.features)
        additions["__total__"] = group.total
        scored: list[tuple[float, str]] = []
        for split in ratios:
            score = 0.0
            for feature, addition in additions.items():
                target = float(targets[split][feature])
                current = float(assigned_counts[split][feature])
                denominator = max(target, 1.0)
                before = ((current - target) / denominator) ** 2
                after = ((current + addition - target) / denominator) ** 2
                score += feature_weight(feature) * (after - before)
            target_total = float(targets[split]["__total__"])
            projected = assigned_counts[split]["__total__"] + group.total
            if projected > target_total + group.total:
                score += (projected - target_total) / max(target_total, 1.0)
            scored.append((score, split))
        selected = min(
            scored,
            key=lambda item: (
                item[0],
                stable_hash(seed, group.key, item[1]),
            ),
        )[1]
        assignment[group.key] = selected
        assigned_counts[selected].update(additions)
    return assignment, assigned_counts


def class_group_counts(
    groups: Mapping[tuple[str, ...], SplitGroup],
) -> dict[str, Counter[str]]:
    result = {
        "species": Counter(),
        "task": Counter(),
        "disease": Counter(),
    }
    for group in groups.values():
        for feature, count in group.features.items():
            if count <= 0:
                continue
            prefix, _, value = feature.partition(":")
            if prefix in result:
                result[prefix][value] += 1
    return result


def maximum_class_deviation_pp(
    assigned: Mapping[str, Counter[str]],
) -> tuple[float, dict[str, float]]:
    total = sum(counter["__total__"] for counter in assigned.values())
    overall = Counter()
    for counter in assigned.values():
        overall.update(counter)
    max_by_family: dict[str, float] = {
        "species": 0.0,
        "task": 0.0,
        "disease": 0.0,
    }
    for split, counter in assigned.items():
        split_total = counter["__total__"]
        if not split_total:
            continue
        for feature, overall_count in overall.items():
            family, separator, _value = feature.partition(":")
            if not separator or family not in max_by_family:
                continue
            overall_rate = overall_count / total
            split_rate = counter[feature] / split_total
            deviation = abs(split_rate - overall_rate) * 100
            max_by_family[family] = max(
                max_by_family[family], deviation
            )
    return max(max_by_family.values()), max_by_family


def split_candidate_metrics(
    database_path: Path,
    *,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    connection = sql_connection(database_path, read_only=True)
    try:
        results: list[dict[str, Any]] = []
        for name, columns in SPLIT_CANDIDATES:
            groups = load_groups(connection, columns)
            assignment, assigned = deterministic_group_assignment(
                groups, seed=seed, stratify_by_first_key=True
            )
            total = sum(group.total for group in groups.values())
            largest = max((group.total for group in groups.values()), default=0)
            group_counts = class_group_counts(groups)
            mixed_official = sum(
                bool(group.official_counts.get("train"))
                and bool(group.official_counts.get("validation"))
                for group in groups.values()
            )
            changed = 0
            for key, group in groups.items():
                selected = assignment[key]
                retained = (
                    group.official_counts.get(selected, 0)
                    if selected in ("train", "validation")
                    else 0
                )
                changed += group.total - retained
            max_deviation, family_deviation = maximum_class_deviation_pp(
                assigned
            )
            if largest / total > 0.10 or max_deviation > 10:
                feasibility = "poor"
            elif largest / total > 0.05 or max_deviation > 5:
                feasibility = "limited"
            elif max_deviation > 2:
                feasibility = "acceptable"
            else:
                feasibility = "good"
            results.append(
                {
                    "candidate_key": name,
                    "group_columns": " + ".join(columns),
                    "seed": seed,
                    "group_count": len(groups),
                    "largest_group_images": largest,
                    "largest_group_fraction": largest / total if total else 0,
                    "species_group_counts": json.dumps(
                        dict(group_counts["species"]),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "task_group_counts": json.dumps(
                        dict(group_counts["task"]),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "disease_group_counts": json.dumps(
                        dict(group_counts["disease"]),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "achieved_train_count": assigned["train"]["__total__"],
                    "achieved_validation_count": assigned["validation"][
                        "__total__"
                    ],
                    "achieved_test_count": assigned["test"]["__total__"],
                    "achieved_train_rate": assigned["train"]["__total__"]
                    / total,
                    "achieved_validation_rate": assigned["validation"][
                        "__total__"
                    ]
                    / total,
                    "achieved_test_rate": assigned["test"]["__total__"]
                    / total,
                    "achieved_train_groups": sum(
                        split == "train" for split in assignment.values()
                    ),
                    "achieved_validation_groups": sum(
                        split == "validation"
                        for split in assignment.values()
                    ),
                    "achieved_test_groups": sum(
                        split == "test" for split in assignment.values()
                    ),
                    "max_class_deviation_pp": max_deviation,
                    "family_deviation_pp": json.dumps(
                        family_deviation, sort_keys=True
                    ),
                    "group_overlap_count": (
                        len(assignment) - len(set(assignment))
                    ),
                    "official_mixed_group_count": mixed_official,
                    "official_assignment_changed_count": changed,
                    "official_assignment_changed_rate": changed / total,
                    "balance_feasibility": feasibility,
                }
            )
        return results
    finally:
        connection.close()


def recommend_split_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    lookup = {item["candidate_key"]: item for item in candidates}
    date_candidate = lookup["species_camera_date"]
    if (
        float(date_candidate["largest_group_fraction"]) <= 0.05
        and float(date_candidate["max_class_deviation_pp"]) <= 5.0
    ):
        return date_candidate
    return lookup["species_camera_date_time"]


def write_split_reports(
    output_dir: Path,
    candidates: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> Mapping[str, Any]:
    fieldnames = (
        "candidate_key",
        "group_columns",
        "seed",
        "group_count",
        "largest_group_images",
        "largest_group_fraction",
        "species_group_counts",
        "task_group_counts",
        "disease_group_counts",
        "achieved_train_count",
        "achieved_validation_count",
        "achieved_test_count",
        "achieved_train_rate",
        "achieved_validation_rate",
        "achieved_test_rate",
        "achieved_train_groups",
        "achieved_validation_groups",
        "achieved_test_groups",
        "max_class_deviation_pp",
        "family_deviation_pp",
        "group_overlap_count",
        "official_mixed_group_count",
        "official_assignment_changed_count",
        "official_assignment_changed_rate",
        "balance_feasibility",
    )
    write_csv(
        output_dir / "split_candidate_comparison.csv",
        fieldnames,
        candidates,
    )
    recommended = recommend_split_candidate(candidates)
    lines = [
        "# 데이터 누수 방지 분할 전략",
        "",
        f"- 목표 비율: Train {SPLIT_RATIOS['train']:.0%}, "
        f"Validation {SPLIT_RATIOS['validation']:.0%}, "
        f"Test {SPLIT_RATIOS['test']:.0%}",
        f"- 고정 seed: {seed}",
        "- 같은 그룹은 정확히 하나의 split에만 배정",
        "- Test는 모델 선택에 사용하지 않는 최종 평가 전용",
        "- 현재 결과는 후보 평가용 임시 배정이며 최종 분할 CSV는 생성하지 않음",
        "",
        "## 후보 비교",
        "",
    ]
    lines.extend(
        markdown_table(
            (
                "후보",
                "그룹 수",
                "최대 그룹",
                "split별 그룹 T/V/Test",
                "달성 비율 T/V/Test",
                "최대 클래스 편차",
                "그룹 중복",
                "공식 split 혼합 그룹",
                "공식 배정 변경",
                "균형 가능성",
            ),
            (
                (
                    item["candidate_key"],
                    f"{item['group_count']:,}",
                    f"{item['largest_group_images']:,} "
                    f"({float(item['largest_group_fraction']):.2%})",
                    f"{item['achieved_train_groups']:,}/"
                    f"{item['achieved_validation_groups']:,}/"
                    f"{item['achieved_test_groups']:,}",
                    f"{float(item['achieved_train_rate']):.2%}/"
                    f"{float(item['achieved_validation_rate']):.2%}/"
                    f"{float(item['achieved_test_rate']):.2%}",
                    f"{float(item['max_class_deviation_pp']):.2f}pp",
                    item["group_overlap_count"],
                    f"{item['official_mixed_group_count']:,}",
                    f"{float(item['official_assignment_changed_rate']):.2%}",
                    item["balance_feasibility"],
                )
                for item in candidates
            ),
        )
    )
    lines.extend(["", "## 후보별 클래스 그룹 수", ""])
    for item in candidates:
        lines.extend(
            [
                f"### {item['candidate_key']}",
                "",
            ]
        )
        for label, field in (
            ("품종", "species_group_counts"),
            ("작업", "task_group_counts"),
            ("병해", "disease_group_counts"),
        ):
            values = json.loads(str(item[field]))
            lines.extend(
                markdown_table(
                    (label, "그룹 수"),
                    (
                        (key, f"{int(value):,}")
                        for key, value in sorted(values.items())
                    ),
                )
            )
            lines.append("")
    lines.extend(
        [
            "",
            "## 권장안",
            "",
            f"권장 그룹 키: `{recommended['group_columns']}`",
            "",
        ]
    )
    if recommended["candidate_key"] == "species_camera_date":
        lines.extend(
            [
                "- 같은 카메라·같은 날의 연속 촬영을 묶어 인접 시점·유사 구도의 누수를 줄입니다.",
                "- 시간까지 포함한 키는 초 단위가 달라도 거의 같은 장면일 수 있어 누수 방지 단위로는 충분히 보수적이지 않습니다.",
                "- 카메라 전체를 묶는 키보다 그룹 크기가 작아 품종·작업·병해 균형을 유지할 여지도 남습니다.",
                "- 현재 greedy 배정은 후보 비교용입니다. 최종 CSV 단계에서는 목표 함수와 허용 편차를 고정하고 국소 최적화해야 합니다.",
            ]
        )
    else:
        lines.extend(
            [
                "- 날짜 단위 그룹의 최대 크기 또는 클래스 편차가 커서 시간까지 포함한 촬영 세션 키를 우선합니다.",
                "- 이후 근접 시간 창을 합치는 민감도 분석이 필요합니다.",
            ]
        )
    lines.extend(
        [
            f"- 임시 배정의 그룹 중복: {recommended['group_overlap_count']}개",
            f"- 임시 달성 비율: "
            f"{float(recommended['achieved_train_rate']):.2%}/"
            f"{float(recommended['achieved_validation_rate']):.2%}/"
            f"{float(recommended['achieved_test_rate']):.2%}",
            f"- 최대 클래스 비율 편차: "
            f"{float(recommended['max_class_deviation_pp']):.2f}pp",
            "",
            "최종 CSV 생성 전에는 품종×작업×병해 교차 분포와 각 split의 대표 시각 샘플을 다시 확인해야 합니다.",
        ]
    )
    (output_dir / "recommended_split_strategy.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return recommended


def row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (row["official_split"], row["archive_id"], row["json_member"])


def select_visual_samples(
    database_path: Path,
    *,
    max_per_combination: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not 1 <= max_per_combination <= MAX_VISUAL_PER_COMBINATION:
        raise ValueError("품종×작업 조합당 시각 감사 범위는 1~20개입니다")
    connection = sql_connection(database_path, read_only=True)
    try:
        combinations = list(
            connection.execute(
                "SELECT DISTINCT species, task FROM quality_records "
                "ORDER BY species, task"
            )
        )
        selected: list[dict[str, Any]] = []
        for combination in combinations:
            rows = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT * FROM quality_records
                    WHERE species=? AND task=?
                    """,
                    (combination["species"], combination["task"]),
                )
            ]
            priority = [
                row for row in rows if int(row["bbox_count"]) == 0
            ]
            issue_fields = (
                "bbox_nonpositive_count",
                "bbox_out_of_bounds_count",
                "bbox_tiny_count",
                "bbox_near_full_count",
                "segmentation_out_of_bounds_count",
                "segmentation_invalid_count",
            )
            for issue_field in issue_fields:
                issue_rows = sorted(
                    (
                        row
                        for row in rows
                        if int(row[issue_field] or 0) > 0
                    ),
                    key=lambda item: stable_hash(
                        seed,
                        issue_field,
                        item["official_split"],
                        item["archive_id"],
                        item["json_member"],
                    ),
                )
                if issue_rows:
                    priority.append(issue_rows[0])
            buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                if row["task"] == "병해":
                    bucket = f"disease:{row['disease_type'] or '<missing>'}"
                else:
                    bucket = (
                        f"seg:{int(row['segmentation_count']) > 0}:"
                        f"{row['official_split']}"
                    )
                buckets[bucket].append(row)
            for bucket_rows in buckets.values():
                bucket_rows.sort(
                    key=lambda row: stable_hash(
                        seed,
                        row["official_split"],
                        row["archive_id"],
                        row["json_member"],
                    )
                )
            chosen: list[dict[str, Any]] = []
            chosen_keys: set[tuple[str, str, str]] = set()
            for row in sorted(
                priority,
                key=lambda item: stable_hash(
                    seed, item["official_split"], item["json_member"]
                ),
            ):
                if len(chosen) >= max_per_combination:
                    break
                chosen.append(row)
                chosen_keys.add(row_key(row))
            position = 0
            ordered_buckets = sorted(buckets)
            while len(chosen) < max_per_combination:
                added = False
                for bucket in ordered_buckets:
                    bucket_rows = buckets[bucket]
                    if position >= len(bucket_rows):
                        continue
                    row = bucket_rows[position]
                    if row_key(row) not in chosen_keys:
                        chosen.append(row)
                        chosen_keys.add(row_key(row))
                        added = True
                    if len(chosen) >= max_per_combination:
                        break
                if not added and all(
                    position >= len(buckets[bucket]) - 1
                    for bucket in ordered_buckets
                ):
                    break
                position += 1
            selected.extend(chosen)
        return selected
    finally:
        connection.close()


def find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("/").joinpath("mnt", "c", "Windows", "Fonts", "malgun.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def overlay_text(
    image: Image.Image,
    lines: Sequence[str],
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    font_size = max(18, round(min(image.size) / 45))
    font = find_font(font_size)
    spacing = max(3, font_size // 6)
    text = "\n".join(lines)
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    padding = max(8, font_size // 3)
    width = min(image.width, box[2] - box[0] + padding * 2)
    height = box[3] - box[1] + padding * 2
    draw.rectangle((0, 0, width, height), fill=(0, 0, 0, 190))
    draw.multiline_text(
        (padding, padding),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        spacing=spacing,
    )


def draw_bbox_overlay(
    image: Image.Image,
    annotations: Sequence[Any],
    label_lines: Sequence[str],
) -> tuple[Image.Image, set[str]]:
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    issues: set[str] = set()
    line_width = max(3, round(min(output.size) / 250))
    for annotation in annotations:
        bbox = bbox_values(annotation)
        if bbox is None:
            continue
        issues.update(validate_bbox(bbox, output.width, output.height))
        x, y, width, height = bbox
        draw.rectangle(
            (x, y, x + width, y + height),
            outline=(255, 40, 40),
            width=line_width,
        )
    overlay_text(output, label_lines)
    return output, issues


def draw_segmentation_overlay(
    image: Image.Image,
    annotations: Sequence[Any],
    label_lines: Sequence[str],
) -> tuple[Image.Image, set[str], int]:
    output = image.copy().convert("RGBA")
    polygon_layer = Image.new("RGBA", output.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(polygon_layer)
    issues: set[str] = set()
    polygon_count = 0
    line_width = max(3, round(min(output.size) / 250))
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        segmentation = annotation.get("SEGMENTATION")
        if segmentation is None:
            continue
        polygons = segmentation_polygons(segmentation)
        if not polygons:
            issues.add("segmentation_invalid")
            continue
        for polygon in polygons:
            polygon_count += 1
            if any(
                x < 0 or y < 0 or x > output.width or y > output.height
                for x, y in polygon
            ):
                issues.add("segmentation_out_of_bounds")
            draw.polygon(
                polygon,
                fill=(30, 220, 80, 75),
                outline=(30, 255, 80, 240),
            )
            draw.line(
                polygon + [polygon[0]],
                fill=(30, 255, 80, 255),
                width=line_width,
            )
    output = Image.alpha_composite(output, polygon_layer).convert("RGB")
    overlay_text(output, label_lines)
    return output, issues, polygon_count


def save_overlay(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        image.save(path, format="JPEG", quality=92, optimize=True)
    else:
        image.save(path)


def relative_artifact_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def visual_manifest_rows(
    selected: Sequence[Mapping[str, Any]],
    refs: Sequence[base.ArchiveRef],
    visual_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    label_lookup = {
        (ref.split, ref.archive_id): ref for ref in refs if ref.kind == "label"
    }
    image_lookup = {
        (ref.split, ref.index): ref for ref in refs if ref.kind == "image"
    }
    selected_by_archive: dict[tuple[str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in selected:
        selected_by_archive[
            (row["official_split"], row["archive_id"])
        ].append(row)
    source_paths = [
        ref.path
        for ref in refs
        if ref.kind in ("label", "image")
    ]
    snapshot = source_zip_snapshot(source_paths)
    manifest: list[dict[str, Any]] = []
    try:
        image_index_cache: dict[
            Path,
            tuple[
                dict[str, list[str]],
                dict[str, list[str]],
                dict[str, list[str]],
            ],
        ] = {}
        for (split, archive_id), rows in tqdm(
            sorted(selected_by_archive.items()),
            desc="시각 감사 ZIP",
            unit="ZIP-pair",
        ):
            label_ref = label_lookup[(split, archive_id)]
            image_ref = image_lookup[(split, label_ref.index)]
            if image_ref.path not in image_index_cache:
                inspection = base.inspect_archive(image_ref)
                image_index_cache[image_ref.path] = base.build_image_indexes(
                    inspection
                )
            indexes = image_index_cache[image_ref.path]
            with zipfile.ZipFile(label_ref.path, "r") as label_zip:
                with zipfile.ZipFile(image_ref.path, "r") as image_zip:
                    info_lookup = {
                        base.normalize_member_name(info.filename): info
                        for info in image_zip.infolist()
                        if not info.is_dir()
                    }
                    for row in rows:
                        json_member = row["json_member"]
                        raw_json = label_zip.read(json_member)
                        text, _encoding = base.decode_json_bytes(raw_json)
                        data = json.loads(text)
                        sample = base.JsonSample(
                            archive=label_ref,
                            member_name=json_member,
                            task=row["task"],
                            path_values={
                                "$.IMAGE.IMAGE_FILE_NAME": [
                                    row["image_filename"]
                                ]
                            },
                        )
                        match = base.match_image(sample, indexes)
                        if match.status != "matched":
                            manifest.append(
                                {
                                    "sample_id": stable_hash(
                                        seed,
                                        split,
                                        archive_id,
                                        json_member,
                                    )[:16],
                                    "official_split": split,
                                    "species": row["species"],
                                    "task": row["task"],
                                    "disease_type": row["disease_type"] or "",
                                    "normality": row["normality"],
                                    "label_archive_id": archive_id,
                                    "source_archive_id": image_ref.archive_id,
                                    "json_member": json_member,
                                    "image_member": "",
                                    "status": "image_unmatched",
                                    "automatic_issues": "image_unmatched",
                                }
                            )
                            continue
                        image_member = match.matched_member
                        image_bytes = image_zip.read(info_lookup[image_member])
                        with Image.open(io.BytesIO(image_bytes)) as loaded:
                            image = loaded.convert("RGB")
                        annotations_value = data.get("ANNOTATION_INFO", [])
                        annotations = (
                            annotations_value
                            if isinstance(annotations_value, list)
                            else []
                        )
                        bbox_count = sum(
                            bbox_values(annotation) is not None
                            for annotation in annotations
                        )
                        has_segmentation = any(
                            isinstance(annotation, dict)
                            and annotation.get("SEGMENTATION") is not None
                            for annotation in annotations
                        )
                        lines = (
                            f"품종: {row['species']}",
                            f"작업: {row['task']}",
                            f"병해: {row['disease_type'] or '없음'}",
                            f"정상 여부: {row['normality']}",
                            f"archive: {archive_id}",
                            f"bbox: {bbox_count}",
                            f"segmentation: {'yes' if has_segmentation else 'no'}",
                        )
                        bbox_overlay, bbox_issues = draw_bbox_overlay(
                            image, annotations, lines
                        )
                        base_root = (
                            Path(split)
                            / f"{image_ref.archive_id}_{image_ref.species}"
                        )
                        original_path = base.safe_destination(
                            visual_dir / "original" / base_root,
                            image_member,
                        )
                        bbox_path = base.safe_destination(
                            visual_dir / "bbox_overlay" / base_root,
                            image_member,
                        )
                        original_path.parent.mkdir(
                            parents=True, exist_ok=True
                        )
                        original_path.write_bytes(image_bytes)
                        save_overlay(bbox_overlay, bbox_path)
                        segmentation_path = ""
                        segmentation_issues: set[str] = set()
                        polygon_count = 0
                        if has_segmentation:
                            segmentation_overlay, segmentation_issues, polygon_count = (
                                draw_segmentation_overlay(
                                    image, annotations, lines
                                )
                            )
                            segmentation_output = base.safe_destination(
                                visual_dir
                                / "segmentation_overlay"
                                / base_root,
                                image_member,
                            )
                            save_overlay(
                                segmentation_overlay, segmentation_output
                            )
                            segmentation_path = relative_artifact_path(
                                segmentation_output
                            )
                        issues = sorted(
                            bbox_issues | segmentation_issues
                        )
                        manifest.append(
                            {
                                "sample_id": stable_hash(
                                    seed,
                                    split,
                                    archive_id,
                                    json_member,
                                )[:16],
                                "official_split": split,
                                "species": row["species"],
                                "task": row["task"],
                                "disease_type": row["disease_type"] or "",
                                "normality": row["normality"],
                                "label_archive_id": archive_id,
                                "source_archive_id": image_ref.archive_id,
                                "json_member": json_member,
                                "image_member": image_member,
                                "original_path": relative_artifact_path(
                                    original_path
                                ),
                                "bbox_overlay_path": relative_artifact_path(
                                    bbox_path
                                ),
                                "segmentation_overlay_path": segmentation_path,
                                "image_width": image.width,
                                "image_height": image.height,
                                "bbox_count": bbox_count,
                                "segmentation_present": has_segmentation,
                                "polygon_count": polygon_count,
                                "automatic_issues": " | ".join(issues),
                                "status": "generated",
                                "visual_review_status": "review_needed",
                                "visual_review_note": (
                                    "좌표 검사는 자동 완료; 실제 버섯과의 의미적 정렬은 사람 검토 필요"
                                ),
                            }
                        )
    finally:
        assert_source_zips_unchanged(snapshot)
    return manifest


def create_contact_sheets(
    manifest: Sequence[Mapping[str, Any]],
    visual_dir: Path,
) -> list[Path]:
    created: list[Path] = []
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in manifest:
        if row.get("status") == "generated":
            grouped[(str(row["species"]), str(row["task"]))].append(row)
    for (species, task), rows in sorted(grouped.items()):
        for kind, field in (
            ("bbox", "bbox_overlay_path"),
            ("segmentation", "segmentation_overlay_path"),
        ):
            available = [row for row in rows if row.get(field)]
            if not available:
                continue
            columns = 5
            cell_width, cell_height = 260, 360
            title_height = 50
            row_count = math.ceil(len(available) / columns)
            sheet = Image.new(
                "RGB",
                (columns * cell_width, title_height + row_count * cell_height),
                "white",
            )
            title_font = find_font(24)
            ImageDraw.Draw(sheet).text(
                (10, 10),
                f"{species} / {task} / {kind}",
                font=title_font,
                fill="black",
            )
            for index, row in enumerate(available):
                path = PROJECT_ROOT / str(row[field])
                with Image.open(path) as loaded:
                    tile = ImageOps.contain(
                        loaded.convert("RGB"),
                        (cell_width - 12, cell_height - 12),
                    )
                x = (index % columns) * cell_width
                y = title_height + (index // columns) * cell_height
                sheet.paste(tile, (x + 6, y + 6))
            output = (
                visual_dir
                / "contact_sheets"
                / kind
                / f"{species}_{task}.jpg"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(output, format="JPEG", quality=90)
            created.append(output)
    return created


def load_manual_review(path: Path = DEFAULT_MANUAL_REVIEW) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_visual_audit_report(
    path: Path,
    manifest: Sequence[Mapping[str, Any]],
    manual_review: Mapping[str, Any],
) -> None:
    generated = [row for row in manifest if row.get("status") == "generated"]
    issues = Counter()
    for row in generated:
        for issue in str(row.get("automatic_issues", "")).split(" | "):
            if issue:
                issues[issue] += 1
    combo_counts = Counter(
        (str(row["species"]), str(row["task"])) for row in generated
    )
    observed_combination_max = max(combo_counts.values(), default=0)
    disease_counts = Counter(
        str(row["disease_type"])
        for row in generated
        if row.get("task") == "병해"
    )
    segmentation_count = sum(
        str(row.get("segmentation_present")).lower() == "true"
        or row.get("segmentation_present") is True
        for row in generated
    )
    lines = [
        "# 시각적 라벨 감사",
        "",
        "## 범위",
        "",
        f"- 원천 이미지 제한 추출: {len(generated):,}개",
        f"- bbox overlay: {len(generated):,}개",
        f"- segmentation overlay: {segmentation_count:,}개",
        f"- 품종×작업 조합별 실제 최대: {observed_combination_max}개 "
        f"(프로그램 상한 {MAX_VISUAL_PER_COMBINATION}개)",
        "- 전체 이미지 추출·디코딩 없음",
        "- 원본 ZIP 크기와 수정시각 불변 확인",
        "",
        "## 조합별 샘플",
        "",
    ]
    lines.extend(
        markdown_table(
            ("품종", "작업", "이미지 수"),
            (
                (species, task, count)
                for (species, task), count in sorted(combo_counts.items())
            ),
        )
    )
    lines.extend(["", "## 병해 종류별 시각 샘플", ""])
    lines.extend(
        markdown_table(
            ("병해 종류", "이미지 수"),
            sorted(disease_counts.items()),
        )
    )
    lines.extend(["", "## 자동 좌표 검사", ""])
    if issues:
        lines.extend(
            markdown_table(
                ("검사 항목", "해당 이미지 수"),
                sorted(issues.items()),
            )
        )
    else:
        lines.append("- 제한 샘플에서 자동 좌표 오류를 발견하지 못함")
    lines.extend(
        [
            "",
            "## 사람 검토가 필요한 의미적 정렬",
            "",
            "자동 좌표 검사는 실제 버섯, 배지, 병반 중 무엇을 표시하는지 확정할 수 없습니다.",
        ]
    )
    if manual_review:
        labels = {
            "bbox_semantics": "bbox 의미",
            "segmentation_semantics": "segmentation 의미",
            "major_findings": "주요 관찰",
            "coordinate_findings": "좌표 이상 대표 사례",
            "review_needed": "추가 검토 필요",
        }
        for key, label in labels.items():
            value = manual_review.get(key)
            if value:
                lines.append(f"- {label}: {value}")
    else:
        lines.append(
            "- contact sheet 사람 검토가 아직 기록되지 않아 모든 샘플을 review_needed로 유지"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_scope_report(
    path: Path,
    stats: Mapping[str, Any],
    manifest: Sequence[Mapping[str, Any]],
    manual_review: Mapping[str, Any],
) -> None:
    total = int(stats["total"])
    bbox_available = total - len(stats["bbox_missing"])
    segmentation_available = sum(
        int(item["non_null_count"])
        for item in stats["segmentation_species_task"]
    )
    normal_count = sum(
        int(item["count"])
        for item in stats["species_normality"]
        if item["normality"] == "normal"
    )
    abnormal_count = sum(
        int(item["count"])
        for item in stats["species_normality"]
        if item["normality"] == "abnormal"
    )
    disease_count = sum(int(item["count"]) for item in stats["disease"])
    disease_values = [int(item["count"]) for item in stats["disease"]]
    disease_imbalance = (
        max(disease_values) / min(disease_values)
        if disease_values and min(disease_values)
        else 0
    )
    bbox_visual = manual_review.get(
        "bbox_semantics", "시각 검토 기록 필요"
    )
    non_culture_count = sum(
        int(item["count"])
        for item in stats["species_task"]
        if item["task"] != "배양"
    )
    candidates = [
        (
            "5품종 객체 탐지",
            f"{bbox_available:,} bbox JSON",
            f"좌표 완전성은 높으나 대상 단위가 작업·품종별로 다름; {bbox_visual}",
            "높음—촬영 세션·카메라 그룹 분할 필수",
            "품종 15.0%~25.6%",
            "높음—고정 카메라/재배사 구도와 스마트폰 도메인 차이",
            "중간",
            "조건부 권장",
        ),
        (
            "정상/병해 이진 분류",
            f"정상 {normal_count:,}, 비정상 {abnormal_count:,}",
            "라벨은 완전하지만 작업 폴더와 강한 대응 가능성",
            "매우 높음—병변 대신 폴더별 구도·환경 단서 학습 위험",
            f"약 {normal_count / abnormal_count:.2f}:1",
            "매우 높음",
            "낮음~중간",
            "첫 모델 비권장",
        ),
        (
            "병해 종류 다중 분류",
            f"{disease_count:,}개, {len(stats['disease'])}종",
            "병해 폴더 내 라벨은 완전",
            "높음—품종·카메라·날짜 confound",
            f"최대/최소 {disease_imbalance:.2f}:1",
            "매우 높음",
            "중간",
            "후속 과제",
        ),
        (
            "segmentation",
            f"{segmentation_available:,} JSON",
            "양송이 생육에만 존재; 제한 샘플에서 개별 버섯 윤곽과 대체로 정렬",
            "높음",
            "품종·작업 편중 가능",
            "높음",
            "높음",
            "첫 모델 비권장",
        ),
        (
            "생육 단계 분류",
            "전용 라벨 0개",
            "라벨 없음",
            "평가 불가",
            "평가 불가",
            "매우 높음",
            "구현 불가",
            "제외",
        ),
        (
            "형태 수치 회귀",
            "유효값 0개",
            "5개 필드 전부 100% 결측",
            "평가 불가",
            "평가 불가",
            "매우 높음",
            "구현 불가",
            "제외",
        ),
    ]
    lines = [
        "# 첫 모델 범위 결정",
        "",
        "첫 모델은 데이터셋 내부 benchmark 범위와 사용자 스마트폰 일반화 범위를 구분해야 합니다.",
        "",
    ]
    lines.extend(
        markdown_table(
            (
                "후보",
                "사용 가능 라벨",
                "라벨 신뢰성",
                "누수 위험",
                "불균형",
                "스마트폰 일반화 위험",
                "난이도",
                "판정",
            ),
            candidates,
        )
    )
    lines.extend(
        [
            "",
            "## 권장 첫 모델",
            "",
            "**권장: 라벨 의미를 정제한 생육·병해 범위의 5품종 객체 탐지 benchmark.**",
            "",
            f"- 1차 포함 후보는 배양을 제외한 생육·병해 {non_culture_count:,}개이며, 실제 버섯이 보이지 않고 배지 상단만 표시한 사례는 추가 제외 규칙이 필요합니다.",
            "- 배양 bbox는 주로 병·봉지·배지를 표시하므로 첫 탐지 모델 범위에서 제외합니다.",
            "- 생육·병해에서도 개체 bbox와 군집 bbox가 혼재하므로 학습 전 annotation 단위 정책을 확정해야 합니다.",
            "- 이는 스마트폰 실사용 성능을 보장하는 모델이 아니라 고정 카메라 데이터의 누수 방지 baseline입니다.",
            "- 정상/병해 분류는 작업 폴더와 라벨의 수집 편향을 먼저 해소하거나 별도 외부 사진 평가셋을 확보한 뒤 진행합니다.",
            "- Test는 모델 선택에 사용하지 않고 마지막 한 번의 평가에만 사용합니다.",
            "",
            f"시각 감사 manifest 이미지: {len(manifest):,}개",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_quality_reports(
    database_path: Path,
    output_dir: Path,
    *,
    seed: int,
    manifest_path: Path,
    manual_review_path: Path = DEFAULT_MANUAL_REVIEW,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = quality_statistics(database_path)
    write_csv(
        output_dir / "disease_class_distribution.csv",
        (
            "scope",
            "species",
            "disease_type",
            "count",
            "rate_within_scope",
            "class_rank",
        ),
        disease_distribution_rows(stats),
    )
    write_csv(
        output_dir / "annotation_quality_summary.csv",
        (
            "section",
            "species",
            "task",
            "metric",
            "value",
            "count",
            "rate",
            "archive_id",
            "json_member",
            "reason",
        ),
        annotation_quality_rows(stats),
    )
    write_full_data_quality_report(
        output_dir / "full_data_quality_review.md", stats
    )
    candidates = split_candidate_metrics(database_path, seed=seed)
    recommended = write_split_reports(
        output_dir, candidates, seed=seed
    )
    manifest = read_manifest(manifest_path)
    manual_review = load_manual_review(manual_review_path)
    write_visual_audit_report(
        output_dir / "visual_label_audit.md",
        manifest,
        manual_review,
    )
    write_model_scope_report(
        output_dir / "model_scope_decision.md",
        stats,
        manifest,
        manual_review,
    )
    return {
        "stats": stats,
        "split_candidates": candidates,
        "recommended": recommended,
        "manifest_count": len(manifest),
    }


def run(args: argparse.Namespace, environ: Mapping[str, str]) -> int:
    refs, issues, directories = base.discover_archives(environ)
    if issues:
        raise ValueError(
            "아카이브 구성이 기대와 다릅니다: "
            + "; ".join(
                base.redact_local_absolute_paths(issue) for issue in issues
            )
        )
    base.validate_output_location(args.output_dir, directories)
    base.validate_output_location(args.visual_dir, directories)
    base.validate_output_location(args.database, directories)
    label_refs = [ref for ref in refs if ref.kind == "label"]
    scan_result = {"processed": 0, "failed": 0, "skipped_complete": 0}
    if args.scan_metadata:
        scan_result = scan_label_quality_metadata(
            label_refs, args.database
        )
        if scan_result["failed"]:
            raise RuntimeError(
                f"품질 메타데이터 JSON 실패 {scan_result['failed']}개"
            )
    if not args.database.is_file():
        raise ValueError("품질 감사 DB가 없습니다. --scan-metadata가 필요합니다")

    manifest_path = args.output_dir / "visual_audit_manifest.csv"
    visual_count = 0
    contact_sheet_count = 0
    if args.visual_audit:
        selected = select_visual_samples(
            args.database,
            max_per_combination=args.max_visual_per_combination,
            seed=args.seed,
        )
        manifest = visual_manifest_rows(
            selected,
            refs,
            args.visual_dir,
            seed=args.seed,
        )
        write_csv(
            manifest_path,
            (
                "sample_id",
                "official_split",
                "species",
                "task",
                "disease_type",
                "normality",
                "label_archive_id",
                "source_archive_id",
                "json_member",
                "image_member",
                "original_path",
                "bbox_overlay_path",
                "segmentation_overlay_path",
                "image_width",
                "image_height",
                "bbox_count",
                "segmentation_present",
                "polygon_count",
                "automatic_issues",
                "status",
                "visual_review_status",
                "visual_review_note",
            ),
            manifest,
        )
        contact_sheets = create_contact_sheets(manifest, args.visual_dir)
        visual_count = sum(
            row.get("status") == "generated" for row in manifest
        )
        contact_sheet_count = len(contact_sheets)

    report_result = write_quality_reports(
        args.database,
        args.output_dir,
        seed=args.seed,
        manifest_path=manifest_path,
    )
    print(
        json.dumps(
            {
                "metadata_scan": scan_result,
                "visual_audit_images": visual_count,
                "contact_sheets": contact_sheet_count,
                "manifest_rows": report_result["manifest_count"],
                "recommended_group_key": report_result["recommended"][
                    "group_columns"
                ],
                "output_dir": base.portable_output_path(args.output_dir),
                "visual_dir": base.portable_output_path(args.visual_dir),
                "source_zip_modified": False,
                "model_training": False,
                "yolo_conversion": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args, os.environ)


if __name__ == "__main__":
    sys.exit(main())
