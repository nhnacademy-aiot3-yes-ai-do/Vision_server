from __future__ import annotations

import csv
import io
import importlib.util
import json
import sqlite3
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = PROJECT_ROOT / "scripts" / "inspect_aihub_archives.py"
AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "audit_mushroom_labels.py"


def load_module(name: str, path: Path) -> object:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


base = load_module("inspect_aihub_archives", BASE_SCRIPT)
audit = load_module("audit_mushroom_labels", AUDIT_SCRIPT)


def write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def label_json(
    image_filename: str,
    *,
    species: str = "느타리",
    normal: bool = True,
    disease: str | None = None,
    camera: int = 1,
    capture_date: str = "2021-11-01",
    capture_time: str = "12:00:00",
    annotations: list[dict[str, object]] | None = None,
) -> bytes:
    payload = {
        "INFO": {"CATEGORY_NAME": species},
        "IMAGE": {
            "IMAGE_FILE_NAME": image_filename,
            "WIDTH": 100,
            "HEIGHT": 80,
        },
        "ANNOTATION_INFO": annotations
        if annotations is not None
        else [
            {
                "BOUNDING_BOX_X_COORDINATE": 10,
                "BOUNDING_BOX_Y_COORDINATE": 10,
                "BOUNDING_BOX_WIDTH": 30,
                "BOUNDING_BOX_HEIGHT": 20,
                "SEGMENTATION": None,
            }
        ],
        "META": {
            "DBYHS_NORMALITY_ALTERNATIVE": normal,
            "DBYHS_SPCHCKN": disease,
            "IP_CAMERA_ID": camera,
            "IMAGE_CREATE_DATE": capture_date,
            "IMAGE_CREATE_TIME": capture_time,
        },
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def archive_ref(
    path: Path,
    *,
    split: str = "train",
    archive_id: str = "TL1",
) -> object:
    return base.ArchiveRef(
        env_name=(
            "TRAIN_LABEL_ARCHIVES"
            if split == "train"
            else "VAL_LABEL_ARCHIVES"
        ),
        split=split,
        kind="label",
        prefix=archive_id[:2],
        index=1,
        species="느타리",
        path=path,
    )


def image_archive_ref(path: Path, *, split: str = "train") -> object:
    return base.ArchiveRef(
        env_name=(
            "TRAIN_IMAGE_ARCHIVES"
            if split == "train"
            else "VAL_IMAGE_ARCHIVES"
        ),
        split=split,
        kind="image",
        prefix="TS" if split == "train" else "VS",
        index=1,
        species="느타리",
        path=path,
    )


def scan_synthetic_database(tmp_path: Path, count: int = 25) -> Path:
    label_path = tmp_path / "TL1_느타리.zip"
    members = {}
    for index in range(count):
        is_disease = index % 5 == 0
        folder = "병해" if is_disease else "생육"
        members[f"{folder}/{index:03d}.json"] = label_json(
            f"{index:03d}.jpg",
            normal=not is_disease,
            disease="테스트병" if is_disease else None,
            camera=(index % 3) + 1,
            capture_date=f"2021-11-{(index % 4) + 1:02d}",
            capture_time=f"12:{index % 5:02d}:00",
        )
    write_zip(label_path, members)
    database = tmp_path / "artifacts" / "quality.sqlite3"
    result = audit.scan_label_quality_metadata(
        [archive_ref(label_path)],
        database,
        show_progress=False,
        commit_interval=3,
    )
    assert result["processed"] == count
    assert result["failed"] == 0
    return database


def test_bbox_range_validation() -> None:
    assert audit.validate_bbox((0, 0, 100, 80), 100, 80) == {
        "bbox_near_full"
    }
    assert audit.validate_bbox((-1, 0, 20, 20), 100, 80) == {
        "bbox_out_of_bounds"
    }
    assert audit.validate_bbox((10, 10, 0, 20), 100, 80) == {
        "bbox_nonpositive"
    }
    assert audit.validate_bbox((10, 10, 1, 1), 100, 80) == {
        "bbox_tiny"
    }


def test_segmentation_coordinate_validation() -> None:
    inside = [0, 0, 99, 0, 99, 79, 0, 79]
    outside = [0, 0, 101, 0, 99, 79]
    assert audit.validate_segmentation(inside, 100, 80) == (0, 0)
    assert audit.validate_segmentation(outside, 100, 80) == (1, 0)
    assert audit.validate_segmentation([0, 0, 1], 100, 80) == (0, 1)
    assert audit.validate_segmentation(None, 100, 80) == (0, 0)


def test_group_assignment_has_no_overlap_and_is_seed_deterministic() -> None:
    groups = {
        (f"species-{index % 2}", f"group-{index}"): audit.SplitGroup(
            key=(f"species-{index % 2}", f"group-{index}"),
            total=index + 1,
            features=Counter(
                {
                    f"species:{'A' if index % 2 else 'B'}": index + 1,
                    f"task:{'생육' if index % 3 else '병해'}": index + 1,
                }
            ),
        )
        for index in range(30)
    }
    first, _ = audit.deterministic_group_assignment(
        groups, seed=1234, stratify_by_first_key=True
    )
    second, _ = audit.deterministic_group_assignment(
        groups, seed=1234, stratify_by_first_key=True
    )
    assert first == second
    split_keys = {
        split: {key for key, assigned in first.items() if assigned == split}
        for split in audit.SPLIT_RATIOS
    }
    assert set.union(*split_keys.values()) == set(groups)
    assert not (
        split_keys["train"] & split_keys["validation"]
        or split_keys["train"] & split_keys["test"]
        or split_keys["validation"] & split_keys["test"]
    )


def test_visual_sample_limit_is_respected(tmp_path: Path) -> None:
    database = scan_synthetic_database(tmp_path)
    selected = audit.select_visual_samples(
        database, max_per_combination=3, seed=777
    )
    counts = Counter((row["species"], row["task"]) for row in selected)
    assert counts
    assert max(counts.values()) <= 3
    with pytest.raises(ValueError):
        audit.select_visual_samples(
            database, max_per_combination=21, seed=777
        )


def test_reports_do_not_expose_absolute_paths(tmp_path: Path) -> None:
    database = scan_synthetic_database(tmp_path)
    reports = tmp_path / "reports"
    audit.write_quality_reports(
        database,
        reports,
        seed=123,
        manifest_path=reports / "visual_audit_manifest.csv",
        manual_review_path=tmp_path / "missing-review.json",
    )
    combined = "\n".join(
        path.read_text(encoding="utf-8-sig")
        for path in reports.iterdir()
    )
    assert str(tmp_path) not in combined
    assert "/mnt/" not in combined
    assert "/home/" not in combined


def test_source_zip_is_unchanged_by_metadata_scan(tmp_path: Path) -> None:
    label_path = tmp_path / "TL1_느타리.zip"
    write_zip(label_path, {"생육/sample.json": label_json("sample.jpg")})
    before = audit.source_zip_snapshot([label_path])
    database = tmp_path / "artifacts" / "quality.sqlite3"
    audit.scan_label_quality_metadata(
        [archive_ref(label_path)],
        database,
        show_progress=False,
        commit_interval=1,
    )
    audit.assert_source_zips_unchanged(before)
    assert audit.source_zip_snapshot([label_path]) == before


def test_limited_visual_audit_keeps_label_and_image_zips_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    label_path = tmp_path / "TL1_느타리.zip"
    image_path = tmp_path / "TS1_느타리.zip"
    image_buffer = io.BytesIO()
    Image.new("RGB", (100, 80), "white").save(
        image_buffer, format="JPEG"
    )
    write_zip(
        label_path,
        {"생육/sample.json": label_json("sample.jpg")},
    )
    write_zip(
        image_path,
        {"생육/sample.jpg": image_buffer.getvalue()},
    )
    label = archive_ref(label_path)
    image = image_archive_ref(image_path)
    database = tmp_path / "quality.sqlite3"
    audit.scan_label_quality_metadata(
        [label], database, show_progress=False
    )
    selected = audit.select_visual_samples(
        database, max_per_combination=1, seed=9
    )
    before = audit.source_zip_snapshot([label_path, image_path])
    manifest = audit.visual_manifest_rows(
        selected,
        [label, image],
        tmp_path / "artifacts" / "visual_audit",
        seed=9,
    )
    assert len(manifest) == 1
    assert manifest[0]["status"] == "generated"
    assert not Path(manifest[0]["original_path"]).is_absolute()
    assert audit.source_zip_snapshot([label_path, image_path]) == before


def test_bbox_and_annotation_aggregation_handles_empty_annotations(
    tmp_path: Path,
) -> None:
    label_path = tmp_path / "TL1_느타리.zip"
    write_zip(
        label_path,
        {
            "생육/with.json": label_json("with.jpg"),
            "생육/empty.json": label_json("empty.jpg", annotations=[]),
        },
    )
    database = tmp_path / "quality.sqlite3"
    audit.scan_label_quality_metadata(
        [archive_ref(label_path)],
        database,
        show_progress=False,
    )
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT annotation_count, bbox_count, bbox_missing_reason "
            "FROM quality_records ORDER BY annotation_count"
        ).fetchall()
    assert rows == [(0, 0, "no_annotations"), (1, 1, "")]


def test_csv_writer_keeps_member_paths_relative(tmp_path: Path) -> None:
    output = tmp_path / "manifest.csv"
    audit.write_csv(
        output,
        ("json_member", "original_path"),
        [
            {
                "json_member": "생육/a.json",
                "original_path": "artifacts/visual_audit/original/a.jpg",
            }
        ],
    )
    with output.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["json_member"] == "생육/a.json"
    assert not Path(row["original_path"]).is_absolute()
