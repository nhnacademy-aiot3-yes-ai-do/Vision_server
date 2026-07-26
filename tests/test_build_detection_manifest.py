from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import zipfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


base = load_module(
    "inspect_aihub_archives",
    PROJECT_ROOT / "scripts" / "inspect_aihub_archives.py",
)
audit = load_module(
    "audit_mushroom_labels",
    PROJECT_ROOT / "scripts" / "audit_mushroom_labels.py",
)
builder = load_module(
    "build_detection_manifest",
    PROJECT_ROOT / "scripts" / "build_detection_manifest.py",
)


def annotation(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    segmentation: object = None,
) -> dict[str, object]:
    return {
        "BOUNDING_BOX_X_COORDINATE": x,
        "BOUNDING_BOX_Y_COORDINATE": y,
        "BOUNDING_BOX_WIDTH": width,
        "BOUNDING_BOX_HEIGHT": height,
        "SEGMENTATION": segmentation,
    }


def label_json(
    image_name: str,
    annotations: list[dict[str, object]],
    *,
    species: str = "느타리",
    camera: int = 1,
    date: str = "2021-11-01",
    time: str = "12:00:00",
    normal: bool = True,
    disease: str | None = None,
) -> bytes:
    return json.dumps(
        {
            "INFO": {"CATEGORY_NAME": species},
            "IMAGE": {
                "IMAGE_FILE_NAME": image_name,
                "WIDTH": 100,
                "HEIGHT": 80,
            },
            "ANNOTATION_INFO": annotations,
            "META": {
                "DBYHS_NORMALITY_ALTERNATIVE": normal,
                "DBYHS_SPCHCKN": disease,
                "IP_CAMERA_ID": camera,
                "IMAGE_CREATE_DATE": date,
                "IMAGE_CREATE_TIME": time,
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")


def write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for member, content in members.items():
            archive.writestr(member, content)


def archive_ref(
    path: Path,
    *,
    kind: str,
    split: str = "train",
    species: str = "느타리",
) -> object:
    prefixes = {
        ("train", "label"): "TL",
        ("train", "image"): "TS",
        ("validation", "label"): "VL",
        ("validation", "image"): "VS",
    }
    prefix = prefixes[(split, kind)]
    return base.ArchiveRef(
        env_name=f"{split}_{kind}",
        split=split,
        kind=kind,
        prefix=prefix,
        index=1,
        species=species,
        path=path,
    )


def synthetic_pair(tmp_path: Path) -> tuple[object, object, Path, Path]:
    label_path = tmp_path / "TL1_느타리.zip"
    image_path = tmp_path / "TS1_느타리.zip"
    labels = {
        "생육/valid.json": label_json(
            "valid.jpg",
            [
                annotation(10, 10, 20, 20),
                annotation(-5, 70, 20, 20),
                annotation(20, 20, 1, 1),
                annotation(30, 30, 0, 4),
            ],
            camera=1,
            date="2021-11-01",
        ),
        "생육/all_removed.json": label_json(
            "all_removed.jpg",
            [annotation(10, 10, 0, 0), annotation(120, 10, 5, 5)],
            camera=1,
            date="2021-11-02",
        ),
        "병해/disease.json": label_json(
            "disease.jpg",
            [annotation(5, 5, 20, 30)],
            camera=2,
            date="2021-11-03",
            normal=False,
            disease="테스트병",
        ),
        "배양/culture.json": b"not read by the manifest builder",
    }
    images = {
        "생육/valid.jpg": b"IMAGE_BYTES_MUST_NOT_BE_READ",
        "생육/all_removed.jpg": b"IMAGE_BYTES_MUST_NOT_BE_READ",
        "병해/disease.jpg": b"IMAGE_BYTES_MUST_NOT_BE_READ",
        "배양/culture.jpg": b"IMAGE_BYTES_MUST_NOT_BE_READ",
    }
    write_zip(label_path, labels)
    write_zip(image_path, images)
    return (
        archive_ref(label_path, kind="label"),
        archive_ref(image_path, kind="image"),
        label_path,
        image_path,
    )


def test_bbox_cleanup_zero_clip_postclip_and_small_policy() -> None:
    result = builder.clean_annotations(
        [
            annotation(10, 10, 0, 2),
            annotation(-5, 70, 20, 20),
            annotation(110, 10, 5, 5),
            annotation(20, 20, 1, 1),
        ],
        100,
        80,
    )
    assert [box["annotation_index"] for box in result.cleaned] == [1, 3]
    assert result.cleaned[0] == {
        "annotation_index": 1,
        "x": 0,
        "y": 70,
        "width": 15,
        "height": 10,
    }
    assert result.counts["removed_nonpositive_bbox"] == 1
    assert result.counts["removed_after_clip_lt_1px"] == 1
    assert result.counts["clipped_to_image_bounds"] == 2
    assert result.counts["retained_small_bbox"] == 1


def test_scan_excludes_all_removed_and_never_reads_image_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    label_ref, image_ref, _label_path, image_path = synthetic_pair(tmp_path)
    original_read = zipfile.ZipFile.read

    def guarded_read(self: zipfile.ZipFile, name: object, *args: object, **kwargs: object) -> bytes:
        assert Path(self.filename).resolve() != image_path.resolve()
        return original_read(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", guarded_read)
    database = tmp_path / "artifacts" / "manifest.sqlite3"
    result = builder.scan_detection_records(
        [(label_ref, image_ref)],
        database,
        show_progress=False,
        commit_interval=1,
    )
    assert result["included"] == 2
    connection = sqlite3.connect(database)
    try:
        reasons = dict(
            connection.execute(
                "SELECT exclusion_reason, COUNT(*) FROM excluded_records "
                "GROUP BY exclusion_reason"
            )
        )
        assert reasons == {
            "all_bboxes_removed": 1,
            "task_excluded_culture": 1,
        }
        assert connection.execute(
            "SELECT SUM(valid_bbox_count) FROM detection_records"
        ).fetchone()[0] == 4
    finally:
        connection.close()


def test_group_split_is_leak_free_reproducible_and_images_unique(
    tmp_path: Path,
) -> None:
    label_ref, image_ref, _label_path, _image_path = synthetic_pair(tmp_path)
    database = tmp_path / "manifest.sqlite3"
    builder.scan_detection_records(
        [(label_ref, image_ref)], database, show_progress=False
    )
    connection = audit.sql_connection(database)
    try:
        builder.assign_splits(connection, seed=20260726)
        first = dict(
            connection.execute(
                "SELECT group_key, new_split FROM detection_records"
            )
        )
        builder.assign_splits(connection, seed=20260726)
        second = dict(
            connection.execute(
                "SELECT group_key, new_split FROM detection_records"
            )
        )
        assert first == second
        validation = builder.validate_database(connection)
        assert validation["group_split_leakage"] == 0
        assert validation["image_split_leakage"] == 0
    finally:
        connection.close()


def test_reports_do_not_expose_local_absolute_paths(tmp_path: Path) -> None:
    label_ref, image_ref, _label_path, _image_path = synthetic_pair(tmp_path)
    database = tmp_path / "manifest.sqlite3"
    output_dir = tmp_path / "reports"
    builder.scan_detection_records(
        [(label_ref, image_ref)], database, show_progress=False
    )
    builder.write_reports(database, output_dir, seed=20260726)
    for path in output_dir.iterdir():
        text = path.read_text(encoding="utf-8-sig")
        assert str(tmp_path) not in text
        assert "/mnt/" not in text
        assert "/home/" not in text
    manifest = (output_dir / "detection_dataset_manifest.csv").read_text(
        encoding="utf-8-sig"
    )
    assert "생육/valid.json" in manifest
    assert "생육/valid.jpg" in manifest


def test_source_zip_size_and_mtime_are_unchanged(tmp_path: Path) -> None:
    label_ref, image_ref, label_path, image_path = synthetic_pair(tmp_path)
    before = audit.source_zip_snapshot((label_path, image_path))
    builder.scan_detection_records(
        [(label_ref, image_ref)],
        tmp_path / "manifest.sqlite3",
        show_progress=False,
    )
    after = audit.source_zip_snapshot((label_path, image_path))
    assert before == after


def test_duplicate_image_across_splits_is_detected(tmp_path: Path) -> None:
    database = tmp_path / "minimal.sqlite3"
    connection = builder.initialize_database(database)
    row = (
        "record-a",
        None,
        '["느타리","1","2021-01-01"]',
        "느타리",
        0,
        "생육",
        "normal",
        "<missing>",
        "1",
        "2021-01-01",
        "12:00:00",
        "TL1",
        "TS1",
        "생육/a.json",
        "생육/shared.jpg",
        "생육/shared.jpg",
        100,
        80,
        1,
        1,
        "[]",
        '[{"annotation_index":0,"x":1,"y":1,"width":2,"height":2}]',
        "[]",
        0,
        "train",
        0,
        0,
        0,
        0,
        0,
    )
    connection.execute(
        "INSERT INTO detection_records VALUES("
        + ",".join("?" for _ in row)
        + ")",
        row,
    )
    other = list(row)
    other[0] = "record-b"
    other[2] = '["느타리","2","2021-01-02"]'
    other[13] = "생육/b.json"
    other[1] = "test"
    connection.execute(
        "INSERT INTO detection_records VALUES("
        + ",".join("?" for _ in other)
        + ")",
        other,
    )
    connection.execute(
        "UPDATE detection_records SET new_split='train' WHERE record_key='record-a'"
    )
    connection.commit()
    with pytest.raises(RuntimeError, match="manifest 검증 실패"):
        builder.validate_database(connection)
    connection.close()
