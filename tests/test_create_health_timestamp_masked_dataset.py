from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image, JpegImagePlugin


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


load_module(
    "inspect_aihub_archives",
    PROJECT_ROOT / "scripts" / "inspect_aihub_archives.py",
)
load_module(
    "audit_mushroom_labels",
    PROJECT_ROOT / "scripts" / "audit_mushroom_labels.py",
)
load_module(
    "build_detection_manifest",
    PROJECT_ROOT / "scripts" / "build_detection_manifest.py",
)
smoke = load_module(
    "create_yolo_smoke_dataset",
    PROJECT_ROOT / "scripts" / "create_yolo_smoke_dataset.py",
)
load_module(
    "create_yolo_pilot_dataset",
    PROJECT_ROOT / "scripts" / "create_yolo_pilot_dataset.py",
)
load_module(
    "create_yolo_camera_holdout_pilot",
    PROJECT_ROOT / "scripts" / "create_yolo_camera_holdout_pilot.py",
)
load_module(
    "audit_mushroom_health",
    PROJECT_ROOT / "scripts" / "audit_mushroom_health.py",
)
raw_health = load_module(
    "create_health_pilot_dataset",
    PROJECT_ROOT / "scripts" / "create_health_pilot_dataset.py",
)
masked = load_module(
    "create_health_timestamp_masked_dataset",
    PROJECT_ROOT / "scripts" / "create_health_timestamp_masked_dataset.py",
)


ORIGINAL_SIZE = (1000, 500)
CROP_SIZE = (200, 100)
OVERLAP_CROP = {"x": 550, "y": 20, "width": 200, "height": 100}
NON_OVERLAP_CROP = {
    "x": 100,
    "y": 100,
    "width": 200,
    "height": 100,
}


def raw_row(
    *,
    species: str,
    split: str,
    health_class_id: int,
    camera_id: str,
    index: int,
    overlaps: bool,
) -> dict[str, str]:
    class_name, class_directory = raw_health.HEALTH_CLASSES[health_class_id]
    split_directory = "train" if split == "train" else "val"
    species_id = smoke.CLASS_NAMES_INV[species]
    filename = (
        f"{split_directory}_{species_id}_{health_class_id}_{index:03d}.jpg"
    )
    task = "생육" if health_class_id == 0 else "병해"
    crop = OVERLAP_CROP if overlaps else NON_OVERLAP_CROP
    return {
        "split": split,
        "health_class_id": str(health_class_id),
        "health_class_name": class_name,
        "species": species,
        "task": task,
        "normality": "normal" if health_class_id == 0 else "abnormal",
        "disease_type": (
            "<missing>" if health_class_id == 0 else "푸른곰팡이병"
        ),
        "camera_id": camera_id,
        "capture_date": "2021-11-01",
        "capture_time": f"12:{index % 60:02d}:00",
        "image_archive_id": f"TS{species_id + 1}",
        "image_member": f"{task}/source_{species_id}_{index:03d}.jpg",
        "original_width": str(ORIGINAL_SIZE[0]),
        "original_height": str(ORIGINAL_SIZE[1]),
        "valid_bbox_count": "1",
        "union_bbox": json.dumps(
            {"x": 570, "y": 40, "width": 100, "height": 40},
            separators=(",", ":"),
        ),
        "padded_crop_bbox": json.dumps(crop, separators=(",", ":")),
        "padding_ratio": "0.15",
        "crop_width": str(CROP_SIZE[0]),
        "crop_height": str(CROP_SIZE[1]),
        "timestamp_region_overlap": str(overlaps),
        "output_relative_path": (
            f"{split_directory}/{class_directory}/{filename}"
        ),
    }


def save_source_jpeg(
    path: Path,
    *,
    color: tuple[int, int, int],
    quality: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", CROP_SIZE, color).save(
        path,
        format="JPEG",
        quality=quality,
        subsampling=2,
        optimize=False,
        progressive=False,
    )


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_health.MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_class_mapping(source_dir: Path) -> None:
    payload = {
        "schema_version": 1,
        "task": "mushroom_health_binary_classification",
        "classes": {
            str(class_id): {"name": name, "directory": directory}
            for class_id, (name, directory) in raw_health.HEALTH_CLASSES.items()
        },
    }
    (source_dir / "class_mapping.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def file_tree_fingerprint(root: Path) -> dict[str, tuple[int, int, str]]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def build_raw_fixture(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    Path,
    list[dict[str, str]],
    list[object],
]:
    artifacts = tmp_path / "artifacts"
    source_dir = artifacts / "health_pilot"
    source_dir.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    index = 0
    for split in ("train", "validation"):
        for species_index, species in enumerate(raw_health.pilot.SPECIES):
            for class_id in raw_health.HEALTH_CLASSES:
                row = raw_row(
                    species=species,
                    split=split,
                    health_class_id=class_id,
                    camera_id=(
                        f"train-{species_index}"
                        if split == "train"
                        else f"val-{species_index}"
                    ),
                    index=index,
                    overlaps=class_id == 1,
                )
                color = (230, 30, 30) if class_id == 1 else (30, 230, 30)
                save_source_jpeg(
                    source_dir / row["output_relative_path"],
                    color=color,
                    quality=35 + index % 4 * 10,
                )
                rows.append(row)
                index += 1
    write_class_mapping(source_dir)
    model_sentinel = (
        source_dir
        / "training_runs"
        / "raw_health_model"
        / "weights"
        / "best.pt"
    )
    model_sentinel.parent.mkdir(parents=True)
    model_sentinel.write_bytes(b"synthetic-model-must-not-change")
    manifest = source_dir / "health_pilot_manifest.csv"
    write_manifest(manifest, rows)
    records = masked.load_source_records(
        manifest,
        source_dir,
        expected_images=len(rows),
    )
    return artifacts, source_dir, manifest, rows, records


def test_roi_intersection_is_transformed_to_crop_coordinates_and_clipped() -> None:
    geometry = masked.timestamp_mask_geometry(
        original_width=1000,
        original_height=500,
        crop_rect_original=(550, 20, 750, 120),
    )
    assert geometry.original_timestamp_rect == (600, 0, 1000, 60)
    assert geometry.crop_rect_original == (550, 20, 750, 120)
    assert geometry.mask_rect_crop == (50, 0, 200, 40)
    assert (geometry.crop_width, geometry.crop_height) == (200, 100)
    assert geometry.mask_area_ratio == pytest.approx(0.30)

    clipped = masked.timestamp_mask_geometry(
        original_width=1000,
        original_height=500,
        crop_rect_original=(900, 0, 1000, 50),
    )
    assert clipped.mask_rect_crop == (0, 0, 100, 50)

    odd_size = masked.timestamp_mask_geometry(
        original_width=101,
        original_height=83,
        crop_rect_original=(0, 0, 101, 83),
    )
    assert odd_size.original_timestamp_rect == (60, 0, 101, 10)
    assert odd_size.mask_rect_crop == (60, 0, 101, 10)


def test_non_overlap_and_boundary_touch_do_not_apply_a_mask() -> None:
    for crop in (
        (100, 100, 300, 200),
        (400, 60, 700, 160),
        (400, 100, 600, 200),
    ):
        geometry = masked.timestamp_mask_geometry(
            original_width=1000,
            original_height=500,
            crop_rect_original=crop,
        )
        assert not geometry.overlaps
        assert geometry.mask_rect_crop is None
        assert geometry.mask_width == 0
        assert geometry.mask_height == 0
        image = Image.new(
            "RGB",
            (geometry.crop_width, geometry.crop_height),
            (20, 220, 40),
        )
        result = masked.apply_timestamp_mask(
            image, geometry, mask_color=127
        )
        assert result.getpixel((10, 10)) == (20, 220, 40)


def test_mask_pixels_jpeg_policy_and_crop_size_are_deterministic(
    tmp_path: Path,
) -> None:
    geometry = masked.timestamp_mask_geometry(
        original_width=1000,
        original_height=500,
        crop_rect_original=(550, 20, 750, 120),
    )
    source = Image.new("RGB", CROP_SIZE, (230, 30, 30))
    result = masked.apply_timestamp_mask(source, geometry, mask_color=127)
    assert result.size == CROP_SIZE
    assert result.getpixel((100, 20)) == (127, 127, 127)
    assert result.getpixel((20, 80)) == (230, 30, 30)

    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    masked.save_uniform_jpeg(result, first, jpeg_quality=95)
    masked.save_uniform_jpeg(result, second, jpeg_quality=95)
    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as opened:
        opened.load()
        assert opened.format == "JPEG"
        assert opened.mode == "RGB"
        assert opened.size == CROP_SIZE
        assert JpegImagePlugin.get_sampling(opened) == 0
        center = opened.convert("RGB").getpixel((100, 20))
    assert all(abs(channel - 127) <= 5 for channel in center)

    wrong_size = Image.new("RGB", (199, 100))
    with pytest.raises(ValueError, match="실제 크기"):
        masked.apply_timestamp_mask(
            wrong_size, geometry, mask_color=127
        )


def test_review_selection_is_seeded_and_has_a_global_100_pair_limit() -> None:
    records: list[object] = []
    index = 0
    for split in ("train", "validation"):
        for species_index, species in enumerate(raw_health.pilot.SPECIES):
            for class_id in raw_health.HEALTH_CLASSES:
                for candidate in range(6):
                    row = raw_row(
                        species=species,
                        split=split,
                        health_class_id=class_id,
                        camera_id=f"{split}-{species_index}",
                        index=index,
                        overlaps=candidate % 2 == 0,
                    )
                    geometry = masked.timestamp_mask_geometry(
                        original_width=ORIGINAL_SIZE[0],
                        original_height=ORIGINAL_SIZE[1],
                        crop_rect_original=(
                            550,
                            20,
                            750,
                            120,
                        )
                        if candidate % 2 == 0
                        else (100, 100, 300, 200),
                    )
                    records.append(
                        masked.SourceRecord(
                            row=row,
                            source_path=Path(row["output_relative_path"]),
                            unique_key=masked.raw_unique_key(row),
                            geometry=geometry,
                        )
                    )
                    index += 1
    first = masked.select_review_records(
        records, max_review_pairs=100, seed=20260726
    )
    second = masked.select_review_records(
        records, max_review_pairs=100, seed=20260726
    )
    assert [record.unique_key for record in first] == [
        record.unique_key for record in second
    ]
    assert len(first) == 100
    assert set(Counter(masked.review_group(record) for record in first).values()) == {
        5
    }
    assert any(record.geometry.overlaps for record in first)
    assert any(not record.geometry.overlaps for record in first)
    with pytest.raises(ValueError, match="100"):
        masked.select_review_records(
            records, max_review_pairs=101, seed=20260726
        )


def test_test_split_is_rejected_before_any_output_is_created(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    source_dir = artifacts / "health_pilot"
    source_dir.mkdir(parents=True)
    row = raw_row(
        species="느타리",
        split="test",
        health_class_id=0,
        camera_id="99",
        index=1,
        overlaps=False,
    )
    manifest = source_dir / "health_pilot_manifest.csv"
    write_manifest(manifest, [row])
    source_before = file_tree_fingerprint(source_dir)
    with pytest.raises(ValueError, match="Test"):
        masked.load_source_records(
            manifest, source_dir, expected_images=1
        )
    assert file_tree_fingerprint(source_dir) == source_before
    assert not (artifacts / "health_pilot_timestamp_masked").exists()


def test_end_to_end_manifest_is_one_to_one_and_raw_artifacts_are_unchanged(
    tmp_path: Path,
) -> None:
    artifacts, source_dir, manifest, raw_rows, records = build_raw_fixture(
        tmp_path
    )
    output = artifacts / "health_pilot_timestamp_masked"
    for name in (
        "yolo_smoke",
        "yolo_pilot",
        "yolo_pilot_camera_holdout",
        "models",
        "unrelated",
    ):
        sentinel = artifacts / name / "nested" / "sentinel.txt"
        sentinel.parent.mkdir(parents=True)
        sentinel.write_text(name, encoding="utf-8")
    raw_before = file_tree_fingerprint(source_dir)
    artifacts_before = raw_health.snapshot_artifacts_excluding(
        artifacts, output
    )

    result = masked.create_masked_dataset(
        records,
        source_dir,
        manifest,
        output,
        mask_color=127,
        jpeg_quality=95,
        max_review_pairs=20,
        seed=20260726,
        overwrite=False,
        artifacts_root=artifacts,
        show_progress=False,
        expected_images=len(records),
    )

    assert result["total_images"] == 20
    assert result["train_images"] == 10
    assert result["validation_images"] == 10
    assert result["masked_images"] == 10
    assert result["review_pairs"] == 20
    assert result["review_contact_sheets"] == 20
    assert result["test_images"] == 0
    assert result["missing_images"] == 0
    assert result["crop_size_changes"] == 0
    assert not result["source_raw_modified"]
    assert not result["protected_artifacts_modified"]
    assert file_tree_fingerprint(source_dir) == raw_before
    assert (
        raw_health.snapshot_artifacts_excluding(artifacts, output)
        == artifacts_before
    )

    with (output / "health_timestamp_masked_manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        output_rows = list(csv.DictReader(handle))
    assert len(output_rows) == len(raw_rows)
    assert {
        masked.raw_unique_key(row) for row in output_rows
    } == {
        masked.raw_unique_key(row) for row in raw_rows
    }
    for source_row, output_row in zip(raw_rows, output_rows):
        assert all(
            source_row[column] == output_row[column]
            for column in raw_health.MANIFEST_COLUMNS
        )
        assert (
            output_row["source_raw_relative_path"]
            == source_row["output_relative_path"]
        )
        expected_overlap = source_row["health_class_id"] == "1"
        assert masked.boolean_text(
            output_row["timestamp_mask_applied"]
        ) is expected_overlap
        expected_rect = (
            {"x": 50, "y": 0, "width": 150, "height": 40}
            if expected_overlap
            else None
        )
        assert json.loads(
            output_row["mask_rect_crop_coordinates"]
        ) == expected_rect

    output_images = [
        output / row["output_relative_path"] for row in output_rows
    ]
    assert all(path.is_file() for path in output_images)
    signatures = set()
    for row, path in zip(output_rows, output_images):
        with Image.open(path) as opened:
            opened.load()
            assert opened.format == "JPEG"
            assert opened.mode == "RGB"
            assert opened.size == (
                int(row["crop_width"]),
                int(row["crop_height"]),
            )
            assert JpegImagePlugin.get_sampling(opened) == 0
            signatures.add(masked.jpeg_quantization_signature(opened))
    assert len(signatures) == 1

    masked_row = next(
        row for row in output_rows if row["timestamp_mask_applied"] == "True"
    )
    with Image.open(output / masked_row["output_relative_path"]) as opened:
        masked_pixel = opened.convert("RGB").getpixel((100, 20))
        outside_pixel = opened.convert("RGB").getpixel((20, 80))
    assert all(abs(channel - 127) <= 5 for channel in masked_pixel)
    assert outside_pixel[0] > 200 and outside_pixel[1] < 70

    unmasked_row = next(
        row for row in output_rows if row["timestamp_mask_applied"] == "False"
    )
    raw_unmasked_path = source_dir / unmasked_row["output_relative_path"]
    output_unmasked_path = output / unmasked_row["output_relative_path"]
    with Image.open(raw_unmasked_path) as before, Image.open(
        output_unmasked_path
    ) as after:
        before.load()
        after.load()
        assert masked.jpeg_quantization_signature(before) != (
            masked.jpeg_quantization_signature(after)
        )
        unmasked_pixel = after.convert("RGB").getpixel((100, 50))
    assert unmasked_pixel[1] > 200 and unmasked_pixel[0] < 70

    sheets = list((output / "review" / "contact_sheets").glob("*.jpg"))
    assert len(sheets) == 20
    for sheet in sheets:
        with Image.open(sheet) as opened:
            assert opened.size == (720, 324)

    text_suffixes = {".csv", ".json", ".md", ".txt", ".yaml"}
    for path in output.rglob("*"):
        if path.is_file() and path.suffix.lower() in text_suffixes:
            text = path.read_text(encoding="utf-8-sig")
            assert str(tmp_path) not in text
            assert "/tmp/" not in text
            assert "/mnt/" not in text
            assert "/home/" not in text
            assert ":\\\\" not in text


def test_empty_source_image_fails_without_leaving_output(
    tmp_path: Path,
) -> None:
    artifacts, source_dir, manifest, _rows, records = build_raw_fixture(
        tmp_path
    )
    records[0].source_path.write_bytes(b"")
    source_before = file_tree_fingerprint(source_dir)
    output = artifacts / "health_pilot_timestamp_masked"

    with pytest.raises(ValueError, match="손상"):
        masked.create_masked_dataset(
            records,
            source_dir,
            manifest,
            output,
            mask_color=127,
            jpeg_quality=95,
            max_review_pairs=20,
            seed=20260726,
            overwrite=False,
            artifacts_root=artifacts,
            show_progress=False,
            expected_images=len(records),
        )

    assert not output.exists()
    assert not list(artifacts.glob(".health_pilot_timestamp_masked.tmp-*"))
    assert file_tree_fingerprint(source_dir) == source_before


def test_overwrite_restores_previous_output_when_post_commit_check_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts, source_dir, manifest, _rows, records = build_raw_fixture(
        tmp_path
    )
    output = artifacts / "health_pilot_timestamp_masked"
    output.mkdir()
    (output / "previous-result.txt").write_text(
        "preserve me", encoding="utf-8"
    )
    previous = file_tree_fingerprint(output)
    original_check = raw_health.assert_artifacts_unchanged
    call_count = 0

    def fail_second_check(*args: object, **kwargs: object) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("synthetic post-commit failure")
        original_check(*args, **kwargs)

    monkeypatch.setattr(
        raw_health, "assert_artifacts_unchanged", fail_second_check
    )
    with pytest.raises(RuntimeError, match="post-commit"):
        masked.create_masked_dataset(
            records,
            source_dir,
            manifest,
            output,
            mask_color=127,
            jpeg_quality=95,
            max_review_pairs=20,
            seed=20260726,
            overwrite=True,
            artifacts_root=artifacts,
            show_progress=False,
            expected_images=len(records),
        )

    assert file_tree_fingerprint(output) == previous
    assert not list(artifacts.glob(".health_pilot_timestamp_masked.tmp-*"))
    assert not list(
        artifacts.glob(".health_pilot_timestamp_masked.backup-*")
    )


def test_output_is_reproducible_for_the_same_seed(
    tmp_path: Path,
) -> None:
    artifacts, source_dir, manifest, _rows, records = build_raw_fixture(
        tmp_path
    )
    first_output = artifacts / "masked_first"
    second_output = artifacts / "masked_second"
    for output in (first_output, second_output):
        masked.create_masked_dataset(
            records,
            source_dir,
            manifest,
            output,
            mask_color=127,
            jpeg_quality=95,
            max_review_pairs=20,
            seed=20260726,
            overwrite=False,
            artifacts_root=artifacts,
            show_progress=False,
            expected_images=len(records),
        )

    def comparable_files(root: Path) -> dict[str, str]:
        return {
            path.relative_to(root).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in root.rglob("*")
            if path.is_file()
        }

    assert comparable_files(first_output) == comparable_files(second_output)
