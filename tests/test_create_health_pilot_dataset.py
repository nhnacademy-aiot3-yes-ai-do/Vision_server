from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image, ImageDraw


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
label_audit = load_module(
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
pilot = load_module(
    "create_yolo_pilot_dataset",
    PROJECT_ROOT / "scripts" / "create_yolo_pilot_dataset.py",
)
holdout = load_module(
    "create_yolo_camera_holdout_pilot",
    PROJECT_ROOT / "scripts" / "create_yolo_camera_holdout_pilot.py",
)
health = load_module(
    "audit_mushroom_health",
    PROJECT_ROOT / "scripts" / "audit_mushroom_health.py",
)
health_pilot = load_module(
    "create_health_pilot_dataset",
    PROJECT_ROOT / "scripts" / "create_health_pilot_dataset.py",
)


DEFAULT_BOXES = [
    {
        "annotation_index": 0,
        "x": 20,
        "y": 20,
        "width": 10,
        "height": 10,
    },
    {
        "annotation_index": 1,
        "x": 50,
        "y": 30,
        "width": 20,
        "height": 10,
    },
]


def source_image_bytes() -> bytes:
    image = Image.new("RGB", (100, 80), (40, 40, 60))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 29, 29), fill=(240, 20, 20))
    draw.rectangle((50, 30, 69, 39), fill=(20, 240, 20))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=98, subsampling=0)
    return buffer.getvalue()


def source_row(
    *,
    species: str,
    camera_id: str,
    task: str,
    index: int,
    split: str = "train",
    disease: str = "<missing>",
    boxes: list[dict[str, int]] | None = None,
) -> dict[str, str]:
    class_id = smoke.CLASS_NAMES_INV[species]
    member = (
        f"{task}/{split}_{class_id}_cam{camera_id}_{task}_{index:03d}.jpg"
    )
    cleaned = boxes or DEFAULT_BOXES
    return {
        "split": split,
        "group_key": f'["{species}","{camera_id}","2021-11-01"]',
        "species": species,
        "class_id": str(class_id),
        "task": task,
        "normality": "normal" if task == "생육" else "abnormal",
        "disease_type": disease,
        "camera_id": camera_id,
        "capture_date": "2021-11-01",
        "capture_time": "12:00:00",
        "label_archive_id": "TL1",
        "image_archive_id": "TS1",
        "json_member": str(Path(member).with_suffix(".json")),
        "image_member": member,
        "image_width": "100",
        "image_height": "80",
        "valid_bbox_count": str(len(cleaned)),
        "bbox_cleaned": json.dumps(cleaned, separators=(",", ":")),
        "official_split": "train",
    }


def synthetic_rows(
    *,
    cameras_per_species: int = 6,
    images_per_status_camera: int = 3,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for species in pilot.SPECIES:
        for camera_number in range(1, cameras_per_species + 1):
            camera_id = str(camera_number)
            for task in pilot.TASKS:
                for index in range(images_per_status_camera):
                    rows.append(
                        source_row(
                            species=species,
                            camera_id=camera_id,
                            task=task,
                            index=index,
                            split=(
                                "train"
                                if (camera_number + index) % 2
                                else "validation"
                            ),
                            disease=(
                                "병A"
                                if task == "병해" and camera_number <= 3
                                else "병B"
                                if task == "병해"
                                else "<missing>"
                            ),
                        )
                    )
    rows.append(
        source_row(
            species="느타리",
            camera_id="99",
            task="생육",
            index=999,
            split="test",
        )
    )
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(dict.fromkeys(pilot.PILOT_SOURCE_COLUMNS))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for member, content in members.items():
            archive.writestr(member, content)


def archive_ref(path: Path) -> object:
    return base.ArchiveRef(
        env_name="TRAIN_IMAGE_ARCHIVES",
        split="train",
        kind="image",
        prefix="TS",
        index=1,
        species="느타리",
        path=path,
    )


def selected_fixture(tmp_path: Path) -> tuple[Path, object]:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, synthetic_rows())
    selection = health_pilot.select_health_rows(
        manifest,
        train_per_species_status=2,
        validation_per_species_status=1,
        seed=20260726,
        max_total_images=30,
    )
    return manifest, selection


def test_union_bbox_padding_clip_and_multiple_boxes() -> None:
    row = source_row(
        species="느타리",
        camera_id="1",
        task="생육",
        index=1,
    )
    geometry = health_pilot.crop_geometry(row, padding_ratio=0.15)
    assert geometry.union_rect == pytest.approx((20, 20, 70, 40))
    assert geometry.crop_rect == (5, 8, 85, 52)
    assert (geometry.crop_width, geometry.crop_height) == (80, 44)

    edge = source_row(
        species="느타리",
        camera_id="1",
        task="생육",
        index=2,
        boxes=[{"x": 2, "y": 3, "width": 20, "height": 10}],
    )
    assert health_pilot.crop_geometry(
        edge, padding_ratio=0.15
    ).crop_rect == (0, 0, 37, 25)


def test_timestamp_region_positive_overlap_not_touching_boundary() -> None:
    overlapping = source_row(
        species="느타리",
        camera_id="1",
        task="생육",
        index=1,
        boxes=[{"x": 50, "y": 20, "width": 10, "height": 10}],
    )
    outside = source_row(
        species="느타리",
        camera_id="1",
        task="생육",
        index=2,
        boxes=[{"x": 0, "y": 40, "width": 10, "height": 10}],
    )
    touching = source_row(
        species="느타리",
        camera_id="1",
        task="생육",
        index=3,
        boxes=[{"x": 35, "y": 22, "width": 10, "height": 10}],
    )
    assert health_pilot.timestamp_candidate_rect(100, 80) == (
        60,
        0,
        100,
        10,
    )
    assert health_pilot.crop_geometry(
        overlapping, padding_ratio=0.15
    ).timestamp_overlap
    assert not health_pilot.crop_geometry(
        outside, padding_ratio=0.15
    ).timestamp_overlap
    assert not health_pilot.crop_geometry(
        touching, padding_ratio=0.15
    ).timestamp_overlap


def test_camera_assignment_sampling_is_exact_deterministic_and_test_free(
    tmp_path: Path,
) -> None:
    manifest, first = selected_fixture(tmp_path)
    second = health_pilot.select_health_rows(
        manifest,
        train_per_species_status=2,
        validation_per_species_status=1,
        seed=20260726,
        max_total_images=30,
    )
    first_keys = [
        smoke.candidate_unique_key(row) for row in first.selected
    ]
    second_keys = [
        smoke.candidate_unique_key(row) for row in second.selected
    ]
    assert first_keys == second_keys
    assert len(first.selected) == 30
    assert first.camera_stats.skipped_test_rows == 1
    assert all(row["split"] != "test" for row in first.selected)
    assert holdout.camera_overlap(first.selected) == set()
    counts = Counter(
        (row["split"], row["species"], row["normality"])
        for row in first.selected
    )
    assert {
        count
        for (split, _species, _normality), count in counts.items()
        if split == "train"
    } == {2}
    assert {
        count
        for (split, _species, _normality), count in counts.items()
        if split == "validation"
    } == {1}
    assert sum(
        row["normality"] == "abnormal" for row in first.selected
    ) == 15


def test_health_label_contradiction_test_and_duplicate_are_rejected(
    tmp_path: Path,
) -> None:
    _manifest, selection = selected_fixture(tmp_path)
    contradictory = dict(selection.selected[0])
    contradictory["normality"] = "abnormal"
    contradictory["disease_type"] = ""
    with pytest.raises(ValueError, match="모순"):
        health_pilot.health_class_for_row(contradictory)

    test_row = dict(selection.selected[0])
    test_row["split"] = "test"
    with pytest.raises(ValueError, match="Test"):
        health_pilot.build_crop_plans(
            [test_row], padding_ratio=0.15
        )

    duplicate = [dict(row) for row in selection.selected]
    duplicate[-1] = dict(duplicate[0])
    with pytest.raises(ValueError, match="중복"):
        health_pilot.validate_selected_health_rows(
            duplicate, train_target=2, validation_target=1
        )


def test_review_selection_global_limit_and_per_group_limit() -> None:
    rows: list[dict[str, str]] = []
    for split in ("train", "validation"):
        for species in pilot.SPECIES:
            for task in pilot.TASKS:
                for index in range(6):
                    rows.append(
                        source_row(
                            species=species,
                            camera_id=str(index + 1),
                            task=task,
                            index=index,
                            split=split,
                            disease="병A" if task == "병해" else "<missing>",
                        )
                    )
    plans = health_pilot.build_crop_plans(rows, padding_ratio=0.15)
    selected = health_pilot.select_review_keys(
        plans, seed=20260726, max_review_images=100
    )
    assert len(selected) == 100
    selected_groups = Counter(
        health_pilot.review_group(plan)
        for plan in plans
        if plan.unique_key in selected
    )
    assert set(selected_groups.values()) == {5}
    with pytest.raises(ValueError, match="100"):
        health_pilot.select_review_keys(
            plans, seed=20260726, max_review_images=101
        )


def test_selected_members_only_crop_output_and_invariants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest, selection = selected_fixture(tmp_path)
    selected = selection.selected
    image_zip = tmp_path / "source" / "TS1_느타리.zip"
    members = {
        row["image_member"]: source_image_bytes() for row in selected
    }
    members["생육/not_selected.jpg"] = b"not an image"
    members["생육/test_poison.jpg"] = b"not an image"
    write_zip(image_zip, members)

    artifacts = tmp_path / "artifacts"
    for name in (
        "yolo_smoke",
        "yolo_pilot",
        "yolo_pilot_camera_holdout",
        "models",
        "unrelated",
    ):
        directory = artifacts / name
        directory.mkdir(parents=True)
        (directory / "sentinel.txt").write_text(name, encoding="utf-8")
    output = artifacts / "health_pilot"
    artifacts_before = health_pilot.snapshot_artifacts_excluding(
        artifacts, output
    )
    zip_before = label_audit.source_zip_snapshot((image_zip,))
    reads: list[str] = []
    original_read = zipfile.ZipFile.read

    def tracked_read(
        self: zipfile.ZipFile,
        name: object,
        *args: object,
        **kwargs: object,
    ) -> bytes:
        member = name.filename if isinstance(name, zipfile.ZipInfo) else str(name)
        reads.append(member)
        return original_read(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", tracked_read)
    result = health_pilot.create_health_dataset(
        selection,
        {"TS1": archive_ref(image_zip)},
        output,
        train_per_species_status=2,
        validation_per_species_status=1,
        padding_ratio=0.15,
        seed=20260726,
        max_total_images=30,
        max_review_images=10,
        overwrite=False,
        artifacts_root=artifacts,
        show_progress=False,
    )

    assert Counter(reads) == Counter(
        row["image_member"] for row in selected
    )
    assert "생육/not_selected.jpg" not in reads
    assert "생육/test_poison.jpg" not in reads
    assert label_audit.source_zip_snapshot((image_zip,)) == zip_before
    assert (
        health_pilot.snapshot_artifacts_excluding(artifacts, output)
        == artifacts_before
    )
    assert result["train_images"] == 20
    assert result["validation_images"] == 10
    assert result["camera_overlap"] == 0
    assert result["test_images"] == 0
    assert result["missing_images"] == 0
    assert result["corrupt_images"] == 0
    assert result["empty_crops"] == 0
    assert result["review_count"] == 10

    with (output / "health_pilot_manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        manifest_rows = list(csv.DictReader(handle))
    assert len(manifest_rows) == 30
    assert set(health_pilot.MANIFEST_COLUMNS) == set(manifest_rows[0])
    assert Counter(
        (
            row["split"],
            row["health_class_name"],
        )
        for row in manifest_rows
    ) == {
        ("train", "HEALTHY"): 10,
        ("train", "DISEASE_SUSPECTED"): 10,
        ("validation", "HEALTHY"): 5,
        ("validation", "DISEASE_SUSPECTED"): 5,
    }
    first_crop = output / manifest_rows[0]["output_relative_path"]
    with Image.open(first_crop) as crop:
        crop.load()
        assert crop.size == (
            int(manifest_rows[0]["crop_width"]),
            int(manifest_rows[0]["crop_height"]),
        )
        rgb = crop.convert("RGB")
        pixels = [
            rgb.getpixel((x, y))
            for y in range(rgb.height)
            for x in range(rgb.width)
        ]
    assert any(r > 180 and g < 80 for r, g, _b in pixels)
    assert any(g > 180 and r < 80 for r, g, _b in pixels)


def test_output_text_is_deidentified_and_class_mapping_is_correct(
    tmp_path: Path,
) -> None:
    _manifest, selection = selected_fixture(tmp_path)
    image_zip = tmp_path / "private" / "TS1_느타리.zip"
    write_zip(
        image_zip,
        {
            row["image_member"]: source_image_bytes()
            for row in selection.selected
        },
    )
    artifacts = tmp_path / "artifacts"
    output = artifacts / "health_pilot"
    health_pilot.create_health_dataset(
        selection,
        {"TS1": archive_ref(image_zip)},
        output,
        train_per_species_status=2,
        validation_per_species_status=1,
        padding_ratio=0.15,
        seed=20260726,
        max_total_images=30,
        max_review_images=10,
        overwrite=False,
        artifacts_root=artifacts,
        show_progress=False,
    )
    mapping = json.loads(
        (output / "class_mapping.json").read_text(encoding="utf-8")
    )
    assert mapping["classes"]["0"]["name"] == "HEALTHY"
    assert mapping["classes"]["1"]["name"] == "DISEASE_SUSPECTED"
    health_pilot.assert_deidentified_output(output)
    for path in output.rglob("*"):
        if path.is_file() and path.suffix.lower() in {
            ".csv",
            ".json",
            ".md",
            ".txt",
            ".yaml",
        }:
            text = path.read_text(encoding="utf-8-sig")
            assert str(tmp_path) not in text
            assert "/mnt/" not in text
            assert "/home/" not in text


def test_hard_limits_and_protected_output_locations() -> None:
    assert health_pilot.validate_requested_limits(500, 100, 6000, 100) == 6000
    with pytest.raises(ValueError, match="6000"):
        health_pilot.validate_requested_limits(500, 101, 6000, 100)
    with pytest.raises(ValueError, match="100"):
        health_pilot.validate_requested_limits(500, 100, 6000, 101)
    with pytest.raises(ValueError, match="기존"):
        health_pilot.validate_output_location(
            health_pilot.ARTIFACTS_ROOT / "yolo_pilot" / "health"
        )
