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
from PIL import Image


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


def jpeg_bytes(size: tuple[int, int] = (100, 80)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (80, 100, 120)).save(buffer, format="JPEG")
    return buffer.getvalue()


def write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for member, content in members.items():
            archive.writestr(member, content)


def manifest_row(
    *,
    split: str,
    species: str,
    task: str,
    index: int,
    disease_type: str = "<missing>",
    archive_id: str = "TS1",
    boxes: list[dict[str, float]] | None = None,
) -> dict[str, str]:
    class_id = smoke.CLASS_NAMES_INV[species]
    cleaned = boxes or [
        {"annotation_index": 0, "x": 10, "y": 10, "width": 20, "height": 30}
    ]
    member = f"{task}/{split}_{class_id}_{task}_{index:04d}.jpg"
    return {
        "split": split,
        "group_key": f'["{species}","1","2021-01-01"]',
        "species": species,
        "class_id": str(class_id),
        "task": task,
        "normality": "normal" if task == "생육" else "abnormal",
        "disease_type": disease_type,
        "camera_id": "1",
        "capture_date": "2021-01-01",
        "capture_time": "12:00:00",
        "label_archive_id": "TL1",
        "image_archive_id": archive_id,
        "json_member": str(Path(member).with_suffix(".json")),
        "image_member": member,
        "image_width": "100",
        "image_height": "80",
        "valid_bbox_count": str(len(cleaned)),
        "bbox_cleaned": json.dumps(cleaned, separators=(",", ":")),
        "official_split": "train",
    }


def synthetic_manifest_rows(
    *,
    per_group: int = 10,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in pilot.SPLITS:
        for species in pilot.SPECIES:
            for task in pilot.TASKS:
                for index in range(per_group):
                    disease = (
                        "병A"
                        if task == "병해" and index < per_group * 0.8
                        else "병B"
                        if task == "병해"
                        else "<missing>"
                    )
                    rows.append(
                        manifest_row(
                            split=split,
                            species=species,
                            task=task,
                            index=index,
                            disease_type=disease,
                        )
                    )
    rows.append(
        manifest_row(
            split="test",
            species="느타리",
            task="생육",
            index=999,
        )
    )
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(pilot.PILOT_SOURCE_COLUMNS))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def exact_selected_rows() -> list[dict[str, str]]:
    return [
        manifest_row(
            split=split,
            species=species,
            task=task,
            index=smoke.CLASS_NAMES_INV[species] * 10
            + (0 if task == "생육" else 1),
            disease_type="병A" if task == "병해" else "<missing>",
        )
        for split in pilot.SPLITS
        for species in pilot.SPECIES
        for task in pilot.TASKS
    ]


def test_exact_group_counts_determinism_and_disease_stratification(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, synthetic_manifest_rows())
    first = pilot.select_pilot_rows(
        manifest,
        train_per_species_task=5,
        validation_per_species_task=5,
        seed=20260726,
        max_total_images=100,
    )
    second = pilot.select_pilot_rows(
        manifest,
        train_per_species_task=5,
        validation_per_species_task=5,
        seed=20260726,
        max_total_images=100,
    )
    assert [
        smoke.candidate_unique_key(row) for row in first
    ] == [smoke.candidate_unique_key(row) for row in second]
    groups = Counter(pilot.source_group(row) for row in first)
    assert len(first) == 100
    assert set(groups.values()) == {5}
    for split in pilot.SPLITS:
        for species in pilot.SPECIES:
            diseases = Counter(
                row["disease_type"]
                for row in first
                if row["split"] == split
                and row["species"] == species
                and row["task"] == "병해"
            )
            assert diseases == {"병A": 4, "병B": 1}
    assert all(row["split"] != "test" for row in first)


def test_six_thousand_hard_limit_and_test_rejection() -> None:
    assert (
        pilot.validate_requested_limits(500, 100, 6000, 100) == 6000
    )
    with pytest.raises(ValueError, match="초과"):
        pilot.validate_requested_limits(500, 101, 6000, 100)
    with pytest.raises(ValueError, match="6000"):
        pilot.validate_requested_limits(500, 100, 6001, 100)
    test_row = manifest_row(
        split="test",
        species="느타리",
        task="생육",
        index=1,
    )
    with pytest.raises(ValueError, match="Test"):
        pilot.validate_source_row(test_row)


def test_duplicate_selection_is_rejected(tmp_path: Path) -> None:
    selected = exact_selected_rows()
    selected[-1] = dict(selected[0])
    with pytest.raises(ValueError, match="중복"):
        pilot.create_pilot_dataset(
            selected,
            {},
            tmp_path / "pilot",
            train_per_species_task=1,
            validation_per_species_task=1,
            seed=20260726,
            max_total_images=20,
            max_overlay_images=10,
            overwrite=False,
            smoke_dir=tmp_path / "smoke",
            show_progress=False,
        )


def test_bbox_conversion_and_multiple_boxes() -> None:
    box = pilot.convert_bbox_to_yolo(
        {"x": 10, "y": 20, "width": 20, "height": 10},
        class_id=0,
        image_width=100,
        image_height=50,
    )
    assert (box.x_center, box.y_center, box.width, box.height) == pytest.approx(
        (0.2, 0.5, 0.2, 0.2)
    )
    row = manifest_row(
        split="train",
        species="느타리",
        task="생육",
        index=1,
        boxes=[
            {"x": 0, "y": 0, "width": 10, "height": 10},
            {"x": 20, "y": 20, "width": 30, "height": 20},
        ],
    )
    assert len(pilot.yolo_boxes_for_row(row)) == 2


def test_serialized_boundary_box_remains_in_range(tmp_path: Path) -> None:
    box = pilot.convert_bbox_to_yolo(
        {"x": 1, "y": 2, "width": 99, "height": 78},
        class_id=0,
        image_width=100,
        image_height=80,
    )
    label = tmp_path / "edge.txt"
    label.write_text(box.line() + "\n", encoding="utf-8")
    count, errors = smoke.parse_label_file(label)
    assert count == 1
    assert errors == 0


def test_overlay_selection_never_exceeds_one_hundred() -> None:
    selected: list[dict[str, str]] = []
    for repetition in range(10):
        for row in exact_selected_rows():
            copied = dict(row)
            copied["image_member"] = (
                f"{Path(row['image_member']).stem}_{repetition}.jpg"
            )
            selected.append(copied)
    keys = pilot.select_overlay_keys(
        selected, seed=20260726, max_overlay_images=100
    )
    assert len(keys) == 100
    with pytest.raises(ValueError, match="100"):
        pilot.select_overlay_keys(
            selected, seed=20260726, max_overlay_images=101
        )


def test_selected_members_only_sources_and_smoke_remain_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = exact_selected_rows()
    image_zip = tmp_path / "TS1_느타리.zip"
    members = {
        row["image_member"]: jpeg_bytes() for row in selected
    }
    members["생육/not_selected.jpg"] = jpeg_bytes()
    write_zip(image_zip, members)
    smoke_dir = tmp_path / "artifacts" / "yolo_smoke"
    smoke_dir.mkdir(parents=True)
    smoke_marker = smoke_dir / "marker.txt"
    smoke_marker.write_text("unchanged", encoding="utf-8")
    source_before = audit.source_zip_snapshot((image_zip,))
    smoke_before = pilot.directory_snapshot(smoke_dir)
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
    output = tmp_path / "artifacts" / "yolo_pilot"
    result = pilot.create_pilot_dataset(
        selected,
        {"TS1": archive_ref(image_zip)},
        output,
        train_per_species_task=1,
        validation_per_species_task=1,
        seed=20260726,
        max_total_images=20,
        max_overlay_images=10,
        overwrite=False,
        smoke_dir=smoke_dir,
        show_progress=False,
    )
    assert set(reads) == {row["image_member"] for row in selected}
    assert len(reads) == len(selected)
    assert audit.source_zip_snapshot((image_zip,)) == source_before
    assert pilot.directory_snapshot(smoke_dir) == smoke_before
    assert result["train_images"] == 10
    assert result["validation_images"] == 10
    assert result["overlay_count"] == 10
    assert result["coordinate_errors"] == 0
    assert result["train_val_source_overlap"] == 0


def test_output_has_no_absolute_paths(tmp_path: Path) -> None:
    selected = exact_selected_rows()
    image_zip = tmp_path / "TS1_느타리.zip"
    write_zip(
        image_zip,
        {row["image_member"]: jpeg_bytes() for row in selected},
    )
    output = tmp_path / "artifacts" / "yolo_pilot"
    pilot.create_pilot_dataset(
        selected,
        {"TS1": archive_ref(image_zip)},
        output,
        train_per_species_task=1,
        validation_per_species_task=1,
        seed=20260726,
        max_total_images=20,
        max_overlay_images=10,
        overwrite=False,
        smoke_dir=tmp_path / "artifacts" / "yolo_smoke",
        show_progress=False,
    )
    assert smoke.text_path_leaks(output) == []
    for path in output.rglob("*"):
        if path.is_file() and path.suffix in {".txt", ".csv", ".yaml", ".md"}:
            text = path.read_text(encoding="utf-8-sig")
            assert str(tmp_path) not in text
            assert "/mnt/d" not in text
            assert "/home/kim75" not in text
