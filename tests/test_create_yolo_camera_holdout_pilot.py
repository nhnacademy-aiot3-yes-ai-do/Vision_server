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
holdout = load_module(
    "create_yolo_camera_holdout_pilot",
    PROJECT_ROOT / "scripts" / "create_yolo_camera_holdout_pilot.py",
)


def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (100, 80), (90, 110, 130)).save(
        buffer, format="JPEG"
    )
    return buffer.getvalue()


def source_row(
    *,
    species: str,
    camera_id: str,
    task: str,
    index: int,
    split: str = "train",
    disease: str = "<missing>",
) -> dict[str, str]:
    class_id = smoke.CLASS_NAMES_INV[species]
    member = (
        f"{task}/{split}_{class_id}_cam{camera_id}_{task}_{index:03d}.jpg"
    )
    bbox = [
        {
            "annotation_index": 0,
            "x": 10,
            "y": 10,
            "width": 20,
            "height": 30,
        }
    ]
    return {
        "split": split,
        "group_key": f'["{species}","{camera_id}","2021-01-01"]',
        "species": species,
        "class_id": str(class_id),
        "task": task,
        "normality": "normal" if task == "생육" else "abnormal",
        "disease_type": disease,
        "camera_id": camera_id,
        "capture_date": "2021-01-01",
        "capture_time": "12:00:00",
        "label_archive_id": "TL1",
        "image_archive_id": "TS1",
        "json_member": str(Path(member).with_suffix(".json")),
        "image_member": member,
        "image_width": "100",
        "image_height": "80",
        "valid_bbox_count": "1",
        "bbox_cleaned": json.dumps(bbox, separators=(",", ":")),
        "official_split": "train",
    }


def synthetic_rows(
    *,
    cameras_per_species: int = 6,
    images_per_task_camera: int = 3,
    missing_disease_species: str | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for species in pilot.SPECIES:
        for camera_number in range(1, cameras_per_species + 1):
            camera_id = str(camera_number)
            for task in pilot.TASKS:
                if task == "병해" and species == missing_disease_species:
                    continue
                for index in range(images_per_task_camera):
                    disease = (
                        "병A"
                        if task == "병해" and camera_number <= 3
                        else "병B"
                        if task == "병해"
                        else "<missing>"
                    )
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
                            disease=disease,
                        )
                    )
    rows.append(
        source_row(
            species="느타리",
            camera_id="99",
            task="생육",
            index=0,
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


def feasible_selection(tmp_path: Path) -> tuple[Path, object, object, list[dict[str, str]]]:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, synthetic_rows())
    stats = holdout.stream_camera_statistics(manifest)
    assignment = holdout.choose_camera_assignment(
        stats,
        train_per_species_task=2,
        validation_per_species_task=1,
        seed=20260726,
    )
    selected = holdout.select_camera_holdout_rows(
        manifest,
        stats,
        assignment,
        train_per_species_task=2,
        validation_per_species_task=1,
        seed=20260726,
        max_total_images=30,
    )
    return manifest, stats, assignment, selected


def test_camera_groups_are_disjoint_deterministic_and_exact(
    tmp_path: Path,
) -> None:
    manifest, stats, first, selected = feasible_selection(tmp_path)
    second = holdout.choose_camera_assignment(
        stats,
        train_per_species_task=2,
        validation_per_species_task=1,
        seed=20260726,
    )
    assert first.feasible is True
    assert first.validation_cameras == second.validation_cameras
    assert holdout.camera_overlap(selected) == set()
    assert len(selected) == 30
    counts = Counter(pilot.source_group(row) for row in selected)
    assert {
        count for (split, _species, _task), count in counts.items()
        if split == "train"
    } == {2}
    assert {
        count for (split, _species, _task), count in counts.items()
        if split == "validation"
    } == {1}
    repeated = holdout.select_camera_holdout_rows(
        manifest,
        stats,
        second,
        train_per_species_task=2,
        validation_per_species_task=1,
        seed=20260726,
        max_total_images=30,
    )
    assert [
        smoke.candidate_unique_key(row) for row in selected
    ] == [smoke.candidate_unique_key(row) for row in repeated]


def test_impossible_assignment_reports_shortfall(tmp_path: Path) -> None:
    manifest = tmp_path / "impossible.csv"
    write_manifest(
        manifest,
        synthetic_rows(missing_disease_species="느타리"),
    )
    stats = holdout.stream_camera_statistics(manifest)
    assignment = holdout.choose_camera_assignment(
        stats,
        train_per_species_task=2,
        validation_per_species_task=1,
        seed=20260726,
    )
    assert assignment.feasible is False
    assert assignment.closest_shortfall > 0
    assert "느타리" in assignment.reason


def test_test_and_duplicate_images_are_rejected(tmp_path: Path) -> None:
    _manifest, stats, assignment, selected = feasible_selection(tmp_path)
    test_selected = [dict(row) for row in selected]
    test_selected[0]["split"] = "test"
    with pytest.raises(ValueError, match="Test"):
        holdout.create_camera_holdout_dataset(
            test_selected,
            stats,
            assignment,
            {},
            tmp_path / "out",
            train_per_species_task=2,
            validation_per_species_task=1,
            seed=20260726,
            max_total_images=30,
            max_overlay_images=10,
            overwrite=False,
            smoke_dir=tmp_path / "smoke",
            pilot_dir=tmp_path / "pilot",
            show_progress=False,
        )
    duplicate = [dict(row) for row in selected]
    duplicate[-1] = dict(duplicate[0])
    with pytest.raises(ValueError, match="중복"):
        holdout.create_camera_holdout_dataset(
            duplicate,
            stats,
            assignment,
            {},
            tmp_path / "out",
            train_per_species_task=2,
            validation_per_species_task=1,
            seed=20260726,
            max_total_images=30,
            max_overlay_images=10,
            overwrite=False,
            smoke_dir=tmp_path / "smoke",
            pilot_dir=tmp_path / "pilot",
            show_progress=False,
        )


def test_yolo_coordinates_and_overlay_limit(tmp_path: Path) -> None:
    box = smoke.convert_bbox_to_yolo(
        {"x": 10, "y": 20, "width": 20, "height": 10},
        class_id=0,
        image_width=100,
        image_height=50,
    )
    smoke.validate_yolo_box(box)
    _manifest, _stats, _assignment, selected = feasible_selection(tmp_path)
    keys = pilot.select_overlay_keys(
        selected * 4, seed=20260726, max_overlay_images=100
    )
    assert len(keys) <= 100
    with pytest.raises(ValueError, match="100"):
        pilot.select_overlay_keys(
            selected, seed=20260726, max_overlay_images=101
        )


def test_selected_members_only_and_protected_directories_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest, stats, assignment, selected = feasible_selection(tmp_path)
    image_zip = tmp_path / "TS1_느타리.zip"
    members = {
        row["image_member"]: jpeg_bytes() for row in selected
    }
    members["생육/not_selected.jpg"] = jpeg_bytes()
    write_zip(image_zip, members)
    smoke_dir = tmp_path / "artifacts" / "yolo_smoke"
    pilot_dir = tmp_path / "artifacts" / "yolo_pilot"
    smoke_dir.mkdir(parents=True)
    pilot_dir.mkdir(parents=True)
    (smoke_dir / "marker.txt").write_text("smoke", encoding="utf-8")
    (pilot_dir / "marker.txt").write_text("pilot", encoding="utf-8")
    protected_before = {
        smoke_dir: pilot.directory_snapshot(smoke_dir),
        pilot_dir: pilot.directory_snapshot(pilot_dir),
    }
    zip_before = audit.source_zip_snapshot((image_zip,))
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
    output = tmp_path / "artifacts" / "holdout"
    result = holdout.create_camera_holdout_dataset(
        selected,
        stats,
        assignment,
        {"TS1": archive_ref(image_zip)},
        output,
        train_per_species_task=2,
        validation_per_species_task=1,
        seed=20260726,
        max_total_images=30,
        max_overlay_images=10,
        overwrite=False,
        smoke_dir=smoke_dir,
        pilot_dir=pilot_dir,
        show_progress=False,
    )
    assert set(reads) == {row["image_member"] for row in selected}
    assert len(reads) == len(selected)
    assert audit.source_zip_snapshot((image_zip,)) == zip_before
    for path, snapshot in protected_before.items():
        assert pilot.directory_snapshot(path) == snapshot
    assert result["camera_overlap"] == 0
    assert result["missing_images"] == 0
    assert result["coordinate_errors"] == 0
    assert result["test_images"] == 0


def test_output_paths_are_deidentified(tmp_path: Path) -> None:
    _manifest, stats, assignment, selected = feasible_selection(tmp_path)
    image_zip = tmp_path / "TS1_느타리.zip"
    write_zip(
        image_zip,
        {row["image_member"]: jpeg_bytes() for row in selected},
    )
    output = tmp_path / "artifacts" / "holdout"
    holdout.create_camera_holdout_dataset(
        selected,
        stats,
        assignment,
        {"TS1": archive_ref(image_zip)},
        output,
        train_per_species_task=2,
        validation_per_species_task=1,
        seed=20260726,
        max_total_images=30,
        max_overlay_images=10,
        overwrite=False,
        smoke_dir=tmp_path / "artifacts" / "smoke",
        pilot_dir=tmp_path / "artifacts" / "pilot",
        show_progress=False,
    )
    assert smoke.text_path_leaks(output) == []
    for path in output.rglob("*"):
        if path.is_file() and path.suffix in {".txt", ".csv", ".yaml", ".md"}:
            text = path.read_text(encoding="utf-8-sig")
            assert str(tmp_path) not in text
            assert "/mnt/d" not in text
            assert "/home/kim75" not in text
