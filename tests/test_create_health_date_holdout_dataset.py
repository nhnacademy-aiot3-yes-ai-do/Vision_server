from __future__ import annotations

import csv
import hashlib
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
VALIDATION_DATES = (
    "2021-11-04",
    "2021-11-26",
    "2021-12-04",
)


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
load_module(
    "create_yolo_camera_holdout_pilot",
    PROJECT_ROOT / "scripts" / "create_yolo_camera_holdout_pilot.py",
)
load_module(
    "audit_mushroom_health",
    PROJECT_ROOT / "scripts" / "audit_mushroom_health.py",
)
health_pilot = load_module(
    "create_health_pilot_dataset",
    PROJECT_ROOT / "scripts" / "create_health_pilot_dataset.py",
)
load_module(
    "audit_health_date_holdout",
    PROJECT_ROOT / "scripts" / "audit_health_date_holdout.py",
)
date_holdout = load_module(
    "create_health_date_holdout_dataset",
    PROJECT_ROOT / "scripts" / "create_health_date_holdout_dataset.py",
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


def first_fixed_camera(values: object) -> str:
    return sorted(str(value) for value in values)[0]


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
    capture_date: str,
    index: int,
    split: str = "train",
    disease: str = "<missing>",
    boxes: list[dict[str, int]] | None = None,
) -> dict[str, str]:
    class_id = smoke.CLASS_NAMES_INV[species]
    member = (
        f"{task}/{capture_date}_{class_id}_cam{camera_id}_"
        f"{task}_{index:04d}.jpg"
    )
    cleaned = boxes or DEFAULT_BOXES
    return {
        "split": split,
        "group_key": json.dumps(
            [species, camera_id, capture_date],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "species": species,
        "class_id": str(class_id),
        "task": task,
        "normality": "normal" if task == "생육" else "abnormal",
        "disease_type": disease,
        "camera_id": camera_id,
        "capture_date": capture_date,
        "capture_time": "12:00:00",
        "label_archive_id": "TL1",
        "image_archive_id": "TS1",
        "json_member": str(Path(member).with_suffix(".json")),
        "image_member": member,
        "image_width": "100",
        "image_height": "80",
        "valid_bbox_count": str(len(cleaned)),
        "bbox_cleaned": json.dumps(
            cleaned, ensure_ascii=False, separators=(",", ":")
        ),
        "official_split": (
            "validation" if split == "validation" else "train"
        ),
    }


def synthetic_rows(
    *,
    candidates_per_cell: int = 3,
) -> list[dict[str, str]]:
    train_camera = first_fixed_camera(
        date_holdout.FIXED_TRAIN_CAMERA_IDS
    )
    validation_camera = first_fixed_camera(
        date_holdout.FIXED_VALIDATION_CAMERA_IDS
    )
    rows: list[dict[str, str]] = []
    serial = 0
    for species_index, species in enumerate(pilot.SPECIES):
        for status_index, task in enumerate(pilot.TASKS):
            disease = "병A" if task == "병해" else "<missing>"
            validation_date = VALIDATION_DATES[
                (species_index * len(pilot.TASKS) + status_index)
                % len(VALIDATION_DATES)
            ]
            for candidate_index in range(candidates_per_cell):
                rows.append(
                    source_row(
                        species=species,
                        camera_id=train_camera,
                        task=task,
                        capture_date=(
                            "2021-10-20"
                            if candidate_index % 2 == 0
                            else "2021-11-01"
                        ),
                        index=serial,
                        split=(
                            "validation"
                            if candidate_index % 2
                            else "train"
                        ),
                        disease=disease,
                    )
                )
                serial += 1
                rows.append(
                    source_row(
                        species=species,
                        camera_id=validation_camera,
                        task=task,
                        capture_date=validation_date,
                        index=serial,
                        split=(
                            "train"
                            if candidate_index % 2
                            else "validation"
                        ),
                        disease=disease,
                    )
                )
                serial += 1

            # Crossed date/camera rows must not be used by the strict
            # intersection of the two global holdout constraints.
            rows.append(
                source_row(
                    species=species,
                    camera_id=validation_camera,
                    task=task,
                    capture_date="2021-10-25",
                    index=serial,
                    disease=disease,
                )
            )
            serial += 1
            rows.append(
                source_row(
                    species=species,
                    camera_id=train_camera,
                    task=task,
                    capture_date=validation_date,
                    index=serial,
                    disease=disease,
                )
            )
            serial += 1

    rows.append(
        source_row(
            species="느타리",
            camera_id=validation_camera,
            task="생육",
            capture_date=VALIDATION_DATES[0],
            index=99_999,
            split="test",
        )
    )
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(dict.fromkeys(pilot.PILOT_SOURCE_COLUMNS)),
        )
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


def selection_fixture(tmp_path: Path) -> tuple[Path, object]:
    manifest = tmp_path / "reports" / "detection_manifest.csv"
    write_manifest(manifest, synthetic_rows())
    selection = date_holdout.select_date_holdout_rows(
        manifest,
        train_per_species_status=2,
        validation_per_species_status=1,
        validation_dates=VALIDATION_DATES,
        seed=20260726,
        max_total_images=30,
    )
    return manifest, selection


def fingerprint(path: Path) -> tuple[int, int, str]:
    return (
        path.stat().st_size,
        path.stat().st_mtime_ns,
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def tree_fingerprint(root: Path) -> dict[str, tuple[int, int, str]]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): fingerprint(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_fixed_dates_global_cameras_exact_counts_determinism_and_test_free(
    tmp_path: Path,
) -> None:
    assert {
        str(camera) for camera in date_holdout.FIXED_TRAIN_CAMERA_IDS
    } == {"3", "4", "5", "7", "8", "9", "11", "12", "15", "17", "19"}
    assert {
        str(camera)
        for camera in date_holdout.FIXED_VALIDATION_CAMERA_IDS
    } == {"1", "2", "6", "10", "13", "14", "16", "18", "20"}

    manifest, first = selection_fixture(tmp_path)
    second = date_holdout.select_date_holdout_rows(
        manifest,
        train_per_species_status=2,
        validation_per_species_status=1,
        validation_dates=VALIDATION_DATES,
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
    assert len(first_keys) == len(set(first_keys)) == 30
    assert all(row["split"] != "test" for row in first.selected)

    counts = Counter(
        (row["split"], row["species"], row["normality"])
        for row in first.selected
    )
    assert {
        counts[("train", species, normality)]
        for species in pilot.SPECIES
        for normality in ("normal", "abnormal")
    } == {2}
    assert {
        counts[("validation", species, normality)]
        for species in pilot.SPECIES
        for normality in ("normal", "abnormal")
    } == {1}

    train_dates = {
        row["capture_date"]
        for row in first.selected
        if row["split"] == "train"
    }
    validation_dates = {
        row["capture_date"]
        for row in first.selected
        if row["split"] == "validation"
    }
    assert validation_dates == set(VALIDATION_DATES)
    assert train_dates.isdisjoint(validation_dates)

    train_cameras = {
        row["camera_id"]
        for row in first.selected
        if row["split"] == "train"
    }
    validation_cameras = {
        row["camera_id"]
        for row in first.selected
        if row["split"] == "validation"
    }
    assert train_cameras <= {
        str(camera) for camera in date_holdout.FIXED_TRAIN_CAMERA_IDS
    }
    assert validation_cameras <= {
        str(camera)
        for camera in date_holdout.FIXED_VALIDATION_CAMERA_IDS
    }
    assert train_cameras.isdisjoint(validation_cameras)


def test_validation_rejects_split_date_camera_duplicate_and_test(
    tmp_path: Path,
) -> None:
    _manifest, selection = selection_fixture(tmp_path)
    selected = [dict(row) for row in selection.selected]

    date_leak = [dict(row) for row in selected]
    next(
        row for row in date_leak if row["split"] == "train"
    )["capture_date"] = VALIDATION_DATES[0]
    with pytest.raises(ValueError, match="날짜"):
        date_holdout.validate_selected_rows(
            date_leak,
            train_target=2,
            validation_target=1,
            validation_dates=VALIDATION_DATES,
        )

    camera_leak = [dict(row) for row in selected]
    next(
        row for row in camera_leak if row["split"] == "train"
    )["camera_id"] = next(
        row["camera_id"]
        for row in camera_leak
        if row["split"] == "validation"
    )
    with pytest.raises(ValueError, match="camera|카메라"):
        date_holdout.validate_selected_rows(
            camera_leak,
            train_target=2,
            validation_target=1,
            validation_dates=VALIDATION_DATES,
        )

    duplicate = [dict(row) for row in selected]
    duplicate[-1] = dict(duplicate[0])
    with pytest.raises(ValueError, match="중복"):
        date_holdout.validate_selected_rows(
            duplicate,
            train_target=2,
            validation_target=1,
            validation_dates=VALIDATION_DATES,
        )

    test_rows = [dict(row) for row in selected]
    test_rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="Test"):
        date_holdout.validate_selected_rows(
            test_rows,
            train_target=2,
            validation_target=1,
            validation_dates=VALIDATION_DATES,
        )


def test_exact_target_is_not_filled_by_splitting_date_or_camera_groups(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    # Every cell has three valid Validation candidates plus one crossed
    # Train-camera candidate on a Validation date.  A target of four can only
    # be reached by violating the global camera partition.
    write_manifest(manifest, synthetic_rows(candidates_per_cell=3))
    with pytest.raises(ValueError, match="부족|불가능|수량"):
        date_holdout.select_date_holdout_rows(
            manifest,
            train_per_species_status=2,
            validation_per_species_status=4,
            validation_dates=VALIDATION_DATES,
            seed=20260726,
            max_total_images=60,
        )


def test_union_bbox_and_fifteen_percent_padding_are_reused() -> None:
    row = source_row(
        species="느타리",
        camera_id=first_fixed_camera(
            date_holdout.FIXED_TRAIN_CAMERA_IDS
        ),
        task="생육",
        capture_date="2021-10-20",
        index=1,
    )
    geometry = date_holdout.crop_geometry(row, padding_ratio=0.15)
    assert geometry.union_rect == pytest.approx((20, 20, 70, 40))
    assert geometry.crop_rect == (5, 8, 85, 52)
    assert (geometry.crop_width, geometry.crop_height) == (80, 44)

    edge = source_row(
        species="느타리",
        camera_id=first_fixed_camera(
            date_holdout.FIXED_TRAIN_CAMERA_IDS
        ),
        task="생육",
        capture_date="2021-10-20",
        index=2,
        boxes=[{"x": 2, "y": 3, "width": 20, "height": 10}],
    )
    assert date_holdout.crop_geometry(
        edge, padding_ratio=0.15
    ).crop_rect == (0, 0, 37, 25)


def test_cli_rejects_date_or_seed_drift() -> None:
    with pytest.raises(SystemExit):
        date_holdout.parse_args(
            [
                "--validation-dates",
                "2021-11-04",
                "2021-11-26",
                "2021-12-05",
            ]
        )
    with pytest.raises(SystemExit):
        date_holdout.parse_args(["--seed", "1"])


def test_selected_zip_members_only_and_all_source_artifacts_invariants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, selection = selection_fixture(tmp_path)
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
        "health_pilot",
        "health_pilot_timestamp_masked",
        "yolo_smoke",
        "yolo_pilot",
        "yolo_pilot_camera_holdout",
        "models",
        "unrelated",
    ):
        sentinel = artifacts / name / "sentinel.bin"
        sentinel.parent.mkdir(parents=True)
        sentinel.write_bytes(name.encode("utf-8"))
    output = artifacts / "health_pilot_date_holdout"
    protected_before = tree_fingerprint(artifacts)
    zip_before = fingerprint(image_zip)
    manifest_before = fingerprint(manifest)
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
    result = date_holdout.create_date_holdout_dataset(
        selection,
        {"TS1": archive_ref(image_zip)},
        output,
        train_per_species_status=2,
        validation_per_species_status=1,
        validation_dates=VALIDATION_DATES,
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
    assert fingerprint(image_zip) == zip_before
    assert fingerprint(manifest) == manifest_before
    assert {
        relative: value
        for relative, value in tree_fingerprint(artifacts).items()
        if not relative.startswith("health_pilot_date_holdout/")
    } == protected_before

    assert result["train_images"] == 20
    assert result["validation_images"] == 10
    assert result["test_images"] == 0
    assert result["date_overlap"] == 0
    assert result["camera_overlap"] == 0
    assert result["missing_images"] == 0
    assert result["corrupt_images"] == 0
    assert result["empty_crops"] == 0

    manifest_path = output / "health_date_holdout_manifest.csv"
    with manifest_path.open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 30
    assert set(rows[0]) == set(date_holdout.MANIFEST_COLUMNS)
    assert {
        row["capture_date"]
        for row in rows
        if row["split"] == "validation"
    } == set(VALIDATION_DATES)
    assert not {
        row["capture_date"]
        for row in rows
        if row["split"] == "train"
    } & set(VALIDATION_DATES)
    assert not {
        row["camera_id"] for row in rows if row["split"] == "train"
    } & {
        row["camera_id"]
        for row in rows
        if row["split"] == "validation"
    }

    first = rows[0]
    assert (int(first["crop_width"]), int(first["crop_height"])) == (
        80,
        44,
    )
    with Image.open(output / first["output_relative_path"]) as crop:
        crop.load()
        assert crop.size == (80, 44)


def test_review_limit_summary_limitations_and_deidentified_output(
    tmp_path: Path,
) -> None:
    _manifest, selection = selection_fixture(tmp_path)
    image_zip = tmp_path / "private" / "TS1_느타리.zip"
    write_zip(
        image_zip,
        {
            row["image_member"]: source_image_bytes()
            for row in selection.selected
        },
    )
    artifacts = tmp_path / "artifacts"
    output = artifacts / "health_pilot_date_holdout"
    result = date_holdout.create_date_holdout_dataset(
        selection,
        {"TS1": archive_ref(image_zip)},
        output,
        train_per_species_status=2,
        validation_per_species_status=1,
        validation_dates=VALIDATION_DATES,
        padding_ratio=0.15,
        seed=20260726,
        max_total_images=30,
        max_review_images=100,
        overwrite=False,
        artifacts_root=artifacts,
        show_progress=False,
    )
    assert result["review_count"] <= 100
    summary = (
        output / "health_date_holdout_summary.md"
    ).read_text(encoding="utf-8")
    for required in (
        "HEALTHY",
        "DISEASE_SUSPECTED",
        "병해 종류 모델",
        "고정 Test",
        "스마트폰",
        "timestamp",
    ):
        assert required.casefold() in summary.casefold()

    date_holdout.assert_deidentified_output(output)
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
            assert "/tmp/" not in text
            assert "/mnt/" not in text
            assert "/home/" not in text
            assert ":\\\\" not in text
