from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
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
load_module(
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
health = load_module(
    "audit_mushroom_health",
    PROJECT_ROOT / "scripts" / "audit_mushroom_health.py",
)
raw_health = load_module(
    "create_health_pilot_dataset",
    PROJECT_ROOT / "scripts" / "create_health_pilot_dataset.py",
)
date_audit = load_module(
    "audit_health_date_holdout",
    PROJECT_ROOT / "scripts" / "audit_health_date_holdout.py",
)


DETECTION_FIELDS = (
    "split",
    "species",
    "task",
    "normality",
    "disease_type",
    "camera_id",
    "capture_date",
    "image_archive_id",
    "image_member",
    "official_split",
)


def write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def detection_row(
    *,
    split: str,
    species: str,
    normality: str,
    capture_date: str,
    index: int,
) -> dict[str, str]:
    task = "생육" if normality == "normal" else "병해"
    return {
        "split": split,
        "species": species,
        "task": task,
        "normality": normality,
        "disease_type": (
            "<missing>" if normality == "normal" else "푸른곰팡이병"
        ),
        "camera_id": str(index % 4 + 1),
        "capture_date": capture_date,
        "image_archive_id": f"TS{index % 5 + 1}",
        "image_member": f"{task}/{capture_date}_{species}_{index:04d}.jpg",
        "official_split": "train" if index % 2 else "validation",
    }


def balanced_detection_rows() -> list[dict[str, str]]:
    """Six atomic dates: one train/validation date for each capacity family."""
    species = tuple(date_audit.SPECIES)
    rows: list[dict[str, str]] = []
    index = 0

    def add(
        capture_date: str,
        selected_species: tuple[str, ...],
        normality: str,
    ) -> None:
        nonlocal index
        for candidate_species in selected_species:
            for repeat in range(3):
                rows.append(
                    detection_row(
                        split="train" if repeat < 2 else "validation",
                        species=candidate_species,
                        normality=normality,
                        capture_date=capture_date,
                        index=index,
                    )
                )
                index += 1

    add("2021-01-01", species, "normal")
    add("2021-01-02", species, "normal")
    add("2021-01-03", species[:-1], "abnormal")
    add("2021-01-04", species[:-1], "abnormal")
    add("2021-01-05", (species[-1],), "abnormal")
    add("2021-01-06", (species[-1],), "abnormal")
    rows.extend(
        detection_row(
            split="test",
            species=species[index % len(species)],
            normality="normal" if index % 2 else "abnormal",
            capture_date="2099-12-31",
            index=10_000 + index,
        )
        for index in range(7)
    )
    return rows


def write_detection_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "reports" / "detection.csv"
    write_csv(path, DETECTION_FIELDS, balanced_detection_rows())
    return path


def raw_pilot_row(
    *,
    split: str,
    species: str,
    normality: str,
    index: int,
    timestamp_overlap: bool,
) -> dict[str, str]:
    class_id = 0 if normality == "normal" else 1
    class_name, class_directory = raw_health.HEALTH_CLASSES[class_id]
    split_directory = "train" if split == "train" else "val"
    filename = f"{split_directory}_{index:04d}.jpg"
    task = "생육" if normality == "normal" else "병해"
    return {
        "split": split,
        "health_class_id": str(class_id),
        "health_class_name": class_name,
        "species": species,
        "task": task,
        "normality": normality,
        "disease_type": (
            "<missing>" if normality == "normal" else "푸른곰팡이병"
        ),
        "camera_id": str(index % 5 + 1),
        "capture_date": f"2021-01-{index % 6 + 1:02d}",
        "capture_time": "12:00:00",
        "image_archive_id": f"TS{index % 5 + 1}",
        "image_member": f"{task}/source_{index:04d}.jpg",
        "original_width": "1000",
        "original_height": "500",
        "valid_bbox_count": "1",
        "union_bbox": '{"x":550,"y":20,"width":100,"height":40}',
        "padded_crop_bbox": '{"x":500,"y":0,"width":250,"height":100}',
        "padding_ratio": "0.15",
        "crop_width": "250",
        "crop_height": "100",
        "timestamp_region_overlap": str(timestamp_overlap),
        "output_relative_path": (
            f"{split_directory}/{class_directory}/{filename}"
        ),
    }


def pilot_manifest_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    index = 0
    for split, repeats in (("train", 2), ("validation", 1)):
        for species in date_audit.SPECIES:
            for normality in date_audit.NORMALITIES:
                for _repeat in range(repeats):
                    rows.append(
                        raw_pilot_row(
                            split=split,
                            species=species,
                            normality=normality,
                            index=index,
                            timestamp_overlap=(
                                normality == "abnormal" or index % 3 == 0
                            ),
                        )
                    )
                    index += 1
    return rows


def masked_row(raw: dict[str, str]) -> dict[str, str]:
    result = dict(raw)
    overlap = raw["timestamp_region_overlap"]
    result.update(
        {
            "timestamp_candidate_overlap": overlap,
            "timestamp_mask_applied": overlap,
            "mask_rect_crop_coordinates": (
                '{"x":100,"y":0,"width":150,"height":60}'
                if overlap == "True"
                else "null"
            ),
            "mask_width": "150" if overlap == "True" else "0",
            "mask_height": "60" if overlap == "True" else "0",
            "mask_area_ratio": "0.3600000000" if overlap == "True" else "0",
            "source_raw_relative_path": raw["output_relative_path"],
        }
    )
    return result


def write_pilot_fixtures(
    tmp_path: Path,
) -> tuple[Path, Path, list[dict[str, str]], list[dict[str, str]]]:
    raw_rows = pilot_manifest_rows()
    masked_rows = [masked_row(row) for row in raw_rows]
    raw_path = tmp_path / "artifacts" / "health_pilot" / "manifest.csv"
    masked_path = (
        tmp_path
        / "artifacts"
        / "health_pilot_timestamp_masked"
        / "manifest.csv"
    )
    write_csv(raw_path, raw_health.MANIFEST_COLUMNS, raw_rows)
    masked_fields = tuple(
        dict.fromkeys(
            (
                *raw_health.MANIFEST_COLUMNS,
                "timestamp_candidate_overlap",
                "timestamp_mask_applied",
                "mask_rect_crop_coordinates",
                "mask_width",
                "mask_height",
                "mask_area_ratio",
                "source_raw_relative_path",
            )
        )
    )
    write_csv(masked_path, masked_fields, masked_rows)
    return raw_path, masked_path, raw_rows, masked_rows


def file_fingerprint(path: Path) -> tuple[int, int, str]:
    return (
        path.stat().st_size,
        path.stat().st_mtime_ns,
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def tree_fingerprint(root: Path) -> dict[str, tuple[int, int, str]]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): file_fingerprint(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_streaming_excludes_test_and_preserves_global_date_buckets(
    tmp_path: Path,
) -> None:
    manifest = write_detection_fixture(tmp_path)
    stats = date_audit.stream_date_statistics(manifest)
    assert stats.development_rows == 60
    assert stats.test_rows == 7
    assert stats.test_dates == {"2099-12-31"}
    assert set(stats.development_by_date) == {
        "2021-01-01",
        "2021-01-02",
        "2021-01-03",
        "2021-01-04",
        "2021-01-05",
        "2021-01-06",
    }
    assert stats.development_by_date["2021-01-01"].rows == 15
    assert stats.development_by_date["2021-01-01"].current_splits == {
        "train": 10,
        "validation": 5,
    }
    assert sum(stats.development_health.values()) == 60
    assert all(
        date != "2099-12-31" for date in stats.development_by_date
    )


def test_global_date_assignment_is_exact_atomic_and_seed_deterministic(
    tmp_path: Path,
) -> None:
    stats = date_audit.stream_date_statistics(
        write_detection_fixture(tmp_path)
    )
    first = date_audit.find_minimal_global_date_assignment(
        stats, train_target=2, validation_target=1, seed=20260726
    )
    second = date_audit.find_minimal_global_date_assignment(
        stats, train_target=2, validation_target=1, seed=20260726
    )
    assert first == second
    assert first.feasible
    assert first.minimum_date_count == 3
    assert first.feasible_combinations_at_minimum == 8
    assert len(first.validation_dates) == 3
    assert set(first.train_dates).isdisjoint(first.validation_dates)
    assert set(first.train_dates) | set(first.validation_dates) == set(
        stats.development_by_date
    )
    assert first.capacity is not None
    assert first.capacity.exact_6000
    assert first.capacity.train_common >= 2
    assert first.capacity.validation_common >= 1
    assert first.capacity.target_capped_images == 30
    assert all(
        date in first.train_dates or date in first.validation_dates
        for date in stats.development_by_date
    )


def test_impossible_exact_assignment_reports_maximum_target_capped_count(
    tmp_path: Path,
) -> None:
    stats = date_audit.stream_date_statistics(
        write_detection_fixture(tmp_path)
    )
    result = date_audit.find_minimal_global_date_assignment(
        stats,
        train_target=4,
        validation_target=3,
        seed=20260726,
    )
    assert not result.feasible
    assert result.capacity is not None
    assert result.capacity.target_capped_images < 70
    assert result.capacity.target_capped_images > 0
    assert set(result.train_dates).isdisjoint(result.validation_dates)
    assert set(result.train_dates) | set(result.validation_dates) == set(
        stats.development_by_date
    )
    assert "그룹" in result.reason


def test_candidate_evaluation_detects_date_overlap_and_temporal_infeasibility(
    tmp_path: Path,
) -> None:
    stats = date_audit.stream_date_statistics(
        write_detection_fixture(tmp_path)
    )
    assignment = date_audit.find_minimal_global_date_assignment(
        stats, train_target=2, validation_target=1, seed=20260726
    )
    candidates = date_audit.evaluate_candidate_splits(
        stats,
        global_assignment=assignment,
        train_target=2,
        validation_target=1,
        seed=20260726,
    )
    by_name = {candidate.candidate: candidate for candidate in candidates}
    global_result = next(
        candidate
        for candidate in candidates
        if candidate.group_key == "capture_date"
        and candidate.exact_6000
    )
    assert global_result.global_date_overlap == 0
    assert global_result.declared_group_overlap == 0
    chronological = [
        candidate
        for candidate in candidates
        if "chronological" in candidate.candidate.lower()
    ]
    contiguous = [
        candidate
        for candidate in candidates
        if "contiguous" in candidate.candidate.lower()
    ]
    assert chronological and all(
        not candidate.exact_6000 for candidate in chronological
    )
    assert contiguous and all(
        not candidate.exact_6000 for candidate in contiguous
    )
    assert any(
        candidate.global_date_overlap > 0
        for candidate in candidates
        if candidate.group_key != "capture_date"
    )
    assert len(by_name) == len(candidates)


def test_raw_and_masked_pilot_manifests_are_one_to_one_and_count_masks(
    tmp_path: Path,
) -> None:
    raw_path, masked_path, raw_rows, _masked_rows = write_pilot_fixtures(
        tmp_path
    )
    audit = date_audit.audit_pilot_manifests(raw_path, masked_path)
    expected_masks = sum(
        row["timestamp_region_overlap"] == "True" for row in raw_rows
    )
    assert audit.raw_rows == 30
    assert audit.masked_rows == 30
    assert audit.identity_mismatches == 0
    assert audit.invariant_mismatches == 0
    assert audit.raw_duplicates == 0
    assert audit.masked_duplicates == 0
    assert audit.timestamp_candidates == expected_masks
    assert audit.masks_applied == expected_masks
    assert audit.mask_flag_mismatches == 0
    assert sum(audit.rows_by_split_species_status.values()) == 30
    assert sum(audit.timestamp_by_split_species_status.values()) == (
        expected_masks
    )
    assert sum(audit.masks_by_split_species_status.values()) == expected_masks


def test_manifest_mismatch_is_detected_without_reading_any_image(
    tmp_path: Path,
) -> None:
    raw_path, masked_path, _raw_rows, masked_rows = write_pilot_fixtures(
        tmp_path
    )
    changed = [dict(row) for row in masked_rows]
    changed[0]["camera_id"] = "999"
    changed.pop()
    with masked_path.open(encoding="utf-8-sig", newline="") as handle:
        fields = tuple(csv.DictReader(handle).fieldnames or ())
    write_csv(masked_path, fields, changed)
    audit = date_audit.audit_pilot_manifests(raw_path, masked_path)
    assert audit.masked_rows == 29
    assert audit.identity_mismatches > 0
    assert audit.invariant_mismatches > 0


def test_run_writes_only_deidentified_reports_and_preserves_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detection = write_detection_fixture(tmp_path)
    raw_path, masked_path, _raw_rows, _masked_rows = write_pilot_fixtures(
        tmp_path
    )
    artifacts = tmp_path / "artifacts"
    sentinel = artifacts / "models" / "nested" / "sentinel.bin"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"do-not-change")
    output = tmp_path / "reports" / "date_audit"
    input_before = {
        path: file_fingerprint(path)
        for path in (detection, raw_path, masked_path)
    }
    artifacts_before = tree_fingerprint(artifacts)
    monkeypatch.setattr(date_audit, "EXPECTED_PILOT_ROWS", 30)
    args = argparse.Namespace(
        detection_manifest=detection,
        raw_health_manifest=raw_path,
        masked_health_manifest=masked_path,
        output_dir=output,
        seed=20260726,
        train_per_species_status=2,
        validation_per_species_status=1,
    )
    result = date_audit.run(args)

    assert result["date_holdout_feasible"]
    assert result["date_overlap"] == 0
    assert result["test_rows_used"] == 0
    assert result["raw_masked_identity_mismatches"] == 0
    assert result["input_files_modified"] is False
    assert result["artifacts_modified"] is False
    assert {
        path.name for path in output.iterdir() if path.is_file()
    } == set(date_audit.OUTPUT_FILES)
    assert all(
        path.suffix.lower() in {".csv", ".md"}
        for path in output.iterdir()
        if path.is_file()
    )
    assert {
        path: file_fingerprint(path)
        for path in (detection, raw_path, masked_path)
    } == input_before
    assert tree_fingerprint(artifacts) == artifacts_before
    for path in output.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8-sig")
        assert str(tmp_path) not in text
        assert "/tmp/" not in text
        assert "/mnt/" not in text
        assert "/home/" not in text
        assert ":\\\\" not in text

    mask_bias = (
        output / "health_timestamp_mask_bias.csv"
    ).read_text(encoding="utf-8-sig")
    assert "mask" in mask_bias.lower()
    assert any(character.isdigit() for character in mask_bias)
