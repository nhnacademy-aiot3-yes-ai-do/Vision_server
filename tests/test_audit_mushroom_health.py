from __future__ import annotations

import csv
import importlib.util
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_mushroom_health.py"
BASE_SCRIPT = PROJECT_ROOT / "scripts" / "inspect_aihub_archives.py"


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
health = load_module("audit_mushroom_health", SCRIPT)


def record(
    *,
    species: str = "느타리",
    task: str = "생육",
    normality: str = "normal",
    disease: str | None = None,
    camera: str = "1",
    date: str = "2021-11-01",
) -> dict[str, object]:
    return {
        "species": species,
        "task": task,
        "normality": normality,
        "disease_type": disease,
        "camera_id": camera,
        "capture_date": date,
    }


def update_many(rows: list[dict[str, object]]) -> object:
    stats = health.HealthStats()
    for row in rows:
        health.update_health_stats(stats, row)
    return stats


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "split",
        "species",
        "task",
        "normality",
        "disease_type",
        "camera_id",
        "capture_date",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_normal_and_disease_health_label_validation() -> None:
    assert health.health_label_issues("생육", "normal", None) == (
        "HEALTHY",
        (),
    )
    assert health.health_label_issues(
        "병해", "abnormal", "푸른곰팡이병"
    ) == ("DISEASE_SUSPECTED", ())
    assert health.health_label_issues("생육", "unknown", None) == (
        "UNCERTAIN",
        ("invalid_normality", "task_normality_mismatch"),
    )


def test_detects_normal_with_disease_and_abnormal_without_disease() -> None:
    _state, normal_issues = health.health_label_issues(
        "생육", "normal", "푸른곰팡이병"
    )
    _state, abnormal_issues = health.health_label_issues(
        "병해", "abnormal", None
    )
    assert "normal_has_disease" in normal_issues
    assert "abnormal_missing_disease" in abnormal_issues


def test_species_and_disease_distribution_aggregation() -> None:
    stats = update_many(
        [
            record(species="느타리"),
            record(
                species="느타리",
                task="병해",
                normality="abnormal",
                disease="푸른곰팡이병",
            ),
            record(
                species="표고",
                task="병해",
                normality="abnormal",
                disease="흰곰팡이병",
                camera="2",
            ),
        ]
    )
    assert stats.total == 3
    assert stats.normality == Counter({"abnormal": 2, "normal": 1})
    assert stats.species_normality[("느타리", "abnormal")] == 1
    assert stats.species_disease[("표고", "흰곰팡이병")] == 1
    assert stats.disease == Counter(
        {"푸른곰팡이병": 1, "흰곰팡이병": 1}
    )
    assert stats.contradictory_records == 0


def test_camera_group_overlap_detection() -> None:
    train = {"느타리": ("1", "2"), "표고": ("3",)}
    validation = {"느타리": ("2", "4"), "표고": ("5",)}
    assert health.camera_overlap(train, validation) == {("느타리", "2")}
    assert not health.camera_overlap(
        {"느타리": ("1",)},
        {"느타리": ("2",)},
    )


def test_camera_assignment_is_seed_deterministic_and_disjoint() -> None:
    health_by_camera: dict[tuple[str, str], Counter[str]] = {}
    disease_by_camera: dict[tuple[str, str], Counter[str]] = {}
    for camera in range(1, 7):
        key = ("테스트품종", str(camera))
        health_by_camera[key] = Counter({"normal": 200, "abnormal": 200})
        disease_by_camera[key] = Counter(
            {
                "병A": 120 if camera % 2 else 80,
                "병B": 80 if camera % 2 else 120,
            }
        )
    first = health.choose_camera_holdout(
        health_by_camera,
        disease_by_camera,
        species_values=("테스트품종",),
        seed=20260726,
        train_target=500,
        validation_target=100,
    )
    second = health.choose_camera_holdout(
        health_by_camera,
        disease_by_camera,
        species_values=("테스트품종",),
        seed=20260726,
        train_target=500,
        validation_target=100,
    )
    assert first.feasible
    assert first == second
    assert not health.camera_overlap(
        first.train_cameras, first.validation_cameras
    )


def test_manifest_stream_skips_test_before_development_aggregation(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            {
                "split": "train",
                "species": "느타리",
                "task": "생육",
                "normality": "normal",
                "disease_type": "",
                "camera_id": "1",
                "capture_date": "2021-11-01",
            },
            {
                "split": "validation",
                "species": "느타리",
                "task": "병해",
                "normality": "abnormal",
                "disease_type": "병A",
                "camera_id": "2",
                "capture_date": "2021-11-02",
            },
            {
                "split": "test",
                "species": "느타리",
                "task": "병해",
                "normality": "abnormal",
                "disease_type": "TEST_ONLY_DISEASE",
                "camera_id": "3",
                "capture_date": "2021-11-03",
            },
        ],
    )
    stats = health.analyze_detection_manifest(manifest)
    assert stats.total_rows == 3
    assert stats.test_rows == 1
    assert stats.development_rows == 2
    assert stats.development_disease == Counter({"병A": 1})
    assert "TEST_ONLY_DISEASE" not in stats.development_disease


def test_report_path_deidentification(tmp_path: Path) -> None:
    safe = tmp_path / "safe.md"
    safe.write_text(
        "source: ${MUSHROOM_DATASET}/labels/TL1.zip\n",
        encoding="utf-8",
    )
    health.assert_deidentified([safe])
    unsafe = tmp_path / "unsafe.md"
    unsafe.write_text("source: /home/example/private.zip\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="절대경로"):
        health.assert_deidentified([unsafe])


def test_synthetic_source_zip_and_existing_tree_remain_unchanged(
    tmp_path: Path,
) -> None:
    source_zip = tmp_path / "source" / "TL1_test.zip"
    source_zip.parent.mkdir()
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("생육/sample.json", b"{}")
    protected = tmp_path / "artifacts"
    protected.mkdir()
    protected_file = protected / "existing.txt"
    protected_file.write_text("unchanged", encoding="utf-8")
    zip_snapshot = health.snapshot_files([source_zip])
    tree_snapshot = health.snapshot_tree(protected)
    with zipfile.ZipFile(source_zip, "r") as archive:
        assert archive.namelist() == ["생육/sample.json"]
    health.assert_snapshot_unchanged(zip_snapshot)
    health.assert_tree_unchanged(protected, tree_snapshot)
