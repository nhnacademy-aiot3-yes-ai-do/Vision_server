from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
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


evaluation = load_module(
    "evaluate_health_end_to_end",
    PROJECT_ROOT / "scripts" / "evaluate_health_end_to_end.py",
)


MANIFEST_COLUMNS = (
    "split",
    "actual_species",
    "actual_health_status",
    "capture_date",
    "camera_id",
    "image_archive_id",
    "image_member",
    "output_relative_path",
)


def row(
    *,
    actual_species: str,
    actual_health_status: str,
    species_detection_success: bool,
    detector_health_status: str,
    gt_crop_health_status: str,
    serial: int,
    split: str = "validation",
) -> dict[str, object]:
    return {
        "split": split,
        "actual_species": actual_species,
        "actual_health_status": actual_health_status,
        "species_detection_success": species_detection_success,
        "detector_health_status": detector_health_status,
        "gt_crop_health_status": gt_crop_health_status,
        "capture_date": "2021-11-04",
        "camera_id": "1",
        "image_archive_id": "VS1",
        "image_member": f"생육/source_{serial:03d}.jpg",
        "output_relative_path": (
            f"val/{'0_healthy' if actual_health_status == 'HEALTHY' else '1_disease_suspected'}"
            f"/image_{serial:03d}.jpg"
        ),
    }


def manifest_row(
    *,
    split: str = "validation",
    species: str = "느타리",
    health_status: str = "HEALTHY",
    serial: int = 0,
) -> dict[str, str]:
    class_directory = (
        "0_healthy"
        if health_status == "HEALTHY"
        else "1_disease_suspected"
    )
    return {
        "split": split,
        "actual_species": species,
        "actual_health_status": health_status,
        "capture_date": "2021-11-04",
        "camera_id": "1",
        "image_archive_id": "VS1",
        "image_member": f"생육/source_{serial:03d}.jpg",
        "output_relative_path": (
            f"val/{class_directory}/image_{serial:03d}.jpg"
        ),
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compute_end_to_end_metrics_includes_detection_species_health() -> None:
    rows = [
        row(
            actual_species="느타리",
            actual_health_status="HEALTHY",
            species_detection_success=True,
            detector_health_status="HEALTHY",
            gt_crop_health_status="HEALTHY",
            serial=0,
        ),
        row(
            actual_species="표고",
            actual_health_status="DISEASE_SUSPECTED",
            species_detection_success=True,
            detector_health_status="HEALTHY",
            gt_crop_health_status="DISEASE_SUSPECTED",
            serial=1,
        ),
        row(
            actual_species="양송이",
            actual_health_status="HEALTHY",
            species_detection_success=True,
            detector_health_status="HEALTHY",
            gt_crop_health_status="HEALTHY",
            serial=2,
        ),
        row(
            actual_species="팽이",
            actual_health_status="DISEASE_SUSPECTED",
            species_detection_success=False,
            detector_health_status="",
            gt_crop_health_status="DISEASE_SUSPECTED",
            serial=3,
        ),
        row(
            actual_species="큰느타리",
            actual_health_status="DISEASE_SUSPECTED",
            species_detection_success=True,
            detector_health_status="UNCERTAIN",
            gt_crop_health_status="DISEASE_SUSPECTED",
            serial=4,
        ),
    ]

    metrics = evaluation.compute_end_to_end_metrics(
        rows,
        health_field="detector_health_status",
        require_detection=True,
    )

    assert metrics["total"] == 5
    assert metrics["detection_success_count"] == 4
    assert metrics["detection_success_rate"] == pytest.approx(0.8)
    assert metrics["health_analyzable_count"] == 4
    assert metrics["health_analyzable_rate"] == pytest.approx(0.8)
    assert metrics["accuracy"] == pytest.approx(0.4)
    assert metrics["macro_f1"] == pytest.approx(0.4)
    assert metrics["healthy_recall"] == pytest.approx(1.0)
    assert metrics["disease_suspected_recall"] == pytest.approx(0.0)
    assert metrics["uncertain_count"] == 1
    assert metrics["uncertain_rate"] == pytest.approx(0.2)
    assert metrics["disease_as_healthy_count"] == 1

    # Ground-truth union crops isolate classifier quality from detector misses.
    gt_crop = evaluation.compute_end_to_end_metrics(
        rows,
        health_field="gt_crop_health_status",
        require_detection=False,
    )
    assert gt_crop["health_analyzable_count"] == 5
    assert gt_crop["health_analyzable_rate"] == pytest.approx(1.0)
    assert gt_crop["accuracy"] == pytest.approx(1.0)
    assert gt_crop["macro_f1"] == pytest.approx(1.0)
    assert gt_crop["disease_as_healthy_count"] == 0


def test_test_rows_are_rejected_before_evaluation(tmp_path: Path) -> None:
    manifest = tmp_path / "evaluation_manifest.csv"
    write_manifest(
        manifest,
        [
            manifest_row(),
            manifest_row(
                split="test",
                species="표고",
                health_status="DISEASE_SUSPECTED",
                serial=1,
            ),
        ],
    )

    with pytest.raises((ValueError, RuntimeError), match="(?i)test"):
        evaluation.load_evaluation_candidates(manifest)


def test_selected_source_in_fixed_detection_test_split_is_rejected(
    tmp_path: Path,
) -> None:
    selected = [
        {
            "actual_species": "느타리",
            "image_archive_id": "VS1",
            "image_member": "생육/selected.jpg",
        }
    ]
    detection_manifest = tmp_path / "detection_manifest.csv"
    with detection_manifest.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "split",
                "species",
                "image_archive_id",
                "image_member",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "test",
                "species": "느타리",
                "image_archive_id": "VS1",
                "image_member": "생육/selected.jpg",
            }
        )

    with pytest.raises((ValueError, RuntimeError), match="(?i)test"):
        evaluation.verify_selected_detection_sources(
            selected,
            detection_manifest,
        )


def test_deterministic_stratified_selection_and_no_group_splitting() -> None:
    candidates: list[dict[str, str]] = []
    serial = 0
    for species in ("느타리", "표고"):
        for health_status in ("HEALTHY", "DISEASE_SUSPECTED"):
            for camera_id in ("1", "2", "3"):
                for index in range(3):
                    candidates.append(
                        {
                            "split": "validation",
                            "actual_species": species,
                            "actual_health_status": health_status,
                            "capture_date": f"2021-11-0{camera_id}",
                            "camera_id": camera_id,
                            "group_key": json.dumps(
                                [species, camera_id],
                                ensure_ascii=False,
                            ),
                            "image_member": (
                                f"{health_status}/{serial:03d}.jpg"
                            ),
                            "output_relative_path": (
                                f"val/{health_status}/{serial:03d}.jpg"
                            ),
                        }
                    )
                    serial += 1

    first = evaluation.deterministic_stratified_sample(
        candidates,
        per_species_status=3,
        seed=20260726,
    )
    second = evaluation.deterministic_stratified_sample(
        list(reversed(candidates)),
        per_species_status=3,
        seed=20260726,
    )

    assert [
        item["image_member"] for item in first
    ] == [item["image_member"] for item in second]
    counts: dict[tuple[str, str], int] = {}
    for item in first:
        key = (
            item["actual_species"],
            item["actual_health_status"],
        )
        counts[key] = counts.get(key, 0) + 1
    assert set(counts.values()) == {3}

    selected_groups = {item["group_key"] for item in first}
    for group in selected_groups:
        assert {
            item["image_member"]
            for item in first
            if item["group_key"] == group
        } == {
            item["image_member"]
            for item in candidates
            if item["group_key"] == group
        }


def test_overlap_audit_detects_image_date_and_camera_overlap() -> None:
    evaluation_rows = [
        {
            "actual_species": "느타리",
            "camera_id": "1",
            "capture_date": "2021-11-04",
            "image_archive_id": "VS1",
            "image_member": "생육/shared.jpg",
        },
        {
            "actual_species": "표고",
            "camera_id": "6",
            "capture_date": "2021-11-26",
            "image_archive_id": "VS5",
            "image_member": "병해/eval_only.jpg",
        },
    ]
    training_rows = [
        {
            "actual_species": "느타리",
            "camera_id": "1",
            "capture_date": "2021-11-01",
            "image_archive_id": "VS1",
            "image_member": "생육/shared.jpg",
        },
        {
            "actual_species": "양송이",
            "camera_id": "9",
            "capture_date": "2021-11-26",
            "image_archive_id": "VS2",
            "image_member": "생육/train_only.jpg",
        },
    ]

    audit = evaluation.audit_evaluation_overlap(
        evaluation_rows,
        training_rows,
    )

    assert audit["image_overlap_count"] == 1
    assert audit["capture_date_overlap_count"] == 1
    assert audit["camera_id_overlap_count"] == 1
    assert audit["species_camera_overlap_count"] == 1


def test_geometry_iou_and_bidirectional_coverage_known_values() -> None:
    half_overlap = evaluation._geometry_comparison(
        (0, 0, 10, 10),
        (5, 0, 15, 10),
    )
    assert half_overlap == pytest.approx(
        {
            "iou": 1 / 3,
            "gt_coverage": 0.5,
            "detector_coverage": 0.5,
            "area_ratio": 1.0,
        }
    )

    detector_contains_gt = evaluation._geometry_comparison(
        (0, 0, 20, 20),
        (5, 5, 15, 15),
    )
    assert detector_contains_gt == pytest.approx(
        {
            "iou": 0.25,
            "gt_coverage": 1.0,
            "detector_coverage": 0.25,
            "area_ratio": 4.0,
        }
    )


def test_error_reasons_include_unexpected_detected_species() -> None:
    reasons = evaluation._error_reasons(
        {
            "actual_species": "느타리",
            "actual_health_status": "HEALTHY",
            "species_detection_success": True,
            "exact_species_match": False,
            "unexpected_species": "표고",
            "detector_health_status": "HEALTHY",
            "gt_crop_health_status": "HEALTHY",
        }
    )

    assert "UNEXPECTED_SPECIES_DETECTED" in reasons


def test_metrics_reject_health_status_when_species_detection_failed() -> None:
    contradictory = row(
        actual_species="느타리",
        actual_health_status="HEALTHY",
        species_detection_success=False,
        detector_health_status="HEALTHY",
        gt_crop_health_status="HEALTHY",
        serial=99,
    )

    with pytest.raises((ValueError, RuntimeError), match="탐지 실패"):
        evaluation.compute_end_to_end_metrics(
            [contradictory],
            health_field="detector_health_status",
            require_detection=True,
        )


def test_json_outputs_are_deidentified(tmp_path: Path) -> None:
    safe = tmp_path / "safe.json"
    safe.write_text(
        json.dumps(
            {
                "image_member": "생육/source.jpg",
                "output_relative_path": "val/0_healthy/image.jpg",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    evaluation.assert_deidentified_json([safe])

    leaked = tmp_path / "leaked.json"
    leaked.write_text(
        json.dumps({"path": "/home/kim75/private.jpg"}),
        encoding="utf-8",
    )
    with pytest.raises((ValueError, RuntimeError), match="(?i)(path|absolute|home)"):
        evaluation.assert_deidentified_json([leaked])


def test_evaluation_with_fake_predictor_preserves_inputs_and_model(
    tmp_path: Path,
) -> None:
    model = tmp_path / "best.pt"
    image = tmp_path / "val" / "0_healthy" / "image_000.jpg"
    manifest = tmp_path / "manifest.csv"
    model.write_bytes(b"fake immutable model")
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake immutable input")
    write_manifest(manifest, [manifest_row()])
    before = {
        path: (
            sha256(path),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in (model, image, manifest)
    }

    class FakePredictor:
        def __init__(self) -> None:
            self.paths: list[Path] = []

        def __call__(self, candidate: dict[str, str]) -> dict[str, object]:
            candidate_path = (
                tmp_path / str(candidate["output_relative_path"])
            )
            self.paths.append(candidate_path)
            return row(
                actual_species=str(candidate["actual_species"]),
                actual_health_status=str(candidate["actual_health_status"]),
                species_detection_success=True,
                detector_health_status=str(
                    candidate["actual_health_status"]
                ),
                gt_crop_health_status=str(
                    candidate["actual_health_status"]
                ),
                serial=0,
            )

    predictor = FakePredictor()
    candidates = evaluation.load_evaluation_candidates(manifest)
    result = evaluation.evaluate_candidates(
        candidates,
        predictor=predictor,
    )

    assert result["detector_metrics"]["total"] == 1
    assert result["detector_metrics"]["accuracy"] == pytest.approx(1.0)
    assert result["gt_crop_metrics"]["accuracy"] == pytest.approx(1.0)
    assert predictor.paths == [image]
    assert {
        path: (
            sha256(path),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in (model, image, manifest)
    } == before
