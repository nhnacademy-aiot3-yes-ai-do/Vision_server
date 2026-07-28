from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
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


evaluation = load_module(
    "evaluate_health_classifier",
    PROJECT_ROOT / "scripts" / "evaluate_health_classifier.py",
)


MANIFEST_COLUMNS = (
    "split",
    "health_class_id",
    "health_class_name",
    "species",
    "task",
    "normality",
    "disease_type",
    "camera_id",
    "capture_date",
    "capture_time",
    "image_archive_id",
    "image_member",
    "original_width",
    "original_height",
    "valid_bbox_count",
    "union_bbox",
    "padded_crop_bbox",
    "padding_ratio",
    "crop_width",
    "crop_height",
    "output_relative_path",
)


def prediction(
    *,
    actual: int,
    predicted: int,
    healthy_probability: float,
    species: str = "느타리",
    capture_date: str = "2021-11-04",
    camera_id: str = "1",
    disease_type: str = "",
    serial: int = 0,
) -> object:
    return evaluation.Prediction(
        actual_class_id=actual,
        predicted_class_id=predicted,
        healthy_probability=healthy_probability,
        disease_probability=1.0 - healthy_probability,
        species=species,
        capture_date=capture_date,
        camera_id=camera_id,
        disease_type=disease_type,
        image_member=f"생육/source_{serial:03d}.jpg",
        output_relative_path=(
            f"val/{actual}_{'healthy' if actual == 0 else 'disease_suspected'}"
            f"/image_{serial:03d}.jpg"
        ),
    )


def manifest_row(
    *,
    split: str = "validation",
    class_id: int = 0,
    relative_path: str = "val/0_healthy/image_000.jpg",
    serial: int = 0,
) -> dict[str, str]:
    healthy = class_id == 0
    return {
        "split": split,
        "health_class_id": str(class_id),
        "health_class_name": (
            "HEALTHY" if healthy else "DISEASE_SUSPECTED"
        ),
        "species": "느타리",
        "task": "생육" if healthy else "병해",
        "normality": "normal" if healthy else "abnormal",
        "disease_type": "" if healthy else "세균갈색무늬병",
        "camera_id": "1",
        "capture_date": "2021-11-04",
        "capture_time": "12:34:56",
        "image_archive_id": "VS1",
        "image_member": f"생육/source_{serial:03d}.jpg",
        "original_width": "100",
        "original_height": "80",
        "valid_bbox_count": "1",
        "union_bbox": '{"x":10,"y":10,"width":50,"height":40}',
        "padded_crop_bbox": '{"x":0,"y":0,"width":75,"height":64}',
        "padding_ratio": "0.15",
        "crop_width": "75",
        "crop_height": "64",
        "output_relative_path": relative_path,
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_image(path: Path, color: tuple[int, int, int] = (30, 90, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (75, 64), color).save(path, format="JPEG")


def test_confusion_matrix_and_binary_metrics() -> None:
    predictions = [
        prediction(
            actual=0,
            predicted=0,
            healthy_probability=0.90,
            serial=0,
        ),
        prediction(
            actual=0,
            predicted=1,
            healthy_probability=0.20,
            serial=1,
        ),
        prediction(
            actual=1,
            predicted=1,
            healthy_probability=0.10,
            disease_type="병A",
            serial=2,
        ),
        prediction(
            actual=1,
            predicted=1,
            healthy_probability=0.40,
            disease_type="병B",
            serial=3,
        ),
    ]

    metrics = evaluation.compute_binary_metrics(predictions)

    assert metrics["total"] == 4
    assert metrics["correct"] == 3
    assert metrics["confusion_matrix"]["labels"] == [
        "HEALTHY",
        "DISEASE_SUSPECTED",
    ]
    assert metrics["confusion_matrix"]["matrix"] == [[1, 1], [0, 2]]
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["macro_precision"] == pytest.approx(5 / 6)
    assert metrics["macro_recall"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx(11 / 15)

    healthy = metrics["per_class"]["HEALTHY"]
    disease = metrics["per_class"]["DISEASE_SUSPECTED"]
    assert (healthy["tp"], healthy["fp"], healthy["fn"]) == (1, 0, 1)
    assert healthy["precision"] == pytest.approx(1.0)
    assert healthy["recall"] == pytest.approx(0.5)
    assert healthy["f1"] == pytest.approx(2 / 3)
    assert (disease["tp"], disease["fp"], disease["fn"]) == (2, 1, 0)
    assert disease["precision"] == pytest.approx(2 / 3)
    assert disease["recall"] == pytest.approx(1.0)
    assert disease["f1"] == pytest.approx(0.8)


def test_species_date_and_camera_group_aggregation() -> None:
    predictions = [
        prediction(
            actual=0,
            predicted=0,
            healthy_probability=0.9,
            species="느타리",
            capture_date="2021-11-04",
            camera_id="1",
            serial=0,
        ),
        prediction(
            actual=1,
            predicted=1,
            healthy_probability=0.1,
            species="느타리",
            capture_date="2021-11-04",
            camera_id="1",
            disease_type="병A",
            serial=1,
        ),
        prediction(
            actual=0,
            predicted=1,
            healthy_probability=0.2,
            species="표고",
            capture_date="2021-11-26",
            camera_id="6",
            serial=2,
        ),
        prediction(
            actual=1,
            predicted=1,
            healthy_probability=0.2,
            species="표고",
            capture_date="2021-12-04",
            camera_id="6",
            disease_type="병B",
            serial=3,
        ),
    ]

    by_species = evaluation.aggregate_group_metrics(predictions, "species")
    by_date = evaluation.aggregate_group_metrics(predictions, "capture_date")
    by_camera = evaluation.aggregate_group_metrics(predictions, "camera_id")

    assert set(by_species) == {"느타리", "표고"}
    assert by_species["느타리"]["accuracy"] == pytest.approx(1.0)
    assert by_species["표고"]["accuracy"] == pytest.approx(0.5)
    assert by_date["2021-11-04"]["total"] == 2
    assert by_date["2021-11-04"]["accuracy"] == pytest.approx(1.0)
    assert by_date["2021-11-26"]["accuracy"] == pytest.approx(0.0)
    assert by_camera["1"]["accuracy"] == pytest.approx(1.0)
    assert by_camera["6"]["accuracy"] == pytest.approx(0.5)
    assert (
        by_camera["6"]["per_class"]["DISEASE_SUSPECTED"]["recall"]
        == pytest.approx(1.0)
    )


def test_uncertain_threshold_analysis() -> None:
    predictions = [
        prediction(
            actual=0,
            predicted=0,
            healthy_probability=0.55,
            serial=0,
        ),
        prediction(
            actual=1,
            predicted=1,
            healthy_probability=0.10,
            disease_type="병A",
            serial=1,
        ),
        prediction(
            actual=1,
            predicted=0,
            healthy_probability=0.96,
            disease_type="병B",
            serial=2,
        ),
    ]

    rows = evaluation.analyze_thresholds(predictions, thresholds=(0.60, 0.95))

    assert [row["threshold"] for row in rows] == [0.60, 0.95]
    at_060, at_095 = rows
    assert at_060["total_count"] == 3
    assert at_060["auto_decision_count"] == 2
    assert at_060["uncertain_count"] == 1
    assert at_060["auto_decision_rate"] == pytest.approx(2 / 3)
    assert at_060["uncertain_rate"] == pytest.approx(1 / 3)
    assert at_060["auto_decision_accuracy"] == pytest.approx(0.5)
    assert at_060["healthy_recall"] == pytest.approx(0.0)
    assert at_060["disease_suspected_recall"] == pytest.approx(0.5)
    assert at_060["disease_as_healthy_count"] == 1

    assert at_095["auto_decision_count"] == 1
    assert at_095["uncertain_count"] == 2
    assert at_095["auto_decision_accuracy"] == pytest.approx(0.0)
    assert at_095["healthy_recall"] == pytest.approx(0.0)
    assert at_095["disease_suspected_recall"] == pytest.approx(0.0)
    assert at_095["disease_as_healthy_count"] == 1


def test_error_review_selection_is_misclassified_and_confidence_sorted() -> None:
    predictions = [
        prediction(
            actual=0,
            predicted=0,
            healthy_probability=0.999,
            serial=0,
        ),
        prediction(
            actual=1,
            predicted=0,
            healthy_probability=0.80,
            disease_type="병A",
            serial=1,
        ),
        prediction(
            actual=0,
            predicted=1,
            healthy_probability=0.10,
            serial=2,
        ),
        prediction(
            actual=1,
            predicted=0,
            healthy_probability=0.99,
            disease_type="병B",
            serial=3,
        ),
    ]

    selected = evaluation.select_error_reviews(predictions, max_errors=2)

    assert len(selected) == 2
    assert all(row.actual_class_id != row.predicted_class_id for row in selected)
    assert [row.output_relative_path for row in selected] == [
        "val/1_disease_suspected/image_003.jpg",
        "val/0_healthy/image_002.jpg",
    ]
    assert [
        max(row.healthy_probability, row.disease_probability)
        for row in selected
    ] == pytest.approx([0.99, 0.90])


def test_load_validation_records_rejects_test_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            manifest_row(),
            manifest_row(
                split="test",
                class_id=1,
                relative_path="test/1_disease_suspected/image_001.jpg",
                serial=1,
            ),
        ],
    )
    write_image(tmp_path / "val" / "0_healthy" / "image_000.jpg")

    with pytest.raises((ValueError, RuntimeError), match="(?i)test"):
        evaluation.load_validation_records(
            manifest,
            tmp_path / "val",
            expected_count=None,
        )


@pytest.mark.parametrize("create_image", [False, True])
def test_manifest_and_validation_image_mismatch_is_detected(
    tmp_path: Path,
    create_image: bool,
) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [manifest_row()])
    if create_image:
        # The expected file exists, but an untracked validation image makes the
        # manifest-to-directory comparison non-bijective.
        write_image(tmp_path / "val" / "0_healthy" / "image_000.jpg")
        write_image(tmp_path / "val" / "0_healthy" / "extra.jpg")

    with pytest.raises((ValueError, RuntimeError), match="(?i)(manifest|image|file)"):
        evaluation.load_validation_records(
            manifest,
            tmp_path / "val",
            expected_count=None,
        )


def test_validation_manifest_and_images_match_one_to_one(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    rows = [
        manifest_row(),
        manifest_row(
            class_id=1,
            relative_path="val/1_disease_suspected/image_001.jpg",
            serial=1,
        ),
    ]
    write_manifest(manifest, rows)
    write_image(tmp_path / "val" / "0_healthy" / "image_000.jpg")
    write_image(
        tmp_path / "val" / "1_disease_suspected" / "image_001.jpg",
        (90, 30, 30),
    )

    records = evaluation.load_validation_records(
        manifest,
        tmp_path / "val",
        expected_count=2,
    )

    assert len(records) == 2
    assert {
        record.output_relative_path
        if hasattr(record, "output_relative_path")
        else record["output_relative_path"]
        for record in records
    } == {
        "val/0_healthy/image_000.jpg",
        "val/1_disease_suspected/image_001.jpg",
    }


@pytest.mark.parametrize(
    "relative_path",
    [
        "../outside.jpg",
        "val/0_healthy/../../../outside.jpg",
        "/mnt/d/private/image.jpg",
        r"C:\Users\private\image.jpg",
    ],
)
def test_manifest_image_reference_cannot_escape_validation_directory(
    tmp_path: Path,
    relative_path: str,
) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [manifest_row(relative_path=relative_path)],
    )

    with pytest.raises((ValueError, RuntimeError), match="(?i)(path|relative|escape)"):
        evaluation.load_validation_records(
            manifest,
            tmp_path / "val",
            expected_count=None,
        )


def test_deidentified_output_scan_accepts_safe_and_rejects_local_paths(
    tmp_path: Path,
) -> None:
    safe = tmp_path / "safe.csv"
    safe.write_text(
        "image_member,output_relative_path\n"
        "생육/source.jpg,val/0_healthy/image.jpg\n",
        encoding="utf-8",
    )
    evaluation.assert_deidentified_outputs([safe])

    leaked = tmp_path / "leaked.md"
    leaked.write_text(
        "source: /mnt/d/버섯/private.jpg\n"
        "home: /home/kim75/projects/mushroom-vision\n",
        encoding="utf-8",
    )
    with pytest.raises((ValueError, RuntimeError), match="(?i)(absolute|path|mnt|home)"):
        evaluation.assert_deidentified_outputs([leaked])


def test_invalid_probabilities_are_rejected(tmp_path: Path) -> None:
    record = evaluation.ValidationRecord(
        actual_class_id=0,
        actual_class_name="HEALTHY",
        species="느타리",
        capture_date="2021-11-04",
        camera_id="1",
        disease_type="",
        image_archive_id="VS1",
        image_member="생육/source.jpg",
        output_relative_path="val/0_healthy/image.jpg",
        image_path=tmp_path / "image.jpg",
    )

    for invalid in ([0.4, 0.4], [1.1, -0.1], [float("nan"), 0.5], [1.0]):
        with pytest.raises(ValueError, match="(?i)(probability|tensor)"):
            evaluation.predictions_from_probabilities([record], [invalid])


def test_error_review_writes_selected_images_and_contact_sheet(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "source.jpg"
    write_image(image_path)
    records: dict[str, object] = {}
    predictions = []
    for serial, confidence in enumerate((0.99, 0.80)):
        item = prediction(
            actual=1,
            predicted=0,
            healthy_probability=confidence,
            disease_type=f"병{serial}",
            serial=serial,
        )
        predictions.append(item)
        records[item.output_relative_path] = evaluation.ValidationRecord(
            actual_class_id=1,
            actual_class_name="DISEASE_SUSPECTED",
            species=item.species,
            capture_date=item.capture_date,
            camera_id=item.camera_id,
            disease_type=item.disease_type,
            image_archive_id="VS1",
            image_member=item.image_member,
            output_relative_path=item.output_relative_path,
            image_path=image_path,
        )

    review_map, contact_sheet, count = evaluation.create_error_review(
        predictions,
        records,
        tmp_path / "evaluation",
        max_errors=1,
    )

    assert count == 1
    assert set(review_map) == {predictions[0].output_relative_path}
    assert len(list((tmp_path / "evaluation" / "error_review").glob("*.jpg"))) == 1
    assert contact_sheet.is_file()
    with Image.open(contact_sheet) as opened:
        opened.verify()


def test_report_metadata_and_zero_error_disease_types(
    tmp_path: Path,
) -> None:
    predictions = [
        prediction(
            actual=0,
            predicted=0,
            healthy_probability=0.9,
            serial=0,
        ),
        prediction(
            actual=1,
            predicted=1,
            healthy_probability=0.1,
            capture_date="2021-11-26",
            disease_type="병A",
            serial=1,
        ),
        prediction(
            actual=1,
            predicted=0,
            healthy_probability=0.9,
            capture_date="2021-11-26",
            disease_type="병B",
            serial=2,
        ),
    ]
    records = [object(), object(), object()]
    report_dir = tmp_path / "reports"

    evaluation._write_evaluation_reports(
        report_dir,
        predictions,
        records,
        model_reference="artifacts/model/best.pt",
        manifest_reference="artifacts/manifest.csv",
        validation_reference="artifacts/val",
        model_class_names=evaluation.MODEL_CLASS_NAMES,
        review_relative_by_output={},
        review_count=0,
        review_limit=7,
        contact_sheet_reference="artifacts/evaluation/contact.jpg",
        image_size=384,
    )

    payload = json.loads(
        (report_dir / "health_classifier_overall_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["scope"]["image_size"] == 384
    assert payload["error_review"]["review_limit"] == 7
    assert payload["disease_as_healthy_by_disease_type"] == {
        "병A": 0,
        "병B": 1,
    }
    date_rows = list(
        csv.DictReader(
            (report_dir / "health_classifier_date_metrics.csv").open(
                encoding="utf-8-sig",
                newline="",
            )
        )
    )
    healthy_only = next(
        row for row in date_rows if row["capture_date"] == "2021-11-04"
    )
    assert healthy_only["disease_suspected_support"] == "0"
    assert healthy_only["disease_suspected_recall"] == ""


def test_output_activation_restores_prior_targets_when_backup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".evaluation.tmp-test"
    staged_evaluation = staging / "evaluation"
    staged_report = staging / "reports" / "report.csv"
    staged_evaluation.mkdir(parents=True)
    staged_report.parent.mkdir()
    (staged_evaluation / "new.txt").write_text("new", encoding="utf-8")
    staged_report.write_text("new report", encoding="utf-8")

    evaluation_target = tmp_path / "evaluation"
    evaluation_target.mkdir()
    (evaluation_target / "old.txt").write_text("old", encoding="utf-8")
    report_target = tmp_path / "report.csv"
    report_target.write_text("old report", encoding="utf-8")
    real_replace = evaluation.os.replace
    failed = False

    def flaky_replace(source: object, destination: object) -> None:
        nonlocal failed
        if Path(source) == report_target and not failed:
            failed = True
            raise OSError("synthetic backup failure")
        real_replace(source, destination)

    monkeypatch.setattr(evaluation.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="synthetic"):
        evaluation._activate_outputs(
            staging,
            staged_evaluation,
            evaluation_target,
            {report_target: staged_report},
        )

    assert (evaluation_target / "old.txt").read_text(encoding="utf-8") == "old"
    assert report_target.read_text(encoding="utf-8") == "old report"


def test_inference_uses_only_validated_paths_and_restores_manifest_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeScalar:
        def __init__(self, value: float):
            self.value = value

        def detach(self) -> "FakeScalar":
            return self

        def cpu(self) -> "FakeScalar":
            return self

        def __float__(self) -> float:
            return self.value

    class FakeVector:
        def __init__(self, values: list[float]):
            self.values = values

        def detach(self) -> "FakeVector":
            return self

        def cpu(self) -> "FakeVector":
            return self

        def tolist(self) -> list[float]:
            return self.values

    class FakeProbs:
        def __init__(self, values: list[float]):
            self.data = FakeVector(values)
            self.top1 = 0 if values[0] >= values[1] else 1
            self.top1conf = FakeScalar(max(values))

    class FakeResult:
        def __init__(self, path: str, values: list[float]):
            self.path = path
            self.probs = FakeProbs(values)

    accessed: list[Path] = []

    class FakeYOLO:
        names = evaluation.MODEL_CLASS_NAMES

        def __init__(self, model_path: str, task: str):
            assert Path(model_path).name == "best.pt"
            assert task == "classify"

        def predict(self, *, source: str, **_kwargs: object) -> list[FakeResult]:
            paths = sorted(
                Path(line)
                for line in Path(source).read_text(encoding="utf-8").splitlines()
            )
            accessed.extend(paths)
            return [
                FakeResult(
                    str(path),
                    [0.9, 0.1] if path.name == "b.jpg" else [0.2, 0.8],
                )
                for path in paths
            ]

    fake_module = types.ModuleType("ultralytics")
    fake_module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"synthetic")
    image_b = tmp_path / "selected" / "b.jpg"
    image_a = tmp_path / "selected" / "a.jpg"
    image_b.parent.mkdir()
    image_b.write_bytes(b"b")
    image_a.write_bytes(b"a")
    records = [
        evaluation.ValidationRecord(
            0,
            "HEALTHY",
            "느타리",
            "2021-11-04",
            "1",
            "",
            "VS1",
            "생육/b.jpg",
            "val/0_healthy/b.jpg",
            image_b,
        ),
        evaluation.ValidationRecord(
            1,
            "DISEASE_SUSPECTED",
            "느타리",
            "2021-11-26",
            "2",
            "병A",
            "VS1",
            "병해/a.jpg",
            "val/1_disease_suspected/a.jpg",
            image_a,
        ),
    ]

    predictions, names = evaluation.run_ultralytics_inference(
        model_path,
        records,
        batch_size=2,
        image_size=320,
        device="cpu",
        runtime_dir=tmp_path / "runtime",
        show_progress=False,
    )

    assert names == evaluation.MODEL_CLASS_NAMES
    assert accessed == sorted((image_a.resolve(), image_b.resolve()))
    assert [item.output_relative_path for item in predictions] == [
        "val/0_healthy/b.jpg",
        "val/1_disease_suspected/a.jpg",
    ]
    assert [item.predicted_class_id for item in predictions] == [0, 1]
