from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sqlite3
import sys
import zipfile
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "inspect_aihub_archives.py"
)
SPEC = importlib.util.spec_from_file_location("inspect_aihub_archives", SCRIPT_PATH)
assert SPEC and SPEC.loader
inspector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inspector
SPEC.loader.exec_module(inspector)


def write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def label_json(image_filename: str = "sample.jpg", species: str = "느타리") -> bytes:
    payload = {
        "INFO": {
            "DATASET_NAME": f"{species} 버섯(생육)",
            "CATEGORY_NAME": species,
            "CONTRIBUTOR": "",
        },
        "IMAGE": {"IMAGE_FILE_NAME": image_filename},
        "ANNOTATION_INFO": [
            {
                "BOUNDING_BOX_X_COORDINATE": 1,
                "BOUNDING_BOX_Y_COORDINATE": 2,
                "BOUNDING_BOX_WIDTH": 3,
                "BOUNDING_BOX_HEIGHT": 4,
                "SEGMENTATION": [1.0, 2.0, 3.0, 4.0],
            }
        ],
        "META": {
            "DBYHS_SPCHCKN": None,
            "DBYHS_NORMALITY_ALTERNATIVE": True,
            "IP_CAMERA_ID": 7,
            "TEMPERATURE": 18.5,
            "HUMIDITY": 91.2,
            "ILLUMINATION_INTENSITY": 0.0,
            "CARBON_DIOXIDE": 1200.0,
            "IMAGE_CREATE_DATE": "2021-11-01",
            "STIPE_LENGTH": None,
            "STIPE_THICKNESS": None,
            "PILEUS_DIAMETER": None,
            "PILEUS_THICKNESS": None,
            "GROSS_WEIGHT": None,
        },
    }
    return b"\xef\xbb\xbf" + json.dumps(payload, ensure_ascii=False).encode("utf-8")


def full_label_json(
    *,
    image_filename: str,
    species: str = "느타리",
    camera_id: int | None = 1,
    capture_date: str | None = "2021-11-01",
    capture_time: str | None = "12:00:00",
    temperature: float | None = 20.0,
    humidity: float | None = 80.0,
    co2: float | None = 1000.0,
    illumination: float | None = 0.0,
    annotations: list[dict[str, object]] | None = None,
    normality: bool | None = True,
    disease: str | None = None,
    extra_info: dict[str, object] | None = None,
) -> bytes:
    info: dict[str, object] = {
        "DATASET_NAME": f"{species} 버섯(생육)",
        "CATEGORY_NAME": species,
        "CONTRIBUTOR": "",
    }
    info.update(extra_info or {})
    payload = {
        "INFO": info,
        "IMAGE": {"IMAGE_FILE_NAME": image_filename},
        "ANNOTATION_INFO": annotations if annotations is not None else [],
        "META": {
            "DBYHS_SPCHCKN": disease,
            "DBYHS_NORMALITY_ALTERNATIVE": normality,
            "IP_CAMERA_ID": camera_id,
            "TEMPERATURE": temperature,
            "HUMIDITY": humidity,
            "ILLUMINATION_INTENSITY": illumination,
            "CARBON_DIOXIDE": co2,
            "IMAGE_CREATE_DATE": capture_date,
            "IMAGE_CREATE_TIME": capture_time,
            "STIPE_LENGTH": None,
            "STIPE_THICKNESS": None,
            "PILEUS_DIAMETER": None,
            "PILEUS_THICKNESS": None,
            "GROSS_WEIGHT": None,
        },
    }
    return b"\xef\xbb\xbf" + json.dumps(payload, ensure_ascii=False).encode("utf-8")


def archive_ref(
    path: Path,
    *,
    split: str = "train",
    kind: str = "label",
    prefix: str = "TL",
    index: int = 1,
    species: str = "느타리",
) -> object:
    env_name = (
        "TRAIN_LABEL_ARCHIVES"
        if split == "train" and kind == "label"
        else "TRAIN_IMAGE_ARCHIVES"
        if split == "train"
        else "VAL_LABEL_ARCHIVES"
        if kind == "label"
        else "VAL_IMAGE_ARCHIVES"
    )
    return inspector.ArchiveRef(
        env_name=env_name,
        split=split,
        kind=kind,
        prefix=prefix,
        index=index,
        species=species,
        path=path,
    )


def test_cli_defaults_are_lightweight() -> None:
    args = inspector.parse_args([])
    assert args.sample_json_per_archive == 1
    assert args.sample_image_pairs == 0
    assert args.check_zip_integrity is False
    assert args.extract_samples is False
    assert args.scan_all_json is False


@pytest.mark.parametrize("value", ["-1", "21", "x"])
def test_sample_count_is_bounded(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        inspector.bounded_sample_count(value)


def test_decode_json_encoding_order_and_cp949_fallback() -> None:
    bom_text, bom_encoding = inspector.decode_json_bytes(
        b"\xef\xbb\xbf" + '{"name":"버섯"}'.encode("utf-8")
    )
    assert json.loads(bom_text)["name"] == "버섯"
    assert bom_encoding == "utf-8-sig"

    cp949_text, cp949_encoding = inspector.decode_json_bytes(
        '{"name":"버섯"}'.encode("cp949")
    )
    assert json.loads(cp949_text)["name"] == "버섯"
    assert cp949_encoding == "cp949"


def test_nested_paths_normalize_array_indices() -> None:
    values = inspector.collect_path_values(
        {"items": [{"box": [1, 2]}, {"box": None}]}
    )
    assert "$.items[]" in values
    assert "$.items[].box" in values
    assert "$.items[].box[]" in values
    assert values["$.items[].box[]"] == [1, 2]


def test_inventory_and_full_path_image_match(tmp_path: Path) -> None:
    label_path = tmp_path / "TL1_느타리.zip"
    image_path = tmp_path / "TS1_느타리.zip"
    write_zip(label_path, {"생육/sample.json": label_json()})
    write_zip(image_path, {"생육/sample.jpg": b"fake-image-bytes"})
    label_ref = archive_ref(label_path)
    image_ref = archive_ref(
        image_path, kind="image", prefix="TS"
    )
    label_inspection = inspector.inspect_archive(label_ref)
    image_inspection = inspector.inspect_archive(image_ref)

    assert label_inspection.zip_open_status == "central_directory_ok"
    assert label_inspection.json_count == 1
    assert image_inspection.image_count == 1
    assert label_inspection.integrity_status == "not_requested"
    assert label_inspection.corrupt_status == "unknown_full_check_not_requested"

    samples = inspector.read_json_samples([label_inspection], per_task=1)
    result = inspector.match_image(
        samples[0], inspector.build_image_indexes(image_inspection)
    )
    assert result.status == "matched"
    assert result.method == "full_path"
    assert result.matched_member == "생육/sample.jpg"


def test_safe_sample_extraction_preserves_member_path(tmp_path: Path) -> None:
    label_path = tmp_path / "TL1_느타리.zip"
    image_path = tmp_path / "TS1_느타리.zip"
    write_zip(label_path, {"생육/sample.json": label_json()})
    write_zip(image_path, {"생육/nested/sample.jpg": b"sample-image"})
    label_ref = archive_ref(label_path)
    image_ref = archive_ref(image_path, kind="image", prefix="TS")
    sample = inspector.JsonSample(
        archive=label_ref,
        member_name="생육/sample.json",
        task="생육",
    )
    result = inspector.MatchResult(
        sample=sample,
        image_filename="sample.jpg",
        matched_member="생육/nested/sample.jpg",
        method="filename",
        status="matched",
        candidate_count=1,
    )
    output_root = tmp_path / "extract"
    errors = inspector.extract_matched_samples(
        [result], [label_ref, image_ref], root=output_root
    )
    extracted = (
        output_root
        / "train"
        / "TS1_느타리"
        / "생육"
        / "nested"
        / "sample.jpg"
    )
    assert errors == []
    assert extracted.read_bytes() == b"sample-image"

    with pytest.raises(ValueError):
        inspector.safe_destination(output_root, "../escape.jpg")


def test_invalid_zip_is_reported_without_crashing(tmp_path: Path) -> None:
    bad_path = tmp_path / "TL1_느타리.zip"
    bad_path.write_bytes(b"not a zip")
    result = inspector.inspect_archive(archive_ref(bad_path))
    assert result.zip_open_status == "open_failed"
    assert result.corrupt_status == "yes"


def test_output_inside_source_tree_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    with pytest.raises(ValueError, match="원본 데이터 경로 밖"):
        inspector.validate_output_location(
            source / "reports", {"MUSHROOM_DATASET": source}
        )


def test_end_to_end_reports_with_synthetic_archives(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_root"
    directories = {
        "TRAIN_LABEL_ARCHIVES": dataset_root / "train_labels",
        "TRAIN_IMAGE_ARCHIVES": dataset_root / "train_images",
        "VAL_LABEL_ARCHIVES": dataset_root / "val_labels",
        "VAL_IMAGE_ARCHIVES": dataset_root / "val_images",
    }
    species_names = ("느타리", "양송이", "큰느타리", "팽이", "표고")
    for index, species in enumerate(species_names, start=1):
        for split, label_prefix, image_prefix, label_key, image_key in (
            (
                "train",
                "TL",
                "TS",
                "TRAIN_LABEL_ARCHIVES",
                "TRAIN_IMAGE_ARCHIVES",
            ),
            (
                "validation",
                "VL",
                "VS",
                "VAL_LABEL_ARCHIVES",
                "VAL_IMAGE_ARCHIVES",
            ),
        ):
            del split
            write_zip(
                directories[label_key] / f"{label_prefix}{index}_{species}.zip",
                {"생육/sample.json": label_json(species=species)},
            )
            write_zip(
                directories[image_key] / f"{image_prefix}{index}_{species}.zip",
                {"생육/sample.jpg": b"image"},
            )
    output_dir = tmp_path / "reports"
    args = argparse.Namespace(
        sample_json_per_archive=1,
        sample_image_pairs=1,
        check_zip_integrity=False,
        extract_samples=False,
        scan_all_json=False,
        output_dir=output_dir,
    )
    environ = {
        "MUSHROOM_DATASET": str(dataset_root),
        **{key: str(value) for key, value in directories.items()},
    }
    status = inspector.run_analysis(args, environ)
    assert status == 0
    expected = {
        "archive_inventory.csv",
        "archive_pairing.md",
        "label_schema.json",
        "label_distribution.csv",
        "unmatched_sample_files.csv",
        "sample_inspection.md",
        "dataset_inventory.md",
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    schema = json.loads((output_dir / "label_schema.json").read_text())
    assert any(
        item["path"] == "$.IMAGE.IMAGE_FILE_NAME"
        for item in schema["all_field_paths"]
    )
    pairing = (output_dir / "archive_pairing.md").read_text()
    assert pairing.count("| ok |") == 10
    inventory = (output_dir / "archive_inventory.csv").read_text(
        encoding="utf-8-sig"
    )
    dataset_report = (output_dir / "dataset_inventory.md").read_text()
    assert str(tmp_path) not in inventory
    assert str(tmp_path) not in dataset_report
    assert "${MUSHROOM_DATASET}/" in inventory


def test_full_json_streaming_aggregates_and_handles_corruption(
    tmp_path: Path,
) -> None:
    bbox = {
        "BOUNDING_BOX_X_COORDINATE": 0,
        "BOUNDING_BOX_Y_COORDINATE": 2,
        "BOUNDING_BOX_WIDTH": 3,
        "BOUNDING_BOX_HEIGHT": 4,
        "SEGMENTATION": None,
    }
    no_bbox_non_null_segmentation = {"SEGMENTATION": []}
    null_segmentation = {"SEGMENTATION": None}
    label_path = tmp_path / "TL1_느타리.zip"
    write_zip(
        label_path,
        {
            "생육/zero.json": full_label_json(
                image_filename="zero.jpg",
                temperature=0.0,
                humidity=None,
                annotations=[bbox, no_bbox_non_null_segmentation],
                extra_info={"FARM_ID": "farm-candidate"},
            ),
            "생육/ten.json": full_label_json(
                image_filename="ten.jpg",
                temperature=10.0,
                humidity=50.0,
                annotations=[],
            ),
            "병해/null.json": full_label_json(
                image_filename="null.jpg",
                temperature=None,
                humidity=0.0,
                annotations=[null_segmentation],
                normality=False,
                disease="테스트병",
            ),
            "병해/broken.json": b"{broken json",
        },
    )
    inspection = inspector.inspect_archive(archive_ref(label_path))
    result = inspector.scan_all_label_json(
        [inspection],
        checkpoint_dir=tmp_path / "artifacts" / "checkpoints",
        database_path=tmp_path / "artifacts" / "scan.sqlite3",
        checkpoint_interval=2,
        show_progress=False,
    )

    stats = result.stats
    assert result.complete is True
    assert stats.attempted_json == 4
    assert stats.processed_json == 3
    assert stats.failed_json == 1
    assert stats.numeric["temperature"].present == 2
    assert stats.numeric["temperature"].missing == 1
    assert stats.numeric["temperature"].minimum == 0.0
    assert stats.numeric["temperature"].mean == 5.0
    assert stats.numeric["temperature"].maximum == 10.0
    assert stats.numeric["humidity"].present == 2
    assert stats.numeric["humidity"].missing == 1
    assert stats.numeric["humidity"].minimum == 0.0
    assert stats.numeric["humidity"].mean == 25.0
    assert stats.bbox_document_counts == {"present": 1, "missing": 2}
    assert stats.annotation_count_distribution == {2: 1, 0: 1, 1: 1}
    assert stats.segmentation_document_counts == {
        "non_null": 1,
        "missing": 1,
        "null": 1,
    }
    assert stats.normality_counts["abnormal"] == 1
    assert stats.disease_counts["테스트병"] == 1
    assert (
        stats.candidate_fields["farm_or_facility_id"]["$.INFO.FARM_ID"]
        .examples
        == ["farm-candidate"]
    )

    leakage = inspector.analyze_data_leakage(result.database_path)
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    inspector.write_full_scan_reports(report_dir, result, leakage)
    expected_reports = {
        "full_label_distribution.csv",
        "full_field_completeness.csv",
        "full_environment_statistics.csv",
        "full_camera_date_distribution.csv",
        "data_leakage_analysis.md",
        "full_dataset_analysis.md",
    }
    assert {path.name for path in report_dir.iterdir()} == expected_reports
    combined = "\n".join(
        path.read_text(encoding="utf-8-sig")
        for path in report_dir.iterdir()
    )
    assert str(tmp_path) not in combined
    with (report_dir / "full_environment_statistics.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        environment_rows = {
            row["field_key"]: row for row in csv.DictReader(handle)
        }
    assert environment_rows["temperature"]["minimum"] == "0.0"
    assert environment_rows["temperature"]["mean"] == "5.0"
    assert environment_rows["temperature"]["missing_count"] == "1"
    assert (
        environment_rows["temperature"]["analysis_scope"]
        == "full_label_json"
    )
    assert environment_rows["temperature"]["image_bytes_read"] == "False"
    assert (
        environment_rows["temperature"]["full_zip_crc_checked"] == "False"
    )


def test_full_scan_checkpoint_resume_skips_already_processed_json(
    tmp_path: Path,
) -> None:
    label_path = tmp_path / "TL1_느타리.zip"
    write_zip(
        label_path,
        {
            "생육/one.json": full_label_json(image_filename="one.jpg"),
            "생육/two.json": full_label_json(image_filename="two.jpg"),
            "생육/three.json": full_label_json(image_filename="three.jpg"),
        },
    )
    inspection = inspector.inspect_archive(archive_ref(label_path))
    checkpoint_dir = tmp_path / "artifacts" / "checkpoints"
    database_path = tmp_path / "artifacts" / "scan.sqlite3"

    partial = inspector.scan_all_label_json(
        [inspection],
        checkpoint_dir=checkpoint_dir,
        database_path=database_path,
        checkpoint_interval=1,
        show_progress=False,
        stop_after=2,
    )
    assert partial.complete is False
    assert partial.stats.processed_json == 2
    checkpoint_path = checkpoint_dir / "full_json_scan_checkpoint.json"
    partial_payload = json.loads(checkpoint_path.read_text())
    assert partial_payload["status"] == "in_progress"
    assert partial_payload["checkpoint_version"] == 1
    assert partial_payload["settings"]["settings_version"] == 1
    assert (
        partial_payload["operational_settings"]["checkpoint_interval_json"]
        == 1
    )
    assert not list(checkpoint_dir.glob("*.tmp-*"))

    resumed = inspector.scan_all_label_json(
        [inspection],
        checkpoint_dir=checkpoint_dir,
        database_path=database_path,
        checkpoint_interval=1,
        show_progress=False,
    )
    assert resumed.complete is True
    assert resumed.resumed is True
    assert resumed.resume_count == 1
    assert resumed.stats.processed_json == 3
    assert resumed.stats.attempted_json == 3
    completed_payload = json.loads(checkpoint_path.read_text())
    assert completed_payload["status"] == "complete"
    assert len(completed_payload["completed_archives"]) == 1
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM label_records"
        ).fetchone()[0] == 3

    completed_rerun = inspector.scan_all_label_json(
        [inspection],
        checkpoint_dir=checkpoint_dir,
        database_path=database_path,
        checkpoint_interval=1,
        show_progress=False,
    )
    assert completed_rerun.stats.processed_json == 3
    assert completed_rerun.stats.attempted_json == 3


def test_train_validation_leakage_detection_uses_sqlite(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "TL1_느타리.zip"
    validation_path = tmp_path / "VL1_느타리.zip"
    write_zip(
        train_path,
        {
            "생육/a.json": full_label_json(
                image_filename="Exact.JPG", capture_time="12:00:00"
            ),
            "생육/b.json": full_label_json(
                image_filename="stem_only.jpg", capture_time="13:00:00"
            ),
        },
    )
    write_zip(
        validation_path,
        {
            "생육/c.json": full_label_json(
                image_filename="exact.jpg", capture_time="12:00:00"
            ),
            "생육/d.json": full_label_json(
                image_filename="stem_only.png", capture_time="14:00:00"
            ),
        },
    )
    train_ref = archive_ref(train_path)
    validation_ref = archive_ref(
        validation_path,
        split="validation",
        prefix="VL",
    )
    inspections = [
        inspector.inspect_archive(train_ref),
        inspector.inspect_archive(validation_ref),
    ]
    result = inspector.scan_all_label_json(
        inspections,
        checkpoint_dir=tmp_path / "checkpoints",
        database_path=tmp_path / "leakage.sqlite3",
        checkpoint_interval=1,
        show_progress=False,
    )
    leakage = inspector.analyze_data_leakage(result.database_path)
    assert leakage["image_filename"]["duplicate_key_count"] == 1
    assert leakage["image_stem"]["duplicate_key_count"] == 2
    assert leakage["capture_session"]["duplicate_key_count"] == 1
    assert leakage["camera_date"]["duplicate_key_count"] == 1
    assert leakage["camera_date"]["train_record_count"] == 2
    assert leakage["camera_date"]["validation_record_count"] == 2


def test_full_scan_never_reads_image_member_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    label_path = tmp_path / "TL1_느타리.zip"
    image_path = tmp_path / "TS1_느타리.zip"
    write_zip(
        label_path,
        {"생육/sample.json": full_label_json(image_filename="sample.jpg")},
    )
    write_zip(image_path, {"생육/sample.jpg": b"must-not-be-read"})
    label_inspection = inspector.inspect_archive(archive_ref(label_path))
    image_inspection = inspector.inspect_archive(
        archive_ref(image_path, kind="image", prefix="TS")
    )
    assert image_inspection.image_count == 1

    read_members: list[str] = []
    original_read = zipfile.ZipFile.read

    def guarded_read(
        archive: zipfile.ZipFile, name: object, *args: object, **kwargs: object
    ) -> bytes:
        member_name = name.filename if isinstance(name, zipfile.ZipInfo) else str(name)
        read_members.append(member_name)
        assert not member_name.lower().endswith((".jpg", ".jpeg", ".png"))
        return original_read(archive, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", guarded_read)
    result = inspector.scan_all_label_json(
        [label_inspection],
        checkpoint_dir=tmp_path / "checkpoints",
        database_path=tmp_path / "scan.sqlite3",
        checkpoint_interval=1,
        show_progress=False,
    )
    assert result.complete is True
    assert read_members == ["생육/sample.json"]


def test_scan_all_json_rejects_crc_or_image_options() -> None:
    with pytest.raises(SystemExit):
        inspector.parse_args(["--scan-all-json", "--check-zip-integrity"])
    with pytest.raises(SystemExit):
        inspector.parse_args(
            ["--scan-all-json", "--sample-image-pairs", "1"]
        )
    with pytest.raises(SystemExit):
        inspector.parse_args(
            [
                "--scan-all-json",
                "--sample-image-pairs",
                "1",
                "--extract-samples",
            ]
        )
