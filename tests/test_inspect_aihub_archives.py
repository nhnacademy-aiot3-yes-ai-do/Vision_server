from __future__ import annotations

import argparse
import importlib.util
import json
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
    directories = {
        "TRAIN_LABEL_ARCHIVES": tmp_path / "train_labels",
        "TRAIN_IMAGE_ARCHIVES": tmp_path / "train_images",
        "VAL_LABEL_ARCHIVES": tmp_path / "val_labels",
        "VAL_IMAGE_ARCHIVES": tmp_path / "val_images",
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
        output_dir=output_dir,
    )
    dataset_root = tmp_path / "dataset_root"
    dataset_root.mkdir()
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
