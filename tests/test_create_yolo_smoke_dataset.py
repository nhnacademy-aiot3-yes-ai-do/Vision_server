from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
import zipfile
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


def jpeg_bytes(size: tuple[int, int] = (100, 80)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (70, 90, 110)).save(buffer, format="JPEG")
    return buffer.getvalue()


def write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for member, content in members.items():
            archive.writestr(member, content)


def manifest_row(
    *,
    split: str = "train",
    species: str = "느타리",
    class_id: int = 0,
    task: str = "생육",
    archive_id: str = "TS1",
    member: str = "생육/a.jpg",
    boxes: list[dict[str, float]] | None = None,
) -> dict[str, str]:
    cleaned = boxes or [
        {"annotation_index": 0, "x": 10, "y": 10, "width": 20, "height": 30}
    ]
    return {
        "split": split,
        "species": species,
        "class_id": str(class_id),
        "task": task,
        "normality": "normal" if task == "생육" else "abnormal",
        "disease_type": "<missing>" if task == "생육" else "테스트병",
        "image_archive_id": archive_id,
        "image_member": member,
        "image_width": "100",
        "image_height": "80",
        "valid_bbox_count": str(len(cleaned)),
        "bbox_cleaned": json.dumps(cleaned, separators=(",", ":")),
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=sorted(smoke.REQUIRED_MANIFEST_COLUMNS)
        )
        writer.writeheader()
        writer.writerows(rows)


def archive_ref(
    path: Path,
    *,
    split: str,
    archive_id: str,
) -> object:
    return base.ArchiveRef(
        env_name=f"{split}_images",
        split=split,
        kind="image",
        prefix=archive_id[:2],
        index=int(archive_id[2:]),
        species="느타리",
        path=path,
    )


def test_bbox_to_yolo_coordinates_and_multiple_boxes() -> None:
    box = smoke.convert_bbox_to_yolo(
        {"x": 10, "y": 20, "width": 20, "height": 10},
        class_id=0,
        image_width=100,
        image_height=50,
    )
    assert box.x_center == pytest.approx(0.2)
    assert box.y_center == pytest.approx(0.5)
    assert box.width == pytest.approx(0.2)
    assert box.height == pytest.approx(0.2)
    row = manifest_row(
        boxes=[
            {"x": 0, "y": 0, "width": 10, "height": 10},
            {"x": 50, "y": 20, "width": 20, "height": 30},
        ]
    )
    boxes = smoke.yolo_boxes_for_row(row)
    assert len(boxes) == 2
    assert len([box.line() for box in boxes]) == 2


def test_class_id_and_normalized_coordinate_validation() -> None:
    with pytest.raises(ValueError, match="class_id"):
        smoke.convert_bbox_to_yolo(
            {"x": 0, "y": 0, "width": 10, "height": 10},
            class_id=5,
            image_width=100,
            image_height=80,
        )
    with pytest.raises(ValueError, match="범위"):
        smoke.convert_bbox_to_yolo(
            {"x": 95, "y": 0, "width": 10, "height": 10},
            class_id=0,
            image_width=100,
            image_height=80,
        )


def test_streaming_selection_is_deterministic_and_group_bounded(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, str]] = []
    for index in range(30):
        rows.append(
            manifest_row(
                member=f"생육/{index:03d}.jpg",
                task="생육",
            )
        )
        rows.append(
            manifest_row(
                member=f"병해/{index:03d}.jpg",
                task="병해",
            )
        )
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    first = smoke.select_manifest_rows(
        manifest,
        splits=("train", "validation"),
        samples_per_species_task=5,
        seed=20260726,
        max_total_images=200,
    )
    second = smoke.select_manifest_rows(
        manifest,
        splits=("train", "validation"),
        samples_per_species_task=5,
        seed=20260726,
        max_total_images=200,
    )
    assert [
        (row["image_archive_id"], row["image_member"]) for row in first
    ] == [
        (row["image_archive_id"], row["image_member"]) for row in second
    ]
    assert len(first) == 10
    assert {row["task"] for row in first} == {"생육", "병해"}


def test_maximum_image_limit_and_duplicate_prevention(tmp_path: Path) -> None:
    rows = [
        manifest_row(member=f"생육/{index:03d}.jpg")
        for index in range(20)
    ]
    rows.append(manifest_row(task="병해", member="생육/000.jpg"))
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    selected = smoke.select_manifest_rows(
        manifest,
        splits=("train",),
        samples_per_species_task=20,
        seed=7,
        max_total_images=7,
    )
    keys = [smoke.candidate_unique_key(row) for row in selected]
    assert len(selected) == 7
    assert len(keys) == len(set(keys))
    with pytest.raises(ValueError, match="200"):
        smoke.select_manifest_rows(
            manifest,
            splits=("train",),
            samples_per_species_task=20,
            seed=7,
            max_total_images=201,
        )


def test_test_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="Test"):
        smoke.normalize_requested_splits(("train", "test"))


def test_only_selected_zip_members_are_read_and_sources_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_zip = tmp_path / "TS1_느타리.zip"
    val_zip = tmp_path / "VS1_느타리.zip"
    write_zip(
        train_zip,
        {
            "생육/selected.jpg": jpeg_bytes(),
            "생육/reassigned_to_validation.jpg": jpeg_bytes(),
            "생육/not_selected.jpg": jpeg_bytes(),
        },
    )
    write_zip(
        val_zip,
        {
            "병해/selected.jpg": jpeg_bytes(),
            "병해/not_selected.jpg": jpeg_bytes(),
        },
    )
    refs = {
        "TS1": archive_ref(train_zip, split="train", archive_id="TS1"),
        "VS1": archive_ref(val_zip, split="validation", archive_id="VS1"),
    }
    selected = [
        manifest_row(
            split="train",
            archive_id="TS1",
            member="생육/selected.jpg",
        ),
        manifest_row(
            split="validation",
            archive_id="TS1",
            member="생육/reassigned_to_validation.jpg",
            task="병해",
        ),
    ]
    before = audit.source_zip_snapshot((train_zip, val_zip))
    reads: list[tuple[str, str]] = []
    original_read = zipfile.ZipFile.read

    def tracked_read(
        self: zipfile.ZipFile,
        name: object,
        *args: object,
        **kwargs: object,
    ) -> bytes:
        member = name.filename if isinstance(name, zipfile.ZipInfo) else str(name)
        reads.append((Path(self.filename).name, member))
        return original_read(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", tracked_read)
    output = tmp_path / "artifacts" / "yolo_smoke"
    result = smoke.create_smoke_dataset(
        selected,
        refs,
        output,
        seed=20260726,
        samples_per_species_task=1,
        max_total_images=2,
        overwrite=False,
    )
    assert set(reads) == {
        (train_zip.name, "생육/selected.jpg"),
        (train_zip.name, "생육/reassigned_to_validation.jpg"),
    }
    assert audit.source_zip_snapshot((train_zip, val_zip)) == before
    assert result["train_images"] == 1
    assert result["validation_images"] == 1
    assert result["bbox_count"] == 2
    assert result["source_zip_modified"] is False


def test_output_structure_and_absolute_path_non_exposure(
    tmp_path: Path,
) -> None:
    train_zip = tmp_path / "TS1_느타리.zip"
    write_zip(train_zip, {"생육/a.jpg": jpeg_bytes()})
    selected = [manifest_row(member="생육/a.jpg")]
    output = tmp_path / "artifacts" / "yolo_smoke"
    smoke.create_smoke_dataset(
        selected,
        {"TS1": archive_ref(train_zip, split="train", archive_id="TS1")},
        output,
        seed=20260726,
        samples_per_species_task=1,
        max_total_images=1,
        overwrite=False,
    )
    assert (output / "data.yaml").read_text(encoding="utf-8") == (
        "train: images/train\n"
        "val: images/val\n"
        "nc: 5\n"
        "names:\n"
        "  0: 느타리\n"
        "  1: 양송이\n"
        "  2: 큰느타리\n"
        "  3: 팽이\n"
        "  4: 표고\n"
    )
    assert len(list((output / "images" / "train").iterdir())) == 1
    assert len(list((output / "labels" / "train").glob("*.txt"))) == 1
    assert len(list((output / "overlays" / "train").glob("*.jpg"))) == 1
    for path in output.rglob("*"):
        if path.is_file() and path.suffix in {".txt", ".csv", ".yaml", ".md"}:
            text = path.read_text(encoding="utf-8-sig")
            assert str(tmp_path) not in text
            assert "/mnt/d" not in text
            assert "/home/kim75" not in text
