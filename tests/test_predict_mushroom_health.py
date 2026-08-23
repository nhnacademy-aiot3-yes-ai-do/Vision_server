# 실제 모델 없이 핵심 추론 알고리즘의 외부 동작을 검증한다.
# 품종별 grouping과 union crop, 15% padding, confidence gate,
# 건강 상태 매핑, 입력·모델 불변성과 경로 비노출을 확인한다.
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 같은 이름으로 이미 로드된 모듈은 재사용하고, 아니면 지정 경로의 스크립트를 직접 import하는 helper이다.
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


# 실제 추론 스크립트를 패키지 설치 여부와 무관하게 테스트 대상 모듈로 불러온다.
prediction = load_module(
    "predict_mushroom_health",
    PROJECT_ROOT / "scripts" / "predict_mushroom_health.py",
)


# 테스트 전후 모델 파일 내용이 바뀌지 않았는지 비교할 SHA-256을 계산하는 helper이다.
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 준비된 탐지 결과를 반환하고 입력 이미지 크기와 호출 횟수를 기록하는 detector 대역이다.
class FakeDetector:
    model_name = "fake-yolo11n-species"

    def __init__(self, detections: list[object]) -> None:
        self.detections = detections
        self.calls: list[tuple[int, int]] = []

    def predict(self, image: Image.Image) -> list[object]:
        self.calls.append(image.size)
        return list(self.detections)


# 품종 그룹별 준비된 확률을 순서대로 반환하고 crop 크기를 기록하는 classifier 대역이다.
class FakeClassifier:
    model_name = "fake-yolo11n-health"

    def __init__(self, probabilities: list[tuple[float, float]]) -> None:
        self.probabilities = list(probabilities)
        self.calls: list[tuple[int, int]] = []

    def predict(self, crop: Image.Image) -> tuple[float, float]:
        self.calls.append(crop.size)
        if not self.probabilities:
            raise AssertionError("unexpected classifier call")
        return self.probabilities.pop(0)


# 테스트가 읽기 쉽게 품종·bbox·신뢰도를 지정한 Detection 객체를 만드는 helper이다.
def detection(
    *,
    class_id: int,
    species: str,
    bbox: tuple[int, int, int, int],
    confidence: float,
) -> object:
    return prediction.Detection(
        class_id=class_id,
        species=species,
        bbox=bbox,
        confidence=confidence,
    )


# 탐지 결과가 품종별로 묶이고 같은 품종의 여러 bbox가 하나의 union bbox가 되는지 검증한다.
def test_species_grouping_and_union_bbox() -> None:
    detections = [
        detection(
            class_id=0,
            species="느타리",
            bbox=(10, 20, 30, 40),
            confidence=0.91,
        ),
        detection(
            class_id=4,
            species="표고",
            bbox=(50, 5, 90, 30),
            confidence=0.88,
        ),
        detection(
            class_id=0,
            species="느타리",
            bbox=(25, 10, 70, 65),
            confidence=0.82,
        ),
    ]

    grouped = prediction.group_detections_by_species(detections)

    assert set(grouped) == {"느타리", "표고"}
    assert len(grouped["느타리"]) == 2
    assert len(grouped["표고"]) == 1
    assert prediction.union_bbox(
        [item.bbox for item in grouped["느타리"]]
    ) == (10, 10, 70, 65)
    assert prediction.union_bbox(
        [item.bbox for item in grouped["표고"]]
    ) == (50, 5, 90, 30)


# padding이 union 크기가 아닌 원본 이미지 크기를 기준으로 계산되고 이미지 경계에서 잘리는지 검증한다.
def test_union_crop_uses_image_relative_15_percent_padding_and_clips() -> None:
    # Padding is based on the source width/height: 15 px horizontally and
    # 12 px vertically, not 15% of the union-box size.
    assert prediction.padded_union_crop_box(
        (100, 80),
        [(10, 20, 30, 40)],
        padding_ratio=0.15,
    ) == (0, 8, 45, 52)

    # A union close to all four edges must be clipped to image bounds.
    assert prediction.padded_union_crop_box(
        (100, 80),
        [(10, 10, 80, 70)],
        padding_ratio=0.15,
    ) == (0, 0, 95, 80)


# 탐지 결과가 없으면 성공적인 미탐지 응답을 만들고 건강 분류기를 호출하지 않는지 검증한다.
def test_no_detection_does_not_call_health_classifier() -> None:
    image = Image.new("RGB", (100, 80), (20, 30, 40))
    detector = FakeDetector([])
    classifier = FakeClassifier([])

    result = prediction.predict_health(
        image,
        detector=detector,
        classifier=classifier,
        health_threshold=0.70,
    )

    assert result["analysis_type"] == "MUSHROOM_HEALTH_CHECK_V1"
    assert result["status"] == "NO_MUSHROOM_DETECTED"
    assert result["results"] == []
    assert result["detector_model"] == detector.model_name
    assert result["health_model"] == classifier.model_name
    assert result["thresholds"]["min_detection_confidence"] == pytest.approx(
        0.50
    )
    assert result["thresholds"]["health_confidence"] == pytest.approx(0.70)
    assert len(detector.calls) == 1
    assert classifier.calls == []
    json.dumps(result, ensure_ascii=False, allow_nan=False)


# 한 품종 그룹의 최소 탐지 신뢰도가 기준 미만이면 분류를 건너뛰고 nullable 결과를 내는지 검증한다.
def test_detection_confidence_below_minimum_skips_classifier() -> None:
    image = Image.new("RGB", (100, 80), "white")
    detector = FakeDetector(
        [
            detection(
                class_id=0,
                species="느타리",
                bbox=(10, 10, 40, 50),
                confidence=0.90,
            ),
            detection(
                class_id=0,
                species="느타리",
                bbox=(45, 15, 80, 60),
                confidence=0.49,
            ),
        ]
    )
    classifier = FakeClassifier([])

    response = prediction.predict_health(
        image,
        detector=detector,
        classifier=classifier,
        min_detection_confidence=0.50,
    )

    result = response["results"][0]
    assert classifier.calls == []
    assert result["detection_confidence"] == pytest.approx(0.90)
    assert result["detection_confidence_min"] == pytest.approx(0.49)
    assert result["health_status"] == "UNCERTAIN"
    assert result["health_confidence"] is None
    assert result["healthy_probability"] is None
    assert result["disease_suspected_probability"] is None
    assert prediction.LOW_DETECTION_CONFIDENCE_WARNING in response["warnings"]
    assert not any(
        warning.startswith("UNCERTAIN: 느타리 건강 confidence")
        for warning in response["warnings"]
    )
    assert prediction.draw_annotated(image, response).size == image.size


# 최소 탐지 신뢰도와 정확히 같은 경계값은 건강 분류 대상으로 포함되는지 검증한다.
def test_detection_confidence_equal_to_minimum_runs_classifier() -> None:
    image = Image.new("RGB", (100, 80), "white")
    detector = FakeDetector(
        [
            detection(
                class_id=0,
                species="느타리",
                bbox=(10, 10, 80, 70),
                confidence=0.50,
            )
        ]
    )
    classifier = FakeClassifier([(0.95, 0.05)])

    response = prediction.predict_health(
        image,
        detector=detector,
        classifier=classifier,
        min_detection_confidence=0.50,
    )

    result = response["results"][0]
    assert len(classifier.calls) == 1
    assert result["health_status"] == "HEALTHY"
    assert result["health_confidence"] == pytest.approx(0.95)
    assert result["healthy_probability"] == pytest.approx(0.95)
    assert result["disease_suspected_probability"] == pytest.approx(0.05)
    assert prediction.LOW_DETECTION_CONFIDENCE_WARNING not in response[
        "warnings"
    ]


# 신뢰도가 낮은 품종만 UNCERTAIN 처리하고 다른 품종 그룹의 분류는 계속하는지 검증한다.
def test_low_confidence_only_skips_its_species_group() -> None:
    image = Image.new("RGB", (100, 80), "white")
    detector = FakeDetector(
        [
            detection(
                class_id=0,
                species="느타리",
                bbox=(5, 5, 40, 50),
                confidence=0.49,
            ),
            detection(
                class_id=4,
                species="표고",
                bbox=(55, 10, 95, 70),
                confidence=0.90,
            ),
        ]
    )
    classifier = FakeClassifier([(0.08, 0.92)])

    response = prediction.predict_health(
        image,
        detector=detector,
        classifier=classifier,
        min_detection_confidence=0.50,
    )

    by_species = {
        item["species"]: item for item in response["results"]
    }
    assert len(classifier.calls) == 1
    assert by_species["느타리"]["health_status"] == "UNCERTAIN"
    assert by_species["느타리"]["health_confidence"] is None
    assert by_species["느타리"]["healthy_probability"] is None
    assert by_species["느타리"]["disease_suspected_probability"] is None
    assert by_species["표고"]["health_status"] == "DISEASE_SUSPECTED"
    assert by_species["표고"]["health_confidence"] == pytest.approx(0.92)


# 같은 품종의 여러 탐지는 한 번만 분류하고 서로 다른 품종은 각각 한 번씩 분류하는지 검증한다.
def test_multiple_species_are_classified_once_per_species_group() -> None:
    image = Image.new("RGB", (100, 80), (50, 60, 70))
    detector = FakeDetector(
        [
            detection(
                class_id=0,
                species="느타리",
                bbox=(10, 10, 25, 30),
                confidence=0.90,
            ),
            detection(
                class_id=0,
                species="느타리",
                bbox=(30, 20, 55, 45),
                confidence=0.80,
            ),
            detection(
                class_id=4,
                species="표고",
                bbox=(60, 25, 90, 70),
                confidence=0.95,
            ),
        ]
    )
    classifier = FakeClassifier([(0.96, 0.04), (0.08, 0.92)])

    response = prediction.predict_health(
        image,
        detector=detector,
        classifier=classifier,
        health_threshold=0.70,
        padding_ratio=0.15,
    )

    assert response["status"] == "SUCCESS"
    assert len(response["results"]) == 2
    assert len(classifier.calls) == 2
    by_species = {
        item["species"]: item for item in response["results"]
    }
    oyster = by_species["느타리"]
    shiitake = by_species["표고"]
    assert oyster["class_id"] == 0
    assert oyster["detected_count"] == 2
    assert oyster["bbox"] == [10, 10, 55, 45]
    assert oyster["crop_bbox"] == [0, 0, 70, 57]
    assert oyster["health_status"] == "HEALTHY"
    assert oyster["healthy_probability"] == pytest.approx(0.96)
    assert oyster["disease_suspected_probability"] == pytest.approx(0.04)
    assert oyster["health_confidence"] == pytest.approx(0.96)

    assert shiitake["class_id"] == 4
    assert shiitake["detected_count"] == 1
    assert shiitake["bbox"] == [60, 25, 90, 70]
    assert shiitake["crop_bbox"] == [45, 13, 100, 80]
    assert shiitake["health_status"] == "DISEASE_SUSPECTED"
    assert shiitake["healthy_probability"] == pytest.approx(0.08)
    assert shiitake["disease_suspected_probability"] == pytest.approx(0.92)
    json.dumps(response, ensure_ascii=False, allow_nan=False)


# 최대 건강 확률과 불확실 임계값 조합이 세 공개 건강 상태로 정확히 매핑되는지 검증한다.
@pytest.mark.parametrize(
    ("probabilities", "expected_status"),
    [
        ((0.60, 0.40), "UNCERTAIN"),
        ((0.31, 0.69), "UNCERTAIN"),
        ((0.70, 0.30), "HEALTHY"),
        ((0.20, 0.80), "DISEASE_SUSPECTED"),
    ],
)
def test_health_probability_mapping_and_uncertain_threshold(
    probabilities: tuple[float, float],
    expected_status: str,
) -> None:
    image = Image.new("RGB", (100, 80), "white")
    detector = FakeDetector(
        [
            detection(
                class_id=1,
                species="양송이",
                bbox=(20, 10, 80, 70),
                confidence=0.90,
            )
        ]
    )
    classifier = FakeClassifier([probabilities])

    response = prediction.predict_health(
        image,
        detector=detector,
        classifier=classifier,
        health_threshold=0.70,
    )
    result = response["results"][0]

    assert result["healthy_probability"] == pytest.approx(probabilities[0])
    assert result["disease_suspected_probability"] == pytest.approx(
        probabilities[1]
    )
    assert result["health_confidence"] == pytest.approx(max(probabilities))
    assert result["health_status"] == expected_status


# 추론이 호출자의 이미지 바이트와 모델 파일의 내용·크기·수정 시각을 변경하지 않는지 검증한다.
def test_prediction_does_not_modify_input_or_fake_model_files(
    tmp_path: Path,
) -> None:
    detector_weight = tmp_path / "detector.pt"
    health_weight = tmp_path / "health.pt"
    detector_weight.write_bytes(b"immutable fake detector")
    health_weight.write_bytes(b"immutable fake classifier")
    before_hashes = (sha256(detector_weight), sha256(health_weight))
    before_stats = (
        (detector_weight.stat().st_size, detector_weight.stat().st_mtime_ns),
        (health_weight.stat().st_size, health_weight.stat().st_mtime_ns),
    )
    image = Image.new("RGB", (100, 80), (11, 22, 33))
    input_before = image.tobytes()
    detector = FakeDetector(
        [
            detection(
                class_id=2,
                species="큰느타리",
                bbox=(10, 10, 90, 70),
                confidence=0.9,
            )
        ]
    )
    classifier = FakeClassifier([(0.8, 0.2)])

    prediction.predict_health(
        image,
        detector=detector,
        classifier=classifier,
        health_threshold=0.70,
    )

    assert image.tobytes() == input_before
    assert (sha256(detector_weight), sha256(health_weight)) == before_hashes
    assert (
        (detector_weight.stat().st_size, detector_weight.stat().st_mtime_ns),
        (health_weight.stat().st_size, health_weight.stat().st_mtime_ns),
    ) == before_stats


# 직렬화 응답에 Path 객체나 Linux·Windows 로컬 절대 경로가 포함되지 않는지 검증한다.
def test_response_contains_no_path_objects_or_local_absolute_paths() -> None:
    image = Image.new("RGB", (100, 80), "white")
    response: dict[str, Any] = prediction.predict_health(
        image,
        detector=FakeDetector(
            [
                detection(
                    class_id=3,
                    species="팽이",
                    bbox=(5, 5, 50, 60),
                    confidence=0.85,
                )
            ]
        ),
        classifier=FakeClassifier([(0.9, 0.1)]),
        health_threshold=0.70,
    )

    serialized = json.dumps(response, ensure_ascii=False, allow_nan=False)
    assert "/mnt/" not in serialized
    assert "/home/" not in serialized
    assert "C:\\\\" not in serialized
    assert not any(
        isinstance(value, Path)
        for result in response["results"]
        for value in result.values()
    )
