# Mushroom Vision

AIHub 버섯 데이터로 개발한 두 고정 모델을 결합해 버섯 품종과 건강 상태를
확인하는 FastAPI v1입니다.

- 품종 탐지: YOLO11n, 5품종
- 건강 분류: YOLO11n-cls, `HEALTHY` / `DISEASE_SUSPECTED`
- 낮은 건강 confidence: `UNCERTAIN`
- 결과는 AI 분석 참고 정보이며 확정 진단이 아닙니다.

수확 적기, 생육 단계, 예상 수확일, 실제 갓·대 크기, 해충 및 병반 위치는
현재 지원하지 않습니다.

## 고정 모델

- `artifacts/models/yolo11n_camera_holdout_v1/best.pt`
- `artifacts/models/yolo11n_health_date_camera_holdout_v1/best.pt`

애플리케이션 lifespan 시작 시 프로세스당 한 번만 모델을 로드하며, class
mapping을 검증합니다. 기본 설정에서는 승인된 SHA-256도 검증합니다. 모델
로딩에 실패하면 서버 시작이 실패합니다.

## 실행

프로젝트 가상환경에는 Python 3.12, FastAPI, Uvicorn, python-multipart,
Pillow, PyTorch 및 Ultralytics가 준비되어 있어야 합니다.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

GPU 모델 객체는 process-local singleton입니다. GPU 메모리 복제와 동시
추론 충돌을 피하기 위해 worker 1개로 실행하고 운영 중 `--reload`는 사용하지
않습니다.

OpenAPI 문서는 서버 실행 중 `/docs`, schema는 `/openapi.json`에서 확인할 수
있습니다.

## 환경변수

| 환경변수 | 기본값 | 허용 범위 |
| --- | ---: | --- |
| `HEALTH_DETECTION_CONFIDENCE` | `0.25` | 0~1 |
| `HEALTH_UNCERTAIN_THRESHOLD` | `0.70` | 0~1 |
| `HEALTH_PADDING_RATIO` | `0.15` | 0~0.5 |
| `HEALTH_MAX_UPLOAD_BYTES` | `10485760` | 1 byte~100 MiB |
| `HEALTH_VERIFY_MODEL_SHA256` | `true` | boolean |
| `HEALTH_DEVICE` | `auto` | Ultralytics device 값 |

잘못된 설정은 모델 로드 전에 검증되며 애플리케이션 시작을 중단합니다.

## API

`POST /api/v1/mushroom/health-check`

- 요청: `multipart/form-data`
- 파일 필드: `image`
- 형식: JPG, JPEG, PNG, WEBP
- 최대 업로드: 기본 10 MiB
- 최대 픽셀: `scripts/predict_mushroom_health.py`의
  `MAX_IMAGE_PIXELS`
- 업로드 원본은 애플리케이션 파일로 저장하지 않습니다.
- EXIF 방향을 반영하고, 손상 이미지·압축폭탄·애니메이션 이미지를 거부합니다.

```bash
curl -X POST \
  http://localhost:8000/api/v1/mushroom/health-check \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@sample.jpg"
```

성공 예시:

```json
{
  "analysisType": "MUSHROOM_HEALTH_CHECK_V1",
  "status": "SUCCESS",
  "detectorModel": "mushroom-yolo11n-camera-holdout-v1",
  "healthModel": "mushroom-health-yolo11n-date-camera-holdout-v1",
  "thresholds": {
    "detection": 0.25,
    "healthUncertain": 0.7
  },
  "results": [
    {
      "species": "느타리",
      "speciesClassId": 0,
      "detectedCount": 1,
      "detectionConfidence": 0.97,
      "detectionConfidenceMin": 0.97,
      "healthStatus": "HEALTHY",
      "healthConfidence": 0.99,
      "healthyProbability": 0.99,
      "diseaseSuspectedProbability": 0.01,
      "bbox": [187, 769, 902, 1372],
      "cropBbox": [25, 481, 1064, 1660]
    }
  ],
  "warnings": [
    "AI 분석 참고 결과이며 확정 진단이 아닙니다."
  ]
}
```

### HTTP 상태

| HTTP | 응답 상태 | 의미 |
| ---: | --- | --- |
| 200 | `SUCCESS` | 품종별 건강 결과 반환 |
| 200 | `NO_MUSHROOM_DETECTED` | 버섯 탐지 없음, `results=[]` |
| 400 | `INVALID_IMAGE` | 빈 파일 또는 손상 이미지 |
| 413 | `INVALID_IMAGE` | 업로드 크기 제한 초과 |
| 415 | `INVALID_IMAGE` | MIME, 확장자 또는 실제 형식 미지원 |
| 422 | FastAPI validation error | `image` 필드 누락 |
| 500 | `INFERENCE_FAILED` | 상세 예외를 숨긴 모델 추론 실패 |

`LOW_DETECTION_CONFIDENCE`, `MULTIPLE_SPECIES_DETECTED` 및 `UNCERTAIN`은
필요할 때 응답 `warnings` 또는 품종별 `healthStatus`로 제공됩니다. 로컬
경로, 모델 파일 경로와 stack trace는 외부 JSON에 포함하지 않습니다.

## 동시성과 안전

이미지 decode와 동기 모델 추론은 threadpool에서 실행하므로 event loop를
직접 차단하지 않습니다. detector→union crop→classifier 전체 구간은 하나의
비동기 inference lock으로 직렬화하여 동일 GPU 모델 객체에 동시 접근하지
않습니다. 파일명은 저장 경로에 사용하지 않으며 ZIP, SVG 및 기타 형식은
지원하지 않습니다.

## 검증 범위

내부 date+camera holdout 원본 100장 end-to-end 평가에서 품종 탐지 성공률,
건강 분석 가능률, Accuracy, Macro F1과 두 클래스 recall이 모두 1.000이었습니다.
detector 학습 이미지와 동일 원본은 0장이었습니다.

이 결과는 AIHub 내부 촬영 환경의 소규모 검증이며 외부 스마트폰 사진에 대한
일반화 성능으로 인정하지 않습니다. 실제 서비스 적용 전 별도의 외부 데이터
검증과 전문가 검수가 필요합니다.

## 테스트

기본 테스트는 fake detector/classifier를 사용하며 실제 모델을 로드하지
않습니다.

```bash
pytest -q
```

실제 모델 registry 통합 확인은 명시적으로 허용할 때만 실행합니다.

```bash
RUN_MODEL_INTEGRATION_TESTS=true \
pytest -q tests/test_model_registry.py -k integration
```
