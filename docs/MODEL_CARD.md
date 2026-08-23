# Mushroom Vision 모델 카드

문서 기준일: 2026-07-30

## 1. 목적과 사용 범위

이 서비스는 한 장의 버섯 이미지에서 다음 두 단계를 수행합니다.

1. 품종 탐지 모델이 지원 품종의 bbox, 개수와 탐지 confidence를 계산합니다.
2. 같은 품종의 bbox를 하나의 영역으로 합친 union crop을 건강 분류 모델에
   입력해 `HEALTHY`, `DISEASE_SUSPECTED` 또는 안전 규칙에 따른
   `UNCERTAIN`을 반환합니다.

결과는 AI-Server가 요약·가공하기 위한 참고 정보입니다. 의료·농업 전문가의
확정 진단이나 방제 처방을 대신하지 않습니다.

## 2. 배포 모델과 무결성

현재 배포 모델의 단일 기준은
[`models/model-manifest.json`](../models/model-manifest.json)입니다. 모델을
교체할 때는 파일만 덮어쓰지 않고 manifest의 모델명, 크기, SHA-256, 입력
크기와 클래스 순서를 함께 갱신해야 합니다. 서비스 시작 시 이 정보와 실제
모델이 일치하지 않으면 준비 완료 상태로 전환하지 않는 것이 운영 원칙입니다.

| 역할 | 모델명 | 입력 크기 | 파일 크기 | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 품종 탐지 | `mushroom-yolo11n-camera-holdout-v1` | 640 | 5,447,706 bytes | `8d17eb493f2eeccccd832c56da2f346dfc730e5605c460016386c6bd0be10d32` |
| 건강 분류 | `mushroom-health-yolo11n-date-camera-holdout-v1` | 320 | 3,186,882 bytes | `720efb30093c2fbaf8866243a60870c7816f3e141c81e0fac015a358edce1f92` |

파일 크기와 SHA-256은 성능 지표가 아니라, 검증한 모델과 배포 모델이 같은
파일인지 확인하기 위한 식별값입니다.

## 3. 품종 탐지 모델

품종 탐지 모델은 Ultralytics YOLO11n 기반 객체 탐지 모델입니다.

| class id | 지원 품종 |
| ---: | --- |
| 0 | 느타리 |
| 1 | 양송이 |
| 2 | 큰느타리 |
| 3 | 팽이 |
| 4 | 표고 |

release manifest에 기록된 내부 `species + camera_id` holdout 검증의
**mAP50-95는 0.899**입니다. 이 수치는 AIHub 고정 카메라 데이터 범위의
내부 평가 결과이며, 외부 스마트폰 이미지 성능을 의미하지 않습니다. 현재
정리 대상 보고서에는 이 수치를 다시 계산할 수 있는 detector 상세 평가
표가 없으므로, 이 모델 카드에서는 release manifest의 검증 기록만
인용합니다.

## 4. 건강 분류 모델

건강 분류 모델은 Ultralytics YOLO11n-cls 기반 이진 분류 모델입니다. 모델
내부 클래스 `0_healthy`, `1_disease_suspected`를 API에서 각각
`HEALTHY`, `DISEASE_SUSPECTED`로 변환합니다. `UNCERTAIN`은 별도의 학습
클래스가 아니라 confidence가 기준보다 낮을 때 서비스가 적용하는 안전
상태입니다.

### 입력과 분할

- 입력은 품종별 bbox union에 context padding을 적용한 320 크기
  이미지입니다.
- 분할 계획은 Train 5,000장, Validation 1,000장입니다.
- 5품종 × 2상태 조합마다 Train 500장, Validation 100장을 선택해 클래스와
  품종 수를 맞췄습니다.
- 같은 촬영 날짜와 global camera ID가 Train과 Validation 양쪽에 들어가지
  않도록 구성했습니다.
- Validation 날짜는 `2021-11-04`, `2021-11-26`, `2021-12-04`입니다.
- 모델 선택과 아래 평가는 Validation만 사용했으며, 고정 Test 이미지는
  사용하지 않았습니다.

### Validation 1,000장 지표

| 지표 | 값 |
| --- | ---: |
| Accuracy | 0.999000 |
| Balanced accuracy | 0.999000 |
| Macro precision | 0.999002 |
| Macro recall | 0.999000 |
| Macro F1 | 0.999000 |

혼동행렬은 실제 클래스가 행, 예측 클래스가 열입니다.

| 실제 \ 예측 | HEALTHY | DISEASE_SUSPECTED |
| --- | ---: | ---: |
| HEALTHY | 499 | 1 |
| DISEASE_SUSPECTED | 0 | 500 |

Validation에서 확인된 오분류는 1,000장 중 1장이었습니다. 이 높은 내부
지표는 아래의 날짜·촬영 환경 편향 때문에 외부 성능으로 확대 해석하면 안
됩니다.

## 5. 15% union crop 정책

건강 라벨은 이미지 단위이지만 원본 bbox는 병반 위치가 아니라 버섯
개체·군집 영역입니다. 따라서 서비스는 같은 품종의 모든 유효 bbox를
합쳐 버섯 전체를 포함하고, 이미지 폭과 높이의 **15%를 각 방향의 context
padding**으로 추가한 뒤 이미지 경계에서 잘라냅니다.

이 정책을 선택한 근거는 다음과 같습니다.

- 가장 큰 bbox 하나만 사용했을 때 생길 수 있는 다른 개체·군집 누락을
  줄입니다.
- padding 없이 union만 사용할 때 빠질 수 있는 배지와 주변 군집 문맥을
  보존합니다.
- 전체 이미지를 입력할 때보다 재배실·카메라·작업 배경이 다시 들어오는
  범위를 줄입니다.
- 15% 정책을 사용한 내부 end-to-end 100장 평가에서 detector crop과
  ground-truth crop의 건강 상태가 100장 모두 일치했고, 두 crop의 평균
  IoU는 0.982953이었습니다.

15%가 10% 또는 20%보다 우수하다고 확정한 비교 실험 결과는 현재 보존할
보고서에 없습니다. 따라서 15%는 검증된 운영 정책이지만 전역 최적값으로
주장하지 않습니다.

## 6. 내부 end-to-end 평가

두 배포 모델을 실제 순서대로 연결한 평가는 health date+camera holdout
Validation에서 100장을 선택해 수행했습니다. 5품종의
`HEALTHY`/`DISEASE_SUSPECTED` 조합별로 10장씩이며, Test 이미지는
사용하지 않았습니다. 평가 threshold는 탐지 0.25, 건강 `UNCERTAIN` 0.70,
padding 0.15였습니다.

| 지표 | 결과 |
| --- | ---: |
| 정답 품종 탐지 성공 | 100/100 |
| 예측 품종 집합 완전 일치 | 100/100 |
| 예상 밖 추가 품종 탐지 이미지 | 0/100 |
| 건강 분석 가능 | 100/100 |
| 건강 Accuracy | 1.000000 |
| 건강 Macro F1 | 1.000000 |
| HEALTHY recall | 1.000000 |
| DISEASE_SUSPECTED recall | 1.000000 |
| 정확한 품종 집합 + 건강 상태 | 100/100 |
| `UNCERTAIN` | 0/100 |
| 병해를 `HEALTHY`로 반환 | 0/100 |

이 결과는 제한된 내부 표본에서 파이프라인 연결과 crop 호환성을 확인한
결과입니다. 실사용 환경 정확도 100%를 보장하는 결과가 아닙니다.

## 7. 알려진 편향과 한계

### 날짜와 카메라 편향

- 건강 Validation에서 정상 이미지는 `2021-11-04`, 병해 이미지는
  `2021-11-26`과 `2021-12-04`에만 존재해 날짜와 상태가 결합되어
  있습니다. 모델이 병해 특징뿐 아니라 날짜별 촬영 환경을 지름길로
  사용했을 가능성을 배제할 수 없습니다.
- end-to-end 100장은 detector Train/Validation과 동일한 원본 이미지를
  제외했지만, 100장 모두 detector Train과 촬영 날짜 및 camera ID가
  겹쳤고 88장은 `species + camera` 조합도 겹쳤습니다.
- 원본 데이터는 고정 카메라와 재배 환경 중심이므로 조명, 배경, 거리와
  구도가 다른 입력에서 성능이 달라질 수 있습니다.

### 외부 일반화

외부 스마트폰 이미지로 구성한 독립 평가가 완료되지 않았습니다. 따라서
내부 mAP, Validation 0.999 Accuracy, end-to-end 1.000 Accuracy를 실제
사용자 촬영 환경의 성능으로 해석하지 않습니다. 배포 후에는 새 농가,
새 카메라, 다양한 스마트폰·조명·거리의 별도 Test 세트가 필요합니다.

### 결과의 의미

`DISEASE_SUSPECTED`는 병해 가능성을 알리는 후보 상태이며 특정 병명을
확정하지 않습니다. `HEALTHY`도 모든 병해가 없음을 보증하지 않습니다.
confidence가 낮거나 품종 탐지가 불충분한 결과는 자동 진단으로 강제하지
말고 `UNCERTAIN` 또는 건강 분석 불가로 처리해야 합니다.

## 8. 지원하지 않는 기능

현재 두 모델과 API는 다음 기능을 지원하지 않습니다.

- 느타리, 양송이, 큰느타리, 팽이, 표고 이외 품종의 신뢰할 수 있는 식별
- 병해 종류의 확정 진단
- 병반 위치 bbox, mask 또는 segmentation
- 해충 탐지
- 수확 적기, 생육 단계, 예상 수확일 또는 남은 일수 계산
- 갓·대의 실제 길이, 두께, 무게 등 물리량 측정
- 방제·투약 처방

## 9. 보존한 수치의 출처

이 문서는 다음 자료에서 서비스 운영에 필요한 수치와 판단만 통합했습니다.
아래 원본 보고서는 현재 작업 트리에서는 제거했으며, 보존용 Git 브랜치
`archive/research-reports-2026-07-30`에서 확인할 수 있습니다.

- `models/model-manifest.json`
- `reports/health_classifier_evaluation.md`
- `reports/health_classifier_overall_metrics.json`
- `reports/health_date_holdout_split_plan.md`
- `reports/health_input_strategy.md`
- `reports/health_end_to_end_evaluation.md`
- `reports/health_end_to_end_metrics.json`
- `reports/model_scope_decision.md`
- `reports/health_model_scope_decision.md`

향후 모델 파일이나 데이터 분할을 변경하면 기존 지표를 그대로 재사용하지
말고 새 모델 버전으로 평가한 뒤 이 문서와 manifest를 함께 갱신해야 합니다.
