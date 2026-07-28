# 건강 체크 분류기 Validation 상세 평가

## 범위

- 모델: `artifacts/health_pilot_date_holdout/training_runs/yolo11n_health_binary_date_camera_holdout_30ep/weights/best.pt` (`best.pt`만 사용)
- manifest: `artifacts/health_pilot_date_holdout/health_date_holdout_manifest.csv`
- Validation: **1,000장**
- Train 평가: 0장
- Test 접근·평가: 0장
- 재학습: 수행하지 않음

## 전체 지표

- Accuracy: **0.999000**
- Balanced accuracy: **0.999000**
- Macro precision: **0.999002**
- Macro recall: **0.999000**
- Macro F1: **0.999000**

Confusion matrix는 행이 실제, 열이 예측이며 순서는 `HEALTHY`, `DISEASE_SUSPECTED`이다.

| 실제\예측 | HEALTHY | DISEASE_SUSPECTED |
| --- | --- | --- |
| HEALTHY | 499 | 1 |
| DISEASE_SUSPECTED | 0 | 500 |

## 클래스별 지표

| 클래스 | support | precision | recall | F1 |
| --- | --- | --- | --- | --- |
| HEALTHY | 500 | 1.000000 | 0.998000 | 0.998999 |
| DISEASE_SUSPECTED | 500 | 0.998004 | 1.000000 | 0.999001 |

## 품종별

| 품종 | N | accuracy | balanced | macro F1 | 병해 recall |
| --- | --- | --- | --- | --- | --- |
| 느타리 | 200 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 양송이 | 200 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 큰느타리 | 200 | 0.995000 | 0.995000 | 0.995000 | 1.000000 |
| 팽이 | 200 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 표고 | 200 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

## 날짜·카메라 최저 성능

- 날짜 최저: `2021-11-04` accuracy=0.998000, 병해 recall=-
- camera_id 최저: `2` accuracy=0.993464, 병해 recall=1.000000
- 2021-11-04는 HEALTHY만, 2021-11-26/12-04는 DISEASE_SUSPECTED만 포함하므로 날짜별 클래스 비교에 주의해야 한다.

## 병해를 HEALTHY로 반환한 오류

| 병해 종류 | 오류 |
| --- | --- |
| 세균갈색무늬병 | 0 |
| 솜털곰팡이병 | 0 |
| 푸른곰팡이병 | 0 |
| 흰곰팡이병 | 0 |

## Confidence 구간

| 구간 | N | 비율 | 정답 | 오류 | accuracy |
| --- | --- | --- | --- | --- | --- |
| [0.50, 0.60) | 1 | 0.10% | 1 | 0 | 1.000000 |
| [0.60, 0.70) | 7 | 0.70% | 6 | 1 | 0.857143 |
| [0.70, 0.80) | 2 | 0.20% | 2 | 0 | 1.000000 |
| [0.80, 0.90) | 16 | 1.60% | 16 | 0 | 1.000000 |
| [0.90, 0.95) | 16 | 1.60% | 16 | 0 | 1.000000 |
| [0.95, 1.00] | 958 | 95.80% | 958 | 0 | 1.000000 |

## UNCERTAIN 후보

규칙은 `top1_confidence < threshold`이다. 클래스 recall은 전체 실제 클래스가 분모이며 UNCERTAIN은 미회수로 처리한다.

| threshold | 자동 비율 | UNCERTAIN | 자동 accuracy | HEALTHY recall | DISEASE recall | 병해→정상 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.60 | 99.90% | 0.10% | 0.998999 | 0.996000 | 1.000000 | 0 |
| 0.70 | 99.20% | 0.80% | 1.000000 | 0.984000 | 1.000000 | 0 |
| 0.80 | 99.00% | 1.00% | 1.000000 | 0.980000 | 1.000000 | 0 |
| 0.90 | 97.40% | 2.60% | 1.000000 | 0.956000 | 0.992000 | 0 |
| 0.95 | 95.80% | 4.20% | 1.000000 | 0.930000 | 0.986000 | 0 |

이 표는 현재 Validation 결과를 설명하는 후보 분석이며 서비스 최종 threshold를 확정하지 않는다.

## 오분류 review

- 전체 오분류: **1장**
- review 저장: **1장**
- 오분류가 100장을 넘으면 top1 confidence가 높은 순으로 최대 100장만 저장한다.
- contact sheet: `artifacts/health_pilot_date_holdout/evaluation/error_contact_sheet.jpg`

## 안전 검증

- 모델 파일 변경: 0
- Validation 이미지 변경: 0
- Test 이미지 접근: 0
- 기존 artifacts 변경: 0
- 허용 출력 외 파일 생성: 0
- 로컬 절대경로 노출: 0
