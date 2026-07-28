# 버섯 건강 체크 end-to-end 평가

## 범위

- 평가 이미지: **100장**
- 품종×상태별 10장, Test 0장
- 선택 ZIP member read: **100개**
- 전체 ZIP 압축 해제·학습·모델 변환: 수행하지 않음

## Detector provenance 중복 감사

- detector Train 동일 원본: 후보 **169장**, 최종 **0장**
- detector Validation 동일 원본: 후보 **15장**, 최종 **0장**
- 최종 표본 중 detector Train과 동일 날짜 영향을 받는 행: **100/100장**
- 최종 표본 중 detector Train과 동일 camera_id 영향을 받는 행: **100/100장**
- 최종 표본 중 detector Train과 동일 species+camera 영향을 받는 행: **88/100장**

## Detector crop end-to-end 지표

- 품종 탐지 성공은 정답 품종이 결과에 하나 존재함을 뜻하며, bbox 품질은 별도 IoU로 본다.
- 정답 품종 탐지 성공률: **1.000000**
- 예측 품종 집합 완전 일치율: **1.000000**
- 예상 밖 추가 품종 탐지 이미지: **0장**
- 건강 분석 가능률: **1.000000**
- 아래 target-species health Accuracy/Macro F1/Recall은 전체 100장을 분모로 하며 UNCERTAIN과 정답 품종 탐지 실패는 오답/FN으로 집계한다. 추가 품종 오탐은 품종 집합 완전 일치율에서 별도 평가한다.
- target-species health E2E Accuracy: **1.000000**
- target-species health E2E Macro F1: **1.000000**
- 정확한 품종 집합 + 건강 상태 Accuracy: **1.000000**
- 자동 판정 대상 Accuracy: **1.000000**
- HEALTHY recall: **1.000000**
- DISEASE_SUSPECTED recall: **1.000000**
- UNCERTAIN: **0장 (0.00%)**
- 병해→HEALTHY: **0장**

## Ground-truth crop 비교

- GT crop accuracy: **1.000000**
- 전체 E2E detector crop accuracy: **1.000000**
- 전체 GT - E2E accuracy: **+0.000000** (탐지 누락과 crop 차이를 함께 포함)
- 탐지 성공 paired 이미지: **100장**
- paired GT - detector accuracy: **+0.000000**
- detector/GT crop 평균 IoU: **0.9829534134402602**

## 품종별 detector crop

| 품종 | N | 탐지 성공 | accuracy | macro F1 | 병해 recall |
| --- | --- | --- | --- | --- | --- |
| 느타리 | 20 | 1.000 | 1.000 | 1.000 | 1.000 |
| 양송이 | 20 | 1.000 | 1.000 | 1.000 | 1.000 |
| 큰느타리 | 20 | 1.000 | 1.000 | 1.000 | 1.000 |
| 팽이 | 20 | 1.000 | 1.000 | 1.000 | 1.000 |
| 표고 | 20 | 1.000 | 1.000 | 1.000 | 1.000 |

## 제한

- 정상은 2021-11-04, 병해는 2021-11-26/12-04로 날짜와 상태가 결합되어 있다.
- 원본 직접 GT crop과 기존 JPEG95 health Validation 입력은 bit-identical하지 않다.
- 동일 원본은 배제했지만 detector Train과 같은 촬영 날짜·카메라 환경은 남아 있다.
- 내부 AIHub 환경 평가이며 외부 스마트폰 일반화 성능으로 해석하지 않는다.

## 안전 검증

- Test 이미지 사용: 0장
- 원본 ZIP·모델·기존 데이터셋·기존 artifacts 변경: 0
- 허용 출력 외 생성 및 로컬 절대경로 노출: 0
