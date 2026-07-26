# 탐지 데이터 manifest 검증 요약

## 범위와 고정 정책

- 대상: 생육·병해 JSON, 5품종(느타리·양송이·큰느타리·팽이·표고)
- 제외: 배양 JSON 및 정제 후 유효 bbox가 없는 JSON
- 클래스: 품종 5개 객체 탐지(class_id 0~4)
- 그룹 키: `species + camera_id + capture_date`
- 고정 seed: `20260726`
- 목표 비율: Train 70% / Validation 15% / Test 15%; 그룹 누수 0을 우선
- Test는 고정된 최종 평가 세트이며 학습·모델 선택에 사용하지 않음

## 최종 규모

- 포함 이미지(JSON): **303,388개**
- 제외 JSON: **28,712개**
- 최종 유효 bbox: **575,702개**

| split | 이미지 수 | 비율 |
| --- | --- | --- |
| train | 210,915 | 69.520% |
| validation | 46,031 | 15.172% |
| test | 46,442 | 15.308% |

## 제외 사유

| 사유 | JSON 수 | 제거 bbox 수 |
| --- | --- | --- |
| all_bboxes_removed | 2 | 3 |
| task_excluded_culture | 28,710 | 0 |

- annotation이 없는 JSON **4개**는 모두 `task_excluded_culture`에 포함된 배양 데이터로, 첫 탐지 후보에서 제외됨

작은 bbox는 자동 제거하지 않았고, annotation이 개체인지 군집인지 자동으로 해석하지 않았다. 원본 annotation 수와 유효 bbox 수를 manifest에 함께 기록했다.

## 검증 결과

- 둘 이상의 split에 걸친 group_key: **0개**
- 둘 이상의 split에 걸친 image_member: **0개**
- 범위 밖 정제 bbox: **0개**
- width/height가 양수가 아닌 정제 bbox: **0개**
- split 미할당 레코드: **0개**

전체 분포 대비 split별 최대 절대 편차:

| 분포 | 최대 절대 편차(pp) |
| --- | --- |
| 품종 | 1.3746 |
| 작업 | 0.2340 |
| 정상 여부 | 0.2340 |
| 병해 종류 | 0.4289 |

## 공식 split 변경

- 최종 포함 레코드 중 공식 Training/Validation과 새 split이 다른 레코드: **107,946개 (35.580%)**

| 공식 split | 새 split | 이미지 수 |
| --- | --- | --- |
| train | test | 40,101 |
| train | train | 189,494 |
| train | validation | 40,083 |
| validation | test | 6,341 |
| validation | train | 21,421 |
| validation | validation | 5,948 |

공식 Validation을 최종 평가셋으로 확정하지 않고, 동일 그룹을 한 split에만 배치해 새 Test를 별도로 고정했다.

## 안전 범위

- 라벨 ZIP의 대상 JSON만 순차적으로 읽음
- 이미지 ZIP은 중앙 디렉터리의 파일명만 조회하고 이미지 바이트를 읽지 않음
- 이미지 디코딩·추출 및 전체 ZIP 압축 해제를 하지 않음
- ZIP 전체 CRC 검사를 하지 않음
- 원본 ZIP을 수정·이동·삭제·이름 변경하지 않음
- YOLO 변환 및 모델 학습을 하지 않음
- 출력에는 ZIP 내부 상대경로와 archive ID만 기록하며 로컬 절대경로를 기록하지 않음
