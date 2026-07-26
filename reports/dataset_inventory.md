# AIHub 지능형 스마트팜 통합 데이터(버섯) 인벤토리

생성 시각: 2026-07-26T19:04:20.159028+09:00

## 데이터 위치

- `MUSHROOM_DATASET`: `${MUSHROOM_DATASET}`
- `TRAIN_LABEL_ARCHIVES`: `${MUSHROOM_DATASET}/1.Training/라벨링데이터`
- `TRAIN_IMAGE_ARCHIVES`: `${MUSHROOM_DATASET}/1.Training/원천데이터`
- `VAL_LABEL_ARCHIVES`: `${MUSHROOM_DATASET}/2.Validation/라벨링데이터`
- `VAL_IMAGE_ARCHIVES`: `${MUSHROOM_DATASET}/2.Validation/원천데이터`

## 전체 요약

- ZIP: 20개, 합계 47.16 GiB
- 중앙 디렉터리 기준 내부 파일: 664,200개
- JSON: 332,100개
- 이미지: 332,100개
- 정상 대응: 10/10쌍
- 읽기 성공 JSON 샘플: 560개
- 이미지 매칭: `{"matched": 200}`

## 안전 범위

- 원본 ZIP은 읽기 전용으로 열었고 이름 변경·이동·삭제·수정하지 않았습니다.
- 전체 압축 해제 및 이미지 디코딩을 수행하지 않았습니다.
- 전체 ZIP CRC 무결성 검사는 수행하지 않았습니다. 인벤토리의 손상 여부는 중앙 디렉터리 판독 범위만 확인되었습니다.
- 제한된 매칭 이미지 샘플만 artifacts/sample_extract 아래에 추출했습니다.

## 샘플 기준 평균 결측률

| 라벨 | 평균 결측률 |
|---|---:|
| 버섯 품종 | 0.00% |
| 배양·생육·병해 구분 | 0.00% |
| 생육 단계 | 100.00% |
| 정상 여부 | 0.00% |
| 병해 종류 | 64.29% |
| bounding box | 0.00% |
| segmentation 또는 polygon | 92.86% |
| 온도 | 0.18% |
| 습도 | 0.18% |
| CO2 | 0.18% |
| 조도 | 6.43% |
| 갓 직경 | 100.00% |
| 갓 두께 | 100.00% |
| 대 길이 | 100.00% |
| 대 두께 | 100.00% |
| 총중량 | 100.00% |
| 촬영 날짜 | 0.00% |
| 카메라 ID | 0.00% |
| 농가 또는 재배사 식별값 | 100.00% |
| 이미지 파일명 | 0.00% |

세부 경로·타입은 `label_schema.json`, 품종/작업별 결측률과 값 분포는 `label_distribution.csv`를 참조하십시오.
