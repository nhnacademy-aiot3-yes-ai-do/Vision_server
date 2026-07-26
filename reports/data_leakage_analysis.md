# Train/Validation 데이터 누수 가능성 검사

분석 범위: 전체 라벨 JSON 스트리밍 스캔

- 처리 성공 JSON: 332,100개
- 실패 JSON: 0개
- 체크포인트에서 재개: 아니요
- 이미지 바이트 읽기/디코딩: 수행하지 않음
- ZIP 전체 CRC 검사: 수행하지 않음
- 중복 키는 프로젝트 내부 SQLite에서 집계했으며 전체 값을 메모리에 보관하지 않았습니다.
- 이미지명과 stem은 경로 구분자를 정규화하고 대소문자를 구분하지 않아 보수적으로 비교했습니다.
- 이 결과는 중복 가능성 신호이며 곧바로 데이터 누수의 원인을 확정하지 않습니다.

| 검사 | 양쪽에 존재하는 키 | train 레코드 | validation 레코드 |
|---|---:|---:|---:|
| 동일 IMAGE_FILE_NAME | 0 | 0 | 0 |
| 동일 이미지 stem | 0 | 0 | 0 |
| 품종+카메라+날짜+시간 촬영 세션 | 994 | 152,141 | 27,563 |
| 동일 카메라+동일 촬영 날짜 | 560 | 236,663 | 35,354 |

## 동일 IMAGE_FILE_NAME 예시

- 발견되지 않음

## 동일 이미지 stem 예시

- 발견되지 않음

## 품종+카메라+날짜+시간 촬영 세션 예시

```json
[
  {
    "species": "양송이",
    "camera_id": "16",
    "capture_date": "2021-11-04",
    "capture_time": "15:20:10",
    "train_count": 2625,
    "validation_count": 233
  },
  {
    "species": "양송이",
    "camera_id": "17",
    "capture_date": "2021-11-04",
    "capture_time": "15:40:12",
    "train_count": 2208,
    "validation_count": 199
  },
  {
    "species": "양송이",
    "camera_id": "7",
    "capture_date": "2021-11-05",
    "capture_time": "11:00:10",
    "train_count": 1592,
    "validation_count": 190
  },
  {
    "species": "양송이",
    "camera_id": "1",
    "capture_date": "2021-11-09",
    "capture_time": "05:20:11",
    "train_count": 1455,
    "validation_count": 178
  },
  {
    "species": "양송이",
    "camera_id": "4",
    "capture_date": "2021-11-09",
    "capture_time": "04:40:13",
    "train_count": 1335,
    "validation_count": 172
  },
  {
    "species": "팽이",
    "camera_id": "7",
    "capture_date": "2021-11-09",
    "capture_time": "15:40:13",
    "train_count": 1068,
    "validation_count": 17
  },
  {
    "species": "양송이",
    "camera_id": "9",
    "capture_date": "2021-11-17",
    "capture_time": "11:00:11",
    "train_count": 947,
    "validation_count": 16
  },
  {
    "species": "양송이",
    "camera_id": "18",
    "capture_date": "2021-11-04",
    "capture_time": "16:00:11",
    "train_count": 840,
    "validation_count": 85
  },
  {
    "species": "팽이",
    "camera_id": "12",
    "capture_date": "2021-11-09",
    "capture_time": "15:40:13",
    "train_count": 894,
    "validation_count": 12
  },
  {
    "species": "팽이",
    "camera_id": "11",
    "capture_date": "2021-11-09",
    "capture_time": "15:40:13",
    "train_count": 897,
    "validation_count": 4
  },
  {
    "species": "팽이",
    "camera_id": "10",
    "capture_date": "2021-11-09",
    "capture_time": "15:40:13",
    "train_count": 895,
    "validation_count": 2
  },
  {
    "species": "팽이",
    "camera_id": "8",
    "capture_date": "2021-11-09",
    "capture_time": "15:40:13",
    "train_count": 720,
    "validation_count": 153
  },
  {
    "species": "양송이",
    "camera_id": "2",
    "capture_date": "2021-11-30",
    "capture_time": "15:00:10",
    "train_count": 603,
    "validation_count": 232
  },
  {
    "species": "팽이",
    "camera_id": "9",
    "capture_date": "2021-11-09",
    "capture_time": "15:40:13",
    "train_count": 538,
    "validation_count": 287
  },
  {
    "species": "양송이",
    "camera_id": "14",
    "capture_date": "2021-11-04",
    "capture_time": "14:40:11",
    "train_count": 752,
    "validation_count": 69
  },
  {
    "species": "팽이",
    "camera_id": "19",
    "capture_date": "2021-11-09",
    "capture_time": "14:40:45",
    "train_count": 732,
    "validation_count": 4
  },
  {
    "species": "큰느타리",
    "camera_id": "1",
    "capture_date": "2021-11-17",
    "capture_time": "16:00:15",
    "train_count": 731,
    "validation_count": 2
  },
  {
    "species": "팽이",
    "camera_id": "14",
    "capture_date": "2021-11-09",
    "capture_time": "14:40:45",
    "train_count": 729,
    "validation_count": 4
  },
  {
    "species": "팽이",
    "camera_id": "16",
    "capture_date": "2021-11-09",
    "capture_time": "14:40:45",
    "train_count": 728,
    "validation_count": 4
  },
  {
    "species": "팽이",
    "camera_id": "18",
    "capture_date": "2021-11-09",
    "capture_time": "14:40:45",
    "train_count": 717,
    "validation_count": 10
  }
]
```

## 동일 카메라+동일 촬영 날짜 예시

```json
[
  {
    "camera_id": "1",
    "capture_date": "2021-11-09",
    "train_count": 3622,
    "validation_count": 805
  },
  {
    "camera_id": "4",
    "capture_date": "2021-11-09",
    "train_count": 3105,
    "validation_count": 725
  },
  {
    "camera_id": "1",
    "capture_date": "2021-11-17",
    "train_count": 3747,
    "validation_count": 78
  },
  {
    "camera_id": "9",
    "capture_date": "2021-11-17",
    "train_count": 2908,
    "validation_count": 277
  },
  {
    "camera_id": "3",
    "capture_date": "2021-11-09",
    "train_count": 2326,
    "validation_count": 720
  },
  {
    "camera_id": "16",
    "capture_date": "2021-11-04",
    "train_count": 2774,
    "validation_count": 244
  },
  {
    "camera_id": "2",
    "capture_date": "2021-11-17",
    "train_count": 2831,
    "validation_count": 28
  },
  {
    "camera_id": "7",
    "capture_date": "2021-11-17",
    "train_count": 2610,
    "validation_count": 203
  },
  {
    "camera_id": "3",
    "capture_date": "2021-11-17",
    "train_count": 2647,
    "validation_count": 110
  },
  {
    "camera_id": "14",
    "capture_date": "2021-11-17",
    "train_count": 2393,
    "validation_count": 343
  },
  {
    "camera_id": "8",
    "capture_date": "2021-11-17",
    "train_count": 2509,
    "validation_count": 154
  },
  {
    "camera_id": "6",
    "capture_date": "2021-11-17",
    "train_count": 2204,
    "validation_count": 446
  },
  {
    "camera_id": "13",
    "capture_date": "2021-11-17",
    "train_count": 2340,
    "validation_count": 294
  },
  {
    "camera_id": "2",
    "capture_date": "2021-11-09",
    "train_count": 2093,
    "validation_count": 516
  },
  {
    "camera_id": "5",
    "capture_date": "2021-11-09",
    "train_count": 2033,
    "validation_count": 537
  },
  {
    "camera_id": "4",
    "capture_date": "2021-11-17",
    "train_count": 2498,
    "validation_count": 69
  },
  {
    "camera_id": "17",
    "capture_date": "2021-11-04",
    "train_count": 2323,
    "validation_count": 199
  },
  {
    "camera_id": "7",
    "capture_date": "2021-11-05",
    "train_count": 2230,
    "validation_count": 217
  },
  {
    "camera_id": "15",
    "capture_date": "2021-11-17",
    "train_count": 1999,
    "validation_count": 439
  },
  {
    "camera_id": "11",
    "capture_date": "2021-11-17",
    "train_count": 2264,
    "validation_count": 40
  }
]
```
