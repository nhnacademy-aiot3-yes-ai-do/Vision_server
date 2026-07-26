# 아카이브 대응 검사

대응은 인덱스, 품종명, ZIP 중앙 디렉터리 열기 성공 여부를 기준으로 판정했습니다.

| 구분 | 기대 대응 | 라벨 ZIP | 원천 ZIP | 품종 일치 | 열기 | 결과 |
|---|---|---|---|---:|---:|---|
| train | TL1 ↔ TS1 | TL1_느타리.zip | TS1_느타리.zip | True | True | ok |
| train | TL2 ↔ TS2 | TL2_양송이.zip | TS2_양송이.zip | True | True | ok |
| train | TL3 ↔ TS3 | TL3_큰느타리.zip | TS3_큰느타리.zip | True | True | ok |
| train | TL4 ↔ TS4 | TL4_팽이.zip | TS4_팽이.zip | True | True | ok |
| train | TL5 ↔ TS5 | TL5_표고.zip | TS5_표고.zip | True | True | ok |
| validation | VL1 ↔ VS1 | VL1_느타리.zip | VS1_느타리.zip | True | True | ok |
| validation | VL2 ↔ VS2 | VL2_양송이.zip | VS2_양송이.zip | True | True | ok |
| validation | VL3 ↔ VS3 | VL3_큰느타리.zip | VS3_큰느타리.zip | True | True | ok |
| validation | VL4 ↔ VS4 | VL4_팽이.zip | VS4_팽이.zip | True | True | ok |
| validation | VL5 ↔ VS5 | VL5_표고.zip | VS5_표고.zip | True | True | ok |

## 발견 사항

- 네 디렉터리에서 각각 ZIP 5개를 확인했습니다.
- 기대한 10개 라벨↔원천 대응이 모두 확인되었습니다.
