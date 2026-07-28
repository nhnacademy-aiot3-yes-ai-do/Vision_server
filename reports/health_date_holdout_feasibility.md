# 건강 체크 날짜 홀드아웃 구성 가능성

## 감사 범위

- detection 개발 후보: 256,946장
- 고정 Test 제외: 46,442장; 최적화·선택에 사용 0장
- 개발 촬영 날짜: 50개
- 이미지/ZIP 바이트 접근, 이미지 추출, 모델 로드·학습: 없음

## 날짜 편향

- 날짜 lookup: 93.79%
- 품종+날짜 lookup: 97.59%
- 날짜+카메라 lookup: 95.30%
- 항상 다수 상태 baseline: 72.58%
- 현재 detection Train/Validation 날짜 중복: 38개
- 현재 raw/masked 6,000장 날짜 중복: 50개
- 정상 날짜 범위: 2021-10-19 ~ 2021-11-30
- 병해 날짜 범위: 2021-11-11 ~ 2021-12-08
- 날짜 유형: 정상 전용 24일 / 병해 전용 17일 / 혼합 9일

위 lookup은 metadata를 같은 데이터에서 다수결한 진단치이며 모델 성능이 아니다.
그럼에도 품종+날짜 97.59%는 수집 날짜·환경 shortcut 위험이 매우 큼을 뜻한다.

## 후보 비교

| 후보 | 그룹 키 | 정확 6,000 | 전역 날짜 중복 | 선언 그룹 중복 | Train 공통 가용 | Validation 공통 가용 | 타깃 상한 가용 합계 | 기존 split 변경 행 | 판단 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| global_capture_date | capture_date | 가능 | 0 | 0 | 8,595 | 161 | 6,000 | 53,018 | - |
| species_capture_date | species + capture_date | 가능 | 5 | 0 | 8,595 | 104 | 6,000 | 50,752 | 선언한 그룹은 분리되지만 동일 날짜가 다른 품종을 통해 양쪽 split에 존재할 수 있음 |
| chronological | capture_date | 불가 | 0 | 0 | 0 | 1,942 | 1,000 | 118,535 | 정상/병해 수집 시기가 분리되어 정확 균형 분할 불가 |
| contiguous_validation_range | capture_date | 불가 | 0 | 0 | 2,038 | 0 | 5,000 | 107,507 | 연속 구간으로는 모든 품종의 정상/병해를 양쪽에 유지 불가 |
| species_camera_capture_date | species + camera_id + capture_date | 가능 | 38 | 0 | 8,559 | 1,973 | 6,000 | 0 | 그룹 중복은 없지만 전역 날짜 분리를 보장하지 않음 |

## 결론

1. 전역 날짜 홀드아웃: **가능**
2. 6,000장 균형 구성: **가능**
3. 가능한 목표 수량: Train 5,000장 / Validation 1,000장
4. 권장안의 Train/Validation 날짜 중복: **0개**
5. 엄격한 global camera_id 분리 추가 시 카메라 중복: **0**
6. chronological 및 연속 날짜 구간: 수집 시기가 상태별로 갈려 정확 균형 구성이 불가능하다.
7. 고정 Test 날짜 36일 중 development와 겹치는 날짜는 36일이며, 권장 Train/Test 34일, Validation/Test 2일이 겹친다. Test 겹침 날짜를 development에서 전부 제외하면 부족 셀은 양송이/HEALTHY 134/600, 큰느타리/HEALTHY 104/600이므로 현재 고정 Test까지 포함한 정확 6,000장 3-way date-isolated split은 불가능하다.
8. 보고된 raw/masked 99.9~100%는 날짜·카메라·배경 shortcut 조합으로 설명될 가능성이 크지만 metadata만으로 인과를 확정할 수 없다.
9. mask-only lookup 59.62%, Train 규칙→Validation 58.00%이므로 회색 mask 존재 자체도 보조 shortcut이 될 수 있다. 이는 완전한 timestamp ablation이 아니다.
10. 외부 스마트폰·새 환경 평가 전 서비스 성능으로 인정하거나 배포 판단에 사용하는 것은 부적절하다.

## Raw/Masked 대응 검증

- raw/masked 행 수: 6,000/6,000
- 이미지 identity 불일치: 0건
- split·class·crop·source 경로·mask geometry 불일치: 0건
- timestamp overlap/mask 적용 플래그 불일치: 0건
