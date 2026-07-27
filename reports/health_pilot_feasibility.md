# 버섯 건강 체크 v1 6,000장 파일럿 가능성

## 결론

정확한 구성이 **가능**하다.

- Test 사용: 0장
- Test 제외 개발 후보: 256,946장
- 제외한 고정 Test: 46,442장
- seed: `20260726`
- 그룹 키: `species + camera_id`
- Train/Validation 그룹 중복: 0개
- 이미지는 추출하지 않았고 manifest도 생성하지 않았다.

## 목표와 가용량

| split | 품종 | 상태 | 계획 수량 | 선택 카메라 가용량 |
| --- | --- | --- | ---: | ---: |
| train | 느타리 | abnormal | 500 | 11,297 |
| train | 느타리 | normal | 500 | 36,763 |
| train | 양송이 | abnormal | 500 | 10,883 |
| train | 양송이 | normal | 500 | 29,508 |
| train | 큰느타리 | abnormal | 500 | 7,271 |
| train | 큰느타리 | normal | 500 | 26,681 |
| train | 팽이 | abnormal | 500 | 10,963 |
| train | 팽이 | normal | 500 | 31,501 |
| train | 표고 | abnormal | 500 | 7,953 |
| train | 표고 | normal | 500 | 23,399 |
| validation | 느타리 | abnormal | 100 | 5,552 |
| validation | 느타리 | normal | 100 | 8,749 |
| validation | 양송이 | abnormal | 100 | 4,863 |
| validation | 양송이 | normal | 100 | 8,148 |
| validation | 큰느타리 | abnormal | 100 | 4,858 |
| validation | 큰느타리 | normal | 100 | 6,814 |
| validation | 팽이 | abnormal | 100 | 4,238 |
| validation | 팽이 | normal | 100 | 9,113 |
| validation | 표고 | abnormal | 100 | 2,579 |
| validation | 표고 | normal | 100 | 5,813 |

Train은 품종별 정상/병해 각 500장, Validation은 각 100장으로 총 5,000/1,000장이다.

## 결정적 카메라 배정

| split | 품종 | camera_id | 카메라 수 |
| --- | --- | --- | ---: |
| train | 느타리 | 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20 | 16 |
| validation | 느타리 | 2, 11, 15, 16 | 4 |
| train | 양송이 | 1, 2, 3, 4, 6, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 | 16 |
| validation | 양송이 | 5, 7, 9, 10 | 4 |
| train | 큰느타리 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 17, 18, 19, 20 | 16 |
| validation | 큰느타리 | 10, 14, 15, 16 | 4 |
| train | 팽이 | 1, 2, 3, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20 | 16 |
| validation | 팽이 | 4, 5, 7, 12 | 4 |
| train | 표고 | 1, 2, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20 | 16 |
| validation | 표고 | 3, 5, 8, 17 | 4 |

## 병해 종류 계획 분포

카메라별 가용 분포를 비례 배분한 feasibility quota다. 아직 실제 이미지 선택 manifest가 아니다.

| split | 병해 종류 | 계획 이미지 |
| --- | --- | ---: |
| train | 세균갈색무늬병 | 990 |
| train | 세균성검은썩음병 | 201 |
| train | 솜털곰팡이병 | 145 |
| train | 푸른곰팡이병 | 718 |
| train | 흰곰팡이병 | 446 |
| validation | 세균갈색무늬병 | 192 |
| validation | 세균성검은썩음병 | 40 |
| validation | 솜털곰팡이병 | 30 |
| validation | 푸른곰팡이병 | 146 |
| validation | 흰곰팡이병 | 92 |

모든 지원 병해가 Train과 Validation에 남도록 카메라 조합을 우선했다.

## 제한

이 파일럿은 Train/Validation 카메라 홀드아웃 가능성을 증명한다. 기존 고정 Test와는 같은 `species+camera_id`가 존재하므로 최종 3-way 카메라 독립 평가로 해석하지 않는다.
