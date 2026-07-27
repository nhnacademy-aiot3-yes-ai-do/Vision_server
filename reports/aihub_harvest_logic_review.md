# AIHub 수확 적기 판단 로직 정적 검토

## 판정 요약

확인된 설계는 **이미지 분류, 회귀, segmentation, 규칙 기반 판정이 아니라 품종+번호 단계를 클래스화한 YOLO 객체 탐지**다.

```text
이미지
→ 608×608×3 Darknet 입력
→ YOLOv3-tiny 또는 YOLOv4 3-head detector
→ bbox + objectness + 12개 품종_단계 클래스 점수
→ YOLOv4 cfg의 greedy NMS
→ 단계 번호를 수확 상태로 바꾸는 로직은 제공되지 않음
```

최종 task 가중치가 없어 이 흐름은 **설정으로 확인한 의도**이며 실행 가능한 완성 파이프라인이 아니다.

## 키워드 검색 결과

문서·설정·코드에서 `수확`, `수확기`, `수확 적기`, `생육 단계`, `성장 단계`, `발이`, `과숙`, `harvest_ready`, `growth_stage`, `maturity`, `overmature`, `cap`, `pileus`, `stipe`의 의미를 정의하는 자료는 발견되지 않았다. 수확 관련성은 디렉터리 약어 `harv`와 `obj.names`의 품종별 번호 단계에서만 확인된다.

`mask` 문자열은 YOLO anchor mask 설정이며 segmentation을 뜻하지 않는다. 범용 binary 내부의 `cap_frame` 같은 OpenCV video capture 문자열도 버섯 갓을 뜻하는 `cap` 라벨 근거로 사용하지 않았다.

## 클래스 구조

Darknet의 0 기반 class index는 `data/_datafile/harv/obj.names`의 행 순서를 따른다.

| class_id 범위 | 원문 패턴 | 해석 가능한 범위 |
| --- | --- | --- |
| 0–2 | `neutali_1..3` | 느타리로 추정되는 3개 번호 단계 |
| 3–5 | `keunneutali_1..3` | 큰느타리로 추정되는 3개 번호 단계 |
| 6–8 | `paeng-i_1..3` | 팽이로 추정되는 3개 번호 단계 |
| 9–11 | `pyogo_1..3` | 표고로 추정되는 3개 번호 단계 |

정확한 12개 순서는 `reports/aihub_harvest_class_mapping.csv`에 있다.

확인할 수 없는 의미:

- `_1`이 초기 생육인지
- `_2`가 중기 또는 수확 적기인지
- `_3`이 후기·수확 적기·과숙 중 무엇인지
- 한 단계 이상을 READY로 처리하는지
- 미성숙/성숙/과숙의 경계 기준

5단계 구조, 명시적인 과숙 클래스, READY/NOT_READY 클래스는 없다.

## 모델별 입력·출력

| 항목 | YOLOv3-tiny harv | YOLOv4 harv |
| --- | --- | --- |
| framework | Darknet | Darknet |
| architecture | YOLOv3-tiny, 3 detection heads | YOLOv4, 3 detection heads |
| input | 608×608×3 | 608×608×3 |
| classes | 12 | 12 |
| head filters | 51 | 51 |
| anchors/head | 3 | 3 |
| output vector/anchor | x,y,w,h, objectness, 12 class scores | x,y,w,h, objectness, 12 class scores |
| grid shape | 구조상 19×19, 38×38, 76×76로 추론 | 구조상 19×19, 38×38, 76×76로 추론 |
| raw candidates | 구조상 합계 22,743개로 추론 | 구조상 합계 22,743개로 추론 |
| NMS | cfg에 방식 미기재 | `greedynms`, `beta_nms=0.6` |
| final weight | 없음 | 없음 |

런타임 API가 반환하는 정확한 tensor 배치/메모리 레이아웃은 소스가 없어 확인하지 못했다. 위 shape는 608 입력과 3-scale YOLO 구조에서 유도한 모델 구조 수준의 설명이다.

## 전처리

### 확인됨

- 입력 크기: 608×608
- 채널 수: 3
- 학습 augmentation:
  - saturation 1.5
  - exposure 1.5
  - hue 0.1
  - YOLOv4는 mosaic 1

### 확인 불가

- RGB 또는 BGR
- 픽셀 normalization과 정확한 scale
- 단순 resize 또는 letterbox
- padding 색과 보간 방식
- 전체 이미지와 객체 crop 중 어떤 배포 입력을 전제로 했는지

cfg의 `random`과 `resize=2.5`는 학습/multiscale 설정이며 배포 전처리 구현을 증명하지 않는다.

## 추론·후처리

### 확인됨

- 모델 유형은 bbox object detector다.
- YOLOv4 yolo layer는 `nms_kind=greedynms`, `beta_nms=0.6`을 가진다.
- `uselib.exe`의 제한적 정적 문자열은 bbox의 class_id, name, absolute coordinates, confidence를 JSON 형태로 표현하는 범용 예제를 보여 준다.

### 확인 불가

- 수확 모델 추론 진입점
- confidence threshold
- 최종 NMS IoU threshold
- multi-label 허용 여부
- 품종과 단계가 충돌할 때의 선택 규칙
- detection이 없을 때의 처리
- 여러 버섯의 단계가 다를 때 사진 전체 수확 상태를 합성하는 규칙
- 단계→READY/NOT_READY 매핑

`ignore_thresh=.7`, `truth_thresh=1`, `iou_thresh=0.213`은 yolo layer의 학습/label-assignment 설정이다. 이를 배포 confidence나 최종 NMS IoU로 보고하지 않는다.

## 수확 판단 유형별 결론

| 후보 | 결과 | 근거 |
| --- | --- | --- |
| 객체 탐지 클래스 | 설계상 해당 | 품종+번호 단계 12개와 bbox YOLO head |
| 이미지 분류 클래스 | 아님 | classification head/분류 코드 없음 |
| segmentation 결과 | 아님 | mask/polygon head 없음 |
| 숫자 회귀 | 아님 | 날짜·일수·크기 scalar output 없음 |
| 명시적 threshold 규칙 | 없음 | 크기·색상·기간 규칙 코드 없음 |
| 여러 모델 후처리 | 확인 불가 | 조합 코드 없음 |

## 날짜·일수·정량 기준

- 정확한 수확 날짜 출력: 없음
- 수확까지 남은 일수 출력: 없음
- 갓 너비 threshold: 없음
- 대 길이 threshold: 없음
- 갓/대 비율: 없음
- bbox/화면 면적 비율: 없음
- 색상값: 없음
- 생육 경과일: 없음
- stage confidence threshold: 없음

신경망 클래스 예측만 있었다고 하더라도 사람이 이해할 수 있는 수확 기준을 weight에서 역추출할 수 없다. 이 패키지는 최종 weight와 단계 의미마저 없으므로 판단 기준을 복원할 근거가 더 부족하다.

## 환경과 실행 가능성

- 실행 파일: Windows x64 Darknet GPU build
- 예상 framework 계열: YOLOv4/CIoU/mosaic/greedy NMS를 지원하는 AlexeyAB 계열 Darknet으로 추정; 정확 release/commit 미상
- 정적 의존성: CUDA 11, cuDNN 8, OpenCV, NVIDIA driver, MSVC runtime
- CPU 전용 지원: 제공 build는 embedded usage상 `-nogpu`를 지원하지 않으며 별도 CPU build 필요
- 정확한 Darknet 버전/commit: 확인되지 않음
- 제공 설정 그대로 실행: 불가
  - 최종 수확 weight 없음
  - train/test 목록 없음
  - `.data`의 일부 경로 불일치
  - 수확용 추론 소스/명령 없음
  - `uselib.exe`는 누락된 COCO/yolov3 파일을 가리키는 범용 demo

따라서 legacy Windows 환경을 구성하는 것만으로 해결되지 않는다. 먼저 공식 최종 weight, 클래스 의미표, 전처리·추론 파라미터를 확보해야 한다.
