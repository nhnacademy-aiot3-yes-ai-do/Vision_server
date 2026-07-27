# AIHub 버섯 활용 모델 읽기 전용 정적 감사

## 감사 범위와 안전성

- 모델 루트 아래의 파일명, 크기, 텍스트 설정, PE 헤더/import, Darknet weight 헤더만 정적으로 확인했다.
- 실행 파일·Python 파일을 실행하거나 import하지 않았다.
- 모델 가중치를 로드·추론·변환하지 않았다.
- 압축 해제, 학습, 패키지 설치, 원본 수정·이동·복사를 수행하지 않았다.
- 경로는 모두 모델 루트 기준 상대경로로 기록했다.

## 최우선 결론

`harv`라는 이름의 **수확 관련 객체 탐지 모델 정의**는 있다. 그러나 패키지 안에는 학습 완료 task 가중치가 없고 `_export/`도 비어 있다. 따라서 **실행 가능한 학습 완료 수확 적기 모델이 실제 포함됐다고 확인할 수 없다.**

별도 생육 단계 모델은 없다. `harv` 설정은 느타리·큰느타리·팽이·표고와 번호 단계 `_1`, `_2`, `_3`을 결합한 12개 YOLO 객체 탐지 클래스를 정의한다. 번호 세 단계의 정확한 의미와 어느 단계가 수확 적기인지를 설명하는 문서·코드가 없으므로 초기/중기/후기 또는 READY/NOT_READY로 임의 매핑할 수 없다.

## 전체 인벤토리

- 파일: **819개**
- 총 크기: **2,114,816,748 bytes** (약 2.115 GB)
- 디렉터리: 20개
- `_export/`: 빈 디렉터리

| 확장자 | 파일 수 | 총 크기(bytes) | 해석 |
| --- | ---: | ---: | --- |
| `.dll` | 38 | 1,882,278,344 | Windows/CUDA/cuDNN/OpenCV 런타임 |
| `.137` | 1 | 170,038,676 | YOLOv4 partial conv 초기 가중치 |
| `.exe` | 4 | 31,064,640 | Darknet·예제·런타임 설치 파일 |
| `.15` | 1 | 30,973,396 | YOLOv3-tiny partial conv 초기 가중치 |
| `.png` | 760 | 389,863 | Darknet 화면 표시용 ASCII glyph |
| `.cfg` | 6 | 48,487 | YOLOv3-tiny/YOLOv4 task 설정 |
| `.lib` | 1 | 21,336 | Darknet Windows import library |
| `.py` | 1 | 963 | glyph 생성 보조 코드; 추론 코드 아님 |
| `.names` | 3 | 559 | class/harv/pest 클래스 이름 |
| `.data` | 3 | 438 | Darknet 데이터 설정 |
| `.txt` | 1 | 46 | 의미가 문서화되지 않은 4개 counter |

전체 파일별 경로와 크기는 `reports/aihub_model_inventory.csv`에 기록했다.

## 문서·압축·모델 파일 확인

| 항목 | 결과 |
| --- | --- |
| README, PDF, DOCX, HWP/HWPX | 없음 |
| JSON, YAML, XML | 없음 |
| CFG | 6개 |
| NAMES | 3개 |
| DATA | 3개 |
| Python | 1개; 추론이 아닌 glyph 생성 코드 |
| ZIP, 7z, RAR, TAR, GZ 등 | 없음 |
| 최종 `.weights`, `.h5`, `.pb`, SavedModel | 없음 |
| `.onnx`, `.pt`, `.pth`, `.ckpt`, checkpoint | 없음 |
| partial conv weights | 2개 |

`data/_weights/yolov3-tiny.conv.15`와 `data/_weights/yolov4.conv.137`의 첫 20바이트는 Darknet weight header `major=0, minor=2, revision=5, seen=0`과 일치한다. 파일명과 저장 계층 범위상 backbone 초기화용 partial weights이며 task별 최종 detector 가중치가 아니다.

## 포함된 모델 정의

| 용도 | 아키텍처 | 클래스 | 입력 | 최종 가중치 |
| --- | --- | ---: | --- | --- |
| `class` 품종 | YOLOv3-tiny 3-head, YOLOv4 | 4 | 608×608×3 | 없음 |
| `harv` 수확/번호 단계 | YOLOv3-tiny 3-head, YOLOv4 | 12 | 608×608×3 | 없음 |
| `pest` 병해 | YOLOv3-tiny 3-head, YOLOv4 | 15 | 608×608×3 | 없음 |

### 수확·생육

- 원문 클래스 순서는 `neutali_1..3`, `keunneutali_1..3`, `paeng-i_1..3`, `pyogo_1..3`이다.
- 이름상 느타리·큰느타리·팽이·표고 4품종×3개 번호 단계다.
- 양송이 수확 클래스는 없다.
- `_1/_2/_3`의 의미, 수확 적기 단계, 과숙 단계, READY 변환 규칙은 없다.
- 별도 `growth_stage` 모델이나 이미지 분류 모델은 없다.

### 병해

`data/_datafile/pest/obj.names`에는 느타리·큰느타리·양송이·팽이·표고의 품종×병해 조합 15개가 있다. 이것도 YOLO bbox detector 정의이고 최종 가중치는 없다.

| class_id | 원문 클래스 |
| ---: | --- |
| 0 | `neutali_segyungalsaegmunui` |
| 1 | `neutali_puleungompang-i` |
| 2 | `neutali_huingompang-i` |
| 3 | `keunneutali_segyungalsaegmunui` |
| 4 | `keunneutali_puleungompang-i` |
| 5 | `keunneutali_huingompang-i` |
| 6 | `yangsong-i_segyungalsaegmunui` |
| 7 | `yangsong-i_puleungompang-i` |
| 8 | `yangsong-i_somteolgompang-i` |
| 9 | `paeng-i_segyunseong-geom-eunsseog-eum` |
| 10 | `paeng-i_puleungompang-i` |
| 11 | `paeng-i_huingompang-i` |
| 12 | `pyogo_segyungalsaegmunui` |
| 13 | `pyogo_puleungompang-i` |
| 14 | `pyogo_huingompang-i` |

### 품종

`data/_datafile/class/obj.names`는 `1, 2, 4, 5` 네 클래스만 제공한다. 수확 모델의 네 품종과 대응하고 누락된 `3`이 양송이일 가능성은 있으나 공식 매핑 문서가 없어 추정으로만 남긴다.

## 실행 완전성

- `data/_datafile/*/obj.data`가 참조하는 모든 `train.txt`와 `test.txt`가 없다.
- 다수의 `names`/`valid` 경로가 실제 `_datafile` 대신 존재하지 않는 `data/datafile/...`를 가리킨다.
- `darknet.exe`와 `darknet.dll`은 Windows x64 PE이고 CUDA 11, cuDNN 8, OpenCV, NVIDIA driver를 import한다.
- YOLOv4, CIoU, mosaic, greedy NMS 옵션상 AlexeyAB 계열 Darknet과 호환되는 빌드로 추정되지만 정확한 release/commit은 없다. 정적 PE timestamp는 2021-09-01이다.
- embedded usage 문자열은 이 build가 `-nogpu`를 지원하지 않는다고 표시한다. 따라서 패키지 binary는 호환 NVIDIA GPU/CUDA 환경을 요구하며 CPU 전용 사용에는 별도 build가 필요하다.
- `uselib.exe`는 정적 문자열상 `data/coco.names`, `cfg/yolov3.cfg`, `yolov3.weights`를 사용하는 범용 예제다. 이 파일들은 없고 `harv` 경로도 참조하지 않으므로 수확 추론 진입점으로 볼 수 없다.
- Python 추론 코드는 없다.

결론적으로 설정 경로, 최종 가중치, 단계 의미, task별 추론 코드가 모두 완성되지 않아 **그대로 실행할 수 없다.**

## 학습 데이터와 평가 근거

확인된 것은 cfg의 학습 계획뿐이다.

- YOLOv3-tiny: batch 64, subdivisions 16, learning rate 0.001, max_batches 40,000, steps 32,000/36,000
- YOLOv4 harv: batch 64, subdivisions 16, learning rate 0.0013, max_batches 100,000, steps 80,000/90,000, mosaic 1

다음은 모두 확인할 수 없다.

- 학습/검증/평가 비율
- 수확 단계별·품종별 이미지 수
- 날짜·카메라·재배 객체 단위 분리 여부
- mAP, accuracy, precision, recall
- 학습 카메라와 평가 카메라의 중복
- 외부 사진·스마트폰 사진 일반화 성능

`counters_per_class.txt`의 네 숫자는 설명이 없고 수확 클래스 수 12와도 맞지 않아 수확 분포로 해석하지 않았다.

## 갓·대 및 segmentation 부가 확인

- `cap`, `pileus`, `stipe` 클래스가 없다.
- segmentation/polygon head나 전체 버섯 polygon이 없다.
- cfg의 `mask=`는 YOLO anchor 선택값이며 segmentation mask가 아니다.
- 갓·대 길이·폭 산출 코드가 없다.
- 실제 cm 환산용 scale, 기준물, 카메라 intrinsic/extrinsic 또는 calibration 설정이 없다.
- `data/labels/*.png` 760개는 버섯 mask가 아니라 Darknet 표시용 ASCII 문자 bitmap이다.

따라서 이 자료만으로 갓과 대를 별도 분할하거나 실제 치수를 측정할 수 없다.

## 필수 최종 결론

1. **수확 적기 모델이 실제 존재하는가?** `harv` 모델 정의는 존재하지만 학습 완료 가중치가 없어 실행 가능한 수확 모델의 존재는 확인되지 않았다.
2. **생육 단계 모델이 실제 존재하는가?** 별도 모델은 없다. `harv`가 품종×3개 미정의 번호 단계를 탐지하도록 설계됐을 뿐이다.
3. **수확기 클래스 이름과 순서를 확인했는가?** 원문 12개 이름과 순서는 확인했다. 번호 단계의 의미는 확인하지 못했다.
4. **정확한 날짜를 출력하는가?** 근거가 없다.
5. **READY/NOT_READY 상태만 출력하는가?** 그런 클래스나 후처리 근거가 없다.
6. **수확까지 남은 일수를 출력하는가?** 근거가 없다.
7. **명시적인 크기·색상·기간 기준이 코드에 존재하는가?** 없다.
8. **가중치만 있고 판단 기준을 확인할 수 없는가?** 그보다 더 제한적이다. 최종 가중치 자체도 없고 단계 정의도 없다. 일반적으로 신경망 가중치만으로 사람 기준을 역추출할 수도 없다.
9. **실제 실행 가능한 파일과 추론 코드가 존재하는가?** 범용 Windows Darknet binary는 있으나 task 완성 파일과 수확 추론 소스가 없어 as-is 실행 불가다.
10. **현재 YOLO11n과 결합 가능한가?** 현재 상태에서는 불가하다. 최종 가중치와 단계 의미를 확보하면 검증을 거쳐 결합 가능성을 다시 평가할 수 있다.
11. **그대로 사용, 변환, 재학습 중 무엇이 적절한가?** 우선 누락된 공식 산출물을 확보한다. 확보 실패 시 전문가 라벨을 새로 만들고 현대 YOLO 또는 crop 분류 모델을 재학습하는 것이 적절하다.
12. **부족한 클래스·라벨·데이터는 무엇인가?** 양송이 수확 클래스, 1/2/3 의미표, READY 기준, 과숙 여부, 최종 가중치, 추론 파라미터, 클래스별 수량, 누수 없는 평가, 외부 스마트폰 검증이 부족하다.
13. **현재 구현 가능한 수확 API 출력은 무엇인가?** 신뢰 가능한 stage/READY 예측은 불가능하며 `UNKNOWN/UNAVAILABLE` 상태와 사유만 반환할 수 있다.
