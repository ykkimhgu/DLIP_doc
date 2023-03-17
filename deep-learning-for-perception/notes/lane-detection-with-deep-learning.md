# Lane Detection with Deep Learning

## Reference

### \[집콕]자율주행 인공지능 시스템

## Introduction

![](<../../.gitbook/assets/image (337).png>)

![](<../../.gitbook/assets/image (336).png>)

![](<../../.gitbook/assets/image (345).png>)

![](<../../.gitbook/assets/image (334).png>)

### Other issues

Highly curved lanes, Occlusions, Road crossings

![](<../../.gitbook/assets/image (335).png>)

## Dateset for Lanes

### CU-LANE: Hongkong dataset

## Deep Learning-based Lane Detection

### Semantic Segmentation

Uses Encoder-Decoder

![](<../../.gitbook/assets/image (344).png>)

#### Drivable Area Segmentation (Pizzati, 2019)

![](<../../.gitbook/assets/image (339).png>)

#### LaneNet (Wang, 2018)

![](<../../.gitbook/assets/image (342).png>)

Deep Learning 기반 Lane Detection 성능 지표 (CU-Lane 데이터셋)

![](<../../.gitbook/assets/image (340).png>)

###

### UltraFast (UFAST) Lane Detection

1. Grid Cell과 ROI (수평선 아래)의 도입으로 전체 연산량 대폭 감소
2. 차선을 따라 기울기가 일정함을 고려한 (Domain Knowledge) 손실 함수 설계로 차선 탐지 성능 향상
3. 학습 시에만 사용하는 Auxiliary Branch로 Global & Local한 Feature 정보를 통합하여 차선을 탐지하고 그 정확도를 Main Branch와 별도의 Loss Function으로 구성하여 전체 학습 성능을 향상 (Down Sampling으로 인한 정보 손실 감소, GoogLeNet과 유사)

![](<../../.gitbook/assets/image (338).png>)

![](<../../.gitbook/assets/image (343).png>)

![](<../../.gitbook/assets/image (341).png>)
