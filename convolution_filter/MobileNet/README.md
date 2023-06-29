# MobileNet

MobileNet은 Google에서 개발한 컴퓨터 비전을 위한 경량화된 딥 러닝 모델 아키텍처입니다.
이 모델은 효율성과 성능 사이의 균형을 맞추기 위해 설계되었으며, 특히 모바일 및 임베디드 비전 애플리케이션에 적합합니다.

## 주요 기술

MobileNet은 두 가지 주요 기술, 즉 **깊이별 컨볼루션(depthwise convolution)**과 **점별 컨볼루션(pointwise convolution)**을 사용하여 모델의 크기와 계산량을 줄입니다.
이러한 모델이 더 적은 리소스를 사용하면서도 높은 정확도를 유지할 수 있게 해줍니다.

### 깊이별 컨볼루션(depthwise convolution)

### 점별 컨볼루션(pointwise convolution)

## 버전(발전 방향)

### MobileNetV1

MobileNet의 첫 번째 버전인 MobileNetV1은 2017년에 소개되었습니다. 이 모델은 <U>기존의 컨볼루션 신경망(CNN)모델들에 비해 훨씬 적은 파라미터</U>를 사용하면서도 비슷한 성능을 보여주었습니다.

### MobileNetV2

MobileNetV2는 2018년에 발표되었으며, 이 모델은 이전 버전에 비해서 성능을 향상 시키기 위해 잔차 연결(residual connection)과 선형 병목(linear bottleneck)을 도입했습니다. 이러한 기술은 모델의 효율성을 높이고, 정보의 손실을 최소화하는 데 도움이 되었습니다.

#### 잔차 연결(residual connection)

#### 선형 병목(linear bottleneck)

### MobileNetV3

MobileNetV3는 2019년에 발표되었으며, 이 모델은 자동화된 검색 알고리즘을 사용하여 아키텍처를 최적화했습니다. 이 알고리즘은 모델의 성능을 더욱 향상시키는 데 도움이 되었습니다.

## 인용

[Mobile-Former: Bridging MobileNet and Transformer](https://arxiv.org/abs/2108.05895)
이 논문에서는 MobileNet과 Transformer 사이의 두 가지 방향의 다리를 설계합니다. 이 구조는 MobileNet의 로컬 처리와 Transformer의 글로벌 상호작용의 장점을 활용합니다.

[A Novel Image Classification Approach via Dense-MobileNet Models](https://www.hindawi.com/journals/misy/2020/7602384/)
이 논문에서는 Dense-MobileNet 모델을 통한 새로운 이미지 분류 방법을 제안합니다. 이 방법은 네트워크의 매개변수 수를 더욱 줄이고 분류 정확도를 향상시킵니다.

## 참고자료

1. []()
