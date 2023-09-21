# Domain Adaptation

**도메인 적응(Domain Adaptation)**은 머신 러닝과 컴퓨터 비전, 자연어 처리 등 다양한 분야에서 사용되는 기술입니다. 이 기술의 주 목적은 하나의 도메인에서 학습된 모델을 다른 도메인에 적용할 수 있도록 하는 것입니다. 여기서 **'도메인'**이라는 용어는 특정한 특성이나 분포를 가진 데이터 집합을 의미합니다.

예를 들어, 어떤 이미지 분류 모델이 도로에서 차량을 분류하는 데 효과적이라고 가정해봅시다. 이 모델을 그대로 주차장의 차량 분류에 사용하려고 하면 성능이 떨어질 수 있습니다. 왜냐하면 도로와 주차장은 다른 환경적 특성과 조명, 배경 등을 가지고 있기 때문입니다. 도메인 적응은 이러한 문제를 해결하기 위해 도로에서 학습된 모델을 주차장 환경에 적응시키는 방법을 제공합니다.

도메인 적응은 주로 라벨이 있는 소스 도메인과 라벨이 없는 타겟 도메인 사이에서 이루어집니다. 이 기술은 소스 도메인에서 학습된 지식을 타겟 도메인에 전이하여, 타겟 도메인에서의 성능을 향상시키는 데 사용됩니다.

이러한 도메인 적응은 다양한 알고리즘과 방법론을 통해 이루어지며, 심층 학습(Deep Learning)을 활용한 심층 도메인 적응(Deep Domain Adaptation) 등도 연구되고 있습니다.

## Domain Adaptation의 문제 정의

Machine learning, Deep learning에서 항상 문제가 되는 것은 무엇일까요? 단순하게 말하면 학습 데이터에서는 잘 동작하는데 테스트 데이터에서는 잘 동작하지 않는 문제가 아마 가장 많을 것 같습니다. 원인은 학습 데이터가 다양하기 때문이기도 하고 model의 over-fitting 때문일 수도 있습니다. 또 하나 다른 이유는 학습 데이터와 테스트 데이터의 Domain-shift가 일어났기 때문입니다.

Domain Shift는 학습 데이터 (Source)와 테스트 데이터 (Target) 의 Distribution의 차이를 의미합니다. 예를 들면 같은 컵을 카메라로 찍었을 때와 캐리커쳐처럼 손으로 그렸을 때의 차이입니다. 물론 Shift가 작은 경우엔 고화질의 DSLR과 Webcam의 이미지도 Domain shift로 볼 수도 있겠습니다. 이 Domain shift가 심할수록 test data의 정확도는 떨어지게 됩니다.

* [출처 - [study] DA(Domain Adaptation)알아보기 기본편](https://lhw0772.medium.com/study-da-domain-adaptation-%EC%95%8C%EC%95%84%EB%B3%B4%EA%B8%B0-%EA%B8%B0%EB%B3%B8%ED%8E%B8-4af4ab63f871)

![Domain Shift (ECCV 2020 Domain Adaptation for Visual Applications Tutorial part 1, 8 page)](./img/domain_shift.png)


### 참고자료
[[study] DA(Domain Adaptation)알아보기 기본편](https://lhw0772.medium.com/study-da-domain-adaptation-%EC%95%8C%EC%95%84%EB%B3%B4%EA%B8%B0-%EA%B8%B0%EB%B3%B8%ED%8E%B8-4af4ab63f871)