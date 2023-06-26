# VGGNet

---

VGGNet은 Visual Geometry Group에서 개발한 딥러닝 모델로, 2014년 ImageNet ILSVRC (ImageNet Large Scale Visual Recognition Challenge)에서 두 번째로 높은 성능을 보였습니다. 이 모델은 그 전에 성공적이었던 **AlexNet의 아이디어를 발전시킨 것**으로, 더 깊은 네트워크를 만들기 위해 **더 작은 크기의 필터를 사용**하는 것이 핵심 아이디어였습니다.

## 연구 배경

---

1. **깊이의 중요성**: VGGNet의 연구자들은 네트워크의 깊이가 성능에 어떤 영향을 미치는지에 대해 깊이 이해하려고 했습니다. 이를 위해, 그들은 3x3 크기의 작은 필터를 사용하는 합성곱 계층을 여러 개 쌓아서 네트워크를 구성하였습니다. 이러한 방식은 네트워크의 깊이를 증가시키면서도 파라미터의 수를 효과적으로 관리할 수 있게 하였습니다.

2. **작은 필터의 사용**: VGGNet은 3x3 크기의 작은 필터를 사용하는 합성곱 계층을 여러 개 쌓는 방식을 사용하였습니다. 이는 AlexNet에서 사용된 11x11 크기의 필터보다 훨씬 작은 크기입니다. 이러한 작은 필터를 사용하는 것은 두 가지 이점이 있습니다. 첫째, 더 작은 필터를 사용하면 더 깊은 네트워크를 만들 수 있습니다. 둘째, 더 작은 필터를 사용하면 더 적은 수의 파라미터를 사용하여 같은 수용 필드(receptive field)를 커버할 수 있습니다.

## 개선 원리

---

**네트워크 깊이의 증가**: VGGNet은 네트워크의 깊이를 증가시키는 것이 성능 향상에 중요하다는 것을 보여주었습니다. 이는 네트워크가 더 복잡한 특징을 학습할 수 있게 하였습니다.

**작은 필터의 사용**: VGGNet은 3x3 크기의 작은 필터를 사용하여 네트워크를 구성하였습니다. 이는 네트워크의 깊이를 증가시키면서도 파라미터의 수를 효과적으로 관리할 수 있게 하였습니다.

## AlexNet

---

AlexNet은 2012년에 발표된 딥러닝 모델로, ImageNet ILSVRC (ImageNet Large Scale Visual Recognition Challenge)에서 우승하며 딥러닝의 가능성을 세계에 알린 모델입니다. AlexNet은 **5개의 합성곱 계층과 3개의 완전 연결 계층**으로 구성되어 있으며, ReLU(Rectified Linear Unit) 활성화 함수, 드롭아웃, 데이터 증강 등의 기법을 사용하여 성능을 향상시켰습니다.

VGGNet은 AlexNet의 아이디어를 발전시킨 모델입니다. AlexNet이 5개의 합성곱 계층을 사용했다면, VGGNet은 이를 확장하여 **16개나 19개의 합성곱 계층을 사용**하는 등 네트워크의 깊이를 늘렸습니다. 이를 통해 더 복잡한 특징을 학습할 수 있게 되었습니다.

또한, AlexNet이 11x11, 5x5, 3x3 등 다양한 크기의 필터를 사용했다면, **VGGNet은 3x3 크기의 작은 필터만을 사용**하였습니다. 이 작은 필터를 여러 개 쌓는 것이 큰 필터를 사용하는 것과 동일한 수용 필드(receptive field)를 가질 수 있지만, 더 적은 파라미터를 사용하고 더 깊은 네트워크를 만들 수 있다는 장점이 있습니다.

이런 아이디어를 발전시킬 수 있는 모델로는 ResNet이 있습니다. ResNet은 VGGNet처럼 3x3 크기의 작은 필터를 사용하지만, 네트워크의 깊이를 더욱 늘릴 수 있도록 "잔차 연결(residual connection)"이라는 기법을 도입하였습니다. 이를 통해 100개 이상의 계층을 가진 네트워크를 효과적으로 학습할 수 있게 되었습니다.

## 활용분야

1. COVID-19 탐지: VGGNet은 COVID-19 탐지를 위해 사용되었습니다. 이 모델은 X-Ray와 CT-Scan 이미지를 처리하는 데 있어 깊이 학습 모델의 성능을 평가하는 데 사용되었습니다. 이 연구에서는 VGGNet 외에도 GoogleNet, ResNet 등의 모델이 사용되었으며, 이들 모델의 성능을 비교하였습니다.

> 참조 [A survey on deep learning models for detection of COVID-19](https://link.springer.com/article/10.1007/s00521-023-08683-x)

2. 골다공증 진단: VGGNet은 골다공증 진단에도 사용되었습니다. 이 모델은 X-ray 이미지를 분류하여 정상, 골다공증 초기 단계, 골다공증 진행 단계를 구분하는 데 사용되었습니다. 이 연구에서는 AlexNet, ResNet 등의 다른 모델과 VGGNet의 성능을 비교하였습니다.

> 참조 [Osteoporosis diagnosis in knee X-rays by transfer learning based on convolution neural network](https://link.springer.com/article/10.1007/s11042-022-13911-y)

3. 반도체 결함 탐지: VGGNet은 반도체 제조 공정에서 결함을 탐지하는 데 사용되었습니다. 이 모델은 SEM (Scanning Electron Microscopy) 이미지에서 다양한 결함 패턴을 분류하고 탐지하는 데 사용되었습니다. 이 연구에서는 VGGNet 외에도 ResNet 등의 다른 모델이 사용되었습니다.

> 참조 [Deep learning-based defect classification and detection in SEM images](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/PC12053/2622550/Deep-learning-based-defect-classification-and-detection-in-SEM-images/10.1117/12.2622550.short?SSO=1)

4. 언어 인식: VGGNet은 언어 인식에도 사용되었습니다. 이 모델은 음성 신호를 이용하여 언어를 인식하는 데 사용되었습니다. 이 연구에서는 VGGNet 외에도 AlexNet, ResNet 등의 다른 모델이 사용되었습니다.

> 참조 [Spoken Language Recognization Based on Features and Classification Methods](https://ijsrcseit.com/CSEIT22839)
