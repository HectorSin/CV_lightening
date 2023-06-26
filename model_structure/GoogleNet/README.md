# GoogleNet

---

GoogleNet, 또는 Inception v1은 2014년에 Google에서 개발한 Convolutional Neural Network (CNN) 모델입니다. 이 모델은 2014년 ImageNet Large Scale Visual Recognition Challenge (ILSVRC)에서 우승하였습니다.

GoogleNet의 주요 개선점은 **"Inception" 구조**입니다. 이 구조는 **네트워크의 깊이와 너비를 증가**시키면서도 계산 효율성을 유지하도록 설계되었습니다. Inception 모듈은 여러 크기의 필터를 동시에 사용하여 다양한 수준의 특징을 추출하고, 이를 병렬로 연산한 후 결과를 합칩니다. 이로 인해 모델은 더욱 복잡한 패턴을 학습할 수 있게 되었습니다.

GoogleNet은 이전의 CNN 모델들, 특히 AlexNet과 VGGNet에서 영감을 받았습니다. AlexNet은 CNN의 기본 구조를 제공하였고, VGGNet은 작은 크기의 필터를 사용하여 깊이 있는 네트워크를 구성하는 아이디어를 제공하였습니다. GoogleNet은 이러한 아이디어를 바탕으로 Inception 모듈을 도입하였고, 이를 통해 모델의 깊이와 너비를 크게 확장하였습니다.

또한, GoogleNet은 네트워크 내에서 여러 위치에서 소프트맥스 분류를 수행하는 **'Auxiliary Classifier'**를 도입하였습니다. 이는 네트워크의 깊이가 깊어짐에 따라 발생할 수 있는 그래디언트 소실 문제를 완화하는 데 도움을 주었습니다.

GoogleNet은 이러한 특징들 덕분에 높은 성능을 보이면서도 모델의 크기와 복잡성을 크게 줄일 수 있었습니다. 이는 모바일 기기와 같은 자원이 제한된 환경에서도 사용할 수 있게 하였습니다.
