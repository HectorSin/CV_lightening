# 간단한 소개

본 연구에서는 인코더-디코더 구조를 기반으로 한 시퀀스 변환 모델인 Transformer를 제안합니다. 이 모델은 전통적인 순환 및 합성곱 계층을 대체하여, 다중 헤드 셀프-어텐션(자기 주의)를 사용합니다. Transformer는 더욱 빠르게 학습되며, WMT 2014 영어-독일어 및 영어-프랑스어 번역 작업에서 최첨단 성능을 달성하였습니다. 또한, 구문 구조분석 및 반지도 학습에서도 높은 성능을 보여주어 이 모델이 다양한 작업에 적용될 수 있음을 확인할 수 있습니다. 이 프로젝트는 향후 모델 확장성과 다른 입력 및 출력 데이터 타입의 적용에 관심을 가지고 진행될 예정입니다.

우선 Transformer는, 쉽고 간단히 말해서, 그냥 번역기입니다. A언어로 쓰인 어떤 문장이 Transformer로 입력되면, 여러 개의 인코더 블록과 디코더 블록(논문에서는 각각 6개)을 통과하여 B언어로 쓰인 것 같은 의미의 문장으로 출력되는 것입니다.

![Transformer 개요](./img/tm1-3.gif)

# 모델

## 모델 정리

# 훈련

# 결과

# 결론

# 참고자료

1. [Transformer는 이렇게 말했다, "Attention is all you need."](https://blog.promedius.ai/transformer/)
2. [Attention Is All You Need(transformer) paper 정리](https://omicro03.medium.com/attention-is-all-you-need-transformer-paper-%EC%A0%95%EB%A6%AC-83066192d9ab)
3. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
4. [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
