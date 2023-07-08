# CBAM: Convolutional Block Attention Module [논문 번역]

[논문 읽으러 가기](https://arxiv.org/abs/1807.06521)

## Abstract

We propose Convolutional Block Attention Module (CBAM),
a simple yet effective attention module for feed-forward convolutional
neural networks. Given an intermediate feature map, our module sequentially infers attention maps along two separate dimensions, channel
and spatial, then the attention maps are multiplied to the input feature
map for adaptive feature refinement. Because CBAM is a lightweight and
general module, it can be integrated into any CNN architectures seamlessly with negligible overheads and is end-to-end trainable along with
base CNNs. We validate our CBAM through extensive experiments on
ImageNet-1K, MS COCO detection, and VOC 2007 detection datasets.
Our experiments show consistent improvements in classification and detection performances with various models, demonstrating the wide applicability of CBAM. The code and models will be publicly available.

Keywords: Object recognition, attention mechanism, gated convolution

우리는 피드포워드 합성곱 신경망에 대한 간단하면서도 효과적인 주의 모듈인 Convolutional Block Attention Module(CBAM)을 제안합니다. 중간 특징 맵이 주어지면, 우리의 모듈은 채널과 공간, 두 가지 별도의 차원을 따라 차례로 주의 맵을 추론하고, 그런 다음 주의 맵이 입력 특징 맵에 곱해져 **적응형 특징 개선**이 이루어집니다. CBAM은 가벼우며 일반적인 모듈이므로, 거의 추가 비용 없이 어떤 CNN 아키텍처에서도 원활하게 통합될 수 있으며 기본 CNN과 함께 끝-끝으로 훈련 가능합니다. 우리는 ImageNet-1K, MS COCO검출, VOC2007 검출 데이터셋에서의 광범위한 실험을 통해 CBAM을 검증합니다. 우리의 실험은 다양한 모델에서의 분류와 검출 성능의 일관된 개선을 보여줌으로써 CBAM의 넓은 적용 가능성을 입증합니다. 코드와 모델은 공개적으로 이용 가능할 것입니다.

## 1. Introduction

Convolutional neural networks (CNNs) have significantly pushed the performance of vision tasks [1,2,3] based on their rich representation power. To enhance performance of CNNs, recent researches have mainly investigated three important factors of networks: depth, width, and cardinality.

합성곱 신경망 (CNNs)은 그들의 풍부한 표현력을 바탕으로 시각 작업의 성능을 크게 향상시켰습니다. CNNs의 성능을 향상시키기 위해, 최근의 연구들은 네트워크의 세 가지 중요한 요소를 주로 조사하였습니다: **깊이**, **너비**, 그리고 **카디널리티**.

From the LeNet architecture [4] to Residual-style Networks [5,6,7,8] so far, the network has become deeper for rich representation. VGGNet [9] shows that stacking blocks with the same shape gives fair results. Following the same spirit, ResNet [5] stacks the same topology of residual blocks along with skip connection to build an extremely deep architecture. GoogLeNet [10] shows that width is another important factor to improve the performance of a model. Zagoruyko and Komodakis [6] propose to increase the width of a network based on the ResNet architecture. They have shown that a 28-layer ResNet with increased width can outperform an extremely deep ResNet with 1001 layers on the CIFAR benchmarks. Xception [11] and ResNeXt [7] come up with to increase the cardinality of a network. They empirically show that cardinality not only saves the total number of parameters but also results in stronger representation power than the other two factors: depth and width.

LeNet 아키텍처 부터 지금까지의 잔여 스타일 네트워크 까지, 네트워크는 풍부한 표현력을 위해 점점 깊어졌습니다. VGGNet은 같은 형태의 블록을 쌓는 것이 공정한 결과를 가져온다는 것을 보여줍니다. 같은 정신을 이어, ResNet은 스킵 연결을 동반한 동일한 토폴로지의 잔차 블록을 쌓아 매우 깊은 아키텍처를 구축합니다. GoogLeNet은 너비가 모델의 성능을 향상시키는 또 다른 중요한 요소임을 보여줍니다. Zagoruyko와 Komodakis는 ResNet 아키텍처를 기반으로 네트워크의 너비를 늘리는 것을 제안합니다. 그들은 너비를 늘린 28 계층의 ResNet이 CIFAR 벤치마크에서 1001 계층의 매우 깊은 ResNet을 능가할 수 있다는 것을 보여주었습니다. Xception과 ResNeXt는 네트워크의 카디널리티를 늘리는 것을 제안합니다. 그들은 카디널리티가 매개변수의 전체 수를 절약하는 것뿐만 아니라 깊이와 너비와 같은 다른 두 요소보다 더 강한 표현력을 가져다 준다는 것을 실증적으로 보여줍니다.

Apart from these factors, we investigate a different aspect of the architecture design, attention. The significance of attention has been studied extensively in the previous literature [12,13,14,15,16,17]. Attention not only tells where to focus, it also improves the representation of interests. Our goal is to increase representation power by using attention mechanism: focusing on important features and suppressing unnecessary ones. In this paper, we propose a new network module, named “Convolutional Block Attention Module”. Since convolution operations extract informative features by blending cross-channel and spatial information together, we adopt our module to emphasize meaningful features along those two principal dimensions: channel and spatial axes. To achieve this, we sequentially apply channel and spatial attention modules (as shown in Fig. 1), so that each of the branches can learn ‘what’ and ‘where’ to attend in the channel and spatial axes respectively. As a result, our module efficiently helps the information flow within the network by learning which information to emphasize or suppress.

In the ImageNet-1K dataset, we obtain accuracy improvement from various baseline networks by plugging our tiny module, revealing the efficacy of CBAM. We visualize trained models using the grad-CAM [18] and observe that CBAMenhanced networks focus on target objects more properly than their baseline networks. We then conduct user study to quantitatively evaluate improvements in interpretability of models. We show that better performance and better interpretability are possible at the same time by using CBAM. Taking this into account, we conjecture that the performance boost comes from accurate attention and noise reduction of irrelevant clutters. Finally, we validate performance improvement of object detection on the MS COCO and the VOC 2007 datasets, demonstrating a wide applicability of CBAM. Since we have carefully designed our module to be light-weight, the overhead of parameters and computation is negligible in most cases.

**Contribution**. Our main contribution is three-fold.

1. We propose a simple yet effective attention module (CBAM) that can be widely applied to boost representation power of CNNs.
2. We validate the effectiveness of our attention module through extensive ablation studies.
3. We verify that performance of various networks is greatly improved on the multiple benchmarks (ImageNet-1K, MS COCO, and VOC 2007) by plugging our light-weight module.

![CBAM overview](./img/CBAM_overview.png)

## 2. Related Work

Network engineering. “Network engineering” has been one of the most important vision research, because well-designed networks ensure remarkable performance improvement in various applications. A wide range of architectures has been proposed since the successful implementation of a large-scale CNN [19]. An intuitive and simple way of extension is to increase the depth of neural networks [9]. Szegedy et al. [10] introduce a deep Inception network using a multi-branch architecture where each branch is customized carefully. While a naive increase in depth comes to saturation due to the difficulty of gradient propagation, ResNet [5] proposes a simple identity skip-connection to ease the optimization issues of deep networks. Based on the ResNet architecture, various models such as WideResNet [6], Inception-ResNet [8], and ResNeXt [7] have been developed. WideResNet [6] proposes a residual network with a larger number of convolutional filters and reduced depth. PyramidNet [20] is a strict generalization of WideResNet where the width of the network gradually increases. ResNeXt [7] suggests to use grouped convolutions and shows that increasing the cardinality leads to better classification accuracy. More recently, Huang et al. [21] propose a new architecture, DenseNet. It iteratively concatenates the input features with the output features, enabling each convolution block to receive raw information from all the previous blocks. While most of recent network engineering methods mainly target on three factors depth [19,9,10,5], width [10,22,6,8], and cardinality [7,11], we focus on the other aspect, ‘attention’, one of the curious facets of a human visual system.

Attention mechanism. It is well known that attention plays an important role in human perception [23,24,25]. One important property of a human visual system is that one does not attempt to process a whole scene at once. Instead, humans exploit a sequence of partial glimpses and selectively focus on salient parts in order to capture visual structure better [26].

Recently, there have been several attempts [27,28] to incorporate attention processing to improve the performance of CNNs in large-scale classification tasks. Wang et al. [27] propose Residual Attention Network which uses an encoderdecoder style attention module. By refining the feature maps, the network not only performs well but is also robust to noisy inputs. Instead of directly computing the 3d attention map, we decompose the process that learns channel attention and spatial attention separately. The separate attention generation process for 3D feature map has much less computational and parameter overhead, and therefore can be used as a plug-and-play module for pre-existing base
CNN architectures.

More close to our work, Hu et al. [28] introduce a compact module to exploit the inter-channel relationship. In their Squeeze-and-Excitation module, they use global average-pooled features to compute channel-wise attention. However, we
show that those are suboptimal features in order to infer fine channel attention, and we suggest to use max-pooled features as well. They also miss the spatial attention, which plays an important role in deciding ‘where’ to focus as shown in [29]. In our CBAM, we exploit both spatial and channel-wise attention based on an efficient architecture and empirically verify that exploiting both is superior to using only the channel-wise attention as [28]. Moreover, we empirically show that our module is effective in detection tasks (MS-COCO and VOC). Especially, we achieve state-of-the-art performance just by placing our module on top of the existing one-shot detector [30] in the VOC2007 test set.

Concurrently, BAM [31] takes a similar approach, decomposing 3D attention map inference into channel and spatial. They place BAM module at every bottleneck of the network while we plug at every convolutional block.

## 3. Convolutional Block Attention Module

Given an intermediate feature map F ∈ R C×H×W as input, CBAM sequentially infers a 1D channel attention map Mc ∈ R C×1×1 and a 2D spatial attention map Ms ∈ R 1×H×W as illustrated in Fig. 1. The overall attention process can be summarized as:

where ⊗ denotes element-wise multiplication. During multiplication, the attention values are broadcasted (copied) accordingly: channel attention values are broadcasted along the spatial dimension, and vice versa. F ′′ is the final refined output. Fig. 2 depicts the computation process of each attention map. The following describes the details of each attention module.

![Convolution_block](./img/Convolution_block.png)

Channel attention module. We produce a channel attention map by exploiting the inter-channel relationship of features. As each channel of a feature map is considered as a feature detector [32], channel attention focuses on ‘what’ is meaningful given an input image. To compute the channel attention efficiently, we squeeze the spatial dimension of the input feature map. For aggregating spatial information, average-pooling has been commonly adopted so far. Zhou et al

![Attention Module](./img/Attention_model.png)

33] suggest to use it to learn the extent of the target object effectively and Hu et al. [28] adopt it in their attention module to compute spatial statistics. Beyond the previous works, we argue that max-pooling gathers another important clue about distinctive object features to infer finer channel-wise attention. Thus, we use both average-pooled and max-pooled features simultaneously. We empirically confirmed that exploiting both features greatly improves representation power of networks rather than using each independently (see Sec. 4.1), showing the effectiveness of our design choice. We describe the detailed operation below. We first aggregate spatial information of a feature map by using both averagepooling and max-pooling operations, generating two different spatial context descriptors: F c avg and F c max, which denote average-pooled features and max-pooled features respectively. Both descriptors are then forwarded to a shared network to produce our channel attention map Mc ∈ R C×1×1 . The shared network is composed of multi-layer perceptron (MLP) with one hidden layer. To reduce parameter overhead, the hidden activation size is set to R C/r×1×1, where r is the reduction ratio. After the shared network is applied to each descriptor, we merge the output feature vectors using element-wise summation. In short, the channel attention is computed as:

![Formula](./img/formula.png)

where σ denotes the sigmoid function, W0 ∈ R C/r×C , and W1 ∈ R C×C/r. Note that the MLP weights, W0 and W1, are shared for both inputs and the ReLU activation function is followed by W0.

Spatial attention module. We generate a spatial attention map by utilizing the inter-spatial relationship of features. Different from the channel attention, the spatial attention focuses on ‘where’ is an informative part, which is complementary to the channel attention. To compute the spatial attention, we first apply average-pooling and max-pooling operations along the channel axis and concatenate them to generate an efficient feature descriptor. Applying pooling operations along the channel axis is shown to be effective in highlighting informative regions [34]. On the concatenated feature descriptor, we apply a convolution layer to generate a spatial attention map Ms(F) ∈ RH×W which encodes where to emphasize or suppress. We describe the detailed operation below.

We aggregate channel information of a feature map by using two pooling operations, generating two 2D maps: F s avg ∈ R 1×H×W and F s max ∈ R 1×H×W . Each denotes average-pooled features and max-pooled features across the channel. Those are then concatenated and convolved by a standard convolution layer, producing our 2D spatial attention map. In short, the spatial attention is computed a

![maxpool](./img/maxpool.png)

where σ denotes the sigmoid function and f 7×7 represents a convolution operation with the filter size of 7 × 7.

Arrangement of attention modules. Given an input image, two attention modules, channel and spatial, compute complementary attention, focusing on ‘what’ and ‘where’ respectively. Considering this, two modules can be placed in a parallel or sequential manner. We found that the sequential arrangement gives a better result than a parallel arrangement. For the arrangement of the sequential process, our experimental result shows that the channel-first order is slightly better than the spatial-first. We will discuss experimental results on network engineering in Sec. 4.1.

## 4. Experiments

## 5. Conclusion

# 논문 정리

# 참고 자료

1. [[논문 읽기] CBAM(2018), Convolutional Block Attention Module](https://deep-learning-study.tistory.com/666)
