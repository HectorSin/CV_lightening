# 2023년 CVPR 주요 논문 분석 [상 받을 후보]

CVPR 2023에서 상을 받을 후보로 제공된 논문을 바탕으로 컴퓨터 비전 연구에서의 추세를 몇가지 식별할 수 있습니다.

## 1. 신경-기호적 접근법(Neuro-symbolic approaches): 'Visual Programming: Compositional Visual Reasoning Without Training'

해당 논문에서는 VISPROG이라는 신경-기호적 접근법을 소개하고 있습니다.

이 방법은 대형 언어 모델을 사용하여 복잡한 시각적 작업을 해결하기 위한 파이썬과 같은 모듈식 프로그램을 생성합니다.

이 추세는 **신경 네트워크와 기호적 추론을 결합**하여 시각적 추론 능력을 향상시키는 탐색을 제안합니다.

![Visual_Programming](./img/visual_programming.png)

[논문 보기](https://openaccess.thecvf.com/content/CVPR2023/html/Gupta_Visual_Programming_Compositional_Visual_Reasoning_Without_Training_CVPR_2023_paper.html)

[Visual Programming: Compositional Visual Reasoning Without Training](./Neuro-symbolic%20approaches/Visual%20Programming)

##### Abstract

We present VISPROG, a neuro-symbolic approach to solving complex and compositional visual tasks given natural language instructions. VISPROG avoids the need for any task-specific training. Instead, it uses the in-context learning ability of large language models to generate python-like modular programs, which are then executed to get both the solution and a comprehensive and interpretable rationale. Each line of the generated program may invoke one of several off-the-shelf computer vision models, image processing routines, or python functions to produce intermediate outputs that may be consumed by subsequent parts of the program. We demonstrate the flexibility of VISPROG on 4 diverse tasks - compositional visual question answering, zero-shot reasoning on image pairs, factual knowledge object tagging, and language-guided image editing. We believe neuro-symbolic approaches like VISPROG are an exciting avenue to easily and effectively expand the scope of AI systems to serve the long tail of complex tasks that people may wish to perform.

##### 초록

우리는 자연어 지시사항에 따른 복잡하고 구성적인 시각적 작업을 해결하기 위한 신경 기호적 접근법인 VISPROG를 제시합니다. VISPROG는 어떠한 작업 특성 훈련의 필요성을 피합니다. 대신에, 이건은 대형 언어 모델의 컨텍스트 내 학습 능력을 사용하여 파이썬 같은 모듈형 프로그램을 생성하고, 그것들을 실행하여 해결책과 함께 포괄적이고 해석 가능한 근거를 제공합니다. 생성된 프로그램의 각 라인은 여러 준비된 컴퓨터 비전 모델, 이미지 처리 루틴, 또는 파이썬 함수 중 하나를 호출하여 프로그램의 후속 부분에서 소비될 수 있는 중간 출력물을 생성할 수 있습니다. 우리는 구성적 시각 질문 응답, 이미지 쌍에 대한 제로샷 추론, 사실적 지식 객체 태깅, 언어 가이드 이미지 편집 - 4가지 다양한 작업에 대해 VISPROG의 유연성을 보여줍니다. 사람들이 수행하고 싶어할 복잡한 작업들의 긴 꼬리를 쉽고 효과적으로 확장하기 위한 **신경 기호적 접근법인 VISPROG** 같은 방법이 흥미진진한 새로운 방향이라고 우리는 믿습니다.

## 2. 데이터 기반 방법(Data-driven methods): 'Data-Driven Feature Tracking for Event Cameras'

해당 논문에서는 이벤트 카메라를 위한 데이터 기반 특징 추적 방법을 제시하고 있습니다.

이 방법은 저지연 이벤트를 활용하고 합성 데이터에서 실제 데이터로 지식을 전달합니다. 이 추세는 대량의 데이터를 활용하고 이로부터 학습하여 성능과 일반화를 향상시키는 것에 초점을 맞추고 있습니다.

![Data-Driven](./img/Data_Feature_Tracking.png)

[논문 보기](https://openaccess.thecvf.com/content/CVPR2023/html/Messikommer_Data-Driven_Feature_Tracking_for_Event_Cameras_CVPR_2023_paper.html)

[Data-Driven Feature Tracking for Event Cameras](./Data-driven%20methods/Data-Driven%20Feature%20Tracking%20for%20Event%20Cameras)

##### Abstract

Because of their high temporal resolution, increased resilience to motion blur, and very sparse output, event cameras have been shown to be ideal for low-latency and low-bandwidth feature tracking, even in challenging scenarios. Existing feature tracking methods for event cameras are either handcrafted or derived from first principles but require extensive parameter tuning, are sensitive to noise, and do not generalize to different scenarios due to unmodeled effects. To tackle these deficiencies, we introduce the first data-driven feature tracker for event cameras, which leverages low-latency events to track features detected in a grayscale frame. We achieve robust performance via a novel frame attention module, which shares information across feature tracks. By directly transferring zero-shot from synthetic to real data, our data-driven tracker outperforms existing approaches in relative feature age by up to 120% while also achieving the lowest latency. This performance gap is further increased to 130% by adapting our tracker to real data with a novel self-supervision strategy.

##### 초록

고시간 해상도, 모션 블러에 대한 높은 내성, 그리고 매우 희박한 출력 덕분에, 이벤트 카메라들은 도전적인 시나리오에서도 저 지연 및 저 대역폭 피처 추적에 이상적임이 증명되었습니다. 이벤트 카메라를 위한 기존의 피처 추적 방법들은 수작업이거나 첫 원리에서 유래했지만, 방대한 매개변수 조정이 필요하고, 노이즈에 민감하며, 모델링 되지 않은 효과로 인해 다른 시나리오에 일반화하지 못합니다. 이런 결점들을 해결하기 위해, 우리는 **그레이스케일 프레임에서 감지된 피처를 추적하기 위해 저지연 이벤트를 활용하는 이벤트 카메라를 위한 첫 번째 데이터 기반 피처 추적기를 소개**합니다. 우리는 피처 추적 간에 정보를 공유하는 새로운 프레임 주의 모듈을 통해 견고한 성능을 달성합니다. 합성 데이터에서 실제 데이터로 직접적인 제로샷 전송을 통해, 우리의 데이터 기반 추적기는 상대적인 피처 연령에서 최대 120% 까지 성능을 향상시키면서도 가장 낮은 지연성을 달성합니다. 이 성능 격차는 새로운 자기 감독 전략을 통해 우리의 추적기를 실제 데이터에 적용합으로써 130%까지 더욱 증가됩니다.

## 3. 효율적인 생성 모델(Efficient generation models): 'On Distillation of Guided Diffusion Models'와 'MobileNeRF: Exploiting the Polygon Rasterization Pipeline for Efficient Neural Field Rendering on Mobile Architectures'

두 논문은 이미지 생성 모델의 효율성을 다룹니다. 이들은 샘플링 과정을 가속화하고 이러한 모델을 더 실용적이고 접근 가능하게 만들기 위한 증류 방법과 대체 표현을 제안합니다.

![Diffusion_Model](./img/diffusion_model.png)

[논문 보기](https://openaccess.thecvf.com/content/CVPR2023/html/Meng_On_Distillation_of_Guided_Diffusion_Models_CVPR_2023_paper.html)

[On Distillation of Guided Diffusion Models](./Efficient%20generation%20models/On%20Distillation%20of%20Guided%20Diffusion%20Models)

##### Abstract

Classifier-free guided diffusion models have recently been shown to be highly effective at high-resolution image generation, and they have been widely used in large-scale diffusion frameworks including DALL\*E 2, Stable Diffusion and Imagen. However, a downside of classifier-free guided diffusion models is that they are computationally expensive at inference time since they require evaluating two diffusion models, a class-conditional model and an unconditional model, tens to hundreds of times. To deal with this limitation, we propose an approach to distilling classifier-free guided diffusion models into models that are fast to sample from: Given a pre-trained classifier-free guided model, we first learn a single model to match the output of the combined conditional and unconditional models, and then we progressively distill that model to a diffusion model that requires much fewer sampling steps. For standard diffusion models trained on the pixel-space, our approach is able to generate images visually comparable to that of the original model using as few as 4 sampling steps on ImageNet 64x64 and CIFAR-10, achieving FID/IS scores comparable to that of the original model while being up to 256 times faster to sample from. For diffusion models trained on the latent-space (e.g., Stable Diffusion), our approach is able to generate high-fidelity images using as few as 1 to 4 denoising steps, accelerating inference by at least 10-fold compared to existing methods on ImageNet 256x256 and LAION datasets. We further demonstrate the effectiveness of our approach on text-guided image editing and inpainting, where our distilled model is able to generate high-quality results using as few as 2-4 denoising steps.

분류기가 없는 가이드된 확산 모델은 최근 고해상도 이미지 생성에 매우 효과적임이 입증되었으며, DALL*E 2, Stable Diffusion 및 Imagen을 포함한 대규모 확산 프레임워크에서 널리 사용되고 있습니다. 그러나 분류기가 없는 가이드된 확산 모델의 단점은 추론 시간에 계산 비용이 많이 든다는 것입니다. 이는 클래스 조건부 모델과 무조건부 모델, 두 가지 확산 모델을 수십에서 수백 번 평가해야 하기 때문입니다. 이 제한을 해결하기 위해, 우리는 미리 훈련된 분류기가 없는 가이드 모델이 주어지면, 우리는 먼저 단일 모델을 학습하여 결합된 조건부 및 무조건부 모델의 출력과 일치시키고, 그런 다음 우리는 그 모델을 점진적으로 샘플링 단계가 훨씬 적은 확산 모델로 정제합니다. 픽셀 공간에서 훈련된 표준 확산 모델의 경우, 우리의 접근법은 ImageNet 64x64와 CIFAR-10에서 샘플링 단계가 4단계로 줄어든 원래 모델과 비교할 수 있는 이미지를 생성할 수 있으며, FID/IS 점수는 원래 모델과 비교하여 최대 256배 빠른 샘플링을 가능하게 합니다. 잠재 공간에서 훈련된 확산 모델(예: Stable Diffusion)의 경우, 우리의 접근법은 14단계의 노이즈 제거 단계만으로 고해상도 이미지를 생성할 수 있어, ImageNet 256x256과 LAION 데이터셋에서 기존 방법에 비해 최소 10배 빠른 추론을 가능하게 합니다. 우리는 또한 텍스트 가이드 이미지 편집 및 인페인팅에서 우리의 접근법의 효과를 보여주며, 우리의 정제된 모델은 노이즈 제거 단계가 2~4단계로 줄어든 고품질 결과를 생성할 수 있습니다.

##### 초록



![MobileNeRF](./img/MobileNeRF.png)

[논문 보기](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_MobileNeRF_Exploiting_the_Polygon_Rasterization_Pipeline_for_Efficient_Neural_Field_CVPR_2023_paper.html)

[MobileNeRF: Exploiting the Polygon Rasterization Pipeline for Efficient Neural Field Rendering on Mobile Architectures](./Efficient%20generation%20models/MobileNeRF%20-%20Exploiting%20the%20Polygon%20Rasterization%20Pipeline%20for%20Efficient%20Neural%20Field%20Rendering%20on%20Mobile%20Architectures)

##### Abstract

Neural Radiance Fields (NeRFs) have demonstrated amazing ability to synthesize images of 3D scenes from novel views. However, they rely upon specialized volumetric rendering algorithms based on ray marching that are mismatched to the capabilities of widely deployed graphics hardware. This paper introduces a new NeRF representation based on textured polygons that can synthesize novel images efficiently with standard rendering pipelines. The NeRF is represented as a set of polygons with textures representing binary opacities and feature vectors. Traditional rendering of the polygons with a z-buffer yields an image with features at every pixel, which are interpreted by a small, view-dependent MLP running in a fragment shader to produce a final pixel color. This approach enables NeRFs to be rendered with the traditional polygon rasterization pipeline, which provides massive pixel-level parallelism, achieving interactive frame rates on a wide range of compute platforms, including mobile phones.

##### Abstract

신경 방사장(NeRFs)은 새로운 시점에서 3D 장면의 이미지를 합성하는 놀라운 능력을 보여주었습니다. 그러나 그들은 광선 행진에 기반한 특수한 부피 렌더링 알고리즘에 의존하며, 이는 널리 배포된 그래픽 하드웨어의 기능과는 맞지 않습니다. 이 논문에서는 텍스처가 있는 다각형에 기반한 새로운 NeRF 표현을 소개하며, 이는 표준 렌더링 파이프라인을 통해 효율적으로 새로운 이미지를 합성할 수 있습니다. NeRF는 이진 불투명도와 피처 벡터를 나타내는 텍스처가 있는 다각형의 집합으로 표현됩니다. 다각형의 전통적인 렌더링은 z-버퍼를 사용하며, 이는 모든 픽셀에서 피처를 가진 이미지를 생성하며, 이는 프래그먼트 쉐이더에서 실행되는 작은, 뷰-의존적인 MLP에 의해 해석되어 최종 픽셀 색상을 생성합니다. 이 접근법은 NeRF가 전통적인 다각형 래스터화 파이프라인으로 렌더링될 수 있게 하며, 이는 대량의 픽셀 수준 병렬성을 제공하며, 모바일 폰을 포함한 다양한 컴퓨트 플랫폼에서 대화형 프레임 속도를 달성합니다.

## 4. 문맥 인식 렌더링 및 합성(Context-aware rendering and synthesis): 'DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation'과 'DynlBaR: Neural Dynamic Image-Based Rendering'

해당 논문들은 각각 텍스트 프롬프트에서 이미지의 합성을 향상시키고 비디오에서 새로운 뷰를 생성하는 데 초점을 맞추고 있습니다.

이 논문들은 이미지 합성과 렌더링 작업에서 문맥 정보와 동적 장면 이해의 중요성을 강조합니다.

![DreamBooth](./img/DreamBooth.png)

[논문 보기](https://openaccess.thecvf.com/content/CVPR2023/html/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.html)

##### Abstract

Large text-to-image models achieved a remarkable leap in the evolution of AI, enabling high-quality and diverse synthesis of images from a given text prompt. However, these models lack the ability to mimic the appearance of subjects in a given reference set and synthesize novel renditions of them in different contexts. In this work, we present a new approach for "personalization" of text-to-image diffusion models. Given as input just a few images of a subject, we fine-tune a pretrained text-to-image model such that it learns to bind a unique identifier with that specific subject. Once the subject is embedded in the output domain of the model, the unique identifier can be used to synthesize novel photorealistic images of the subject contextualized in different scenes. By leveraging the semantic prior embedded in the model with a new autogenous class-specific prior preservation loss, our technique enables synthesizing the subject in diverse scenes, poses, views and lighting conditions that do not appear in the reference images. We apply our technique to several previously-unassailable tasks, including subject recontextualization, text-guided view synthesis, and artistic rendering, all while preserving the subject's key features. We also provide a new dataset and evaluation protocol for this new task of subject-driven generation. Project page: https://dreambooth.github.io/

대형 텍스트-이미지 모델은 주어진 텍스트 프롬프트에서 이미지를 고품질로 다양하게 합성하는 능력을 통해 AI의 진화에서 놀라운 도약을 이루었습니다. 그러나 이러한 모델들은 주어진 참조 세트에서 주제의 외형을 모방하고 다른 맥락에서 그들의 새로운 표현을 합성하는 능력이 부족합니다. 이 연구에서는 텍스트-이미지 확산 모델의 "개인화"를 위한 새로운 접근법을 제시합니다. 주제의 몇 가지 이미지만 입력으로 주면, 우리는 사전 훈련된 텍스트-이미지 모델을 미세 조정하여 특정 주제와 고유 식별자를 연결하도록 학습합니다. 주제가 모델의 출력 도메인에 포함되면, 고유 식별자는 다른 장면에서 주제의 새로운 사실적인 이미지를 합성하는 데 사용될 수 있습니다. 새로운 자기 생성적인 클래스 특정 사전 보존 손실과 함께 모델에 포함된 의미론적 사전을 활용함으로써, 우리의 기술은 참조 이미지에 나타나지 않는 다양한 장면, 포즈, 뷰, 조명 조건에서 주제를 합성하는 것을 가능하게 합니다. 우리는 주제 재구성, 텍스트 가이드 뷰 합성, 예술적 렌더링 등 여러 이전에 불가능했던 작업에 우리의 기술을 적용하며, 주제의 핵심 특징을 유지하면서 고품질의 결과를 생성할 수 있습니다. 또한 우리는 주제 중심 생성이라는 새로운 작업을 위한 새로운 데이터셋과 평가 프로토콜을 제공합니다.

## 참고자료

1. [CVPR 2023 — Summary](https://medium.com/@dobko_m/cvpr-2023-summary-ad271d383404)

2. [CVPR](https://cvpr2023.thecvf.com/)
