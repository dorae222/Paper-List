**Title**: FILIP: BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation
**Venue**: ICML 2022

**Reviewer**: HyeongJun Do
**Last Updated**: 22.05.2024
**Refence**
> Paper Link:https://arxiv.org/abs/2201.12086
> Github: https://github.com/salesforce/BLIP?tab=readme-ov-file
---
# BLIP: Bootstrap Language-Image Pre-training

## 1. Introduction

### 1.1 Background

멀티모달 학습은 텍스트와 이미지 데이터를 결합하여 모델이 다양한 태스크를 수행할 수 있도록 하는 기술입니다. CLIP(Contrastive Language-Image Pre-training)는 대규모 텍스트-이미지 쌍을 사용하여 텍스트와 이미지의 상관관계를 학습함으로써 뛰어난 성능을 보였습니다. 그러나 CLIP는 데이터의 양에 크게 의존하며, 데이터 효율성 측면에서 개선이 필요합니다.

### 1.2 Motivation

BLIP(Bootstrap Language-Image Pre-training)는 데이터 효율성을 극대화하고, 텍스트-이미지 쌍을 스스로 생성하여 모델의 성능을 개선하는 것을 목표로 합니다. BLIP는 부트스트래핑 방법을 도입하여 초기 학습 데이터에 의존하지 않고, 점진적으로 더 나은 텍스트-이미지 쌍을 생성하여 학습합니다.

## 2. Methodology

### 2.1 Architecture

BLIP의 아키텍처는 두 가지 주요 컴포넌트로 구성됩니다:

1. **Vision Encoder**: 이미지를 인코딩하여 고차원 특징 벡터로 변환합니다. 이는 일반적으로 CNN(Convolutional Neural Network) 구조를 사용하여 구현됩니다.
2. **Language Encoder**: 텍스트를 인코딩하여 고차원 특징 벡터로 변환합니다. Transformer 기반의 언어 모델을 사용하여 구현됩니다.

**Figure 2: BLIP Architecture**
![[figure2.png]]
### 2.2 Bootstrap Mechanism

BLIP는 학습 과정에서 부트스트래핑을 사용하여 스스로 텍스트-이미지 쌍을 생성합니다. 이는 다음과 같은 단계를 포함합니다:

1. **Initial Training**: 초기 학습 단계에서 기존의 텍스트-이미지 쌍을 사용하여 모델을 학습합니다.
2. **Bootstrap Data Generation**: 학습된 모델을 사용하여 새로운 텍스트-이미지 쌍을 생성합니다. 이 과정에서는 이미지에 대해 텍스트 설명을 생성하거나, 텍스트에 맞는 이미지를 생성합니다.
3. **Iterative Training**: 생성된 텍스트-이미지 쌍을 사용하여 모델을 다시 학습합니다. 이 과정을 반복함으로써 모델은 점차 더 나은 텍스트-이미지 쌍을 생성하고 학습합니다.

### 2.3 CapFilt Mechanism

BLIP는 CapFilt(Captioning and Filtering) 방법을 통해 웹 이미지에 대한 캡션을 생성하고, 노이즈가 있는 캡션을 제거합니다.

1. **Captioner (Cap)**: 웹 이미지에 대해 캡션을 생성합니다.
2. **Filter (Filt)**: 생성된 캡션 중 노이즈가 있는 캡션을 제거합니다.

**Figure 1: CapFilt Mechanism**
![[source/BLIP/figure1.png]]
### 2.4 Loss Function

BLIP는 다음과 같은 손실 함수를 사용하여 학습합니다:

- **Contrastive Loss (ITC)**: 텍스트와 이미지 쌍의 일치성을 학습하는 손실 함수입니다. 텍스트와 이미지가 일치할 때 낮은 값을 가지며, 일치하지 않을 때 높은 값을 가집니다.
- **Image-Text Matching Loss (ITM)**: 이미지와 텍스트가 매칭되는지 여부를 예측하는 손실 함수입니다.
- **Language Modeling Loss (LM)**: 텍스트 생성의 정확성을 학습하는 손실 함수입니다. 이미지로부터 텍스트를 생성할 때, 생성된 텍스트가 실제 텍스트와 얼마나 일치하는지를 측정합니다.

**Figure 3: Learning framework of BLIP**
![[figure3.png]]
## 3. Experiments

### 3.1 Datasets

BLIP는 다양한 대규모 데이터셋을 사용하여 평가되었습니다. 대표적인 데이터셋으로는 다음과 같습니다:

- **COCO (Common Objects in Context)**: 일상 생활에서 접할 수 있는 다양한 객체들이 포함된 데이터셋입니다.
- **Flickr30k**: 사람들이 일상에서 찍은 사진과 그에 대한 설명이 포함된 데이터셋입니다.

### 3.2 Baselines

BLIP는 다음과 같은 기존 모델들과 비교 평가되었습니다:

- **CLIP**: 대규모 텍스트-이미지 쌍을 사용하여 학습한 모델입니다.
- **ViLBERT**: 이미지와 텍스트의 상관관계를 학습하는 멀티모달 모델입니다.
- **UNITER**: 텍스트와 이미지의 결합 표현을 학습하는 모델입니다.

### 3.3 Results

BLIP는 이미지 캡셔닝, 텍스트-이미지 검색 등 다양한 멀티모달 태스크에서 기존 모델들보다 우수한 성능을 보였습니다. 특히, 데이터 효율성 측면에서 적은 양의 데이터로도 높은 성능을 발휘하였습니다.

**Table 5: Comparison with state-of-the-art image-text retrieval methods** 
![[table5.png]]
**시사점:**

- **BLIP의 성능 우수성**: 이 테이블은 COCO와 Flickr30k 데이터셋에서 이미지-텍스트 검색 태스크에 대해 BLIP가 최신 모델들과 비교하여 뛰어난 성능을 보임을 보여줍니다. 특히, Recall@1, Recall@5, Recall@10과 같은 지표에서 BLIP는 대부분의 경우에서 최고 성능을 기록했습니다.
- **대규모 데이터 학습의 효과**: BLIP는 129M 이미지 데이터로 학습되었으며, 이는 데이터의 양이 성능 향상에 중요한 역할을 한다는 것을 시사합니다.
- **CapFilt-L의 효과**: CapFilt-L을 사용한 BLIP 모델이 더 높은 성능을 기록함으로써, 부트스트래핑된 데이터셋을 사용하는 것이 성능 향상에 효과적임을 보여줍니다.

**Table 6: Zero-shot image-text retrieval results on Flickr30K** Flickr30K
![[table6.png]]
**시사점:**

- **Zero-shot 설정에서의 강력한 성능**: BLIP는 zero-shot 설정에서 Flickr30K 데이터셋을 사용한 이미지-텍스트 검색에서 매우 높은 성능을 보여줍니다. 이는 BLIP가 사전 학습된 데이터 없이도 새로운 데이터셋에 잘 일반화된다는 것을 의미합니다.
- **다양한 모델 간의 비교**: BLIP는 기존의 CLIP, ALIGN, ALBEF 모델들과 비교하여 더 나은 성능을 보였습니다. 특히, Recall@1, Recall@5, Recall@10에서의 성능이 이를 입증합니다.

**Table 7: Comparison with state-of-the-art image captioning methods on NoCaps and COCO Caption** 
![[table7.png]]
**시사점:**

- **이미지 캡셔닝 성능**: BLIP는 NoCaps와 COCO Caption 데이터셋에서 이미지 캡셔닝 태스크에 대해 높은 성능을 보였습니다. 이는 CIDEr, SPICE, BLEU@4 지표에서 두드러집니다.
- **CapFilt-L의 효용성**: CapFilt-L을 적용한 BLIP 모델이 성능 향상에 기여한 것을 볼 수 있습니다. 이는 데이터 부트스트래핑이 이미지 캡셔닝 성능에 긍정적인 영향을 미친다는 것을 시사합니다.
- **비교 모델과의 성능 격차**: LEMON_large 및 SimVLM_huge와 같은 최신 모델들과 비교했을 때, BLIP는 더 적은 데이터로도 경쟁력 있는 성능을 보였습니다.

**Table 8: Comparison with State-of-the-Art Methods on VQA and NLVR2**
![[table8.png]]
**시사점:**

- **VQA(Visual Question Answering) 성능**: BLIP는 VQA 태스크에서 뛰어난 성능을 보였습니다. 특히 BLIP는 테스트-데브(test-dev)와 테스트-스탠드(test-std)에서 모두 높은 정확도를 기록하였습니다. 이는 BLIP가 시각적 질문 응답 문제에서 매우 강력한 모델임을 시사합니다.
- **NLVR2(Natural Language Visual Reasoning for Real) 성능**: NLVR2 태스크에서도 BLIP는 매우 높은 성능을 보여줍니다. 테스트-데브(dev)와 테스트-P(test-P)에서 BLIP의 성능은 다른 최신 모델들과 비교하여 우수함을 입증하였습니다. 이는 BLIP가 복잡한 시각적 추론 작업에서도 탁월한 성능을 발휘한다는 것을 의미합니다.
- **전반적인 성능 우수성**: BLIP는 다양한 멀티모달 태스크에서 일관되게 우수한 성능을 보여주고 있으며, 특히 VQA와 NLVR2와 같은 고난이도의 태스크에서 두드러진 성과를 보이고 있습니다.

## 4. Analysis

### 4.1 Data Efficiency

BLIP는 기존의 모델들에 비해 더 적은 양의 데이터로도 높은 성능을 발휘할 수 있음을 보였습니다. 이는 부트스트래핑 방법이 효과적임을 입증합니다. BLIP는 초기 데이터에 크게 의존하지 않고도 점진적으로 성능을 향상시킬 수 있습니다.

### 4.2 Model Robustness

BLIP는 다양한 도메인과 태스크에서 일관된 성능을 보이며, 모델의 견고성을 입증하였습니다. 이는 BLIP의 부트스트래핑 방법이 멀티모달 학습에 유효함을 나타냅니다. 다양한 데이터 분포와 환경에서 BLIP는 높은 적응력을 보입니다.

## 5. Conclusion

BLIP는 멀티모달 학습에서 텍스트와 이미지의 상관관계를 효과적으로 학습하는 혁신적인 모델입니다. 부트스트래핑을 통해 데이터 효율성을 높이고, 다양한 태스크에서 우수한 성능을 보였습니다. 이는 멀티모달 학습의 새로운 방향을 제시하며, 향후 연구와 응용에 중요한 기여를 할 것으로 기대됩니다.