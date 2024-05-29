**Title**: BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
**Venue**: ICML 2023

**Reviewer**: HyeongJun Do
**Last Updated**: 29.05.2024
**Refence**
> Paper Link: https://arxiv.org/abs/2301.12597
> Github: https://github.com/salesforce/LAVIS/tree/main/projects/blip2
---
## 1. Introduction

### 1.1 Background

멀티모달 학습은 텍스트와 이미지 데이터를 결합하여 모델이 다양한 태스크를 수행할 수 있도록 하는 기술임. 최근 CLIP(Contrastive Language-Image Pre-training)와 같은 모델들은 대규모 텍스트-이미지 쌍을 사용하여 텍스트와 이미지의 상관관계를 학습함으로써 뛰어난 성능을 보였음. 그러나 이러한 모델들은 데이터의 양에 크게 의존하며, 데이터 효율성 측면에서 개선이 필요함. BLIP-2는 이러한 문제를 해결하기 위해 설계된 모델로, 사전 훈련된 이미지 인코더와 대형 언어 모델을 사용하여 효율적이고 강력한 비전-언어 사전 학습을 제공함.

### 1.2 Motivation

BLIP-2의 목표는 사전 훈련된 이미지 인코더와 대형 언어 모델을 활용하여 비전-언어 태스크의 성능을 향상시키는 것임. BLIP-2는 경량화된 Q-Former를 도입하여 두 단계의 사전 학습 전략을 통해 텍스트와 이미지 간의 모달리티 격차를 줄임. 이를 통해 기존 모델들보다 적은 학습 가능한 파라미터로 뛰어난 성능을 달성함. 특히, 모델이 사전 학습 데이터에 의존하지 않고도 스스로 텍스트-이미지 쌍을 생성하여 점진적으로 성능을 향상시킬 수 있다는 점에서 큰 장점을 가짐.

## 2. Methodology

### 2.1 Architecture

BLIP-2의 아키텍처는 세 가지 주요 컴포넌트로 구성됨:

1. **Image Encoder**: 이미지를 인코딩하여 고차원 특징 벡터로 변환함. BLIP-2는 사전 훈련된 CLIP ViT-L/14 또는 EVA-CLIP ViT-g/14 모델을 사용함. 이 모델들은 고품질의 시각적 표현을 제공함.
2. **Q-Former**: 이미지 인코더와 언어 모델 사이에서 정보를 전달하는 경량화된 Transformer 모듈임. 32개의 학습 가능한 쿼리 벡터를 사용하여 이미지 인코더의 시각적 특징을 추출함. Q-Former는 이미지 인코더와 언어 모델 사이에서 정보의 병목 역할을 하여, 가장 중요한 시각 정보를 선택적으로 언어 모델에 전달함.
3. **Language Model**: 대형 언어 모델로, 텍스트를 생성하거나 입력 텍스트를 인코딩함. BLIP-2는 사전 훈련된 OPT 또는 FlanT5 모델을 사용함. 이 모델들은 강력한 언어 생성 능력과 제로샷 전이 능력을 제공함.

**Figure 1: BLIP-2 Architecture**
![[source/BLIP2/figure1.png]]

### 2.2 Bootstrap Mechanism

BLIP-2는 두 단계의 사전 학습 전략을 사용하여 비전-언어 태스크를 학습함:

1. **첫 번째 단계: 비전-언어 표현 학습**: Q-Former를 이미지 인코더에 연결하여 이미지-텍스트 쌍을 사용해 학습함. 이 단계에서는 세 가지 주요 학습 목표가 사용됨:
    
    - **Image-Text Contrastive Learning (ITC)**: 텍스트와 이미지의 쌍을 정렬하여 상호 정보를 최대화하는 학습.
    - **Image-Grounded Text Generation (ITG)**: 이미지를 조건으로 텍스트를 생성하는 학습.
    - **Image-Text Matching (ITM)**: 이미지와 텍스트 쌍이 매칭되는지 여부를 예측하는 이진 분류 학습.
2. **두 번째 단계: 비전-언어 생성 학습**: Q-Former를 언어 모델에 연결하여 이미지-텍스트 생성 작업을 학습함. 이 단계에서는 두 가지 유형의 언어 모델과 함께 학습함:
    
    - **디코더 기반 LLM**: 시각적 정보를 바탕으로 텍스트를 생성하는 디코더 모델 (예: OPT).
    - **인코더-디코더 기반 LLM**: 시각적 정보를 입력 텍스트와 결합하여 텍스트를 생성하는 인코더-디코더 모델 (예: FlanT5).
![[source/BLIP2/figure2.png]]

### 2.3 CapFilt Mechanism

BLIP-2는 CapFilt 방법을 통해 웹 이미지에 대한 캡션을 생성하고, 노이즈가 있는 캡션을 제거함:

1. **Captioner (Cap)**: 웹 이미지에 대해 캡션을 생성함.
2. **Filter (Filt)**: 생성된 캡션 중 노이즈가 있는 캡션을 제거함. 이를 통해 보다 정제된 학습 데이터를 확보할 수 있음.

**CapFilt Mechanism** 
![[source/BLIP/figure3.png]]
### 2.4 Loss Functions

BLIP-2는 다음과 같은 손실 함수를 사용하여 학습함:

- **Image-Text Contrastive Loss (ITC)**: 텍스트와 이미지 쌍의 일치성을 학습함. 텍스트와 이미지가 일치할 때 낮은 값을 가지며, 일치하지 않을 때 높은 값을 가짐.
- **Image-Grounded Text Generation Loss (ITG)**: 이미지를 기반으로 텍스트를 생성하는 손실 함수임. 생성된 텍스트가 실제 텍스트와 얼마나 일치하는지를 측정함.
- **Image-Text Matching Loss (ITM)**: 이미지와 텍스트가 매칭되는지 여부를 예측하는 손실 함수임. 이진 분류 문제로 접근하여 이미지와 텍스트가 매칭되는 경우와 그렇지 않은 경우를 구분함.

## 3. Experiments

### 3.1 Datasets

BLIP-2는 다양한 대규모 데이터셋을 사용하여 평가되었음. 대표적인 데이터셋으로는 다음과 같음:

- **COCO (Common Objects in Context)**: 일상 생활에서 접할 수 있는 다양한 객체들이 포함된 데이터셋임. 주로 이미지 캡셔닝 및 객체 검출에 사용됨.
- **Flickr30k**: 사람들이 일상에서 찍은 사진과 그에 대한 설명이 포함된 데이터셋임. 이미지-텍스트 매칭과 검색에 주로 사용됨.
- **Visual Genome**: 다양한 이미지와 그에 대한 상세한 설명 및 질문 응답 데이터가 포함된 데이터셋임.
- **CC3M (Conceptual Captions 3M)**: 웹에서 수집된 이미지-텍스트 쌍을 포함한 데이터셋임.
- **CC12M (Conceptual Captions 12M)**: CC3M의 확장판으로, 더 많은 이미지-텍스트 쌍을 포함함.
- **SBU Captioned Photo Dataset**: SBU에서 수집된 이미지와 캡션이 포함된 데이터셋임.
- **LAION400M**: 대규모 이미지-텍스트 쌍을 포함한 공개 데이터셋임.

### 3.2 Baselines

BLIP-2는 다음과 같은 기존 모델들과 비교 평가되었음:

- **CLIP**: 대규모 텍스트-이미지 쌍을 사용하여 학습한 모델임. 텍스트와 이미지의 상관관계를 학습하여 강력한 제로샷 성능을 보임.
- **Flamingo**: 대규모 비전-언어 모델로, 다양한 비전-언어 태스크에서 뛰어난 성능을 보임.
- **BEIT-3**: 비전-언어 통합 학습 모델로, 다양한 비전-언어 태스크에서 우수한 성능을 보임.
	- VQA Task SOTA Model(2024.05.29)
- **UNITER**: 텍스트와 이미지의 결합 표현을 학습하는 모델임.
- **SimVLM**: 간단한 비전-언어 모델로, 약한 감독 학습을 사용하여 강력한 성능을 보임.

### 3.3 Results

BLIP-2는 이미지 캡셔닝, 텍스트-이미지 검색, VQA(Visual Question Answering) 등 다양한 멀티모달 태스크에서 기존 모델들보다 우수한 성능을 보였음.

**Table 1: Performance Comparison on Various Zero-Shot Vision-Language Tasks** 
![[source/BLIP2/table1.png]]
**시사점**:

- **BLIP-2의 성능 우수성**: BLIP-2는 다양한 비전-언어 태스크에서 기존 최신 모델들과 비교하여 뛰어난 성능을 보였음. 특히, 적은 수의 학습 가능한 파라미터로도 높은 성능을 달성했음.
- **대규모 데이터의 효율적 활용**: BLIP-2는 129M 이미지 데이터로 학습되었으며, 이는 데이터의 양이 성능 향상에 중요한 역할을 한다는 것을 시사함.
- **다양한 데이터셋에서의 일반화 능력**: BLIP-2는 COCO, Flickr30k, Visual Genome 등 다양한 데이터셋에서 일관된 성능을 보여주었음.

## 4. Analysis

### 4.1 Data Efficiency

BLIP-2는 기존의 모델들에 비해 더 적은 양의 데이터로도 높은 성능을 발휘할 수 있음을 보였음. 이는 부트스트래핑 방법이 효과적임을 입증함. BLIP-2는 초기 데이터에 크게 의존하지 않고도 점진적으로 성능을 향상시킬 수 있음.

### 4.2 Model Robustness

BLIP-2는 다양한 도메인과 태스크에서 일관된 성능을 보이며, 모델의 견고성을 입증하였음. 이는 BLIP-2의 부트스트래핑 방법이 멀티모달 학습에 유효함을 나타냄. 다양한 데이터 분포와 환경에서 BLIP-2는 높은 적응력을 보임.

## 5. Conclusion

BLIP-2는 텍스트와 이미지의 상관관계를 효과적으로 학습하는 혁신적인 비전-언어 모델임. 부트스트래핑을 통해 데이터 효율성을 높이고, 다양한 태스크에서 우수한 성능을 보였음. 이는 멀티모달 학습의 새로운 방향을 제시하며, 향후 연구와 응용에 중요한 기여를 할 것으로 기대됨. 그런데 Paper-with-Code를 살펴보니 더 우수한 모델들에 대한 수치가 필터링되서 한계가 많다고 생각됨.


















```python
@classmethod

def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):

    encoder_config = BertConfig.from_pretrained("bert-base-uncased")

    encoder_config.encoder_width = vision_width

    # insert cross-attention layer every other block

    encoder_config.add_cross_attention = True

    encoder_config.cross_attention_freq = cross_attention_freq

    encoder_config.query_length = num_query_token

    Qformer = BertLMHeadModel.from_pretrained(

        "bert-base-uncased", config=encoder_config

    )

    query_tokens = nn.Parameter(

        torch.zeros(1, num_query_token, encoder_config.hidden_size)

    )

    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

    return Qformer, query_tokens
```