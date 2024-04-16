**Title**: AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE
**Venue**: 

**Reviwer**: HyeongJun Do
**Last Updated**: 12.09.2023

### 1. Introduction

![[ViT.1.png]]
- 논문은 CNN(Convolutional Neural Networks)을 대체할 수 있는 새로운 아키텍처로 Transformer를 제시합니다.
    
    - 기존 NLP분야에서 우수한 성능을 보이고 있는 Transformer를 CNN 대신 사용하고자 합니다.
- CNN은 지역적인 정보를 중점적으로 다루지만, Transformer는 전역적인 정보도 쉽게 활용할 수 있다는 장점이 있습니다.
    - Inductive Bias 해소 → 기존 CNN의 문제점
        - Translation Equivariance & Locality
    - Inductive Bias란
        - 정리하기에 시간이 부족해 정리가 잘 된 링크 첨부합니다.
          [[머신러닝/딥러닝] Inductive Bias란?](https://velog.io/@euisuk-chung/Inductive-Bias%EB%9E%80)
- 이미지를 패치로 나누고, 각 패치의 Linear Embedding에 순서를 제공합니다.

### 2. Related Work

- CNN은 이미지 분류, 객체 검출, 세그멘테이션 등 다양한 비전 태스크에서 활용되고 있으나, 복잡한 아키텍처와 많은 연산이 필요합니다.
    - CNN은 이미지 하나를 한 번에 인식하지만, ViT의 경우 이미지를 패치 단위로 쪼개기 때문에 Inductive Bias를 줄일 수 있습니다.
- the model of Cordonnier et al. (2020)
    - 논문에서는 ViT와 가장 유사한 Paper라고 언급되어 있습니다.
    - 입력 이미지로 부터 $2*2$ 패치 사이즈를 추출하고, full self-attention을 적용합니다.
    - 그리고 ViT에 비해 패치 사이즈가 작아, small-resolution 이미지에만 적용 가능합니다.(단점)
- 기존 CNN과 self-attention을 결합한 모델의 경우 많은 연산이 요구되며, 이전에 이러한 연구들이 있었다고 함.
    - by augmenting feature maps for image classification (Bello et al., 2019) or by further processing the output of a CNN using self-attention, e.g. for object detection (Hu et al., 2018; Carion et al., 2020)
    - video processing (Wang et al., 2018; Sun et al., 2019), image classification (Wu et al., 2020)
    - unsupervised object discovery (Locatello et al., 2020), or unified text-vision tasks (Chen et al., 2020c; Lu et al., 2019; Li et al., 2019)
- image GPT (iGPT) (Chen et al., 2020a)
    - 이미지 해상도와 색 공간을 줄인 후 이미지 픽셀에 트랜스포머를 적용합니다.
    - ImageNet에서 72%의 최대 정확도를 달성했습니다.

### 3. Method

- Vision Transformer(ViT)
    
    ![[ViT.2.png]]
    - ViT(Vision Transformer)는 이미지를 여러 개의 작은 패치로 나눈 후, 이러한 패치를 Transformer 모델에 입력으로 제공하는 방식을 사용합니다. 각 패치는 일렬로 펼쳐진 벡터로 변환되며, 그러한 패치들은 시퀀스로 취급됩니다. 이 시퀀스는 Transformer의 인코더를 통과하고, 그 결과는 이미지의 글로벌한 특성을 포착하는 것이 가능합니다.
    - ViT는 또한 위치 정보를 잃지 않기 위해 각 패치에 위치 임베딩(Positional Embedding)을 추가합니다. 이는 자연어 처리에서 문장 내 단어의 순서를 고려하는 것과 유사한 역할을 합니다.
- Fine-Tuning and Higher Resolution
    - ViT는 대량의 데이터와 계산 능력을 필요로 하는 모델이기 때문에, 미리 훈련된 모델을 다른 작업이나 더 작은 데이터셋에 적용(fine-tuning)하기가 일반적입니다. Fine-tuning은 사전에 훈련된 모델의 가중치를 초기값으로 사용하고, 특정 작업에 대한 성능을 높이기 위해 추가로 훈련을 수행하는 과정입니다.
    - 또한, 높은 해상도의 이미지를 처리하기 위해서는 더 큰 패치 크기나 더 많은 레이어를 추가할 수 있습니다. 이러한 변화는 모델의 파라미터 수를 늘리고, 그에 따라 계산 복잡성도 증가시키지만, 성능 향상을 가져올 수 있습니다.

### 4. Experiments

- SetUp
- Comparison to SOTA
- PreTraining Data Requirements
- Scaling Study
- Inspecting Vision Transformer

→ 시간이 없어서 정리를 못했습니다…

### 5. Conclusion

- 이미지 분류 작업에서도 Transformer는 뛰어난 성능을 보이며,
  이를 통해 다양한 비전 task에 Transformer를 활용할 수 있을 것이라는 가능성을 제시