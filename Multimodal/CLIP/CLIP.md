**Title**: Learning Transferable Visual Models From Natural Language Supervision
**Venue**: ICML 2021

**Reviewer**: HyeongJun Do
**Last Updated**: 17.04.2024

**Review Objective**
> 현재 GPT-4-vision, Claude3, Gemini, LLaVA등 다양한 **Large Multimodal Model(LMM)**, Diffusion 및 GAN 베이스의 다양한 **이미지 생성 모델**이 있음. 이에 따라, 현 Text2Image Task의 근간이 되는 OPENAI가 2021년에 발표한 CLIP에 대해 리뷰하고자 함.

**Reference**
> OpenAI Blog: https://openai.com/research/clip
> Paper Link: https://arxiv.org/abs/2103.00020
> CLIP Github: https://github.com/openai/CLIP

**A Brief Overview**
> - 자연어 기반 지도 학습으로 Vision 모델에 대한 새로운 Pretraining 방법론
> - Text Embedding Vector와 Image Embedding Vector간의 거리를 학습하여,
>   두 종류 데이터의 representation을 하나의 공간에서 학습하는 개념
> - Clip = **C**onstrastive **L**anguage-**I**mage **P**re-training

---
## Abstract

> **기존 SOTA Computer Vision 시스템들은 Fixed Object Category를 예측하도록 훈련됨.**
- 즉, 다른 Visual Concept을 설명하기 위해서는 추가적인 lableling 작업을 필요로 함.
- 이는 **확장성(usability) 및 일반성(generality) 등 측면에서 성능 저하**가 발생하며, **데이터 레이블링에도 어려움이 존재**함.

> **자연어(Raw Text)로부터 이미지를 직접적으로 학습함으로써 지시문(reference) 제공시 zero-shot으로 downstream task에 적용**
- 연구팀은 웹 상의 4억 개의 Text&Image Pair를 사용해 이미지 표현을 사전 학습하는 간단하고 확장 가능한 방법을 제안
- 사전 훈련된 모델은 Text를 사용하여 Zero-shot 예측으로 다양한 Visual Task에 적용 가능하며, 다수의 데이터 셋에서 높은 성능을 보여줌.
- 특히, 추가 훈련 없이 RestNet-50의 ImageNet 정확도에 달성함을 확인할 수 있음.
---
## 1. Introduction and Motivating Work

> Natural Language에서 Task-agnostic objective에서 유의미한 성과를 보이고 있음
- **Task-agnostic (Autoregressive[GPT], MLM[BERT])기반의 기존 연구들은 대규모 텍스트로 Pretraining 진행 후 Finetuning하는 방식**으로 뛰어난 성과를 얻고 있었음.

> 이러한 방식이 **Computer Vision에서도 동일하게 잘 적용 가능할까**?
- 지난 20년 간 Vision 분야에서 꾸준히 발전되었으며, **CNN based 모델이 우수한 성능을 보여주고 있었으나 zero-shot과 관련해서는 다소 낮은 정확도**를 보였음.
- 하지만 **weak supervised learning [Mahajan et al. (2018)] 에서 성과가 나타났지만, zero-shot관련 학습 능력을 제한한다고 주장**함.
- 위 논문에서는 **Natural Language supervision을 통해 Image representation Learning**에서 유의미한 결과가 나타났음을 설명함.

> 1) Weakly supervised-learning과 2) Natural Language Supervision의 주요 차이는 **Scale**임.
- CLIP은 온라인 상에서 수집하고 정제한 새롭게 만든 400m 데이터 셋으로 훈련한  [Simplified vesion of ConVIRT](https://arxiv.org/abs/2010.00747)임.

> CLIP은 Decoder Centric한 GPT 계열의 모델과 같이 **다양한 Task 수행 가능**

- **Zero-shot transfer Performance** with 30 existing datasets
		![[clip.figure1_1.png]]
		 - Transformer Language Model
		 - Prediction with BOW Encoding
		 - Prediction with BOW Encoding and **the contrastive objective**(CLIP)

>  CLIP는 성능, 계산량, Robustness에서 기존 대비 향상된 결과 제공
1. Outperform the best pubilicy available ImageNet Model
2. More computationally Efficient
3. More Robust
---
## 2. Approach

>  Introduction과 같이 어떻게 Vison Task에서 좋은 결과를 얻을 수 있었을까?

### 2.1 Natural Language Supervision

> 'Natural Language Supervision' Concept은 기존에도 존재했음
- 단지 unsupervised, self-supervised, weakly supervised, and supervised가 종합적으로 정리되지 않았을 뿐임.
	- [Contrastive Learning of Medical Visual Representations from Paired Images and Text.2020.arXiv](https://arxiv.org/abs/2010.00747)
	- [Self-supervised learning of visual features through embedding images into text topic spaces.2017. IEEE ](https://arxiv.org/abs/1705.08631)
	- [VirTex: Learning Visual Representations from Textual Annotations.2020.arXiv](https://arxiv.org/abs/2006.06666)

>  **Natural Language Supervision**: the appreciation of natural language as a training signal
- 과거 topic model과 n-gram representation방식을 사용했을 때 복잡성으로 인해 어려움이 많았지만, deep contextual representation learning이 제안되며 text를 문맥적으로 학습하는 것이 가능해짐.
	- [Learned in Translation: Contextualized Word Vectors.2017.NIPS](https://papers.nips.cc/paper_files/paper/2017/hash/20c86a628232a67e7bd46f76fba7ce12-Abstract.html)
- 이미지에 Labeling작업을 하는 것 대비 상대적으로 자연어 기반 Guide가 더 쉬움
	- 즉, 방대한 양의 Text를 인터넷 상에서 수집해야 됨.
> Natural Language은 **표현을 단순히 학습**하는 것을 넘어 unsupervised or self-supervised learning 방식에서 **zero-shot transfer이 가능**하다는 이점이 있음
- zero-shot transfer란?
	- 특별한 추가 학습 없이 새로운 작업에 모델을 적용할 수 있는 능력
	- CLIP은 이미지 분류 작업에서 특정 분류 클래스에 대한 명시적인 학습 없이도, 단순히 클래스 이름을 텍스트로 제공함으로써 이미지를 해당 클래스로 분류 가능

### 2.2 Creating a Suffiently Large Dataset
> 기존 MS-COCO, Visual Genome은 퀄리티는 좋지만, 개수가 적음
> YFCC100M은 개수는 충분하지만, metadata가 자동 생성되어 품질이 일관되지 않음

- 즉, Natural Language을 위해 사용 가능한 데이터셋이 충분하지 않음을 의미
- 이에 따라, CLIP에서는 **WIT(WebImageText)라는 새로운 데이터셋 구성**
	- 다양한 인터넷에서 수집한 4억 개의 (Image, Text) Pair로 구성
### 2.3 Selecting an Efficient Pre-Traning Method
> 과거 SOTA Comupter Vision System은 대량의 컴퓨팅 리소스를 필요로 함
- VirTex와 같이, CNN 기반 이미지와 Transformer 텍스트를 joint하여 예측하도록 함
> 그러나 위 방식은 효율적이지 않았음.
- 이는 특정 클래스에 대한 **exact** word를 예측하도록 했기 때문임
- 이미지는 Cross Entropy Loss를 구할 수 있음.
- 하지만 자연어는 Label을 구분하기가 어렵고 softmax를 사용한 분류 방식으로 학습이 어려움
> 이에 따라, Contrastive Learning 방식을 적용
- 특정 클래스에 대한 **exact**한 것이 아니라, 벡터 공간 상에서 가까운 위치에 존재하는 것끼리 매칭되도록 함.
- Contrastive Learning은 Self-Supervised-Learning에서 유용함
#### Contrastive Learning
	![[clip.figure1_1.png]]
- **방법**: 입력 이미지에 Augmentation을 적용하여, 동일한 이미지 버전끼리는 가까워지고, 다른 이미지 버전과는 멀어지도록 학습
- **1) 데이터셋 구조 및 Feature 추출**
	- 이미지: Image Encoder를 사용하여 Feature를 추출하고, 이를 초록색 사각형($I_N$)으로 표시
	- 자연어: Text Encoder를 사용하여 Feature를 추출하고, 이를 보라색 사각형($T_N$)으로 표시
	- $N$은 배치 개수
- **2) Feature 조합 및 학습 목표**
	- **조합 생성**: 각 배치에서 $N$개의 이미지 Feature와 $N$개의 텍스트 Feature가 있으므로, 가능한 모든 조합은 $N*N$개
	- **학습 방식**:
		- **매칭되는 조합**: 자신에게 매칭되는 이미지와 텍스트 Feature 사이의 Cosine Similarity를 최대화하도록 학습
		- **비매칭 조합**: 나머지 조합에서는 Cosine Similarity를 최소화하도록 학습
- **3) Cosine Similarity의 역할**
	-  **의미**: 두 Feature가 공간상에서 얼마나 가까운 각도에 위치하는지
	- **적용**: Cosine Similarity 값이 크다는 것은 두 Feature가 서로 가까워지고 있다는 것을 의미하며, 이는 두 데이터가 서로 매칭되어 있다는 뜻
	- **성과**: 레이블 정보 없이도, 레이블 정보를 사용한 학습 모델과 비슷한 수준의 표현력을 실험적으로 증명
### 2.4 Choosing and Scaling a Model
- 이미지 Encoder와 텍스트 Encoder 학습 Pseudo Lv 코드
	![[clip.figure3.png]]
	- 이전의 연구들과 다르게 Image와 Text represention을 multi-modal embedding으로 보낼 때 non-linear projection을 하지 않고 **linear하게 projection**을 하였다.
	  (np.dot($I_f$, $W_i$), np.dot($T_f$, $W_t$))
		- non-linear와 linear 모두 학습 시, 효율성에서 차이가 크게 드러나지 않았기 때문
#### Image Encoder 구성

- **다양한 Vision 모델 활용 가능**:
    - **ResNet**: 표현력을 강화하기 위해, 기존 ResNet 구조에서 마지막 Global Average Pooling을 수정하고, Attention Pooling을 도입하여 사용함. 이로 인해 더 세밀하고 효율적인 특성 추출이 가능
    - **Vision Transformer ([[ViT]])** 
	    - 기본 구조를 거의 그대로 유지하면서 사용
	    - ViT는 입력 이미지를 패치로 분할하고 이들을 시퀀스처럼 처리하는 방식으로, 이미지 내 위치적 정보를 잘 캐치해냄
- **실험 모델**: 5가지 종류의 ResNet과 3가지 종류의 ViT를 사용하여 다양한 실험을 진행함

### Text Encoder 구성

- **Transformer 사용**:
    - Transformer 모델은 입력된 텍스트의 전반적인 컨텍스트를 이해하는 데 강점을 가지며, 각 단어의 연관성과 의미를 효과적으로 분석
    - 마지막 Token에서 추출된 Feature는 Linear Projection을 통해 차원 조정을 거쳐, Image Feature와 차원을 일치시킴
### 2.5 Training
- **모델 선택 및 규모 확장**: ResNet-50, ResNet-101, EfficientNet-style 모델인 RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14를 훈련
- **훈련 과정**: 모든 모델을 32 에폭 동안 훈련
- **옵티마이저 및 학습률**: Adam 옵티마이저와 코사인 스케줄을 사용하여 학습률을 조정
- **하이퍼파라미터 초기화**: 그리드 서치, 랜덤 서치, 및 ResNet-50 모델을 1 에폭 동안 훈련시켜 수동으로 초기 하이퍼파라미터를 설정
- **기타 훈련 방법**: Mixed-precision, Gradient Checkpointing, half-precision Adam statistics, half-precision stochastically rounded 텍스트 인코더 가중치 등을 사용하여 훈련을 가속화하고 메모리를 절약함
- **훈련 시간**: 가장 큰 모델인 RN50x64는 592개의 V100 GPU에서 18일 동안, 가장 큰 Vision Transformer는 256개의 V100 GPU에서 12일 동안 훈련됨
- **성능 향상**: ViT-L/14 모델은 추가적으로 336 픽셀 해상도에서 1 에폭 동안 사전 훈련을 진행하여 성능을 향상

---
## 3. Experiments
## 3.1 Zero-shot Transfer
- **CLIP 모델 성능 비교 분석**
	![[clip.figure5.png]]
	- **Linear Probe 방식 소개**: 학습 완료된 Encoder를 기반으로 Classifier만 추가 학습하는 방법. Encoder가 효과적인 표현을 학습했다면, Classifier의 간단한 조정만으로도 높은 성능 달성 가능.
	- **Zero Shot Prediction vs. Linear Probe 성능 비교**
	    - 데이터셋의 절반 가까이에서 **Zero Shot Prediction** 성능이 **Linear Probe** 성능보다 우수함.
	    - **Fine Grained Classification 데이터셋**: 세밀한 표현 학습 필요로 함으로써 Linear Probe 방식으로는 낮은 성능을 보임.
	    - **일반적인 표현 학습이 가능한 데이터셋**: 높은 성능을 보임, 이는 Zero Shot 방식이 Label 정보를 사용하지 않고도 우수한 성능을 나타낼 수 있음을 의미.
	- **결론**: 모든 데이터셋에서 우수한 결과를 보인 것은 아니지만, 레이블 없이도 레이블을 사용한 학습 방식을 초월하는 결과를 보여줌으로써, Zero Shot Prediction의 유용성을 강조.
- **CLIP 모델과 다른 모델들의 Linear Probing 성능 비교**
	![[clip.figure6.png]]
	- **실험 개요**: 사전 학습 완료 후, 클래스당 제한된 수의 데이터만을 사용하여 Classifier를 재학습하고 성능 비교.
	- **x축 정보**: Linear Probing에 사용된 클래스당 데이터 개수.
	- **CLIP 모델의 성능**
	    - CLIP 모델은 다른 모델들에 비해 모든 면에서 우수한 성능을 보임.
	    - 다른 유명한 모델들(SimCLR[13], BiT 등)보다 뛰어난 표현 학습 능력을 검증.
	- **CLIP의 Zero Shot 성능**
	    - 클래스당 데이터 4개만 학습한 CLIP의 Linear Probing 성능과 유사.
	    - 다른 모델들은 훨씬 많은 데이터를 필요로 함에도 불구하고 CLIP의 Zero Shot 성능에 비교될 정도의 수준을 달성하기 어려움.
	    - CLIP의 Zero Shot 성능의 우수성을 강조, 레이블이 없는 환경에서도 뛰어난 성능을 보임을 의미.
- **CLIP 자체 Zero Shot 성능과 Linear Probing 성능 비교**
	![[clip.figure8.png]]
	-  **실험 개요**: 동일한 사전 학습을 받은 CLIP 모델을 사용하여 Zero Shot 성능과 Linear Probing 성능을 비교.
	- **일반 결과**
	    - 대부분의 경우, Linear Probing의 성능이 Zero Shot 성능보다 우수함을 보임.
	- **특이 사례**
	    - 몇몇 데이터셋에서는 Zero Shot 성능과 Linear Probing 성능이 매우 유사하게 나타남.
	- **결론**
	    - 이러한 결과는 CLIP의 Zero Shot Prediction 기능이 강력함을 다시 한번 입증.
	    - 일부 데이터셋에서 Linear Probing과 유사한 성능을 보이는 것은 Zero-Shot Prediction의 효과와 능력을 강조함.
## 3.2 Representation Learning
> CLIP의 Zero-Shot Transfer를 통해 광범위하게 분석했지만, 모델의 Representation Learning 능력을 보는 것이 일반적임
	- Representation의 품질을 평가하는 방법은 다양하지만, 일반적으로 Linear Probing을 사용
	- End-to-End Finetuning의 성능 측정은 또 다른 대안
- Linera Probing을 사용한 이유
	- Pretraining 단계에서 general하고 robust한 representation을 학습하지 못한 것이 더욱 잘 드러남
	- 다른 모델과의 비교가 용이하며, Zero-Shot Classifier와의 유사성
	- Fine-tuning opens up a much larger design and hyper- parameter space
- CLIP은 효율적으로 확장되며, ViT는 CNN보다 계산이 효율적임
	- Finetuning된 ViT-L/14 모델 제안
- CLIP은 이전 모델들이 수행하지 못했던 더 넓은 범위의 Task 수행
	![[clip.figure10.png]]
> Linear Probing Test는 Feature Extaractor를 fixed해 놓고 Classifier만 재학습함
	- 즉, Feature Extractor가 얼마나 범용적이고 효과적인 표현을 학습했는지 평가 가능
	- 이를 통해, Image-Text Pair를 Contrastive Learning으로 학습하는 것이 우수하다고 할 수 있음
## 3.3 Robustness to Natural Distribution Shift
- 2015년 ImageNet대회에서 DL모델이 인간을 뛰어넘었다고 발표됨
- 하지만, 이 모델들은 간단한 Task에 대해 실수하고, 새로운 벤치마크에서는 사람보다 좋지 않은 정확도를 보임
- 이는 over-fitting으로 인해 학습 데이터와 비슷한 분포의 데이터에서만 좋은 성능을 내던 것
![[clip.figure12.png]]
- 위 그림을 통해, CLIP이 다른 모델들과 비교하여 다른 데이터 분포에서도 robust함을 보여줌

---
## 4. Comparison to Human Performance
![[Table2.png]]
- 본 연구에서는 사람 5명을 대상으로 Oxford IIT Pets dataset에서 3669개의 이미지를 보고 37개의 class로 classification 진행
- 사람의 경우, Zero-shot에서 One-shot으로 갈 때 유의미한 성능 향상을 보이지만, One-shot에서 Two-shot으로 갈 때는 성능 변화가 적었음
	- 이는 인간이 자신이 아는 것과 모르는 것을 무엇인지 알고 있음을 의미함
- CLIP은 이와 다르게 Shot을 늘리면 늘릴수록 성능이 좋아지며, 추가적인 개선 방안이 있을 것이라고 판단됨
---
## 5. Limitations
- 특정 Task에서는 여전히 좋지 않은 성능을 보임
- 복잡한 Task(Fine-grained classification)에서 성능 저하가 더욱 시함
- 또한 CLIP은 온라인 상의 Image-Text Pair를  통해 학습했지만, Filtering이 되지 않아 문제가 발생함
	- 이에 따라, social biases를 그대로 학습할 수 있다
	- 쉽게 말하면, 인간이라는 Text에 해당하는 이미지에 백인이 많으면 백인 이미지만 인간이라고 판단할 확률이 높아진 다는 것이다
- 추가적으로 Human Performacne 실험에서도 알 수 있듯, CLIP에서 Zero-Shot 성능을 향상시키기 위해 Few-Shot 방법에 대한 후속 연구가 필요하다고 언급됨

