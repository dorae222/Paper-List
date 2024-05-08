**Title**: FILIP: Fine-grained Interactive Language-Image Pre-Training
**Venue**: ICLR 2022

**Reviewer**: HyeongJun Do
**Last Updated**: 08.05.2024
**Refence**
> Paper Link: https://arxiv.org/abs/2111.07783

**A Brief Overview**
> - 기존 Unsupervised large-scale vision-language pre-training 방식은 downstream tasks에서 뛰어난 성능을 발휘함
> 	- 이는 각 modality 간의 global feature의 관계를 학습을 하지만, 더 세밀한(finer-grained) 부분을 학습하지 못함
> - 또한 viasual and textual tokens를 사용한 cross/self-attention 방식은 training과 inference 두 과정 모두 효율적이지 못한 결과를 보임
> - 이에 따라, cross-modal late interaction 방식을 제안함
> 	- token-wise maximun similarity to guide contrastive objective
> 	- contrastive objective만 수정

## 1. Introduction
- CLIP 및 ALIGN과 같은 VLP는 다양한 downstream task에서 우수한 성능을 보여줌.
- 그러나 **각 Modality의 유사성 만을 통해 interatcion을 모델링하므로, visual objects와 textual words 간의 관계와 같은 세밀한 수준의 정보를 capture하지 못함.**
- 이를 보완하기 위해 과거에는 2가지 방법이 사용됨.
		1) Pre-trained된 object detector를 통해 ROI(Region-Of-Interest) 특징을 추출한 다음 VLP 모델을 통해 pairing된 텍스트와 융합
		    - ROI 특징을 사전에 미리 계산하고 저장하기 때문에, 사전 학습이 복잡함.
		2) the token-wise or patch-wise representations from both modalities into the same space,
		   Cross/self-attention을 통해 finer-grained interatcions을 모델
			- Training과 Inference시 효율성이 떨어짐
			- 특히 Traning시 Cross-Attention은 Encoder-Decoder 구조에서 수행하는 반면, Self-Attention은 연결 sequence 길이에 따라 qudratic하게 증가
			- Inference시에는 두 종류의 데이터 간 Self/Cross-Attention을 dual-stream하게 계산할 수 없음.
- 이에 따라  **FILIP**(a large-scale Fine-grained Interactive Language-Image Pre-training framework) 방법론 제안
	- ==Contrastive loss에서 기존과 같이 Cross/Self-Attention을 사용하는 것이 아닌 새로운 corss-modal late interatcion mechanism 제안==
		- Visual Token과 Text Token 간의 Token-wise maximum similarity을 사용하여 Contrastive loss를 guide함
		- 이미지 패치와 텍스트 단어 사이의 세밀한 표현력을 성공적으로 활용
		- 오프라인에서 이미지와 텍스트 표현을 사전 계산하는 기능을 확보
		- Alignment 계산시 패딩된 토큰을 버리고 토큰별 최대 유사도의 평균 합계를 사용하여 훈련을 안정화 시킴

## 2. Related Work
### 2.1 Vision-Language Pre-training Models

- Pre-training 이후 Finetuning하는 방식은 NLP, Vision 도메인 모두 효과가 입증됨
- 이에 따라, Multi-Modality로 자연스럽게 확장됨
- **최근 VLP는 2가지 양상이 있음**
	1) **Image-Text Constastive Learning**
			- CLIP, ALIGN, UNIMO
			- 텍스트와 시각 정보를 통합된 semantic space로 정렬하는 cross-modal contrastive learning을 활용
	2) **Language Modeling based Tasks**
			- VisualBERT, UNITER, M6, and DALL-E
			- LM과 유사한 objective 설정
- 이외에도 여러가지 방법이 존재
	- Faster-RCNN
		- 사전 학습된 Object Dectection 모델에 의존해 오프라인으로 이미지 영역 특징을 추출
		- 이는 레이블이 지정된 경계 데이터가 추가로 필요하며 확장성이 떨어짐
	- SOHO, SimVLM
		- Visual Dictionary나 PrefixLM을 통해 Object Detection에 대한 부담을 제거하기 위한 시도
- 위 논문은 위의 ==**추론 효율성이라는 이점을 유지하면서 세분화된 vision-language representation을 end-to-end 방식으로 더 단순하게 직접 학습**==
### 2.2 Multi-Modality Interaction Mechanism

- **Cross-modal interaction Architecture**
	1) Single-Stream models
		- VisualBERT, ViLT
		- 패치별 또는 지역별 시각적 특징과 텍스트 임베딩을 직접 연결하여 Transformer 기반 모델에 feeding
	2) **Dual-Stream models**
		- ViLBERT, CLIP
		- **Separate encoders for different modalities**
		- **Encoder를 분리하여, 사전 학습 된 모델들을 유연하게 사용 가능**
		- **Image-Text Retreival과 같은 downstream에도 적용하기 편리함**
- **==위 논문은 Dual-Stream 방식을 따르면서, 세분화된 표현을 capture하기 위한 new-multimodal interatcion mechanism을 제안하고자 함==**
## 3. Method
![[figure1.png]]

- ==**FILIP은 Transformer 기반 이미지 인코더와 텍스트 인코더를 사용하는 Dual-Stream 모델
		![[figure1.png]]**==
- **Visual Modality**
	![[visual modality.png]]
	- 추가 `[CLS]` 토큰 임베딩과 선형적으로 투영된 이 미지 패치의 연결을 입력으로 받는Image Encoder로 Vision Transformer를 사용
- **Text Modality**
	![[text modality.png]]
	- 49408 Vocab Size에 소문자 BPE를 통해 토큰
	- 각 텍스트 시퀀스는 `[BOS]` 토큰으로 시작하여 `[EOS]` 토큰으로 종료
	- 단어 임베딩 레이어 이후, 토큰 임베딩은 modified decoder-only Transformer model에 feed
- L2 정규화
	![[l2 layer.png]]
	- 이미지 및 텍스트 인코더 위에 텍스트 토큰과 시각 토큰의 표현은 멀티모달 common space에 선형적으로 투영

### 3.1 FINE-GRAINED CONSTRASTIVE LEARNING

- global feature 상에서 단순히 $I$번째 이미지와 $J$번째 텍스트 각각을  Encoding
	- Image: $\boldsymbol{x}^I \in \mathcal{I}$
	- Text: $\boldsymbol{x}^T \in \mathcal{T}$
-  거리 메트릭에 따라 관련되어 있으면 가깝고 그렇지 않으면 멀리 떨어져 있도록 함
- 각 traning batch에서 $b$(Image and Text Pair)를 샘플링함.
	-  $\left\{\boldsymbol{x}_k^I, \boldsymbol{x}_k^T\right\}_{k=1}^b$
- Image-to-text contrastive loss
	- $\mathcal{L}_k^I\left(\boldsymbol{x}_k^I,\left\{\boldsymbol{x}_j^T\right\}_{j=1}^b\right)=-\frac{1}{b} \log \frac{\exp \left(s_{k, k}^I\right)}{\sum_j \exp \left(s_{k, j}^I\right)}$
- Text-to-image contrastive loss
	-  $\mathcal{L}_k^T\left(\boldsymbol{x}_k^T,\left\{\boldsymbol{x}_j^I\right\}_{j=1}^b\right)=-\frac{1}{b} \log \frac{\exp \left(s_{k, k}^T\right)}{\sum_j \exp \left(s_{j, k}^T\right)}$
- Total Loss
	-  $\mathcal{L}=\frac{1}{2} \sum_{k=1}^{b}\left(\mathcal{L}_k^I+\mathcal{L}_k^T\right)$
#### 3.1.1. Cross-Modal Late Interaction
- 이전 CLIP과 ALIGN은 아래와 같은 Constrastive Loss를 사용함
	- $f_\theta\left(\boldsymbol{x}_i^I\right) \in \mathbb{R}^d$ and $g_\phi\left(\boldsymbol{x}_j^T\right) \in \mathbb{R}^d$
	-  $s_{i, j}^I=s_{i, j}^T=f_\theta\left(\boldsymbol{x}_i^I\right)^{\top} g_\phi\left(\boldsymbol{x}_j^T\right)$과 같이 구함
	- 이는 두 modality 간의 interatcion(word-patch alignment)을 무시하는 것과 동일
- 위 문제를 해결하고자 **the token-wise cross-modal interaction**를 제안
	- $n_1$, $n_2$는 각각 $i$번째 이미지와 $j$번째 텍스트의 패딩되지 않은 토큰수
	- **$k$번째 visual token은 $x_{J}^{T}$와 유사성을 계산한 후,**
	  **가장 큰 $\max _{0 \leq r<n_2}\left[f_\theta\left(\boldsymbol{x}_i^I\right)\right]_k^{\top}\left[g_\phi\left(\boldsymbol{x}_j^T\right)\right]_r$ 와의 토큰 단위 최대 유사도 계산**
	- **이후, 이미지(또는 텍스트)에 있는 패딩 되지 않은 모든 토큰의 토큰 단위 최대 유사도 평균을 Image-Text의 유사도로 사용**
	- **따라서 $i$ 번째 이미지와 $j$번째 텍스트의 유사도는 다음과 같이 공식화**
		- $s_{i, j}^I\left(\boldsymbol{x}_i^I, \boldsymbol{x}_j^T\right)=\frac{1}{n_1} \sum_{k=1}^{n_1}\left[f_\theta\left(\boldsymbol{x}_i^I\right)\right]_k^{\top}\left[g_\phi\left(\boldsymbol{x}_j^T\right)\right]_{m_k^I}$
			- where $m_k^I=\arg \max _{0 \leq r<n_2}\left[f_\theta\left(\boldsymbol{x}_i^I\right)\right]_k^{\top}\left[g_\phi\left(\boldsymbol{x}_j^T\right)\right]_r$
		- $s_{i, j}^T\left(\boldsymbol{x}_i^I, \boldsymbol{x}_j^T\right)=\frac{1}{n_2} \sum_{k=1}^{n_2}\left[f_\theta\left(\boldsymbol{x}_i^I\right)\right]_{m_k^T}^{\top}\left[g_\phi\left(\boldsymbol{x}_j^T\right)\right]_k$[[]]
			- where $m_k^T=\arg \max _{0 \leq r<n_1}\left[f_\theta\left(\boldsymbol{x}_i^I\right)\right]_r^{\top}\left[g_\phi\left(\boldsymbol{x}_j^T\right)\right]_k$
	- 위와 같은 방법을 통해 Dual-Stream 모델은 이미지 패치와 텍스트 토큰 간의 세분화 된 Alignment를 학습
- 기존 Late Interaction Mechanism
	- masked token으로 채워진 query에 대한 문서의 관련성 점수를 토큰별 최대 유사도의 합으로 계산
	- 또한 pairwise sofmax cross-entropy loss를 통해 최적화
- **==제안하고자 하는 Late Interaction Mechanism==**
	1) 유사도 계산 시 패딩된 텍스트 토큰은 제외(성능 저하)
		- 이는 패딩된 토큰도 텍스트 표현을 학습하기 때문에 의미 있는 non 패딩 단어 대신 의미 없는 패딩된 토큰에 이미지 패치를 정렬하도록 모델을 잘못 leading할 수 있기 때문
	2) $s_{i, j}^I\left(\boldsymbol{x}_i^I, \boldsymbol{x}_j^T\right)$ & $s_{i, j}^T\left(\boldsymbol{x}_i^I, \boldsymbol{x}_j^T\right)$ 계산 시 summation 대신 토큰별 최대 유사도의 평균 사용
		- 패딩되지 않은 토큰의 수가 텍스트마다 다르고, 패딩되지 않은 모든 토큰에 대한 이 합계의 크기가 상당히 달라 학습이 불안정해지고 성능 저하로 이어질 수 있기 때문
	3) 기존 pairwise loss 대신 $\mathcal{L}=\frac{1}{2} \sum_{k=1}^{b}\left(\mathcal{L}_k^I+\mathcal{L}_k^T\right)$를 통해 VLP의 Late Interaction Mechanism 최적화
- **Training Efficiency**
	- 두 모달의 토큰 단위 표현에 의존하는 과정에서 배치 크기가 클 경우, 통신, 메모리, 그리고 계산 측면에서 비효율적
	1. **임베딩 크기 조정**: 임베딩의 크기를 256으로 줄임으로써 메모리 사용량과 계산 비용을 감소시
	2. **정밀도 감소**: 두 모달리티의 마지막 레이어 특징의 정밀도를 fp32에서 fp16으로 낮춰 계산 효율을 높이고, 노드 간 통신 비용을 줄임
	3. **유사도 계산 최적화**: 텍스트 토큰과 이미지 패치 간의 유사도 계산 복잡성을 줄이기 위해, 각 샘플에 대해 유사도 점수가 가장 높은 상위 25%의 토큰만을 선택하여 처리
#### 3.1.2. PROMPT ENSEMBLE AND TEMPLATES
1. **프롬프트 템플릿의 사용**
	- **시각화를 위한 접근법**: 논문 전반에 걸쳐 단일 프롬프트 템플릿 사용: "a photo of a {label}."
	- **다른 실험에서의 접근법**: 여러 프롬프트를 사용하여 결과를 보고하며, 다른 프롬프트 템플릿의 평균 토큰 유사도를 통해 앙상블하는 방식 적용
2. **프롬프트 템플릿 구성**
	- **구성 요소**: `[prefix] {label}, [category description], [suffix]`
	    - `[prefix]`: 예) "a photo of a"
	    - `{label}`: 데이터셋의 클래스 라벨
	    - `[category description]`: 카테고리 설명, 예) "a type of pet"
	    - `[suffix]`: 참조어 포함, 예) "I like it."
3. **프롬프트의 효과**
	- 참조어 "it" 사용은 제로샷 분류 성능을 향상시키는데 기여. "it"은 대상 객체의 이미지 패치에도 맞춰질 수 있어 미세 조정된 교차 모달 정렬을 강화시킴
### 3.2 IMAGE AND TEXT AUGMENTATION
- AutoAugment 적용
	- 데이터셋에서 자동으로 최적의 데이터 증강 정책을 학습하는 기법
		![[autoaugment.png]]
- SOTA를 달성했다!...
		![[table 1.png]]
### 3.3 PRE-TRAINING DATASET
1. **데이터셋 개요**
	- **FILIP300M** 데이터셋은 300M 개의 이미지-텍스트 쌍을 포함하며, 다양한 시각 및 언어 개념을 포괄
	- 이는 인터넷에서 이미지-텍스트 쌍을 수집하여 구축
2. **필터링 규칙**
	- **이미지 기반 필터링**: 짧은 측면이 200 픽셀 미만이거나 종횡비가 3을 초과하는 이미지 제거
	- **텍스트 기반 필터링**: 영어 텍스트만 유지하고, 의미 없는 텍스트(예: img 0.jpg) 제외. 10회 이상 반복된 텍스트가 포함된 이미지-텍스트 쌍도 제거
3. **추가 데이터셋 활용**
	- **Conceptual Captions 3M (CC3M)**, **Conceptual 12M (CC12M)**, **Yahoo Flickr Creative Commons 100M (YFCC100M)** 등 3개의 공개 데이터셋을 추가로 사용하며, YFCC100M에도 동일한 필터링 규칙 적용
4. **사전 훈련 데이터**
	- 최종적으로 약 340M 개의 이미지-텍스트 쌍을 사전 훈련에 사용
	- CLIP과 ALIGN이 사용한 데이터셋 보다는 작지만, 대부분의 하류 작업에서 이들 모델을 능가하는 성능을 보임.
## 4. Experiments

1. **Experimental Setup**
	- **Model Architectures**: FILIP은 두 가지 모델 구조(FILIPbase, FILIPlarge)를 사용, CLIP의 구조를 따라 이미지 인코더로는 ViT-B/32 및 ViT-L/14 사용
	- **Pre-training Details**: automatic mixed-precision와 gradient checkpoint를 사용하여 메모리 사용을 최소화하고 배치 크기를 늘림. 최적화는 LAMB optimizer와 cosine learning rate schedule 사용
2. **Zero-Shot Image Classification**
	- FILIP은 12개의 다운스트림 분류 데이터셋에서 제로샷 평가를 수행
	- CLIP과 비교하여 FILIP은 평균적으로 상당한 성능 향상을 보였으며, 특히 ImageNet에서 더 높은 정확도 달성
3. **Image-Text Retrieval**
	- Flickr30K와 MSCOCO 데이터셋에서 이미지-텍스트 검색 작업에 대한 FILIP의 성능을 평가
	- 제로샷 설정과 미세 조정 설정 모두에서 FILIP은 기존 모델을 상회하는 결과를 보임
4. **Ablation Study**
	- 이미지/텍스트 증강, Cross-modal Late Interaction 같은 FILIP의 각 컴포넌트의 효과를 평가
5. **Visualization of Fine-Grained Alignment**
	- FILIP의 이미지 패치와 텍스트 토큰 간의 미세 정렬 능력을 시각화하여 보여줌.
	- 이를 통해 FILIP이 이미지와 관련 텍스트 사이의 미세한 의미적 연결을 어떻게 학습하는지를 보여줌.
