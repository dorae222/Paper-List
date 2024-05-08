**Title**: Supervision Exists Everywhere: A Data Efficient Contrastive Language-Image Pre-training Paradigm
**Venue**: ICLR 2022

**Reviewer**: HyeongJun Do
**Last Updated**: 30.04.2024

**Reference**
> OpenAI Blog: https://openai.com/research/clip
> Paper Link: https://arxiv.org/abs/2110.05208
> Github: https://github.com/Sense-GVT/DeCLIP

**A Brief Overview**
> -  위 논문에서 기존 CLIP의 경우는 Image와 Text 데이터 사이에 효율적인 사전 학습이 이뤄지지 않고 있음을 언급한다.
> 	- ![[clip.figure1_1.png]]
> - 이에 따라, 아래와 같이 3가지 방법을 통해 더 많은 이미지와 텍스트 사이의 Similarity를 학습하고 나아가 더 많은 Modality를 다룰 수 있음을 다룬다.
> - (1) self-supervision within each modality
> - (2) multi-view supervision across modalities
> - (3) nearest-neighbor supervision from other similar pair
---
## 1. Abstract
![[figule1and2.png]]
최근 언어와 이미지를 결합한 사전 학습 모델이 컴퓨터 비전과 자연어 처리 분야에서 주목받고 있다. 특히, CLIP 같은 모델은 제로샷 학습 능력으로 인해 다양한 시각적 작업에 적용 가능하다. 그러나 이러한 모델들은 막대한 양의 훈련 데이터를 필요로 하며, 이는 자원 제약이 있는 환경에서의 사용을 제한한다. 본 연구는 이미지와 텍스트 간의 내재된 상호작용과 관계를 효과적으로 활용하여 데이터 요구량을 현저하게 줄이는 방법을 제시한다.

---
## 2. RELATED WORK
언어-이미지 사전 학습 분야에서의 주요 발전과 함께, 1) self-supervised learning, 2) multi-view learning, 그리고 3) nearest-neighbor learning 같은 기술들이 소개되고 있다. 이 연구들은 주로 단일 모달리티에서의 특징을 추출하는 데 중점을 둔 반면, 본 논문에서는 이러한 개념을 언어와 이미지의 결합된 컨텍스트에서 탐구한다.

---
## 3. Approach
![[figure4.png]]
- **Self-Supervision**: 각 모달리티에서 발생하는 일관된 패턴을 학습하여, 더 일반적이고 강인한 특징을 추출한다.
	- **Image**: 이미지 자체에서 두 가지 다른 view를 생성하고, 이 view들 사이의 유사성을 최대화하는 방식으로 신경망을 학습합니다. 이를 통해 모델은 동일 이미지의 다른 표현에 대해 roburtness를 갖추게 됩니다.
	- **Text** : Masked Language Modeling(MLM)과 같은 NLP 기법을 사용하여 텍스트 데이터 내에서 유용한 언어적 특징을 학습합니다. 텍스트에서 일부 단어를 가리고 네트워크가 이를 예측하도록 함으로써, 모델은 더 나은 언어 이해를 개발합니다.
- **Multi-View Supervision**: 하나의 이미지나 텍스트를 여러 방식으로 변형시켜 동일 객체의 다양한 표현을 학습한다. 이를 통해 모델은 더 넓은 범위의 변형에 강해진다.
	- **Image**: 이미지에 여러 가지 데이터 증강 기법을 적용하여 다양한 이미지 view를 생성합니다. 예를 들어, 크롭, 회전, 색상 조정 등을 사용합니다.
	- **Text**: 텍스트에도 동일하게 sthochastic augmentation을 적용하여 다양한 텍스트 묘사를 생성합니다. 이는 텍스트의 동의어 치환, 구조 재배치 등을 포함할 수 있습니다.
- **Nearest-Neighbor Supervision**: 유사한 이미지 또는 텍스트 쌍을 찾아내어 보다 정확한 맥락적 연관성을 학습한다.
	- **Embedding Space Exploration**: 학습 과정에서 생성된 특징 임베딩을 사용하여, 각 텍스트 또는 이미지 샘플에 대한 최근접 이웃을 찾습니다.
	- **Utilization as Supervisory Signals**: 선택된 최근접 이웃을 추가적인 학습 목표로 사용하여, 모델이 더 세밀하고 다양한 데이터 표현을 학습하도록 돕습니다.

---
## 4. EXPREIMENTS
본 연구에서는 다양한 데이터셋에 대한 DeCLIP의 성능을 CLIP과 비교하여 평가한다. 실험 결과, DeCLIP은 CLIP에 비해 훨씬 적은 데이터를 사용하면서도 동등하거나 더 높은 정확도를 달성하였다. 이는 제안한 감독 방법들이 모델의 데이터 효율성을 크게 향상시킨다는 것을 보여준다.

### 4.1 Datasets
- **ImageNet**: 이 데이터셋은 비전 모델의 표준 벤치마크로 사용되며, DeCLIP의 제로샷 인식 능력을 평가하는 데 사용되었습니다.
- **CIFAR-100**: 다양한 카테고리의 이미지를 포함하고 있어, 모델의 분류 능력을 평가하는 데 사용되었습니다.
- **Visual Task Adaptation Benchmark (VTAB)**: 다양한 시각적 작업을 포함하여 모델의 전이 학습 능력을 평가합니다.
### 4.2 Zero-shot Recognition
![[source/DeCLIP/table2.png]]
- DeCLIP은 제로샷 설정에서 특히 뛰어난 성능을 보였으며, 이는 모델이 학습 중 보지 못한 이미지나 카테고리에 대해서도 높은 인식 정확도를 달성할 수 있음을 의미합니다.
- CLIP 대비, DeCLIP은 훨씬 적은 데이터를 사용하면서도 동등하거나 더 나은 성능을 보였습니다.
### 4.3 Downstream Task Evaluation
![[main_result.png]]
- **Linear Probe**: 선형 분류기를 사용하여 모델이 추출한 특징의 품질을 평가했습니다. DeCLIP은 CLIP에 비해 더 나은 또는 비슷한 성능을 여러 다운스트림 작업에서 보여주었습니다.
- **Fine-tuning**: 다양한 작업에 대해 모델을 미세 조정하였을 때, DeCLIP은 일반적인 학습 방식보다 우수한 결과를 나타냈습니다. 이는 모델이 다양한 시각적 작업에 적응할 수 있는 강력한 기능을 학습했음을 시사합니다.
---
## 5. Conclusion

DeCLIP은 언어-이미지 사전 학습 분야에서 데이터 효율성을 크게 개선할 수 있는 강력한 방법을 제시한다. 본 연구의 접근 방식은 제한된 자원으로도 효과적인 시각적 표현을 학습할 수 있는 가능성을 열어준다. 추가 연구를 통해 다양한 언어와 시각적 작업에 대한 모델의 적용성을 더욱 확장할 수 있을 것으로 기대된다.