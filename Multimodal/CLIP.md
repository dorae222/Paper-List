**Title**: Learning Transferable Visual Models From Natural Language Supervision
**Venue**: ICML. 2021

**Reviwer**: HyeongJun Do
**Last Updated**: 13.04.2024

**Review Objective**
> 현재 GPT-4-vision, Claude3, Gemini, LLaVA등 다양한 **Large Multimodal Model(LMM)** 및 Diffusion 및 GAN 베이스의 다양한 **이미지 생성 모델**이 있음. 이에 따라, 현 Text2Image Task의 근간이 되는 OPENAI가 2021년에 발표한 CLIP에 대해 리뷰하고자 함.

**Reference**
> OpenAI Blog: https://openai.com/research/clip
> Paper Link: https://arxiv.org/abs/2103.00020
> CLIP Github: https://github.com/openai/CLIP

**A Brief Overview**
> - 자연어 기반 지도 학습으로 Vision 모델을 새로운 Pretraining 방법론
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
		![[Pasted image 20240413104818.png]]
		 - Transformer Language Model
		 - Prediction with BOW Encoding
		 - Prediction with BOW Encoding and **the contrastive objective**(CLIP)

>  CLIP는 성능, 계산량, Robustness에서 기존 대비 향상된 결과 제공
1. Outperform the best pubilicy available ImageNet Model
2. More computationally Efficient
3. More Robust
---
## 2. Approach

>  어떻게 Vison Task에서 좋은 결과를 얻을 수 자세하게 알아보도록 하자.

### 2.1 Natural Language Supervision

> 'Natural Language Supervision' Concept은 기존에도 존재했음
- 단지 unsupervised, self-supervised, weakly supervised, and supervised를 개별적으로 언급함.
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

### 2.4 Choosing and Scaling a Model


### 2.5 Training


---
## 3. Experiments
## 3.1 Zero-shot Transfer

## 3.2 Representation Learning

## 3.3 Robustness to Natural Distribution Shift

---
## 4. Comparison to Human Performance
---
## 5. Data Overlap Analysis
---
## 6. Limitations

---
## 7. Broader Impacts
### 7.1 Bias

### 7.2 Surveillance

### 7.3 Future Work

---
## 8. Related Work