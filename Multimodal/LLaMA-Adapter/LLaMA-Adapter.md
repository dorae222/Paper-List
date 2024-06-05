**Title**: LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention
**Venue**: ICLR 2024

**Reviewer**: HyeongJun Do
**Last Updated**: 05.06.2024
**Refence**
> Paper Link: https://arxiv.org/abs/2303.16199
> Github: https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/alpaca_finetuning_v1
---
### 1. Introduction

- **대규모 언어 모델(LLMs)**: 최근 LLMs는 언어 이해 및 생성 능력에서 큰 진전을 보였음.
- **문제점**: 기존의 LLMs 미세 조정은 많은 파라미터와 시간이 소요됨.
- **LLaMA-Adapter**: 이를 해결하기 위해 경량화된 미세 조정 방법을 제안함.

대규모 언어 모델은 최근 몇 년간 Transformer 모델의 도입으로 인해 급격히 발전함. Encoder-based Model(GPT 등) 및 Encoder-based(BERT, RoBERTa) 등의 모델은 다양한 언어 처리 작업에서 높은 성능을 보였지만, 이러한 모델의 미세 조정은 여전히 많은 자원과 시간이 필요함. LLaMA-Adapter는 이러한 문제를 해결하기 위해 제안된 방법으로, 적은 파라미터로도 기존의 높은 성능을 유지하며 미세 조정 시간을 크게 단축할 수 있음.

### 2. Related Work

- **Instruction-Following Models**: 다양한 연구에서 언어 모델을 훈련시켜 자연어 명령을 이해하고 이에 응답하는 능력을 향상시키고 있음. 예: FLAN, InstructGPT, Stanford Alpaca 등.
- **Parameter-Efficient Fine-Tuning**: 대부분의 파라미터를 동결시키고 일부 파라미터만 학습하는 방법들. 예: Prompt Tuning, LoRA, Adapters 등.

LLaMA-Adapter는 이러한 관련 연구들에 기반을 두고 있으며, 특히 Alpaca와 같은 기존의 완전 미세 조정 모델에 비해 경량화된 어댑터를 사용하여 더 효율적인 성능을 제공함. 제로 초기화 주의 메커니즘은 학습 초기에 기존 모델 지식을 보존하면서 새로운 지시 신호를 점진적으로 통합할 수 있게 함.

### 3. Methodology

#### 3.1 LLaMA-Adapter Design

- **경량화된 구조**: 학습 가능한 Adaption Prompt 세트를 사용.
- **효율성**: 적은 파라미터로도 높은 성능 유지.

LLaMA-Adapter는 LLaMA 모델의 상위 변환기 층에 학습 가능한 어댑션 프롬프트 세트를 추가하는 방법을 사용함. 이를 통해 모델의 파라미터 수를 크게 줄여 효율성을 높임. 이러한 경량화된 구조는 기존의 많은 파라미터를 사용하는 방법들에 비해 매우 효율적임.

**수식 1**: Adaption Prompt의 삽입
$$\left[P_l ; T_l\right] \in \mathbb{R}^{(K+M) \times C}$$
**Figure 1**: LLaMA-Adapter의 구조 다이어그램 ![[source/llama_adpter/figure1.png]]

- LLaMA 모델의 상위 변환기 층에 학습 가능한 어댑션 프롬프트가 추가된 모습을 다이어그램으로 설명.

#### 3.2 Zero-initialized Attention Mechanism

- **Zero-initialized attention**: 초기화된 주의 메커니즘 사용.
- **기존 지식 보존**: 새로운 지시 신호 통합과 기존 지식의 효과적 보존.

Zero-initialized attention 메커니즘은 초기화된 Attention 메커니즘을 통해 새로운 지시 신호를 모델에 통합. 이를 통해 기존의 사전 학습된 지식을 효과적으로 보존하면서도 새로운 지시 신호를 적응적으로 주입 가능. 이 메커니즘은 zero gating을 사용하여 학습 초기에는 기존 모델의 성능을 유지하고 점진적으로 새로운 신호를 통합.
**수식 2**: Zero-initialized Attention
$$S_l=Q_{l}K_{l}^{T}/\sqrt{C}$$​ 여기서 $Q_l$ ​, $K_l$​는 쿼리와 키 매트릭스.

**수식 3**: Gating Factor 적용
$$S_g=\left[\operatorname{softmax}\left(S_l^K\right) \cdot g_l ; \operatorname{softmax}\left(S_l^{M+1}\right)\right]$$

**Figure 2**: Zero-initialized attention 메커니즘 다이어그램 ![[source/llama_adpter/figure2.png]]

- 초기화된 Attention 메커니즘과 zero gating이 적용된 모습을 다이어그램으로 설명.

#### 3.3 Multi-modal Extension

- **다중 모달 학습**: 이미지 기반 학습 적용.
- **확장 가능성**: 다양한 모달리티와의 통합.

LLaMA-Adapter는 다중 모달 학습으로 확장 가능하며, 텍스트와 이미지의 조합을 처리할 수 있습니다. 이를 위해 이미지 기반 프롬프트를 추가하여, 모델이 이미지 조건부 언어 생성 작업을 수행할 수 있습니다. 이 방법은 ScienceQA와 COCO Caption 벤치마크에서 우수한 성능을 보였으며, 이는 다중 모달 학습의 가능성을 증명합니다.

**Figure 3**: Multi-modal extension 다이어그램 ![[source/llama_adpter/figure3.png]]

- 이미지 기반 프롬프트와 텍스트 프롬프트가 통합된 다이어그램.
### 4. Experiments

#### 4.1 Language Command Tasks

- **언어 명령 작업**: LLaMA-Adapter의 성능 평가.
- **비교 모델**: Alpaca 모델과의 성능 비교.

LLaMA-Adapter는 언어 명령 작업에서 높은 성능을 보임. 완전히 미세 조정된 7B 파라미터 모델인 Alpaca와 비교하여 유사한 품질의 응답을 생성할 수 있음. 성능 지표로는 정확도, 반응 시간 등이 사용되었으며, LLaMA-Adapter는 적은 파라미터로도 높은 성능을 유지함.

**Table 1**: 언어 명령 작업 성능 비교 표 ![[source/llama_adpter/table1.png]]

- 모델, 파라미터 수, 정확도, 반응 시간 등의 성능 지표를 비교한 표.

#### 4.2 Multi-modal Tasks

- **Multi Modality 작업**: 이미지 기반 모델과의 성능 비교.
- **벤치마크**: ScienceQA, COCO Caption.

LLaMA-Adapter는 다중 모달 작업에서도 우수한 성능을 보임. 이미지 조건부 언어 생성 작업에서 ScienceQA와 COCO Caption 벤치마크를 사용하여 평가되었으며, 기존의 이미지 기반 모델들과 비교하여 경쟁력 있는 성능을 입증.

**Table 2**: Multi modality 작업 성능 비교 표 ![[source/llama_adpter/table2.png]]

- 모델, 파라미터 수, ScienceQA 및 COCO Caption 벤치마크 성능 지표를 비교한 표.
#### 4.3 Generalization to Other Models
- **일반화 성능**: 다른 사전 학습 모델에의 적용.
- **적용 사례**: ViT, RoBERTa.

Zero-initialized attention 메커니즘은 ViT, RoBERTa와 같은 다른 사전 학습 모델에도 적용 가능. 이들 모델의 전통적인 비전 및 언어 작업에서 우수한 성능을 보였으며, 제안된 메커니즘의 일반화 능력을 입증.

**Table 3**: 다른 모델에의 일반화 성능 비교 표
![[table3_4.png]]

- ViT, RoBERTa 등의 모델에 대해 zero-initialized attention 메커니즘 적용 후 성능을 비교한 표.

---

### 5. Conclusion

- **연구 요약**: LLaMA-Adapter의 주요 기여.
- **향후 연구 방향**: 추가적인 연구 가능성.

LLaMA-Adapter는 효율적인 미세 조정 방법으로, 적은 파라미터로도 높은 성능을 유지할 수 있는 방법을 제안함. Zero-initialized attention 메커니즘을 통해 기존 지식을 보존하면서 새로운 지시 신호를 통합하는 혁신적인 방법을 소개. 향후 연구에서는 더 다양한 모델과 작업에 대한 적용 가능성을 탐구할 예정.

---
### 6. Code
#### zero-initialized attention 메커니즘
```python
import torch
import torch.nn as nn
from transformers import LlamaModel

class LLaMAAdapterV1(nn.Module):
    def __init__(self, llama_model, prompt_length, adapter_dim):
        super(LLaMAAdapterV1, self).__init__()
        self.llama = llama_model
        self.prompt_embeddings = nn.Parameter(torch.randn(prompt_length, llama_model.config.hidden_size))
        self.adapter = nn.Linear(llama_model.config.hidden_size, adapter_dim)
        self.zero_gating = nn.Parameter(torch.zeros(adapter_dim))
    
    def forward(self, input_ids, attention_mask=None):
        # LLaMA 모델의 출력
        hidden_states = self.llama(input_ids, attention_mask=attention_mask).last_hidden_state
        
        # Adaption Prompt 추가
        prompts = self.prompt_embeddings.expand(hidden_states.size(0), -1, -1)
        hidden_states = torch.cat([prompts, hidden_states], dim=1)
        
        # Zero-initialized Attention 적용
        attention_scores = torch.matmul(hidden_states, hidden_states.transpose(-1, -2)) / self.llama.config.hidden_size ** 0.5
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        gated_attention = attention_probs * self.zero_gating
        hidden_states = torch.matmul(gated_attention, hidden_states)
        
        return hidden_states

```
 - **prompt_embeddings**: 학습 가능한 프롬프트 임베딩을 초기화
- **adapter**: LLaMA 모델의 출력 차원을 어댑터 차원으로 변환
- **zero_gating**: 제로 초기화된 게이팅 파라미터를 정의
- **forward 메서드**: 입력 토큰에 대한 LLaMA 모델의 출력을 얻고, 프롬프트 임베딩을 추가하여 zero-initialized attention 메커니즘을 적용
#### Finetuning(Training)
```python
import torch
from transformers import Trainer, TrainingArguments, LlamaTokenizer, LlamaForSequenceClassification
from models_llama_adapter import LLaMAAdapterV1

# 사전 학습된 LLaMA 모델 로드
llama_model = LlamaForSequenceClassification.from_pretrained('llama-base')
tokenizer = LlamaTokenizer.from_pretrained('llama-base')

# LLaMA-Adapter V1 초기화
llama_adapter = LLaMAAdapterV1(llama_model, prompt_length=10, adapter_dim=512)

# 학습 설정
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# 트레이너 초기화
trainer = Trainer(
    model=llama_adapter,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 학습 시작
trainer.train()

```

- **TrainingArguments**: 학습 설정을 정의 (출력 디렉토리, 에포크 수, 배치 크기 등)
- **Trainer**: 모델, 학습 설정, 데이터셋을 사용하여 트레이너를 초기화
- **trainer.train()**: 모델 학습을 시작
#### Evaluation
```python
import torch
from transformers import Trainer, TrainingArguments, LlamaTokenizer, LlamaForSequenceClassification
from models_llama_adapter import LLaMAAdapterV1

# 사전 학습된 LLaMA 모델 로드
llama_model = LlamaForSequenceClassification.from_pretrained('llama-base')
tokenizer = LlamaTokenizer.from_pretrained('llama-base')

# LLaMA-Adapter V1 초기화
llama_adapter = LLaMAAdapterV1(llama_model, prompt_length=10, adapter_dim=512)

# 평가 설정
training_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=8,
)

# 트레이너 초기화
trainer = Trainer(
    model=llama_adapter,
    args=training_args,
    eval_dataset=eval_dataset,
)

# 평가 시작
results = trainer.evaluate()
print(results)

```
- **Trainer**: 평가 설정과 데이터셋을 사용하여 트레이너를 초기화
- **trainer.evaluate()**: 모델 평가를 수행하고 결과를 출력