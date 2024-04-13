## 0. Abstract

- 가장 널리 사용되는 고성능 텍스트 임베딩 모델인 BERT를 소개한다.
- BERT가 무엇이며 다른 임베딩 모델과 어떻게 다른지 이해하는데서부터 시작한 다음 BERT의 동작 방식과 구조를 살펴본다.
- 마스크 언어모델링(MLM)과 다음 문장 예측(NSP)이라는 두 가지 테스크 기반 BERT 모델이 어떻게 사전 학습을 진행하는지 알아본다.
- 하위단어 토큰화 알고리즘에 대해 알아본다.
- 앞 선 내용들을 이해하고, 논문을 다시 본다.

---

## 1. BERT의 기본 컨셉

### 1-1. BERT란

**BERT(Bidirectional Encoder Representation from Transformer)**는 구글에서 발표한 **텍스트 임베딩 모델**입니다.

NLP 분야에서 많은 기여를 해왔으며, BERT를 기반으로 한 다양한 모델들이 존재합니다.

### 1-2. BERT's Usefulness

이전의 **word2vec**과 같은 문맥 독립(context-free) 임베딩 모델은 해당 단어의 의미를 파악하기 힘들었습니다.

하지만 **BERT**는 모든 단어의 문맥 기반(context-based) 임베딩 모델로 **문맥 정보**를 고려할 수 있숩니다.

![[Untitled.png]]

위의 그림을 보면 이해가 쉬울 것 같지만 아래의 예시를 통해 BERT를 조금 더 자세하게 컨셉을 이해해보도록 하겠습니다.

- 추가 예시
    
    아래와 같이 두 문장이 있다고 하자.
    
    > A: 나는 항구에 들러 배를 탔다.
    
    > B: 나는 사과와 배를 먹었다.
    
    두 문장에서 '배'라는 단어의 의미가 서로 다르다는 것을 알 수 있다.
    
    BERT는 모든 단어의 문맥상 의미를 이해하기 위해 아래의 그림과 같이, 문장의 각 단어를 문장의 다른 모든 단어와 연결 시켜 이해합니다.
    
    ![https://blog.kakaocdn.net/dn/nSUtl/btstsQHE2Hp/eIeuptU4eJw5HHWsCL7VK1/img.png](https://blog.kakaocdn.net/dn/nSUtl/btstsQHE2Hp/eIeuptU4eJw5HHWsCL7VK1/img.png)
    
    A 케이스의 경우, 항구와 함께 언급되었기에 해당 '배'의 의미는 ship이라는 의미로 이해할 수 있습니다.
    
    또한 B케이스에서는 아마도 사과나 먹었다와 관련성을 가지고 pear라는 의미로 이해할 수 있습니다.
    
    이렇게 문맥을 고려하면 **다의어 및 동음이의어**를 구분할 수 있게 됩니다.
    

---

## 2. BERT의 동작 방식 및 구조

### 2-1. BERT의 이름 알아보기

- **B** — **B**idrectional
        - RNN을 공부해보셨다면 양방향 RNN을 떠올리시면 쉽습니다.
    - 트랜스포머 인코더는 원래 양방향으로 문장을 읽을 수 있습니다.
    - 따라서 BERT는 기본적으로 트랜스포머에서 얻은 양방향 인코더 표현입니다.
- **ER** — **E**ncoder + **R**epresentation
    
    BERT는 이름에서 알 수 있듯이 트랜스포머 모델을 기반으로 하며, 인코더-디코더가 있는 트랜스포머 모델과 달리 인코더만 사용합니다.
    
    문장을 트랜스포머 인코더에 입력하고, 문장의 각 단어에 대한 표현 벡터를 출력으로 반환하는 것을 확인하였습니다.
    
    즉, 트랜스포머의 인코더는 BERT의 **표현 벡터** 입니다.
    
- from
    
- **T** — **T**ransformer
    
    - Introduction
        
        - SInce 2016,
            
            - In 2016, Google puplished thier nueral machine translation system(GNMT), which outperforms previous traditional MT system.
            
            [Google’s Neural Machine Translation System.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3acf089a-afaa-4cf8-8aee-c3a6cd364982/Googles_Neural_Machine_Translation_System.pdf)
            
            ![[Untitled 1.png]]
            
            
            - However, RNN based Sequence-to-sequence reveals its limitations.
            ![[Untitled 2.png]]

            
        - Fully Convilutional Seq2Seq[Gehring et al.2017p]
            
            [Convolutional Sequence to Sequence Learning.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e3cc7dc7-cf90-4146-8f2a-cf36742cd3a0/Convolutional_Sequence_to_Sequence_Learning.pdf)
            
            ![[Untitled 3.png]]
            
        - Attention is all you need
            
            [Attention Is All You Need.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/41b6edfe-428a-4fe2-bff7-7754f8dab242/Attention_Is_All_You_Need.pdf)
            
        - Transformer
            
            - Encoder + decoder
                
                ![[Untitled 4.png]]
                
            - 성능과 속도 모두 기존 모델을 압도
                
                ![[Untitled 5.png]]
                
        - Transformer and MLP
            
            - Pretraining and finetuning (Transfer Learning) with Big-LM.
            
            ![[Untitled 6.png]]
            
    - Multi-head Attention
        
        - Attention: Query Generation
            
            - Example
            
            ![[Untitled 6.png]]
            
            ![[Untitled 7.png]] ![[Untitled 8.png]]
            
            
            
            - 마음의 상태(state)를 잘 반영하면서 좋은 검색 결과를 이끌어내는 쿼리를 얻기 위함
            - 만약 검색을 다양하게 할 수 있다면?
        - Transformer & Attention
            
            - Scaled Dot-product Attention
                
                ![[Untitled 9.png]]
                
                
            - Multi-Head Attention
                
                ![[Untitled 10.png]]
                
            - Transformer
                
                ![[Untitled 11.png]]
                
            - Encoder&Decoder가 여러 층인 것을 굳이 그려보자면
                
                ![[Untitled 12.png]]
                
        - Equations
            
            [Multihead_Attention.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2af6f0f6-515f-4b84-b7f9-b8eaf9a24092/Multihead_Attention.pdf)
            
            ![[Untitled 13.png]]
            
        - Summary
            
            - Previous method: attention in sequence to sequence
                - Query를 잘 만들어 key-value를 잘 matching시키자
            - Multi-head Attention
                - 여러개의 Query를 만들어 다양한 정보를 잘 얻어오자
            - Attention 자체로도 정보의 encoding과 decoding이 가능함을 보여줌
    - Encoder block
        
        - Transformer
            
            ![[Untitled 14.png]]
            
            
        - Equations
            
            - Q,K and V are from previous alayer:
                
                - Residual connections and Layer Normalizations are used.
                    
                    [Deep Residual Learning for Image Recognition.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/94cae579-4b3d-4b43-8267-8c93d153fd63/Deep_Residual_Learning_for_Image_Recognition.pdf)
                    
                    [Layer Normalization.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cd879c33-928d-4973-a013-85d6964ecbc9/Layer_Normalization.pdf)
                    
                
                ![[Untitled 14.png]]
            - Encoder is stack of encoder blocks:
                
                ![[Untitled 15.png]]
                
        - Summary
            
            - Encoder는 self-attention으로 구성되어 있음
                - Q,K,V는 이전 레이어의 출력값 - 즉, 같은 값
            - Seq2Seq의 Attention과 달리, Q도 모든 time-step을 동시에 연산
                - 빠르지만 메모리를 많이 먹게 됨
            - Residual connection으로 인해 깊은 네트워크 구성 가능
                - Big LM의 토대 마련
    - Decoder with Masking
        
        - Transformer
            
            ![[Untitled 16.png]]
            
        - Equations
            
            - Given Dataset.
                
                ![[Untitled 17.png]]
                
            - What we want is
                
                ![[Untitled 18.png]]
                
            - Before we start
                
                - Using mask, assign -∞ to make 0s for softmax results..
                
                ![[Untitled 19.png]]
                
            - Decoder Self-attention with mask
                
                - 모든 attention에는 <pad>에 마스킹이 들어간다.
                - 단, 디코더에서는 AuroRegressive한 특성으로 인해, 다음스텝을 보는 것을 방지하는 마스킹을 함께 해줘야 한다.
                
                ![[Untitled 20.png]]
                
            - Attention from encoder with mask for <pad>
                
                ![[Untitled 21.png]]
                
            - FC layers
                
                ![[Untitled 22.png]]
                
            - Decoder is stack of decoder blocks.
                
                ![[Untitled 23.png]]
                
            - Generator
                
                ![[Untitled 24.png]]
                
        - Summary
            
            - Decoder는 2가지의 Attention으로 구성됨
                - Attention from encoder:
                    - K와 V는 encoder의 최종 출력 값, Q는 이전 레이어의 출력 값
                - Self-Attention with mask:
                    - Q,K,V는 이전 레이어의 출력 값
                    - Attention weight 계산 시, softmax 연산 이전에 masking을 통해 음의 무한대를 주어, 미래 time-step을 보는 것을 방지
            - 추론 때에는 self-attention의 mask는 필요 없으나, 모든 layer의 t 시점 이전의 모든 time-step(<t)의 hidden_state가 필요
    - Positional Encoding
        
        - Unlike RNN,
            
            - Transformer는 위치 정보를 스스로 처리하지 않음(Conv2S도 마찬가지)
                - $h_{t}=f(x_{t},h_{t-1})$ → RNN 계열은 위치 정보를 스스로 처리했음
                - 마치 FC layer의 입력 feature 순서를 바꿔 학습해도 성능이 똑같은 것과 같음
            - 입력 순서를 바꿔 넣으면 출력도 순서가 바뀐 채 같은 값이 나올 것
            - 따라서 위치순서 정보를 따로 인코딩해서 넣어줘야 함
        - Positonal Encoding
            
            - 기존의 word embedding 값에 positonal encoding 값을 더해줌
            
            ![[Untitled 25.png]]
            
        - vs Positional Embedding → 학습도 가능
            
            - 사실 위치 정보도 integer 값이므로 embedding layer를 통해 임베딩 할 수 있음
            - BERT와 같은 모델은 positional encoding 대신에 positional embedding을 사용하기 도함
        - Summary
            
            - RNN과 달리, 순서(위치) 정보를 encoding해주는 작업이 필요
                - 학습이 아닌 단순 계산 후 encoding
            - 학습에 의해 달라지는 값이 아니므로, 한번만 계산해 놓으면 됨
        
        [transfomer-positional-encoding](https://www.blossominkyung.com/deeplearning/transfomer-positional-encoding)
        
    - Learning rate warm-up and linear decay
        
        - Previous Method
            
            **SGD+Gradient Clipping**
            
            - 가장 기본적인 방법
            
            $\theta \leftarrow \theta-\gamma \bigtriangledown_{\theta}L(\theta)$
            
            - Learning rate에 따른 성능 변화
            - 학습 후반부에 LR decay 해주기도
            
            **Adam**
            
            - Adaptive하게 LR을 조절
            
            ![[Untitled 26.png]]
            
            - 일부 깊은 네트워크(e.g.Transformer)에서 서 성능이 낮음
                - 문제는 지금은 Transformer의 세상
        - Warm-up and Linear Decay(Noam Decay)
            
            - Heuristic Methods
                - Control learning rate for Adam with hyper-params
            - 학습 초기 불안정한 gradient를 통해 잘못된 momentum을 갖는 것을 방지
                - Residual Connection을 하는 과정에서 발생??
                - 대체로 5% 근처
            
            ![[Untitled 27.png]]
            
            - 결국 Trial&Error방식으로 Hyper-parameter 튜닝을 해야함
                - 가장 핵심은 #warm-up steps와 #total iterations.
                - 이외에도 다양한 hyper-parmas: init LR, batch size
            - 심지어 튜닝에 따라 SGD+Gradient Clipping이 더 나은 결과를 얻기도 함
        - Rectified Adam[Liu et al., 2020]
            
            [On The Variance of the Adaptive Learning Rate and Beyond.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e17bb8aa-84f6-49e5-bc33-d0768d7073f6/On_The_Variance_of_the_Adaptive_Learning_Rate_and_Beyond.pdf)
            
            - Adam이 잘 동작하지 않는 이유(가설)
            - Due to the lack of samples in the early stage, the adaptive learning rate has an undesirably large variance, which leads to suspicious/bad local optima. – [Liu et al., 2020]
            
            ![[Untitled 28.png]]
            
            - Pytorch 구현
                
                [https://github.com/LiyuanLucasLiu/RAdam](https://github.com/LiyuanLucasLiu/RAdam)
                
                - `$pip install torch-optimizer`
    - Appendix: Beyond the paper
        
        - Transformer의 단점
            
            - 학습이 까다롭다.
                
                - Bad local optima에 빠지기 매우 쉬움
                    
                - 그런데 paper에서 이것을 언급하지 않음
                    
                    -  warm-up step, learning rate
                    
                    ![[Untitled 29.png]]
                    
            - 오죽하면,
                
                [Training Tips for the Transformer Model.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/769d69f2-b212-4045-81c2-a19503c572f3/Training_Tips_for_the_Transformer_Model.pdf)
                
                [Transformers without Tears_Improving the Normalization of Self-Attention.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/65ab91ae-94b7-4221-8962-a77b9c90d160/Transformers_without_Tears_Improving_the_Normalization_of_Self-Attention.pdf)
                
                [On the Variance of the Adaptive Learning Rate and Beyond.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/efe85efb-8dcb-491c-984c-541a34675367/On_the_Variance_of_the_Adaptive_Learning_Rate_and_Beyond.pdf)
                
        - On Layer Normalization in Transformer Architecture[Xiong et al., 2020]
            
            [On Layer Normalization in Transformer Architecture.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/40f54878-9814-4339-843a-4c284da92f6d/On_Layer_Normalization_in_Transformer_Architecture.pdf)
            
            - Previous Work:
                
                - Use Noam decay(warm-up and linear decay)
                - Rectified Adam(RAdam)
            - Propose:
                
                - Layer Norm의 위치에 따라 학습이 수월해짐
                    - LN이 gradient를 평탄하게 바꾸는 효과
                
                ![[Untitled 30.png]]
                
                ![[Untitled 31.png]]
                
            - Evaluation Results
                
                ![[Untitled 32.png]]
                
        - Summary
            
            - Pre-Norm 방식을 통해 warm-up 및 LR 튜닝 제거 가능
                - LR decay는 여전히 필요
            - 그 밖에도 Layer Norm을 대체하거나, weight initialization을 활용하여 좀 더 나은 성능을 확보할 수 있음

### 2-2. 동작 방식

- 이전 예시 **나는 항구에 들러 배를 탔다**라는 A 문장을 트랜스 포머의 인코더에 입력으로 제공하고 문장의 각 단어데 대한 임베딩을 출력으로 가져온다.
- 인코더에 문장을 입력하면 인코더는 멀티헤드 어텐션 메커니즘을 사용해 문장의 각 단어의 문맥을 이해해 문장에 잇는 각 단어의 문맥 표현을 출력으로 반환한다.

![[Untitled 33.png]]

### 2-3. 구조

BERT는 크기에 따라 아래의 두 모델로 나뉜다.

- `Bert-base`: OpenAI GPT와 동일한 하이퍼파라미터를 가짐. GPT와의 성능 비교를 위해 설계됨
- `Bert-large`: BERT의 최대 성능을 보여주기 위해 만들어짐

![[Untitled 34.png]]

![[Untitled 35.png]]

- 모든 task에 대해 SOTA 달성
    
- BERT-large가 일반적으로 base 모델보다 성능이 뛰어남
    
- 사전학습 덕분에 데이터셋의 크기가 작아도 모델의 크기가 클수록 정확도가 상승
    
    - 사전학습을 한다는 것은 General하게 사람들의 언어체계를 학습한다라고 이해하면 쉽습니다.
    - 즉, 문맥과 표현을 사전학습을 통해 배워 놓고, Down stream task를 목적에 따라 파인튜닝하는 것입니다.
- 그 밖의 여러 BERT 기본 구조
    
    |BERT-tiny|L=2,A=2,H=128|
    |---|---|
    |BERT-mini|L=4,A=4,H=256|
    |BERT-small|L=4,A=8,H=521|
    |BERT-medium|L=8,A=8,H=521|
    

---

## 3. BERT의 사전 학습

### 3-1. BERT의 입력 표현

이제 우리는 BERT에 데이터를 입력하기 전에 임베딩 레이어를 기반으로 입력 데이터를 임베딩으로 변환해야 합니다.

그리고 아래와 같이 3가지 방법이 있습니다.

- Token Embedding (2)
    
    - Sentence Pair는 합쳐져서 단일 Sequence로 입력되고, Pair는 한 개 혹은 2개의 문장으로 이루어져 있다.
        
        - e.g. Translation
            - 1: My dog is cute, he likes playing.
            - 2: 나의 강아지는 귀엽고, 노는 것을 좋아한다.
    - 문장의 시작 부분에는 `[CLS]` 라는 토큰을 추가합니다.
        
        - [CLS] 토큰의 경우, 분류 테스크에서만 사용되지만 다른 테스크에서도 반드시 추가해줘야 합니다.
    - 문장의 끝에는 `[SEP]` 라는 토큰을 추가합니다.
        
    - 예시
        
        ![[Untitled 36.png]]
        
        - 기존 토큰: `token = [my,dog,is,cute,he,likes,playing]`
        - 토큰 임베딩을 거칠 경우: `token_embedding=[[CLS], my, dog, is, cute, [SEP], he, likes, playing ,[SEP]]`
- Segment Embedding (3)
    
    - Segment Embedding은 주어진 두 문장을 구별하는데 사용됩니다.
        
        - 토큰 임베딩이 진행되었다고 가정합니다.
            - `token_embedding=[[CLS], my, dog, is, cute, [SEP], he, likes, playing ,[SEP]]`
        - 예시 문장1: My dog is cute
        - 예시 문장2: He likes playing
    - Segment Embedding Layer는 입력에 대한 출력으로 $E_A$와 $E_B$만 반환합니다.
        
        ![[Untitled 37.png]]
        
        - 1번 문장에 속할 경우 $E_A$를 반환
        - 2번 문장에 속할 경우 $E_B$를 반환
- Position Embedding (4)
    
    [vs Positional Embedding → 학습도 가능](https://www.notion.so/vs-Positional-Embedding-7d912d5c51834c1b9ee42445393553c6?pvs=21)
    
    - 주의 하실 점은 Positional Encoding과는 조금 다른 Positional Embedding은 다른 개념입니다.
    - Transformer의 메커니즘을 이해한다면, 어떤 반복 메커니즘을 사용하지 않고 모든 단어를 병렬 처리함을 알고 있을 것입니다.
    - 이에 따라, 단어의 순서가 중요하므로 위치에 대한 정보를 제공한다.
    
    ![[Untitled 38.png]]
    
    - 위 그림이 최종적으로 BERT 모델에 들어가는 INPUT 값이 됩니다.
- WordPeice Tokenizer (1)
    
    - 그런데 Token Embedding층에 들어갈 Input으로 사용될 토큰이 어떻게 쪼개지는지 알아야겠죠?
    - BERT는 하위 단어 토큰화 알고리즘([4. 하위 단어 토큰화 알고리즘](https://www.notion.so/4-eb353e9dc9064723b2045ec89933c2b0?pvs=21))을 기반으로 작동합니다.
    - 한글을 기준으로 형태소를 분석하듯이, 영어의 경우 `pretraining`이라는 단어를 토큰화 해보겠습니다.
        - pretraining = pre + train + ing
        - `tokens = [ pre, ##train, ##ing ]`
    - 그렇다면 왜 하위 단어로 쪼개는 것일까요?
        - 하위 단어로 쪼갤 경우, 어휘 사전 이외(OOV, Out-Of-Vocabulary)의 단어를 처리하는데 효과적입니다.
        - 기본적으로 BERT의 어휘 사전은 3만 토큰이므로, 왠만한 어휘들은 토큰화가 가능합니다.
        - 만약 존재하지 않는 토큰이라면 어떻게되는지는 [4. 하위 단어 토큰화 알고리즘](https://www.notion.so/4-eb353e9dc9064723b2045ec89933c2b0?pvs=21) 에서 추가적으로 다뤄보겠습니다.

### 3-2. 사전 학습 전략

- 기존의 사전 학습 방법론
    
    ![[Untitled 39.png]]
    
    - 전통적인 언어 모델링(Language Modeling): **n-gram**, 앞의 N-1개의 단어로 뒤에 올 단어를 예측하는 모델
    - 필연적으로 단방향일수 밖에 없고, BiLM을 사용하는 ELMo더라도 순방향, 순방향의 언어 모델을 둘 다 학습해 활용하지만,
    - 단방향 언어 모델의 출력을 concat하여 사용하는 정도이므로 제한적인 양방향성을 가짐

---

**NEW**

- 마스크 언어 모델링(**MLM**, Masked Language Modeling)
    - MLM은 일반적으로 임의의 문장이 주어지고 단어를 순서대로 보면서 다음 단어를 예측 하도록 모델을 학습 시키는 것입니다.
        
    - MLM의 종류 2가지
        
        - 자귀 회귀 언어 모델링 — (단방향)
            - 앞 → 끝 방향으로 예측(전방 예측)
            - 끝 → 앞 방향으로 예측(후방 예측)
            - 예시
                
                - `Paris is a beautiful city. I love Paris.` 라는 두 문장이 있습니다.
                
                1. 처음에는 city라는 단어에 공백을 추가한다.
                    1. `Paris is a beautiful __. I love Paris.`
                2. 이제 모델은 공백을 예측한다.
                    1. 전방 예측은 Paris부터 문장을 읽는다.
                        1. (→) `Paris is a beautiful __.`
                    2. 후방 예측은 끝의 Paris부터 문장을 읽는다.
                        1. (←) `__. I love Paris.`
            - 자동 회귀 언어 모델은 원래 단 방향이므로 한 방향으로만 문장을 읽습니다.
        - 자동 인코딩 언어 모델링 — (양방향)
            - 이 방식의 경우, 전방 및 후방 예측을 모두 활용합니다.
            - 즉, 양방향으로 문장을 읽습니다.
            - (→)`Paris is a beautiful __. I love Paris.`(←)
            - 당연히 단방향 보다 문장 이해 측면에서 나으므로, 더 정확한 결과를 제공합니다.
    - **BERT는 자동 인코딩 언어 모델로, 예측을 위해 양방향으로 문장을 읽습니다.**
        
    - **주어진 문장에서 전체 단어의 15%를 무작위로 마스킹하고, 마스크된 단어를 예측하도록 모델을 학습하는 것입니다.**
        
        - `Paris is a beautiful city. I love Paris.`
        - 위의 예시에서 들었던 문장을 토큰화 하고 마스킹하도록 하겠습니다.
            - `tokens = [ [CLS], Paris, is, a, beatiful, [MASK], [SEP], I, love, Paris, [SEP] ]`
    - 하지만 위와 같이 토큰화를 하게 될 경우, 사전학습과 파인튜닝 사이에서 불일치가 발생합니다. `[MASK]`토큰이 없기 때문입니다.
        
    - 이 문제를 극복하기 위해 **80-10-10% 규칙**을 사용합니다.
        
        - 기존 15% 중 **80%**의 토큰을 [MASK] 토큰으로 교체한다.
            - `tokens = [ [CLS], Paris, is, a, beatiful, [MASK], [SEP], I, love, Paris, [SEP] ]`
        - 15% 중 **10%**의 토큰을 임의의 토큰으로 교체한다.
            - `tokens = [ [CLS], Paris, is, a, beatiful, **love**, [SEP], I, love, Paris, [SEP] ]`
        - 15% 중 나머지 10%의 토큰은 어떤 변경도 하지 않는다.
            - `tokens = [ [CLS], Paris, is, a, beatiful, city, [SEP], I, love, Paris, [SEP] ]`
    - 이후, 앞서 언급된 (Token, Segment, Position) Embedding 층을 거쳐 입력 임베딩(토큰의 표현 벡터)를 반환합니다.
        
        - $R_{[CLS]}$는 [CLS] 토큰의 표현 벡터를 의미하고, $R_{[Paris]}$는 Paris의 표현 벡터를 의미합니다.
        
        ![[Untitled 40.png]]
        
    - 이제 우리는 토큰의 표현 벡터를 얻었으므로, 마스크된 토큰을 예측해야한다.
        
        - BERT에서 반환된 마스크된 토큰 $R_{[MASK]}$의 표현을 ($softmax$활성화+$feed-forward$) 네크워크에 입력한다.
        - 이후, 우리는 해당 마스크 위치의 단어가 될 확률을 얻을 수 있게 된다.
        
        ![[Untitled 41.png]]
        
    - 전체 단어 마스킹(WWM, Whole Word Masking)
        
        - MLM에서 조금 더 나아가 심화 내용인 전체 단어 마스킹에 대해서 알아보자.
            - Let us start pretraining the model이라는 문장을 워드피스 토크나이저를 사용해 문장을 토큰화하면 다음과 같은 토큰을 얻을 수 있다.
            - `tokens = [let, us, start, pre, ##train, ##ing, the, model]`
            - `tokens = [ [CLS], let, us, start, pre, ##train, ##ing, the, model, [SEP]]`
            - `tokens = [ [CLS], [MASK], us, start, pre, [MASK], ##ing, the, model, [SEP]]`
        - WWM 방법에서는 하위 단어가 마스킹 되면 관련된 모든 단어가 마스킹된다.
            - `tokens = [ [CLS], [MASK], us, start, [MASK], [MASK], [MASK], the, model, [SEP]]`
        - 마스크 비율(15%)를 초과하면 다른 단어의 마스크를 무시한다. 이 때는 let을 무시한다.
            - `tokens = [ [CLS], let, us, start, [MASK], [MASK], [MASK], the, model, [SEP]]`
        - 이후는 동일하게 마스크 된 토큰을 학습하도록 한다.
- 다음 문장 예측(**NSP**, Next Sentence Prediction)
    - NSP는 BERT 학습에서 사용되는 다른 테스크로, 이진 분류 테스트다.
        
        - isNext
        - NotNext
        
        |문장 쌍|레이블|
        |---|---|
        |She cooked pasta||
        |It was delicious|isNext|
        |Birds fly in the sky.||
        |He was reading|NotNext|
        
    - NSP는 두 문장 사이의 관계를 파악하며, 질문-응답 및 유사문장탐지와 같은 다운 스트림 테스크에서 유용하다.
        
    - 예시를 통해 살펴보자.
        
        ![[Untitled 42.png]]
        
    - BERT는 [CLS] 토큰만 가져와 분류 작업한다.
        
        - [CLS]토큰은 기본적으로 모든 토큰의 집계 표현을 보유하고 있으므로 문장 전체에 대한 표현을 담고 있다.
        - 학습 초기에는 물론 피드포워드 네트워크 및 인코더 계층의 가중치가 최적이 아니라 올바른 확률을 반환하지 못하지만,
        - 역전파를 기반으로 반복 학습을 통해 최적의 가중치를 찾게 되면 아래의 그림과 같은 반환 값을 내놓게 된다.
        
        ![[Untitled 43.png]]
        

---

### 3-3. 사전 학습 절차

1. 말뭉치에서 두 문장 A, B를 샘플링한다.
    - A와 B의 총 토큰 수의 합은 512보다 작거나 같아야 한다.
    - 전체의 50%은 B 문장이 A 문장과 이어지는 문장(`IsNext`)이 되도록 샘플링하고, 나머지 50%은 B 문장이 A 문장의 후속 문장이 아닌 것(`NotNext`)으로 샘플링한다.
2. 워드피스 토크나이저로 문장을 토큰화하고, 토큰 임베딩-세그먼트 임베딩-위치 임베딩 레이어를 거친다.
    - 시작 부분에 `[CLS]` 토큰을, 문장 끝에 `[SEP]` 토큰을 추가한다.
    - `80-10-10%` 규칙에 따라 토큰의 15%를 무작위 마스킹한다.
3. BERT에 토큰을 입력하고, MLM과 NSP 태스크를 동시에 수행한다
    - WarmUp Step(= 1만): 초기 1만 스텝은 학습률이 0에서 1e - 4로 선형 증가, 1만 스텝 이후 선형 감소
        
    - DropOut(0.1)
        
    - **GeLU Activation Func** : 음수에 대해서도 미분이 가능해 약간의 그래디언트를 전달할 수 있음
        
        - GELU는 가우시안 오차 선형 유닛(Gaussian Error Linear Unit)을 사용한다고 합니다.
        - $G E L U(x)=x \Phi(x)$
        - $\Phi$은 표준 가우시안 누적 분포이며, GELU함수는 다음 수식의 근사치라고 합니다.
        - $\operatorname{GELU}(x)=0.5 x\left(1+\tanh \left[\sqrt{\frac{2}{\pi}}\left(\mathrm{x}+0.044715 x^3\right)\right]\right)$
        
        
        ![[Untitled 44.png]]

---

## 4. 하위 단어 토큰화 알고리즘

### 4-1. 하위 단어 토큰화를 하는 이유

- BERT 및 GPT-3를 포함한 많은 최신 LLM 모델에서는 하위 단어 토큰화를 사용합니다.
- 그 이유는 바로 OOV단어 처리에 매우 효과적입니다.
    - **OoV(**Out of Vocabulary): 단어 집합에 존재하지 않는 단어들이 생기는 상황 (Inference시 TrainSet에 없던 단어가 있을 경우)
- OOV가 발생한다면 어떻게 할까?
    - OoV 단어 발생 시 `<UNK>` 토큰으로 처리하고, `<UNK>` 토큰은 모델 학습에 있어 매우 치명적으로 작용합니다.
    - 하지만 일반적으로 Vocab 자체가 매우 크기 때문에 신조어가 아닌 이상 대부분의 단어는 하위 단어로 토큰화 할 수 있습니다.

### 4-2. 바이트 쌍 인코딩(BPE)

- **BPE**은 빈도수가 가장 높은 쌍의 문자나 문자열을 하나로 합치는 알고리즘입니다.
- 이 방법은 초기에 데이터 압축에 사용되었으나, 나중에 자연어 처리에서 텍스트 토큰화에도 적용되었습니다.

1. Dataset에서 모든 단어를 빈도수와 함께 추출
2. 모든 단어를 문자로 나누고 문자 시퀀스로 만든다.
3. 어휘 사전 크기를 정의한다.
4. 문자 시퀀스에 있는 모든 고유문자를 어휘 사전에 추가한다.
5. 가장 빈도수가 큰 기호 쌍을 식별하고, 해당 쌍을 병합해서 어휘 사전에 추가한다.
6. 어휘 사전 크기에 도달할 때 까지 5번 과정을 반복한다.

- 참고 링크
    
    [Byte pair encoding 설명 (BPE tokenizer, BPE 설명, BPE 예시)](https://process-mining.tistory.com/189)
    

### 4-3. 바이트 수준 바이트 쌍 인코딩(BBPE)

- BPE와 동작 방식이 거의 유사하지만, 단어를 문자 시퀀스로 변환하지 않고 바이트 수준 시퀀스로 변환합니다.
- 이를 통해, 다국어에 대해 어휘 사전을 공유할 수 있게 됩니다.

1. Dataset에서 모든 단어를 빈도수와 함께 추출
2. ~~모든 단어를 문자로 나누고 문자 시퀀스로 만든다.~~ → 모든 단어를 문자로 나누고 바이트 수준 시퀀스 만든다.
3. 어휘 사전 크기를 정의한다.
4. 문자 시퀀스에 있는 모든 고유문자를 어휘 사전에 추가한다.
5. 가장 빈도수가 큰 기호 쌍을 식별하고, 해당 쌍을 병합해서 어휘 사전에 추가한다.
6. 어휘 사전 크기에 도달할 때 까지 5번 과정을 반복한다.

### 4-4. 워드 피스

- 워드 피스는 BPE와 유사하게 동작하지만, 한 가지 차이점이 있다.
- BPE에서는 데이터셋에서 단어의 빈도를 추출하고 단어를 문자 시퀀스로 나눈다. 이후, 어휘 사전 크기에 도달할 때 까지 고빈도 기호 쌍을 병합한다.
- 하지만 워드피스는 심볼 쌍을 병합하지 않고, 가능도를 기준으로 병합한다.
- 심볼쌍의 가능도를 계산
    - $argmax(\frac{p(st)}{p(s)p(t)})$

---

## 5. Code Review

### Transformer(직접 구현)

[](https://github.com/dorae222/DeepLearning/blob/main/4.%20Lv4_NLG/simple_nmt/models/transformer.py)

### BERT

- 코드를 자세하게 분석하고 작성하기에 시간이 부족하여, 모델 아키텍처와 코드를 매칭 시켜 이해한대로 편집하였습니다.
- [코드 링크](https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891)
- Full Architecture
    - 토큰화
    - PositionalEmbedding
    - Transformer Encoder
    - BERT(MLM + NSP)

---

- 토큰화
    
    ![[Untitled 45.png]]
    
- PositionalEmbedding
    
    ![[Untitled 46.png]]
    
- Transformer Encoder
    
    ![[Untitled 47.png]]
    
- BERT(MLM + NSP)
    
    ![[Untitled 48.png]]
    
    ![[Untitled 49.png]]
    
    ![[Untitled 50.png]]
    
    ![[Untitled 51.png]]
    

---

- FineTuning 예시 코드(교내 동아리에서 진행된 방언 분류 코드입니다.)
    
    [](https://github.com/dorae222/HAI_Kaggle_Competition/blob/main/v_1_hai_kaggle_summer.ipynb)
    

---

## 6. 논문 리뷰

<aside> 💡 앞의 1~5까지 내용을 보시고 아래 내용을 보시면 아래 논문을 이해하기 쉬울 것 같습니다.

</aside>

[BERT: Pre-training of Deep Bidirectional Transformers for Language...](https://arxiv.org/abs/1810.04805)

### **1. Introduction (서론)**

- BERT(Bidirectional Encoder Representations from Transformers)는 양방향 Transformer 아키텍처를 기반으로 한 모델입니다.
- 사전 학습(pre-training)과 미세 조정(fine-tuning)의 두 단계로 학습되며, 다양한 NLP 작업에 적용할 수 있습니다.

### **2. Related Work (관련 연구)**

### 2.1 Unsupervised Feature-based Approaches

- Word2Vec, GloVe와 같은 비지도 학습(unsupervised learning) 방법을 통해 단어 임베딩을 생성하는 방식이 언급됩니다.
- 이러한 방법은 각 단어를 독립적으로 임베딩하는데, 문맥 정보가 제대로 반영되지 않는다는 한계가 있습니다.
- **단어 혹은 문장의 representation 학습**
    - non-neural method
        
        [Class-Based n-gram Models of Natural Language](https://aclanthology.org/J92-4003/)
        
        [](https://www.jmlr.org/papers/volume6/ando05a/ando05a.pdf)
        
        [Domain Adaptation with Structural Correspondence Learning](https://aclanthology.org/W06-1615/)
        
    - neural method
        
        [One Billion Word Benchmark for Measuring Progress in Statistical...](https://arxiv.org/abs/1312.3005)
        
        [](https://nlp.stanford.edu/pubs/glove.pdf)
        

### 2.2 Unsupervised Fine-tuning Approaches

- ELMo, GPT와 같이 미리 큰 데이터셋으로 사전 학습을 시킨 후, 특정 작업에 대해 미세 조정(fine-tuning)을 하는 방법을 다룹니다. GPT는 특히 Transformer의 디코더를 사용해 언어 모델을 학습시킵니다. 하지만 이는 주로 단방향(왼쪽에서 오른쪽 또는 그 반대) 정보만을 고려합니다.

### 2.3 Transfer Learning from Supervised Data

- 지도 학습(supervised learning)에서 얻은 지식을 다른 작업에 적용하는 전이 학습(transfer learning)에 관한 내용입니다. 예를 들어, 문장 분류 작업에서 학습된 모델을 다른 NLP 작업에도 적용할 수 있습니다. 그러나 이런 방법은 항상 레이블이 있는 데이터가 필요하다는 한계가 있습니다.

### **3. BERT (모델 설명)**

![[Untitled 52.png]]

### 3.1 Pre-training BERT

- **데이터 및 아키텍처**: BERT는 대규모의 언어 데이터(예: Wikipedia)를 사용하여 Transformer의 인코더 구조를 사전 학습합니다. 이때 아키텍처는 다양한 크기를 가질 수 있지만, 대표적으로 BERT-Base와 BERT-Large가 있습니다.
- **Masked Language Model (MLM)**: 전통적인 언어 모델은 단방향(왼쪽에서 오른쪽 또는 그 반대)만을 고려합니다. BERT는 양방향 정보를 고려하기 위해 MLM 작업을 사용합니다. 입력 문장에서 일부 단어를 무작위로 가린 뒤(hidden or 'masked') 이 가려진 단어를 예측하도록 학습됩니다.
- **Next Sentence Prediction (NSP)**: 두 문장이 주어졌을 때, 두 번째 문장이 첫 번째 문장 다음에 오는 문장인지를 판단하는 작업입니다. 이를 통해 모델은 문장 간의 관계를 더 잘 이해할 수 있습니다.

### 3.2 Fine-tuning BERT

- **작업 특화**: 사전 학습된 BERT 모델을 특정 NLP 작업에 맞게 미세 조정합니다. 여기에는 문장 분류, 개체 명명, 질문 응답 등이 포함될 수 있습니다.
- **데이터 및 학습**: 미세 조정은 일반적으로 작은 데이터셋에서도 효과적으로 이루어질 수 있습니다. BERT의 Transformer 인코더는 각 작업의 특성을 캡처할 수 있도록 학습됩니다.
- **양방향성의 이점**: 기존 모델들이 주로 단방향 정보만을 활용했다면, BERT는 양방향 정보를 활용하여 문맥을 더 정확하게 파악합니다. 이러한 문맥 정보는 특히 의미가 애매한 단어나 문장에서 더 정확한 결과를 도출하는 데 도움이 됩니다.

### 4**. Experimental Results (실험 결과)**

- 저도 정리를 하면서 찾은 내용인데, Downstream tasks에 관해 잘 정리된 내용이 있어 첨부합니다.
    
    [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://thejb.ai/bert/)
    

### 4.1 GLUE

- GLUE는 General Language Understanding Evaluation의 약자로 다양한 분야의 general language understanding task를 포함하고 이를 평가하고 있습니다
- 여기에는 문장 분류, 개체 명명 등 다양한 벤치마크가 존재합니다.

![[Untitled 53.png]]

- Table 1
    
    ![[Untitled 54.png]]
    
    - 실험 결과 BERT Base,Large 모두 기존의 방법보다 좋은 성능을 보이고 있다.
        
    - 혹시 벤치마크가 각 어떤 테스크인지 모르실 수도 있을 것 같아 첨부합니다.
        
        [NLP 이해하기](https://hryang06.github.io/nlp/NLP/)
        

### 4.2 SQuAD v1.1

- SQuAD v1.1(Stanford Question Answering Dataset)에서의 성능을 보여줍니다.
- 입력 : Context / question 쌍의 형태로 제공되며
- 출력 : Answer, 정수 쌍으로, Context 내에 포함된 답변 Text의 시작과 끝을 색인화 합니다.
- 주로 Exact Match(EM)과 F1 score 지표로 성능을 측정하며, BERT와 다른 모델들과의 성능 차이를 보여줍니다.

![[Untitled 55.png]]

- Table 2
    
    ![[Untitled 56.png]]
    
- Table 3
    
    ![[Untitled 57.png]]
    

### 4.3 SQuAD v2.0

- SQuAD는 여러 소스들에서 모인 질문/답변 쌍 데이터이다. 하나의 질문과, 그에 대한 답변 문단이 들어있고, 답변 문단 중에서 질문에 대한 답이 되는 구문을 찾는 것이 목적이다.

![[Untitled 58.png]]

- Table 4
    
    ![[Untitled 59.png]]
    

### 4.4 SWAG

- SWAG(Situation With Adversarial Generations)는 문맥을 이해하고 논리적 추론을 하는 능력을 테스트하는 데이터셋입니다.
- 이 테이블에서는 BERT가 얼마나 잘 추론을 하는지 다른 모델과 비교하여 보여줍니다.

![[Untitled 60.png]]

- Table 5
    
    ![[Untitled 61.png]]
    
    - **No NSP**: MLM 사용 / NSP 미사용
    - **LTR & No NSP**: MLM 대신 Left-to-Right 사용 / NLP 미사용
    
    ---
    
    - NSP 태스크를 진행하지 않으면 자연어 추론 태스크(QNLI, MNLI)와 QA 태스크(SQuAD)에서 큰 성능 하락이 있음
    - MLM 대신 LTR이나 BiLSTM을 사용했을 때 MRPC와 SQuAD에서의 성능이 크게 하락함. MLM이 LTR과 BiLSTM보다 훨씬 깊은 양방향성을 띈다.

### 5**. Ablation Studies (성능 분석)**

### 5.1 Effect of Pre-training Tasks

- 사전 훈련 작업의 종류가 모델 성능에 어떻게 영향을 미치는지를 분석합니다.
- 예를 들어, Masked Language Modeling(MLM)과 Next Sentence Prediction(NSP) 같은 다양한 작업을 이용해 어떤 것이 더 유용한지를 평가합니다.

### 5.2 Effect of Model Size

- 모델 크기 (예: 레이어 수, 파라미터 수 등)가 성능에 어떻게 영향을 미치는지 분석합니다.
- 이를 통해 모델 크기가 커질수록 성능이 얼마나 향상되는지, 그리고 언제 그 성능이 더 이상 향상되지 않는지를 확인할 수 있습니다.

### 5.3 Feature-based Approach with BERT

- BERT를 feature-based 방식으로 사용할 경우 성능이 어떻게 변하는지를 살펴봅니다.
    
- 예를 들어, BERT의 출력을 다른 모델의 입력으로 사용하는 경우와 BERT를 end-to-end로 훈련시키는 경우의 성능을 비교합니다.
    
- Table 6
    

![[Untitled 62.png]]

- Table 7

![[Untitled 63.png]]

### 6**. Conclusion (결론)**

- BERT는 사전 학습과 미세 조정을 통해 다양한 NLP 작업에서 뛰어난 성능을 보이며,
- 이를 통해 새로운 연구 및 애플리케이션의 가능성이 확대될 것이라는 결론을 내립니다.

---

## 7. 추후 방향성

- 추가 정리 목표
    - 사전 학습된 BERT 모델 추가 탐색([10. BERT 이후 관련 논문](https://www.notion.so/10-BERT-84c7fcdb22914ab892d65c6ab3d0715b?pvs=21))
    - 사전 학습된 BERT에서 임베딩을 추출하는 방법
    - BERT의 모든 인코더 레이어에서 임베딩을 추출하는 방법
    - 다운스트림 태스크를 위한 BERT 파인 튜닝 방법
    - 다른 언어에 적용하는 법

---

## 8. 참고 링크

- 리스트
    - 아래 링크들을 주요 내용으로 참고하여 내용을 작성하였습니다.
        
        [Word2vec vs BERT](https://medium.com/@ankiit/word2vec-vs-bert-d04ab3ade4c9)
        
        [GitHub - kh-kim/nlp_with_pytorch_examples: 도서 내의 코드들을 모아 놓은 repo입니다.](https://github.com/kh-kim/nlp_with_pytorch_examples)
        
        [GitHub - PacktPublishing/Getting-Started-with-Google-BERT: Getting Started with Google BERT, published by Packt](https://github.com/PacktPublishing/Getting-Started-with-Google-BERT)
        

---

## 9. BERT 이전에 참고할 만한 논문

- 리스트
    - [Semi-supervised sequence tagging with bidirectional language models. : ELMo](https://arxiv.org/abs/1705.00108)
    - [Attention is all you need. : Transformer](https://arxiv.org/abs/1706.03762)
    - [Class-based n-gram models of natural language : non-neural word representation (1)](https://aclanthology.org/J92-4003/)
    - [A framework for learning predictive structures from multiple tasks and unlabeled data : non-neural word representation (2)](https://www.jmlr.org/papers/volume6/ando05a/ando05a.pdf)
    - [Domain adaptation with structural correspondence learning. : non-neural word representation (3)](https://aclanthology.org/W06-1615.pdf)
    - [Distributed representations of words and phrases and their compositionality : neural word representation (1) (word2vec)](https://arxiv.org/abs/1310.4546)
    - [Glove: Global vectors for word representation. : neural word representation (2) (GloVe)](https://nlp.stanford.edu/pubs/glove.pdf)
    - [Semi-supervised sequence learning. : fine-tuning representation (1)](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/44267.pdf)
    - [Universal language model fine-tuning for text classification : fine-tuning representation (2)](https://arxiv.org/abs/1801.06146)
    - [Improving language understanding with Generative Pre-Training. : fine-tuning representation (3) (GPT)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
    - [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data : NLI 를 이용해서 학습](https://arxiv.org/abs/1705.02364)
    - [Learned in translation: Contextualized word vectors. : MT를 이용해서 학습 (CoVe)](https://arxiv.org/abs/1708.00107)
    - [Google’s neural machine translation system: Bridging the gap between human and machine translation. : WordPiece embedding](https://arxiv.org/abs/1609.08144)

---

## 10. BERT 이후 관련 논문

- 파생 모델 1
    - [Multi-Task Deep Neural Networks for Natural Language Understanding](https://arxiv.org/pdf/1901.11504.pdf)
    - [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf)
    - [ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS](https://arxiv.org/pdf/1909.11942.pdf)
    - [ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS](https://arxiv.org/pdf/2003.10555.pdf)
    - [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/pdf/1909.11942.pdf)
- 파생 모델 2(지식 증류 기반)
    - [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)
    - [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)

---

## 11. 한국어 관련 LLM 모음

- 한국어 Task할 때 참고하면 좋아요
    
    [한국어 언어모델 (Korean Pre-trained Language Models) 톺아보기 (1)](https://www.letr.ai/blog/tech-20220908)
    
    [한국어 언어모델 (Korean Pre-trained Language Models) 톺아보기 (2)](https://www.letr.ai/blog/tech-20221124)
    
    - **Encoder-Centric Models: BERT 계열**
        
        ![[Untitled 64.png]]
        
    - **Decoder-Centric Models: GPT 계열**
        
        ![[Untitled 65.png]]
        
    - **Encoder-Decoder Models: Seq2seq 계열**
        
        ![[Untitled 66.png]]
        
- 금융권 한국어 BERT 기반 LM 모델
    
    - 이번 학기에 KB국민은행과 산학 협력 프로젝트를 진행 중입니다.
    - 금융권에서 언어모델을 어떻게 만들고 활용하는지 궁금하신 분들은 참고하시면 좋을 것 같습니다.
    
    [KB-BERT.pdf](https://prod-files-secure.s3.us-west-2.amazonaws.com/462d6414-91e6-4eeb-9ba6-f72fd930483f/ea661402-57cf-44ea-9ecf-e3a4c0c37a1c/KB-BERT.pdf)
    

---

## 12. 리뷰 후기

<aside> 💡 평소 NLP에 관심이 많아 BERT를 자주 마주치곤 했는데, 이번 기회에 자세하게 리뷰를 하면서 다시금 깊게 이해할 수 있어 좋은 경험이었습니다. 그리고 평소 전이 학습 된 모델을 주로 가져와서 쓰다 보니, 놓치는 점이 많았던 것 같았는데 이번 기회로 BERT 계열 모델들을 꾸준히 정리해보고자 합니다.

</aside>