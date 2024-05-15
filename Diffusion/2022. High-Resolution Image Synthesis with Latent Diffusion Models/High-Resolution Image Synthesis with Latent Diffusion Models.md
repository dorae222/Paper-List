**Title**: High-Resolution Image Synthesis with Latent Diffusion Models
**Venue**: CVPR 2022

**Reviewer**: HyeongJun Do
**Last Updated**: 15.05.2024
**Refence**
> Paper Link: https://arxiv.org/abs/2112.10752
> Github: https://github.com/CompVis/latent-diffusion


## Abstract

- **문제점**: Diffusion models (DMs)는 이미지 데이터를 위한 최신 합성 결과를 제공하지만, 픽셀 공간에서 직접 작동하기 때문에 학습과 추론이 매우 비싸다.
- **해결책**: 강력한 사전 학습된 오토인코더의 잠재 공간에서 DMs를 적용하여 계산 리소스를 줄인다.
- **기여**:
    1. 잠재 공간에서의 DMs는 복잡성과 디테일 보존 사이의 최적점을 달성.
    2. Cross Attention Layer를 도입하여 텍스트나 바운딩 박스와 같은 일반적인 조건 입력을 위한 유연한 생성기.
    3. 새로운 최고 성능을 달성하면서도 계산 요구 사항을 크게 줄임.

## 1. Introduction

- **현황**: 이미지 합성은 최근 컴퓨터 비전 분야에서 가장 큰 발전을 보였지만, 높은 계산 요구 사항을 갖는다.
- **문제점**: 고해상도 합성을 위해서는 수백 또는 수천 GPU일의 계산이 필요.
- **목표**: 제한된 계산 자원으로도 DMs를 학습할 수 있는 방법을 제시하고, 그 품질과 유연성을 유지.

## 2. Related Work

### Generative Models for Image Synthesis

- **GANs**: 높은 해상도 이미지를 효율적으로 샘플링 가능하지만, 최적화가 어려움.
- **Likelihood-based methods**: 최적화가 잘 되지만 샘플 품질이 낮음.
- **Autoregressive Models (ARM)**: 강력한 성능을 보이지만 계산 요구가 높음.
- **Diffusion Models (DM)**: 최근 높은 샘플 품질과 밀도 추정 성능을 보임.

### Two-Stage Image Synthesis

- **VQ-VAEs**: 표현력이 높은 잠재 공간을 학습.
- **VQGANs**: 더 큰 이미지로의 확장 가능.
- **우리의 접근법**: 더 나은 재구성을 위해 압축 수준을 최적화하고 높은 충실도를 보장.

## 3. Method

### 3.1 Perceptual Image Compression

- **모델**: 오토인코더 기반의 압축 모델, 지각적 손실과 패치 기반의 적대적 목적을 결합.
- **구조**: 인코더(E)와 디코더(D)로 구성, 다양한 다운샘플링 팩터(f)를 실험.

### 3.2 Latent Diffusion Models

- **DMs**: 데이터를 점진적으로 denoising하여 학습.
- **목표**: 고주파수의 감지 불가능한 세부 사항을 추상화하고 더 낮은 차원의 공간에서 학습하여 계산 효율성을 높임.

### 3.3 Conditioning Mechanisms

- **Cross Attention**: 다양한 조건 입력을 처리할 수 있는 유연한 생성기.
- **구현**: 언어 프롬프트 등의 입력을 처리하는 도메인 특화 인코더(τθ) 도입.

## 4. Experiments

### 4.1 Perceptual Compression Tradeoffs

- **실험**: 다양한 다운샘플링 팩터(f)를 사용한 클래스 조건 모델의 성능 비교.
- **결과**: LDM-4와 LDM-8이 최고의 성능과 효율성 균형을 달성.

### 4.2 Image Generation with Latent Diffusion

- **데이터셋**: CelebA-HQ, FFHQ, LSUN-Churches, LSUN-Bedrooms.
- **결과**: 새로운 최고 성능을 기록하며 기존 모델들보다 뛰어난 성능을 보임.

### 4.3 Conditional Latent Diffusion

- **텍스트-이미지 합성**: LAION-400M 데이터베이스를 사용한 모델 훈련.
- **결과**: 강력한 AR 및 GAN 기반 방법들을 능가하는 성능.

### 4.4 Super-Resolution with Latent Diffusion

- **이미지 업스케일링**: LDM-SR이 FID에서 경쟁력 있는 성능을 보이며, SR3보다 더 나은 성능을 기록.

### 4.5 Inpainting with Latent Diffusion

- **이미지 인페인팅**: 다양한 접근 방식과의 비교 실험에서 우수한 성능을 보임.

## 5. Limitations & Societal Impact

### Limitations

- **속도**: GAN보다 샘플링 과정이 느림.
- **정밀도**: 고정밀 작업에는 부적합할 수 있음.

### Societal Impact

- **긍정적 영향**: 창의적 응용 가능성 증가.
- **부정적 영향**: 조작된 데이터의 생성 및 확산 용이성 증가, 개인정보 노출 위험성 존재.

## 6. Conclusion

- **결론**: Latent Diffusion Models는 DMs의 학습 및 샘플링 효율성을 크게 개선하면서도 높은 품질을 유지하는 효과적인 방법이다.
- **기여**: 다양한 조건 이미지 합성 작업에서 좋은 결과를 보여주었으며, 특정 작업에 맞춘 아키텍처 없이도 높은 성능을 발휘.