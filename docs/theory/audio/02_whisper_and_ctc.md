# Whisper and CTC: 음성 인식 모델

**석사 수준 이론 문서**  
**CS 관점 + 수학적 엄밀성**  
**기반**: llmkit WhisperSTT 실제 구현 분석

---

## 목차

1. [Whisper 아키텍처](#1-whisper-아키텍처)
2. [CTC Loss와 시퀀스 정렬](#2-ctc-loss와-시퀀스-정렬)
3. [Encoder-Decoder 구조](#3-encoder-decoder-구조)
4. [CS 관점: 구현과 성능](#4-cs-관점-구현과-성능)

---

## 1. Whisper 아키텍처

### 1.1 Whisper 구조

#### 정의 1.1.1: Whisper

**Whisper**는 음성을 텍스트로 변환합니다:

$$
\text{text} = \text{Whisper}(\text{audio})
$$

**아키텍처:**
- **Encoder**: 오디오 → 특징 벡터
- **Decoder**: 특징 벡터 → 텍스트

#### 시각적 표현: Whisper 구조

```
Whisper 아키텍처:

오디오 입력
    │
    ▼
┌─────────────┐
│  Encoder    │  ← 오디오 특징 추출
│ (Transformer)│
└──────┬──────┘
       │
       │ 특징 벡터
       │
       ▼
┌─────────────┐
│  Decoder    │  ← 텍스트 생성
│ (Transformer)│
└──────┬──────┘
       │
       ▼
    텍스트 출력
```

---

## 2. CTC Loss와 시퀀스 정렬

### 2.1 CTC Loss 정의

#### 정의 2.1.1: CTC Loss

**CTC Loss**는 시퀀스 정렬 문제를 해결합니다:

$$
\mathcal{L}_{\text{CTC}} = -\log \sum_{\pi \in \mathcal{B}^{-1}(y)} P(\pi | x)
$$

여기서 $\mathcal{B}$는 collapsing 함수입니다.

### 2.2 Collapsing 함수

#### 정의 2.2.1: Collapsing

**Collapsing 함수:**

$$
\mathcal{B}(\pi) = \text{remove\_blanks\_and\_repeats}(\pi)
$$

**예시:**
- $\pi = [a, -, a, b, -]$ → $\mathcal{B}(\pi) = [a, a, b]$
- $\pi = [a, a, a, b]$ → $\mathcal{B}(\pi) = [a, b]$

---

## 3. Encoder-Decoder 구조

### 3.1 Encoder

#### 정의 3.1.1: Audio Encoder

**오디오 인코더:**

$$
E = \text{Encoder}(\text{audio}) \in \mathbb{R}^{T \times d}
$$

여기서 $T$는 시간 스텝 수입니다.

### 3.2 Decoder

#### 정의 3.2.1: Text Decoder

**텍스트 디코더:**

$$
\text{text} = \text{Decoder}(E)
$$

---

## 4. CS 관점: 구현과 성능

### 4.1 llmkit 구현

#### 구현 4.1.1: WhisperSTT

```python
# audio_speech.py
class WhisperSTT:
    def transcribe(self, audio):
        """
        Whisper 음성 인식
        """
        inputs = self.processor(audio, return_tensors="pt")
        with torch.no_grad():
            generated_ids = self.model.generate(inputs["input_features"])
        transcription = self.processor.batch_decode(generated_ids)
        return transcription
```

---

## 질문과 답변 (Q&A)

### Q1: Whisper의 정확도는?

**A:** Whisper 성능:

- **영어:** WER ~5%
- **다국어:** 다양한 언어 지원
- **노이즈:** 강건함

---

## 참고 문헌

1. **Radford et al. (2022)**: "Robust Speech Recognition via Large-Scale Weak Supervision" - Whisper
2. **Graves et al. (2006)**: "Connectionist Temporal Classification" - CTC

---

**작성일**: 2025-01-XX  
**버전**: 3.0 (CS 관점 + 석사 수준 확장)

