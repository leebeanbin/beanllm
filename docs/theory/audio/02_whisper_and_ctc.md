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

**llmkit 구현:**
```python
# facade/audio_facade.py: WhisperSTT
# service/impl/audio_service_impl.py: AudioServiceImpl.transcribe()
# handler/audio_handler.py: AudioHandler.handle_transcribe()
from typing import Union, Optional
from pathlib import Path
import asyncio

class WhisperSTT:
    """
    Whisper Speech-to-Text: text = Whisper(audio)
    
    아키텍처:
    - Encoder: 오디오 → 특징 벡터 (Mel spectrogram → Transformer)
    - Decoder: 특징 벡터 → 텍스트 (Transformer → Token sequence)
    
    수학적 표현:
    E = Encoder(Mel(STFT(audio)))  # 오디오 → 특징 벡터
    text = Decoder(E)               # 특징 벡터 → 텍스트
    
    실제 구현:
    - facade/audio_facade.py: WhisperSTT (사용자 API)
    - service/impl/audio_service_impl.py: AudioServiceImpl.transcribe() (비즈니스 로직)
    - handler/audio_handler.py: AudioHandler.handle_transcribe() (입력 검증)
    - openai-whisper 라이브러리 사용
    """
    def __init__(
        self,
        model: Union[str, WhisperModel] = WhisperModel.BASE,
        device: Optional[str] = None,
        language: Optional[str] = None,
    ):
        """
        Args:
            model: Whisper 모델 크기 ('tiny', 'base', 'small', 'medium', 'large')
            device: 디바이스 ('cpu', 'cuda', 'mps')
            language: 언어 지정 (None이면 자동 감지)
        """
        self.model_name = model
        self.device = device
        self.language = language
        # 내부적으로 AudioHandler와 AudioService 사용
        self._init_services()
    
    def transcribe(
        self,
        audio: Union[str, Path, AudioSegment, bytes],
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs,
    ) -> TranscriptionResult:
        """
        음성 → 텍스트 변환: text = Whisper(audio)
        
        Process:
        1. Audio preprocessing (16kHz 샘플링, 정규화)
        2. STFT → Mel spectrogram 변환
        3. Whisper Encoder (Transformer) → 특징 벡터 E ∈ ℝ^(T×d)
        4. Whisper Decoder (Transformer) → 텍스트 토큰
        5. Token decoding → 최종 텍스트
        
        실제 구현:
        - facade/audio_facade.py: WhisperSTT.transcribe()
        - service/impl/audio_service_impl.py: AudioServiceImpl.transcribe()
        - openai-whisper 라이브러리 사용
        """
        # 동기 메서드이지만 내부적으로는 비동기 사용
        response = asyncio.run(
            self._audio_handler.handle_transcribe(
                audio=audio,
                language=language or self.language,
                task=task,
                model=self.model_name,
                device=self.device,
                **kwargs,
            )
        )
        return response.transcription_result
```

**AudioServiceImpl 구현:**
```python
# service/impl/audio_service_impl.py: AudioServiceImpl.transcribe()
class AudioServiceImpl(IAudioService):
    """
    Audio 서비스 구현체: Whisper 전사
    
    실제 구현:
    - service/impl/audio_service_impl.py: AudioServiceImpl
    - openai-whisper 라이브러리 사용
    """
    async def transcribe(self, request: AudioRequest) -> AudioResponse:
        """
        음성 전사: text = Whisper(audio)
        
        실제 구현:
        - service/impl/audio_service_impl.py: AudioServiceImpl.transcribe()
        - openai-whisper의 transcribe() 메서드 사용
        """
        self._load_whisper_model()
        
        # 오디오 준비
        audio_path = self._prepare_audio(request.audio)
        
        # Whisper 전사 실행
        result = self._whisper_model.transcribe(
            audio_path,
            language=request.language,
            task=request.task,
            **request.extra_params or {}
        )
        
        return AudioResponse(
            transcription_result=TranscriptionResult(
                text=result["text"],
                language=result.get("language"),
                segments=result.get("segments", [])
            )
        )
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

