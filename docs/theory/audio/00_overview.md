# Audio & Speech Theory: 음성 처리의 수학적 기초

**석사 수준 이론 문서**  
**기반**: llmkit AudioSpeech, WhisperSTT 실제 구현 분석

---

## 목차

### Part I: 신호 처리 이론
1. [푸리에 변환과 주파수 분석](#part-i-신호-처리-이론)
2. [STFT와 시간-주파수 표현](#12-stft와-시간-주파수-표현)
3. [MFCC 특징 추출](#13-mfcc-특징-추출)

### Part II: 음성 인식
4. [Whisper 아키텍처](#part-ii-음성-인식)
5. [CTC Loss와 시퀀스 정렬](#42-ctc-loss와-시퀀스-정렬)
6. [Audio RAG 파이프라인](#43-audio-rag-파이프라인)

### Part III: 음성 합성
7. [Text-to-Speech 모델](#part-iii-음성-합성)
8. [Vocoder와 파형 생성](#72-vocoder와-파형-생성)

---

## Part I: 신호 처리 이론

### 1.1 푸리에 변환과 주파수 분석

#### 정의 1.1.1: 푸리에 변환 (Fourier Transform)

**연속 푸리에 변환:**

$$
F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} dt
$$

**이산 푸리에 변환 (DFT):**

$$
X[k] = \sum_{n=0}^{N-1} x[n] e^{-i2\pi kn/N}
$$

**llmkit 구현:**
```python
# audio_speech.py: Line 10-14 (주석에 수학 공식 포함)
"""
Fourier Transform:
F(ω) = ∫_{-∞}^{∞} f(t) e^{-iωt} dt

Discrete Fourier Transform (DFT):
X[k] = Σ_{n=0}^{N-1} x[n] e^{-i2πkn/N}
"""
```

---

### 1.2 STFT와 시간-주파수 표현

#### 정의 1.2.1: Short-Time Fourier Transform (STFT)

**STFT**는 시간에 따른 주파수 변화를 분석합니다:

$$
\text{STFT}\{x[n]\}(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] w[n - m] e^{-i\omega n}
$$

여기서 $w[n]$은 윈도우 함수입니다.

#### 시각적 표현: STFT Spectrogram

```
┌─────────────────────────────────────────────────────────┐
│                  STFT Spectrogram                       │
└─────────────────────────────────────────────────────────┘

시간 →
주파수
  ↑
  │  ████░░░░░░░░░░░░░░░░  (고주파 성분)
  │  ████████░░░░░░░░░░░░
  │  ████████████░░░░░░░░
  │  ████████████████░░░░  (중주파 성분)
  │  ████████████████████
  │  ████████████████████  (저주파 성분)
  │  ████████████████████
  │
  └──────────────────────────────→ 시간

각 픽셀 = 주파수 성분의 강도
██ = 강함, ░░ = 약함
```

#### 구체적 수치 예시

**예시 1.2.1: STFT 계산**

**입력 신호:**
- 샘플링 레이트: 16,000 Hz
- 윈도우 크기: 512 샘플
- 오버랩: 256 샘플

**처리 과정:**

1. **프레임 분할:**
   ```
   프레임 1: 샘플 0-511
   프레임 2: 샘플 256-767  (256 오버랩)
   프레임 3: 샘플 512-1023
   ...
   ```

2. **각 프레임에 FFT 적용:**
   $$
   X[k] = \sum_{n=0}^{511} x[n] w[n] e^{-i2\pi kn/512}
   $$

3. **주파수 빈 계산:**
   $$
   f_k = \frac{k \times 16000}{512} \text{ Hz}
   $$
   - $k=0$: 0 Hz
   - $k=1$: 31.25 Hz
   - $k=256$: 8,000 Hz (Nyquist frequency)

**결과:**
- 시간-주파수 행렬: $[T \times F]$ (T: 프레임 수, F: 주파수 빈 수)
- 예: 1초 오디오 → 약 62 프레임 × 256 주파수 빈

**llmkit 구현:**
```python
# audio_speech.py: Line 16-19 (주석에 수학 공식 포함)
"""
Short-Time Fourier Transform (STFT):
STFT{x[n]}(m, ω) = Σ_{n=-∞}^{∞} x[n] w[n - m] e^{-iωn}

where w[n] is window function
"""
```

---

### 1.3 MFCC 특징 추출

#### 정의 1.3.1: Mel-Frequency Cepstral Coefficients (MFCC)

**Mel 스케일 변환:**

$$
\text{mel}(f) = 2595 \times \log_{10}\left(1 + \frac{f}{700}\right)
$$

**MFCC 추출 단계:**

1. 프레임 분할
2. FFT 적용
3. Mel 필터뱅크
4. 로그 변환
5. DCT → MFCC

**llmkit 구현:**
```python
# audio_speech.py: Line 21-29 (주석에 수학 공식 포함)
"""
Mel-Frequency Cepstral Coefficients (MFCC):
mel(f) = 2595 × log₁₀(1 + f/700)

Steps:
1. Frame signal
2. Apply FFT
3. Mel filterbank
4. Log
5. DCT → MFCC
"""
```

---

## Part II: 음성 인식

### 2.1 Whisper 아키텍처

#### 정의 2.1.1: Whisper 모델

**Whisper**는 음성을 텍스트로 변환합니다:

$$
\text{text} = \text{Whisper}(\text{audio})
$$

**아키텍처:**
- **Encoder**: 오디오 → 특징 벡터
- **Decoder**: 특징 벡터 → 텍스트

**llmkit 구현:**
```python
# audio_speech.py: WhisperSTT
class WhisperSTT:
    """
    Whisper Speech-to-Text: text = Whisper(audio)
    """
    def __init__(self, model: WhisperModel = WhisperModel.BASE):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model.value}")
        self.model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model.value}")
    
    def transcribe(self, audio: Union[str, Path, AudioSegment]) -> TranscriptionResult:
        """
        음성 → 텍스트 변환
        """
        # 오디오 전처리
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=16000)
        
        # Whisper 모델 실행
        with torch.no_grad():
            generated_ids = self.model.generate(inputs["input_features"])
        
        # 텍스트 디코딩
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return TranscriptionResult(text=transcription)
```

---

### 2.2 CTC Loss와 시퀀스 정렬

#### 정의 2.2.1: CTC Loss (Connectionist Temporal Classification)

**CTC Loss**는 시퀀스 정렬 문제를 해결합니다:

$$
\mathcal{L}_{\text{CTC}} = -\log \sum_{\pi \in \mathcal{B}^{-1}(y)} P(\pi | x)
$$

여기서 $\mathcal{B}$는 collapsing 함수 (공백과 반복 제거)입니다.

**llmkit 구현:**
```python
# audio_speech.py: Line 36-39 (주석에 수학 공식 포함)
"""
CTC Loss (Connectionist Temporal Classification):
L_CTC = -log Σ_{π ∈ B^{-1}(y)} P(π|x)

where B is collapsing function (removing blanks and repeats)
"""
```

---

### 2.3 Audio RAG 파이프라인

#### 정의 2.3.1: Audio RAG

**Audio RAG 파이프라인:**

$$
\text{AudioRAG}(Q) = \text{LLM}(Q, \text{Retrieve}(Q, \text{Transcribe}(\mathcal{A})))
$$

**단계별 분해:**

1. **음성 전사**: $T = \text{Whisper}(\mathcal{A})$
2. **임베딩**: $E = \text{Embed}(T)$
3. **저장**: $V = \text{Store}(E)$
4. **검색**: $R = \text{Retrieve}(Q, V, k)$
5. **생성**: $A = \text{LLM}(Q, R)$

**llmkit 구현:**
```python
# audio_speech.py: AudioRAG
class AudioRAG:
    """
    Audio RAG 파이프라인:
    1. Audio → Transcription (Whisper)
    2. Transcription → Embeddings
    3. Store in Vector DB
    4. Query → Retrieve segments
    5. Generate response
    """
    def __init__(self, stt: Optional[WhisperSTT] = None, vector_store=None):
        self.stt = stt or WhisperSTT()
        self.vector_store = vector_store
    
    def add_audio(self, audio: Union[str, Path, AudioSegment]) -> TranscriptionResult:
        """
        1. 음성 전사: T = Whisper(A)
        2. 임베딩: E = Embed(T)
        3. 저장: V = Store(E)
        """
        transcription = self.stt.transcribe(audio)
        embeddings = self.embedding_model.embed_sync([transcription.text])
        self.vector_store.add_texts([transcription.text], embeddings=embeddings)
        return transcription
    
    async def query(self, query: str, k: int = 5) -> str:
        """
        4. 검색: R = Retrieve(Q, V, k)
        5. 생성: A = LLM(Q, R)
        """
        results = self.vector_store.similarity_search(query, k=k)
        context = "\n\n".join([r.document.content for r in results])
        answer = await self.llm.chat([{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }])
        return answer.content
```

---

## Part III: 음성 합성

### 3.1 Text-to-Speech 모델

#### 정의 3.1.1: Text-to-Speech (TTS)

**TTS**는 텍스트를 음성으로 변환합니다:

$$
\text{audio} = \text{TTS}(\text{text}, \text{voice})
$$

**llmkit 구현:**
```python
# audio_speech.py: TextToSpeech
class TextToSpeech:
    """
    Text-to-Speech: audio = TTS(text, voice)
    """
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0
    ) -> AudioSegment:
        """
        텍스트 → 음성 변환
        """
        # TTS 모델 실행 (예: gTTS, pyttsx3 등)
        audio_data = self._generate_audio(text, voice, speed)
        return AudioSegment(
            audio_data=audio_data,
            sample_rate=22050,
            format="wav"
        )
```

---

### 3.2 Vocoder와 파형 생성

#### 정의 3.2.1: Vocoder

**Vocoder**는 특징 벡터를 파형으로 변환합니다:

$$
\text{waveform} = \text{Vocoder}(\text{features})
$$

**llmkit 구현:**
```python
# audio_speech.py: TTS 내부 구현
def _generate_audio(self, text: str, voice: str, speed: float) -> bytes:
    """
    Vocoder: waveform = Vocoder(features)
    """
    # 1. 텍스트 → 특징 벡터 (Mel spectrogram)
    features = self.text_to_features(text, voice)
    
    # 2. 특징 벡터 → 파형 (Vocoder)
    waveform = self.vocoder(features)
    
    # 3. 속도 조절
    waveform = self._adjust_speed(waveform, speed)
    
    return waveform
```

---

## 참고 문헌

1. **Rabiner (1989)**: "A tutorial on hidden Markov models" - 음성 인식 기초
2. **Graves et al. (2006)**: "Connectionist Temporal Classification" - CTC
3. **Radford et al. (2022)**: "Robust Speech Recognition via Large-Scale Weak Supervision" - Whisper

---

**작성일**: 2025-01-XX  
**버전**: 2.0 (석사 수준 확장)
