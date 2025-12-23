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
# service/impl/audio_service_impl.py: AudioServiceImpl
# facade/audio_facade.py: WhisperSTT
"""
Fourier Transform:
F(ω) = ∫_{-∞}^{∞} f(t) e^{-iωt} dt

Discrete Fourier Transform (DFT):
X[k] = Σ_{n=0}^{N-1} x[n] e^{-i2πkn/N}

Whisper 모델은 내부적으로 FFT를 사용하여 오디오를 주파수 도메인으로 변환합니다.
실제 구현은 openai-whisper 라이브러리의 transcribe() 메서드에서 처리됩니다.
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
- 예: 1초 오디오 (16kHz) → 약 62 프레임 × 256 주파수 빈
- 프레임 수 계산: $T = \frac{\text{샘플 수} - \text{윈도우 크기}}{\text{오버랩}} + 1 = \frac{16000 - 512}{256} + 1 \approx 62$

**llmkit 구현:**
```python
# service/impl/audio_service_impl.py: AudioServiceImpl
"""
Short-Time Fourier Transform (STFT):
STFT{x[n]}(m, ω) = Σ_{n=-∞}^{∞} x[n] w[n - m] e^{-iωn}

where w[n] is window function

Whisper 모델은 내부적으로 STFT를 사용하여 오디오를 시간-주파수 표현으로 변환합니다.
실제 구현은 openai-whisper 라이브러리의 transcribe() 메서드에서 처리됩니다.
"""
```

---

### 1.3 MFCC 특징 추출

#### 정의 1.3.1: Mel-Frequency Cepstral Coefficients (MFCC)

**Mel 스케일 변환:**

$$
\text{mel}(f) = 2595 \times \log_{10}\left(1 + \frac{f}{700}\right)
$$

**역변환 (Hz → Mel):**

$$
f = 700 \times (10^{\text{mel}/2595} - 1)
$$

**MFCC 추출 단계:**

1. **프레임 분할**: 오디오를 짧은 프레임으로 분할 (예: 25ms, 10ms 오버랩)
2. **FFT 적용**: 각 프레임에 FFT 적용하여 주파수 도메인 변환
3. **Mel 필터뱅크**: 주파수 스펙트럼을 Mel 스케일로 변환 (일반적으로 40개 필터)
4. **로그 변환**: 에너지의 로그를 취하여 동적 범위 압축
5. **DCT (Discrete Cosine Transform)**: MFCC 계수 추출 (일반적으로 13개 계수)

**구체적 수치 예시:**

**예시 1.3.1: MFCC 계산**

**입력:**
- 주파수: $f = 1000$ Hz

**Mel 변환:**
$$
\text{mel}(1000) = 2595 \times \log_{10}\left(1 + \frac{1000}{700}\right) = 2595 \times \log_{10}(2.429) \approx 2595 \times 0.385 \approx 999 \text{ mel}
$$

**Mel 필터뱅크 (40개 필터, 0-8000 Hz):**
- 필터 1: 0-200 mel (0-133 Hz)
- 필터 2: 200-400 mel (133-267 Hz)
- ...
- 필터 40: 3800-4000 mel (≈7000-8000 Hz)

**llmkit 구현:**
```python
# service/impl/audio_service_impl.py: AudioServiceImpl
# facade/audio_facade.py: WhisperSTT
"""
Mel-Frequency Cepstral Coefficients (MFCC):
mel(f) = 2595 × log₁₀(1 + f/700)

Steps:
1. Frame signal
2. Apply FFT
3. Mel filterbank
4. Log
5. DCT → MFCC

Whisper 모델은 내부적으로 Mel spectrogram을 사용합니다.
실제 구현은 openai-whisper 라이브러리에서 처리됩니다.
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
- **Encoder**: 오디오 → 특징 벡터 (Transformer 기반)
- **Decoder**: 특징 벡터 → 텍스트 (Transformer 기반)

#### 정의 2.1.2: Whisper Transformer 구조

**Encoder (Audio Transformer):**
- 입력: Mel spectrogram $M \in \mathbb{R}^{T \times F}$ (T: 시간 프레임, F: 주파수 빈)
- 출력: 특징 벡터 $E \in \mathbb{R}^{T \times d}$ (d: 임베딩 차원)
- 구조: Multi-head self-attention + Feed-forward
- 수학적 표현:
  $$
  E = \text{Encoder}(M) = \text{Transformer}_{\text{enc}}(M)
  $$

**Decoder (Text Transformer):**
- 입력: 특징 벡터 $E$ + 이전 토큰들
- 출력: 다음 토큰 확률 $P(\text{token}_t | E, \text{token}_{<t})$
- 구조: Multi-head cross-attention + Self-attention + Feed-forward
- 수학적 표현:
  $$
  \text{text} = \text{Decoder}(E) = \text{Transformer}_{\text{dec}}(E, \text{prefix})
  $$

**Whisper 모델 크기별 파라미터:**

| 모델 | 파라미터 | Encoder Layers | Decoder Layers | Embedding Dim |
|------|----------|----------------|----------------|---------------|
| tiny  | 39M      | 4              | 4              | 384           |
| base  | 74M      | 6              | 6              | 512           |
| small | 244M     | 12             | 12             | 768           |
| medium| 769M     | 24             | 24             | 1024          |
| large | 1550M    | 32             | 32             | 1280          |

**llmkit 구현:**
```python
# facade/audio_facade.py: WhisperSTT
# service/impl/audio_service_impl.py: AudioServiceImpl
class WhisperSTT:
    """
    Whisper Speech-to-Text: text = Whisper(audio)
    
    아키텍처:
    - Encoder: 오디오 → 특징 벡터 (Mel spectrogram → Transformer)
    - Decoder: 특징 벡터 → 텍스트 (Transformer → Token sequence)
    
    수학적 표현:
    E = Encoder(Mel(STFT(audio)))  # 오디오 → 특징 벡터
    text = Decoder(E)               # 특징 벡터 → 텍스트
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
    
    def transcribe(
        self,
        audio: Union[str, Path, AudioSegment, bytes],
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs,
    ) -> TranscriptionResult:
        """
        음성 → 텍스트 변환
        
        Process:
        1. Audio preprocessing (16kHz 샘플링, 정규화)
        2. STFT → Mel spectrogram 변환
        3. Whisper Encoder (Transformer) → 특징 벡터 E ∈ ℝ^(T×d)
        4. Whisper Decoder (Transformer) → 텍스트 토큰
        5. Token decoding → 최종 텍스트
        
        내부적으로 AudioHandler.handle_transcribe() 호출
        실제 Whisper 모델은 openai-whisper 라이브러리 사용
        """
        # 내부 구현은 service/impl/audio_service_impl.py 참조
        pass
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
# service/impl/audio_service_impl.py: AudioServiceImpl
"""
CTC Loss (Connectionist Temporal Classification):
L_CTC = -log Σ_{π ∈ B^{-1}(y)} P(π|x)

where B is collapsing function (removing blanks and repeats)

Whisper는 CTC Loss를 사용하여 학습되었지만, 
실제 추론 시에는 greedy decoding 또는 beam search를 사용합니다.
llmkit은 Whisper의 transcribe() 메서드를 호출하여 이를 처리합니다.

실제 구현:
- domain/audio/enums.py: WhisperModel (tiny, base, small, medium, large, large-v2, large-v3)
- service/impl/audio_service_impl.py: AudioServiceImpl.transcribe()
- facade/audio_facade.py: WhisperSTT.transcribe()
"""
```

#### 정의 2.2.2: Greedy Decoding vs Beam Search

**Greedy Decoding:**
각 시간 스텝에서 가장 높은 확률의 토큰을 선택:
$$
\text{token}_t = \arg\max_{w} P(w | E, \text{token}_{<t})
$$

**장점:**
- 빠른 추론 속도
- 메모리 사용량 적음

**단점:**
- 지역 최적해에 빠질 수 있음
- 긴 시퀀스에서 오류 누적 가능

**Beam Search:**
각 시간 스텝에서 상위 $k$개 후보를 유지:
$$
\text{Beam}_t = \text{TopK}_{w} P(w | E, \text{token}_{<t})
$$

**수학적 표현:**
$$
\text{score}(\text{sequence}) = \sum_{t=1}^{T} \log P(\text{token}_t | E, \text{token}_{<t})
$$

**Beam Search 파라미터:**
- `beam_size`: 유지할 후보 수 (기본값: 5)
- `length_penalty`: 길이 패널티 (기본값: 1.0)
- `temperature`: 샘플링 온도 (기본값: 0.0 = greedy)

**구체적 수치 예시:**

**예시 2.2.1: Greedy vs Beam Search**

**입력 오디오:** "안녕하세요"

**Greedy Decoding:**
```
Step 1: P("안") = 0.95 → 선택
Step 2: P("녕") = 0.90 → 선택
Step 3: P("하") = 0.85 → 선택
Step 4: P("세") = 0.80 → 선택
Step 5: P("요") = 0.90 → 선택
결과: "안녕하세요" (확률: 0.95 × 0.90 × 0.85 × 0.80 × 0.90 ≈ 0.52)
```

**Beam Search (beam_size=3):**
```
Step 1: Top 3 = ["안"(0.95), "않"(0.03), "앞"(0.02)]
Step 2: 각 후보에 대해:
  - "안" + "녕"(0.90) = 0.855
  - "안" + "녕"(0.05) = 0.048
  - "않" + "녕"(0.01) = 0.0003
  Top 3 = ["안녕"(0.855), "안녕"(0.048), ...]
...
최종: "안녕하세요" (확률: 0.52, 더 안정적)
```

**llmkit 구현:**
```python
# service/impl/audio_service_impl.py: AudioServiceImpl.transcribe()
# openai-whisper의 transcribe() 메서드는 beam_size 파라미터 지원
result = self._whisper_model.transcribe(
    audio_path,
    language=language,
    task=task,
    beam_size=5,  # Beam search 크기 (기본값: 5)
    best_of=5,     # 후보 수 (기본값: 5)
    temperature=0.0,  # 0.0 = greedy, >0 = sampling
    **kwargs
)
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
   - 입력: 오디오 파일 $\mathcal{A}$ (예: WAV, MP3)
   - 출력: 전사 텍스트 $T$ (예: "회의에서 논의된 내용은...")
   
2. **임베딩**: $E = \text{Embed}(T)$
   - 입력: 텍스트 $T$
   - 출력: 임베딩 벡터 $E \in \mathbb{R}^d$ (예: $d = 1536$)
   
3. **저장**: $V = \text{Store}(E)$
   - 벡터 데이터베이스에 저장 (Chroma, FAISS, Pinecone 등)
   
4. **검색**: $R = \text{Retrieve}(Q, V, k)$
   - 쿼리 $Q$의 임베딩과 유사도 계산
   - 상위 $k$개 문서 반환
   
5. **생성**: $A = \text{LLM}(Q, R)$
   - 검색된 컨텍스트 $R$와 쿼리 $Q$를 결합하여 답변 생성

**llmkit 구현:**
```python
# facade/audio_facade.py: AudioRAG
class AudioRAG:
    """
    Audio RAG 파이프라인:
    AudioRAG(Q) = LLM(Q, Retrieve(Q, Transcribe(A)))
    
    단계별 분해:
    1. 음성 전사: T = Whisper(A)
    2. 임베딩: E = Embed(T)
    3. 저장: V = Store(E)
    4. 검색: R = Retrieve(Q, V, k)
    5. 생성: A = LLM(Q, R)
    """
    def __init__(
        self,
        audio_service: Optional[IAudioService] = None,
        vector_store: Optional[VectorStoreProtocol] = None,
        embedding_model: Optional[BaseEmbedding] = None,
        client: Optional[Client] = None,
    ):
        self.audio_service = audio_service or AudioServiceImpl()
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.client = client
    
    async def add_audio(
        self, 
        audio: Union[str, Path, AudioSegment, bytes],
        **kwargs
    ) -> TranscriptionResult:
        """
        1. 음성 전사: T = Whisper(A)
        2. 임베딩: E = Embed(T)
        3. 저장: V = Store(E)
        """
        # 1. 전사
        request = AudioRequest(audio=audio, **kwargs)
        response = await self.audio_service.transcribe(request)
        transcription = response.transcription_result
        
        # 2. 임베딩
        if self.embedding_model and self.vector_store:
            embeddings = await self.embedding_model.embed([transcription.text])
            
            # 3. 저장
            await self.vector_store.add_texts(
                [transcription.text],
                embeddings=embeddings,
                metadatas=[{"source": "audio", "timestamp": transcription.segments[0].start if transcription.segments else None}]
            )
        
        return transcription
    
    async def query(self, query: str, k: int = 5) -> str:
        """
        4. 검색: R = Retrieve(Q, V, k)
        5. 생성: A = LLM(Q, R)
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        # 4. 검색
        results = await self.vector_store.similarity_search(query, k=k)
        context = "\n\n".join([r.page_content for r in results])
        
        # 5. 생성
        if not self.client:
            raise ValueError("LLM client not initialized")
        
        answer = await self.client.chat([{
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

#### 정의 3.1.2: TTS 파이프라인 수학적 모델

**전체 TTS 파이프라인:**

$$
\text{audio} = \text{PostProcess}(\text{Vocoder}(\text{TextEncoder}(\text{text})))
$$

**단계별 분해:**

1. **텍스트 인코딩:**
   $$
   T = \text{TextEncoder}(\text{text}) \in \mathbb{R}^{L \times d_t}
   $$
   - $L$: 텍스트 길이 (토큰 수)
   - $d_t$: 텍스트 임베딩 차원

2. **Mel Spectrogram 생성:**
   $$
   M = \text{MelGenerator}(T) \in \mathbb{R}^{T \times F}
   $$
   - $T$: 시간 프레임 수
   - $F$: Mel 주파수 빈 수 (일반적으로 80)

3. **Vocoder (파형 생성):**
   $$
   W = \text{Vocoder}(M) \in \mathbb{R}^{S}
   $$
   - $S$: 샘플 수 (예: 24kHz × duration)

4. **후처리:**
   $$
   \text{audio} = \text{PostProcess}(W)
   $$
   - 샘플링 레이트 조정
   - 포맷 변환 (WAV, MP3 등)

**구체적 수치 예시:**

**예시 3.1.1: TTS 파이프라인**

**입력:** "안녕하세요" (5자)

**1. 텍스트 인코딩:**
- 토큰화: ["안", "녕", "하", "세", "요"] (L=5)
- 임베딩: $T \in \mathbb{R}^{5 \times 512}$ (d_t=512)

**2. Mel Spectrogram:**
- 시간 프레임: $T = 50$ (약 2초, 25ms 프레임)
- Mel 빈: $F = 80$
- 출력: $M \in \mathbb{R}^{50 \times 80}$

**3. Vocoder:**
- 샘플링 레이트: 24,000 Hz
- 길이: 2초
- 샘플 수: $S = 24,000 \times 2 = 48,000$
- 출력: $W \in \mathbb{R}^{48,000}$

**llmkit 구현:**
```python
# facade/audio_facade.py: TextToSpeech
# service/impl/audio_service_impl.py: AudioServiceImpl
class TextToSpeech:
    """
    Text-to-Speech: audio = TTS(text, voice)
    
    지원하는 Provider:
    - OpenAI TTS (tts-1, tts-1-hd)
    - Google Cloud TTS
    - Azure TTS
    - ElevenLabs TTS
    
    수학적 표현:
    Mel = TextEncoder(text)        # 텍스트 → Mel spectrogram
    waveform = Vocoder(Mel)         # Mel spectrogram → 파형
    audio = PostProcess(waveform)   # 후처리 (샘플링 레이트, 포맷)
    """
    def __init__(
        self,
        provider: TTSProvider = TTSProvider.OPENAI,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        voice: Optional[str] = None,
    ):
        """
        Args:
            provider: TTS 제공자 (OPENAI, GOOGLE, AZURE, ELEVENLABS)
            api_key: API 키
            model: 모델 이름 (예: "tts-1", "tts-1-hd")
            voice: 음성 ID (예: "alloy", "echo", "fable", "onyx", "nova", "shimmer")
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.voice = voice
        # 내부적으로 AudioHandler와 AudioService 사용
    
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        **kwargs,
    ) -> AudioSegment:
        """
        텍스트 → 음성 변환
        
        Process:
        1. 텍스트 전처리 (토큰화, 정규화)
        2. TTS 모델 실행 (Provider별로 다름)
           - OpenAI: Neural vocoder 사용
           - Google: WaveNet 또는 Tacotron
        3. 오디오 후처리 (샘플링 레이트 조정, 포맷 변환)
        
        내부적으로 AudioHandler.handle_synthesize() 사용
        """
        # 내부 구현은 service/impl/audio_service_impl.py 참조
        pass
```

---

### 3.2 Vocoder와 파형 생성

#### 정의 3.2.1: Vocoder

**Vocoder**는 특징 벡터를 파형으로 변환합니다:

$$
\text{waveform} = \text{Vocoder}(\text{features})
$$

#### 정의 3.2.2: Neural Vocoder (WaveNet 기반)

**WaveNet Vocoder**는 확률적 생성 모델입니다:

$$
P(\text{waveform} | M) = \prod_{t=1}^{T} P(w_t | w_{<t}, M)
$$

여기서:
- $M$: Mel spectrogram (조건)
- $w_t$: 시간 $t$의 샘플
- $w_{<t}$: 이전 샘플들

**WaveNet 구조:**
- **Dilated Convolutions**: 다양한 시간 스케일 포착
- **Residual Connections**: 깊은 네트워크 학습
- **Skip Connections**: 고주파 성분 보존

**수학적 표현:**
$$
\text{output} = \text{ReLU}(\text{Conv1D}(\text{input})) + \text{input}
$$

**구체적 수치 예시:**

**예시 3.2.1: Vocoder 계산**

**입력 Mel Spectrogram:**
- $M \in \mathbb{R}^{50 \times 80}$ (50 프레임, 80 Mel 빈)

**WaveNet 처리:**
- Dilated convolutions: [1, 2, 4, 8, 16, 32] (6 레이어)
- 각 레이어: 256 채널
- 출력: $W \in \mathbb{R}^{48,000}$ (24kHz × 2초)

**샘플링:**
- 각 시간 스텝 $t$에서:
  $$
  w_t \sim \text{Categorical}(P(w_t | w_{<t}, M))
  $$
- 256 레벨 양자화 (8-bit)

**llmkit 구현:**
```python
# service/impl/audio_service_impl.py: AudioServiceImpl._synthesize_openai
async def _synthesize_openai(
    self, 
    text: str, 
    voice: str, 
    speed: float
) -> bytes:
    """
    Vocoder: waveform = Vocoder(features)
    
    OpenAI TTS API는 내부적으로 다음 과정을 수행합니다:
    1. 텍스트 → 특징 벡터 (Mel spectrogram)
       - Transformer 기반 텍스트 인코더
       - Mel spectrogram 생성기
    2. 특징 벡터 → 파형 (Vocoder)
       - Neural vocoder (WaveNet 또는 유사)
       - 샘플링 레이트: 24kHz
    3. 속도 조절
       - 시간 스트레칭/압축
    """
    from openai import OpenAI
    
    client = OpenAI(api_key=self.api_key)
    
    response = client.audio.speech.create(
        model=self.model or "tts-1",
        voice=voice or "alloy",
        input=text,
        speed=speed,
    )
    
    return response.content  # WAV bytes (24kHz, 16-bit PCM)
```

---

## Part IV: 오디오 전처리와 후처리

### 4.1 오디오 전처리

#### 정의 4.1.1: 오디오 정규화

**오디오 정규화**는 신호의 진폭을 정규화합니다:

$$
x_{\text{norm}}[n] = \frac{x[n]}{\max(|x[n]|)}
$$

**샘플링 레이트 변환:**

Whisper는 16kHz 샘플링 레이트를 요구합니다:

$$
x_{\text{resampled}}[m] = \text{Resample}(x[n], f_s, 16000)
$$

여기서 $f_s$는 원본 샘플링 레이트입니다.

**구체적 수치 예시:**

**예시 4.1.1: 오디오 전처리**

**입력 오디오:**
- 샘플링 레이트: 44,100 Hz (CD 품질)
- 길이: 5초
- 샘플 수: 220,500

**1. 리샘플링 (44.1kHz → 16kHz):**
- 다운샘플링 비율: $r = \frac{44100}{16000} = 2.75625$
- 출력 샘플 수: $\frac{220,500}{2.75625} \approx 80,000$
- 길이: 5초 (유지)

**2. 정규화:**
- 최대 진폭: $\max(|x[n]|) = 0.8$
- 정규화: $x_{\text{norm}}[n] = \frac{x[n]}{0.8}$

**llmkit 구현:**
```python
# service/impl/audio_service_impl.py: AudioServiceImpl.transcribe()
# openai-whisper는 자동으로 전처리 수행:
# 1. 리샘플링 (16kHz로 변환)
# 2. 정규화 ([-1, 1] 범위)
# 3. Mel spectrogram 변환
result = self._whisper_model.transcribe(audio_path, **options)
```

---

### 4.2 Audio RAG 실제 사용 예시

#### 예시 4.2.1: 회의록 Audio RAG

**시나리오:** 회의 오디오에서 특정 주제 검색

**1. 오디오 추가:**
```python
from llmkit import AudioRAG, WhisperSTT, ChromaVectorStore, OpenAIEmbedding

# AudioRAG 초기화
rag = AudioRAG(
    stt=WhisperSTT(model="base"),
    vector_store=ChromaVectorStore(),
    embedding_model=OpenAIEmbedding()
)

# 회의 오디오 추가
transcription = rag.add_audio("meeting_2024_01_15.wav")
# 출력: TranscriptionResult(
#   text="오늘 회의에서는 프로젝트 일정과 예산에 대해 논의했습니다...",
#   segments=[...],
#   duration=3600.0  # 1시간
# )
```

**2. 검색:**
```python
# 특정 주제 검색
results = rag.search("예산은 얼마인가요?", top_k=3)
# 출력: 관련 오디오 세그먼트들 (타임스탬프 포함)
```

**수학적 표현:**
- 전사: $T = \text{Whisper}(A)$
- 임베딩: $E = \text{Embed}(T)$
- 검색: $R = \text{Retrieve}(Q, E, k=3)$
- 답변: $A = \text{LLM}(Q, R)$

---

## 참고 문헌

1. **Rabiner (1989)**: "A tutorial on hidden Markov models" - 음성 인식 기초
2. **Graves et al. (2006)**: "Connectionist Temporal Classification" - CTC
3. **Radford et al. (2022)**: "Robust Speech Recognition via Large-Scale Weak Supervision" - Whisper
4. **van den Oord et al. (2016)**: "WaveNet: A Generative Model for Raw Audio" - Neural Vocoder
5. **Shen et al. (2018)**: "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions" - Tacotron

---

**작성일**: 2025-01-XX  
**버전**: 3.0 (석사 수준 확장 + 상세 예시)
