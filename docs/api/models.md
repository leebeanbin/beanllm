# Supported Models

beanllm이 지원하는 모든 모델 목록.

`Client(model=<model_id>)`에 사용할 수 있는 모델 ID입니다.

---

## OpenAI

설치: `pip install beanllm[openai]`
환경변수: `OPENAI_API_KEY`

| Model ID | Context | Notes |
|----------|---------|-------|
| `gpt-4o` | 128k | 최고 성능. 비용 높음 |
| `gpt-4o-mini` | 128k | 권장. 성능·비용 균형 |
| `gpt-4-turbo` | 128k | 이전 세대 고성능 |
| `gpt-3.5-turbo` | 16k | 저비용, 간단한 작업 |
| `o1` | 200k | 추론 특화. temperature 미지원 |
| `o1-mini` | 128k | 경량 추론 모델 |
| `o3-mini` | 200k | 최신 추론 모델 |

임베딩:

| Model ID | Dimensions | Notes |
|----------|-----------|-------|
| `text-embedding-3-small` | 1536 | 저비용 임베딩 |
| `text-embedding-3-large` | 3072 | 고성능 임베딩 |
| `text-embedding-ada-002` | 1536 | 구세대 (하위호환) |

---

## Claude (Anthropic)

설치: `pip install beanllm[anthropic]`
환경변수: `ANTHROPIC_API_KEY`

| Model ID | Context | Notes |
|----------|---------|-------|
| `claude-opus-4-5` | 200k | 최고 성능 |
| `claude-sonnet-4-6` | 200k | 권장. 성능·속도 균형 |
| `claude-haiku-3-5` | 200k | 빠른 응답, 저비용 |
| `claude-3-5-sonnet-20241022` | 200k | 직전 세대 Sonnet |
| `claude-3-haiku-20240307` | 200k | 직전 세대 Haiku |

---

## Gemini (Google)

설치: `pip install beanllm[gemini]`
환경변수: `GEMINI_API_KEY`

| Model ID | Context | Notes |
|----------|---------|-------|
| `gemini-2.0-flash` | 1M | 빠른 응답, 비용 효율 |
| `gemini-2.0-flash-thinking-exp` | 1M | 추론 모드 실험 |
| `gemini-1.5-pro` | 1M | 최대 컨텍스트 |
| `gemini-1.5-flash` | 1M | 빠른 응답 |
| `gemini-1.5-flash-8b` | 1M | 최경량 |

---

## Grok (xAI)

설치: `pip install beanllm[all]`
환경변수: `XAI_API_KEY`

| Model ID | Context | Notes |
|----------|---------|-------|
| `grok-3` | 131k | 최고 성능 |
| `grok-3-mini` | 131k | 경량, 추론 지원 |
| `grok-beta` | 131k | 실험적 |
| `grok-vision-beta` | 8k | 비전 입력 지원 |

---

## DeepSeek

설치: `pip install beanllm[all]`
환경변수: `DEEPSEEK_API_KEY`

| Model ID | Context | Notes |
|----------|---------|-------|
| `deepseek-chat` | 64k | 범용. 비용 매우 저렴 |
| `deepseek-reasoner` | 64k | R1 추론 모델. reasoning 토큰 반환 |

---

## Perplexity

설치: `pip install beanllm[all]`
환경변수: `PERPLEXITY_API_KEY`

| Model ID | Context | Notes |
|----------|---------|-------|
| `llama-3.1-sonar-large-128k-online` | 127k | 웹 검색 통합, 고성능 |
| `llama-3.1-sonar-small-128k-online` | 127k | 웹 검색 통합, 저비용 |
| `llama-3.1-sonar-huge-128k-online` | 127k | 웹 검색 통합, 최고 성능 |

---

## Ollama (로컬)

설치: 별도 SDK 불필요 (기본 포함)
환경변수: `OLLAMA_HOST` (선택, 기본 `http://localhost:11434`)

Ollama 레지스트리의 모든 모델을 사용할 수 있습니다.

```bash
# 모델 다운로드
ollama pull llama3
ollama pull mistral
ollama pull qwen2.5
ollama pull gemma2
ollama pull phi3
```

| 권장 Model ID | Parameters | Notes |
|--------------|-----------|-------|
| `llama3` | 8B | Meta LLaMA 3. 범용 |
| `llama3:70b` | 70B | 고성능 (GPU 필요) |
| `mistral` | 7B | 빠른 응답 |
| `qwen2.5` | 7B | 한국어 포함 다국어 강함 |
| `qwen2.5:14b` | 14B | 고성능 다국어 |
| `gemma2` | 9B | Google Gemma 2 |
| `phi3` | 3.8B | Microsoft, 경량·빠름 |
| `deepseek-r1` | 7B | 로컬 추론 모델 |

```python
# 로컬 Ollama 사용
client = Client(model="llama3", provider="ollama")
```

---

## HuggingFace

설치: `pip install beanllm[all]`
환경변수: `HUGGINGFACE_API_KEY`

HuggingFace Inference API를 통해 Hub의 모든 모델을 사용할 수 있습니다.

```python
client = Client(model="meta-llama/Llama-3.1-8B-Instruct", provider="huggingface")
```

---

## ModelParameterStrategy — 파라미터 지원 표

| Provider | temperature | max_tokens | top_p | stream | Notes |
|----------|-------------|-----------|-------|--------|-------|
| OpenAI (GPT-4o, 3.5) | O | O | O | O | — |
| OpenAI (o1, o3) | X | O | X | X | reasoning 모델 |
| Claude | O | O | O | O | system은 별도 파라미터 |
| Gemini | O | O | O | O | — |
| Grok | O | O | O | O | — |
| DeepSeek (chat) | O | O | O | O | — |
| DeepSeek (reasoner) | X | O | X | O | thinking 토큰 |
| Perplexity | O | O | O | O | 웹 검색 자동 통합 |
| Ollama | O | O | O | O | 모델마다 상이 |
| HuggingFace | O | O | O | 모델별 상이 | — |

`X`: 해당 파라미터를 API가 지원하지 않음. `ModelParameterStrategy`가 자동으로 제거합니다.

---

## 관련 문서

- [wiki/providers.md](../../wiki/providers.md) — 프로바이더별 상세 설정
- [docs/api/client.md](client.md) — Client 레퍼런스
