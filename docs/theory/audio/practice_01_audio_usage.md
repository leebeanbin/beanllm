# Audio & Speech 실무 가이드: 음성 처리 시스템

**실무 적용 문서**

---

## 목차

1. [음성 인식 (STT)](#1-음성-인식-stt)
2. [음성 합성 (TTS)](#2-음성-합성-tts)
3. [Audio RAG](#3-audio-rag)

---

## 1. 음성 인식 (STT)

```python
from llmkit.audio_speech import WhisperSTT

stt = WhisperSTT()

# 음성 → 텍스트
result = stt.transcribe("audio.wav")
print(result.text)
```

---

## 2. 음성 합성 (TTS)

```python
from llmkit.audio_speech import TextToSpeech

tts = TextToSpeech()

# 텍스트 → 음성
audio = tts.synthesize("안녕하세요", voice="ko")
audio.save("output.wav")
```

---

## 3. Audio RAG

```python
from llmkit.audio_speech import AudioRAG

rag = AudioRAG()

# 오디오 추가
rag.add_audio("meeting.wav")

# 질문하기
answer = await rag.query("회의에서 논의된 내용은?")
```

---

**작성일**: 2025-01-XX  
**버전**: 1.0

