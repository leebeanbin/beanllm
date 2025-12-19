"""
Audio & Speech 실습 튜토리얼

이 튜토리얼에서는 Whisper STT, TTS, Audio RAG 등
음성 처리 기능을 실습합니다.

Prerequisites:
- Python 3.10+
- openai-whisper, openai 설치
- (Optional) Audio files for testing

Install:
    # Whisper (CPU)
    pip install openai-whisper

    # Whisper (GPU - NVIDIA)
    pip install openai-whisper torch

    # TTS
    pip install openai  # For OpenAI TTS

    # Audio processing
    pip install numpy soundfile pydub

Author: LLMKit Team
"""

import os
import time
from pathlib import Path

# ============================================================================
# Part 1: Audio Basics
# ============================================================================

print("=" * 80)
print("Part 1: Audio Basics")
print("=" * 80)

from llmkit.audio_speech import AudioSegment

# Example 1.1: Create AudioSegment from file
print("\n1.1 Loading audio from file:")

# Note: You need an audio file for this example
# You can create a sample WAV file or use an existing one

# Create a simple test audio file programmatically
try:
    import wave
    import numpy as np

    # Create 1-second 440Hz tone (A note)
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0

    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    # Save as WAV
    test_audio_path = "test_audio.wav"
    with wave.open(test_audio_path, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    print(f"Created test audio: {test_audio_path}")

    # Load with AudioSegment
    audio = AudioSegment.from_file(test_audio_path)
    print(f"Sample rate: {audio.sample_rate} Hz")
    print(f"Duration: {audio.duration:.2f}s")
    print(f"Format: {audio.format}")
    print(f"Channels: {audio.channels}")
    print(f"Data size: {len(audio.audio_data)} bytes")

except ImportError:
    print("NumPy not installed. Skipping audio generation.")
except Exception as e:
    print(f"Error: {e}")


# Example 1.2: Base64 encoding
print("\n1.2 Base64 encoding:")
if 'audio' in locals():
    base64_str = audio.to_base64()
    print(f"Base64 (first 100 chars): {base64_str[:100]}...")
    print(f"Total length: {len(base64_str)} characters")


# ============================================================================
# Part 2: Speech-to-Text with Whisper
# ============================================================================

print("\n" + "=" * 80)
print("Part 2: Speech-to-Text with Whisper")
print("=" * 80)

from llmkit.audio_speech import WhisperSTT, WhisperModel, transcribe_audio

# Example 2.1: Basic transcription
print("\n2.1 Basic Whisper transcription:")

# Check if whisper is installed
try:
    import whisper

    print("Whisper is installed. Starting transcription...")

    # Note: First run will download the model (~140MB for 'base')
    stt = WhisperSTT(model=WhisperModel.BASE)

    # Transcribe the test audio
    if 'test_audio_path' in locals() and os.path.exists(test_audio_path):
        print("Transcribing test audio...")
        result = stt.transcribe(test_audio_path, language='en')

        print(f"\nTranscription: {result.text}")
        print(f"Language: {result.language}")
        print(f"Duration: {result.duration:.2f}s")
        print(f"Model: {result.model}")
        print(f"Number of segments: {len(result.segments)}")

        # Print segments
        if result.segments:
            print("\nSegments:")
            for seg in result.segments:
                print(f"  {seg}")

    else:
        print("⚠️  Test audio file not found. Please provide an audio file.")

except ImportError:
    print("\n⚠️  openai-whisper not installed.")
    print("   Install with: pip install openai-whisper")
    print("   Note: First run will download model weights")


# Example 2.2: Transcribe with different languages
print("\n2.2 Multi-language transcription:")

sample_texts = {
    'en': "This is an English test.",
    'ko': "이것은 한국어 테스트입니다.",
    'es': "Esta es una prueba en español.",
    'fr': "Ceci est un test en français."
}

print("In real scenario, you would transcribe audio in different languages:")
for lang_code, sample_text in sample_texts.items():
    print(f"  {lang_code}: {sample_text}")


# Example 2.3: Translation to English
print("\n2.3 Translation mode (translate to English):")
print("Whisper can translate any language to English using task='translate'")
print("Example: Korean audio → English text")


# Example 2.4: Convenience function
print("\n2.4 Using convenience function:")
print("Quick transcription with transcribe_audio():")

# if os.path.exists('audio.mp3'):
#     result = transcribe_audio('audio.mp3', model='base', language='en')
#     print(result.text)


# ============================================================================
# Part 3: Text-to-Speech
# ============================================================================

print("\n" + "=" * 80)
print("Part 3: Text-to-Speech")
print("=" * 80)

from llmkit.audio_speech import TextToSpeech, TTSProvider, text_to_speech

openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    print("\n3.1 OpenAI TTS:")

    try:
        tts = TextToSpeech(
            provider=TTSProvider.OPENAI,
            api_key=openai_api_key,
            voice='alloy'
        )

        # Synthesize speech
        text = "Hello! This is a test of text to speech synthesis."
        audio = tts.synthesize(text)

        print(f"Generated audio:")
        print(f"  Format: {audio.format}")
        print(f"  Sample rate: {audio.sample_rate} Hz")
        print(f"  Data size: {len(audio.audio_data)} bytes")

        # Save to file
        output_file = "tts_output.mp3"
        audio.to_file(output_file)
        print(f"  Saved to: {output_file}")

    except Exception as e:
        print(f"Error: {e}")


    # Example 3.2: Different voices
    print("\n3.2 OpenAI TTS voices:")

    voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
    print("Available voices:")
    for voice in voices:
        print(f"  - {voice}")

    print("\nExample: Generate with different voice")
    try:
        audio_nova = tts.synthesize("Hello from Nova!", voice='nova')
        audio_nova.to_file("tts_nova.mp3")
        print("Generated tts_nova.mp3")
    except Exception as e:
        print(f"Error: {e}")


    # Example 3.3: Speed control
    print("\n3.3 Speech speed control:")

    speeds = [0.5, 1.0, 1.5, 2.0]
    print("Generating same text at different speeds:")

    for speed in speeds:
        try:
            audio_speed = tts.synthesize(
                "This is a speed test.",
                speed=speed
            )
            filename = f"tts_speed_{speed}.mp3"
            audio_speed.to_file(filename)
            print(f"  Speed {speed}x: {filename}")
        except Exception as e:
            print(f"  Speed {speed}x: Error - {e}")


    # Example 3.4: Convenience function
    print("\n3.4 Convenience function:")

    try:
        audio_quick = text_to_speech(
            "Quick TTS test",
            provider='openai',
            voice='alloy',
            output_file='tts_quick.mp3'
        )
        print("Generated tts_quick.mp3 using convenience function")
    except Exception as e:
        print(f"Error: {e}")

else:
    print("\n⚠️  OPENAI_API_KEY not set.")
    print("   Set environment variable to use OpenAI TTS.")
    print("   Other TTS providers: Google, Azure, ElevenLabs")


# ============================================================================
# Part 4: Audio RAG
# ============================================================================

print("\n" + "=" * 80)
print("Part 4: Audio RAG")
print("=" * 80)

from llmkit.audio_speech import AudioRAG

print("\n4.1 Audio RAG concept:")
print("""
Audio RAG workflow:
1. Transcribe audio files with Whisper
2. Store transcriptions with metadata
3. Search transcriptions by query
4. Retrieve relevant audio segments
5. Use with LLM for Q&A
""")


# Example 4.1: Basic Audio RAG
print("\n4.2 Creating Audio RAG system:")

try:
    import whisper

    # Create Audio RAG instance
    audio_rag = AudioRAG(
        stt=WhisperSTT(model=WhisperModel.BASE)
    )

    print("Audio RAG system created (without vector store)")

    # Add audio
    if 'test_audio_path' in locals() and os.path.exists(test_audio_path):
        print(f"\nAdding audio: {test_audio_path}")

        transcription = audio_rag.add_audio(
            test_audio_path,
            audio_id='test_001',
            metadata={'source': 'generated', 'topic': 'test'}
        )

        print(f"Transcription added:")
        print(f"  Text: {transcription.text}")
        print(f"  Segments: {len(transcription.segments)}")

        # List audios
        print(f"\nStored audios: {audio_rag.list_audios()}")

        # Get transcription
        retrieved = audio_rag.get_transcription('test_001')
        print(f"Retrieved transcription: {retrieved.text[:100]}...")

except ImportError:
    print("Whisper not installed, skipping Audio RAG example")


# Example 4.3: Audio RAG with Vector Store
print("\n4.3 Audio RAG with Vector Store:")

print("""
For production, integrate with vector store:

from llmkit import FAISSVectorStore, OpenAIEmbedding

vector_store = FAISSVectorStore.create(dimension=1536)
embedding = OpenAIEmbedding()

audio_rag = AudioRAG(
    stt=WhisperSTT(model='base'),
    vector_store=vector_store,
    embedding_model=embedding
)

# Now you can search semantically
results = audio_rag.search("topic of interest", top_k=5)
""")


# Example 4.4: Search in transcriptions
print("\n4.4 Searching transcriptions:")

if 'audio_rag' in locals() and audio_rag.list_audios():
    # Simple text search (without vector store)
    query = "test"
    results = audio_rag.search(query, top_k=3)

    print(f"Query: '{query}'")
    print(f"Results: {len(results)}")

    for result in results:
        print(f"\n  Audio ID: {result['audio_id']}")
        print(f"  Segment: {result['segment']}")
        print(f"  Score: {result['score']}")


# ============================================================================
# Part 5: Real-World Applications
# ============================================================================

print("\n" + "=" * 80)
print("Part 5: Real-World Applications")
print("=" * 80)

# Example 5.1: Meeting transcription
print("\n5.1 Meeting Transcription System:")

def transcribe_meeting(audio_file: str, output_file: str = "transcript.txt"):
    """
    Transcribe meeting audio with timestamps

    Args:
        audio_file: Path to audio file
        output_file: Path to save transcript
    """
    print(f"Transcribing meeting: {audio_file}")

    try:
        stt = WhisperSTT(model=WhisperModel.MEDIUM)  # Better accuracy
        result = stt.transcribe(audio_file)

        # Format transcript with timestamps
        transcript_lines = ["Meeting Transcript", "=" * 50, ""]

        for segment in result.segments:
            timestamp = f"[{int(segment.start//60):02d}:{int(segment.start%60):02d}]"
            transcript_lines.append(f"{timestamp} {segment.text}")

        transcript = "\n".join(transcript_lines)

        # Save
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(transcript)

        print(f"Transcript saved to: {output_file}")
        return result

    except Exception as e:
        print(f"Error: {e}")
        return None

print("Usage: transcribe_meeting('meeting.mp3', 'transcript.txt')")


# Example 5.2: Voice assistant
print("\n5.2 Voice Assistant Pattern:")

print("""
Complete voice assistant workflow:

1. Capture audio (microphone)
2. STT: Audio → Text (Whisper)
3. LLM: Generate response
4. TTS: Text → Audio
5. Play audio (speaker)

Code example:

async def voice_assistant(audio_input: str):
    # 1. Transcribe
    stt = WhisperSTT()
    text = stt.transcribe(audio_input).text

    # 2. Generate response
    from llmkit import create_client
    llm = create_client(model='gpt-4o')
    response = llm.chat([{"role": "user", "content": text}])

    # 3. Synthesize
    tts = TextToSpeech(provider='openai')
    audio_response = tts.synthesize(response.content)

    # 4. Play or save
    audio_response.to_file('response.mp3')

    return response.content
""")


# Example 5.3: Podcast search engine
print("\n5.3 Podcast Search Engine:")

class PodcastSearchEngine:
    """
    Search engine for podcast episodes

    Features:
    - Transcribe episodes
    - Index transcriptions
    - Search by keyword or semantic query
    - Return relevant segments with timestamps
    """

    def __init__(self):
        self.audio_rag = AudioRAG(
            stt=WhisperSTT(model=WhisperModel.SMALL)
        )

    def add_episode(self, audio_file: str, episode_info: dict):
        """Add podcast episode"""
        print(f"Adding episode: {episode_info.get('title', 'Unknown')}")

        self.audio_rag.add_audio(
            audio_file,
            audio_id=episode_info.get('id'),
            metadata=episode_info
        )

    def search(self, query: str, top_k: int = 10):
        """Search across all episodes"""
        results = self.audio_rag.search(query, top_k=top_k)

        # Format results
        formatted = []
        for result in results:
            segment = result['segment']
            formatted.append({
                'episode_id': result['audio_id'],
                'timestamp': segment.start,
                'text': segment.text,
                'score': result['score']
            })

        return formatted

print("Usage:")
print("  engine = PodcastSearchEngine()")
print("  engine.add_episode('ep1.mp3', {'id': 'ep1', 'title': 'Episode 1'})")
print("  results = engine.search('machine learning')")


# Example 5.4: Multilingual subtitle generator
print("\n5.4 Multilingual Subtitle Generator:")

def generate_subtitles(
    video_file: str,
    output_format: str = "srt",
    translate_to: str = None
):
    """
    Generate subtitles from video

    Args:
        video_file: Path to video file
        output_format: 'srt', 'vtt', or 'txt'
        translate_to: Target language for translation (optional)

    Returns:
        Path to subtitle file
    """
    print(f"Generating subtitles for: {video_file}")

    try:
        # Extract audio from video (requires ffmpeg)
        # audio_file = extract_audio(video_file)

        # Transcribe
        stt = WhisperSTT(model=WhisperModel.MEDIUM)

        task = 'translate' if translate_to == 'en' else 'transcribe'
        result = stt.transcribe(video_file, task=task)

        # Format as SRT
        if output_format == 'srt':
            srt_content = []
            for i, segment in enumerate(result.segments, 1):
                start = format_timestamp(segment.start)
                end = format_timestamp(segment.end)

                srt_content.append(f"{i}")
                srt_content.append(f"{start} --> {end}")
                srt_content.append(segment.text)
                srt_content.append("")  # Blank line

            subtitle_file = video_file.rsplit('.', 1)[0] + '.srt'
            with open(subtitle_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(srt_content))

            print(f"Subtitles saved: {subtitle_file}")
            return subtitle_file

    except Exception as e:
        print(f"Error: {e}")
        return None

def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

print("Usage: generate_subtitles('video.mp4', output_format='srt')")


# ============================================================================
# Part 6: Performance Comparison
# ============================================================================

print("\n" + "=" * 80)
print("Part 6: Performance Comparison")
print("=" * 80)

# Example 6.1: Model size vs accuracy tradeoff
print("\n6.1 Whisper model comparison:")

model_info = {
    'tiny': {'params': '39M', 'speed': 'Very Fast', 'accuracy': 'Low'},
    'base': {'params': '74M', 'speed': 'Fast', 'accuracy': 'Good'},
    'small': {'params': '244M', 'speed': 'Medium', 'accuracy': 'Better'},
    'medium': {'params': '769M', 'speed': 'Slow', 'accuracy': 'Very Good'},
    'large': {'params': '1550M', 'speed': 'Very Slow', 'accuracy': 'Best'}
}

print("Model  | Params | Speed      | Accuracy")
print("-------|--------|------------|----------")
for model, info in model_info.items():
    print(f"{model:7}| {info['params']:6} | {info['speed']:10} | {info['accuracy']}")

print("\nRecommendation:")
print("  - Real-time apps: tiny or base")
print("  - General use: small or medium")
print("  - High accuracy: large")


# Example 6.2: Benchmarking
print("\n6.2 Benchmarking transcription speed:")

def benchmark_transcription(audio_file: str, models: list):
    """Benchmark different Whisper models"""
    results = []

    for model_name in models:
        print(f"\nTesting {model_name}...")

        try:
            stt = WhisperSTT(model=model_name)

            start = time.time()
            result = stt.transcribe(audio_file)
            elapsed = time.time() - start

            results.append({
                'model': model_name,
                'time': elapsed,
                'duration': result.duration,
                'rtf': elapsed / result.duration if result.duration > 0 else 0
            })

            print(f"  Time: {elapsed:.2f}s")
            print(f"  Audio duration: {result.duration:.2f}s")
            print(f"  RTF: {results[-1]['rtf']:.2f}x")

        except Exception as e:
            print(f"  Error: {e}")

    return results

print("Usage: benchmark_transcription('audio.mp3', ['tiny', 'base', 'small'])")
print("RTF (Real-Time Factor): time_to_transcribe / audio_duration")
print("  RTF < 1.0: Faster than real-time")
print("  RTF = 1.0: Same as real-time")
print("  RTF > 1.0: Slower than real-time")


# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("TUTORIAL SUMMARY")
print("=" * 80)

summary = """
이 튜토리얼에서 다룬 내용:

1. Audio Basics
   - AudioSegment class
   - Loading/saving audio files
   - Base64 encoding

2. Speech-to-Text (Whisper)
   - WhisperSTT class
   - Multi-language transcription
   - Translation to English
   - Segment-level timestamps

3. Text-to-Speech
   - OpenAI TTS
   - Multiple voices
   - Speed control
   - Other providers (Google, Azure, ElevenLabs)

4. Audio RAG
   - Transcription indexing
   - Semantic search in audio
   - Vector store integration

5. Real-World Applications
   - Meeting transcription
   - Voice assistant
   - Podcast search engine
   - Subtitle generation

6. Performance
   - Model size tradeoffs
   - Benchmarking
   - Real-Time Factor (RTF)

Key Concepts:
- Sampling & Quantization
- Fourier Transform
- MFCC features
- CTC loss
- Attention mechanism
- Transformer architecture

다음 단계:
- 실제 오디오 파일로 테스트
- Vector store와 통합
- Production 배포 (API 서버)
- Real-time streaming transcription
"""

print(summary)

print("\n" + "=" * 80)
print("튜토리얼 완료!")
print("=" * 80)

# Cleanup
try:
    if 'test_audio_path' in locals() and os.path.exists(test_audio_path):
        os.remove(test_audio_path)
        print(f"\nCleaned up: {test_audio_path}")
except:
    pass
