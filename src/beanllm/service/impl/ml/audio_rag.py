"""
Audio RAG - Audio retrieval-augmented generation logic

Extracted from AudioServiceImpl for SRP compliance.
Handles: add_audio, search_audio, get_transcription, list_audios.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

from beanllm.domain.audio import TranscriptionResult
from beanllm.dto.request.ml.audio_request import AudioRequest
from beanllm.dto.response.ml.audio_response import AudioResponse
from beanllm.utils.logging import get_logger

if TYPE_CHECKING:
    from beanllm.domain.embeddings import BaseEmbedding
    from beanllm.service.types import VectorStoreProtocol

logger = get_logger(__name__)


class AudioRAGMixin:
    """
    Mixin providing AudioRAG: transcribe, index, and search audio segments.

    Expects on self (set by AudioServiceImpl):
        _vector_store: Optional VectorStoreProtocol
        _embedding_model: Optional BaseEmbedding
        _transcriptions: Dict[str, TranscriptionResult]
        transcribe: Async method (from AudioTranscriptionMixin)
    """

    _vector_store: Optional[Any] = None
    _embedding_model: Optional[Any] = None
    _transcriptions: Dict[str, TranscriptionResult] = {}

    async def add_audio(self, request: AudioRequest) -> AudioResponse:
        """
        Transcribe audio and add to RAG system.

        Args:
            request: Audio request DTO.

        Returns:
            AudioResponse with transcription field.
        """
        audio = request.audio
        audio_id = request.audio_id
        metadata = request.metadata or {}

        transcribe_request = AudioRequest(
            audio=audio,
            language=request.language,
            task=request.task,
            model=request.model,
            device=request.device,
            extra_params=request.extra_params,
        )
        transcription = await self.transcribe(transcribe_request)  # type: ignore[attr-defined]
        transcription_result = transcription.transcription_result

        if not transcription_result:
            raise ValueError("Transcription failed")

        if audio_id is None:
            if isinstance(audio, (str, Path)):
                audio_id = str(Path(audio).stem)
            else:
                audio_id = f"audio_{len(self._transcriptions)}"

        self._transcriptions[audio_id] = transcription_result

        if self._vector_store is not None and self._embedding_model is not None:
            from beanllm.domain.loaders import Document

            documents = []
            for i, segment in enumerate(transcription_result.segments):
                doc = Document(
                    content=segment.text,
                    metadata={
                        "audio_id": audio_id,
                        "segment_id": i,
                        "start": segment.start,
                        "end": segment.end,
                        "language": segment.language,
                        **metadata,
                    },
                )
                documents.append(doc)

            store = cast(Any, self._vector_store)
            store.add_documents(documents, self._embedding_model)

        return AudioResponse(transcription=transcription_result)

    async def search_audio(self, request: AudioRequest) -> AudioResponse:
        """
        Search audio segments by query.

        Args:
            request: Audio request DTO (query, top_k).

        Returns:
            AudioResponse with search_results field.
        """
        query = request.query or ""
        top_k = request.top_k
        kwargs = request.extra_params or {}

        if self._vector_store is None:
            results = []
            for audio_id, transcription in self._transcriptions.items():
                for i, segment in enumerate(transcription.segments):
                    if query.lower() in segment.text.lower():
                        results.append({"audio_id": audio_id, "segment": segment, "score": 1.0})
            return AudioResponse(search_results=results[:top_k])

        store = cast(Any, self._vector_store)
        search_results = store.search(query, k=top_k, **kwargs)

        results = []
        for result in search_results:
            metadata = result.metadata
            audio_id = metadata.get("audio_id")
            segment_id = metadata.get("segment_id")

            if audio_id in self._transcriptions:
                transcription = self._transcriptions[audio_id]
                segment = transcription.segments[segment_id]
                results.append(
                    {
                        "audio_id": audio_id,
                        "segment": segment,
                        "score": result.score,
                        "text": result.content,
                    }
                )

        return AudioResponse(search_results=results)

    async def get_transcription(self, request: AudioRequest) -> AudioResponse:
        """
        Get transcription by audio_id.

        Args:
            request: Audio request DTO (audio_id required).

        Returns:
            AudioResponse with transcription field.
        """
        audio_id = request.audio_id
        if not audio_id:
            raise ValueError("audio_id is required")

        transcription = self._transcriptions.get(audio_id)
        return AudioResponse(transcription=transcription)

    async def list_audios(self, request: AudioRequest) -> AudioResponse:
        """
        List all stored audio IDs.

        Args:
            request: Audio request DTO.

        Returns:
            AudioResponse with audio_ids field.
        """
        audio_ids = list(self._transcriptions.keys())
        return AudioResponse(audio_ids=audio_ids)
