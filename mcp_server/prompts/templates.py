"""
Prompt Templates - 재사용 가능한 프롬프트 템플릿

MCP Prompts는 자주 사용되는 프롬프트를 템플릿으로 저장하여 재사용할 수 있게 합니다.
"""
from typing import Dict, Any, List
from fastmcp import FastMCP

# FastMCP 인스턴스
mcp = FastMCP("Prompt Templates")


@mcp.prompt()
def rag_system_builder(
    documents_path: str, collection_name: str = "default"
) -> List[Dict[str, str]]:
    """
    RAG 시스템 구축 워크플로우 프롬프트

    Args:
        documents_path: 문서 경로
        collection_name: 컬렉션 이름

    Returns:
        List[Dict[str, str]]: 프롬프트 메시지 목록

    Usage in Claude:
        "rag_system_builder" 프롬프트를 사용해줘
    """
    return [
        {
            "role": "system",
            "content": "당신은 beanllm RAG 시스템 구축을 돕는 AI 어시스턴트입니다. "
            "사용자가 제공한 문서로 RAG 시스템을 구축하고 질의응답을 할 수 있도록 도와주세요.",
        },
        {
            "role": "user",
            "content": f"""
다음 경로의 문서들로 RAG 시스템을 구축해주세요:
- 경로: {documents_path}
- 컬렉션 이름: {collection_name}

단계:
1. build_rag_system() 도구를 사용하여 RAG 시스템 구축
2. 구축 결과 확인 (문서 개수, 청크 개수)
3. 테스트 질의 실행 (예: "이 문서들의 주제는 무엇인가요?")
4. 결과 요약
""",
        },
    ]


@mcp.prompt()
def multiagent_debate(
    topic: str, agent_count: int = 3
) -> List[Dict[str, str]]:
    """
    다중 에이전트 토론 워크플로우 프롬프트

    Args:
        topic: 토론 주제
        agent_count: 에이전트 개수

    Returns:
        List[Dict[str, str]]: 프롬프트 메시지 목록

    Usage in Claude:
        "multiagent_debate" 프롬프트를 사용해줘
    """
    return [
        {
            "role": "system",
            "content": "당신은 beanllm Multi-Agent 시스템을 활용한 토론 진행자입니다. "
            "다양한 관점의 에이전트들을 만들어 깊이 있는 토론을 진행하세요.",
        },
        {
            "role": "user",
            "content": f"""
다음 주제로 다중 에이전트 토론을 진행해주세요:
- 주제: {topic}
- 에이전트 수: {agent_count}

단계:
1. create_multiagent_system() 도구를 사용하여 {agent_count}명의 에이전트 생성
   - 각 에이전트에게 다른 관점 부여 (예: 낙관론자, 비판론자, 실용주의자)
   - 전략: debate
2. run_multiagent_task()를 사용하여 토론 실행
3. 각 에이전트의 의견 요약
4. 최종 결론 도출
""",
        },
    ]


@mcp.prompt()
def knowledge_graph_explorer(
    documents_path: str, query: str
) -> List[Dict[str, str]]:
    """
    지식 그래프 구축 및 탐색 프롬프트

    Args:
        documents_path: 문서 경로
        query: 탐색 질의

    Returns:
        List[Dict[str, str]]: 프롬프트 메시지 목록

    Usage in Claude:
        "knowledge_graph_explorer" 프롬프트를 사용해줘
    """
    return [
        {
            "role": "system",
            "content": "당신은 beanllm Knowledge Graph 시스템을 활용한 지식 탐색 전문가입니다. "
            "문서에서 엔티티와 관계를 추출하여 구조화된 지식 그래프를 구축하세요.",
        },
        {
            "role": "user",
            "content": f"""
다음 문서들로 지식 그래프를 구축하고 탐색해주세요:
- 문서 경로: {documents_path}
- 탐색 질의: {query}

단계:
1. build_knowledge_graph() 도구를 사용하여 지식 그래프 구축
2. get_kg_stats()로 엔티티/관계 통계 확인
3. query_knowledge_graph()로 질의 실행
4. visualize_knowledge_graph()로 시각화 (선택)
5. 결과 해석 및 인사이트 도출
""",
        },
    ]


@mcp.prompt()
def audio_transcription_batch(
    audio_folder: str, output_format: str = "text"
) -> List[Dict[str, str]]:
    """
    음성 파일 일괄 전사 프롬프트

    Args:
        audio_folder: 음성 파일 폴더 경로
        output_format: 출력 형식 (text, srt, vtt)

    Returns:
        List[Dict[str, str]]: 프롬프트 메시지 목록

    Usage in Claude:
        "audio_transcription_batch" 프롬프트를 사용해줘
    """
    return [
        {
            "role": "system",
            "content": "당신은 beanllm Audio 처리 전문가입니다. "
            "음성 파일을 정확하게 전사하고 필요한 형식으로 변환하세요.",
        },
        {
            "role": "user",
            "content": f"""
다음 폴더의 음성 파일들을 일괄 전사해주세요:
- 폴더: {audio_folder}
- 출력 형식: {output_format}

단계:
1. 폴더 내 모든 .mp3, .wav, .m4a 파일 찾기
2. batch_transcribe_audio() 도구로 일괄 전사
3. 전사 결과를 {output_format} 형식으로 저장
4. 통계 요약 (총 파일 수, 평균 신뢰도, 총 시간)
""",
        },
    ]


@mcp.prompt()
def model_comparison(
    models: List[str], test_prompt: str
) -> List[Dict[str, str]]:
    """
    모델 성능 비교 프롬프트

    Args:
        models: 비교할 모델 목록
        test_prompt: 테스트 프롬프트

    Returns:
        List[Dict[str, str]]: 프롬프트 메시지 목록

    Usage in Claude:
        "model_comparison" 프롬프트를 사용해줘
    """
    models_str = ", ".join(models)

    return [
        {
            "role": "system",
            "content": "당신은 beanllm 모델 평가 전문가입니다. "
            "여러 LLM 모델의 성능을 객관적으로 비교하고 분석하세요.",
        },
        {
            "role": "user",
            "content": f"""
다음 모델들을 비교 평가해주세요:
- 모델: {models_str}
- 테스트 프롬프트: "{test_prompt}"

단계:
1. compare_model_outputs() 도구로 동일한 프롬프트에 대한 응답 비교
2. benchmark_models()로 표준 벤치마크 실행
3. 각 모델의 강점/약점 분석:
   - 응답 품질
   - 응답 속도
   - 토큰 효율성
4. 사용 사례별 추천 모델 제시
""",
        },
    ]


@mcp.prompt()
def google_workspace_exporter(
    chat_content: str, export_type: str = "docs"
) -> List[Dict[str, str]]:
    """
    Google Workspace 내보내기 프롬프트

    Args:
        chat_content: 내보낼 채팅 내용
        export_type: 내보내기 타입 (docs, drive, gmail)

    Returns:
        List[Dict[str, str]]: 프롬프트 메시지 목록

    Usage in Claude:
        "google_workspace_exporter" 프롬프트를 사용해줘
    """
    return [
        {
            "role": "system",
            "content": "당신은 beanllm Google Workspace 통합 전문가입니다. "
            "채팅 내역을 적절한 형식으로 Google 서비스에 저장하세요.",
        },
        {
            "role": "user",
            "content": f"""
다음 채팅 내역을 Google {export_type}로 내보내주세요:

[채팅 내용]
{chat_content[:500]}...

단계:
1. 채팅 내역을 적절한 형식으로 변환 (마크다운)
2. {"export_to_google_docs()" if export_type == "docs" else "save_to_google_drive()" if export_type == "drive" else "share_via_gmail()"}() 도구 사용
3. 내보내기 결과 확인 (URL, ID 등)
4. 사용자에게 결과 링크 제공
""",
        },
    ]


@mcp.prompt()
def rag_optimization(
    collection_name: str = "default"
) -> List[Dict[str, str]]:
    """
    RAG 시스템 최적화 프롬프트

    Args:
        collection_name: 최적화할 RAG 시스템 이름

    Returns:
        List[Dict[str, str]]: 프롬프트 메시지 목록

    Usage in Claude:
        "rag_optimization" 프롬프트를 사용해줘
    """
    return [
        {
            "role": "system",
            "content": "당신은 beanllm RAG 최적화 전문가입니다. "
            "RAG 시스템의 성능을 분석하고 개선 방안을 제시하세요.",
        },
        {
            "role": "user",
            "content": f"""
'{collection_name}' RAG 시스템을 최적화해주세요:

분석 단계:
1. get_rag_stats()로 현재 상태 확인
2. 테스트 질의 실행 및 응답 품질 평가
3. 다음 항목 검토:
   - 청크 크기 적정성
   - 검색 정확도 (top_k 설정)
   - 임베딩 모델 선택
   - 벡터 스토어 타입

최적화 제안:
- Chunk size 조정 (현재 → 제안)
- Top-k 파라미터 튜닝
- HyDE query expansion 적용 검토
- Reranking 추가 검토

예상 효과:
- 검색 정확도 향상: X%
- 응답 품질 개선: Y%
""",
        },
    ]
