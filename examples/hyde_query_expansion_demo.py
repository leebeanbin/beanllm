"""
HyDE Query Expansion Demo

HyDE (Hypothetical Document Embeddings)를 사용한 쿼리 확장 예제입니다.

HyDE는 쿼리와 문서 간의 의미적 갭을 해소하여 검색 품질을 30-40% 향상시킵니다.

Requirements:
    pip install openai  # or anthropic, google-generativeai
"""

from typing import List

from beanllm.domain.embeddings import OpenAIEmbedding
from beanllm.domain.retrieval import HybridRetriever, HyDEExpander


def create_llm_function():
    """
    LLM 함수 생성 (OpenAI 예제)

    다른 LLM 사용 가능:
    - Claude (anthropic)
    - Gemini (google)
    - Ollama (로컬)
    """
    try:
        from openai import OpenAI

        client = OpenAI()

        def llm_generate(prompt: str) -> str:
            """OpenAI LLM으로 가상 문서 생성"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512,
            )
            return response.choices[0].message.content

        return llm_generate

    except ImportError:
        print("OpenAI not installed. Install with: pip install openai")
        return None


def demo_hyde_basic():
    """
    HyDE 기본 사용법

    1. HyDE로 쿼리 확장
    2. 확장된 쿼리로 검색
    """
    print("=" * 60)
    print("HyDE Basic Demo")
    print("=" * 60)

    # LLM 함수 생성
    llm_function = create_llm_function()
    if not llm_function:
        return

    # HyDE Expander 생성
    expander = HyDEExpander(
        llm_function=llm_function,
        num_documents=1,
        temperature=0.7,
    )

    # 쿼리 확장
    query = "What is machine learning?"
    print(f"\n원본 쿼리: {query}")

    hypothetical_doc = expander.expand(query)
    print(f"\n가상 문서:\n{hypothetical_doc}")
    print("\n" + "=" * 60)


def demo_hyde_with_retrieval():
    """
    HyDE + HybridRetriever 사용

    1. 문서 컬렉션 준비
    2. HyDE로 쿼리 확장
    3. 확장된 쿼리로 검색
    """
    print("=" * 60)
    print("HyDE + HybridRetriever Demo")
    print("=" * 60)

    # 문서 컬렉션
    documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers to process complex patterns.",
        "Natural language processing helps computers understand and generate human language.",
        "Computer vision enables machines to interpret and understand visual information.",
        "Reinforcement learning teaches agents to make decisions through trial and error.",
        "Supervised learning uses labeled data to train predictive models.",
        "Unsupervised learning finds patterns in unlabeled data.",
        "Transfer learning applies knowledge from one task to another related task.",
    ]

    # LLM 함수 생성
    llm_function = create_llm_function()
    if not llm_function:
        return

    # 임베딩 모델
    embedding_model = OpenAIEmbedding(model="text-embedding-3-small")

    # HybridRetriever 생성
    retriever = HybridRetriever(
        documents=documents,
        embedding_function=embedding_model.embed,
        fusion_method="rrf",
    )

    # 쿼리
    query = "How do machines learn?"
    print(f"\n원본 쿼리: {query}")

    # 1. 일반 검색 (쿼리 그대로)
    print("\n1. 일반 검색:")
    normal_results = retriever.search(query, top_k=3)
    for i, result in enumerate(normal_results, 1):
        print(f"  {i}. [Score: {result.score:.4f}] {result.text[:80]}...")

    # 2. HyDE 검색
    print("\n2. HyDE 검색:")

    # HyDE Expander
    expander = HyDEExpander(llm_function=llm_function, temperature=0.7)

    # 가상 문서 생성
    hypothetical_doc = expander.expand(query)
    print(f"\n가상 문서:\n{hypothetical_doc[:200]}...")

    # 가상 문서로 검색
    hyde_results = retriever.search(hypothetical_doc, top_k=3)
    print("\n검색 결과:")
    for i, result in enumerate(hyde_results, 1):
        print(f"  {i}. [Score: {result.score:.4f}] {result.text[:80]}...")

    print("\n" + "=" * 60)


def demo_multi_query():
    """
    Multi-Query Expansion Demo

    하나의 쿼리를 여러 관점에서 재구성합니다.
    """
    print("=" * 60)
    print("Multi-Query Expansion Demo")
    print("=" * 60)

    from beanllm.domain.retrieval import MultiQueryExpander

    # LLM 함수 생성
    llm_function = create_llm_function()
    if not llm_function:
        return

    # Multi-Query Expander 생성
    expander = MultiQueryExpander(llm_function=llm_function, num_queries=3)

    # 쿼리 확장
    query = "How does AI work?"
    print(f"\n원본 쿼리: {query}")

    expanded_queries = expander.expand(query)
    print("\n확장된 쿼리들:")
    for i, q in enumerate(expanded_queries, 1):
        print(f"  {i}. {q}")

    print("\n" + "=" * 60)


def demo_step_back():
    """
    Step-back Prompting Demo

    구체적인 쿼리를 더 넓은 맥락으로 재구성합니다.
    """
    print("=" * 60)
    print("Step-back Prompting Demo")
    print("=" * 60)

    from beanllm.domain.retrieval import StepBackExpander

    # LLM 함수 생성
    llm_function = create_llm_function()
    if not llm_function:
        return

    # Step-back Expander 생성
    expander = StepBackExpander(llm_function=llm_function)

    # 쿼리 확장
    query = "What was the impact of COVID-19 on the tech industry in 2020?"
    print(f"\n구체적 쿼리: {query}")

    step_back_query = expander.expand(query)
    print(f"\nStep-back 쿼리: {step_back_query}")

    print("\n" + "=" * 60)


def demo_custom_prompt():
    """
    Custom Prompt Template Demo

    도메인 특화 프롬프트를 사용한 HyDE 예제
    """
    print("=" * 60)
    print("Custom Prompt Template Demo")
    print("=" * 60)

    # LLM 함수 생성
    llm_function = create_llm_function()
    if not llm_function:
        return

    # 의료 도메인 특화 프롬프트
    medical_prompt = """You are a medical expert. Please provide a detailed,
accurate answer to the following medical question.

Question: {query}

Detailed Answer:"""

    # HyDE Expander 생성 (커스텀 프롬프트)
    expander = HyDEExpander(
        llm_function=llm_function,
        prompt_template=medical_prompt,
        temperature=0.3,  # 낮은 온도로 정확성 향상
    )

    # 쿼리 확장
    query = "What are the symptoms of type 2 diabetes?"
    print(f"\n원본 쿼리: {query}")

    hypothetical_doc = expander.expand(query)
    print(f"\n가상 문서 (의료 도메인):\n{hypothetical_doc}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # 데모 실행
    print("\n" + "=" * 60)
    print("HyDE Query Expansion Demo")
    print("=" * 60 + "\n")

    # 1. HyDE 기본
    demo_hyde_basic()
    print("\n")

    # 2. HyDE + Retrieval
    demo_hyde_with_retrieval()
    print("\n")

    # 3. Multi-Query
    demo_multi_query()
    print("\n")

    # 4. Step-back
    demo_step_back()
    print("\n")

    # 5. Custom Prompt
    demo_custom_prompt()
    print("\n")

    print("All demos completed!")
