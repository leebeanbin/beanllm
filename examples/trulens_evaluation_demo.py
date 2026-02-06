"""
TruLens RAG Evaluation Demo

TruLens를 사용한 RAG 시스템 평가 예제입니다.

TruLens는 RAG Triad (Context Relevance, Groundedness, Answer Relevance)를
사용하여 RAG 시스템을 종합적으로 평가합니다.

Requirements:
    pip install trulens-eval openai
"""

from beanllm.domain.evaluation import TruLensWrapper


def demo_rag_triad():
    """
    RAG Triad 평가 데모

    3가지 핵심 메트릭을 한번에 평가합니다:
    1. Context Relevance: 검색된 컨텍스트가 질문과 관련있는지
    2. Groundedness: 답변이 컨텍스트에 근거하는지 (Hallucination 방지)
    3. Answer Relevance: 답변이 질문에 적절한지
    """
    print("=" * 60)
    print("RAG Triad Evaluation Demo")
    print("=" * 60)

    # TruLens Evaluator 생성
    evaluator = TruLensWrapper(provider="openai", model="gpt-4o-mini")

    # 평가 데이터
    question = "What is the capital of France?"
    answer = "Paris is the capital of France."
    contexts = [
        "Paris is the capital and largest city of France.",
        "France is a country in Western Europe.",
    ]

    print(f"\n질문: {question}")
    print(f"답변: {answer}")
    print("컨텍스트:")
    for i, ctx in enumerate(contexts, 1):
        print(f"  {i}. {ctx}")

    # RAG Triad 평가
    result = evaluator.evaluate_rag_triad(question=question, answer=answer, contexts=contexts)

    print("\nRAG Triad 결과:")
    print(f"  Context Relevance: {result['context_relevance']:.3f}")
    print(f"  Groundedness: {result['groundedness']:.3f}")
    print(f"  Answer Relevance: {result['answer_relevance']:.3f}")

    print("\n" + "=" * 60)


def demo_context_relevance():
    """
    Context Relevance 평가 데모

    검색된 컨텍스트가 질문과 관련있는지 평가합니다.
    관련 없는 컨텍스트를 필터링하는데 유용합니다.
    """
    print("=" * 60)
    print("Context Relevance Evaluation Demo")
    print("=" * 60)

    evaluator = TruLensWrapper(provider="openai", model="gpt-4o-mini")

    # 좋은 예시 (관련있는 컨텍스트)
    print("\n1. 관련있는 컨텍스트:")
    question = "What is machine learning?"
    contexts = [
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "ML algorithms improve automatically through experience.",
    ]

    print(f"질문: {question}")
    print(f"컨텍스트: {contexts}")

    result = evaluator.evaluate_context_relevance(question=question, contexts=contexts)
    print(f"Context Relevance: {result['context_relevance']:.3f} (높음 = 좋음)")

    # 나쁜 예시 (관련없는 컨텍스트)
    print("\n2. 관련없는 컨텍스트:")
    question = "What is machine learning?"
    contexts = [
        "Paris is the capital of France.",
        "The weather is nice today.",
    ]

    print(f"질문: {question}")
    print(f"컨텍스트: {contexts}")

    result = evaluator.evaluate_context_relevance(question=question, contexts=contexts)
    print(f"Context Relevance: {result['context_relevance']:.3f} (낮음 = 나쁨)")

    print("\n" + "=" * 60)


def demo_groundedness():
    """
    Groundedness 평가 데모 (Hallucination 체크)

    답변이 컨텍스트에 근거하는지 평가합니다.
    Hallucination을 방지하는데 중요합니다.
    """
    print("=" * 60)
    print("Groundedness Evaluation Demo (Hallucination Check)")
    print("=" * 60)

    evaluator = TruLensWrapper(provider="openai", model="gpt-4o-mini")

    # 좋은 예시 (근거있는 답변)
    print("\n1. 근거있는 답변:")
    contexts = ["Paris is the capital and largest city of France."]
    answer = "Paris is the capital of France."

    print(f"컨텍스트: {contexts}")
    print(f"답변: {answer}")

    result = evaluator.evaluate_groundedness(answer=answer, contexts=contexts)
    print(f"Groundedness: {result['groundedness']:.3f} (높음 = 근거있음)")

    # 나쁜 예시 (Hallucination)
    print("\n2. Hallucination 예시:")
    contexts = ["Paris is the capital and largest city of France."]
    answer = "Paris is the capital of France and has exactly 10 million people."

    print(f"컨텍스트: {contexts}")
    print(f"답변: {answer}")

    result = evaluator.evaluate_groundedness(answer=answer, contexts=contexts)
    print(f"Groundedness: {result['groundedness']:.3f} (낮음 = Hallucination)")

    print("\n" + "=" * 60)


def demo_answer_relevance():
    """
    Answer Relevance 평가 데모

    답변이 질문에 적절한지 평가합니다.
    """
    print("=" * 60)
    print("Answer Relevance Evaluation Demo")
    print("=" * 60)

    evaluator = TruLensWrapper(provider="openai", model="gpt-4o-mini")

    # 좋은 예시 (적절한 답변)
    print("\n1. 적절한 답변:")
    question = "What is the capital of France?"
    answer = "Paris is the capital of France."

    print(f"질문: {question}")
    print(f"답변: {answer}")

    result = evaluator.evaluate_answer_relevance(question=question, answer=answer)
    print(f"Answer Relevance: {result['answer_relevance']:.3f} (높음 = 적절함)")

    # 나쁜 예시 (부적절한 답변)
    print("\n2. 부적절한 답변:")
    question = "What is the capital of France?"
    answer = "France is a country in Europe with many beautiful cities."

    print(f"질문: {question}")
    print(f"답변: {answer}")

    result = evaluator.evaluate_answer_relevance(question=question, answer=answer)
    print(f"Answer Relevance: {result['answer_relevance']:.3f} (낮음 = 부적절함)")

    print("\n" + "=" * 60)


def demo_batch_evaluation():
    """
    배치 평가 데모

    여러 RAG 결과를 한번에 평가합니다.
    """
    print("=" * 60)
    print("Batch RAG Evaluation Demo")
    print("=" * 60)

    evaluator = TruLensWrapper(provider="openai", model="gpt-4o-mini")

    # 테스트 데이터셋
    test_cases = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris is the capital of France.",
            "contexts": ["Paris is the capital and largest city of France."],
        },
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of AI.",
            "contexts": ["Machine learning is a subset of AI that learns from data."],
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "answer": "William Shakespeare wrote Romeo and Juliet.",
            "contexts": ["Romeo and Juliet is a tragedy written by William Shakespeare."],
        },
    ]

    print(f"\n{len(test_cases)}개의 테스트 케이스 평가 중...\n")

    # 배치 평가
    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"{i}. 질문: {case['question']}")

        # RAG Triad 평가
        result = evaluator.evaluate_rag_triad(
            question=case["question"],
            answer=case["answer"],
            contexts=case["contexts"],
        )

        results.append(result)

        print(
            f"   CR: {result['context_relevance']:.3f}, "
            f"G: {result['groundedness']:.3f}, "
            f"AR: {result['answer_relevance']:.3f}"
        )

    # 평균 점수
    avg_context_relevance = sum(r["context_relevance"] for r in results) / len(results)
    avg_groundedness = sum(r["groundedness"] for r in results) / len(results)
    avg_answer_relevance = sum(r["answer_relevance"] for r in results) / len(results)

    print("\n평균 점수:")
    print(f"  Context Relevance: {avg_context_relevance:.3f}")
    print(f"  Groundedness: {avg_groundedness:.3f}")
    print(f"  Answer Relevance: {avg_answer_relevance:.3f}")

    print("\n" + "=" * 60)


def demo_comparison_ragas_vs_trulens():
    """
    RAGAS vs TruLens 비교 데모

    같은 데이터를 RAGAS와 TruLens로 평가하여 비교합니다.
    """
    print("=" * 60)
    print("RAGAS vs TruLens Comparison Demo")
    print("=" * 60)

    from beanllm.domain.evaluation import RAGASWrapper

    # 평가 데이터
    question = "What is the capital of France?"
    answer = "Paris is the capital of France."
    contexts = ["Paris is the capital and largest city of France."]

    print(f"\n질문: {question}")
    print(f"답변: {answer}")
    print(f"컨텍스트: {contexts}\n")

    # TruLens 평가
    print("1. TruLens 평가:")
    trulens = TruLensWrapper(provider="openai", model="gpt-4o-mini")
    trulens_result = trulens.evaluate_rag_triad(question=question, answer=answer, contexts=contexts)

    print(f"   Context Relevance: {trulens_result['context_relevance']:.3f}")
    print(f"   Groundedness: {trulens_result['groundedness']:.3f}")
    print(f"   Answer Relevance: {trulens_result['answer_relevance']:.3f}")

    # RAGAS 평가
    try:
        print("\n2. RAGAS 평가:")
        ragas = RAGASWrapper(model="gpt-4o-mini", embeddings="text-embedding-3-small")

        faithfulness = ragas.evaluate_faithfulness(
            question=question, answer=answer, contexts=contexts
        )
        answer_relevancy = ragas.evaluate_answer_relevancy(
            question=question, answer=answer, contexts=contexts
        )

        print(f"   Faithfulness: {faithfulness.get('faithfulness', 'N/A')}")
        print(f"   Answer Relevancy: {answer_relevancy.get('answer_relevancy', 'N/A')}")

    except Exception as e:
        print(f"   RAGAS evaluation failed: {e}")

    print("\n비교:")
    print("  - TruLens: RAG Triad, 시각화, 트레이싱")
    print("  - RAGAS: 더 많은 메트릭, reference-free")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TruLens RAG Evaluation Demo")
    print("=" * 60 + "\n")

    # 1. RAG Triad
    demo_rag_triad()
    print("\n")

    # 2. Context Relevance
    demo_context_relevance()
    print("\n")

    # 3. Groundedness (Hallucination Check)
    demo_groundedness()
    print("\n")

    # 4. Answer Relevance
    demo_answer_relevance()
    print("\n")

    # 5. Batch Evaluation
    demo_batch_evaluation()
    print("\n")

    # 6. RAGAS vs TruLens
    try:
        demo_comparison_ragas_vs_trulens()
    except ImportError:
        print("RAGAS not installed. Skipping comparison demo.")
    print("\n")

    print("All demos completed!")
