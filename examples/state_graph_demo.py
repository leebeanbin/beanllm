"""
StateGraph - LangGraph 스타일 워크플로우
TypedDict 기반 타입 안전 상태 + Checkpointing
"""
from typing_extensions import TypedDict
from typing import Optional
from beanllm import (
    StateGraph,
    END,
    create_state_graph,
    Checkpoint
)


def demo_basic():
    """기본 사용법"""
    print("\n" + "="*60)
    print("1️⃣  기본 StateGraph")
    print("="*60)

    # State 정의 (TypedDict)
    class MyState(TypedDict):
        input: str
        output: str
        count: int

    # 그래프 생성
    graph = StateGraph(MyState)

    # 노드 정의
    def process(state: MyState) -> MyState:
        """입력을 대문자로 변환"""
        state["output"] = state["input"].upper()
        state["count"] += 1
        return state

    # 노드 & 엣지 추가
    graph.add_node("process", process)
    graph.add_edge("process", END)
    graph.set_entry_point("process")

    # 실행
    print("\n[실행]")
    result = graph.invoke({
        "input": "hello world",
        "output": "",
        "count": 0
    })

    print(f"  Input:  {result['input']}")
    print(f"  Output: {result['output']}")
    print(f"  Count:  {result['count']}")

    print("\n💡 TypedDict로 타입 안전!")


def demo_sequential():
    """순차 실행"""
    print("\n" + "="*60)
    print("2️⃣  순차 실행 (여러 노드)")
    print("="*60)

    class ProcessState(TypedDict):
        text: str
        step1_result: str
        step2_result: str
        step3_result: str

    graph = StateGraph(ProcessState)

    # 3단계 처리
    def step1(state: ProcessState) -> ProcessState:
        state["step1_result"] = state["text"].upper()
        return state

    def step2(state: ProcessState) -> ProcessState:
        state["step2_result"] = state["step1_result"] + "!!!"
        return state

    def step3(state: ProcessState) -> ProcessState:
        state["step3_result"] = f"[{state['step2_result']}]"
        return state

    # 그래프 구성
    graph.add_node("step1", step1)
    graph.add_node("step2", step2)
    graph.add_node("step3", step3)

    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.add_edge("step3", END)

    graph.set_entry_point("step1")

    # 실행
    print("\n[실행]")
    result = graph.invoke({
        "text": "hello",
        "step1_result": "",
        "step2_result": "",
        "step3_result": ""
    })

    print(f"  원본:   {result['text']}")
    print(f"  Step1:  {result['step1_result']}")
    print(f"  Step2:  {result['step2_result']}")
    print(f"  Step3:  {result['step3_result']}")

    print("\n💡 step1 → step2 → step3 순차 실행!")


def demo_conditional():
    """조건부 분기"""
    print("\n" + "="*60)
    print("3️⃣  조건부 분기 (Conditional Edge)")
    print("="*60)

    class RouterState(TypedDict):
        value: int
        path: str
        result: str

    graph = StateGraph(RouterState)

    # 노드들
    def check_value(state: RouterState) -> RouterState:
        """값 체크"""
        state["path"] = "checked"
        return state

    def process_small(state: RouterState) -> RouterState:
        """작은 값 처리"""
        state["result"] = f"Small: {state['value']}"
        return state

    def process_large(state: RouterState) -> RouterState:
        """큰 값 처리"""
        state["result"] = f"Large: {state['value']}"
        return state

    # 라우팅 함수
    def route(state: RouterState) -> str:
        """조건에 따라 다음 노드 결정"""
        if state["value"] < 10:
            return "small"
        else:
            return "large"

    # 그래프 구성
    graph.add_node("check", check_value)
    graph.add_node("small", process_small)
    graph.add_node("large", process_large)

    # 조건부 엣지
    graph.add_conditional_edge(
        "check",
        route,
        {
            "small": "small",
            "large": "large"
        }
    )

    graph.add_edge("small", END)
    graph.add_edge("large", END)

    graph.set_entry_point("check")

    # 작은 값 테스트
    print("\n[작은 값: 5]")
    result = graph.invoke({"value": 5, "path": "", "result": ""})
    print(f"  Path:   {result['path']}")
    print(f"  Result: {result['result']}")

    # 큰 값 테스트
    print("\n[큰 값: 15]")
    result = graph.invoke({"value": 15, "path": "", "result": ""})
    print(f"  Path:   {result['path']}")
    print(f"  Result: {result['result']}")

    print("\n💡 조건에 따라 다른 경로로 실행!")


def demo_loop():
    """루프 (반복)"""
    print("\n" + "="*60)
    print("4️⃣  루프 (반복 실행)")
    print("="*60)

    class LoopState(TypedDict):
        count: int
        max_count: int
        result: str

    graph = StateGraph(LoopState)

    # 노드들
    def increment(state: LoopState) -> LoopState:
        """카운트 증가"""
        state["count"] += 1
        state["result"] = f"Count: {state['count']}"
        return state

    # 라우팅: 계속 or 종료
    def should_continue(state: LoopState) -> str:
        if state["count"] < state["max_count"]:
            return "continue"
        else:
            return "end"

    # 그래프 구성
    graph.add_node("increment", increment)

    # 조건부 엣지 (루프)
    graph.add_conditional_edge(
        "increment",
        should_continue,
        {
            "continue": "increment",  # 자기 자신으로 (루프)
            "end": END
        }
    )

    graph.set_entry_point("increment")

    # 실행
    print("\n[3번 반복]")
    result = graph.invoke({
        "count": 0,
        "max_count": 3,
        "result": ""
    })

    print(f"  Final Count: {result['count']}")
    print(f"  Result:      {result['result']}")

    print("\n💡 조건이 만족될 때까지 반복!")


def demo_checkpointing():
    """체크포인팅"""
    print("\n" + "="*60)
    print("5️⃣  Checkpointing (상태 저장/복원)")
    print("="*60)

    from beanllm import GraphConfig
    from pathlib import Path
    import shutil

    class CheckpointState(TypedDict):
        step: int
        data: str

    # 체크포인팅 활성화
    config = GraphConfig(
        enable_checkpointing=True,
        checkpoint_dir=Path(".demo_checkpoints")
    )

    graph = StateGraph(CheckpointState, config=config)

    # 노드들
    def step1(state: CheckpointState) -> CheckpointState:
        state["step"] = 1
        state["data"] = "Step 1 completed"
        return state

    def step2(state: CheckpointState) -> CheckpointState:
        state["step"] = 2
        state["data"] = "Step 2 completed"
        return state

    def step3(state: CheckpointState) -> CheckpointState:
        state["step"] = 3
        state["data"] = "Step 3 completed"
        return state

    # 그래프 구성
    graph.add_node("step1", step1)
    graph.add_node("step2", step2)
    graph.add_node("step3", step3)

    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.add_edge("step3", END)

    graph.set_entry_point("step1")

    # 첫 실행
    print("\n[첫 실행]")
    execution_id = "test_exec_001"
    result = graph.invoke(
        {"step": 0, "data": ""},
        execution_id=execution_id
    )
    print(f"  Step: {result['step']}")
    print(f"  Data: {result['data']}")

    # 체크포인트 확인
    print("\n[저장된 체크포인트]")
    checkpoints = graph.checkpoint.list_checkpoints(execution_id)
    print(f"  {len(checkpoints)}개 체크포인트")
    for cp in checkpoints[:3]:
        print(f"    • {cp}")

    # 특정 지점에서 복원
    print("\n[step2에서 복원]")
    restored_state = graph.checkpoint.load(execution_id, "step2")
    if restored_state:
        print(f"  Step: {restored_state['step']}")
        print(f"  Data: {restored_state['data']}")

    # 정리
    if config.checkpoint_dir and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        print("\n  (체크포인트 정리 완료)")

    print("\n💡 각 노드 실행 후 상태 자동 저장!")


def demo_streaming():
    """스트리밍 실행"""
    print("\n" + "="*60)
    print("6️⃣  스트리밍 (각 노드 실행 결과 즉시 확인)")
    print("="*60)

    class StreamState(TypedDict):
        value: int

    graph = StateGraph(StreamState)

    def multiply_by_2(state: StreamState) -> StreamState:
        state["value"] *= 2
        return state

    def add_10(state: StreamState) -> StreamState:
        state["value"] += 10
        return state

    def multiply_by_3(state: StreamState) -> StreamState:
        state["value"] *= 3
        return state

    # 그래프 구성
    graph.add_node("multiply2", multiply_by_2)
    graph.add_node("add10", add_10)
    graph.add_node("multiply3", multiply_by_3)

    graph.add_edge("multiply2", "add10")
    graph.add_edge("add10", "multiply3")
    graph.add_edge("multiply3", END)

    graph.set_entry_point("multiply2")

    # 스트리밍 실행
    print("\n[스트리밍 실행]")
    print("  초기값: 5")
    print("\n  진행:")

    for node_name, state in graph.stream({"value": 5}):
        print(f"    {node_name}: {state['value']}")

    print("\n💡 각 노드 실행마다 중간 결과 확인!")


def demo_visualization():
    """그래프 시각화"""
    print("\n" + "="*60)
    print("7️⃣  그래프 시각화")
    print("="*60)

    class VizState(TypedDict):
        data: str

    graph = StateGraph(VizState)

    # 노드 추가
    graph.add_node("start", lambda s: s)
    graph.add_node("process", lambda s: s)
    graph.add_node("check", lambda s: s)
    graph.add_node("finish", lambda s: s)

    # 엣지
    graph.add_edge("start", "process")
    graph.add_edge("process", "check")

    # 조건부 엣지
    graph.add_conditional_edge(
        "check",
        lambda s: "finish",
        {"finish": "finish", "retry": "process"}
    )

    graph.add_edge("finish", END)
    graph.set_entry_point("start")

    # 시각화
    print("\n" + graph.visualize())

    print("\n💡 그래프 구조를 텍스트로 확인!")


def demo_practical_workflow():
    """실전 예제: 문서 처리 워크플로우"""
    print("\n" + "="*60)
    print("8️⃣  실전 예제 - 문서 처리 워크플로우")
    print("="*60)

    class DocumentWorkflowState(TypedDict):
        document: str
        cleaned: str
        summary: str
        quality_score: int
        approved: bool

    graph = StateGraph(DocumentWorkflowState)

    # 노드들
    def clean_document(state: DocumentWorkflowState) -> DocumentWorkflowState:
        """문서 정제"""
        state["cleaned"] = state["document"].strip().lower()
        return state

    def summarize(state: DocumentWorkflowState) -> DocumentWorkflowState:
        """요약 생성"""
        # 실제로는 LLM 사용
        state["summary"] = f"Summary of: {state['cleaned'][:30]}..."
        return state

    def quality_check(state: DocumentWorkflowState) -> DocumentWorkflowState:
        """품질 체크"""
        # 간단한 점수
        state["quality_score"] = len(state["summary"]) % 10
        return state

    def approve_or_reject(state: DocumentWorkflowState) -> DocumentWorkflowState:
        """승인 처리"""
        state["approved"] = state["quality_score"] >= 5
        return state

    # 라우팅
    def route_by_quality(state: DocumentWorkflowState) -> str:
        if state["quality_score"] >= 5:
            return "approve"
        else:
            return "reject"

    # 그래프 구성
    graph.add_node("clean", clean_document)
    graph.add_node("summarize", summarize)
    graph.add_node("quality_check", quality_check)
    graph.add_node("approve", approve_or_reject)
    graph.add_node("reject", approve_or_reject)

    graph.add_edge("clean", "summarize")
    graph.add_edge("summarize", "quality_check")

    graph.add_conditional_edge(
        "quality_check",
        route_by_quality,
        {
            "approve": "approve",
            "reject": "reject"
        }
    )

    graph.add_edge("approve", END)
    graph.add_edge("reject", END)

    graph.set_entry_point("clean")

    # 실행
    print("\n[문서 처리]")
    result = graph.invoke({
        "document": "  This is a TEST DOCUMENT  ",
        "cleaned": "",
        "summary": "",
        "quality_score": 0,
        "approved": False
    })

    print(f"  원본:      {result['document']}")
    print(f"  정제:      {result['cleaned']}")
    print(f"  요약:      {result['summary']}")
    print(f"  품질 점수: {result['quality_score']}")
    print(f"  승인 여부: {result['approved']}")

    print("\n💡 복잡한 워크플로우를 간단하게!")


def main():
    """모든 데모 실행"""
    print("="*60)
    print("🚀 StateGraph - LangGraph 스타일")
    print("="*60)
    print("\n8가지 기능:")
    print("  1. 기본 StateGraph")
    print("  2. 순차 실행")
    print("  3. 조건부 분기")
    print("  4. 루프 (반복)")
    print("  5. Checkpointing")
    print("  6. 스트리밍")
    print("  7. 그래프 시각화")
    print("  8. 실전 워크플로우")

    demo_basic()
    demo_sequential()
    demo_conditional()
    demo_loop()
    demo_checkpointing()
    demo_streaming()
    demo_visualization()
    demo_practical_workflow()

    print("\n" + "="*60)
    print("🎉 StateGraph 데모 완료!")
    print("="*60)
    print("\n✨ 핵심 기능:")
    print("  • TypedDict 기반 타입 안전 상태")
    print("  • 조건부 분기 (Conditional Edge)")
    print("  • 루프 지원 (자기 참조 엣지)")
    print("  • Checkpointing (상태 저장/복원)")
    print("  • 스트리밍 실행")
    print("  • 시각화")
    print("\n💡 복잡한 워크플로우를 간단하고 타입 안전하게!")


if __name__ == "__main__":
    main()
