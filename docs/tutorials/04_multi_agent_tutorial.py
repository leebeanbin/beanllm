"""
Multi-Agent Systems Tutorial
=============================

이 튜토리얼은 llmkit의 Multi-Agent 시스템을 실습합니다.

Topics:
1. Message Passing - Agent 간 통신
2. Sequential Coordination - 순차 협업
3. Parallel Coordination - 병렬 실행 (투표)
4. Hierarchical Coordination - 계층적 조직
5. Debate Strategy - 토론과 합의
6. Advanced Patterns - 실전 응용
"""

import asyncio
from typing import Dict, Any, List

# llmkit imports
# from llmkit import (
#     Agent, MultiAgentCoordinator, CommunicationBus,
#     AgentMessage, MessageType, create_coordinator, quick_debate
# )

print("="*80)
print("Multi-Agent Systems Tutorial")
print("="*80)


# =============================================================================
# Part 1: Message Passing - Agent 간 통신
# =============================================================================

print("\n" + "="*80)
print("Part 1: Message Passing - 기본 통신")
print("="*80)

"""
Theory:
    Message Passing Model에서 메시지는 튜플로 표현됩니다:
    m = (sender, receiver, content, timestamp)

    Communication Patterns:
    - Unicast (1:1): 하나의 수신자
    - Broadcast (1:N): 모든 수신자
    - Multicast (1:M): 일부 수신자
"""


async def demo_message_passing():
    """기본 메시지 통신 예제"""
    from llmkit import CommunicationBus, AgentMessage, MessageType

    # 통신 버스 생성
    bus = CommunicationBus(delivery_guarantee="exactly-once")

    # 메시지 핸들러들
    received_messages = {"agent1": [], "agent2": [], "agent3": []}

    def create_handler(agent_id: str):
        def handler(msg: AgentMessage):
            received_messages[agent_id].append(msg)
            print(f"  [{agent_id}] Received from {msg.sender}: {msg.content}")
        return handler

    # Agent들을 버스에 구독
    bus.subscribe("agent1", create_handler("agent1"))
    bus.subscribe("agent2", create_handler("agent2"))
    bus.subscribe("agent3", create_handler("agent3"))

    print("\n--- 1. Unicast (1:1) ---")
    # Agent1 → Agent2
    msg1 = AgentMessage(
        sender="agent1",
        receiver="agent2",
        message_type=MessageType.INFORM,
        content="Hello Agent2!"
    )
    await bus.publish(msg1)

    print("\n--- 2. Broadcast (1:N) ---")
    # Agent1 → All
    msg2 = AgentMessage(
        sender="agent1",
        receiver=None,  # None = broadcast
        message_type=MessageType.BROADCAST,
        content="Hello everyone!"
    )
    await bus.publish(msg2)

    print("\n--- 3. Request-Response ---")
    # Agent2가 Agent3에게 요청
    request = AgentMessage(
        sender="agent2",
        receiver="agent3",
        message_type=MessageType.REQUEST,
        content="Can you help me?"
    )
    await bus.publish(request)

    # Agent3가 응답
    response = request.reply(
        content="Sure, I can help!",
        message_type=MessageType.RESPONSE
    )
    await bus.publish(response)

    # 히스토리 확인
    print("\n--- Message History ---")
    history = bus.get_history(limit=10)
    for i, msg in enumerate(history):
        print(f"{i+1}. {msg.sender} → {msg.receiver or 'ALL'}: {msg.content}")


# 실행
if __name__ == "__main__":
    # asyncio.run(demo_message_passing())
    pass


# =============================================================================
# Part 2: Sequential Coordination - 순차 협업
# =============================================================================

print("\n" + "="*80)
print("Part 2: Sequential Coordination - 작업의 순차 전달")
print("="*80)

"""
Theory:
    Function composition:
    output = f₃ ∘ f₂ ∘ f₁(input)

    각 agent는 이전 agent의 출력을 입력으로 받습니다.

    Time Complexity: O(Σ Tᵢ) - 모든 agent 시간의 합
"""


async def demo_sequential():
    """순차 협업 예제: 연구 → 작성 → 편집"""
    from llmkit import Agent, MultiAgentCoordinator

    # Agents 생성
    researcher = Agent(
        model="gpt-4o-mini",
        system_prompt="You are a researcher. Gather key facts about the topic."
    )

    writer = Agent(
        model="gpt-4o-mini",
        system_prompt="You are a writer. Turn research into a clear article."
    )

    editor = Agent(
        model="gpt-4o-mini",
        system_prompt="You are an editor. Polish the article and fix errors."
    )

    # Coordinator 생성
    coordinator = MultiAgentCoordinator(agents={
        "researcher": researcher,
        "writer": writer,
        "editor": editor
    })

    # 순차 실행
    task = "Write about the benefits of exercise"

    print("\n순차 실행 시작...")
    print(f"Task: {task}\n")

    result = await coordinator.execute_sequential(
        task=task,
        agent_order=["researcher", "writer", "editor"]
    )

    print("\n" + "="*60)
    print("Results:")
    print("="*60)

    for i, answer in enumerate(result["intermediate_results"]):
        agent_name = ["researcher", "writer", "editor"][i]
        print(f"\n[{agent_name}]:")
        print(answer[:200] + "..." if len(answer) > 200 else answer)

    print("\n" + "="*60)
    print("Final Result:")
    print("="*60)
    print(result["final_result"])


# 실행
if __name__ == "__main__":
    # asyncio.run(demo_sequential())
    pass


# =============================================================================
# Part 3: Parallel Coordination - 병렬 실행과 투표
# =============================================================================

print("\n" + "="*80)
print("Part 3: Parallel Coordination - 병렬 실행과 합의")
print("="*80)

"""
Theory:
    Parallel execution:
    {r₁, r₂, ..., rₙ} = {f₁(x), f₂(x), ..., fₙ(x)} concurrently

    Speedup: S = T_sequential / T_parallel
    Ideal: S = n

    Voting (다수결):
    result = argmax_r |{i | fᵢ(x) = r}|
"""


async def demo_parallel_voting():
    """병렬 실행과 투표 예제"""
    from llmkit import Agent, MultiAgentCoordinator
    import time

    # 같은 모델로 3개 agents 생성
    agents = {
        f"agent_{i}": Agent(
            model="gpt-4o-mini",
            system_prompt=f"You are expert {i+1}. Answer concisely."
        )
        for i in range(3)
    }

    coordinator = MultiAgentCoordinator(agents=agents)

    # 테스트 질문 (답이 명확한 것)
    task = "What is the capital of France? Answer with just the city name."

    print("\n병렬 실행 + 투표...")
    start_time = time.time()

    result = await coordinator.execute_parallel(
        task=task,
        aggregation="vote"
    )

    elapsed = time.time() - start_time

    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"All Answers: {result['all_answers']}")
    print(f"Vote Counts: {result['vote_counts']}")
    print(f"Final Result (by vote): {result['final_result']}")
    print(f"Agreement Rate: {result['agreement_rate']:.1%}")
    print(f"Time: {elapsed:.2f}s")

    # Consensus 테스트
    print("\n--- Consensus Mode ---")
    result2 = await coordinator.execute_parallel(
        task=task,
        aggregation="consensus"
    )

    print(f"Consensus Reached: {result2['consensus']}")
    if result2['consensus']:
        print(f"Agreed Answer: {result2['final_result']}")


# 실행
if __name__ == "__main__":
    # asyncio.run(demo_parallel_voting())
    pass


# =============================================================================
# Part 4: Hierarchical Coordination - 계층적 조직
# =============================================================================

print("\n" + "="*80)
print("Part 4: Hierarchical Coordination - 매니저와 워커")
print("="*80)

"""
Theory:
    Tree structure:
         Manager
         /  |  \
       W₁  W₂  W₃

    Process:
    1. Manager decomposes task → subtasks
    2. Workers execute subtasks in parallel
    3. Manager synthesizes results → final answer

    Time: O(3 × T_agent) for depth=2 tree
"""


async def demo_hierarchical():
    """계층적 조직 예제: 여행 계획"""
    from llmkit import Agent, MultiAgentCoordinator

    # Manager
    manager = Agent(
        model="gpt-4o",
        system_prompt="You are a travel planning manager. Coordinate your team."
    )

    # Workers (각자 전문 분야)
    workers = {
        "accommodation": Agent(
            model="gpt-4o-mini",
            system_prompt="You are a hotel expert. Find best accommodations."
        ),
        "activities": Agent(
            model="gpt-4o-mini",
            system_prompt="You are an activities expert. Suggest things to do."
        ),
        "dining": Agent(
            model="gpt-4o-mini",
            system_prompt="You are a food expert. Recommend restaurants."
        )
    }

    # Coordinator
    all_agents = {"manager": manager, **workers}
    coordinator = MultiAgentCoordinator(agents=all_agents)

    # 작업
    task = "Plan a 3-day trip to Tokyo, Japan. Budget: $2000."

    print("\n계층적 실행 시작...")
    print(f"Task: {task}\n")

    result = await coordinator.execute_hierarchical(
        task=task,
        manager_id="manager",
        worker_ids=list(workers.keys())
    )

    print("\n" + "="*60)
    print("Subtasks (Manager's delegation):")
    print("="*60)
    for i, subtask in enumerate(result["subtasks"], 1):
        print(f"{i}. {subtask}")

    print("\n" + "="*60)
    print("Worker Results:")
    print("="*60)
    for i, (worker_id, answer) in enumerate(zip(workers.keys(), result["worker_results"]), 1):
        print(f"\n[{worker_id}]:")
        print(answer[:150] + "..." if len(answer) > 150 else answer)

    print("\n" + "="*60)
    print("Final Plan (Manager's synthesis):")
    print("="*60)
    print(result["final_result"])


# 실행
if __name__ == "__main__":
    # asyncio.run(demo_hierarchical())
    pass


# =============================================================================
# Part 5: Debate Strategy - 토론과 개선
# =============================================================================

print("\n" + "="*80)
print("Part 5: Debate Strategy - 여러 관점의 토론")
print("="*80)

"""
Theory:
    Iterative refinement:
    x₀⁽ⁱ⁾ = fᵢ(task)  (초기 답변)
    xₜ⁽ⁱ⁾ = gᵢ(xₜ₋₁⁽¹⁾, ..., xₜ₋₁⁽ⁿ⁾)  (개선)

    Convergence (이상적):
    lim(t→∞) distance(xₜ⁽ⁱ⁾, xₜ⁽ʲ⁾) = 0

    실제로는 fixed rounds로 종료
"""


async def demo_debate():
    """토론 예제: 논쟁적 주제"""
    from llmkit import Agent, MultiAgentCoordinator

    # 다른 관점을 가진 agents
    agents = {
        "optimist": Agent(
            model="gpt-4o-mini",
            system_prompt="You are an optimist. Focus on positive aspects and opportunities."
        ),
        "pessimist": Agent(
            model="gpt-4o-mini",
            system_prompt="You are a pessimist. Focus on risks and challenges."
        ),
        "realist": Agent(
            model="gpt-4o-mini",
            system_prompt="You are a realist. Focus on balanced, practical views."
        )
    }

    # Judge (선택적)
    judge = Agent(
        model="gpt-4o",
        system_prompt="You are an impartial judge. Evaluate arguments objectively."
    )

    all_agents = {**agents, "judge": judge}
    coordinator = MultiAgentCoordinator(agents=all_agents)

    # 논쟁적 주제
    task = "Should companies adopt 4-day work weeks?"

    print("\n토론 시작 (3 rounds)...")
    print(f"Topic: {task}\n")

    result = await coordinator.execute_debate(
        task=task,
        agent_ids=list(agents.keys()),
        rounds=3,
        judge_id="judge"
    )

    # 토론 과정 출력
    print("\n" + "="*60)
    print("Debate Progress:")
    print("="*60)

    for round_data in result["debate_history"]:
        round_num = round_data["round"]
        print(f"\n--- Round {round_num} ---")

        for agent_id, answer in round_data["answers"].items():
            agent_name = agent_id.split("_")[1] if "_" in agent_id else agent_id
            print(f"\n[{agent_name}]:")
            print(answer[:200] + "..." if len(answer) > 200 else answer)

    print("\n" + "="*60)
    print(f"Final Decision (by {result['decision_method']}):")
    print("="*60)
    print(result["final_result"])


# 실행
if __name__ == "__main__":
    # asyncio.run(demo_debate())
    pass


# =============================================================================
# Part 6: Advanced Patterns - 실전 응용
# =============================================================================

print("\n" + "="*80)
print("Part 6: Advanced Patterns - 복잡한 Multi-Agent 시스템")
print("="*80)

"""
실전 시나리오: 종합 리포트 작성 시스템

Architecture:
    Project Manager (orchestrator)
    ├─ Research Team (parallel)
    │  ├─ Data Researcher
    │  ├─ Literature Researcher
    │  └─ Market Researcher
    ├─ Analysis Team (sequential)
    │  ├─ Data Analyst
    │  └─ Statistician
    └─ Writing Team (debate)
       ├─ Technical Writer
       └─ Business Writer
"""


async def demo_advanced_multi_agent():
    """고급 Multi-Agent 시스템: 리포트 작성"""
    from llmkit import Agent, MultiAgentCoordinator

    # Project Manager
    pm = Agent(
        model="gpt-4o",
        system_prompt="You are a project manager. Coordinate teams and synthesize results."
    )

    # Research Team (병렬)
    research_team = {
        "data_researcher": Agent(
            model="gpt-4o-mini",
            system_prompt="Find and summarize relevant data and statistics."
        ),
        "lit_researcher": Agent(
            model="gpt-4o-mini",
            system_prompt="Find relevant academic papers and key findings."
        ),
        "market_researcher": Agent(
            model="gpt-4o-mini",
            system_prompt="Analyze market trends and competitive landscape."
        )
    }

    # Analysis Team (순차)
    analysis_team = {
        "data_analyst": Agent(
            model="gpt-4o-mini",
            system_prompt="Analyze data patterns and extract insights."
        ),
        "statistician": Agent(
            model="gpt-4o-mini",
            system_prompt="Perform statistical analysis and validate findings."
        )
    }

    # Writing Team (토론)
    writing_team = {
        "tech_writer": Agent(
            model="gpt-4o-mini",
            system_prompt="Write technical, detailed content."
        ),
        "biz_writer": Agent(
            model="gpt-4o-mini",
            system_prompt="Write business-focused, accessible content."
        )
    }

    # 모든 agents
    all_agents = {
        "pm": pm,
        **research_team,
        **analysis_team,
        **writing_team
    }

    coordinator = MultiAgentCoordinator(agents=all_agents)

    # 프로젝트 주제
    topic = "The impact of AI on job market in 2025"

    print(f"\n프로젝트 시작: {topic}")
    print("="*60)

    # Phase 1: Research (병렬)
    print("\n[Phase 1] Research Team (Parallel Execution)...")
    research_result = await coordinator.execute_parallel(
        task=f"Research: {topic}",
        agent_ids=list(research_team.keys()),
        aggregation="all"
    )

    # Research 결과 요약
    research_summary = "\n".join([
        f"- {name}: {result[:100]}..."
        for name, result in zip(research_team.keys(), research_result["final_result"])
    ])

    print(f"Research completed: {len(research_result['final_result'])} reports")

    # Phase 2: Analysis (순차)
    print("\n[Phase 2] Analysis Team (Sequential Processing)...")
    analysis_input = f"Analyze this research:\n{research_summary}"

    analysis_result = await coordinator.execute_sequential(
        task=analysis_input,
        agent_order=list(analysis_team.keys())
    )

    print("Analysis completed")

    # Phase 3: Writing (토론)
    print("\n[Phase 3] Writing Team (Debate & Refinement)...")
    writing_input = f"Write final report:\nTopic: {topic}\nAnalysis: {analysis_result['final_result'][:300]}..."

    writing_result = await coordinator.execute_debate(
        task=writing_input,
        agent_ids=list(writing_team.keys()),
        rounds=2
    )

    print("Writing completed")

    # Phase 4: PM Synthesis
    print("\n[Phase 4] Project Manager (Final Synthesis)...")
    final_input = f"""Synthesize final report from team outputs:

Research: {research_summary[:200]}...
Analysis: {analysis_result['final_result'][:200]}...
Writing: {writing_result['final_result'][:200]}...

Create executive summary:
"""

    final_result = await all_agents["pm"].run(final_input)

    # 결과 출력
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print(final_result.answer)

    # 프로세스 통계
    print("\n" + "="*60)
    print("Process Statistics:")
    print("="*60)
    print(f"Research Team: {len(research_team)} agents (parallel)")
    print(f"Analysis Team: {len(analysis_team)} agents (sequential)")
    print(f"Writing Team: {len(writing_team)} agents (debate, 2 rounds)")
    print(f"Total Agents: {len(all_agents)}")
    print(f"Total Steps: {final_result.total_steps}")


# =============================================================================
# Part 7: Quick Helpers - 빠른 사용
# =============================================================================

print("\n" + "="*80)
print("Part 7: Quick Helpers - 편의 함수")
print("="*80)


async def demo_quick_helpers():
    """빠른 실행 헬퍼 함수들"""
    from llmkit import quick_debate, create_coordinator

    # 1. Quick Debate
    print("\n--- Quick Debate ---")
    result = await quick_debate(
        task="Is remote work better than office work?",
        num_agents=3,
        rounds=2,
        model="gpt-4o-mini"
    )

    print(f"Debate Result: {result['final_result'][:150]}...")

    # 2. Create Coordinator from configs
    print("\n--- Create Coordinator ---")
    coordinator = create_coordinator([
        {"id": "agent1", "model": "gpt-4o-mini", "system_prompt": "You are helpful."},
        {"id": "agent2", "model": "gpt-4o-mini", "system_prompt": "You are creative."},
    ])

    result = await coordinator.execute_parallel(
        task="What is 2+2?",
        aggregation="vote"
    )

    print(f"Result: {result['final_result']}")


# =============================================================================
# Performance Analysis - 병렬화 효과 측정
# =============================================================================

print("\n" + "="*80)
print("Part 8: Performance Analysis - Speedup 측정")
print("="*80)


async def demo_performance():
    """병렬화 성능 분석"""
    from llmkit import Agent, MultiAgentCoordinator
    import time

    # 3개 agents
    agents = {
        f"agent_{i}": Agent(model="gpt-4o-mini")
        for i in range(3)
    }

    coordinator = MultiAgentCoordinator(agents=agents)
    task = "Explain machine learning in one sentence."

    # Sequential
    print("\n순차 실행...")
    start = time.time()
    seq_result = await coordinator.execute_sequential(
        task=task,
        agent_order=list(agents.keys())
    )
    seq_time = time.time() - start
    print(f"Sequential Time: {seq_time:.2f}s")

    # Parallel
    print("\n병렬 실행...")
    start = time.time()
    par_result = await coordinator.execute_parallel(
        task=task,
        aggregation="vote"
    )
    par_time = time.time() - start
    print(f"Parallel Time: {par_time:.2f}s")

    # Speedup
    speedup = seq_time / par_time
    efficiency = speedup / len(agents)

    print("\n" + "="*60)
    print("Performance Metrics:")
    print("="*60)
    print(f"Speedup: {speedup:.2f}x")
    print(f"Efficiency: {efficiency:.1%}")
    print(f"Ideal Speedup: {len(agents)}x")

    # Amdahl's Law 예측
    # S = 1 / ((1-p) + p/n)
    # 가정: 90% 병렬 가능
    p = 0.9
    n = len(agents)
    theoretical_speedup = 1 / ((1-p) + p/n)

    print(f"Theoretical Speedup (Amdahl, p={p}): {theoretical_speedup:.2f}x")


# =============================================================================
# 전체 실행
# =============================================================================

async def run_all_demos():
    """모든 데모 실행"""
    demos = [
        ("Message Passing", demo_message_passing),
        ("Sequential Coordination", demo_sequential),
        ("Parallel Voting", demo_parallel_voting),
        ("Hierarchical", demo_hierarchical),
        ("Debate", demo_debate),
        ("Advanced Multi-Agent", demo_advanced_multi_agent),
        ("Quick Helpers", demo_quick_helpers),
        ("Performance Analysis", demo_performance),
    ]

    for name, demo in demos:
        print("\n" + "="*80)
        print(f"Running: {name}")
        print("="*80)
        try:
            await demo()
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(1)


if __name__ == "__main__":
    print("""
이 튜토리얼을 실행하려면:

1. llmkit 설치:
   pip install -e .

2. API 키 설정:
   export OPENAI_API_KEY="your-key"

3. 실행:
   python docs/tutorials/04_multi_agent_tutorial.py

개별 데모 실행은 해당 함수의 주석을 해제하세요.
    """)

    # 전체 실행
    # asyncio.run(run_all_demos())

    # 개별 실행 예시:
    asyncio.run(demo_message_passing())
    # asyncio.run(demo_sequential())
    # asyncio.run(demo_parallel_voting())
    # asyncio.run(demo_hierarchical())
    # asyncio.run(demo_debate())
    # asyncio.run(demo_advanced_multi_agent())


"""
연습 문제:

1. Consensus 알고리즘 구현
   - 모든 agent가 합의할 때까지 반복
   - Byzantine fault tolerance 고려

2. Dynamic Team Formation
   - 작업에 따라 최적의 agent 팀 자동 구성
   - Skill matching

3. Load Balancing
   - 작업을 agent들에게 균등 분배
   - 각 agent의 처리 속도 고려

4. Failure Recovery
   - Agent 실패 시 재시도 로직
   - 다른 agent로 failover

5. Communication Cost 최적화
   - 불필요한 메시지 줄이기
   - Gossip protocol 구현

6. Multi-round Negotiation
   - Agent들이 협상하여 합의 도출
   - Game theory 활용
"""
