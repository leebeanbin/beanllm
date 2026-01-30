# 시각적 워크플로우 구성 (n8n 스타일)

## 🎯 목표

n8n처럼 드래그 앤 드롭으로 워크플로우를 시각적으로 구성하고, 챗의 진행 상황을 그래프 노드로 실시간 표시

**기능**:
- 드래그 앤 드롭으로 노드 배치
- 노드 간 연결 (엣지)
- 실시간 진행 상황 시각화
- 데이터 흐름 추적
- 워크플로우 저장 및 재사용

---

## 📊 현재 상태

### 구현된 기능
- ✅ `WorkflowGraph`: 노드 기반 워크플로우 (`src/beanllm/domain/orchestrator/workflow_graph.py`)
- ✅ `WorkflowVisualizer`: 워크플로우 시각화 (`src/beanllm/ui/visualizers/workflow_viz.py`)
- ✅ `Visualization.tsx`: 프론트엔드 시각화 컴포넌트 (기본)
- ✅ `StateGraph`: 상태 그래프 실행 (`src/beanllm/facade/advanced/state_graph_facade.py`)

### 없는 기능
- ❌ 드래그 앤 드롭 노드 편집기
- ❌ 실시간 진행 상황 그래프 표시
- ❌ 노드 클릭으로 데이터 확인
- ❌ 워크플로우 저장/로드 UI

---

## ✅ 구현 방안

### 1. 노드 편집기 컴포넌트 (React Flow 사용)

**프론트엔드**: `playground/frontend/src/components/WorkflowEditor.tsx`

```typescript
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
} from 'reactflow';
import 'reactflow/dist/style.css';

interface WorkflowEditorProps {
  initialNodes?: Node[];
  initialEdges?: Edge[];
  onSave?: (nodes: Node[], edges: Edge[]) => void;
  readOnly?: boolean; // 읽기 전용 (진행 상황 표시용)
}

export function WorkflowEditor({
  initialNodes = [],
  initialEdges = [],
  onSave,
  readOnly = false,
}: WorkflowEditorProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  return (
    <div style={{ width: '100%', height: '600px' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={readOnly ? undefined : onNodesChange}
        onEdgesChange={readOnly ? undefined : onEdgesChange}
        onConnect={readOnly ? undefined : onConnect}
        fitView
      >
        <Background />
        <Controls />
      </ReactFlow>
      
      {!readOnly && onSave && (
        <button onClick={() => onSave(nodes, edges)}>
          저장
        </button>
      )}
    </div>
  );
}
```

### 2. 노드 타입 정의

**프론트엔드**: `playground/frontend/src/types/workflow.ts`

```typescript
export type NodeType =
  | "chat"           // 일반 채팅
  | "rag"            // RAG 검색
  | "agent"          // Agent 실행
  | "multi_agent"    // 멀티 에이전트
  | "kg"             // Knowledge Graph
  | "web_search"     // 웹 검색
  | "audio"          // 음성 처리
  | "ocr"            // OCR
  | "vision"         // 이미지 분석
  | "code"           // 코드 생성
  | "decision"       // 조건 분기
  | "merge"          // 결과 병합
  | "start"          // 시작
  | "end";           // 종료

export interface WorkflowNode extends Node {
  type: NodeType;
  data: {
    label: string;
    tool?: string;
    config?: Record<string, any>;
    status?: "pending" | "running" | "completed" | "failed";
    result?: any;
    executionTime?: number;
  };
}
```

### 3. 실시간 진행 상황 업데이트

**프론트엔드**: `playground/frontend/src/components/LiveWorkflowView.tsx`

```typescript
interface LiveWorkflowViewProps {
  workflowId: string;
  sessionId: string;
}

export function LiveWorkflowView({ workflowId, sessionId }: LiveWorkflowViewProps) {
  const [nodes, setNodes] = useState<WorkflowNode[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);

  useEffect(() => {
    // SSE로 진행 상황 수신
    const eventSource = new EventSource(
      `/api/chat/stream?session_id=${sessionId}&workflow_id=${workflowId}`
    );

    eventSource.addEventListener("tool_start", (e) => {
      const data = JSON.parse(e.data);
      // 노드 상태 업데이트: pending → running
      setNodes((prev) =>
        prev.map((node) =>
          node.id === data.step_id
            ? { ...node, data: { ...node.data, status: "running" } }
            : node
        )
      );
    });

    eventSource.addEventListener("tool_result", (e) => {
      const data = JSON.parse(e.data);
      // 노드 상태 업데이트: running → completed
      setNodes((prev) =>
        prev.map((node) =>
          node.id === data.step_id
            ? {
                ...node,
                data: {
                  ...node.data,
                  status: "completed",
                  result: data.result,
                  executionTime: data.execution_time,
                },
              }
            : node
        )
      );
    });

    return () => eventSource.close();
  }, [sessionId, workflowId]);

  return (
    <WorkflowEditor
      initialNodes={nodes}
      initialEdges={edges}
      readOnly={true} // 진행 상황만 표시
    />
  );
}
```

### 4. 노드 데이터 확인 (팝업)

**프론트엔드**: `playground/frontend/src/components/NodeDataViewer.tsx`

```typescript
interface NodeDataViewerProps {
  node: WorkflowNode;
  onClose: () => void;
}

export function NodeDataViewer({ node, onClose }: NodeDataViewerProps) {
  return (
    <Dialog open={true} onClose={onClose}>
      <DialogTitle>{node.data.label}</DialogTitle>
      <DialogContent>
        <div>
          <h4>상태</h4>
          <p>{node.data.status}</p>
        </div>
        
        {node.data.result && (
          <div>
            <h4>결과</h4>
            <pre>{JSON.stringify(node.data.result, null, 2)}</pre>
          </div>
        )}
        
        {node.data.executionTime && (
          <div>
            <h4>실행 시간</h4>
            <p>{node.data.executionTime}ms</p>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
```

### 5. 워크플로우 저장/로드

**백엔드**: `playground/backend/routers/workflow_router.py` (신규)

```python
@router.post("/api/workflow/save")
async def save_workflow(request: SaveWorkflowRequest) -> Dict[str, Any]:
    """워크플로우 저장"""
    workflow = {
        "workflow_id": request.workflow_id,
        "name": request.name,
        "nodes": request.nodes,
        "edges": request.edges,
        "created_at": datetime.now(),
        "user_id": request.user_id,
    }
    
    # MongoDB에 저장
    await db.workflows.insert_one(workflow)
    
    return {"workflow_id": request.workflow_id, "status": "saved"}

@router.get("/api/workflow/{workflow_id}")
async def load_workflow(workflow_id: str) -> Dict[str, Any]:
    """워크플로우 로드"""
    workflow = await db.workflows.find_one({"workflow_id": workflow_id})
    if not workflow:
        raise HTTPException(404, "Workflow not found")
    
    return {
        "workflow_id": workflow["workflow_id"],
        "name": workflow["name"],
        "nodes": workflow["nodes"],
        "edges": workflow["edges"],
    }
```

### 6. Orchestrator에서 워크플로우 생성

**백엔드**: `playground/backend/services/orchestrator.py` (수정)

```python
class AgenticOrchestrator:
    async def execute_with_visualization(
        self,
        context: OrchestratorContext,
        workflow_nodes: Optional[List[Dict]] = None
    ) -> AsyncGenerator[AgenticEvent, None]:
        """
        시각화를 위한 워크플로우 실행
        
        Args:
            context: Orchestrator 컨텍스트
            workflow_nodes: 워크플로우 노드 정의 (없으면 자동 생성)
        """
        # 워크플로우 노드 자동 생성 (없으면)
        if not workflow_nodes:
            workflow_nodes = self._generate_workflow_nodes(context)
        
        # 워크플로우 그래프 생성
        workflow_graph = WorkflowGraph(name=context.query)
        for node_def in workflow_nodes:
            workflow_graph.add_node(
                node_type=NodeType(node_def["type"]),
                name=node_def["name"],
                config=node_def.get("config", {})
            )
        
        # 엣지 추가
        for edge_def in workflow_nodes.get("edges", []):
            workflow_graph.add_edge(
                source=edge_def["from"],
                target=edge_def["to"]
            )
        
        # 워크플로우 실행 및 이벤트 스트리밍
        yield AgenticEvent(
            type=EventType.PARALLEL_START,
            data={"workflow": workflow_graph.to_dict()}
        )
        
        # 각 노드 실행
        for node_id in workflow_graph.get_topological_order():
            node = workflow_graph.nodes[node_id]
            
            yield AgenticEvent(
                type=EventType.TOOL_START,
                data={"step": node_id, "tool": node.name}
            )
            
            # 노드 실행
            result = await self._execute_workflow_node(node, context)
            
            yield AgenticEvent(
                type=EventType.TOOL_RESULT,
                data={
                    "step": node_id,
                    "result": result,
                    "execution_time": result.get("duration_ms", 0)
                }
            )
```

### 7. Chat UI 통합

**프론트엔드**: `playground/frontend/src/app/chat/page.tsx` (수정)

```typescript
export default function ChatPage() {
  const [showWorkflow, setShowWorkflow] = useState(false);
  const [workflowNodes, setWorkflowNodes] = useState<Node[]>([]);

  return (
    <div className="chat-container">
      <div className="chat-main">
        {/* 기존 채팅 UI */}
        <ChatMessages />
        <ChatInput />
      </div>
      
      {/* 워크플로우 뷰 토글 */}
      <button onClick={() => setShowWorkflow(!showWorkflow)}>
        {showWorkflow ? "워크플로우 숨기기" : "워크플로우 보기"}
      </button>
      
      {showWorkflow && (
        <div className="workflow-panel">
          <LiveWorkflowView
            workflowId={currentWorkflowId}
            sessionId={sessionId}
          />
        </div>
      )}
    </div>
  );
}
```

---

## 📋 구현 체크리스트 및 상태

### ✅ 구현됨
- [x] `reactflow` 패키지 설치 ✅ (`package.json`에 `reactflow@11.11.4`)
- [x] 기본 시각화 컴포넌트 (`components/Visualization.tsx`)
- [x] WorkflowGraph 구현 (`src/beanllm/domain/orchestrator/workflow_graph.py`)
- [x] WorkflowVisualizer 구현 (`src/beanllm/ui/visualizers/workflow_viz.py`)

### ❌ 미구현
- [ ] **`WorkflowEditor` 컴포넌트 (드래그 앤 드롭)**
  - **위치**: `playground/frontend/src/components/WorkflowEditor.tsx` (신규 생성 필요)
  - **구현 방향**:
    1. React Flow 기본 설정 (Controls, Background)
    2. 노드 타입별 커스텀 노드 컴포넌트
    3. 드래그 앤 드롭으로 노드 추가
    4. 엣지 연결 (노드 간 드래그)
  - **방법**: 문서의 "1. 노드 편집기 컴포넌트" 섹션 참조
  - **의존성**: `reactflow` (이미 설치됨)
- [ ] **`LiveWorkflowView` 컴포넌트 (실시간 진행 상황)**
  - **위치**: `playground/frontend/src/components/LiveWorkflowView.tsx` (신규 생성 필요)
  - **구현 방향**:
    1. SSE로 진행 상황 수신
    2. 노드 상태 실시간 업데이트 (pending → running → completed)
    3. 진행률 표시
  - **방법**: 문서의 "3. 실시간 진행 상황 업데이트" 섹션 참조
  - **통합**: `chat_router.py`의 `/api/chat/agentic` SSE 이벤트 활용
- [ ] **`NodeDataViewer` 컴포넌트 (노드 데이터 확인)**
  - **위치**: `playground/frontend/src/components/NodeDataViewer.tsx` (신규 생성 필요)
  - **구현 방향**: 노드 클릭 시 팝업으로 데이터 표시
  - **방법**: 문서의 "4. 노드 데이터 확인" 섹션 참조
- [ ] **워크플로우 저장/로드 UI**
  - **위치**: `WorkflowEditor` 컴포넌트 내부
  - **방법**: 저장/로드 버튼 추가, API 호출
- [ ] **Chat UI에 워크플로우 뷰 통합**
  - **위치**: `playground/frontend/src/app/chat/page.tsx`
  - **방법**: 워크플로우 뷰 토글 버튼, 패널 추가
- [ ] **`workflow_router.py` 생성 (워크플로우 CRUD)**
  - **위치**: `playground/backend/routers/workflow_router.py` (신규 생성 필요)
  - **구현 방향**:
    1. MongoDB `workflows` 컬렉션에 저장
    2. 워크플로우 ID로 조회/수정/삭제
  - **방법**: 문서의 "5. 워크플로우 저장/로드" 섹션 참조
- [ ] **`orchestrator.py`에 워크플로우 생성 로직 추가**
  - **통합 위치**: `services/orchestrator.py`의 `execute()` 메서드
  - **구현 방향**:
    1. Intent 분류 결과로 워크플로우 노드 자동 생성
    2. `WorkflowGraph` 활용 (이미 구현됨)
    3. 각 단계를 노드로 표현
  - **방법**: 문서의 "6. Orchestrator에서 워크플로우 생성" 섹션 참조
- [ ] **워크플로우 저장 (MongoDB)**
  - **통합 위치**: `workflow_router.py`의 `save_workflow` 엔드포인트
  - **방법**: MongoDB `workflows` 컬렉션에 저장
- [ ] **SSE 이벤트에 워크플로우 정보 포함**
  - **통합 위치**: `orchestrator.py`의 각 핸들러
  - **방법**: `AgenticEvent`에 `workflow_node_id` 필드 추가
- [ ] **Orchestrator 실행 → 워크플로우 노드 자동 생성**
  - **통합 위치**: `orchestrator.py`의 `execute()` 메서드 시작 부분
  - **방법**: Intent와 Tool 선택 결과로 노드 생성
- [ ] **실시간 진행 상황 → 노드 상태 업데이트**
  - **통합 위치**: 프론트엔드 `LiveWorkflowView` 컴포넌트
  - **방법**: SSE 이벤트 수신하여 노드 상태 업데이트
- [ ] **노드 클릭 → 데이터 확인**
  - **통합 위치**: `WorkflowEditor` 또는 `LiveWorkflowView`
  - **방법**: 노드 클릭 이벤트 핸들러에서 `NodeDataViewer` 모달 표시
- [ ] **워크플로우 템플릿 저장/재사용**
  - **통합 위치**: `workflow_router.py`
  - **방법**: 템플릿 플래그로 저장, 템플릿 목록 조회 API

---

## 🎯 우선순위

**중간**: 사용자 경험 개선, 디버깅 용이성 향상

---

## 💡 추가 기능 (선택)

### 1. 워크플로우 템플릿
- 자주 사용하는 워크플로우를 템플릿으로 저장
- 템플릿에서 빠르게 워크플로우 생성

### 2. 워크플로우 공유
- 워크플로우를 다른 사용자와 공유
- 커뮤니티 템플릿 라이브러리

### 3. 워크플로우 최적화
- 실행 시간 분석
- 병목 지점 시각화
- 자동 최적화 제안

### 4. 조건부 분기
- Decision 노드로 조건부 실행
- If-else 분기 시각화

---

## 🔗 관련 문서

- [16_PLAN_MODE.md](./16_PLAN_MODE.md): Plan 모드 (계획 검토)
- [02_AGENTIC_MODE.md](./02_AGENTIC_MODE.md): Agentic 모드 기본 구조
- [14_SEARCH_ARCHITECTURE.md](./14_SEARCH_ARCHITECTURE.md): 검색 시스템 구조

---

## 📚 참고 라이브러리

- **React Flow**: https://reactflow.dev/ (드래그 앤 드롭 그래프)
- **n8n**: https://n8n.io/ (워크플로우 자동화 플랫폼)
- **LangGraph**: https://langchain-ai.github.io/langgraph/ (그래프 기반 LLM 애플리케이션)
