"""
VisualBuilder - ASCII 워크플로우 디자이너
SOLID 원칙:
- SRP: 워크플로우 시각화만 담당
- OCP: 새로운 시각화 스타일 추가 가능
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .workflow_graph import NodeType, WorkflowGraph, WorkflowNode


class VisualBuilder:
    """
    ASCII 워크플로우 시각화 빌더

    책임:
    - 워크플로우를 ASCII 다이어그램으로 변환
    - 노드 배치 알고리즘 (레이어 기반)
    - 엣지 그리기

    Example:
        ```python
        workflow = WorkflowGraph(name="Research")
        # ... add nodes and edges ...

        builder = VisualBuilder(workflow)
        diagram = builder.build_diagram()
        print(diagram)
        ```

        Output:
        ```
        ┌─────────────┐
        │    START    │
        └──────┬──────┘
               │
        ┌──────▼──────────┐
        │   Researcher    │
        └──────┬──────────┘
               │
        ┌──────▼──────────┐
        │    Analyzer     │
        └──────┬──────────┘
               │
        ┌──────▼──────┐
        │     END     │
        └─────────────┘
        ```
    """

    # Box drawing characters
    BOX_TOP_LEFT = "┌"
    BOX_TOP_RIGHT = "┐"
    BOX_BOTTOM_LEFT = "└"
    BOX_BOTTOM_RIGHT = "┘"
    BOX_HORIZONTAL = "─"
    BOX_VERTICAL = "│"
    BOX_T_DOWN = "┬"
    BOX_T_UP = "┴"
    BOX_T_RIGHT = "├"
    BOX_T_LEFT = "┤"
    BOX_CROSS = "┼"

    # Arrows
    ARROW_DOWN = "▼"
    ARROW_RIGHT = "►"
    ARROW_UP = "▲"
    ARROW_LEFT = "◄"

    def __init__(self, workflow: WorkflowGraph) -> None:
        """
        Args:
            workflow: 시각화할 워크플로우
        """
        self.workflow = workflow
        self.layers: List[List[str]] = []  # Layered nodes
        self.node_positions: Dict[str, Tuple[int, int]] = {}  # (layer, offset)

    def build_diagram(
        self,
        style: str = "box",
        max_width: int = 80,
        show_config: bool = False,
    ) -> str:
        """
        다이어그램 생성

        Args:
            style: 스타일 ("box", "simple", "compact")
            max_width: 최대 너비
            show_config: 노드 설정 표시 여부

        Returns:
            str: ASCII 다이어그램
        """
        # Assign layers (topological sort)
        self._assign_layers()

        if style == "box":
            return self._build_box_diagram(show_config)
        elif style == "simple":
            return self._build_simple_diagram()
        elif style == "compact":
            return self._build_compact_diagram()
        else:
            return self._build_box_diagram(show_config)

    def _assign_layers(self) -> None:
        """노드를 레이어별로 배치 (BFS)"""
        if not self.workflow.nodes:
            return

        # Start nodes
        start_nodes = self.workflow.get_start_nodes()

        if not start_nodes:
            # No explicit start, use nodes with no incoming edges
            start_nodes = [
                nid for nid in self.workflow.nodes if not self.workflow.reverse_adjacency.get(nid)
            ]

        # BFS to assign layers
        visited = set()
        layer_map: Dict[str, int] = {}

        queue = [(nid, 0) for nid in start_nodes]

        while queue:
            node_id, layer = queue.pop(0)

            if node_id in visited:
                continue

            visited.add(node_id)
            layer_map[node_id] = max(layer, layer_map.get(node_id, 0))

            # Add children
            for edge_id in self.workflow.adjacency.get(node_id, []):
                edge = self.workflow.edges[edge_id]
                queue.append((edge.target, layer + 1))

        # Group by layers
        max_layer = max(layer_map.values()) if layer_map else 0
        self.layers = [[] for _ in range(max_layer + 1)]

        for node_id, layer in layer_map.items():
            self.layers[layer].append(node_id)

        # Store positions
        for layer_idx, layer_nodes in enumerate(self.layers):
            for offset, node_id in enumerate(layer_nodes):
                self.node_positions[node_id] = (layer_idx, offset)

    def _build_box_diagram(self, show_config: bool = False) -> str:
        """박스 스타일 다이어그램"""
        lines = []

        for layer_idx, layer_nodes in enumerate(self.layers):
            # Draw nodes in this layer
            for node_id in layer_nodes:
                node = self.workflow.nodes[node_id]

                # Node box
                node_lines = self._draw_node_box(node, show_config)
                lines.extend(node_lines)

                # Edges to next layer
                outgoing_edges = self.workflow.adjacency.get(node_id, [])
                if outgoing_edges:
                    # Draw connector
                    lines.append(self._center_text("│", 20))

        return "\n".join(lines)

    def _draw_node_box(self, node: WorkflowNode, show_config: bool = False) -> List[str]:
        """노드 박스 그리기"""
        lines = []

        # Box width
        label = node.name
        config_str = ""

        if show_config and node.config:
            config_items = [f"{k}={v}" for k, v in node.config.items()]
            config_str = ", ".join(config_items[:2])  # First 2 items

        content_width = max(len(label), len(config_str)) + 4
        box_width = max(content_width, 15)

        # Top border
        top = f"{self.BOX_TOP_LEFT}{self.BOX_HORIZONTAL * (box_width - 2)}{self.BOX_TOP_RIGHT}"
        lines.append(self._center_text(top, 20))

        # Label
        padded_label = label.center(box_width - 2)
        lines.append(self._center_text(f"{self.BOX_VERTICAL}{padded_label}{self.BOX_VERTICAL}", 20))

        # Config (if shown)
        if show_config and config_str:
            padded_config = config_str.center(box_width - 2)
            lines.append(
                self._center_text(f"{self.BOX_VERTICAL}{padded_config}{self.BOX_VERTICAL}", 20)
            )

        # Bottom border
        bottom = (
            f"{self.BOX_BOTTOM_LEFT}{self.BOX_HORIZONTAL * (box_width - 2)}{self.BOX_BOTTOM_RIGHT}"
        )
        lines.append(self._center_text(bottom, 20))

        return lines

    def _build_simple_diagram(self) -> str:
        """간단한 스타일 다이어그램"""
        lines = []

        for layer_idx, layer_nodes in enumerate(self.layers):
            for node_id in layer_nodes:
                node = self.workflow.nodes[node_id]

                # Simple format: [Name] (Type)
                node_str = f"[{node.name}] ({node.node_type.value})"
                lines.append(node_str)

                # Edge
                outgoing_edges = self.workflow.adjacency.get(node_id, [])
                if outgoing_edges:
                    lines.append("    ↓")

        return "\n".join(lines)

    def _build_compact_diagram(self) -> str:
        """컴팩트 스타일 (한 줄로)"""
        parts = []

        try:
            order = self.workflow.get_topological_order()
        except ValueError:
            order = list(self.workflow.nodes.keys())

        for i, node_id in enumerate(order):
            node = self.workflow.nodes[node_id]
            parts.append(node.name)

            if i < len(order) - 1:
                parts.append("→")

        return " ".join(parts)

    def _center_text(self, text: str, width: int) -> str:
        """텍스트 중앙 정렬"""
        text_len = len(text)
        if text_len >= width:
            return text

        padding = (width - text_len) // 2
        return " " * padding + text

    def build_mermaid_diagram(self) -> str:
        """
        Mermaid.js 다이어그램 생성

        Returns:
            str: Mermaid 코드

        Example:
            ```mermaid
            graph TD
                A[Start] --> B[Researcher]
                B --> C[Analyzer]
                C --> D[End]
            ```
        """
        lines = ["graph TD"]

        # Nodes
        for node_id, node in self.workflow.nodes.items():
            # Mermaid node shape based on type
            if node.node_type == NodeType.START:
                shape = f"({node.name})"
            elif node.node_type == NodeType.END:
                shape = f"({node.name})"
            elif node.node_type == NodeType.DECISION:
                shape = f"{{{node.name}}}"
            else:
                shape = f"[{node.name}]"

            # Use short ID for Mermaid
            short_id = node_id[:8]
            lines.append(f"    {short_id}{shape}")

        # Edges
        for edge_id, edge in self.workflow.edges.items():
            source_short = edge.source[:8]
            target_short = edge.target[:8]

            # Edge label (condition)
            if edge.condition.value != "always":
                label = f"|{edge.condition.value}|"
                lines.append(f"    {source_short} -->{label} {target_short}")
            else:
                lines.append(f"    {source_short} --> {target_short}")

        return "\n".join(lines)

    def build_code(self, language: str = "python") -> str:
        """
        워크플로우를 코드로 생성

        Args:
            language: 코드 언어 ("python" only for now)

        Returns:
            str: 코드

        Example:
            ```python
            workflow = WorkflowGraph(name="Research")
            start = workflow.add_node(NodeType.START, "start")
            research = workflow.add_node(NodeType.AGENT, "researcher", ...)
            ...
            ```
        """
        if language != "python":
            return "# Only Python code generation supported"

        lines = []
        lines.append("from beanllm.domain.orchestrator import WorkflowGraph, NodeType")
        lines.append("")
        lines.append("# Create workflow")
        lines.append(f'workflow = WorkflowGraph(name="{self.workflow.name}")')
        lines.append("")

        # Add nodes
        lines.append("# Add nodes")
        for node_id, node in self.workflow.nodes.items():
            config_str = repr(node.config) if node.config else "{}"
            lines.append(
                f"{node_id} = workflow.add_node("
                f"NodeType.{node.node_type.name}, "
                f'"{node.name}", '
                f"config={config_str})"
            )

        lines.append("")

        # Add edges
        lines.append("# Add edges")
        for edge in self.workflow.edges.values():
            lines.append(f"workflow.add_edge({edge.source}, {edge.target})")

        lines.append("")
        lines.append("# Execute")
        lines.append("# result = await workflow.execute(agents=..., task=...)")

        return "\n".join(lines)

    def get_statistics(self) -> Dict[str, any]:
        """워크플로우 통계"""
        node_type_counts = {}
        for node in self.workflow.nodes.values():
            node_type = node.node_type.value
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

        return {
            "num_nodes": len(self.workflow.nodes),
            "num_edges": len(self.workflow.edges),
            "num_layers": len(self.layers),
            "node_types": node_type_counts,
            "max_layer_width": max(len(layer) for layer in self.layers) if self.layers else 0,
            "start_nodes": len(self.workflow.get_start_nodes()),
            "end_nodes": len(self.workflow.get_end_nodes()),
        }

    def validate_workflow(self) -> List[str]:
        """
        워크플로우 검증

        Returns:
            List[str]: 검증 경고/에러 목록
        """
        warnings = []

        # Check for start nodes
        start_nodes = self.workflow.get_start_nodes()
        if not start_nodes:
            warnings.append("Warning: No START nodes found")

        # Check for end nodes
        end_nodes = self.workflow.get_end_nodes()
        if not end_nodes:
            warnings.append("Warning: No END nodes found")

        # Check for isolated nodes
        for node_id in self.workflow.nodes:
            in_edges = self.workflow.reverse_adjacency.get(node_id, [])
            out_edges = self.workflow.adjacency.get(node_id, [])

            if not in_edges and not out_edges:
                warnings.append(f"Warning: Isolated node: {node_id}")

        # Check for cycles
        try:
            self.workflow.get_topological_order()
        except ValueError:
            warnings.append("Error: Workflow contains cycles")

        return warnings


def create_simple_workflow(
    nodes: List[Tuple[str, NodeType]],
    name: str = "Simple Workflow",
) -> WorkflowGraph:
    """
    간단한 순차 워크플로우 생성

    Args:
        nodes: [(node_name, node_type), ...] 리스트
        name: 워크플로우 이름

    Returns:
        WorkflowGraph: 생성된 워크플로우

    Example:
        ```python
        workflow = create_simple_workflow([
            ("start", NodeType.START),
            ("process", NodeType.AGENT),
            ("end", NodeType.END),
        ])
        ```
    """
    workflow = WorkflowGraph(name=name)

    node_ids = []
    for node_name, node_type in nodes:
        node_id = workflow.add_node(node_type, node_name)
        node_ids.append(node_id)

    # Connect sequentially
    for i in range(len(node_ids) - 1):
        workflow.add_edge(node_ids[i], node_ids[i + 1])

    return workflow
