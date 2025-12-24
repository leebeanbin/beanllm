"""
RAG Pipeline Visualization - RAG 파이프라인 시각화
"""

from typing import Any, Dict, List, Optional


class RAGPipelineVisualizer:
    """RAG 파이프라인 시각화"""

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []

    def add_step(
        self,
        step_name: str,
        step_type: str,  # "load", "split", "embed", "store", "search", "llm"
        input_data: Any = None,
        output_data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        파이프라인 단계 추가

        Args:
            step_name: 단계 이름
            step_type: 단계 타입
            input_data: 입력 데이터
            output_data: 출력 데이터
            metadata: 추가 메타데이터
        """
        self.steps.append(
            {
                "name": step_name,
                "type": step_type,
                "input": input_data,
                "output": output_data,
                "metadata": metadata or {},
            }
        )

    def visualize_pipeline(self, format: str = "mermaid") -> str:
        """
        파이프라인 흐름 시각화

        Args:
            format: "mermaid" 또는 "graphviz"

        Returns:
            시각화 코드 (Mermaid 또는 DOT)
        """
        if format == "mermaid":
            return self._generate_mermaid()
        elif format == "graphviz":
            return self._generate_graphviz()
        else:
            raise ValueError(f"Unknown format: {format}")

    def _generate_mermaid(self) -> str:
        """Mermaid 다이어그램 생성"""
        lines = ["graph TD"]

        # 노드 정의
        node_ids = {}
        for i, step in enumerate(self.steps):
            node_id = f"step{i}"
            node_ids[step["name"]] = node_id

            # 노드 스타일 (타입별)
            style = self._get_node_style(step["type"])
            label = f"{step['name']}\\n({step['type']})"

            lines.append(f'    {node_id}["{label}"]')
            if style:
                lines.append(f"    style {node_id} {style}")

        # 엣지 정의
        for i in range(len(self.steps) - 1):
            current_id = f"step{i}"
            next_id = f"step{i + 1}"
            lines.append(f"    {current_id} --> {next_id}")

        return "\n".join(lines)

    def _get_node_style(self, step_type: str) -> str:
        """노드 스타일 (타입별)"""
        styles = {
            "load": "fill:#e1f5ff",
            "split": "fill:#fff4e1",
            "embed": "fill:#ffe1f5",
            "store": "fill:#e1ffe1",
            "search": "fill:#f5e1ff",
            "llm": "fill:#ffe1e1",
        }
        color = styles.get(step_type, "")
        return f"fill:{color}" if color else ""

    def _generate_graphviz(self) -> str:
        """Graphviz DOT 코드 생성"""
        lines = ["digraph RAGPipeline {"]
        lines.append("    rankdir=LR;")
        lines.append("    node [shape=box, style=rounded];")

        # 노드 정의
        for i, step in enumerate(self.steps):
            node_id = f"step{i}"
            label = f"{step['name']}\\n({step['type']})"
            color = self._get_graphviz_color(step["type"])
            lines.append(f'    {node_id} [label="{label}", fillcolor="{color}", style="filled"];')

        # 엣지 정의
        for i in range(len(self.steps) - 1):
            current_id = f"step{i}"
            next_id = f"step{i + 1}"
            lines.append(f"    {current_id} -> {next_id};")

        lines.append("}")
        return "\n".join(lines)

    def _get_graphviz_color(self, step_type: str) -> str:
        """Graphviz 색상 (타입별)"""
        colors = {
            "load": "lightblue",
            "split": "lightyellow",
            "embed": "lightpink",
            "store": "lightgreen",
            "search": "lavender",
            "llm": "lightcoral",
        }
        return colors.get(step_type, "lightgray")

    def export_graph(
        self,
        output_path: str,
        format: str = "png",
        diagram_format: str = "graphviz",
    ):
        """
        그래프를 이미지로 내보내기

        Args:
            output_path: 출력 경로
            format: 이미지 포맷 ("png", "svg", "pdf")
            diagram_format: 다이어그램 포맷 ("mermaid" 또는 "graphviz")
        """
        if diagram_format == "mermaid":
            # Mermaid는 온라인 서비스 또는 mermaid-cli 필요
            # 여기서는 Graphviz 사용
            diagram_code = self._generate_graphviz()
        else:
            diagram_code = self._generate_graphviz()

        try:
            import graphviz

            # Graphviz로 렌더링
            graph = graphviz.Source(diagram_code)
            graph.render(output_path, format=format, cleanup=True)

        except ImportError:
            raise ImportError(
                "Graphviz 필요:\n"
                "  pip install graphviz\n"
                "  그리고 시스템에 Graphviz 설치 필요:\n"
                "  - macOS: brew install graphviz\n"
                "  - Ubuntu: sudo apt-get install graphviz\n"
                "  - Windows: https://graphviz.org/download/"
            )

    def clear(self):
        """단계 초기화"""
        self.steps.clear()
