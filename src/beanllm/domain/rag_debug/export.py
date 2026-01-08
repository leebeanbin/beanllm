"""
Exporter - RAG 디버그 리포트 내보내기
SOLID 원칙:
- SRP: 리포트 내보내기만 담당
- OCP: 새로운 포맷 추가 가능
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from beanllm.utils.logger import get_logger

logger = get_logger(__name__)


class DebugReportExporter:
    """
    디버그 리포트 내보내기

    책임:
    - 디버그 결과를 다양한 포맷으로 내보내기
    - JSON, Markdown, HTML 지원
    """

    @staticmethod
    def export_json(
        data: Dict[str, Any], output_path: str, pretty: bool = True
    ) -> str:
        """
        JSON 포맷으로 내보내기

        Args:
            data: 내보낼 데이터
            output_path: 출력 파일 경로
            pretty: Pretty print 여부

        Returns:
            str: 저장된 파일 경로
        """
        logger.info(f"Exporting debug report to JSON: {output_path}")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            else:
                json.dump(data, f, ensure_ascii=False, default=str)

        logger.info(f"JSON report saved: {output_file}")
        return str(output_file)

    @staticmethod
    def export_markdown(data: Dict[str, Any], output_path: str) -> str:
        """
        Markdown 포맷으로 내보내기

        Args:
            data: 내보낼 데이터
            output_path: 출력 파일 경로

        Returns:
            str: 저장된 파일 경로
        """
        logger.info(f"Exporting debug report to Markdown: {output_path}")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Generate Markdown
        md_lines = []
        md_lines.append("# RAG Debug Report")
        md_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_lines.append("\n---\n")

        # Session Info
        if "session" in data:
            session = data["session"]
            md_lines.append("## Session Information")
            md_lines.append(f"- **Session ID**: {session.get('session_id', 'N/A')}")
            md_lines.append(f"- **Session Name**: {session.get('session_name', 'N/A')}")
            md_lines.append(f"- **Created At**: {session.get('created_at', 'N/A')}")
            md_lines.append("")

        # Metadata
        if "metadata" in data:
            metadata = data["metadata"]
            md_lines.append("## Vector Store Metadata")
            md_lines.append(f"- **Documents**: {metadata.get('num_documents', 0)}")
            md_lines.append(f"- **Embeddings**: {metadata.get('num_embeddings', 0)}")
            md_lines.append(
                f"- **Embedding Dimension**: {metadata.get('embedding_dim', 0)}"
            )
            md_lines.append(
                f"- **VectorStore Type**: {metadata.get('vector_store_type', 'N/A')}"
            )
            md_lines.append("")

        # Embedding Analysis
        if "embedding_analysis" in data:
            analysis = data["embedding_analysis"]
            md_lines.append("## Embedding Analysis")
            md_lines.append(
                f"- **Method**: {analysis.get('method', 'N/A').upper()}"
            )
            md_lines.append(
                f"- **Components**: {analysis.get('n_components', 0)}"
            )

            if "cluster_stats" in analysis:
                stats = analysis["cluster_stats"]
                md_lines.append(
                    f"- **Clusters**: {stats.get('n_clusters', 0)}"
                )
                md_lines.append(f"- **Noise Points**: {stats.get('n_noise', 0)}")
                md_lines.append(
                    f"- **Noise Ratio**: {stats.get('noise_ratio', 0):.2%}"
                )

            if "silhouette_score" in analysis:
                md_lines.append(
                    f"- **Silhouette Score**: {analysis['silhouette_score']:.4f}"
                )

            md_lines.append("")

        # Chunk Validation
        if "chunk_validation" in data:
            validation = data["chunk_validation"]
            md_lines.append("## Chunk Validation")
            md_lines.append(
                f"- **Total Chunks**: {validation.get('total_chunks', 0)}"
            )
            md_lines.append(
                f"- **Valid Chunks**: {validation.get('valid_chunks', 0)}"
            )

            if "size_issues" in validation:
                md_lines.append(
                    f"- **Size Issues**: {len(validation['size_issues'])}"
                )

            if "duplicate_chunks" in validation:
                md_lines.append(
                    f"- **Duplicate Chunks**: {len(validation['duplicate_chunks'])}"
                )

            if "recommendations" in validation:
                md_lines.append("\n### Recommendations")
                for rec in validation["recommendations"]:
                    md_lines.append(f"- {rec}")

            md_lines.append("")

        # Parameter Tuning
        if "parameter_tuning" in data:
            tuning = data["parameter_tuning"]
            md_lines.append("## Parameter Tuning Results")

            if "best_params" in tuning:
                md_lines.append("\n### Best Parameters")
                for key, value in tuning["best_params"].items():
                    md_lines.append(f"- **{key}**: {value}")

            if "improvement_pct" in tuning:
                md_lines.append(
                    f"\n**Improvement**: {tuning['improvement_pct']:.2f}%"
                )

            md_lines.append("")

        # Write file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        logger.info(f"Markdown report saved: {output_file}")
        return str(output_file)

    @staticmethod
    def export_html(data: Dict[str, Any], output_path: str) -> str:
        """
        HTML 포맷으로 내보내기

        Args:
            data: 내보낼 데이터
            output_path: 출력 파일 경로

        Returns:
            str: 저장된 파일 경로
        """
        logger.info(f"Exporting debug report to HTML: {output_path}")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Generate HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Debug Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-left: 4px solid #4CAF50;
            padding-left: 10px;
        }}
        .metric {{
            display: inline-block;
            background-color: #f0f0f0;
            padding: 10px 15px;
            margin: 5px;
            border-radius: 5px;
        }}
        .metric-label {{
            font-weight: bold;
            color: #666;
        }}
        .metric-value {{
            color: #4CAF50;
            font-size: 1.2em;
        }}
        .recommendation {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
        }}
        .error {{
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 10px;
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 RAG Debug Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>📊 Summary</h2>
        <div class="metric">
            <span class="metric-label">Documents:</span>
            <span class="metric-value">{data.get('metadata', {}).get('num_documents', 0)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Embeddings:</span>
            <span class="metric-value">{data.get('metadata', {}).get('num_embeddings', 0)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Dimension:</span>
            <span class="metric-value">{data.get('metadata', {}).get('embedding_dim', 0)}</span>
        </div>

        <h2>🔬 Analysis Results</h2>
        <pre>{json.dumps(data, indent=2, default=str)}</pre>
    </div>
</body>
</html>
"""

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"HTML report saved: {output_file}")
        return str(output_file)

    @staticmethod
    def create_full_report(
        session_data: Dict[str, Any],
        output_dir: str,
        formats: Optional[list] = None,
    ) -> Dict[str, str]:
        """
        전체 디버그 리포트 생성 (여러 포맷)

        Args:
            session_data: 세션 데이터
            output_dir: 출력 디렉토리
            formats: 내보낼 포맷 목록 (None이면 모두)

        Returns:
            Dict[str, str]: 포맷별 파일 경로
        """
        formats = formats or ["json", "markdown", "html"]
        logger.info(f"Creating full debug report in formats: {formats}")

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        session_id = session_data.get("session", {}).get("session_id", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results = {}

        if "json" in formats:
            json_path = output_dir_path / f"debug_report_{session_id}_{timestamp}.json"
            results["json"] = DebugReportExporter.export_json(
                session_data, str(json_path)
            )

        if "markdown" in formats:
            md_path = output_dir_path / f"debug_report_{session_id}_{timestamp}.md"
            results["markdown"] = DebugReportExporter.export_markdown(
                session_data, str(md_path)
            )

        if "html" in formats:
            html_path = output_dir_path / f"debug_report_{session_id}_{timestamp}.html"
            results["html"] = DebugReportExporter.export_html(
                session_data, str(html_path)
            )

        logger.info(f"Full report created: {len(results)} files")
        return results
