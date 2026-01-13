"""
RAG Debug Commands - Rich CLI 인터페이스
SOLID 원칙:
- SRP: CLI 명령어 처리만 담당
- DIP: Facade 인터페이스에 의존
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from beanllm.facade.advanced.rag_debug_facade import RAGDebug
from beanllm.ui.components import Badge, Divider, OutputBlock, StatusIcon
from beanllm.ui.console import get_console
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class RAGDebugCommands:
    """
    RAG 디버깅 CLI 명령어 모음

    책임:
    - CLI 명령어 파싱 및 실행
    - Rich 포맷팅된 출력
    - 진행상황 표시

    Example:
        ```python
        # In REPL
        commands = RAGDebugCommands(vector_store)

        # Start session
        await commands.cmd_start(session_name="my_debug")

        # Analyze embeddings
        await commands.cmd_analyze(method="umap", n_clusters=5)

        # Validate chunks
        await commands.cmd_validate()

        # Export report
        await commands.cmd_export(output_dir="./reports")
        ```
    """

    def __init__(
        self,
        vector_store: Any = None,
        console: Optional[Console] = None,
    ) -> None:
        """
        Args:
            vector_store: VectorStore 인스턴스 (optional, cmd_start에서 설정 가능)
            console: Rich Console (optional)
        """
        self.vector_store = vector_store
        self.console = console or get_console()
        self._debug: Optional[RAGDebug] = None
        self._session_active = False

    # ========================================
    # Command: Start Debug Session
    # ========================================

    async def cmd_start(
        self,
        vector_store: Any = None,
        session_name: Optional[str] = None,
    ) -> None:
        """
        디버그 세션 시작

        Args:
            vector_store: VectorStore 인스턴스
            session_name: 세션 이름 (optional)

        Example:
            ```
            await cmd_start(vector_store=my_store, session_name="prod_debug")
            ```
        """
        if vector_store:
            self.vector_store = vector_store

        if not self.vector_store:
            self.console.print(
                f"{StatusIcon.error()} [red]VectorStore가 제공되지 않았습니다.[/red]"
            )
            return

        self.console.print(f"\n{StatusIcon.LOADING} [cyan]디버그 세션 시작 중...[/cyan]")

        try:
            # Create RAGDebug instance
            self._debug = RAGDebug(
                vector_store=self.vector_store,
                session_name=session_name,
            )

            # Start session
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]세션 초기화 중...[/cyan]"),
                console=self.console,
                transient=True,
            ) as progress:
                progress.add_task("Starting", total=None)
                response = await self._debug.start()

            self._session_active = True

            # Display session info
            self._display_session_info(response)

        except Exception as e:
            self.console.print(f"{StatusIcon.error()} [red]세션 시작 실패: {e}[/red]")
            logger.error(f"Failed to start debug session: {e}")

    def _display_session_info(self, response: Any) -> None:
        """세션 정보 표시"""
        # Create info table
        table = Table(
            title=f"🔍 RAG Debug Session: {response.session_name or 'Unnamed'}",
            title_style="bold cyan",
            box=box.ROUNDED,
            show_header=False,
        )

        table.add_column("Key", style="bold cyan", width=20)
        table.add_column("Value", style="white")

        table.add_row("Session ID", response.session_id[:12] + "...")
        table.add_row("Status", f"{Badge.success('ACTIVE')}")
        table.add_row("Documents", f"{response.num_documents:,}")
        table.add_row("Embeddings", f"{response.num_embeddings:,}")
        table.add_row("Embedding Dim", str(response.embedding_dim))
        table.add_row("Created At", response.created_at)

        self.console.print()
        self.console.print(table)
        self.console.print()

    # ========================================
    # Command: Analyze Embeddings
    # ========================================

    async def cmd_analyze(
        self,
        method: str = "umap",
        n_clusters: int = 5,
        detect_outliers: bool = True,
        sample_size: Optional[int] = None,
    ) -> None:
        """
        Embedding 분석 (UMAP/t-SNE + 클러스터링)

        Args:
            method: 차원 축소 방법 ("umap" or "tsne")
            n_clusters: 클러스터 수
            detect_outliers: 이상치 탐지 여부
            sample_size: 샘플 크기 (None이면 전체)

        Example:
            ```
            await cmd_analyze(method="umap", n_clusters=5)
            ```
        """
        if not self._check_session():
            return

        self.console.print(
            f"\n{StatusIcon.LOADING} [cyan]Embedding 분석 중... (method={method.upper()}, clusters={n_clusters})[/cyan]"
        )

        try:
            # Run analysis
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[cyan]{method.upper()} 차원 축소 및 클러스터링...[/cyan]"),
                console=self.console,
                transient=True,
            ) as progress:
                progress.add_task("Analyzing", total=None)
                response = await self._debug.analyze_embeddings(
                    method=method,
                    n_clusters=n_clusters,
                    detect_outliers=detect_outliers,
                    sample_size=sample_size,
                )

            # Display results
            self._display_embedding_analysis(response)

        except ImportError:
            self.console.print(
                f"{StatusIcon.error()} [red]고급 기능을 사용하려면 추가 패키지가 필요합니다:[/red]"
            )
            self.console.print(
                f"  [yellow]pip install beanllm[advanced][/yellow]"
            )
        except Exception as e:
            self.console.print(f"{StatusIcon.error()} [red]분석 실패: {e}[/red]")
            logger.error(f"Embedding analysis failed: {e}")

    def _display_embedding_analysis(self, response: Any) -> None:
        """Embedding 분석 결과 표시"""
        # Summary table
        table = Table(
            title=f"📊 Embedding Analysis ({response.method.upper()})",
            title_style="bold green",
            box=box.ROUNDED,
            show_header=False,
        )

        table.add_column("Metric", style="bold cyan", width=25)
        table.add_column("Value", style="white")

        table.add_row("Clusters Found", str(response.num_clusters))
        table.add_row("Outliers Detected", str(len(response.outliers)))
        table.add_row(
            "Silhouette Score",
            f"{response.silhouette_score:.4f}" if response.silhouette_score else "N/A",
        )

        # Cluster sizes
        cluster_sizes_str = ", ".join(
            f"C{k}: {v}" for k, v in sorted(response.cluster_sizes.items())
        )
        table.add_row("Cluster Sizes", cluster_sizes_str)

        self.console.print()
        self.console.print(table)

        # Quality assessment
        if response.silhouette_score:
            self._display_quality_assessment(response.silhouette_score)

        self.console.print()

    def _display_quality_assessment(self, silhouette_score: float) -> None:
        """클러스터링 품질 평가 표시"""
        self.console.print()
        self.console.print("[bold]Clustering Quality:[/bold]")

        if silhouette_score > 0.7:
            assessment = f"{StatusIcon.success()} Excellent (강력한 클러스터 구조)"
            color = "green"
        elif silhouette_score > 0.5:
            assessment = f"{StatusIcon.success()} Good (명확한 클러스터)"
            color = "cyan"
        elif silhouette_score > 0.25:
            assessment = f"{StatusIcon.warning()} Fair (약한 클러스터 구조)"
            color = "yellow"
        else:
            assessment = f"{StatusIcon.error()} Poor (클러스터가 불명확)"
            color = "red"

        self.console.print(f"  [{color}]{assessment}[/{color}]")

    # ========================================
    # Command: Validate Chunks
    # ========================================

    async def cmd_validate(
        self,
        size_threshold: int = 2000,
        check_size: bool = True,
        check_overlap: bool = True,
        check_metadata: bool = True,
        check_duplicates: bool = True,
    ) -> None:
        """
        청크 검증 (크기, 중복, 메타데이터)

        Args:
            size_threshold: 최대 청크 크기
            check_size: 크기 검증 여부
            check_overlap: Overlap 검증 여부
            check_metadata: 메타데이터 검증 여부
            check_duplicates: 중복 검증 여부

        Example:
            ```
            await cmd_validate(size_threshold=2000)
            ```
        """
        if not self._check_session():
            return

        self.console.print(f"\n{StatusIcon.LOADING} [cyan]청크 검증 중...[/cyan]")

        try:
            # Run validation
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]크기, 중복, 메타데이터 검증...[/cyan]"),
                console=self.console,
                transient=True,
            ) as progress:
                progress.add_task("Validating", total=None)
                response = await self._debug.validate_chunks(
                    size_threshold=size_threshold,
                    check_size=check_size,
                    check_overlap=check_overlap,
                    check_metadata=check_metadata,
                    check_duplicates=check_duplicates,
                )

            # Display results
            self._display_chunk_validation(response)

        except Exception as e:
            self.console.print(f"{StatusIcon.error()} [red]검증 실패: {e}[/red]")
            logger.error(f"Chunk validation failed: {e}")

    def _display_chunk_validation(self, response: Any) -> None:
        """청크 검증 결과 표시"""
        # Summary table
        table = Table(
            title="📝 Chunk Validation Results",
            title_style="bold blue",
            box=box.ROUNDED,
            show_header=False,
        )

        table.add_column("Metric", style="bold cyan", width=25)
        table.add_column("Value", style="white")

        table.add_row("Total Chunks", f"{response.total_chunks:,}")
        table.add_row("Valid Chunks", f"{response.valid_chunks:,}")
        table.add_row("Issues Found", str(len(response.issues)))
        table.add_row("Duplicate Chunks", str(len(response.duplicate_chunks)))

        self.console.print()
        self.console.print(table)

        # Issues
        if response.issues:
            self.console.print()
            self.console.print(f"{StatusIcon.warning()} [yellow bold]Issues Found:[/yellow bold]")
            for issue in response.issues[:10]:  # Show first 10
                self.console.print(f"  • [yellow]{issue}[/yellow]")
            if len(response.issues) > 10:
                self.console.print(f"  [dim]... and {len(response.issues) - 10} more[/dim]")

        # Recommendations
        if response.recommendations:
            self.console.print()
            self.console.print(f"{StatusIcon.info()} [cyan bold]Recommendations:[/cyan bold]")
            for rec in response.recommendations:
                self.console.print(f"  💡 [cyan]{rec}[/cyan]")

        self.console.print()

    # ========================================
    # Command: Tune Parameters
    # ========================================

    async def cmd_tune(
        self,
        parameters: Dict[str, Any],
        test_queries: Optional[List[str]] = None,
    ) -> None:
        """
        파라미터 실시간 튜닝

        Args:
            parameters: 테스트할 파라미터 (예: {"top_k": 10, "score_threshold": 0.7})
            test_queries: 테스트 쿼리 목록

        Example:
            ```
            await cmd_tune(
                parameters={"top_k": 10, "score_threshold": 0.7},
                test_queries=["What is RAG?", "How does it work?"]
            )
            ```
        """
        if not self._check_session():
            return

        self.console.print(
            f"\n{StatusIcon.LOADING} [cyan]파라미터 튜닝 중... {parameters}[/cyan]"
        )

        try:
            # Run tuning
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]파라미터 테스트 및 비교...[/cyan]"),
                console=self.console,
                transient=True,
            ) as progress:
                progress.add_task("Tuning", total=None)
                response = await self._debug.tune_parameters(
                    parameters=parameters,
                    test_queries=test_queries,
                )

            # Display results
            self._display_tuning_results(response)

        except Exception as e:
            self.console.print(f"{StatusIcon.error()} [red]튜닝 실패: {e}[/red]")
            logger.error(f"Parameter tuning failed: {e}")

    def _display_tuning_results(self, response: Any) -> None:
        """파라미터 튜닝 결과 표시"""
        # Summary table
        table = Table(
            title="⚙️  Parameter Tuning Results",
            title_style="bold magenta",
            box=box.ROUNDED,
            show_header=False,
        )

        table.add_column("Metric", style="bold cyan", width=25)
        table.add_column("Value", style="white")

        table.add_row("New Parameters", str(response.parameters))
        table.add_row("Average Score", f"{response.avg_score:.4f}")

        # Comparison
        if response.comparison_with_baseline:
            comparison = response.comparison_with_baseline
            improvement = comparison.get("improvement_pct", 0.0)

            improvement_str = f"{improvement:+.2f}%"
            if improvement > 5:
                improvement_str = f"[green]{improvement_str} {StatusIcon.SUCCESS}[/green]"
            elif improvement < -5:
                improvement_str = f"[red]{improvement_str} {StatusIcon.ERROR}[/red]"
            else:
                improvement_str = f"[yellow]{improvement_str}[/yellow]"

            table.add_row("vs Baseline", improvement_str)

        self.console.print()
        self.console.print(table)

        # Recommendations
        if response.recommendations:
            self.console.print()
            self.console.print(f"{StatusIcon.info()} [cyan bold]Recommendations:[/cyan bold]")
            for rec in response.recommendations:
                self.console.print(f"  {rec}")

        self.console.print()

    # ========================================
    # Command: Export Report
    # ========================================

    async def cmd_export(
        self,
        output_dir: str,
        formats: Optional[List[str]] = None,
    ) -> None:
        """
        디버그 리포트 내보내기

        Args:
            output_dir: 출력 디렉토리
            formats: 내보낼 포맷 목록 (None이면 ["json", "markdown", "html"])

        Example:
            ```
            await cmd_export(output_dir="./reports", formats=["json", "markdown"])
            ```
        """
        if not self._check_session():
            return

        formats = formats or ["json", "markdown", "html"]

        self.console.print(
            f"\n{StatusIcon.LOADING} [cyan]리포트 내보내기 중... (formats={formats})[/cyan]"
        )

        try:
            # Export report
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]리포트 생성 중...[/cyan]"),
                console=self.console,
                transient=True,
            ) as progress:
                progress.add_task("Exporting", total=None)
                results = await self._debug.export_report(
                    output_dir=output_dir,
                    formats=formats,
                )

            # Display results
            self._display_export_results(results)

        except Exception as e:
            self.console.print(f"{StatusIcon.error()} [red]내보내기 실패: {e}[/red]")
            logger.error(f"Report export failed: {e}")

    def _display_export_results(self, results: Dict[str, str]) -> None:
        """리포트 내보내기 결과 표시"""
        self.console.print()
        self.console.print(
            f"{StatusIcon.success()} [green bold]리포트가 성공적으로 내보내졌습니다![/green bold]"
        )
        self.console.print()

        # Files table
        table = Table(
            title="📁 Exported Files",
            title_style="bold green",
            box=box.ROUNDED,
        )

        table.add_column("Format", style="bold cyan")
        table.add_column("File Path", style="white")

        for fmt, path in results.items():
            table.add_row(fmt.upper(), path)

        self.console.print(table)
        self.console.print()

    # ========================================
    # Command: Full Analysis (One-Stop)
    # ========================================

    async def cmd_run_all(
        self,
        analyze_embeddings: bool = True,
        validate_chunks: bool = True,
        tune_parameters: bool = False,
        tuning_params: Optional[Dict[str, Any]] = None,
        test_queries: Optional[List[str]] = None,
    ) -> None:
        """
        전체 분석 실행 (원스톱)

        Args:
            analyze_embeddings: Embedding 분석 실행 여부
            validate_chunks: 청크 검증 실행 여부
            tune_parameters: 파라미터 튜닝 실행 여부
            tuning_params: 튜닝할 파라미터
            test_queries: 테스트 쿼리

        Example:
            ```
            await cmd_run_all(
                analyze_embeddings=True,
                validate_chunks=True,
                tune_parameters=True,
                tuning_params={"top_k": 10},
                test_queries=["test query"]
            )
            ```
        """
        if not self._check_session():
            return

        self.console.print()
        self.console.print(
            Panel(
                "[bold cyan]전체 RAG 디버그 분석 시작[/bold cyan]",
                box=box.DOUBLE,
                style="cyan",
            )
        )

        try:
            # Run full analysis
            results = await self._debug.run_full_analysis(
                analyze_embeddings=analyze_embeddings,
                validate_chunks=validate_chunks,
                tune_parameters=tune_parameters,
                tuning_params=tuning_params,
                test_queries=test_queries,
            )

            # Display summary
            self._display_full_analysis_summary(results)

        except Exception as e:
            self.console.print(f"{StatusIcon.error()} [red]전체 분석 실패: {e}[/red]")
            logger.error(f"Full analysis failed: {e}")

    def _display_full_analysis_summary(self, results: Dict[str, Any]) -> None:
        """전체 분석 요약 표시"""
        self.console.print()
        self.console.print(Divider.thick())
        self.console.print("[bold green]✅ 전체 분석 완료![/bold green]")
        self.console.print(Divider.thick())
        self.console.print()

        # Summary
        completed = []
        if "embedding_analysis" in results:
            completed.append("📊 Embedding Analysis")
        if "chunk_validation" in results:
            completed.append("📝 Chunk Validation")
        if "parameter_tuning" in results:
            completed.append("⚙️  Parameter Tuning")

        for item in completed:
            self.console.print(f"{StatusIcon.success()} {item}")

        self.console.print()

    # ========================================
    # Utilities
    # ========================================

    def _check_session(self) -> bool:
        """세션 활성화 확인"""
        if not self._session_active or not self._debug:
            self.console.print(
                f"{StatusIcon.error()} [red]활성 세션이 없습니다. 먼저 'cmd_start()'를 실행하세요.[/red]"
            )
            return False
        return True
