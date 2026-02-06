#!/usr/bin/env python3
"""
Code Duplication Analyzer for beanllm
Analyzes patterns like caching, rate limiting, error handling, etc.
"""

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


class DuplicationAnalyzer(ast.NodeVisitor):
    """Analyzes AST for common duplication patterns"""

    def __init__(self):
        self.caching_patterns = []
        self.rate_limiting_patterns = []
        self.event_streaming_patterns = []
        self.error_handling_patterns = []
        self.validation_patterns = []
        self.current_file = None
        self.current_function = None

    def analyze_file(self, file_path: Path):
        """Analyze a single file for duplication patterns"""
        self.current_file = file_path
        try:
            with open(file_path, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(file_path))
                self.visit(tree)
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"⚠️  Error parsing {file_path}: {e}")

    def visit_AsyncFunctionDef(self, node):
        """Visit async function definitions"""
        self.current_function = node.name
        self._check_patterns(node)
        self.generic_visit(node)
        self.current_function = None

    def visit_FunctionDef(self, node):
        """Visit regular function definitions"""
        self.current_function = node.name
        self._check_patterns(node)
        self.generic_visit(node)
        self.current_function = None

    def _check_patterns(self, node):
        """Check for various duplication patterns in function body"""
        func_body = ast.unparse(node)

        # Pattern 1: Caching pattern
        if re.search(r"cache\.get\(|cache_key\s*=|cached\s*=", func_body):
            self.caching_patterns.append(
                {
                    "file": str(self.current_file),
                    "function": self.current_function,
                    "lines": len(func_body.split("\n")),
                    "pattern": "caching",
                }
            )

        # Pattern 2: Rate limiting pattern
        if re.search(r"rate_limit|RateLimiter|acquire\(|wait\(", func_body):
            self.rate_limiting_patterns.append(
                {
                    "file": str(self.current_file),
                    "function": self.current_function,
                    "lines": len(func_body.split("\n")),
                    "pattern": "rate_limiting",
                }
            )

        # Pattern 3: Event streaming pattern
        if re.search(r"event.*publish|log_event|EventPublisher|EventLogger", func_body):
            self.event_streaming_patterns.append(
                {
                    "file": str(self.current_file),
                    "function": self.current_function,
                    "lines": len(func_body.split("\n")),
                    "pattern": "event_streaming",
                }
            )

        # Pattern 4: Error handling pattern (try-except blocks)
        try_count = func_body.count("try:")
        except_count = func_body.count("except")
        if try_count >= 2 or except_count >= 2:
            self.error_handling_patterns.append(
                {
                    "file": str(self.current_file),
                    "function": self.current_function,
                    "lines": len(func_body.split("\n")),
                    "try_blocks": try_count,
                    "except_blocks": except_count,
                    "pattern": "error_handling",
                }
            )

        # Pattern 5: Validation pattern
        if re.search(r"if not.*raise|ValueError|TypeError.*validate", func_body):
            self.validation_patterns.append(
                {
                    "file": str(self.current_file),
                    "function": self.current_function,
                    "lines": len(func_body.split("\n")),
                    "pattern": "validation",
                }
            )


def analyze_directory(base_path: Path) -> DuplicationAnalyzer:
    """Analyze all Python files in directory"""
    analyzer = DuplicationAnalyzer()

    python_files = list(base_path.rglob("*.py"))
    print(f"📁 Scanning {len(python_files)} Python files...\n")

    for py_file in python_files:
        # Skip __pycache__ and test files for now
        if "__pycache__" in str(py_file) or "test_" in py_file.name:
            continue
        analyzer.analyze_file(py_file)

    return analyzer


def print_report(analyzer: DuplicationAnalyzer):
    """Print detailed duplication report"""

    print("=" * 80)
    print("🔍 CODE DEDUPLICATION REPORT - beanllm")
    print("=" * 80)
    print()

    # Statistics
    total_patterns = (
        len(analyzer.caching_patterns)
        + len(analyzer.rate_limiting_patterns)
        + len(analyzer.event_streaming_patterns)
        + len(analyzer.error_handling_patterns)
        + len(analyzer.validation_patterns)
    )

    print("📊 STATISTICS:")
    print(f"  Total duplication patterns found: {total_patterns}")
    print()

    # Pattern 1: Caching
    if analyzer.caching_patterns:
        print("=" * 80)
        print(f"1️⃣  CACHING PATTERN ({len(analyzer.caching_patterns)} occurrences)")
        print("=" * 80)
        print()
        total_lines = sum(p["lines"] for p in analyzer.caching_patterns)
        print(f"📍 Locations ({len(analyzer.caching_patterns)} files):")
        for pattern in analyzer.caching_patterns[:10]:  # Show first 10
            print(f"   - {pattern['file']}:{pattern['function']} ({pattern['lines']} lines)")
        if len(analyzer.caching_patterns) > 10:
            print(f"   ... and {len(analyzer.caching_patterns) - 10} more")
        print()
        print(f"📏 Total lines: {total_lines}")
        print("💡 Recommendation: Use @with_cache or @with_distributed_features decorator")
        print(
            f"   Estimated savings: {total_lines} → ~{len(analyzer.caching_patterns) * 5} lines ({int((total_lines - len(analyzer.caching_patterns) * 5) / total_lines * 100)}% reduction)"
        )
        print()

    # Pattern 2: Rate Limiting
    if analyzer.rate_limiting_patterns:
        print("=" * 80)
        print(f"2️⃣  RATE LIMITING PATTERN ({len(analyzer.rate_limiting_patterns)} occurrences)")
        print("=" * 80)
        print()
        total_lines = sum(p["lines"] for p in analyzer.rate_limiting_patterns)
        print(f"📍 Locations ({len(analyzer.rate_limiting_patterns)} files):")
        for pattern in analyzer.rate_limiting_patterns[:10]:
            print(f"   - {pattern['file']}:{pattern['function']} ({pattern['lines']} lines)")
        if len(analyzer.rate_limiting_patterns) > 10:
            print(f"   ... and {len(analyzer.rate_limiting_patterns) - 10} more")
        print()
        print(f"📏 Total lines: {total_lines}")
        print("💡 Recommendation: Use @with_distributed_features(enable_rate_limiting=True)")
        print(
            f"   Estimated savings: {total_lines} → ~{len(analyzer.rate_limiting_patterns) * 5} lines ({int((total_lines - len(analyzer.rate_limiting_patterns) * 5) / total_lines * 100)}% reduction)"
        )
        print()

    # Pattern 3: Event Streaming
    if analyzer.event_streaming_patterns:
        print("=" * 80)
        print(f"3️⃣  EVENT STREAMING PATTERN ({len(analyzer.event_streaming_patterns)} occurrences)")
        print("=" * 80)
        print()
        total_lines = sum(p["lines"] for p in analyzer.event_streaming_patterns)
        print(f"📍 Locations ({len(analyzer.event_streaming_patterns)} files):")
        for pattern in analyzer.event_streaming_patterns[:10]:
            print(f"   - {pattern['file']}:{pattern['function']} ({pattern['lines']} lines)")
        if len(analyzer.event_streaming_patterns) > 10:
            print(f"   ... and {len(analyzer.event_streaming_patterns) - 10} more")
        print()
        print(f"📏 Total lines: {total_lines}")
        print("💡 Recommendation: Use @with_distributed_features(enable_event_streaming=True)")
        print(
            f"   Estimated savings: {total_lines} → ~{len(analyzer.event_streaming_patterns) * 3} lines"
        )
        print()

    # Pattern 4: Error Handling
    if analyzer.error_handling_patterns:
        print("=" * 80)
        print(f"4️⃣  ERROR HANDLING PATTERN ({len(analyzer.error_handling_patterns)} occurrences)")
        print("=" * 80)
        print()
        total_lines = sum(p["lines"] for p in analyzer.error_handling_patterns)
        print(f"📍 Locations ({len(analyzer.error_handling_patterns)} files):")
        for pattern in analyzer.error_handling_patterns[:10]:
            try_blocks = pattern.get("try_blocks", 0)
            except_blocks = pattern.get("except_blocks", 0)
            print(
                f"   - {pattern['file']}:{pattern['function']} ({pattern['lines']} lines, {try_blocks} try, {except_blocks} except)"
            )
        if len(analyzer.error_handling_patterns) > 10:
            print(f"   ... and {len(analyzer.error_handling_patterns) - 10} more")
        print()
        print(f"📏 Total lines: {total_lines}")
        print("💡 Recommendation: Use @provider_error_handler or custom error decorators")
        print()

    # Pattern 5: Validation
    if analyzer.validation_patterns:
        print("=" * 80)
        print(f"5️⃣  VALIDATION PATTERN ({len(analyzer.validation_patterns)} occurrences)")
        print("=" * 80)
        print()
        total_lines = sum(p["lines"] for p in analyzer.validation_patterns)
        print(f"📍 Locations ({len(analyzer.validation_patterns)} files):")
        for pattern in analyzer.validation_patterns[:10]:
            print(f"   - {pattern['file']}:{pattern['function']} ({pattern['lines']} lines)")
        if len(analyzer.validation_patterns) > 10:
            print(f"   ... and {len(analyzer.validation_patterns) - 10} more")
        print()
        print(f"📏 Total lines: {total_lines}")
        print("💡 Recommendation: Extract to validation utility functions or decorators")
        print()

    # Summary
    print("=" * 80)
    print("🎯 RECOMMENDED ACTIONS")
    print("=" * 80)
    print()

    actions = []
    if analyzer.caching_patterns:
        actions.append(
            f"1. Apply @with_cache decorator to {len(analyzer.caching_patterns)} methods"
        )
    if analyzer.rate_limiting_patterns:
        actions.append(
            f"2. Apply @with_distributed_features to {len(analyzer.rate_limiting_patterns)} methods"
        )
    if analyzer.event_streaming_patterns:
        actions.append(
            f"3. Consolidate event streaming in {len(analyzer.event_streaming_patterns)} methods"
        )
    if analyzer.error_handling_patterns:
        actions.append(
            f"4. Refactor error handling in {len(analyzer.error_handling_patterns)} methods"
        )
    if analyzer.validation_patterns:
        actions.append(
            f"5. Extract validation logic from {len(analyzer.validation_patterns)} methods"
        )

    for action in actions:
        print(f"   {action}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    import sys

    base_path = Path("src/beanllm")
    if len(sys.argv) > 1:
        base_path = Path(sys.argv[1])

    analyzer = analyze_directory(base_path)
    print_report(analyzer)
