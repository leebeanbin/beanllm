#!/usr/bin/env python3
"""
Automated script to refactor Domain → Infrastructure dependencies.

This script updates Domain layer classes to use Protocol injection instead
of directly importing from Infrastructure layer.
"""

import re
from pathlib import Path
from typing import Dict, List

# Mapping of infrastructure imports to protocols
IMPORT_TO_PROTOCOL_MAP = {
    "get_event_logger": "EventLoggerProtocol",
    "get_event_bus": "EventBusProtocol",
    "get_lock_manager": "LockManagerProtocol",
    "get_rate_limiter": "RateLimiterProtocol",
    "get_distributed_cache": "CacheProtocol",
    "get_cache": "CacheProtocol",
    "BatchProcessor": "BatchProcessorProtocol",
    "ConcurrencyController": "ConcurrencyControllerProtocol",
    "get_distributed_config": "DistributedConfigProtocol",
}


def add_protocol_imports(file_path: Path, protocols_needed: List[str]) -> str:
    """Add protocol imports to TYPE_CHECKING block."""
    with open(file_path) as f:
        content = f.read()

    # Check if TYPE_CHECKING already exists
    if "TYPE_CHECKING" in content:
        # Find the TYPE_CHECKING block and add protocols
        pattern = r"if TYPE_CHECKING:(.*?)(?=\nelse:|\n\n[^from\s])"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            existing_imports = match.group(1)
            # Add new protocol imports
            protocols_import = f"from beanllm.domain.protocols import {', '.join(protocols_needed)}"
            if protocols_import not in existing_imports:
                # Insert after last 'from' statement in TYPE_CHECKING
                updated_block = existing_imports.rstrip() + f"\n    {protocols_import}"
                content = content.replace(match.group(0), f"if TYPE_CHECKING:{updated_block}")
    else:
        # Add TYPE_CHECKING block after other imports
        protocols_import = f"from beanllm.domain.protocols import {', '.join(protocols_needed)}"
        type_checking_block = f"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    {protocols_import}
"""
        # Find the last import statement
        import_pattern = r"(from .+ import .+\n)"
        matches = list(re.finditer(import_pattern, content))
        if matches:
            last_import = matches[-1]
            insert_pos = last_import.end()
            content = content[:insert_pos] + type_checking_block + content[insert_pos:]

    return content


def refactor_infrastructure_import(content: str, import_name: str, protocol_name: str) -> str:
    """
    Refactor a specific infrastructure import to use protocol injection.

    Example:
        from beanllm.infrastructure.distributed import get_event_logger
        event_logger = get_event_logger()
        →
        # event_logger is injected via __init__(event_logger=...)
        if self._event_logger:
            ...
    """
    # Remove the infrastructure import line
    import_pattern = rf"from beanllm\.infrastructure.*import.*{import_name}"
    content = re.sub(import_pattern, f"# {import_name} is now injected via {protocol_name}", content)

    # Replace get_X() calls with self._X usage
    if import_name.startswith("get_"):
        attr_name = f"_{import_name[4:]}"  # get_event_logger → _event_logger
        usage_pattern = rf"{import_name}\(\)"
        content = re.sub(usage_pattern, f"self.{attr_name}", content)

    return content


def main():
    """Main refactoring function."""
    domain_path = Path("src/beanllm/domain")

    print("🔧 Refactoring Domain → Infrastructure dependencies...")
    print()

    # Files to refactor (based on grep results)
    files_to_refactor = [
        "embeddings/base.py",
        "embeddings/local/local_embeddings.py",
        "embeddings/utils/cache.py",
        "evaluation/evaluator.py",
        "graph/node_cache.py",
        "loaders/core/directory.py",
        "multi_agent/communication.py",
        "ocr/bean_ocr.py",
        "prompts/cache.py",
        "vision/embeddings.py",
    ]

    for file_rel_path in files_to_refactor:
        file_path = domain_path / file_rel_path
        if not file_path.exists():
            print(f"⚠️  Skipping {file_rel_path} (not found)")
            continue

        print(f"📝 Processing {file_rel_path}...")

        with open(file_path) as f:
            original_content = f.read()

        # Detect which protocols are needed
        protocols_needed = set()
        for import_name, protocol in IMPORT_TO_PROTOCOL_MAP.items():
            if import_name in original_content:
                protocols_needed.add(protocol)

        if not protocols_needed:
            print(f"  ✅ No infrastructure imports found")
            continue

        print(f"  📋 Protocols needed: {', '.join(protocols_needed)}")

        # Note: Full automated refactoring is complex
        # This script provides a starting point
        # Manual review is recommended

        print(f"  ⚠️  Manual refactoring recommended for this file")
        print(f"     Add these protocols to __init__: {', '.join(f'{p.lower().replace('protocol', '')}' for p in protocols_needed)}")
        print()

    print("=" * 60)
    print("Summary:")
    print("  This script identifies files needing refactoring.")
    print("  For complete refactoring, follow the pattern in:")
    print("  - src/beanllm/domain/vector_stores/base.py")
    print("  - src/beanllm/domain/vector_stores/factory.py")
    print("  - src/beanllm/domain/vector_stores/local/chroma.py")
    print()
    print("Pattern:")
    print("  1. Add protocol parameter to __init__")
    print("  2. Store as self._protocol")
    print("  3. Use 'if self._protocol:' for optional features")
    print("  4. Remove 'from beanllm.infrastructure...' imports")


if __name__ == "__main__":
    main()
