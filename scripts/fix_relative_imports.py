#!/usr/bin/env python3
"""
Fix relative imports to absolute imports in beanllm codebase.
Converts patterns like 'from ...dto.request' to 'from beanllm.dto.request'
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


def fix_relative_imports(file_path: Path) -> Tuple[bool, int]:
    """
    Fix relative imports in a single file.

    Args:
        file_path: Path to Python file

    Returns:
        Tuple of (changed, count) where changed is True if file was modified
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return False, 0

    original_content = content
    changes = 0

    # Pattern 1: from ...module import X
    # from ...dto.request.core.chat_request import ChatRequest
    # → from beanllm.dto.request.core.chat_request import ChatRequest
    pattern1 = r"from \.\.\.([a-zA-Z_][a-zA-Z0-9_.]*)"
    replacement1 = r"from beanllm.\1"
    content, count1 = re.subn(pattern1, replacement1, content)
    changes += count1

    # Pattern 2: from ..module import X (two dots)
    # Need to infer the correct path based on file location
    parts = file_path.parts
    if "beanllm" in parts:
        beanllm_index = parts.index("beanllm")

        # Get all parent directories from beanllm root
        # e.g., for beanllm/domain/loaders/core/pdf_loader.py
        # parent_dirs = ['domain', 'loaders', 'core']
        parent_dirs = list(parts[beanllm_index + 1 : -1])

        # Replace .. imports (going up one level)
        # For a file at beanllm/domain/loaders/core/pdf_loader.py:
        # from ..base → from beanllm.domain.loaders.base
        # from ..types → from beanllm.domain.loaders.types
        if len(parent_dirs) >= 1:
            # Go up one level from current directory
            parent_path = parent_dirs[:-1] if len(parent_dirs) > 1 else []
            base_module = "beanllm." + ".".join(parent_path) if parent_path else "beanllm"

            # Replace from ..module with from beanllm.parent.module
            pattern2 = r"from \.\.([a-zA-Z_][a-zA-Z0-9_.]*)"
            matches = list(re.finditer(pattern2, content))
            for match in matches:
                module_name = match.group(1)
                old_import = match.group(0)
                new_import = f"from {base_module}.{module_name}"
                content = content.replace(old_import, new_import, 1)
                changes += 1

        # Pattern 3: from ....module (four dots - go up 3 levels)
        # Used in service/impl/ files
        if len(parent_dirs) >= 3:
            parent_path = parent_dirs[:-3] if len(parent_dirs) > 3 else []
            base_module = "beanllm." + ".".join(parent_path) if parent_path else "beanllm"

            pattern4 = r"from \.\.\.\.([a-zA-Z_][a-zA-Z0-9_.]*)"
            matches = list(re.finditer(pattern4, content))
            for match in matches:
                module_name = match.group(1)
                old_import = match.group(0)
                new_import = f"from {base_module}.{module_name}"
                content = content.replace(old_import, new_import, 1)
                changes += 1

    # Save if changed
    if content != original_content:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True, changes
        except Exception as e:
            print(f"Error writing {file_path}: {e}", file=sys.stderr)
            return False, 0

    return False, 0


def main():
    """Main function to fix all Python files."""
    src_dir = Path(__file__).parent.parent / "src" / "beanllm"

    if not src_dir.exists():
        print(f"Error: {src_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    print("🔧 Fixing relative imports in beanllm codebase...")
    print(f"📂 Scanning: {src_dir}")
    print()

    python_files = list(src_dir.rglob("*.py"))
    total_files = len(python_files)
    modified_files = 0
    total_changes = 0

    for i, py_file in enumerate(python_files, 1):
        relative_path = py_file.relative_to(src_dir.parent.parent)
        changed, count = fix_relative_imports(py_file)

        if changed:
            modified_files += 1
            total_changes += count
            print(f"✅ [{i}/{total_files}] {relative_path} ({count} imports fixed)")
        else:
            if (i % 50) == 0:  # Progress indicator
                print(f"⏳ [{i}/{total_files}] Processing...")

    print()
    print("=" * 60)
    print(f"✨ Fixed {total_changes} relative imports in {modified_files}/{total_files} files")
    print("=" * 60)


if __name__ == "__main__":
    main()
