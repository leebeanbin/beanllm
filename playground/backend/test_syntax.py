#!/usr/bin/env python3
"""
Quick syntax check for main.py
This script checks if the code can be parsed without import errors
"""

import sys
import ast
from pathlib import Path

def check_syntax(file_path):
    """Check if Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Parse AST
        ast.parse(code)
        print(f"✅ Syntax check passed: {file_path}")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path}:")
        print(f"   Line {e.lineno}: {e.text}")
        print(f"   {e.msg}")
        return False
    except Exception as e:
        print(f"❌ Error checking {file_path}: {e}")
        return False

if __name__ == "__main__":
    main_file = Path(__file__).parent / "main.py"
    success = check_syntax(main_file)
    sys.exit(0 if success else 1)
