"""
Document Loaders - Re-exports

All loader implementations have been moved to separate files:
- text.py - TextLoader (mmap optimized)
- pdf_loader.py - PDFLoader
- csv.py - CSVLoader (with helper methods)
- directory.py - DirectoryLoader (recursive scan)
- html.py - HTMLLoader (BeautifulSoup)
- jupyter.py - JupyterLoader
- docling_loader.py - DoclingLoader (advanced document processing)

This file re-exports all implementations for backward compatibility.
"""

# Re-export all loaders
from .advanced import DoclingLoader
from .core import (
    CSVLoader,
    DirectoryLoader,
    HTMLLoader,
    JupyterLoader,
    PDFLoader,
    TextLoader,
)

__all__ = [
    "TextLoader",
    "PDFLoader",
    "CSVLoader",
    "DirectoryLoader",
    "HTMLLoader",
    "JupyterLoader",
    "DoclingLoader",
]
