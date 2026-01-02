"""
Dependency Manager - Centralized dependency checking

Replaces 261 duplicate try/except ImportError patterns across the codebase.
"""

from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

F = TypeVar('F', bound=Callable[..., Any])


class DependencyManager:
    """
    Centralized dependency management with decorators

    Eliminates duplicate ImportError handling patterns:
    - Before: 261 occurrences of try/except ImportError
    - After: 1 centralized implementation

    Example:
        >>> class HuggingFaceEmbedding:
        ...     @DependencyManager.require("transformers", "torch")
        ...     def _load_model(self):
        ...         from transformers import AutoModel
        ...         # ... model loading logic
    """

    # Installation messages for common packages
    _INSTALL_MSGS: Dict[str, str] = {
        # Deep Learning Frameworks
        "transformers": "pip install transformers",
        "torch": "pip install torch",
        "torchvision": "pip install torchvision",
        "tensorflow": "pip install tensorflow",

        # Embeddings
        "sentence-transformers": "pip install sentence-transformers",
        "openai": "pip install openai",

        # Vector Stores
        "chromadb": "pip install chromadb",
        "faiss": "pip install faiss-cpu  # or faiss-gpu",
        "pinecone": "pip install pinecone-client",
        "qdrant-client": "pip install qdrant-client",
        "weaviate-client": "pip install weaviate-client",
        "pymilvus": "pip install pymilvus",
        "lancedb": "pip install lancedb",
        "psycopg2": "pip install psycopg2-binary",
        "pgvector": "pip install pgvector",

        # PDF Processing
        "marker": "pip install marker-pdf",
        "pdfplumber": "pip install pdfplumber",
        "pymupdf": "pip install PyMuPDF",
        "fitz": "pip install PyMuPDF",
        "pypdf": "pip install pypdf",
        "docling": "pip install docling",

        # Vision
        "cv2": "pip install opencv-python",
        "PIL": "pip install Pillow",
        "sam3": "pip install segment-anything-3",
        "ultralytics": "pip install ultralytics",

        # Audio
        "whisper": "pip install openai-whisper",
        "librosa": "pip install librosa",
        "soundfile": "pip install soundfile",

        # Web
        "playwright": "pip install playwright",
        "selenium": "pip install selenium",

        # LLM Providers
        "anthropic": "pip install anthropic",
        "google.generativeai": "pip install google-generativeai",
        "ollama": "pip install ollama",

        # Utilities
        "pandas": "pip install pandas",
        "openpyxl": "pip install openpyxl",
        "python-pptx": "pip install python-pptx",
        "python-docx": "pip install python-docx",
    }

    @staticmethod
    def require(*packages: str) -> Callable[[F], F]:
        """
        Decorator to check required packages before function execution

        Args:
            *packages: Package names to check

        Returns:
            Decorated function that checks dependencies first

        Raises:
            ImportError: If any required package is not installed

        Example:
            >>> @DependencyManager.require("transformers", "torch")
            ... def load_model():
            ...     from transformers import AutoModel
            ...     return AutoModel.from_pretrained("bert-base-uncased")
        """
        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Check all packages before execution
                for pkg in packages:
                    try:
                        __import__(pkg)
                    except ImportError as e:
                        install_cmd = DependencyManager._INSTALL_MSGS.get(
                            pkg,
                            f"pip install {pkg}"
                        )
                        raise ImportError(
                            f"{pkg} is required but not installed. "
                            f"Install with: {install_cmd}"
                        ) from e

                # All dependencies satisfied, execute function
                return func(*args, **kwargs)

            return wrapper  # type: ignore

        return decorator

    @staticmethod
    def check_available(*packages: str) -> bool:
        """
        Check if packages are available without raising error

        Args:
            *packages: Package names to check

        Returns:
            True if all packages are available, False otherwise

        Example:
            >>> if DependencyManager.check_available("torch", "transformers"):
            ...     print("Using HuggingFace embeddings")
            ... else:
            ...     print("Using OpenAI embeddings")
        """
        for pkg in packages:
            try:
                __import__(pkg)
            except ImportError:
                return False
        return True

    @staticmethod
    def get_install_command(package: str) -> str:
        """
        Get installation command for a package

        Args:
            package: Package name

        Returns:
            pip install command string

        Example:
            >>> cmd = DependencyManager.get_install_command("transformers")
            >>> print(cmd)
            pip install transformers
        """
        return DependencyManager._INSTALL_MSGS.get(
            package,
            f"pip install {package}"
        )

    @staticmethod
    def require_any(*package_groups: tuple) -> Callable[[F], F]:
        """
        Decorator requiring at least one package from each group

        Args:
            *package_groups: Tuples of alternative packages

        Returns:
            Decorated function

        Example:
            >>> @DependencyManager.require_any(
            ...     ("torch", "tensorflow"),  # Need either torch or tensorflow
            ...     ("transformers",)          # Need transformers
            ... )
            ... def load_model():
            ...     pass
        """
        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                for group in package_groups:
                    if not any(DependencyManager.check_available(pkg) for pkg in group):
                        pkg_list = " or ".join(group)
                        install_cmds = " or ".join(
                            DependencyManager.get_install_command(pkg)
                            for pkg in group
                        )
                        raise ImportError(
                            f"At least one of {pkg_list} is required. "
                            f"Install with: {install_cmds}"
                        )

                return func(*args, **kwargs)

            return wrapper  # type: ignore

        return decorator


# Convenience aliases
require = DependencyManager.require
check_available = DependencyManager.check_available
require_any = DependencyManager.require_any


__all__ = [
    "DependencyManager",
    "require",
    "check_available",
    "require_any",
]
