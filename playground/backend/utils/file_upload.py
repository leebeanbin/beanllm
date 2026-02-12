"""
File Upload Utilities

Centralized file upload handling with:
- Filename sanitization (path traversal prevention)
- Streaming file save (memory efficient, no full-file load)
- File size validation during upload
- Consistent temp directory cleanup via context managers
"""

import logging
import re
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import FrozenSet, Optional

from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

try:
    from beanllm.utils.constants import MAX_FILE_SIZE, MAX_UPLOAD_CHUNK_SIZE
except ImportError:
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    MAX_UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB

# Regex: allow only alphanumeric, hyphens, underscores, dots
_SAFE_FILENAME_RE = re.compile(r"[^\w\-.]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sanitize_filename(name: Optional[str], fallback_ext: str = "") -> str:
    """
    Sanitize an upload filename to prevent path traversal and special chars.

    Args:
        name: Original filename from the client (may be None).
        fallback_ext: Extension to use when name is None (e.g. ".jpg").

    Returns:
        A safe filename string such as ``a3f2c1_report.pdf``.
    """
    if not name:
        unique = uuid.uuid4().hex[:8]
        return f"{unique}{fallback_ext}"

    # Strip directory components (path traversal)
    basename = Path(name).name

    # Replace unsafe chars with underscore
    safe = _SAFE_FILENAME_RE.sub("_", basename)

    # Collapse consecutive underscores and strip leading/trailing
    safe = re.sub(r"_+", "_", safe).strip("_")

    if not safe:
        safe = uuid.uuid4().hex[:8]

    # Prepend short UUID to avoid collisions
    prefix = uuid.uuid4().hex[:8]
    return f"{prefix}_{safe}"


@asynccontextmanager
async def temp_directory():
    """
    Async context manager for a temporary directory.

    Automatically removes the directory and all contents on exit.
    """
    temp_dir = tempfile.mkdtemp(prefix="beanllm_")
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def save_upload_to_temp(
    file: UploadFile,
    temp_dir: str,
    *,
    allowed_extensions: Optional[FrozenSet[str]] = None,
    max_size: int = MAX_FILE_SIZE,
    fallback_ext: str = "",
) -> Path:
    """
    Save an uploaded file to *temp_dir* using chunked streaming.

    - Sanitizes the filename.
    - Validates extension against *allowed_extensions* (if provided).
    - Validates cumulative size while writing (never loads full file into memory).

    Args:
        file: FastAPI UploadFile instance.
        temp_dir: Path to the temporary directory.
        allowed_extensions: Frozenset of allowed lowercase extensions (e.g. {".pdf", ".txt"}).
        max_size: Maximum allowed file size in bytes.
        fallback_ext: Default extension when filename is absent.

    Returns:
        Path to the saved file.

    Raises:
        HTTPException 400: Unsupported file type.
        HTTPException 413: File exceeds *max_size*.
    """
    # Determine extension
    raw_name = file.filename or ""
    ext = Path(raw_name).suffix.lower() if raw_name else fallback_ext

    if allowed_extensions is not None and ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(allowed_extensions))}",
        )

    safe_name = sanitize_filename(raw_name, fallback_ext=ext)
    dest = Path(temp_dir) / safe_name

    # Stream chunks to disk
    total_size = 0
    with open(dest, "wb") as f:
        while True:
            chunk = await file.read(MAX_UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > max_size:
                # Remove partial file and reject
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large (>{max_size // (1024 * 1024)}MB limit)",
                )
            f.write(chunk)

    logger.debug("Saved upload %s (%d bytes) -> %s", raw_name, total_size, dest.name)
    return dest
