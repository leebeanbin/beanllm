# DL-002: Optional Extras for Lightweight Core Install

**Date:** 2026-06  
**Status:** Active

## Decision

Core `pip install beanllm` installs only httpx, pydantic, tiktoken, PyMuPDF (~5MB). Each provider SDK and ML dependency is an optional extra (`beanllm[openai]`, `beanllm[ml]`, etc.).

## Why

- **PyPI library hygiene**: A library that pulls in `torch` or the full OpenAI SDK by default pollutes user environments and causes dependency conflicts.
- **Selective activation**: Users who only need Claude don't pay for the OpenAI SDK binary weight, and vice versa.
- **try/except import guard**: Provider SDKs are imported with `try/except` — missing packages emit a `WARNING` log and disable only that provider, not the whole library.

## Trade-offs

- Users must know which extras to install. Mitigated by `beanllm[all]` shortcut.
- `pyproject.toml` optional extras section grows with each new provider.

## How to apply

New provider → add to `pyproject.toml` extras, guard import with `try/except`, document in README provider table and `docs/api/models.md`.
