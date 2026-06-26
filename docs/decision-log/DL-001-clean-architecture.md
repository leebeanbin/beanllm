# DL-001: Clean Architecture Layer Separation

**Date:** 2026-06  
**Status:** Active

## Decision

Apply Clean Architecture across all layers: Facade → Handler → Service → Domain ← Infrastructure. Dependencies only point inward. No layer may import from an outer layer.

## Why

- **Testability**: Service and Domain layers have zero HTTP dependencies. 80% test coverage without mocking real APIs — only the Provider (Infrastructure) layer needs HTTP mocks.
- **Provider swappability**: Adding a new LLM provider means implementing `BaseLLMProvider` and registering in `ProviderFactory` — zero changes to Service or Domain.
- **Forced separation**: The architecture makes it structurally impossible to call an HTTP client from business logic.

## Trade-offs

- More boilerplate: HandlerFactory, ServiceFactory, ProviderFactory add indirection.
- Steeper onboarding: new contributors must understand which layer their code belongs to before writing it.

## How to apply

When adding new functionality, determine its layer first:
- Pure business rule with no I/O → Domain
- Orchestration of services → Handler
- Use-case logic with interfaces → Service
- External API / DB call → Infrastructure
