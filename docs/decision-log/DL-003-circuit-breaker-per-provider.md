# DL-003: Per-Provider CircuitBreaker Instances

**Date:** 2026-06  
**Status:** Active

## Decision

Each `BaseLLMProvider` subclass holds its own `CircuitBreaker` instance. Thresholds: 5 consecutive failures → OPEN (60s) → HALF_OPEN → 2 successes → CLOSED.

## Why

- **Blast radius isolation**: An OpenAI outage opens the OpenAI circuit but leaves Claude and Gemini fully operational. A single shared breaker would block all providers.
- **Fallback chain**: When `fallback=True`, `ProviderFactory.get_provider()` skips OPEN providers and moves to the next available one automatically.

## Trade-offs

- N provider instances = N independent breaker state machines in memory. Negligible overhead.
- Circuit state is process-local — restarting the process resets all breakers to CLOSED. Intentional: a restart is evidence the issue was resolved.

## How to apply

- Default thresholds (`failure_threshold=5`, `recovery_timeout=60`) are set in `BaseLLMProvider.__init__`. Override per-provider if needed.
- Monitor with `provider.circuit_breaker.state` in health-check endpoints.
- See [playbooks/01-provider-circuit-open.md](../playbooks/01-provider-circuit-open.md) for incident response.
