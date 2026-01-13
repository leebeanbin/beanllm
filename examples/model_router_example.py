"""
Model Router Example

Intelligent model selection based on request characteristics
"""

from beanllm.infrastructure.routing import (
    ModelRouter,
    RoutingStrategy,
    ModelInfo,
    RequestCharacteristics,
    ComplexityRule,
    CostRule,
    CapabilityRule,
    LatencyRule,
    CompositeRule,
)
from beanllm.infrastructure.routing.model_router import create_default_router


def example_1_basic_routing():
    """Example 1: Basic Routing with Different Strategies"""
    print("=" * 70)
    print("Example 1: Basic Routing")
    print("=" * 70)

    # Create router with balanced strategy
    router = create_default_router(strategy=RoutingStrategy.BALANCED)

    # Simple text request
    request = RequestCharacteristics(
        prompt_length=500,
        complexity_score=0.5,
        context_window_needed=4000,
    )

    decision = router.route(request)
    print(f"\n📋 Request: Simple text task (500 chars)")
    print(f"✅ Selected: {decision.selected_model.provider}:{decision.selected_model.model_id}")
    print(f"💰 Estimated cost: ${decision.estimated_cost:.6f}")
    print(f"🎯 Reason: {decision.reason}")
    print(f"📊 Confidence: {decision.confidence_score:.3f}")


def example_2_cost_optimization():
    """Example 2: Cost-Optimized Routing"""
    print("\n" + "=" * 70)
    print("Example 2: Cost-Optimized Routing")
    print("=" * 70)

    # Cost-optimized router
    router = create_default_router(strategy=RoutingStrategy.COST_OPTIMIZED)

    # Simple tasks
    simple_requests = [
        ("Translation", 200, 0.3),
        ("Summarization", 500, 0.4),
        ("Q&A", 100, 0.3),
    ]

    print("\n💰 Routing simple tasks with cost optimization:")
    for task_name, length, complexity in simple_requests:
        request = RequestCharacteristics(
            prompt_length=length,
            complexity_score=complexity,
        )
        decision = router.route(request)
        print(f"\n  {task_name}:")
        print(f"    Model: {decision.selected_model.model_id}")
        print(f"    Cost: ${decision.estimated_cost:.6f}")
        print(f"    Quality: {decision.selected_model.quality_score:.2f}")


def example_3_quality_optimization():
    """Example 3: Quality-Optimized Routing"""
    print("\n" + "=" * 70)
    print("Example 3: Quality-Optimized Routing")
    print("=" * 70)

    # Quality-optimized router
    router = create_default_router(strategy=RoutingStrategy.QUALITY_OPTIMIZED)

    # Complex tasks
    complex_requests = [
        ("Code Generation", 1000, 0.9),
        ("Mathematical Reasoning", 800, 0.95),
        ("Creative Writing", 600, 0.85),
    ]

    print("\n🎯 Routing complex tasks with quality optimization:")
    for task_name, length, complexity in complex_requests:
        request = RequestCharacteristics(
            prompt_length=length,
            complexity_score=complexity,
        )
        decision = router.route(request)
        print(f"\n  {task_name}:")
        print(f"    Model: {decision.selected_model.model_id}")
        print(f"    Quality: {decision.selected_model.quality_score:.2f}")
        print(f"    Cost: ${decision.estimated_cost:.6f}")


def example_4_capability_matching():
    """Example 4: Capability-Based Routing"""
    print("\n" + "=" * 70)
    print("Example 4: Capability-Based Routing")
    print("=" * 70)

    router = create_default_router(strategy=RoutingStrategy.CAPABILITY_MATCH)

    # Different capability requirements
    capability_tests = [
        ("Text only", False, False, False),
        ("Vision task", True, False, False),
        ("Function calling", False, True, False),
        ("Vision + Functions", True, True, False),
    ]

    print("\n🔧 Routing by capabilities:")
    for task_name, vision, functions, json_mode in capability_tests:
        request = RequestCharacteristics(
            prompt_length=500,
            requires_vision=vision,
            requires_function_calling=functions,
            requires_json_mode=json_mode,
        )
        decision = router.route(request)
        print(f"\n  {task_name}:")
        print(f"    Model: {decision.selected_model.model_id}")
        print(f"    Vision: {decision.selected_model.supports_vision}")
        print(f"    Functions: {decision.selected_model.supports_function_calling}")
        print(f"    Cost: ${decision.estimated_cost:.6f}")


def example_5_complexity_based():
    """Example 5: Complexity-Based Routing"""
    print("\n" + "=" * 70)
    print("Example 5: Complexity-Based Adaptive Routing")
    print("=" * 70)

    router = create_default_router(strategy=RoutingStrategy.COMPLEXITY_BASED)

    # Varying complexity
    complexity_levels = [
        ("Trivial (0.1)", 0.1),
        ("Simple (0.3)", 0.3),
        ("Moderate (0.5)", 0.5),
        ("Complex (0.7)", 0.7),
        ("Very Complex (0.9)", 0.9),
    ]

    print("\n📊 Routing by task complexity:")
    for task_name, complexity in complexity_levels:
        request = RequestCharacteristics(
            prompt_length=500,
            complexity_score=complexity,
        )
        decision = router.route(request)
        print(f"\n  {task_name}:")
        print(f"    Model: {decision.selected_model.model_id}")
        print(f"    Model Quality: {decision.selected_model.quality_score:.2f}")
        print(f"    Cost: ${decision.estimated_cost:.6f}")


def example_6_custom_rules():
    """Example 6: Custom Routing Rules"""
    print("\n" + "=" * 70)
    print("Example 6: Custom Routing Rules")
    print("=" * 70)

    # Create router without default strategy
    router = ModelRouter(strategy=RoutingStrategy.BALANCED)

    # Register a few models manually
    router.register_model(ModelInfo(
        provider="openai",
        model_id="gpt-4",
        context_window=8000,
        cost_per_1k_input=0.03,
        cost_per_1k_output=0.06,
        quality_score=0.95,
        supports_function_calling=True,
        latency_score=0.5,
        reliability_score=0.99,
    ))
    router.register_model(ModelInfo(
        provider="openai",
        model_id="gpt-3.5-turbo",
        context_window=4000,
        cost_per_1k_input=0.0015,
        cost_per_1k_output=0.002,
        quality_score=0.7,
        supports_function_calling=True,
        latency_score=0.2,
        reliability_score=0.95,
    ))

    # Create composite rule
    composite_rule = CompositeRule([
        (ComplexityRule(), 0.3),
        (CostRule(), 0.3),
        (LatencyRule(), 0.2),
        (CapabilityRule(), 0.2),
    ])

    print("\n🎨 Using custom composite rule (complexity + cost + latency + capability):")

    request = RequestCharacteristics(
        prompt_length=800,
        complexity_score=0.6,
        requires_function_calling=True,
    )

    # Score models using custom rule
    for model in router.models:
        score = composite_rule.evaluate(model, request)
        print(f"\n  {model.model_id}:")
        print(f"    Score: {score:.3f}")
        print(f"    Quality: {model.quality_score}")
        print(f"    Cost/1k: ${model.cost_per_1k_input}")
        print(f"    Latency: {model.latency_score}")


def example_7_fallback():
    """Example 7: Fallback Strategy"""
    print("\n" + "=" * 70)
    print("Example 7: Fallback Strategy")
    print("=" * 70)

    router = create_default_router(strategy=RoutingStrategy.BALANCED)

    request = RequestCharacteristics(
        prompt_length=1000,
        complexity_score=0.7,
        requires_vision=True,
    )

    decision = router.route(request)

    print(f"\n🔄 Primary model: {decision.selected_model.model_id}")
    print(f"   Cost: ${decision.estimated_cost:.6f}")

    print(f"\n📋 Fallback models (if primary fails):")
    for i, fallback in enumerate(decision.fallback_models, 1):
        print(f"   {i}. {fallback.model_id} (quality: {fallback.quality_score:.2f})")


def example_8_cost_constraints():
    """Example 8: Cost-Constrained Routing"""
    print("\n" + "=" * 70)
    print("Example 8: Cost-Constrained Routing")
    print("=" * 70)

    router = create_default_router(strategy=RoutingStrategy.BALANCED)

    # Try with different cost constraints
    cost_limits = [0.001, 0.005, 0.02, None]  # None = no limit

    print("\n💵 Routing with different cost constraints:")
    for max_cost in cost_limits:
        request = RequestCharacteristics(
            prompt_length=1000,
            complexity_score=0.6,
            max_cost_per_1k=max_cost,
        )

        try:
            decision = router.route(request)
            cost_str = f"${max_cost:.4f}" if max_cost else "unlimited"
            print(f"\n  Max cost/1k: {cost_str}")
            print(f"    Selected: {decision.selected_model.model_id}")
            print(f"    Actual cost/1k: ${decision.selected_model.cost_per_1k_input:.4f}")
            print(f"    Quality: {decision.selected_model.quality_score:.2f}")
        except ValueError as e:
            print(f"\n  Max cost/1k: ${max_cost:.4f}")
            print(f"    ❌ Error: {e}")


def example_9_statistics():
    """Example 9: Router Statistics"""
    print("\n" + "=" * 70)
    print("Example 9: Router Statistics and Adaptive Routing")
    print("=" * 70)

    router = create_default_router(strategy=RoutingStrategy.BALANCED)

    # Simulate some successful and failed requests
    print("\n📊 Simulating requests to build statistics...")

    for i in range(10):
        request = RequestCharacteristics(
            prompt_length=500,
            complexity_score=0.5,
        )
        decision = router.route(request)

        # Simulate success/failure (90% success rate for demo)
        import random
        success = random.random() < 0.9
        latency = random.uniform(0.5, 2.0)

        router.record_result(
            model=decision.selected_model,
            success=success,
            latency=latency,
        )

    # Print statistics
    stats = router.get_stats()
    print(f"\n📈 Router Statistics:")
    print(f"   Strategy: {stats['strategy']}")
    print(f"   Registered models: {stats['registered_models']}")
    print(f"   Fallback enabled: {stats['enable_fallback']}")

    print(f"\n📊 Model Performance:")
    for model_key, model_stats in stats['model_stats'].items():
        total = model_stats['total_requests']
        if total > 0:
            success_rate = model_stats['successful_requests'] / total * 100
            print(f"\n   {model_key}:")
            print(f"      Total requests: {total}")
            print(f"      Success rate: {success_rate:.1f}%")
            print(f"      Avg latency: {model_stats['avg_latency']:.2f}s")


def main():
    """Run all examples"""
    examples = [
        example_1_basic_routing,
        example_2_cost_optimization,
        example_3_quality_optimization,
        example_4_capability_matching,
        example_5_complexity_based,
        example_6_custom_rules,
        example_7_fallback,
        example_8_cost_constraints,
        example_9_statistics,
    ]

    for i, example in enumerate(examples, 1):
        example()
        if i < len(examples):
            print("\n" + "-" * 70 + "\n")

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
