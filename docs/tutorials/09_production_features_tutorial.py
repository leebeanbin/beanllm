"""
Production-Ready Features Tutorial
프로덕션 기능 튜토리얼

이 튜토리얼은 llmkit의 프로덕션 기능을 실습합니다:
- Token Counting & Cost Management
- Prompt Templates
- Evaluation Metrics
- Fine-tuning
- Error Handling
"""

import os
import time
from typing import List, Dict

# llmkit imports
import llmkit as lk

# ============================================================================
# Part 1: Token Counting & Cost Management
# ============================================================================

def part1_token_counting():
    """Part 1: Token Counting"""
    print("\n" + "="*60)
    print("Part 1: Token Counting & Cost Management")
    print("="*60)

    # 1.1 Basic Token Counting
    print("\n### 1.1 Basic Token Counting ###")

    text = "Hello, world! How are you doing today?"

    # Count tokens for different models
    models = ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-20241022"]

    for model in models:
        try:
            token_count = lk.count_tokens(text, model=model)
            print(f"{model}: {token_count} tokens")
        except Exception as e:
            print(f"{model}: Error - {e}")

    # 1.2 Message Token Counting
    print("\n### 1.2 Message Token Counting ###")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is..."},
        {"role": "user", "content": "Can you give me an example?"}
    ]

    # Count tokens including message overhead
    total_tokens = lk.count_message_tokens(messages, model="gpt-4o")
    print(f"Total message tokens: {total_tokens}")

    # 1.3 Cost Estimation
    print("\n### 1.3 Cost Estimation ###")

    input_text = "Explain quantum computing in simple terms."
    output_text = """Quantum computing uses quantum bits (qubits) that can exist
    in superposition of states, unlike classical bits that are either 0 or 1.
    This allows quantum computers to process vast amounts of possibilities
    simultaneously, making them potentially much more powerful for certain tasks."""

    # Estimate cost for different models
    for model in ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-20241022"]:
        try:
            estimate = lk.estimate_cost(
                input_text=input_text,
                output_text=output_text,
                model=model
            )

            print(f"\n{model}:")
            print(f"  Input tokens: {estimate.input_tokens}")
            print(f"  Output tokens: {estimate.output_tokens}")
            print(f"  Input cost: ${estimate.input_cost:.6f}")
            print(f"  Output cost: ${estimate.output_cost:.6f}")
            print(f"  Total cost: ${estimate.total_cost:.6f}")
        except Exception as e:
            print(f"\n{model}: Error - {e}")

    # 1.4 Find Cheapest Model
    print("\n### 1.4 Find Cheapest Model ###")

    models_to_compare = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022"
    ]

    cheapest = lk.get_cheapest_model(
        input_text=input_text,
        output_tokens=100,
        models=models_to_compare
    )

    print(f"Cheapest model: {cheapest}")

    # 1.5 Context Window Management
    print("\n### 1.5 Context Window Management ###")

    model = "gpt-4o"
    context_window = lk.get_context_window(model)
    print(f"{model} context window: {context_window:,} tokens")

    # Calculate available tokens
    counter = lk.TokenCounter(model)
    input_tokens = counter.count_tokens_from_messages(messages)
    reserved_tokens = 1000  # Reserve for output

    available = counter.get_available_tokens(messages, reserved=reserved_tokens)
    print(f"Input tokens: {input_tokens}")
    print(f"Available for output: {available:,}")


# ============================================================================
# Part 2: Prompt Templates
# ============================================================================

def part2_prompt_templates():
    """Part 2: Prompt Templates"""
    print("\n" + "="*60)
    print("Part 2: Prompt Templates")
    print("="*60)

    # 2.1 Basic Template
    print("\n### 2.1 Basic Template ###")

    template = lk.PromptTemplate(
        template="Translate the following text from {source_lang} to {target_lang}:\n\n{text}",
        input_variables=["source_lang", "target_lang", "text"]
    )

    prompt = template.format(
        source_lang="English",
        target_lang="Korean",
        text="Hello, how are you?"
    )

    print(prompt)

    # 2.2 Few-Shot Template
    print("\n### 2.2 Few-Shot Template ###")

    examples = [
        lk.PromptExample(input="2+2", output="4"),
        lk.PromptExample(input="5*3", output="15"),
        lk.PromptExample(input="10-7", output="3")
    ]

    few_shot = lk.FewShotPromptTemplate(
        examples=examples,
        example_template=lk.PromptTemplate(
            template="Q: {input}\nA: {output}",
            input_variables=["input", "output"]
        ),
        prefix="Solve the math problem:",
        suffix="Q: {input}\nA:",
        input_variables=["input"]
    )

    prompt = few_shot.format(input="8*4")
    print(prompt)

    # 2.3 Chat Template
    print("\n### 2.3 Chat Template ###")

    chat_template = lk.ChatPromptTemplate.from_messages([
        ("system", "You are a {role} that helps with {task}."),
        ("user", "{user_input}")
    ])

    messages = chat_template.format_messages(
        role="Python expert",
        task="code debugging",
        user_input="How do I fix a NameError?"
    )

    for msg in messages:
        print(f"{msg.role}: {msg.content}")

    # 2.4 Predefined Templates
    print("\n### 2.4 Predefined Templates ###")

    # Translation template
    translation = lk.PredefinedTemplates.translation()
    prompt = translation.format(
        source_lang="English",
        target_lang="Spanish",
        text="Good morning!"
    )
    print(f"\nTranslation:\n{prompt}")

    # Chain-of-Thought template
    cot = lk.PredefinedTemplates.chain_of_thought()
    prompt = cot.format(question="If John has 5 apples and gives away 2, how many does he have left?")
    print(f"\nChain-of-Thought:\n{prompt[:200]}...")

    # 2.5 Prompt Optimization
    print("\n### 2.5 Prompt Optimization ###")

    base_prompt = "Write a Python function to check if a number is prime."

    # Add JSON output format
    optimized = lk.PromptOptimizer.add_json_output(
        prompt=base_prompt,
        schema={
            "function_name": "str",
            "code": "str",
            "explanation": "str"
        }
    )

    print(optimized[:200] + "...")

    # 2.6 Prompt Caching
    print("\n### 2.6 Prompt Caching ###")

    template = lk.PromptTemplate(
        template="Summarize: {text}",
        input_variables=["text"]
    )

    # First call (cache miss)
    result1 = lk.get_cached_prompt(template, text="Sample text", use_cache=True)

    # Second call (cache hit)
    result2 = lk.get_cached_prompt(template, text="Sample text", use_cache=True)

    stats = lk.get_cache_stats()
    print(f"Cache stats: {stats}")

    lk.clear_cache()


# ============================================================================
# Part 3: Evaluation Metrics
# ============================================================================

def part3_evaluation_metrics():
    """Part 3: Evaluation Metrics"""
    print("\n" + "="*60)
    print("Part 3: Evaluation Metrics")
    print("="*60)

    # 3.1 BLEU Score
    print("\n### 3.1 BLEU Score ###")

    reference = "The cat is sitting on the mat"
    prediction = "The cat sits on the mat"

    bleu = lk.BLEUMetric()
    result = bleu.compute(prediction, reference)

    print(f"BLEU Score: {result.score:.4f}")
    print(f"Metadata: {result.metadata}")

    # 3.2 ROUGE Score
    print("\n### 3.2 ROUGE Score ###")

    reference = """Machine learning is a subset of artificial intelligence that
    enables systems to learn and improve from experience."""

    prediction = """Machine learning allows systems to learn from data and
    improve their performance over time."""

    rouge1 = lk.ROUGEMetric("rouge-1")
    rouge2 = lk.ROUGEMetric("rouge-2")
    rougel = lk.ROUGEMetric("rouge-l")

    for metric in [rouge1, rouge2, rougel]:
        result = metric.compute(prediction, reference)
        print(f"{metric.name}: {result.score:.4f} (P={result.metadata['precision']:.4f}, R={result.metadata['recall']:.4f})")

    # 3.3 F1 Score
    print("\n### 3.3 F1 Score ###")

    f1 = lk.F1ScoreMetric()
    result = f1.compute(prediction, reference)

    print(f"F1 Score: {result.score:.4f}")
    print(f"Precision: {result.metadata['precision']:.4f}")
    print(f"Recall: {result.metadata['recall']:.4f}")

    # 3.4 Exact Match
    print("\n### 3.4 Exact Match ###")

    exact_match = lk.ExactMatchMetric()

    cases = [
        ("hello world", "hello world", True),
        ("hello world", "Hello World", False),
        ("hello  world", "hello world", True),  # Whitespace normalized
    ]

    for pred, ref, expected in cases:
        result = exact_match.compute(pred, ref)
        print(f"'{pred}' == '{ref}': {result.score} (expected: {expected})")

    # 3.5 Combined Evaluation
    print("\n### 3.5 Combined Evaluation ###")

    # Create evaluator with multiple metrics
    evaluator = lk.create_evaluator(["bleu", "rouge-1", "rouge-l", "f1"])

    result = evaluator.evaluate(prediction, reference)

    print(f"\nCombined Evaluation Results:")
    print(f"Average Score: {result.average_score:.4f}")
    for metric_result in result.results:
        print(f"  {metric_result.metric_name}: {metric_result.score:.4f}")

    # 3.6 Batch Evaluation
    print("\n### 3.6 Batch Evaluation ###")

    predictions = [
        "The cat is on the mat",
        "Machine learning is awesome",
        "Python is a programming language"
    ]

    references = [
        "The cat sits on the mat",
        "Machine learning is great",
        "Python is a popular programming language"
    ]

    batch_results = evaluator.batch_evaluate(predictions, references)

    print(f"\nBatch Evaluation ({len(predictions)} examples):")
    for i, result in enumerate(batch_results):
        print(f"\nExample {i+1}: Avg = {result.average_score:.4f}")

    # 3.7 Convenience Function
    print("\n### 3.7 Convenience Function ###")

    result = lk.evaluate_text(
        prediction="Hello world",
        reference="Hello world!",
        metrics=["exact_match", "f1", "bleu"]
    )

    print(f"Quick evaluation: {result.average_score:.4f}")


# ============================================================================
# Part 4: Fine-tuning (Preparation Only)
# ============================================================================

def part4_finetuning_prep():
    """Part 4: Fine-tuning Preparation"""
    print("\n" + "="*60)
    print("Part 4: Fine-tuning (Data Preparation)")
    print("="*60)

    # 4.1 Create Training Examples
    print("\n### 4.1 Create Training Examples ###")

    # From Q&A pairs
    qa_pairs = [
        {"question": "What is Python?", "answer": "Python is a high-level programming language."},
        {"question": "What is a list?", "answer": "A list is a mutable sequence data type."},
        {"question": "What is a function?", "answer": "A function is a reusable block of code."}
    ]

    examples = lk.DatasetBuilder.from_qa_pairs(
        qa_pairs,
        system_message="You are a Python programming expert."
    )

    print(f"Created {len(examples)} examples")
    print(f"First example: {examples[0].messages}")

    # 4.2 Validate Dataset
    print("\n### 4.2 Validate Dataset ###")

    report = lk.DataValidator.validate_dataset(examples)

    print(f"Validation Report:")
    print(f"  Valid: {report['is_valid']}")
    print(f"  Total: {report['total_examples']}")
    print(f"  Invalid: {report['invalid_count']}")

    if report['errors']:
        print(f"  Errors: {report['errors']}")

    # 4.3 Estimate Tokens
    print("\n### 4.3 Estimate Tokens ###")

    token_estimate = lk.DataValidator.estimate_tokens(examples)

    print(f"Token Estimate:")
    print(f"  Total tokens: {token_estimate['total_tokens']}")
    print(f"  Avg per example: {token_estimate['average_per_example']:.1f}")

    # 4.4 Split Dataset
    print("\n### 4.4 Split Dataset ###")

    train, val = lk.DatasetBuilder.split_dataset(
        examples,
        train_ratio=0.8,
        shuffle=True
    )

    print(f"Train set: {len(train)} examples")
    print(f"Validation set: {len(val)} examples")

    # 4.5 Save to JSONL
    print("\n### 4.5 Save to JSONL ###")

    # Note: In real use, would use provider.prepare_data()
    print("Tip: Use FineTuningManager to prepare and upload data")
    print("Example:")
    print("  provider = lk.create_finetuning_provider('openai')")
    print("  manager = lk.FineTuningManager(provider)")
    print("  file_id = manager.prepare_and_upload(train, 'train.jsonl')")

    # 4.6 Cost Estimation
    print("\n### 4.6 Fine-tuning Cost Estimation ###")

    n_tokens = token_estimate['total_tokens']
    n_epochs = 3

    cost_info = lk.FineTuningCostEstimator.estimate_training_cost(
        model="gpt-3.5-turbo",
        n_tokens=n_tokens,
        n_epochs=n_epochs
    )

    print(f"Fine-tuning Cost Estimate:")
    print(f"  Model: {cost_info['model']}")
    print(f"  Total tokens (with {n_epochs} epochs): {cost_info['total_tokens']:,}")
    print(f"  Price per 1M tokens: ${cost_info['price_per_1m']:.2f}")
    print(f"  Estimated cost: ${cost_info['estimated_cost_usd']:.4f}")


# ============================================================================
# Part 5: Error Handling
# ============================================================================

def part5_error_handling():
    """Part 5: Error Handling"""
    print("\n" + "="*60)
    print("Part 5: Error Handling")
    print("="*60)

    # 5.1 Retry with Exponential Backoff
    print("\n### 5.1 Retry with Exponential Backoff ###")

    attempt_count = [0]

    @lk.retry(max_retries=3, strategy=lk.RetryStrategy.EXPONENTIAL)
    def unreliable_function():
        attempt_count[0] += 1
        print(f"  Attempt {attempt_count[0]}")
        if attempt_count[0] < 3:
            raise ConnectionError("Simulated failure")
        return "Success!"

    try:
        result = unreliable_function()
        print(f"Result: {result}")
    except lk.MaxRetriesExceededError as e:
        print(f"Failed after retries: {e}")

    # 5.2 Circuit Breaker
    print("\n### 5.2 Circuit Breaker ###")

    breaker = lk.CircuitBreaker(
        lk.CircuitBreakerConfig(
            failure_threshold=3,
            timeout=5.0
        )
    )

    def flaky_service():
        import random
        if random.random() < 0.7:  # 70% failure rate
            raise Exception("Service error")
        return "OK"

    # Simulate calls
    for i in range(10):
        try:
            result = breaker.call(flaky_service)
            print(f"Call {i+1}: {result}")
        except lk.CircuitBreakerError as e:
            print(f"Call {i+1}: Circuit OPEN - {e}")
        except Exception as e:
            print(f"Call {i+1}: Failed - {e}")

    # Check state
    state = breaker.get_state()
    print(f"\nCircuit state: {state['state']}")
    print(f"Failure count: {state['failure_count']}")
    print(f"Success rate: {state['success_rate']:.2%}")

    # 5.3 Rate Limiting
    print("\n### 5.3 Rate Limiting ###")

    limiter = lk.RateLimiter(
        lk.RateLimitConfig(
            max_calls=3,
            time_window=5.0  # 3 calls per 5 seconds
        )
    )

    def api_call():
        return "API response"

    # Make calls
    for i in range(5):
        try:
            result = limiter.call(api_call)
            print(f"Call {i+1}: {result}")
        except lk.RateLimitError as e:
            print(f"Call {i+1}: Rate limited - {e}")

    # Check status
    status = limiter.get_status()
    print(f"\nRate limiter status:")
    print(f"  Current calls: {status['current_calls']}/{status['max_calls']}")
    print(f"  Calls remaining: {status['calls_remaining']}")

    # 5.4 Combined Error Handling
    print("\n### 5.4 Combined Error Handling ###")

    @lk.with_error_handling(
        max_retries=3,
        failure_threshold=5,
        max_calls=10,
        time_window=60
    )
    def production_api_call():
        import random
        if random.random() < 0.3:
            raise Exception("Random failure")
        return "Production data"

    # Make calls with full error handling
    for i in range(5):
        try:
            result = production_api_call()
            print(f"Call {i+1}: {result}")
        except Exception as e:
            print(f"Call {i+1}: Error - {e}")

    # 5.5 Error Tracking
    print("\n### 5.5 Error Tracking ###")

    tracker = lk.get_error_tracker()

    # Simulate errors
    for i in range(3):
        try:
            if i % 2 == 0:
                raise ValueError("Validation error")
            else:
                raise ConnectionError("Network error")
        except Exception as e:
            tracker.record(e, metadata={"request_id": f"req_{i}"})

    # Get summary
    summary = tracker.get_error_summary()

    print(f"\nError Summary:")
    print(f"  Total errors: {summary['total_errors']}")
    print(f"  Error types: {summary['error_types']}")
    print(f"  Most common: {summary.get('most_common_error', 'N/A')}")

    # Get recent errors
    recent = tracker.get_recent_errors(n=3)
    print(f"\nRecent Errors:")
    for error in recent:
        print(f"  [{error.timestamp}] {error.error_type}: {error.error_message}")

    # Clear tracker
    tracker.clear()


# ============================================================================
# Part 6: Production Best Practices
# ============================================================================

def part6_best_practices():
    """Part 6: Production Best Practices"""
    print("\n" + "="*60)
    print("Part 6: Production Best Practices")
    print("="*60)

    # 6.1 Cost-Optimized RAG
    print("\n### 6.1 Cost-Optimized RAG ###")

    # Strategy: Use cheaper model for simple queries, expensive for complex
    def smart_model_selection(query: str) -> str:
        """Select model based on query complexity"""
        # Simple heuristic: length and keywords
        complex_keywords = ["complex", "detailed", "comprehensive", "analyze", "compare"]

        is_complex = (
            len(query.split()) > 20 or
            any(kw in query.lower() for kw in complex_keywords)
        )

        if is_complex:
            return "gpt-4o"  # Expensive but powerful
        else:
            return "gpt-4o-mini"  # Cheap and fast

    queries = [
        "What is Python?",
        "Provide a comprehensive analysis comparing Python and Java for enterprise applications, considering performance, ecosystem, and maintainability."
    ]

    for query in queries:
        model = smart_model_selection(query)
        tokens = lk.count_tokens(query, model)
        print(f"\nQuery: {query[:50]}...")
        print(f"  Selected model: {model}")
        print(f"  Tokens: {tokens}")

    # 6.2 Monitoring Setup
    print("\n### 6.2 Monitoring Setup ###")

    class ProductionMetrics:
        """Simple metrics collector"""
        def __init__(self):
            self.requests = []

        def record_request(self, model, input_tokens, output_tokens, latency_ms, cost):
            self.requests.append({
                "timestamp": time.time(),
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "cost": cost
            })

        def get_summary(self):
            if not self.requests:
                return {"error": "No requests recorded"}

            total_cost = sum(r["cost"] for r in self.requests)
            avg_latency = sum(r["latency_ms"] for r in self.requests) / len(self.requests)

            return {
                "total_requests": len(self.requests),
                "total_cost": total_cost,
                "avg_latency_ms": avg_latency,
                "total_tokens": sum(r["input_tokens"] + r["output_tokens"] for r in self.requests)
            }

    metrics = ProductionMetrics()

    # Simulate requests
    for i in range(3):
        model = "gpt-4o-mini"
        input_text = f"Query {i}"
        output_text = f"Response {i}"

        cost_estimate = lk.estimate_cost(input_text, output_text, model)

        metrics.record_request(
            model=model,
            input_tokens=cost_estimate.input_tokens,
            output_tokens=cost_estimate.output_tokens,
            latency_ms=100 + i*10,
            cost=cost_estimate.total_cost
        )

    summary = metrics.get_summary()
    print(f"\nMetrics Summary:")
    print(f"  Total requests: {summary['total_requests']}")
    print(f"  Total cost: ${summary['total_cost']:.6f}")
    print(f"  Avg latency: {summary['avg_latency_ms']:.2f}ms")
    print(f"  Total tokens: {summary['total_tokens']}")

    # 6.3 Quality Assurance
    print("\n### 6.3 Quality Assurance ###")

    print("Best Practices:")
    print("  1. Always validate input/output with schemas")
    print("  2. Use evaluation metrics in CI/CD")
    print("  3. A/B test prompt changes")
    print("  4. Monitor error rates and latency")
    print("  5. Set up cost alerts")
    print("  6. Use circuit breakers for external APIs")
    print("  7. Implement graceful degradation")
    print("  8. Cache frequent queries")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run all tutorial parts"""
    print("\n" + "="*60)
    print("Production-Ready Features Tutorial")
    print("llmkit - Production Best Practices")
    print("="*60)

    parts = [
        ("Token Counting & Cost Management", part1_token_counting),
        ("Prompt Templates", part2_prompt_templates),
        ("Evaluation Metrics", part3_evaluation_metrics),
        ("Fine-tuning Preparation", part4_finetuning_prep),
        ("Error Handling", part5_error_handling),
        ("Production Best Practices", part6_best_practices)
    ]

    for name, func in parts:
        try:
            func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("Tutorial Complete!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Review theory: docs/theory/09_production_features_theory.md")
    print("2. Implement a production RAG system with cost optimization")
    print("3. Set up monitoring and error handling")
    print("4. Experiment with prompt templates and evaluation")
    print("5. Try fine-tuning on your own dataset")


if __name__ == "__main__":
    main()
