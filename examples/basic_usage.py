"""
Basic Usage Example
beanllm의 기본 사용법
"""

from beanllm import get_registry

def main():
    print("=== beanllm Basic Usage ===\n")

    # 1. Get registry
    print("1. Getting model registry...")
    registry = get_registry()
    print("✅ Registry loaded\n")

    # 2. Check active providers
    print("2. Checking active providers...")
    active_providers = registry.get_active_providers()
    print(f"Active providers: {[p.name for p in active_providers]}")
    print(f"Total: {len(active_providers)} providers\n")

    # 3. Get all available models
    print("3. Getting available models...")
    models = registry.get_available_models()
    print(f"Total models: {len(models)}\n")

    # Show first 5 models
    print("First 5 models:")
    for model in models[:5]:
        print(f"  - {model.model_name} ({model.provider})")
    print()

    # 4. Get specific model info
    print("4. Getting specific model info...")
    model_name = "gpt-4o-mini"
    model_info = registry.get_model_info(model_name)

    if model_info:
        print(f"\nModel: {model_info.model_name}")
        print(f"Provider: {model_info.provider}")
        print(f"Display Name: {model_info.display_name}")
        print(f"Description: {model_info.description}")
        print(f"\nCapabilities:")
        print(f"  - Streaming: {model_info.supports_streaming}")
        print(f"  - Temperature: {model_info.supports_temperature}")
        print(f"  - Max Tokens: {model_info.supports_max_tokens}")
        print(f"  - Uses max_completion_tokens: {model_info.uses_max_completion_tokens}")
        print(f"  - Max tokens value: {model_info.max_tokens}")
    else:
        print(f"Model {model_name} not found")

    # 5. Get summary
    print("\n5. Getting summary...")
    summary = registry.get_summary()
    print(f"\nSummary:")
    print(f"  Total providers: {summary['total_providers']}")
    print(f"  Active providers: {summary['active_providers']}")
    print(f"  Total models: {summary['total_models']}")

    print("\n✅ Done!")

if __name__ == "__main__":
    main()
