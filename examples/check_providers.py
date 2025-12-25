"""
Check Providers Example
각 Provider의 상태와 사용 가능한 모델 확인
"""

from beanllm import get_registry

def main():
    print("=== Provider Status Check ===\n")

    registry = get_registry()

    # Get all providers
    providers = registry.get_all_providers()

    for name, provider in providers.items():
        status_icon = "✅" if provider.status.value == "active" else "❌"

        print(f"{status_icon} {name.upper()}")
        print(f"   Status: {provider.status.value}")
        print(f"   Env Key: {provider.env_key}")
        print(f"   API Key Set: {'Yes' if provider.env_value_set else 'No'}")
        print(f"   Available Models: {len(provider.available_models)}")

        if provider.default_model:
            print(f"   Default Model: {provider.default_model}")

        if provider.available_models and len(provider.available_models) > 0:
            print(f"   Models:")
            for model in provider.available_models[:3]:  # Show first 3
                print(f"     - {model}")
            if len(provider.available_models) > 3:
                print(f"     ... and {len(provider.available_models) - 3} more")

        print()

if __name__ == "__main__":
    main()
