"""
Model Parameters Example
모델별 파라미터 정보 확인
"""

from beanllm import get_registry

def main():
    print("=== Model Parameters Check ===\n")

    registry = get_registry()

    # Check different model types
    models_to_check = [
        "gpt-4o",           # Standard OpenAI
        "gpt-5-mini",       # New OpenAI (max_completion_tokens)
        "gpt-5-nano",       # New OpenAI (no temperature)
        "claude-3-5-sonnet-20241022",  # Claude
        "gemini-2.5-flash", # Gemini
        "phi3.5",           # Ollama
    ]

    for model_name in models_to_check:
        model_info = registry.get_model_info(model_name)

        if not model_info:
            print(f"❌ {model_name}: Not found")
            print()
            continue

        print(f"📦 {model_name}")
        print(f"   Provider: {model_info.provider}")
        print(f"   Type: {model_info.model_type}")
        print(f"\n   Capabilities:")
        print(f"     Streaming: {'✅' if model_info.supports_streaming else '❌'}")
        print(f"     Temperature: {'✅' if model_info.supports_temperature else '❌'}")

        if model_info.supports_temperature:
            print(f"       Range: {model_info.default_temperature}")

        print(f"     Max Tokens: {'✅' if model_info.supports_max_tokens else '❌'}")

        if model_info.uses_max_completion_tokens:
            print(f"     ⚠️ Uses 'max_completion_tokens' instead of 'max_tokens'")

        print(f"     Max Value: {model_info.max_tokens}")

        print(f"\n   Parameters:")
        for param in model_info.parameters:
            status = '✅' if param.supported else '❌'
            print(f"     {status} {param.name} ({param.type})")
            if param.notes:
                print(f"        Note: {param.notes}")

        print()

if __name__ == "__main__":
    main()
