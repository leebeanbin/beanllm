"""
Test Import Example
다른 프로젝트에서 beanllm을 import해서 사용하는 예제
"""

def test_import():
    """패키지 import 테스트"""
    print("=== Testing beanllm Import ===\n")

    # 1. Basic imports
    print("1. Testing basic imports...")
    try:
        from beanllm import get_registry
        print("   ✅ get_registry imported")
    except ImportError as e:
        print(f"   ❌ Failed to import get_registry: {e}")
        return False

    try:
        from beanllm import ProviderFactory
        print("   ✅ ProviderFactory imported")
    except ImportError as e:
        print(f"   ❌ Failed to import ProviderFactory: {e}")
        return False

    try:
        from beanllm import ModelCapabilityInfo, ProviderInfo
        print("   ✅ Data classes imported")
    except ImportError as e:
        print(f"   ❌ Failed to import data classes: {e}")
        return False

    # 2. Test utils
    print("\n2. Testing utils imports...")
    try:
        from beanllm.utils import EnvConfig
        print("   ✅ EnvConfig imported")
        print(f"   Active providers: {EnvConfig.get_active_providers()}")
    except ImportError as e:
        print(f"   ❌ Failed to import EnvConfig: {e}")
        return False

    try:
        from beanllm.utils import ProviderError, retry, get_logger
        print("   ✅ Utils imported (ProviderError, retry, get_logger)")
    except ImportError as e:
        print(f"   ❌ Failed to import utils: {e}")
        return False

    # 3. Test functionality
    print("\n3. Testing functionality...")
    try:
        registry = get_registry()
        models = registry.get_available_models()
        print(f"   ✅ Registry works: {len(models)} models found")
    except Exception as e:
        print(f"   ❌ Registry failed: {e}")
        return False

    # 4. Test CLI
    print("\n4. Testing CLI...")
    try:
        from beanllm.cli import main
        print("   ✅ CLI main imported")
    except ImportError as e:
        print(f"   ❌ Failed to import CLI: {e}")
        return False

    print("\n✅ All imports successful!")
    return True

if __name__ == "__main__":
    success = test_import()
    exit(0 if success else 1)
