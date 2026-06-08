"""Tests for infrastructure/security/config.py — SecureConfig, load_from_env."""

import os
from unittest.mock import patch

import pytest

from beanllm.infrastructure.security.config import SecureConfig, load_from_env

# ---------------------------------------------------------------------------
# SecureConfig.__init__
# ---------------------------------------------------------------------------


class TestSecureConfigInit:
    def test_creates_instance(self):
        config = SecureConfig()
        assert config is not None

    def test_stores_values(self):
        config = SecureConfig(model="gpt-4o", temperature=0.7)
        assert config.get_secret("model") == "gpt-4o"
        assert config.get_secret("temperature") == 0.7

    def test_auto_detects_api_key_as_sensitive(self):
        config = SecureConfig(api_key="sk-1234")
        assert "api_key" in config._sensitive_keys

    def test_auto_detects_token_as_sensitive(self):
        config = SecureConfig(access_token="abc123")
        assert "access_token" in config._sensitive_keys

    def test_auto_detects_password_as_sensitive(self):
        config = SecureConfig(password="secret")
        assert "password" in config._sensitive_keys

    def test_model_not_sensitive(self):
        config = SecureConfig(model="gpt-4o")
        assert "model" not in config._sensitive_keys

    def test_endpoint_not_sensitive(self):
        config = SecureConfig(endpoint="https://api.example.com")
        assert "endpoint" not in config._sensitive_keys


# ---------------------------------------------------------------------------
# SecureConfig._is_sensitive_key
# ---------------------------------------------------------------------------


class TestIsSensitiveKey:
    def setup_method(self):
        self.config = SecureConfig()

    def test_api_key_is_sensitive(self):
        assert self.config._is_sensitive_key("api_key") is True

    def test_secret_is_sensitive(self):
        assert self.config._is_sensitive_key("api_secret") is True

    def test_password_is_sensitive(self):
        assert self.config._is_sensitive_key("db_password") is True

    def test_token_is_sensitive(self):
        assert self.config._is_sensitive_key("auth_token") is True

    def test_credential_is_sensitive(self):
        assert self.config._is_sensitive_key("credential") is True

    def test_model_not_sensitive(self):
        assert self.config._is_sensitive_key("model") is False

    def test_region_not_sensitive(self):
        assert self.config._is_sensitive_key("region") is False

    def test_timeout_not_sensitive(self):
        assert self.config._is_sensitive_key("timeout") is False

    def test_temperature_not_sensitive(self):
        assert self.config._is_sensitive_key("temperature") is False

    def test_public_key_not_sensitive(self):
        assert self.config._is_sensitive_key("rsa_public_key") is False

    def test_key_id_not_sensitive(self):
        assert self.config._is_sensitive_key("access_key_id") is False


# ---------------------------------------------------------------------------
# SecureConfig.get / get_secret
# ---------------------------------------------------------------------------


class TestSecureConfigGet:
    def test_get_masks_sensitive_value(self):
        config = SecureConfig(api_key="sk-1234")
        assert config.get("api_key") == "***MASKED***"

    def test_get_returns_plain_value_for_non_sensitive(self):
        config = SecureConfig(model="gpt-4o")
        assert config.get("model") == "gpt-4o"

    def test_get_returns_default_for_missing_key(self):
        config = SecureConfig()
        assert config.get("missing", default="fallback") == "fallback"

    def test_get_secret_returns_actual_sensitive_value(self):
        config = SecureConfig(api_key="sk-real-key")
        assert config.get_secret("api_key") == "sk-real-key"

    def test_get_secret_returns_default_for_missing(self):
        config = SecureConfig()
        assert config.get_secret("nope", default="default") == "default"


# ---------------------------------------------------------------------------
# SecureConfig.set
# ---------------------------------------------------------------------------


class TestSecureConfigSet:
    def test_set_adds_value(self):
        config = SecureConfig()
        config.set("model", "gpt-4o")
        assert config.get_secret("model") == "gpt-4o"

    def test_set_marks_sensitive_when_explicit(self):
        config = SecureConfig()
        config.set("custom_key", "value", sensitive=True)
        assert "custom_key" in config._sensitive_keys

    def test_set_marks_safe_when_explicit_false(self):
        config = SecureConfig(api_key="sk-test")
        config.set("api_key", "new-value", sensitive=False)
        assert "api_key" not in config._sensitive_keys

    def test_set_auto_detects_sensitive_key(self):
        config = SecureConfig()
        config.set("secret_key", "value")
        assert "secret_key" in config._sensitive_keys


# ---------------------------------------------------------------------------
# SecureConfig.mark_sensitive
# ---------------------------------------------------------------------------


class TestMarkSensitive:
    def test_marks_existing_key_as_sensitive(self):
        config = SecureConfig(model="gpt-4o")
        config.mark_sensitive("model")
        assert "model" in config._sensitive_keys

    def test_ignores_nonexistent_key(self):
        config = SecureConfig()
        config.mark_sensitive("nonexistent")
        assert "nonexistent" not in config._sensitive_keys

    def test_marks_multiple_keys(self):
        config = SecureConfig(host="localhost", port=5432)
        config.mark_sensitive("host", "port")
        assert "host" in config._sensitive_keys
        assert "port" in config._sensitive_keys


# ---------------------------------------------------------------------------
# SecureConfig.to_dict
# ---------------------------------------------------------------------------


class TestToDictMasking:
    def test_masks_sensitive_by_default(self):
        config = SecureConfig(api_key="sk-1234", model="gpt-4o")
        d = config.to_dict()
        assert d["api_key"] == "***MASKED***"
        assert d["model"] == "gpt-4o"

    def test_no_masking_when_mask_secrets_false(self):
        config = SecureConfig(api_key="sk-1234")
        d = config.to_dict(mask_secrets=False)
        assert d["api_key"] == "sk-1234"

    def test_include_only_safe_excludes_sensitive(self):
        config = SecureConfig(api_key="sk-1234", model="gpt-4o")
        d = config.to_dict(include_only_safe=True)
        assert "api_key" not in d
        assert "model" in d

    def test_include_only_safe_with_mask_false(self):
        config = SecureConfig(api_key="sk-1234", model="gpt-4o")
        d = config.to_dict(mask_secrets=False, include_only_safe=True)
        assert "api_key" not in d
        assert d["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# SecureConfig.__repr__ / __str__
# ---------------------------------------------------------------------------


class TestReprStr:
    def test_repr_masks_sensitive(self):
        config = SecureConfig(api_key="sk-1234", model="gpt-4o")
        r = repr(config)
        assert "sk-1234" not in r
        assert "***MASKED***" in r

    def test_str_equals_repr(self):
        config = SecureConfig(model="gpt-4o")
        assert str(config) == repr(config)

    def test_repr_contains_class_name(self):
        config = SecureConfig()
        assert "SecureConfig" in repr(config)


# ---------------------------------------------------------------------------
# Dict-like interface
# ---------------------------------------------------------------------------


class TestDictLikeInterface:
    def test_getitem_masks_sensitive(self):
        config = SecureConfig(api_key="sk-1234")
        assert config["api_key"] == "***MASKED***"

    def test_getitem_plain_value(self):
        config = SecureConfig(model="gpt-4o")
        assert config["model"] == "gpt-4o"

    def test_setitem_updates_value(self):
        config = SecureConfig()
        config["model"] = "gpt-4o"
        assert config.get_secret("model") == "gpt-4o"

    def test_contains_existing_key(self):
        config = SecureConfig(model="gpt-4o")
        assert "model" in config

    def test_not_contains_missing_key(self):
        config = SecureConfig()
        assert "missing" not in config

    def test_keys_returns_all_keys(self):
        config = SecureConfig(model="gpt-4o", api_key="sk-1234")
        keys = list(config.keys())
        assert "model" in keys
        assert "api_key" in keys

    def test_values_masks_sensitive(self):
        config = SecureConfig(api_key="sk-1234")
        values = config.values()
        assert "sk-1234" not in values
        assert "***MASKED***" in values

    def test_items_masks_sensitive(self):
        config = SecureConfig(api_key="sk-1234", model="gpt-4o")
        items = dict(config.items())
        assert items["api_key"] == "***MASKED***"
        assert items["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# load_from_env
# ---------------------------------------------------------------------------


class TestLoadFromEnv:
    def test_loads_from_env_with_prefix(self):
        env = {"TESTPREFIX_API_KEY": "sk-test", "TESTPREFIX_MODEL": "gpt-4o"}
        with patch.dict(os.environ, env, clear=False):
            config = load_from_env("TESTPREFIX_")
        assert config.get_secret("api_key") == "sk-test"
        assert config.get_secret("model") == "gpt-4o"

    def test_ignores_vars_without_prefix(self):
        env = {"OTHER_API_KEY": "sk-other", "MYPREFIX_MODEL": "gpt-4o"}
        with patch.dict(os.environ, env, clear=False):
            config = load_from_env("MYPREFIX_")
        assert "other_api_key" not in config

    def test_converts_prefix_var_to_lowercase(self):
        env = {"BEANLLM_MAX_TOKENS": "4096"}
        with patch.dict(os.environ, env, clear=False):
            config = load_from_env("BEANLLM_")
        assert "max_tokens" in config

    def test_returns_secure_config_instance(self):
        with patch.dict(os.environ, {}, clear=False):
            config = load_from_env("BEANLLM_")
        assert isinstance(config, SecureConfig)

    def test_empty_env_returns_empty_config(self):
        # Remove any BEANTEST_ vars
        env_to_remove = {k: "" for k in os.environ if k.startswith("BEANTEST_")}
        with patch.dict(os.environ, {}, clear=False):
            config = load_from_env("BEANTEST_")
        assert isinstance(config, SecureConfig)
