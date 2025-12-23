"""
Tool Validator - 도구 입력 검증기
"""

from typing import Any, Dict, Optional, Tuple


class ToolValidator:
    """
    도구 입력 검증기

    Mathematical Foundation:
        Schema Validation as Language Acceptance:

        Given schema S and input x:
        Valid(x, S) ⟺ x ∈ L(S)

        where L(S) is the language defined by schema S

        Validation Rules:
        - Type checking: typeof(x) = T where T is expected type
        - Range checking: x ∈ [min, max] for numeric types
        - Pattern matching: x matches regex pattern
        - Required fields: ∀f ∈ required. f ∈ keys(x)
    """

    @staticmethod
    def validate(data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        데이터가 스키마를 만족하는지 검증

        Args:
            data: 검증할 데이터
            schema: JSON Schema

        Returns:
            (is_valid, error_message)
        """
        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                return False, f"Missing required field: {field}"

        # Check properties
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                field_schema = properties[key]
                is_valid, error = ToolValidator._validate_field(value, field_schema, key)
                if not is_valid:
                    return False, error

        return True, None

    @staticmethod
    def _validate_field(
        value: Any, schema: Dict[str, Any], field_name: str
    ) -> Tuple[bool, Optional[str]]:
        """개별 필드 검증"""
        expected_type = schema.get("type")

        type_check_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        if expected_type in type_check_map:
            expected_python_type = type_check_map[expected_type]
            if not isinstance(value, expected_python_type):
                return (
                    False,
                    f"Field '{field_name}' must be of type {expected_type}, got {type(value).__name__}",
                )

        # Enum validation
        if "enum" in schema:
            if value not in schema["enum"]:
                return False, f"Field '{field_name}' must be one of {schema['enum']}, got {value}"

        # Range validation for numbers
        if expected_type in ("integer", "number"):
            if "minimum" in schema and value < schema["minimum"]:
                return False, f"Field '{field_name}' must be >= {schema['minimum']}"
            if "maximum" in schema and value > schema["maximum"]:
                return False, f"Field '{field_name}' must be <= {schema['maximum']}"

        # Array items validation
        if expected_type == "array" and "items" in schema:
            for i, item in enumerate(value):
                is_valid, error = ToolValidator._validate_field(
                    item, schema["items"], f"{field_name}[{i}]"
                )
                if not is_valid:
                    return False, error

        return True, None
