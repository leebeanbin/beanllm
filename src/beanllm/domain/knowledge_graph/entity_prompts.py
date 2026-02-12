"""
Entity Extraction Prompts - LLM 프롬프트 템플릿

EntityExtractor에서 사용하는 LLM 프롬프트를 중앙 관리.
프롬프트 튜닝 시 이 파일만 수정하면 됨.
"""

ENTITY_EXTRACTION_PROMPT = """Extract named entities from the following text.

Text:
{text}

Entity types to extract: {entity_types}

Return a JSON array with the following structure:
[
  {{
    "name": "Entity name",
    "type": "person|organization|location|concept|event|date|product|technology|other",
    "description": "Brief description of the entity",
    "aliases": ["alternative names"],
    "confidence": 0.95
  }}
]

Rules:
1. Only extract entities that clearly appear in the text
2. Assign appropriate entity types based on context
3. Include common aliases if applicable
4. Set confidence based on how certain you are (0.0-1.0)
5. Return ONLY valid JSON, no explanation

JSON:"""

COREFERENCE_RESOLUTION_PROMPT = """Resolve coreferences in the following text.

Text:
{text}

Known entities:
{entities}

Identify pronouns (he, she, it, they, etc.) and other references that point to the known entities.

Return a JSON array mapping references to entity names:
[
  {{
    "reference": "He",
    "resolved_to": "Steve Jobs",
    "start_position": 45,
    "confidence": 0.95
  }}
]

Return ONLY valid JSON:"""

PROPERTY_EXTRACTION_PROMPT = """Extract properties for the entity "{entity_name}" from the following text.

Text:
{text}

Entity Type: {entity_type}

Extract relevant properties based on the entity type:
- For PERSON: role, title, affiliation, birth_date, nationality, etc.
- For ORGANIZATION: industry, founded, headquarters, ceo, employees, etc.
- For LOCATION: country, population, coordinates, etc.
- For PRODUCT: manufacturer, release_date, price, category, etc.
- For TECHNOLOGY: developer, version, release_date, etc.

Return a JSON object with extracted properties:
{{
  "property_name": "property_value",
  ...
}}

Return ONLY valid JSON:"""
