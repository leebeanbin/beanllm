# 라이브러리 고급 기능 활용 가이드

## ✅ 각 라이브러리의 세부 기능 완전 지원

beanPDFLoader는 PyMuPDF와 pdfplumber의 **모든 고급 기능**을 활용할 수 있도록 설계되었습니다.

## 🎯 PyMuPDF 고급 기능

### 1. 텍스트 추출 모드

```python
from beanllm.domain.loaders.pdf import beanPDFLoader

# 기본 텍스트
loader = beanPDFLoader("doc.pdf", pymupdf_text_mode="text")

# 구조화된 텍스트 (블록, 라인, 스팬 정보)
loader = beanPDFLoader("doc.pdf", pymupdf_text_mode="dict")
# → structured_text에 블록, 라인, 스팬 정보 포함

# HTML 형식
loader = beanPDFLoader("doc.pdf", pymupdf_text_mode="html")

# XML 형식
loader = beanPDFLoader("doc.pdf", pymupdf_text_mode="xml")

# JSON 형식
loader = beanPDFLoader("doc.pdf", pymupdf_text_mode="json")
```

### 2. 폰트 정보 추출

```python
loader = beanPDFLoader("doc.pdf", pymupdf_extract_fonts=True)
docs = loader.load()

# 각 페이지의 폰트 정보
for doc in docs:
    if "fonts" in doc.metadata:
        for font in doc.metadata["fonts"]:
            print(f"Font: {font['name']}, Type: {font['type']}")
```

### 3. 링크 추출

```python
loader = beanPDFLoader("doc.pdf", pymupdf_extract_links=True)
docs = loader.load()

# 각 페이지의 링크 정보
for doc in docs:
    if "links" in doc.metadata:
        for link in doc.metadata["links"]:
            print(f"Link: {link['uri']}, Page: {link['page']}")
```

## 🎯 pdfplumber 고급 기능

### 1. 레이아웃 보존 텍스트

```python
loader = beanPDFLoader("doc.pdf", pdfplumber_layout=True)
# 또는
loader = beanPDFLoader("doc.pdf", layout_analysis=True)  # 자동 활성화
```

### 2. 문자 단위 정보

```python
loader = beanPDFLoader("doc.pdf", pdfplumber_extract_chars=True)
docs = loader.load()

# 각 문자의 위치, 크기 정보
for doc in docs:
    if "chars" in doc.metadata:
        for char in doc.metadata["chars"]:
            print(f"Char: {char['text']}, Position: ({char['x0']}, {char['y0']})")
```

### 3. 단어 단위 정보

```python
loader = beanPDFLoader("doc.pdf", pdfplumber_extract_words=True)
docs = loader.load()

# 각 단어의 위치 정보
for doc in docs:
    if "words" in doc.metadata:
        for word in doc.metadata["words"]:
            print(f"Word: {word['text']}, BBox: ({word['x0']}, {word['y0']}, {word['x1']}, {word['y1']})")
```

### 4. 하이퍼링크 추출

```python
loader = beanPDFLoader("doc.pdf", pdfplumber_extract_hyperlinks=True)
docs = loader.load()

# 각 페이지의 하이퍼링크
for doc in docs:
    if "hyperlinks" in doc.metadata:
        for link in doc.metadata["hyperlinks"]:
            print(f"Link: {link['uri']}, Position: ({link['x0']}, {link['y0']})")
```

### 5. 공백 허용도 조정

```python
# 수평/수직 공백 허용도 조정 (밀집된 텍스트 처리)
loader = beanPDFLoader(
    "doc.pdf",
    pdfplumber_x_tolerance=5.0,  # 수평 공백 허용도 증가
    pdfplumber_y_tolerance=5.0,  # 수직 공백 허용도 증가
)
```

## 📊 통합 사용 예시

### 모든 고급 기능 활성화

```python
loader = beanPDFLoader(
    "document.pdf",
    # 기본 옵션
    extract_tables=True,
    extract_images=True,
    layout_analysis=True,  # 자동으로 여러 고급 기능 활성화
    
    # PyMuPDF 고급 옵션
    pymupdf_text_mode="dict",  # 구조화된 텍스트
    pymupdf_extract_fonts=True,
    pymupdf_extract_links=True,
    
    # pdfplumber 고급 옵션
    pdfplumber_layout=True,
    pdfplumber_extract_chars=True,
    pdfplumber_extract_words=True,
    pdfplumber_extract_hyperlinks=True,
)

docs = loader.load()

# 모든 정보 활용
for doc in docs:
    print(f"Page {doc.metadata['page']}:")
    print(f"  Text: {doc.content[:100]}...")
    
    if "structured_text" in doc.metadata:
        print(f"  Blocks: {len(doc.metadata['structured_text']['blocks'])}")
    
    if "fonts" in doc.metadata:
        print(f"  Fonts: {len(doc.metadata['fonts'])}")
    
    if "links" in doc.metadata:
        print(f"  Links: {len(doc.metadata['links'])}")
    
    if "chars" in doc.metadata:
        print(f"  Chars: {len(doc.metadata['chars'])}")
    
    if "words" in doc.metadata:
        print(f"  Words: {len(doc.metadata['words'])}")
```

## 🚀 Factory 패턴에서도 사용 가능

```python
from beanllm.domain.loaders import DocumentLoader

# 고급 옵션 자동 감지
docs = DocumentLoader.load(
    "document.pdf",
    extract_tables=True,  # beanPDFLoader 자동 사용
    layout_analysis=True,  # 모든 고급 기능 활성화
    pymupdf_extract_fonts=True,  # PyMuPDF 고급 옵션
    pdfplumber_extract_chars=True,  # pdfplumber 고급 옵션
)
```

## 📝 지원되는 모든 옵션

### PyMuPDF 옵션
- `pymupdf_text_mode`: "text" | "dict" | "rawdict" | "html" | "xml" | "json"
- `pymupdf_extract_fonts`: bool
- `pymupdf_extract_links`: bool

### pdfplumber 옵션
- `pdfplumber_layout`: bool
- `pdfplumber_extract_chars`: bool
- `pdfplumber_extract_words`: bool
- `pdfplumber_extract_hyperlinks`: bool
- `pdfplumber_x_tolerance`: float
- `pdfplumber_y_tolerance`: float

## 💡 자동 활성화

`layout_analysis=True`로 설정하면 다음 기능들이 자동으로 활성화됩니다:
- `pymupdf_text_mode="dict"` (구조화된 텍스트)
- `pymupdf_extract_fonts=True`
- `pymupdf_extract_links=True`
- `pdfplumber_layout=True`
- `pdfplumber_extract_chars=True`
- `pdfplumber_extract_words=True`
- `pdfplumber_extract_hyperlinks=True`


