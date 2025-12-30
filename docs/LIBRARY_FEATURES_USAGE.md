# 라이브러리 세부 기능 활용 가이드

## ✅ 현재 활용 중인 고급 기능

### PyMuPDF (fitz)

#### 1. 구조화된 텍스트 추출
```python
# layout_analysis=True일 때
structured_text = page.get_text("dict")
# 블록, 라인, 스팬 정보 포함
# - blocks: 텍스트 블록 리스트
# - lines: 각 블록의 라인
# - spans: 각 라인의 텍스트 스팬 (폰트, 크기, 위치)
```

#### 2. 폰트 정보 추출
```python
fonts = page.get_fonts()
# 각 폰트의 이름, 타입, 확장자 정보
```

#### 3. 링크 추출
```python
links = page.get_links()
# 하이퍼링크 URI, 페이지 번호, 타입
```

#### 4. 정확한 이미지 위치
```python
bbox = page.get_image_bbox(img)
# 이미지의 정확한 bounding box 좌표
```

### pdfplumber

#### 1. 레이아웃 보존 텍스트
```python
# layout_analysis=True일 때
text = page.extract_text(layout=True)
# 레이아웃 구조 보존
```

#### 2. 문자 단위 정보
```python
chars = page.chars
# 각 문자의 위치 (x0, y0, x1, y1), 크기, 폰트
```

#### 3. 단어 단위 정보
```python
words = page.words
# 각 단어의 위치 정보
```

#### 4. 하이퍼링크 추출
```python
hyperlinks = page.hyperlinks
# 링크 URI 및 위치 정보
```

## 📊 사용 예시

### 기본 사용 (고급 기능 자동 활성화)
```python
from beanllm.domain.loaders import load_pdf

# 레이아웃 분석 활성화
docs = load_pdf("document.pdf", layout_analysis=True)

# 첫 번째 페이지의 구조화된 정보
page = docs[0]
if "structured_text" in page.metadata:
    # PyMuPDF의 구조화된 텍스트
    blocks = page.metadata["structured_text"]["blocks"]
    
if "chars" in page.metadata:
    # pdfplumber의 문자 단위 정보
    chars = page.metadata["chars"]
    
if "words" in page.metadata:
    # pdfplumber의 단어 단위 정보
    words = page.metadata["words"]
```

### 폰트 정보 활용
```python
docs = load_pdf("document.pdf", strategy="fast")
page = docs[0]

if "fonts" in page.metadata:
    fonts = page.metadata["fonts"]
    # 폰트별 텍스트 스타일 분석 가능
    for font in fonts:
        print(f"Font: {font['name']}, Type: {font['type']}")
```

### 링크 정보 활용
```python
docs = load_pdf("document.pdf", strategy="fast")
page = docs[0]

if "links" in page.metadata:
    links = page.metadata["links"]
    for link in links:
        print(f"Link: {link['uri']}, Page: {link['page']}")
```

## 🎯 활용 시나리오

### 1. 레이아웃 분석
```python
# 다단 문서 처리
docs = load_pdf("two_column.pdf", layout_analysis=True)
# structured_text로 블록 위치 분석 가능
```

### 2. 폰트 기반 구조 인식
```python
# 제목/본문 구분 (폰트 크기로)
docs = load_pdf("document.pdf", strategy="fast")
# fonts 정보로 텍스트 스타일 분석
```

### 3. 정확한 위치 정보
```python
# 이미지/텍스트 정확한 위치
docs = load_pdf("document.pdf", extract_images=True)
# bbox 정보로 정확한 위치 파악
```

## 📝 메타데이터 구조

### PyMuPDF (strategy="fast")
```python
{
    "source": "file.pdf",
    "page": 0,
    "metadata": {
        "fonts": [...],  # layout_analysis=True일 때
        "links": [...],  # 링크가 있을 때
        "structured_text": {...}  # layout_analysis=True일 때
    }
}
```

### pdfplumber (strategy="accurate")
```python
{
    "source": "file.pdf",
    "page": 0,
    "metadata": {
        "hyperlinks": [...],  # 링크가 있을 때
        "chars": [...],  # layout_analysis=True일 때
        "words": [...]  # layout_analysis=True일 때
    }
}
```

