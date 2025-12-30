# 라이브러리 세부 기능 활용 분석

## 현재 상태 분석

### PyMuPDF (fitz) - 현재 사용 중인 기능

✅ **사용 중:**
- `page.get_text()` - 기본 텍스트 추출
- `page.get_images()` - 이미지 리스트
- `doc.extract_image()` - 이미지 데이터 추출
- `doc.metadata` - 문서 메타데이터
- `page.rect` - 페이지 크기

❌ **미사용 (고급 기능):**
- `page.get_text("dict")` - 구조화된 텍스트 (블록, 라인, 스팬 정보)
- `page.get_text("rawdict")` - 더 상세한 정보 (폰트, 색상, 크기)
- `page.get_text("html")` - HTML 형식 추출
- `page.get_text("xml")` - XML 형식 추출
- `page.get_text("json")` - JSON 형식 추출
- `page.get_text("textdict")` - 텍스트 + 딕셔너리
- `page.get_fonts()` - 폰트 정보 추출
- `page.get_links()` - 링크 추출
- `page.get_annotations()` - 주석 추출
- `page.get_drawings()` - 도형 추출
- `page.get_image_bbox()` - 이미지 정확한 위치
- `page.search_for()` - 텍스트 검색
- `page.get_text_blocks()` - 텍스트 블록 추출

### pdfplumber - 현재 사용 중인 기능

✅ **사용 중:**
- `page.extract_text()` - 기본 텍스트 추출
- `page.extract_tables()` - 테이블 추출
- `page.find_tables()` - 테이블 위치 찾기
- `page.bbox` - 페이지 경계

❌ **미사용 (고급 기능):**
- `page.chars` - 문자 단위 추출 (위치, 폰트, 크기)
- `page.words` - 단어 단위 추출 (위치 정보 포함)
- `page.lines` - 줄 단위 추출
- `page.rects` - 사각형 도형 추출
- `page.lines` - 선 도형 추출
- `page.curves` - 곡선 도형 추출
- `page.hyperlinks` - 하이퍼링크 추출
- `page.images` - 이미지 정보
- `page.crop(bbox)` - 특정 영역만 추출
- `page.within_bbox(bbox)` - 특정 영역 내 요소만
- `page.extract_text(layout=True)` - 레이아웃 보존
- `page.extract_text(x_tolerance=3, y_tolerance=3)` - 공백 허용도 조정
- `page.extract_words()` - 단어 단위 추출
- `page.extract_text_lines()` - 줄 단위 추출

## 개선 방안

### 1. PyMuPDF 고급 기능 활용

#### 레이아웃 분석
```python
# 현재: page.get_text()
# 개선: page.get_text("dict") - 구조화된 정보
blocks = page.get_text("dict")["blocks"]
for block in blocks:
    if "lines" in block:
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"]
                font = span["font"]  # 폰트 정보
                size = span["size"]  # 폰트 크기
                bbox = span["bbox"]  # 정확한 위치
```

#### 폰트 정보 추출
```python
fonts = page.get_fonts()
# 폰트별 텍스트 스타일 분석 가능
```

#### 링크 추출
```python
links = page.get_links()
# 하이퍼링크 정보 추출
```

### 2. pdfplumber 고급 기능 활용

#### 문자/단어 단위 추출
```python
# 현재: page.extract_text()
# 개선: page.chars, page.words
chars = page.chars  # 각 문자의 위치, 폰트, 크기
words = page.words  # 각 단어의 위치 정보
```

#### 레이아웃 보존 텍스트
```python
# 현재: page.extract_text()
# 개선: page.extract_text(layout=True)
text = page.extract_text(layout=True)  # 레이아웃 보존
```

#### 특정 영역만 추출
```python
# 특정 영역만 추출
cropped = page.crop((x0, y0, x1, y1))
text = cropped.extract_text()
```

## 구현 우선순위

### P0 (즉시 구현)
1. PyMuPDF: `get_text("dict")` - 구조화된 텍스트 추출
2. pdfplumber: `extract_text(layout=True)` - 레이아웃 보존
3. pdfplumber: `chars`, `words` - 문자/단어 단위 정보

### P1 (중요)
4. PyMuPDF: `get_fonts()` - 폰트 정보
5. PyMuPDF: `get_links()` - 링크 추출
6. pdfplumber: `hyperlinks` - 하이퍼링크

### P2 (향후)
7. PyMuPDF: `get_annotations()` - 주석
8. pdfplumber: `crop()`, `within_bbox()` - 영역 추출


