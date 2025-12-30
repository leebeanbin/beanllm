# PDF 테스트 픽스처

이 디렉토리는 beanPDFLoader 테스트에 사용되는 PDF 파일들을 저장합니다.

## 필요한 테스트 파일

### 1. simple.pdf
- **용도**: 기본 텍스트 추출 테스트
- **특징**: 텍스트만 포함, 테이블/이미지 없음
- **페이지 수**: 1-5페이지

### 2. tables.pdf
- **용도**: 테이블 추출 테스트
- **특징**: 다양한 형태의 테이블 포함
- **페이지 수**: 3-10페이지

### 3. images.pdf
- **용도**: 이미지 추출 테스트
- **특징**: 이미지와 텍스트 혼합
- **페이지 수**: 2-5페이지

### 4. large.pdf
- **용도**: 대용량 처리 및 성능 테스트
- **특징**: 100+ 페이지
- **비고**: 선택적 (성능 테스트용)

## 테스트 파일 준비 방법

1. **온라인 샘플 다운로드**:
   - [PDF24 샘플](https://tools.pdf24.org/en/create-pdf)
   - [Adobe 샘플](https://www.adobe.com/acrobat/resources/sample-pdf-files.html)

2. **자체 생성**:
   - Word/Google Docs에서 PDF로 내보내기
   - Python으로 PDF 생성 (reportlab 등)

3. **주의사항**:
   - 저작권 없는 파일만 사용
   - 파일 크기 제한 (각 10MB 이하 권장)

