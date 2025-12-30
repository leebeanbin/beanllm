"""
테스트 픽스처 PDF 파일 생성 스크립트

이 스크립트는 beanPDFLoader 테스트에 필요한 PDF 파일들을 자동으로 생성합니다.
"""

import fitz  # PyMuPDF
from pathlib import Path


def generate_simple_pdf():
    """simple.pdf - 기본 텍스트 추출 테스트용"""
    doc = fitz.open()

    # 페이지 1
    page = doc.new_page(width=595, height=842)  # A4 크기
    text = """beanPDFLoader Test Document

This is a simple PDF document for testing text extraction.

Section 1: Introduction
This document contains plain text without any tables or images.
It is designed to test basic text extraction functionality.

Section 2: Features
- Fast text extraction
- Multiple page support
- Unicode character support: 한글, 中文, 日本語

Section 3: Technical Details
The beanPDFLoader uses a 3-layer architecture:
1. Fast Layer (PyMuPDF)
2. Accurate Layer (pdfplumber)
3. ML Layer (future implementation)

End of page 1."""

    page.insert_text((50, 50), text, fontsize=12, fontname="helv")

    # 페이지 2
    page2 = doc.new_page(width=595, height=842)
    text2 = """Page 2: Additional Content

Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
nisi ut aliquip ex ea commodo consequat.

End of document."""

    page2.insert_text((50, 50), text2, fontsize=12, fontname="helv")

    output_path = Path(__file__).parent / "simple.pdf"
    doc.save(output_path)
    doc.close()
    print(f"✓ Created: {output_path}")


def generate_tables_pdf():
    """tables.pdf - 테이블 추출 테스트용"""
    doc = fitz.open()

    # 페이지 1 - 간단한 테이블
    page = doc.new_page(width=595, height=842)

    # 제목
    page.insert_text((50, 50), "Test Document with Tables", fontsize=16, fontname="helv")

    # 테이블 1: 간단한 2x3 테이블
    page.insert_text((50, 100), "Table 1: Simple Table", fontsize=14, fontname="helv")

    # 테이블 그리기 (수동으로 선 그리기)
    table_data = [
        ["Name", "Age", "City"],
        ["Alice", "30", "New York"],
        ["Bob", "25", "London"],
        ["Charlie", "35", "Tokyo"],
    ]

    x_start = 50
    y_start = 130
    col_widths = [150, 100, 150]
    row_height = 25

    # 테이블 테두리 그리기
    for i, row in enumerate(table_data):
        y = y_start + i * row_height
        x = x_start

        # 셀 그리기
        for j, cell in enumerate(row):
            # 텍스트
            page.insert_text((x + 5, y + 18), cell, fontsize=11, fontname="helv")

            # 테두리
            rect = fitz.Rect(x, y, x + col_widths[j], y + row_height)
            page.draw_rect(rect, color=(0, 0, 0), width=0.5)

            x += col_widths[j]

    # 테이블 2: 숫자 데이터
    page.insert_text((50, 280), "Table 2: Sales Data", fontsize=14, fontname="helv")

    table_data2 = [
        ["Product", "Q1", "Q2", "Q3", "Q4"],
        ["Product A", "100", "150", "200", "180"],
        ["Product B", "80", "90", "120", "110"],
        ["Product C", "120", "140", "160", "150"],
    ]

    y_start2 = 310
    col_widths2 = [120, 80, 80, 80, 80]

    for i, row in enumerate(table_data2):
        y = y_start2 + i * row_height
        x = x_start

        for j, cell in enumerate(row):
            page.insert_text((x + 5, y + 18), cell, fontsize=11, fontname="helv")
            rect = fitz.Rect(x, y, x + col_widths2[j], y + row_height)
            page.draw_rect(rect, color=(0, 0, 0), width=0.5)
            x += col_widths2[j]

    output_path = Path(__file__).parent / "tables.pdf"
    doc.save(output_path)
    doc.close()
    print(f"✓ Created: {output_path}")


def generate_images_pdf():
    """images.pdf - 이미지 추출 테스트용"""
    doc = fitz.open()

    page = doc.new_page(width=595, height=842)

    # 제목
    page.insert_text((50, 50), "Test Document with Images", fontsize=16, fontname="helv")

    # 간단한 "이미지" 생성 (사각형으로 대체)
    # 실제 이미지 대신 색상 사각형을 그려서 이미지처럼 보이게 함
    page.insert_text((50, 100), "Image 1: Red Rectangle", fontsize=12, fontname="helv")

    # 빨간 사각형 (이미지 역할)
    rect1 = fitz.Rect(50, 120, 250, 270)
    page.draw_rect(rect1, color=(1, 0, 0), fill=(1, 0, 0))

    page.insert_text((50, 300), "Image 2: Blue Circle", fontsize=12, fontname="helv")

    # 파란 원 (이미지 역할)
    # PyMuPDF는 fill 옵션으로 도형을 채울 수 있습니다
    center = fitz.Point(150, 400)
    page.draw_circle(center, 50, color=(0, 0, 1), fill=(0, 0, 1))

    page.insert_text((50, 480), "This document contains graphical elements.", fontsize=11, fontname="helv")
    page.insert_text((50, 500), "beanPDFLoader should be able to extract metadata about these elements.", fontsize=11, fontname="helv")

    output_path = Path(__file__).parent / "images.pdf"
    doc.save(output_path)
    doc.close()
    print(f"✓ Created: {output_path}")


def main():
    """모든 테스트 픽스처 생성"""
    print("Generating test PDF fixtures...")
    print("-" * 50)

    generate_simple_pdf()
    generate_tables_pdf()
    generate_images_pdf()

    print("-" * 50)
    print("✓ All test fixtures generated successfully!")
    print("\nGenerated files:")
    print("  - simple.pdf  (2 pages, plain text)")
    print("  - tables.pdf  (1 page, 2 tables)")
    print("  - images.pdf  (1 page, graphics)")


if __name__ == "__main__":
    main()
