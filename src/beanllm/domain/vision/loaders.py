"""
Vision Document Loaders - 이미지 및 멀티모달 문서 로딩
"""

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from ...domain.loaders import BaseDocumentLoader, Document


@dataclass
class ImageDocument(Document):
    """
    이미지 문서

    텍스트와 이미지를 함께 포함
    """

    image_path: Optional[str] = None
    image_data: Optional[bytes] = None
    image_base64: Optional[str] = None
    caption: Optional[str] = None  # 이미지 캡션 (자동 생성 가능)

    def get_image_base64(self) -> str:
        """이미지를 Base64로 인코딩"""
        if self.image_base64:
            return self.image_base64

        if self.image_data:
            return base64.b64encode(self.image_data).decode("utf-8")

        if self.image_path:
            with open(self.image_path, "rb") as f:
                image_bytes = f.read()
                return base64.b64encode(image_bytes).decode("utf-8")

        raise ValueError("No image data available")


class ImageLoader(BaseDocumentLoader):
    """
    이미지 로더

    단일 이미지 또는 디렉토리의 이미지들을 로드

    Example:
        # 단일 이미지
        loader = ImageLoader()
        docs = loader.load("image.jpg")

        # 디렉토리
        docs = loader.load("images/")

        # 캡션 자동 생성
        loader = ImageLoader(generate_captions=True)
        docs = loader.load("image.jpg")
    """

    def __init__(self, generate_captions: bool = False, caption_model: Optional[str] = None):
        """
        Args:
            generate_captions: 이미지 캡션 자동 생성 여부
            caption_model: 캡션 생성 모델 (기본: BLIP)
        """
        self.generate_captions = generate_captions
        self.caption_model = caption_model or "Salesforce/blip-image-captioning-base"

    def load(self, source: Union[str, Path]) -> List[ImageDocument]:
        """
        이미지 로드

        Args:
            source: 이미지 파일 또는 디렉토리 경로

        Returns:
            ImageDocument 리스트
        """
        source_path = Path(source)

        if source_path.is_file():
            return [self._load_image(source_path)]
        elif source_path.is_dir():
            return self._load_directory(source_path)
        else:
            raise ValueError(f"Invalid source: {source}")

    def _load_image(self, image_path: Path) -> ImageDocument:
        """단일 이미지 로드"""
        # 이미지 읽기
        with open(image_path, "rb") as f:
            image_data = f.read()

        # 캡션 생성
        caption = None
        if self.generate_captions:
            caption = self._generate_caption(image_path)

        return ImageDocument(
            content=caption or f"Image: {image_path.name}",
            metadata={
                "source": str(image_path),
                "type": "image",
                "format": image_path.suffix[1:],  # .jpg -> jpg
            },
            image_path=str(image_path),
            image_data=image_data,
            caption=caption,
        )

    def _load_directory(self, directory: Path) -> List[ImageDocument]:
        """디렉토리의 모든 이미지 로드"""
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        documents = []

        for file_path in directory.rglob("*"):
            if file_path.suffix.lower() in image_extensions:
                documents.append(self._load_image(file_path))

        return documents

    def _generate_caption(self, image_path: Path) -> str:
        """이미지 캡션 자동 생성"""
        try:
            from PIL import Image
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError:
            raise ImportError("transformers 및 Pillow 필요:\npip install transformers pillow")

        # 모델 로드
        processor = BlipProcessor.from_pretrained(self.caption_model)
        model = BlipForConditionalGeneration.from_pretrained(self.caption_model)

        # 이미지 로드
        image = Image.open(image_path)

        # 캡션 생성
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

        return caption


class PDFWithImagesLoader(BaseDocumentLoader):
    """
    PDF 로더 (이미지 포함)

    PDF에서 텍스트와 이미지를 함께 추출

    Example:
        loader = PDFWithImagesLoader()
        docs = loader.load("document.pdf")

        # 이미지 포함 여부
        for doc in docs:
            if isinstance(doc, ImageDocument):
                print(f"Image page: {doc.metadata['page']}")
    """

    def __init__(self, extract_images: bool = True):
        """
        Args:
            extract_images: 이미지 추출 여부
        """
        self.extract_images = extract_images

    def load(self, source: Union[str, Path]) -> List[Union[Document, ImageDocument]]:
        """
        PDF 로드

        Args:
            source: PDF 파일 경로

        Returns:
            Document 및 ImageDocument 리스트
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF 필요:\npip install pymupdf")

        source_path = Path(source)
        documents = []

        # PDF 열기
        pdf_document = fitz.open(source_path)

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]

            # 텍스트 추출
            text = page.get_text()
            if text.strip():
                documents.append(
                    Document(
                        content=text,
                        metadata={"source": str(source_path), "page": page_num + 1, "type": "text"},
                    )
                )

            # 이미지 추출
            if self.extract_images:
                images = page.get_images(full=True)
                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_data = base_image["image"]

                    documents.append(
                        ImageDocument(
                            content=f"Image from page {page_num + 1}",
                            metadata={
                                "source": str(source_path),
                                "page": page_num + 1,
                                "image_index": img_index,
                                "type": "image",
                            },
                            image_data=image_data,
                        )
                    )

        pdf_document.close()
        return documents


# 편의 함수
def load_images(source: Union[str, Path], generate_captions: bool = False) -> List[ImageDocument]:
    """
    이미지 로드 (간편 함수)

    Args:
        source: 이미지 파일 또는 디렉토리
        generate_captions: 캡션 자동 생성

    Returns:
        ImageDocument 리스트

    Example:
        docs = load_images("images/", generate_captions=True)
    """
    loader = ImageLoader(generate_captions=generate_captions)
    return loader.load(source)


def load_pdf_with_images(source: Union[str, Path]) -> List[Union[Document, ImageDocument]]:
    """
    PDF 로드 (이미지 포함)

    Args:
        source: PDF 파일 경로

    Returns:
        Document 및 ImageDocument 리스트

    Example:
        docs = load_pdf_with_images("document.pdf")
    """
    loader = PDFWithImagesLoader()
    return loader.load(source)
