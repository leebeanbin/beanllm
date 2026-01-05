"""
OCR Interactive Tuning App (Streamlit)

실시간으로 OCR 파라미터를 조정하고 결과를 확인할 수 있는 대시보드.

Usage:
    streamlit run src/beanllm/domain/ocr/tuner_app.py

Features:
- 이미지 업로드
- 실시간 파라미터 조정 (슬라이더)
- 전처리 전/후 비교
- OCR 결과 + BoundingBox 시각화
- 설정 저장/로드 (프리셋)
"""

try:
    import streamlit as st
except ImportError:
    raise ImportError(
        "Streamlit is required for OCR Tuner App. "
        "Install it with: pip install streamlit"
    )

import io
from pathlib import Path

import numpy as np
from PIL import Image

from .bean_ocr import beanOCR
from .models import (
    BinarizeConfig,
    ContrastConfig,
    DenoiseConfig,
    DeskewConfig,
    OCRConfig,
    ResizeConfig,
    SharpenConfig,
)
from .presets import ConfigPresets


def main():
    """메인 Streamlit 앱"""
    st.set_page_config(
        page_title="OCR Parameter Tuner",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔧 OCR Interactive Parameter Tuner")
    st.markdown("**실시간으로 OCR 파라미터를 조정하고 결과를 확인하세요**")

    # 사이드바: 파라미터 조정
    with st.sidebar:
        st.header("⚙️ Parameters")

        # 프리셋 선택
        presets = ConfigPresets()
        preset_names = ["Custom"] + presets.list()
        selected_preset = st.selectbox("Preset", preset_names, index=0)

        if selected_preset != "Custom":
            config = presets.get(selected_preset)
            st.success(f"Loaded preset: **{selected_preset}**")
        else:
            # 커스텀 파라미터 조정
            st.subheader("🎚️ Preprocessing")

            # Denoise
            denoise_enabled = st.checkbox("Denoise", value=True)
            if denoise_enabled:
                denoise_strength = st.select_slider(
                    "Denoise Strength",
                    options=["light", "medium", "strong"],
                    value="medium",
                )
            else:
                denoise_strength = "medium"

            # Contrast
            contrast_enabled = st.checkbox("Contrast Adjustment (CLAHE)", value=True)
            if contrast_enabled:
                clip_limit = st.slider(
                    "CLAHE Clip Limit",
                    min_value=0.5,
                    max_value=5.0,
                    value=2.0,
                    step=0.5,
                )
            else:
                clip_limit = 2.0

            # Binarize
            binarize_enabled = st.checkbox("Binarize", value=False)
            if binarize_enabled:
                binarize_method = st.select_slider(
                    "Binarize Method",
                    options=["otsu", "adaptive", "manual"],
                    value="otsu",
                )
                if binarize_method == "manual":
                    threshold = st.slider(
                        "Manual Threshold",
                        min_value=0,
                        max_value=255,
                        value=127,
                        step=1,
                    )
                else:
                    threshold = 127
            else:
                binarize_method = "otsu"
                threshold = 127

            # Deskew
            deskew_enabled = st.checkbox("Deskew (Rotation Correction)", value=True)
            if deskew_enabled:
                angle_threshold = st.slider(
                    "Angle Threshold (degrees)",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.5,
                    step=0.1,
                )
            else:
                angle_threshold = 0.5

            # Sharpen
            sharpen_enabled = st.checkbox("Sharpen", value=False)
            if sharpen_enabled:
                sharpen_strength = st.slider(
                    "Sharpen Strength",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )
            else:
                sharpen_strength = 0.5

            # Resize
            resize_enabled = st.checkbox("Resize", value=False)
            if resize_enabled:
                max_size = st.slider(
                    "Max Size (px)",
                    min_value=100,
                    max_value=2000,
                    value=1000,
                    step=100,
                )
            else:
                max_size = None

            # OCRConfig 생성
            config = OCRConfig(
                engine="paddleocr",
                language="auto",
                denoise=denoise_enabled,
                denoise_config=DenoiseConfig(
                    enabled=denoise_enabled,
                    strength=denoise_strength,
                ),
                contrast_adjustment=contrast_enabled,
                contrast_config=ContrastConfig(
                    enabled=contrast_enabled,
                    clip_limit=clip_limit,
                ),
                binarize=binarize_enabled,
                binarize_config=BinarizeConfig(
                    enabled=binarize_enabled,
                    method=binarize_method,
                    threshold=threshold,
                ),
                deskew=deskew_enabled,
                deskew_config=DeskewConfig(
                    enabled=deskew_enabled,
                    angle_threshold=angle_threshold,
                ),
                sharpen=sharpen_enabled,
                sharpen_config=SharpenConfig(
                    enabled=sharpen_enabled,
                    strength=sharpen_strength,
                ),
                resize_config=ResizeConfig(
                    enabled=resize_enabled,
                    max_size=max_size,
                ),
            )

        # 설정 저장
        st.subheader("💾 Save Config")
        preset_name = st.text_input("Preset Name", placeholder="my_preset")
        if st.button("Save Preset"):
            if preset_name:
                try:
                    presets.save(preset_name, config)
                    st.success(f"Preset **{preset_name}** saved!")
                except Exception as e:
                    st.error(f"Failed to save: {e}")
            else:
                st.warning("Please enter a preset name")

    # 메인 영역: 이미지 업로드 & 결과
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
        )

        if uploaded_file is not None:
            # 이미지 로드
            image = Image.open(uploaded_file)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_np = np.array(image)

            st.image(image, caption="Original Image", use_container_width=True)

    with col2:
        st.header("🎯 OCR Result")

        if uploaded_file is not None:
            # OCR 실행
            with st.spinner("Running OCR..."):
                ocr = beanOCR(config=config)
                result = ocr.recognize(image_np)

            # 결과 표시
            st.subheader("📝 Recognized Text")
            st.text_area(
                "Text",
                value=result.text,
                height=200,
                disabled=True,
            )

            # 통계
            col2_1, col2_2, col2_3 = st.columns(3)
            with col2_1:
                st.metric("Lines", result.line_count)
            with col2_2:
                st.metric("Confidence", f"{result.confidence:.2%}")
            with col2_3:
                st.metric("Processing Time", f"{result.processing_time:.2f}s")

            # BoundingBox 시각화
            st.subheader("📦 BoundingBox Visualization")
            show_bbox = st.checkbox("Show BoundingBox", value=True)
            show_confidence = st.checkbox("Show Confidence Colors", value=True)

            if show_bbox:
                try:
                    from .visualizer import OCRVisualizer

                    viz = OCRVisualizer()

                    # 시각화 이미지 생성 (메모리에 저장)
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                    ax.imshow(image_np)

                    if result.lines:
                        for line in result.lines:
                            bbox = line.bbox
                            confidence = line.confidence

                            # 신뢰도 기반 색상
                            if show_confidence:
                                color = viz._confidence_to_color(confidence)
                            else:
                                color = "green"

                            # BoundingBox 그리기
                            from matplotlib.patches import Rectangle

                            rect = Rectangle(
                                (bbox.x0, bbox.y0),
                                bbox.width,
                                bbox.height,
                                linewidth=2,
                                edgecolor=color,
                                facecolor="none",
                                alpha=0.8,
                            )
                            ax.add_patch(rect)

                            # 신뢰도 텍스트
                            if show_confidence:
                                ax.text(
                                    bbox.x0,
                                    bbox.y0 - 5,
                                    f"{confidence:.2f}",
                                    color=color,
                                    fontsize=8,
                                    weight="bold",
                                    bbox=dict(
                                        boxstyle="round,pad=0.3",
                                        facecolor="white",
                                        alpha=0.7,
                                    ),
                                )

                    ax.axis("off")
                    plt.tight_layout()

                    # Streamlit에 표시
                    st.pyplot(fig)
                    plt.close()

                except ImportError:
                    st.warning(
                        "matplotlib is required for visualization. "
                        "Install it with: pip install matplotlib"
                    )

    # 하단: 전처리 단계 비교
    if uploaded_file is not None:
        st.header("🔬 Preprocessing Steps")
        show_steps = st.checkbox("Show Preprocessing Pipeline", value=False)

        if show_steps:
            try:
                import cv2

                from .preprocessing import ImagePreprocessor

                preprocessor = ImagePreprocessor()

                # 전처리 단계별 이미지
                steps = []
                titles = []

                # 원본
                steps.append(image_np)
                titles.append("Original")

                # Grayscale
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                current = gray.copy()

                # Denoise
                if config.denoise_config.enabled:
                    current = preprocessor._denoise(current, config.denoise_config)
                    steps.append(cv2.cvtColor(current, cv2.COLOR_GRAY2RGB))
                    titles.append(f"Denoised ({config.denoise_config.strength})")

                # Contrast
                if config.contrast_config.enabled:
                    current = preprocessor._adjust_contrast(current, config.contrast_config)
                    steps.append(cv2.cvtColor(current, cv2.COLOR_GRAY2RGB))
                    titles.append(f"Contrast (CLAHE {config.contrast_config.clip_limit})")

                # Binarize
                if config.binarize_config.enabled:
                    current = preprocessor._binarize(current, config.binarize_config)
                    steps.append(cv2.cvtColor(current, cv2.COLOR_GRAY2RGB))
                    titles.append(f"Binarized ({config.binarize_config.method})")

                # 시각화
                cols = st.columns(min(len(steps), 3))
                for idx, (step_img, title) in enumerate(zip(steps, titles)):
                    with cols[idx % 3]:
                        st.image(step_img, caption=title, use_container_width=True)

            except ImportError:
                st.warning(
                    "opencv-python is required for preprocessing visualization. "
                    "Install it with: pip install opencv-python"
                )


if __name__ == "__main__":
    main()
