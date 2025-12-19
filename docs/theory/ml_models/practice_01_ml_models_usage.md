# ML Models 실무 가이드: 통합 모델 활용

**실무 적용 문서**

---

## 목차

1. [모델 로딩](#1-모델-로딩)
2. [예측 실행](#2-예측-실행)
3. [프레임워크별 사용](#3-프레임워크별-사용)

---

## 1. 모델 로딩

```python
from llmkit.ml_models import MLModel

# TensorFlow
model = MLModel.from_tensorflow("model.h5")

# PyTorch
model = MLModel.from_pytorch("model.pth")

# Scikit-learn
model = MLModel.from_sklearn("model.pkl")
```

---

## 2. 예측 실행

```python
# 예측
predictions = model.predict(input_data)

# 배치 예측
batch_predictions = model.predict_batch(batch_data)
```

---

## 3. 프레임워크별 사용

### 3.1 TensorFlow

```python
import tensorflow as tf

model = tf.keras.models.load_model("model.h5")
predictions = model.predict(data)
```

### 3.2 PyTorch

```python
import torch

model = torch.load("model.pth")
model.eval()
with torch.no_grad():
    predictions = model(data)
```

---

**작성일**: 2025-01-XX  
**버전**: 1.0

