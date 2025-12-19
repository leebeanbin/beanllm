# ML Models Theory: 머신러닝 모델 통합의 수학적 기초

**석사 수준 이론 문서**  
**기반**: llmkit MLModels 실제 구현 분석

---

## 목차

### Part I: 모델 인터페이스 이론
1. [통합 인터페이스의 수학적 정의](#part-i-모델-인터페이스-이론)
2. [예측 함수의 형식적 정의](#12-예측-함수의-형식적-정의)
3. [모델 로딩과 직렬화](#13-모델-로딩과-직렬화)

### Part II: 프레임워크별 구현
4. [TensorFlow/Keras 모델](#part-ii-프레임워크별-구현)
5. [PyTorch 모델](#42-pytorch-모델)
6. [Scikit-learn 모델](#43-scikit-learn-모델)

---

## Part I: 모델 인터페이스 이론

### 1.1 통합 인터페이스의 수학적 정의

#### 정의 1.1.1: 예측 함수 (Prediction Function)

**예측 함수**는 다음과 같이 정의됩니다:

$$
\hat{y} = f(x; \theta)
$$

여기서:
- $x$: 입력 데이터
- $\theta$: 모델 파라미터
- $\hat{y}$: 예측 결과

#### 시각적 표현: 모델 예측 과정

```
┌─────────────────────────────────────────────────────────┐
│                  모델 예측 과정                           │
└─────────────────────────────────────────────────────────┘

입력 데이터 x
    │
    │ x = [1.2, 3.4, 5.6, ...]  (특징 벡터)
    │
    ▼
┌─────────────────┐
│   모델 M(θ)     │
│                 │
│  ┌───────────┐  │
│  │ Layer 1   │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ Layer 2   │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ Output    │  │
│  └─────┬─────┘  │
└─────────┼────────┘
          │
          ▼
    예측 결과 ŷ
    ŷ = [0.85, 0.12, 0.03]  (확률 분포)
```

#### 구체적 수치 예시

**예시 1.1.1: 이미지 분류 모델**

**입력:**
- 이미지: $x \in \mathbb{R}^{224 \times 224 \times 3}$ (RGB)
- 모델: ResNet50 (사전 학습)

**처리 과정:**

1. **전처리:**
   $$
   x \rightarrow \text{normalize} \rightarrow x' \in [0, 1]^{224 \times 224 \times 3}
   $$

2. **모델 통과:**
   $$
   x' \rightarrow \text{ResNet50} \rightarrow \text{features} \in \mathbb{R}^{1000}
   $$

3. **Softmax:**
   $$
   \hat{y} = \text{softmax}(\text{features}) = [0.85, 0.12, 0.03, \ldots]
   $$

4. **예측:**
   $$
   \text{class} = \arg\max(\hat{y}) = 0 \text{ (고양이)}
   $$

**llmkit 구현:**
```python
# ml_models.py: BaseMLModel
class BaseMLModel(ABC):
    """
    통합 인터페이스: f(x; θ) = ŷ
    """
    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """
        예측 함수: ŷ = f(x; θ)
        """
        pass
```

**실제 사용 예시:**
```python
from llmkit.ml_models import TensorFlowModel
import numpy as np

# 모델 로드
model = TensorFlowModel("resnet50.h5")

# 입력 준비
image = np.random.rand(1, 224, 224, 3)  # 배치 크기 1

# 예측
predictions = model.predict(image)
# 출력: [[0.85, 0.12, 0.03, ...]]  (1000개 클래스 확률)

# 최고 확률 클래스
predicted_class = np.argmax(predictions[0])
print(f"예측 클래스: {predicted_class}")  # 예: 0 (고양이)
```

---

### 1.2 예측 함수의 형식적 정의

#### 정의 1.2.1: 모델 타입

**모델 타입**은 입력과 출력의 타입으로 정의됩니다:

$$
M: X \rightarrow Y
$$

**llmkit 구현:**
```python
# ml_models.py: 각 프레임워크별 구현
class TensorFlowModel(BaseMLModel):
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        TensorFlow: X (numpy array) → Y (numpy array)
        """
        return self.model.predict(inputs)

class PyTorchModel(BaseMLModel):
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        PyTorch: X (Tensor) → Y (Tensor)
        """
        with torch.no_grad():
            return self.model(inputs)

class SklearnModel(BaseMLModel):
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Scikit-learn: X (numpy array) → Y (numpy array)
        """
        return self.model.predict(inputs)
```

---

### 1.3 모델 로딩과 직렬화

#### 정의 1.3.1: 모델 직렬화

**모델 직렬화**는 파라미터를 저장하는 과정입니다:

$$
\text{serialize}: \Theta \rightarrow \text{File}
$$

**역직렬화:**

$$
\text{deserialize}: \text{File} \rightarrow \Theta
$$

**llmkit 구현:**
```python
# ml_models.py: BaseMLModel
class BaseMLModel(ABC):
    @abstractmethod
    def load(self, model_path: Union[str, Path]):
        """
        역직렬화: File → Θ
        """
        pass
    
    @abstractmethod
    def save(self, save_path: Union[str, Path]):
        """
        직렬화: Θ → File
        """
        pass
```

---

## Part II: 프레임워크별 구현

### 2.1 TensorFlow/Keras 모델

#### 정의 2.1.1: Keras 모델

**Keras 모델**은 다음과 같이 정의됩니다:

$$
M_{\text{keras}} = (L_1, L_2, \ldots, L_n)
$$

여기서 $L_i$는 레이어입니다.

**llmkit 구현:**
```python
# ml_models.py: TensorFlowModel
class TensorFlowModel(BaseMLModel):
    def load(self, model_path: Union[str, Path]):
        """
        Keras 모델 로드: M = load_model(path)
        """
        import tensorflow as tf
        self.model = tf.keras.models.load_model(str(model_path))
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        예측: ŷ = M(x)
        """
        return self.model.predict(inputs)
```

---

### 2.2 PyTorch 모델

#### 정의 2.2.1: PyTorch 모델

**PyTorch 모델**은 `nn.Module`을 상속받습니다:

$$
M_{\text{pytorch}} = \text{nn.Module}(\theta)
$$

**llmkit 구현:**
```python
# ml_models.py: PyTorchModel
class PyTorchModel(BaseMLModel):
    def load(self, model_path: Union[str, Path]):
        """
        PyTorch 모델 로드: M = torch.load(path)
        """
        import torch
        self.model = torch.load(str(model_path))
        self.model.eval()  # 평가 모드
    
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        예측: ŷ = M(x)
        """
        with torch.no_grad():
            return self.model(inputs)
```

---

### 2.3 Scikit-learn 모델

#### 정의 2.3.1: Scikit-learn 모델

**Scikit-learn 모델**은 `fit`과 `predict` 메서드를 가집니다:

$$
M_{\text{sklearn}} = \text{fit}(X, y)
$$

$$
\hat{y} = M.\text{predict}(X)
$$

**llmkit 구현:**
```python
# ml_models.py: SklearnModel
class SklearnModel(BaseMLModel):
    def load(self, model_path: Union[str, Path]):
        """
        Scikit-learn 모델 로드: M = joblib.load(path)
        """
        import joblib
        self.model = joblib.load(str(model_path))
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        예측: ŷ = M.predict(X)
        """
        return self.model.predict(inputs)
```

---

## 참고 문헌

1. **Abadi et al. (2016)**: "TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems"
2. **Paszke et al. (2019)**: "PyTorch: An Imperative Style, High-Performance Deep Learning Library"
3. **Pedregosa et al. (2011)**: "Scikit-learn: Machine Learning in Python"

---

**작성일**: 2025-01-XX  
**버전**: 2.0 (석사 수준 확장)
