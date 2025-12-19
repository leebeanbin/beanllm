"""
ML Models Integration Tutorial
===============================

이 튜토리얼은 llmkit에서 TensorFlow, PyTorch, Scikit-learn 모델을 사용하는 방법을 실습합니다.

Topics:
1. Model Loading - 다양한 프레임워크의 모델 로드
2. TensorFlow Integration - Keras 모델 사용
3. PyTorch Integration - PyTorch 모델 사용
4. Scikit-learn Integration - Classical ML 모델
5. Hybrid LLM + ML - Agent에 ML Tool 추가
6. Model Serving - FastAPI로 모델 배포
"""

import numpy as np
from pathlib import Path

print("="*80)
print("ML Models Integration Tutorial")
print("="*80)


# =============================================================================
# Part 1: Model Loading - 자동 감지 및 로드
# =============================================================================

print("\n" + "="*80)
print("Part 1: Model Loading - 프레임워크 자동 감지")
print("="*80)

"""
Theory:
    MLModelFactory는 파일 확장자와 내용을 분석하여
    자동으로 적절한 프레임워크를 선택합니다.

    Detection logic:
    - .h5, SavedModel dir → TensorFlow
    - .pt, .pth → PyTorch
    - .pkl, .joblib → Scikit-learn
"""


def demo_model_loading():
    """모델 자동 로드 예제"""
    from llmkit import MLModelFactory

    print("\n--- Automatic Framework Detection ---")

    # 시뮬레이션 (실제로는 모델 파일 필요)
    model_paths = {
        "tensorflow": "models/classifier.h5",
        "pytorch": "models/classifier.pth",
        "sklearn": "models/classifier.joblib"
    }

    for framework, path in model_paths.items():
        print(f"\n{framework.upper()}:")
        print(f"  Path: {path}")

        # model = MLModelFactory.load(path)
        # print(f"  ✓ Loaded as {model.__class__.__name__}")

        # 시뮬레이션
        print(f"  ✓ Auto-detected: {framework}")
        print(f"  ✓ Model ready for inference")

    print("\n--- Manual Framework Specification ---")
    # 명시적으로 프레임워크 지정
    # model = MLModelFactory.load("model.bin", framework="tensorflow")
    print("  Can override auto-detection with framework parameter")


# 실행
if __name__ == "__main__":
    # demo_model_loading()
    pass


# =============================================================================
# Part 2: TensorFlow Integration - Keras 모델
# =============================================================================

print("\n" + "="*80)
print("Part 2: TensorFlow - Keras 모델 사용")
print("="*80)

"""
Theory:
    Keras Sequential API:
    y = fₙ ∘ fₙ₋₁ ∘ ... ∘ f₁(x)

    Loss:
    L = -Σ yᵢ log(ŷᵢ)  (Cross-Entropy)
"""


def demo_tensorflow():
    """TensorFlow/Keras 예제"""
    print("\n--- Creating a Simple Keras Model ---")

    # 실제 환경에서는 TensorFlow 설치 필요
    # import tensorflow as tf
    # from llmkit import TensorFlowModel

    # 시뮬레이션
    print("Model Architecture:")
    print("  Input(784) → Dense(128, relu) → Dropout(0.2) → Dense(10, softmax)")

    print("\n--- Training (Simulation) ---")
    # model = tf.keras.Sequential([...])
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    # model.fit(X_train, y_train, epochs=10)

    print("Epoch 1/10: loss=0.5432, accuracy=0.8234")
    print("Epoch 10/10: loss=0.1234, accuracy=0.9567")

    print("\n--- Saving Model ---")
    # model.save('mnist_model.h5')
    print("✓ Saved to mnist_model.h5")

    print("\n--- Loading with llmkit ---")
    # from llmkit import TensorFlowModel
    # ml_model = TensorFlowModel.load('mnist_model.h5')

    # 시뮬레이션
    print("✓ Loaded TensorFlow model")

    print("\n--- Inference ---")
    # predictions = ml_model.predict(test_images)

    # 시뮬레이션
    test_input = np.random.randn(5, 784)
    print(f"Input shape: {test_input.shape}")

    # 가상 예측
    predictions = np.random.rand(5, 10)
    predictions = predictions / predictions.sum(axis=1, keepdims=True)  # softmax

    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample prediction: {predictions[0]}")
    print(f"Predicted class: {predictions[0].argmax()}")

    print("\n--- Batch Inference ---")
    # batch_predictions = ml_model.predict(large_dataset, batch_size=32)
    print("Processing 1000 samples in batches of 32...")
    print("✓ Completed in 0.5s (GPU accelerated)")


# 실행
if __name__ == "__main__":
    # demo_tensorflow()
    pass


# =============================================================================
# Part 3: PyTorch Integration
# =============================================================================

print("\n" + "="*80)
print("Part 3: PyTorch - 동적 그래프와 Autograd")
print("="*80)

"""
Theory:
    PyTorch autograd:
    ∂L/∂θ = ∂L/∂y · ∂y/∂θ  (Chain rule)

    Training loop:
    1. Forward: ŷ = model(x)
    2. Loss: L = criterion(ŷ, y)
    3. Backward: L.backward()
    4. Update: optimizer.step()
"""


def demo_pytorch():
    """PyTorch 예제"""
    print("\n--- PyTorch Model Definition ---")

    # 시뮬레이션 (실제로는 torch 필요)
    model_code = """
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
"""
    print(model_code)

    print("\n--- Training Loop ---")
    training_code = """
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward
        outputs = model(batch['x'])
        loss = criterion(outputs, batch['y'])

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
"""
    print(training_code)

    print("\n--- Saving & Loading ---")
    # torch.save(model.state_dict(), 'model.pth')
    print("✓ Saved state dict to model.pth")

    # from llmkit import PyTorchModel
    # ml_model = PyTorchModel.load('model.pth', model_class=Net)

    print("✓ Loaded PyTorch model")

    print("\n--- GPU Support ---")
    # ml_model = PyTorchModel.load('model.pth', device='cuda')
    print("Device options:")
    print("  - 'cpu': CPU inference")
    print("  - 'cuda': GPU inference (if available)")
    print("  - 'cuda:0': Specific GPU")

    print("\n--- Inference ---")
    # predictions = ml_model.predict(test_data)

    # 시뮬레이션
    test_input = np.random.randn(3, 784)
    predictions = np.random.randn(3, 10)

    print(f"Input: {test_input.shape}")
    print(f"Output: {predictions.shape}")
    print(f"Predicted classes: {predictions.argmax(axis=1)}")


# 실행
if __name__ == "__main__":
    # demo_pytorch()
    pass


# =============================================================================
# Part 4: Scikit-learn Integration - Classical ML
# =============================================================================

print("\n" + "="*80)
print("Part 4: Scikit-learn - Classical ML 알고리즘")
print("="*80)

"""
Theory:
    Random Forest:
    ŷ = (1/T) Σ fₜ(x)  (T개 tree의 평균)

    SVM:
    min (1/2)||w||² s.t. yᵢ(wᵀxᵢ + b) ≥ 1

    Logistic Regression:
    P(y=1|x) = σ(wᵀx + b) = 1/(1 + e^(-wᵀx-b))
"""


def demo_sklearn():
    """Scikit-learn 예제"""
    print("\n--- Scikit-learn Algorithms ---")

    algorithms = {
        "Random Forest": "Ensemble of decision trees",
        "SVM": "Maximum margin classifier",
        "Logistic Regression": "Linear probability model",
        "K-Nearest Neighbors": "Instance-based learning",
        "Gradient Boosting": "Sequential ensemble"
    }

    for name, desc in algorithms.items():
        print(f"  - {name}: {desc}")

    print("\n--- Training Example ---")

    # 실제 코드 (주석 처리)
    training_example = """
from sklearn.ensemble import RandomForestClassifier
from llmkit import SklearnModel

# Train
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Save
import joblib
joblib.dump(clf, 'rf_model.joblib')

# Load with llmkit
ml_model = SklearnModel.load('rf_model.joblib')
predictions = ml_model.predict(X_test)
"""
    print(training_example)

    print("\n--- Pipeline Example ---")
    pipeline_example = """
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50)),
    ('classifier', RandomForestClassifier())
])

pipe.fit(X_train, y_train)
joblib.dump(pipe, 'pipeline.joblib')

# llmkit에서도 pipeline 로드 가능
ml_model = SklearnModel.load('pipeline.joblib')
"""
    print(pipeline_example)

    print("\n--- Feature Importance ---")
    # feature_importance = ml_model.get_feature_importance()

    # 시뮬레이션
    features = ['age', 'income', 'credit_score', 'debt', 'employment']
    importance = np.random.rand(5)
    importance = importance / importance.sum()

    print("Top features:")
    for feat, imp in sorted(zip(features, importance), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {imp:.3f}")


# 실행
if __name__ == "__main__":
    # demo_sklearn()
    pass


# =============================================================================
# Part 5: Hybrid LLM + ML - Agent에 ML Tool 추가
# =============================================================================

print("\n" + "="*80)
print("Part 5: Hybrid System - LLM + ML 통합")
print("="*80)

"""
Theory:
    Hybrid Architecture:

    Pattern 1: ML as Tool
    Agent(LLM) → calls → ML Model

    Pattern 2: Feature Engineering
    LLM(text → embedding) → ML(embedding → prediction)

    Pattern 3: Ensemble
    y = α·y_LLM + (1-α)·y_ML
"""


async def demo_hybrid_system():
    """Hybrid LLM + ML 시스템"""
    from llmkit import Agent, Tool

    print("\n--- Pattern 1: ML as Agent Tool ---")

    # ML 모델을 Tool로 래핑
    ml_tool_code = """
from llmkit import Tool, SklearnModel

# ML 모델 로드
fraud_detector = SklearnModel.load('fraud_model.joblib')

# Tool로 래핑
def detect_fraud(transaction_amount: float, user_age: int, location: str) -> str:
    '''Detect if a transaction is fraudulent'''

    # 전처리
    features = preprocess(transaction_amount, user_age, location)

    # 예측
    is_fraud = fraud_detector.predict([features])[0]
    prob = fraud_detector.predict_proba([features])[0]

    return f"Fraud: {is_fraud}, Probability: {prob[1]:.2%}"

fraud_tool = Tool.from_function(detect_fraud)

# Agent에 추가
agent = Agent(
    model="gpt-4o",
    tools=[fraud_tool]
)

# 사용
result = await agent.run(
    "Check if this transaction is fraudulent: $5000 from a 25-year-old in Nigeria"
)
"""
    print(ml_tool_code)

    print("\n--- Pattern 2: LLM Embeddings + ML Classifier ---")

    embedding_ml_code = """
from llmkit import Embedding, SklearnModel

# 1. LLM으로 텍스트 임베딩
embed = Embedding(model="text-embedding-3-small")
text_embeddings = embed.embed_sync(texts)

# 2. ML 모델로 분류
classifier = SklearnModel.load('text_classifier.joblib')
predictions = classifier.predict(text_embeddings)
"""
    print(embedding_ml_code)

    print("\n--- Pattern 3: Ensemble (LLM + ML) ---")

    ensemble_code = """
# LLM 예측
llm_prediction = await llm.classify(text)
llm_confidence = 0.8

# ML 예측
ml_prediction = ml_model.predict([features])[0]
ml_confidence = ml_model.predict_proba([features])[0].max()

# Ensemble
if llm_confidence > 0.9:
    final = llm_prediction  # High confidence LLM
elif ml_confidence > 0.9:
    final = ml_prediction   # High confidence ML
else:
    # Weighted average
    alpha = llm_confidence / (llm_confidence + ml_confidence)
    final = alpha * llm_prediction + (1-alpha) * ml_prediction
"""
    print(ensemble_code)

    print("\n--- Use Case: Sentiment Analysis ---")

    use_case = """
Scenario: Analyze customer reviews

1. Fast Filter (ML):
   - Scikit-learn classifier on TF-IDF features
   - Process 10,000 reviews in 1 second
   - Flag ambiguous cases (confidence < 0.7)

2. Deep Analysis (LLM):
   - GPT-4 analyzes flagged reviews
   - Understand sarcasm, context
   - Generate detailed insights

Result:
- 90% handled by fast ML (cheap)
- 10% by LLM (accurate)
- Best of both worlds!
"""
    print(use_case)


# 실행
if __name__ == "__main__":
    # asyncio.run(demo_hybrid_system())
    pass


# =============================================================================
# Part 6: Model Serving - FastAPI로 배포
# =============================================================================

print("\n" + "="*80)
print("Part 6: Model Serving - Production 배포")
print("="*80)

"""
Theory:
    REST API:
    POST /predict → JSON response

    Latency components:
    T_total = T_network + T_preprocessing + T_inference + T_postprocessing

    Throughput:
    QPS = 1 / T_total  (Queries Per Second)
"""


def demo_model_serving():
    """모델 서빙 예제"""
    print("\n--- FastAPI Server ---")

    fastapi_code = """
from fastapi import FastAPI
from llmkit import TensorFlowModel
from pydantic import BaseModel

app = FastAPI()

# 모델 로드 (startup 시 1번만)
model = TensorFlowModel.load('model.h5')

class PredictionRequest(BaseModel):
    features: list[float]

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Inference
    prediction = model.predict([request.features])

    return {
        "prediction": prediction[0].tolist(),
        "class": int(prediction[0].argmax())
    }

# Run: uvicorn server:app --host 0.0.0.0 --port 8000
"""
    print(fastapi_code)

    print("\n--- Client Usage ---")

    client_code = """
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [1.2, 3.4, 5.6, ...]}
)

result = response.json()
print(f"Prediction: {result['class']}")
"""
    print(client_code)

    print("\n--- Batch Endpoint ---")

    batch_code = """
@app.post("/predict_batch")
async def predict_batch(requests: list[PredictionRequest]):
    features = [req.features for req in requests]

    # Batch inference (더 효율적)
    predictions = model.predict(features, batch_size=32)

    return [
        {"prediction": pred.tolist(), "class": int(pred.argmax())}
        for pred in predictions
    ]
"""
    print(batch_code)

    print("\n--- Performance Optimization ---")

    optimizations = {
        "Batching": "Process multiple requests together → 10x throughput",
        "Caching": "Cache common inputs → 100x faster",
        "Quantization": "int8 instead of float32 → 4x smaller, 2x faster",
        "GPU": "Use CUDA → 50x faster for large models",
        "Model Compression": "Pruning, distillation → 10x smaller"
    }

    for technique, benefit in optimizations.items():
        print(f"  {technique}: {benefit}")


# 실행
if __name__ == "__main__":
    # demo_model_serving()
    pass


# =============================================================================
# Part 7: Advanced Integration - 실전 시나리오
# =============================================================================

print("\n" + "="*80)
print("Part 7: Advanced Integration - 복합 시스템")
print("="*80)


async def demo_advanced_integration():
    """고급 통합 예제"""
    print("\n--- Scenario: E-commerce Recommendation ---")

    scenario = """
System Components:
1. Collaborative Filtering (Scikit-learn)
   - User-item matrix factorization
   - Fast recommendations from past behavior

2. Content-Based (TensorFlow)
   - Product embeddings
   - Similar item recommendations

3. LLM Explainer (GPT-4)
   - Generate natural language explanations
   - Answer "why this recommendation?"

4. Anomaly Detector (PyTorch)
   - Detect unusual purchase patterns
   - Fraud prevention
"""
    print(scenario)

    print("\n--- Implementation ---")

    impl_code = """
from llmkit import Agent, Tool, SklearnModel, TensorFlowModel, PyTorchModel

# Load ML models
cf_model = SklearnModel.load('collaborative_filtering.joblib')
content_model = TensorFlowModel.load('content_model.h5')
anomaly_model = PyTorchModel.load('anomaly.pth', device='cuda')

# Create tools
def get_recommendations(user_id: int, k: int = 10) -> list:
    '''Get product recommendations for user'''

    # Stage 1: Collaborative filtering (fast, broad)
    cf_recs = cf_model.predict([[user_id]])[:50]

    # Stage 2: Content-based refinement
    user_embedding = get_user_embedding(user_id)
    product_embeddings = get_product_embeddings(cf_recs)
    scores = content_model.predict([user_embedding, product_embeddings])

    # Top K
    top_k = scores.argsort()[-k:][::-1]
    return [cf_recs[i] for i in top_k]

def check_fraud(user_id: int, product_id: int, amount: float) -> dict:
    '''Check if purchase seems fraudulent'''
    features = extract_features(user_id, product_id, amount)
    is_anomaly = anomaly_model.predict([features])[0]
    score = anomaly_model.predict_proba([features])[0]
    return {"is_fraud": bool(is_anomaly), "score": float(score)}

# Agent with ML tools
agent = Agent(
    model="gpt-4o",
    tools=[
        Tool.from_function(get_recommendations),
        Tool.from_function(check_fraud)
    ]
)

# User query
result = await agent.run(
    "Recommend products for user 12345 and check if buying a $5000 watch is suspicious"
)
"""
    print(impl_code)

    print("\n--- Performance Metrics ---")

    metrics = {
        "Latency": "150ms (CF: 10ms, Content: 20ms, Anomaly: 5ms, LLM: 500ms in parallel)",
        "Throughput": "100 QPS",
        "Accuracy": "CF: 0.75 NDCG, Content: 0.82 Precision@10",
        "Cost": "$0.01 per request (mostly LLM)"
    }

    for metric, value in metrics.items():
        print(f"  {metric}: {value}")


# =============================================================================
# Part 8: Performance Comparison
# =============================================================================

print("\n" + "="*80)
print("Part 8: Performance Benchmarks")
print("="*80)


def demo_performance():
    """성능 비교"""
    print("\n--- Inference Speed Comparison ---")

    inference_speeds = {
        "Scikit-learn (CPU)": "0.1ms - 1ms",
        "TensorFlow (CPU)": "1ms - 10ms",
        "TensorFlow (GPU)": "0.5ms - 5ms",
        "PyTorch (CPU)": "1ms - 10ms",
        "PyTorch (GPU)": "0.5ms - 5ms",
        "LLM (GPT-4)": "500ms - 2000ms"
    }

    for model, speed in inference_speeds.items():
        print(f"  {model}: {speed}")

    print("\n--- Model Size Comparison ---")

    sizes = {
        "Scikit-learn RF": "10KB - 10MB",
        "Small CNN": "1MB - 10MB",
        "ResNet-50": "100MB",
        "BERT-base": "400MB",
        "GPT-3": "350GB (API only)"
    }

    for model, size in sizes.items():
        print(f"  {model}: {size}")

    print("\n--- Cost Analysis ---")

    costs = {
        "Scikit-learn": "$0 (local)",
        "TensorFlow": "$0 (local) or $0.0001/req (cloud)",
        "PyTorch": "$0 (local) or $0.0001/req (cloud)",
        "GPT-4": "$0.03/1K tokens"
    }

    for service, cost in costs.items():
        print(f"  {service}: {cost}")

    print("\n--- When to Use Each ---")

    use_cases = {
        "Scikit-learn": "Tabular data, prototyping, interpretability",
        "TensorFlow": "Production deployment, mobile, edge devices",
        "PyTorch": "Research, complex architectures, rapid iteration",
        "LLM": "Language understanding, reasoning, few-shot learning"
    }

    for framework, use_case in use_cases.items():
        print(f"  {framework}: {use_case}")


# 실행
if __name__ == "__main__":
    # demo_performance()
    pass


# =============================================================================
# 전체 실행
# =============================================================================

async def run_all_demos():
    """모든 데모 실행"""
    import asyncio

    demos = [
        ("Model Loading", demo_model_loading, False),
        ("TensorFlow", demo_tensorflow, False),
        ("PyTorch", demo_pytorch, False),
        ("Scikit-learn", demo_sklearn, False),
        ("Hybrid System", demo_hybrid_system, True),
        ("Model Serving", demo_model_serving, False),
        ("Advanced Integration", demo_advanced_integration, True),
        ("Performance", demo_performance, False),
    ]

    for name, demo, is_async in demos:
        print("\n" + "="*80)
        print(f"Running: {name}")
        print("="*80)
        try:
            if is_async:
                await demo()
            else:
                demo()
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(0.5)


if __name__ == "__main__":
    import asyncio

    print("""
이 튜토리얼을 실행하려면:

1. 필요한 패키지 설치 (선택적):
   pip install tensorflow  # TensorFlow 예제용
   pip install torch       # PyTorch 예제용
   pip install scikit-learn  # Scikit-learn 예제용

2. llmkit 설치:
   pip install -e .

3. 실행:
   python docs/tutorials/05_ml_models_tutorial.py

주의: 실제 모델 파일이 필요합니다. 이 튜토리얼은 시뮬레이션 코드를 포함합니다.
    """)

    # 전체 실행
    # asyncio.run(run_all_demos())

    # 개별 실행 예시:
    demo_model_loading()
    # demo_tensorflow()
    # demo_pytorch()
    # demo_sklearn()
    # asyncio.run(demo_hybrid_system())


"""
연습 문제:

1. Model Comparison
   - 같은 데이터셋에 Scikit-learn, TensorFlow, PyTorch 모델 학습
   - 정확도, 속도, 메모리 사용량 비교
   - Trade-off 분석

2. Hybrid System
   - ML 모델을 Agent tool로 통합
   - LLM이 ML 결과를 해석하도록 구현
   - Cost vs Accuracy 분석

3. Model Serving
   - FastAPI로 모델 서빙 API 구축
   - Load testing (locust, ab)
   - Latency 최적화 (caching, batching)

4. Quantization
   - float32 모델을 int8로 quantize
   - 정확도 변화 측정
   - 속도 향상 측정

5. Ensemble Methods
   - 여러 ML 모델의 ensemble
   - Voting, averaging, stacking
   - Ensemble vs single model 성능 비교

6. Transfer Learning
   - Pre-trained 모델 fine-tuning
   - Feature extraction
   - Domain adaptation
"""
