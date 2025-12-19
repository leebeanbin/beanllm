"""
Embeddings Tutorial - 임베딩 실습
이론을 코드로 구현하며 배우기

이 튜토리얼은 docs/theory/01_embeddings_theory.md의 내용을
실제로 구현하고 시각화합니다.

실행: python 01_embeddings_tutorial.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. 벡터 공간과 기본 연산
# ============================================================================

def section_1():
    """벡터 공간 기초"""
    print("="*70)
    print("Section 1: 벡터 공간과 기본 연산")
    print("="*70)

    # 1.1 벡터 정의
    print("\n[1.1] 벡터 정의")
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])

    print(f"v1 = {v1}")
    print(f"v2 = {v2}")

    # 1.2 벡터 연산
    print("\n[1.2] 벡터 연산")

    # 덧셈
    v_sum = v1 + v2
    print(f"v1 + v2 = {v_sum}")

    # 스칼라 곱
    v_scaled = 2 * v1
    print(f"2 * v1 = {v_scaled}")

    # 내적
    dot_product = np.dot(v1, v2)
    print(f"v1 · v2 = {dot_product}")

    # 노름 (크기)
    norm_v1 = np.linalg.norm(v1)
    print(f"||v1|| = {norm_v1:.4f}")


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """
    코사인 유사도 계산

    수식: cos(θ) = (u·v) / (||u|| ||v||)

    Args:
        u: 벡터 1
        v: 벡터 2

    Returns:
        코사인 유사도 [-1, 1]
    """
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u == 0 or norm_v == 0:
        return 0.0

    return dot_product / (norm_u * norm_v)


def section_2():
    """코사인 유사도"""
    print("\n" + "="*70)
    print("Section 2: 코사인 유사도")
    print("="*70)

    # 2.1 기본 예제
    print("\n[2.1] 기본 예제")

    u = np.array([1, 0, 0])
    v = np.array([1, 1, 0])

    similarity = cosine_similarity(u, v)
    print(f"u = {u}")
    print(f"v = {v}")
    print(f"cosine_similarity(u, v) = {similarity:.4f}")

    # 각도 계산
    angle_rad = np.arccos(np.clip(similarity, -1, 1))
    angle_deg = np.degrees(angle_rad)
    print(f"각도 = {angle_deg:.2f}도")

    # 2.2 다양한 경우
    print("\n[2.2] 다양한 경우")

    test_cases = [
        ([1, 0], [1, 0], "Same direction"),
        ([1, 0], [-1, 0], "Opposite direction"),
        ([1, 0], [0, 1], "Orthogonal"),
        ([1, 1], [2, 2], "Same direction, different magnitude"),
    ]

    for u, v, desc in test_cases:
        u = np.array(u)
        v = np.array(v)
        sim = cosine_similarity(u, v)
        print(f"{desc:40s} → {sim:7.4f}")

    # 2.3 시각화
    print("\n[2.3] 2D 시각화")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Cosine Similarity Visualization', fontsize=16)

    angles = [0, 45, 90, 135]

    for idx, (ax, angle) in enumerate(zip(axes.flat, angles)):
        # 벡터 정의
        u = np.array([1, 0])
        v = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])

        # 코사인 유사도 계산
        sim = cosine_similarity(u, v)

        # 그리기
        ax.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1,
                  color='blue', width=0.01, label='u')
        ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                  color='red', width=0.01, label='v')

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)

        ax.set_title(f'Angle: {angle}°, Cosine: {sim:.4f}')
        ax.legend()

    plt.tight_layout()
    plt.savefig('cosine_similarity_visualization.png', dpi=150, bbox_inches='tight')
    print("  → 저장됨: cosine_similarity_visualization.png")
    plt.close()


# ============================================================================
# 3. Softmax와 Temperature
# ============================================================================

def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Temperature를 적용한 Softmax

    수식: softmax(x/τ)ᵢ = exp(xᵢ/τ) / Σⱼ exp(xⱼ/τ)

    Args:
        x: 로짓 벡터
        temperature: Temperature parameter

    Returns:
        확률 분포
    """
    x_scaled = x / temperature

    # 수치 안정성을 위해 최대값 빼기
    x_shifted = x_scaled - np.max(x_scaled)

    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


def section_3():
    """Softmax와 Temperature"""
    print("\n" + "="*70)
    print("Section 3: Softmax와 Temperature")
    print("="*70)

    # 3.1 기본 Softmax
    print("\n[3.1] 기본 Softmax")

    logits = np.array([2.0, 1.0, 0.1])
    probs = softmax(logits)

    print(f"Logits: {logits}")
    print(f"Softmax: {probs}")
    print(f"Sum: {np.sum(probs):.6f}")

    # 3.2 Temperature 효과
    print("\n[3.2] Temperature 효과")

    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

    print("\nTemperature | Probabilities")
    print("-" * 50)

    for tau in temperatures:
        probs = softmax(logits, tau)
        probs_str = ", ".join([f"{p:.4f}" for p in probs])
        print(f"τ = {tau:4.1f}   | [{probs_str}]")

    # 3.3 시각화
    print("\n[3.3] Temperature 시각화")

    fig, axes = plt.subplots(1, len(temperatures), figsize=(15, 3))
    fig.suptitle('Softmax Temperature Effect', fontsize=16)

    for ax, tau in zip(axes, temperatures):
        probs = softmax(logits, tau)

        bars = ax.bar(range(len(probs)), probs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_title(f'τ = {tau}')
        ax.set_ylim([0, 1])
        ax.set_ylabel('Probability')
        ax.set_xticks(range(len(probs)))
        ax.set_xticklabels([f'Class {i}' for i in range(len(probs))])

        # 값 표시
        for i, (bar, p) in enumerate(zip(bars, probs)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{p:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('softmax_temperature.png', dpi=150, bbox_inches='tight')
    print("  → 저장됨: softmax_temperature.png")
    plt.close()

    # 3.4 수학적 특성
    print("\n[3.4] 수학적 특성")

    print("\nTemperature → 0 (Sharp):")
    probs_low = softmax(logits, 0.01)
    print(f"  Probabilities: {probs_low}")
    print(f"  Max prob: {np.max(probs_low):.6f}")

    print("\nTemperature → ∞ (Uniform):")
    probs_high = softmax(logits, 100)
    print(f"  Probabilities: {probs_high}")
    print(f"  Max prob: {np.max(probs_high):.6f}")


# ============================================================================
# 4. Word Embeddings 시뮬레이션
# ============================================================================

def create_toy_embeddings() -> Tuple[dict, np.ndarray]:
    """
    간단한 toy embeddings 생성

    단어의 의미적 관계를 반영하는 2D 임베딩
    """
    words = {
        # 왕족
        'king': [0.9, 0.1],
        'queen': [0.9, 0.9],
        'prince': [0.7, 0.1],
        'princess': [0.7, 0.9],

        # 일반인
        'man': [0.1, 0.1],
        'woman': [0.1, 0.9],
        'boy': [0.3, 0.1],
        'girl': [0.3, 0.9],
    }

    embeddings = {word: np.array(vec) for word, vec in words.items()}
    return embeddings


def section_4():
    """Word Embeddings"""
    print("\n" + "="*70)
    print("Section 4: Word Embeddings")
    print("="*70)

    # 4.1 Toy embeddings 생성
    print("\n[4.1] Toy Embeddings")

    embeddings = create_toy_embeddings()

    for word, vec in embeddings.items():
        print(f"{word:10s}: {vec}")

    # 4.2 유사도 계산
    print("\n[4.2] 단어 간 유사도")

    word_pairs = [
        ('king', 'queen'),
        ('king', 'man'),
        ('queen', 'woman'),
        ('king', 'woman'),
    ]

    for w1, w2 in word_pairs:
        sim = cosine_similarity(embeddings[w1], embeddings[w2])
        print(f"  sim('{w1}', '{w2}') = {sim:.4f}")

    # 4.3 벡터 연산
    print("\n[4.3] 벡터 연산 (Analogy)")

    print("\n수식: king - man + woman ≈ queen")

    result = embeddings['king'] - embeddings['man'] + embeddings['woman']
    print(f"\nking - man + woman = {result}")

    # 가장 가까운 단어 찾기
    best_word = None
    best_sim = -1

    for word, vec in embeddings.items():
        if word in ['king', 'man', 'woman']:  # 연산에 사용된 단어 제외
            continue

        sim = cosine_similarity(result, vec)
        if sim > best_sim:
            best_sim = sim
            best_word = word

    print(f"가장 가까운 단어: '{best_word}' (유사도: {best_sim:.4f})")

    # 4.4 시각화
    print("\n[4.4] Embedding Space 시각화")

    plt.figure(figsize=(10, 8))

    for word, vec in embeddings.items():
        plt.scatter(vec[0], vec[1], s=200, alpha=0.6)
        plt.annotate(word, (vec[0], vec[1]), fontsize=12,
                     ha='center', va='bottom')

    # 벡터 연산 시각화
    plt.arrow(embeddings['king'][0], embeddings['king'][1],
              -embeddings['man'][0], -embeddings['man'][1],
              head_width=0.05, head_length=0.05, fc='red', ec='red',
              alpha=0.5, linestyle='--', label='king - man')

    plt.arrow(embeddings['king'][0] - embeddings['man'][0],
              embeddings['king'][1] - embeddings['man'][1],
              embeddings['woman'][0], embeddings['woman'][1],
              head_width=0.05, head_length=0.05, fc='blue', ec='blue',
              alpha=0.5, linestyle='--', label='+ woman')

    plt.scatter(result[0], result[1], s=300, marker='*',
                c='green', edgecolors='black', linewidths=2,
                label='Result', zorder=5)

    plt.xlabel('Dimension 1 (Royalty)', fontsize=12)
    plt.ylabel('Dimension 2 (Gender: Female)', fontsize=12)
    plt.title('Word Embeddings - Analogy Visualization', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)

    plt.tight_layout()
    plt.savefig('word_embeddings_analogy.png', dpi=150, bbox_inches='tight')
    print("  → 저장됨: word_embeddings_analogy.png")
    plt.close()


# ============================================================================
# 5. Attention Mechanism
# ============================================================================

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled Dot-Product Attention

    수식: Attention(Q,K,V) = softmax(QKᵀ / √dₖ) V

    Args:
        Q: Query matrix [seq_len, d_k]
        K: Key matrix [seq_len, d_k]
        V: Value matrix [seq_len, d_v]

    Returns:
        (output, attention_weights)
    """
    d_k = K.shape[-1]

    # QKᵀ / √dₖ
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)

    # Softmax
    attention_weights = softmax(scores)

    # Attention weights × V
    output = np.matmul(attention_weights, V)

    return output, attention_weights


def section_5():
    """Attention Mechanism"""
    print("\n" + "="*70)
    print("Section 5: Attention Mechanism")
    print("="*70)

    # 5.1 간단한 예제
    print("\n[5.1] 간단한 Attention 예제")

    # 시퀀스: "I love AI"
    # 3개 토큰, 4차원 임베딩
    seq_len, d_model = 3, 4

    # Query, Key, Value (임의로 생성)
    np.random.seed(42)
    Q = np.random.randn(seq_len, d_model)
    K = np.random.randn(seq_len, d_model)
    V = np.random.randn(seq_len, d_model)

    print(f"Query shape: {Q.shape}")
    print(f"Key shape: {K.shape}")
    print(f"Value shape: {V.shape}")

    # Attention 계산
    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    print("\nAttention Weights:")
    print(attention_weights)

    # 5.2 시각화
    print("\n[5.2] Attention Weights 시각화")

    tokens = ["I", "love", "AI"]

    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Attention Weight')

    plt.xticks(range(len(tokens)), tokens)
    plt.yticks(range(len(tokens)), tokens)
    plt.xlabel('Key/Value')
    plt.ylabel('Query')
    plt.title('Self-Attention Weights')

    # 값 표시
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            plt.text(j, i, f'{attention_weights[i, j]:.2f}',
                     ha='center', va='center', color='white' if attention_weights[i, j] > 0.5 else 'black')

    plt.tight_layout()
    plt.savefig('attention_weights.png', dpi=150, bbox_inches='tight')
    print("  → 저장됨: attention_weights.png")
    plt.close()


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """모든 섹션 실행"""
    print("\n" + "="*70)
    print("Embeddings Tutorial - 임베딩 실습")
    print("="*70)
    print("\nThis tutorial demonstrates:")
    print("  1. Vector Space Operations")
    print("  2. Cosine Similarity")
    print("  3. Softmax & Temperature")
    print("  4. Word Embeddings & Analogies")
    print("  5. Attention Mechanism")
    print()

    section_1()
    section_2()
    section_3()
    section_4()
    section_5()

    print("\n" + "="*70)
    print("Tutorial Complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - cosine_similarity_visualization.png")
    print("  - softmax_temperature.png")
    print("  - word_embeddings_analogy.png")
    print("  - attention_weights.png")
    print("\n이론 문서: docs/theory/01_embeddings_theory.md")
    print("="*70)


if __name__ == "__main__":
    main()
