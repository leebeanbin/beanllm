# Fourier Transform and STFT: 푸리에 변환과 단시간 푸리에 변환

**석사 수준 이론 문서**  
**CS 관점 + 수학적 엄밀성**  
**기반**: 음성 처리 이론

---

## 목차

1. [푸리에 변환의 수학적 정의](#1-푸리에-변환의-수학적-정의)
2. [이산 푸리에 변환 (DFT)](#2-이산-푸리에-변환-dft)
3. [단시간 푸리에 변환 (STFT)](#3-단시간-푸리에-변환-stft)
4. [스펙트로그램](#4-스펙트로그램)
5. [CS 관점: FFT 알고리즘](#5-cs-관점-fft-알고리즘)

---

## 1. 푸리에 변환의 수학적 정의

### 1.1 연속 푸리에 변환

#### 정의 1.1.1: Fourier Transform

**연속 푸리에 변환:**

$$
F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} dt
$$

**역변환:**

$$
f(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} F(\omega) e^{i\omega t} d\omega
$$

### 1.2 이산 푸리에 변환

#### 정의 1.2.1: DFT

**이산 푸리에 변환 (DFT):**

$$
X[k] = \sum_{n=0}^{N-1} x[n] e^{-i2\pi kn/N}
$$

**역변환 (IDFT):**

$$
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{i2\pi kn/N}
$$

---

## 2. 이산 푸리에 변환 (DFT)

### 2.1 DFT의 기하학적 해석

#### 정리 2.1.1: DFT의 의미

**DFT는 신호를 주파수 성분으로 분해합니다:**

$$
x[n] = \sum_{k=0}^{N-1} X[k] \cdot e^{i2\pi kn/N}
$$

**해석:**
- $X[k]$: $k$번째 주파수 성분의 크기
- $e^{i2\pi kn/N}$: 복소 지수 (주파수 $k/N$)

### 2.2 구체적 수치 예시

**예시 2.2.1: DFT 계산**

신호: $x = [1, 0, -1, 0]$ ($N = 4$)

**DFT 계산:**

$$
X[0] = 1 + 0 + (-1) + 0 = 0
$$

$$
X[1] = 1 \cdot e^{-i0} + 0 \cdot e^{-i\pi/2} + (-1) \cdot e^{-i\pi} + 0 \cdot e^{-i3\pi/2}
$$

$$
= 1 - (-1) = 2
$$

$$
X[2] = 1 - 0 - (-1) - 0 = 2
$$

$$
X[3] = 1 \cdot e^{-i0} + 0 \cdot e^{-i3\pi/2} + (-1) \cdot e^{-i3\pi} + 0 \cdot e^{-i9\pi/2}
$$

$$
= 1 - (-1) = 2
$$

**결과:** $X = [0, 2, 2, 2]$

---

## 3. 단시간 푸리에 변환 (STFT)

### 3.1 STFT 정의

#### 정의 3.1.1: STFT

**STFT**는 시간에 따른 주파수 변화를 분석합니다:

$$
\text{STFT}\{x[n]\}(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] w[n - m] e^{-i\omega n}
$$

여기서 $w[n]$은 윈도우 함수입니다.

### 3.2 윈도우 함수

#### 정의 3.2.1: 윈도우 함수

**Hamming 윈도우:**

$$
w[n] = 0.54 - 0.46 \cos\left(\frac{2\pi n}{N-1}\right)
$$

**Hanning 윈도우:**

$$
w[n] = 0.5 \left(1 - \cos\left(\frac{2\pi n}{N-1}\right)\right)
$$

---

## 4. 스펙트로그램

### 4.1 스펙트로그램 정의

#### 정의 4.1.1: Spectrogram

**스펙트로그램**은 STFT의 크기입니다:

$$
\text{Spectrogram}(m, k) = |\text{STFT}(m, \omega_k)|
$$

#### 시각적 표현: 스펙트로그램

```
스펙트로그램:

시간 →
주파수
  ↑
  │  ████░░░░░░░░░░░░░░░░  (고주파)
  │  ████████░░░░░░░░░░░░
  │  ████████████░░░░░░░░
  │  ████████████████░░░░  (중주파)
  │  ████████████████████
  │  ████████████████████  (저주파)
  │  ████████████████████
  │
  └──────────────────────────────→ 시간

각 픽셀 = 주파수 성분의 강도
```

---

## 5. CS 관점: FFT 알고리즘

### 5.1 FFT (Fast Fourier Transform)

#### 정의 5.1.1: FFT

**FFT**는 DFT를 $O(N \log N)$ 시간에 계산합니다:

**Naive DFT:** $O(N^2)$
**FFT:** $O(N \log N)$

#### 알고리즘 5.1.1: Cooley-Tukey FFT

```
Algorithm: FFT(x, N)
1. if N == 1:
2.     return x
3. 
4. // 분할
5. even ← FFT(x[0::2], N/2)
6. odd ← FFT(x[1::2], N/2)
7. 
8. // 결합
9. for k = 0 to N/2 - 1:
10.    t ← exp(-2πik/N) × odd[k]
11.    X[k] ← even[k] + t
12.    X[k + N/2] ← even[k] - t
13. 
14. return X
```

**시간 복잡도:** $O(N \log N)$

---

## 질문과 답변 (Q&A)

### Q1: FFT는 왜 빠른가요?

**A:** FFT의 속도:

1. **분할 정복:**
   - 문제를 반으로 분할
   - 재귀적 해결

2. **중복 계산 제거:**
   - 같은 값 재사용
   - 캐싱 효과

3. **복잡도:**
   - Naive: $O(N^2)$
   - FFT: $O(N \log N)$
   - $N=1024$: 약 100배 빠름

---

## 참고 문헌

1. **Oppenheim & Schafer (2010)**: "Discrete-Time Signal Processing" - FFT
2. **Cooley & Tukey (1965)**: "An algorithm for the machine calculation of complex Fourier series"

---

**작성일**: 2025-01-XX  
**버전**: 3.0 (CS 관점 + 석사 수준 확장)

