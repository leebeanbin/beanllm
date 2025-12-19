# CS 기초: AI 엔지니어를 위한 컴퓨터 과학 기초

**학습 가이드 문서**  
**대상**: AI 엔지니어, ML 엔지니어, 초보 개발자

---

## 목차

1. [데이터 구조와 알고리즘](#1-데이터-구조와-알고리즘)
2. [시간 복잡도와 공간 복잡도](#2-시간-복잡도와-공간-복잡도)
3. [시스템 설계 원칙](#3-시스템-설계-원칙)
4. [네트워크와 분산 시스템](#4-네트워크와-분산-시스템)
5. [데이터베이스 기초](#5-데이터베이스-기초)
6. [프로덕션 환경 이해](#6-프로덕션-환경-이해)

---

## 1. 데이터 구조와 알고리즘

### 1.1 필수 데이터 구조

**1. 배열 (Array)**
```python
# Python 리스트
arr = [1, 2, 3, 4, 5]
# 접근: O(1)
# 삽입/삭제: O(n)
```

**2. 해시 테이블 (Hash Table)**
```python
# Python 딕셔너리
d = {"key": "value"}
# 접근: O(1) 평균
# 삽입/삭제: O(1) 평균
```

**3. 힙 (Heap)**
```python
import heapq
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 2)
# 최소값 추출: O(log n)
```

**4. 트리 (Tree)**
```python
# 이진 트리
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
```

**5. 그래프 (Graph)**
```python
# 인접 리스트
graph = {
    1: [2, 3],
    2: [1, 4],
    3: [1],
    4: [2]
}
```

### 1.2 필수 알고리즘

**1. 정렬**
```python
# 퀵소트: O(n log n) 평균
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```

**2. 검색**
```python
# 이진 검색: O(log n)
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

**3. 그래프 탐색**
```python
# BFS: O(V + E)
from collections import deque

def bfs(graph, start):
    queue = deque([start])
    visited = {start}
    
    while queue:
        node = queue.popleft()
        print(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

---

## 2. 시간 복잡도와 공간 복잡도

### 2.1 Big-O 표기법

**복잡도 비교:**

| 복잡도 | 이름 | 예시 |
|--------|------|------|
| O(1) | 상수 | 배열 접근 |
| O(log n) | 로그 | 이진 검색 |
| O(n) | 선형 | 배열 순회 |
| O(n log n) | 선형 로그 | 퀵소트 |
| O(n²) | 제곱 | 버블소트 |
| O(2ⁿ) | 지수 | 피보나치 (재귀) |

### 2.2 실제 예시

**임베딩 검색:**
```python
# O(n·d): n개 문서, d차원
def similarity_search(query, candidates):
    similarities = []
    for candidate in candidates:  # O(n)
        sim = cosine_similarity(query, candidate)  # O(d)
        similarities.append(sim)
    return similarities
```

**최적화:**
```python
# NumPy 벡터화: O(n·d) 하지만 더 빠름
import numpy as np
def similarity_search_optimized(query, candidates):
    query = np.array(query)
    candidates = np.array(candidates)
    similarities = np.dot(candidates, query)  # 벡터화
    return similarities
```

---

## 3. 시스템 설계 원칙

### 3.1 확장성 (Scalability)

**수직 확장 (Vertical Scaling)**
- 더 강력한 서버
- 제한적

**수평 확장 (Horizontal Scaling)**
- 더 많은 서버
- 무한 확장 가능

### 3.2 가용성 (Availability)

**고가용성 설계:**
- 로드 밸런싱
- 복제 (Replication)
- 장애 복구 (Failover)

### 3.3 일관성 (Consistency)

**CAP 정리:**
- Consistency: 모든 노드가 같은 데이터
- Availability: 항상 응답
- Partition tolerance: 네트워크 분할 견딤

**선택:**
- CP: 일관성 우선 (은행 시스템)
- AP: 가용성 우선 (소셜 미디어)

---

## 4. 네트워크와 분산 시스템

### 4.1 HTTP 기초

**RESTful API:**
```python
# GET: 조회
GET /api/documents/123

# POST: 생성
POST /api/documents
Body: {"title": "New Document"}

# PUT: 전체 업데이트
PUT /api/documents/123
Body: {"title": "Updated Document"}

# DELETE: 삭제
DELETE /api/documents/123
```

### 4.2 비동기 처리

**비동기 프로그래밍:**
```python
import asyncio

async def fetch_data(url):
    # 비동기 HTTP 요청
    response = await aiohttp.get(url)
    return response.json()

async def main():
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results
```

---

## 5. 데이터베이스 기초

### 5.1 SQL 기초

**기본 쿼리:**
```sql
-- SELECT
SELECT * FROM documents WHERE id = 1;

-- INSERT
INSERT INTO documents (title, content) VALUES ('Title', 'Content');

-- UPDATE
UPDATE documents SET title = 'New Title' WHERE id = 1;

-- DELETE
DELETE FROM documents WHERE id = 1;
```

### 5.2 NoSQL

**문서 DB (MongoDB):**
```python
# 문서 저장
document = {
    "title": "Document",
    "content": "Content",
    "embedding": [0.1, 0.2, ...]
}
collection.insert_one(document)

# 검색
results = collection.find({"title": "Document"})
```

---

## 6. 프로덕션 환경 이해

### 6.1 컨테이너화

**Docker:**
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

### 6.2 모니터링

**로깅:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("Application started")
```

**메트릭:**
```python
# Prometheus 메트릭
from prometheus_client import Counter, Histogram

request_count = Counter('requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')
```

---

## 학습 경로

### 초급
1. Python 기초
2. 데이터 구조와 알고리즘
3. 데이터베이스 기초

### 중급
1. 시스템 설계
2. 네트워크 프로그래밍
3. 분산 시스템

### 고급
1. 마이크로서비스 아키텍처
2. 클라우드 인프라
3. 성능 최적화

---

## 추천 자료

- **알고리즘**: "Introduction to Algorithms" (CLRS)
- **시스템 설계**: "Designing Data-Intensive Applications"
- **네트워크**: "Computer Networks" (Tanenbaum)

---

**작성일**: 2025-01-XX  
**버전**: 1.0

