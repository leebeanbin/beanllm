"""
Vector DB vs MongoDB 성능 및 정확도 비교 테스트

메시지 저장 및 검색의 성능과 정확도를 종합적으로 테스트합니다.

사용법:
    python3 tests/test_vector_db_performance.py

필수 요구사항:
    - Ollama 실행 중 (ollama serve)
    - 임베딩 모델 설치 (ollama pull nomic-embed-text)

선택적:
    - MongoDB (비교 테스트용)
"""

import asyncio
import json
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "playground" / "backend"))
sys.path.insert(0, str(project_root / "src"))  # beanllm 패키지 경로

# 환경 변수 설정 (테스트용)
import os

if not os.getenv("MONGODB_URI"):
    os.environ["MONGODB_URI"] = "mongodb://localhost:27017/beanllm_test"
if not os.getenv("OLLAMA_BASE_URL"):
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

# Faker는 선택적 의존성 (없으면 기본 데이터 사용)
try:
    from faker import Faker

    fake = Faker("ko_KR")  # 한국어 데이터 생성
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False

    # 기본 데이터 생성 함수
    def fake_sentence():
        return "테스트 문장입니다."

    def fake_paragraph():
        return "테스트 문단입니다. 여러 문장으로 구성된 긴 텍스트입니다."


class VectorDBPerformanceTest:
    """Vector DB 성능 테스트 클래스"""

    def __init__(self):
        self.test_results = {
            "storage_performance": {},
            "search_performance": {},
            "accuracy": {},
            "scalability": {},
        }
        self.test_messages = []

    async def setup(self):
        """테스트 환경 설정"""
        print("\n" + "=" * 80)
        print("🔧 테스트 환경 설정")
        print("=" * 80)

        # 서비스 초기화
        try:
            # .env 파일 로드 (있는 경우)
            try:
                from dotenv import load_dotenv

                env_path = Path(__file__).parent.parent / ".env"
                if env_path.exists():
                    load_dotenv(env_path)
            except ImportError:
                pass

            # message_vector_store 직접 import (의존성 최소화)
            # services/__init__.py를 거치지 않고 직접 로드
            import importlib.util
            import sys

            # beanllm 패키지 경로 설정
            beanllm_src = project_root / "src"
            if beanllm_src.exists() and (beanllm_src / "beanllm").exists():
                # src/beanllm 경로를 Python 경로에 추가
                if str(beanllm_src) not in sys.path:
                    sys.path.insert(0, str(beanllm_src))
                print(f"✅ beanllm 소스 경로 추가: {beanllm_src}")
            else:
                print("⚠️ beanllm 소스 경로를 찾을 수 없습니다.")
                print(f"   예상 경로: {beanllm_src}")
                print("   beanllm 패키지가 설치되어 있는지 확인하세요.")

            # message_vector_store 직접 import 시도
            try:
                # services 디렉토리를 Python 경로에 추가
                services_path = Path(__file__).parent.parent / "services"
                backend_path = Path(__file__).parent.parent
                if str(backend_path) not in sys.path:
                    sys.path.insert(0, str(backend_path))

                # message_vector_store 모듈 직접 로드
                message_store_path = services_path / "message_vector_store.py"
                if not message_store_path.exists():
                    print(f"❌ message_vector_store.py를 찾을 수 없습니다: {message_store_path}")
                    return False

                spec = importlib.util.spec_from_file_location(
                    "message_vector_store_module", message_store_path
                )
                message_store_module = importlib.util.module_from_spec(spec)

                # beanllm 모듈이 필요한 경우를 대비해 미리 설정
                # message_vector_store가 beanllm을 import하므로 경로가 설정되어 있어야 함
                spec.loader.exec_module(message_store_module)
                self.message_vector_store = message_store_module.message_vector_store

                if not self.message_vector_store:
                    print("⚠️ Vector DB 서비스가 None입니다 (Ollama 또는 임베딩 모델 확인 필요)")
                    print("   해결 방법:")
                    print("   1. Ollama 실행: ollama serve")
                    print("   2. 임베딩 모델 설치: ollama pull mxbai-embed-large:335m")
                    print("   3. 또는 다른 임베딩 모델 사용")
                    # Vector DB가 없어도 테스트는 진행 (에러만 표시)
                    return False

            except Exception as e:
                print(f"❌ message_vector_store 로드 실패: {e}")
                import traceback

                traceback.print_exc()
                return False

            # database 모듈 직접 로드 (motor 의존성 처리)
            try:
                database_path = Path(__file__).parent.parent / "database.py"
                spec = importlib.util.spec_from_file_location("database_module", database_path)
                database_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(database_module)
                self.mongodb = database_module.get_mongodb_database()
            except ImportError as e:
                if "motor" in str(e):
                    print("⚠️ MongoDB 모듈(motor)이 설치되지 않았습니다.")
                    print("   설치: pip install motor")
                    print("   또는 MongoDB 테스트를 건너뜁니다.")
                    self.mongodb = None
                else:
                    raise

            if not self.mongodb:
                print("⚠️ MongoDB 연결 실패 (MONGODB_URI 확인 필요 또는 motor 미설치)")
                print("   환경 변수 확인: MONGODB_URI=mongodb://localhost:27017/beanllm_test")
                print("   또는 MongoDB 테스트를 건너뜁니다.")
                # MongoDB 없이도 Vector DB 테스트는 가능

            if not self.message_vector_store:
                print("⚠️ Vector DB 서비스 초기화 실패")
                print("   Ollama가 실행 중인지 확인: ollama serve")
                print("   또는 임베딩 모델이 설치되어 있는지 확인")
                return False

            print("✅ Vector DB 서비스 초기화 완료")
            if self.mongodb:
                print("✅ MongoDB 연결 완료")
            else:
                print("⚠️ MongoDB 연결 없음 (Vector DB 테스트만 진행)")
            return True
        except Exception as e:
            print(f"❌ 초기화 실패: {e}")
            import traceback

            traceback.print_exc()
            return False

    def generate_test_messages(
        self, num_sessions: int = 10, messages_per_session: int = 50
    ) -> List[Dict[str, Any]]:
        """테스트용 메시지 생성"""
        print(
            f"\n📝 테스트 메시지 생성 중... ({num_sessions}개 세션, 각 {messages_per_session}개 메시지)"
        )

        messages = []
        topics = [
            "인공지능과 머신러닝",
            "데이터베이스 설계",
            "웹 개발 프레임워크",
            "클라우드 컴퓨팅",
            "사이버 보안",
            "블록체인 기술",
            "사이버 보안",
            "데이터 분석",
            "소프트웨어 아키텍처",
            "데브옵스 실무",
        ]

        for session_idx in range(num_sessions):
            session_id = f"test_session_{session_idx}_{uuid.uuid4().hex[:8]}"
            topic = topics[session_idx % len(topics)]

            for msg_idx in range(messages_per_session):
                # 사용자 메시지
                if FAKER_AVAILABLE:
                    user_content = f"{topic}에 대해 {fake.sentence()}"
                else:
                    user_content = f"{topic}에 대해 질문합니다. {msg_idx}번째 메시지입니다."

                messages.append(
                    {
                        "session_id": session_id,
                        "message_id": f"{session_id}_user_{msg_idx}",
                        "role": "user",
                        "content": user_content,
                        "model": "gpt-4o",
                        "timestamp": datetime.now(timezone.utc)
                        - timedelta(minutes=messages_per_session - msg_idx),
                        "metadata": {"topic": topic},
                    }
                )

                # Assistant 응답
                if FAKER_AVAILABLE:
                    assistant_content = f"{topic}에 대한 답변: {fake.paragraph()}"
                else:
                    assistant_content = f"{topic}에 대한 답변입니다. {msg_idx}번째 응답입니다. 상세한 설명과 예시를 포함합니다."

                messages.append(
                    {
                        "session_id": session_id,
                        "message_id": f"{session_id}_assistant_{msg_idx}",
                        "role": "assistant",
                        "content": assistant_content,
                        "model": "gpt-4o",
                        "timestamp": datetime.now(timezone.utc)
                        - timedelta(minutes=messages_per_session - msg_idx),
                        "metadata": {"topic": topic},
                    }
                )

        self.test_messages = messages
        print(f"✅ {len(messages)}개 메시지 생성 완료")
        return messages

    async def test_storage_performance(self):
        """저장 성능 테스트"""
        print("\n" + "=" * 80)
        print("📊 저장 성능 테스트")
        print("=" * 80)

        if not self.test_messages:
            self.generate_test_messages(num_sessions=5, messages_per_session=20)

        # Vector DB 저장 성능
        print("\n1️⃣ Vector DB 저장 성능 테스트")
        vector_start = time.time()
        vector_success = 0
        vector_failed = 0

        for msg in self.test_messages[:100]:  # 처음 100개만 테스트
            try:
                await self.message_vector_store.save_message(
                    session_id=msg["session_id"],
                    message_id=msg["message_id"],
                    role=msg["role"],
                    content=msg["content"],
                    model=msg["model"],
                    timestamp=msg["timestamp"],
                    metadata=msg.get("metadata"),
                )
                vector_success += 1
            except Exception as e:
                vector_failed += 1
                print(f"   ⚠️ 저장 실패: {e}")

        vector_elapsed = time.time() - vector_start
        vector_avg = vector_elapsed / len(self.test_messages[:100]) * 1000  # ms

        print(f"   ✅ Vector DB: {vector_success}개 성공, {vector_failed}개 실패")
        print(f"   ⏱️  총 시간: {vector_elapsed:.2f}초")
        print(f"   📈 평균: {vector_avg:.2f}ms/메시지")

        # MongoDB 저장 성능 (비교용)
        if self.mongodb:
            print("\n2️⃣ MongoDB 저장 성능 테스트 (비교용)")
            mongo_start = time.time()
            mongo_success = 0
            mongo_failed = 0

            for msg in self.test_messages[:100]:
                try:
                    await self.mongodb.chat_sessions.update_one(
                        {"session_id": msg["session_id"]},
                        {
                            "$push": {
                                "messages": {
                                    "message_id": msg["message_id"],
                                    "role": msg["role"],
                                    "content": msg["content"],
                                    "timestamp": msg["timestamp"],
                                }
                            },
                            "$setOnInsert": {
                                "session_id": msg["session_id"],
                                "created_at": datetime.now(timezone.utc),
                                "updated_at": datetime.now(timezone.utc),
                            },
                        },
                        upsert=True,
                    )
                    mongo_success += 1
                except Exception:
                    mongo_failed += 1

            mongo_elapsed = time.time() - mongo_start
            mongo_avg = mongo_elapsed / len(self.test_messages[:100]) * 1000  # ms

            print(f"   ✅ MongoDB: {mongo_success}개 성공, {mongo_failed}개 실패")
            print(f"   ⏱️  총 시간: {mongo_elapsed:.2f}초")
            print(f"   📈 평균: {mongo_avg:.2f}ms/메시지")
        else:
            print("\n2️⃣ MongoDB 저장 성능 테스트 (건너뜀 - MongoDB 연결 없음)")
            mongo_elapsed = 0
            mongo_avg = 0
            mongo_success = 0
            mongo_failed = 0

        # 결과 저장
        self.test_results["storage_performance"] = {
            "vector_db": {
                "total_time": vector_elapsed,
                "avg_time_ms": vector_avg,
                "success": vector_success,
                "failed": vector_failed,
                "throughput": len(self.test_messages[:100]) / vector_elapsed
                if vector_elapsed > 0
                else 0,  # messages/sec
            },
            "mongodb": {
                "total_time": mongo_elapsed,
                "avg_time_ms": mongo_avg,
                "success": mongo_success,
                "failed": mongo_failed,
                "throughput": len(self.test_messages[:100]) / mongo_elapsed
                if mongo_elapsed > 0
                else 0,  # messages/sec
            },
            "comparison": {
                "vector_db_faster": vector_elapsed < mongo_elapsed if mongo_elapsed > 0 else True,
                "speedup": mongo_elapsed / vector_elapsed
                if vector_elapsed > 0 and mongo_elapsed > 0
                else 0,
            },
        }

        print("\n📊 비교 결과:")
        if mongo_elapsed > 0:
            print(f"   {'Vector DB가' if vector_elapsed < mongo_elapsed else 'MongoDB가'} 더 빠름")
            print(f"   속도 차이: {abs(mongo_elapsed - vector_elapsed):.2f}초")
        else:
            print("   MongoDB 비교 불가 (MongoDB 연결 없음)")
            print(f"   Vector DB 저장 성능: {vector_avg:.2f}ms/메시지")

    async def test_search_performance(self):
        """검색 성능 테스트"""
        print("\n" + "=" * 80)
        print("🔍 검색 성능 테스트")
        print("=" * 80)

        # 테스트 쿼리들
        test_queries = [
            "인공지능과 머신러닝",
            "데이터베이스 설계 방법",
            "웹 개발 프레임워크 비교",
            "클라우드 서비스",
            "보안 취약점",
        ]

        # Vector DB 검색 성능
        print("\n1️⃣ Vector DB Semantic Search 성능")
        vector_results = {}
        vector_total_time = 0

        for query in test_queries:
            start = time.time()
            try:
                results = await self.message_vector_store.search_messages(query=query, k=10)
                elapsed = time.time() - start
                vector_total_time += elapsed
                vector_results[query] = {
                    "time": elapsed,
                    "count": len(results),
                    "results": results[:3],  # 처음 3개만 저장
                }
                print(f"   ✅ '{query}': {elapsed*1000:.2f}ms, {len(results)}개 결과")
            except Exception as e:
                print(f"   ❌ '{query}': 실패 - {e}")
                vector_results[query] = {"time": 0, "count": 0, "error": str(e)}

        vector_avg = vector_total_time / len(test_queries) * 1000  # ms
        print(f"   📈 평균 검색 시간: {vector_avg:.2f}ms")

        # MongoDB 텍스트 검색 성능 (비교용)
        if self.mongodb:
            print("\n2️⃣ MongoDB Text Search 성능 (비교용)")
            mongo_results = {}
            mongo_total_time = 0

            for query in test_queries:
                start = time.time()
                try:
                    # MongoDB 텍스트 검색 (정규표현식 사용)
                    results = await self.mongodb.chat_sessions.find(
                        {
                            "messages.content": {"$regex": query, "$options": "i"},
                        }
                    ).to_list(length=10)

                    elapsed = time.time() - start
                    mongo_total_time += elapsed
                    mongo_results[query] = {
                        "time": elapsed,
                        "count": len(results),
                    }
                    print(f"   ✅ '{query}': {elapsed*1000:.2f}ms, {len(results)}개 결과")
                except Exception as e:
                    print(f"   ❌ '{query}': 실패 - {e}")
                    mongo_results[query] = {"time": 0, "count": 0, "error": str(e)}

            mongo_avg = mongo_total_time / len(test_queries) * 1000  # ms
            print(f"   📈 평균 검색 시간: {mongo_avg:.2f}ms")
        else:
            print("\n2️⃣ MongoDB Text Search 성능 (건너뜀 - MongoDB 연결 없음)")
            mongo_results = {
                q: {"time": 0, "count": 0, "error": "MongoDB not available"} for q in test_queries
            }
            mongo_total_time = 0
            mongo_avg = 0

        # 결과 저장
        self.test_results["search_performance"] = {
            "vector_db": {
                "total_time": vector_total_time,
                "avg_time_ms": vector_avg,
                "queries": vector_results,
            },
            "mongodb": {
                "total_time": mongo_total_time,
                "avg_time_ms": mongo_avg,
                "queries": mongo_results,
            },
            "comparison": {
                "vector_db_faster": vector_total_time < mongo_total_time,
                "speedup": mongo_total_time / vector_total_time if vector_total_time > 0 else 0,
            },
        }

        print("\n📊 비교 결과:")
        if mongo_total_time > 0:
            print(
                f"   {'Vector DB가' if vector_total_time < mongo_total_time else 'MongoDB가'} 더 빠름"
            )
            print(f"   속도 차이: {abs(mongo_total_time - vector_total_time)*1000:.2f}ms")
        else:
            print("   MongoDB 비교 불가 (MongoDB 연결 없음)")
            print(f"   Vector DB 검색 성능: {vector_avg:.2f}ms")

    async def test_search_accuracy(self):
        """검색 정확도 테스트"""
        print("\n" + "=" * 80)
        print("🎯 검색 정확도 테스트")
        print("=" * 80)

        # 정확도 테스트 쿼리 (의미적으로 유사한 쿼리)
        accuracy_tests = [
            {
                "query": "인공지능",
                "expected_topics": ["인공지능과 머신러닝", "AI", "머신러닝"],
                "description": "의미 기반 검색 (동의어 포함)",
            },
            {
                "query": "데이터베이스",
                "expected_topics": ["데이터베이스 설계", "DB", "데이터 저장"],
                "description": "주제 기반 검색",
            },
            {
                "query": "보안",
                "expected_topics": ["사이버 보안", "보안 취약점", "보안 정책"],
                "description": "관련 주제 검색",
            },
        ]

        vector_accuracy = []
        mongo_accuracy = []

        for test in accuracy_tests:
            query = test["query"]
            expected = test["expected_topics"]
            print(f"\n📝 테스트: '{query}' ({test['description']})")

            # Vector DB 검색
            try:
                vector_results = await self.message_vector_store.search_messages(query=query, k=10)
                vector_matched = sum(
                    1
                    for result in vector_results
                    if any(topic.lower() in result.get("content", "").lower() for topic in expected)
                )
                vector_precision = vector_matched / len(vector_results) if vector_results else 0
                vector_accuracy.append(vector_precision)
                print(
                    f"   ✅ Vector DB: {vector_matched}/{len(vector_results)} 매칭 (정확도: {vector_precision*100:.1f}%)"
                )
            except Exception as e:
                print(f"   ❌ Vector DB 검색 실패: {e}")
                vector_accuracy.append(0)

            # MongoDB 검색
            if self.mongodb:
                try:
                    mongo_results = await self.mongodb.chat_sessions.find(
                        {
                            "messages.content": {"$regex": query, "$options": "i"},
                        }
                    ).to_list(length=10)

                    # 메시지 내용 추출
                    mongo_contents = []
                    for session in mongo_results:
                        for msg in session.get("messages", []):
                            if query.lower() in msg.get("content", "").lower():
                                mongo_contents.append(msg.get("content", ""))

                    mongo_matched = sum(
                        1
                        for content in mongo_contents
                        if any(topic.lower() in content.lower() for topic in expected)
                    )
                    mongo_precision = mongo_matched / len(mongo_contents) if mongo_contents else 0
                    mongo_accuracy.append(mongo_precision)
                    print(
                        f"   ✅ MongoDB: {mongo_matched}/{len(mongo_contents)} 매칭 (정확도: {mongo_precision*100:.1f}%)"
                    )
                except Exception as e:
                    print(f"   ❌ MongoDB 검색 실패: {e}")
                    mongo_accuracy.append(0)
            else:
                print("   ⚠️ MongoDB 검색 건너뜀 (MongoDB 연결 없음)")
                mongo_accuracy.append(0)

        avg_vector_accuracy = sum(vector_accuracy) / len(vector_accuracy) if vector_accuracy else 0
        avg_mongo_accuracy = sum(mongo_accuracy) / len(mongo_accuracy) if mongo_accuracy else 0

        self.test_results["accuracy"] = {
            "vector_db": {
                "individual": vector_accuracy,
                "average": avg_vector_accuracy,
            },
            "mongodb": {
                "individual": mongo_accuracy,
                "average": avg_mongo_accuracy,
            },
            "comparison": {
                "vector_db_better": avg_vector_accuracy > avg_mongo_accuracy,
                "accuracy_diff": avg_vector_accuracy - avg_mongo_accuracy,
            },
        }

        print("\n📊 정확도 비교:")
        print(f"   Vector DB 평균 정확도: {avg_vector_accuracy*100:.1f}%")
        print(f"   MongoDB 평균 정확도: {avg_mongo_accuracy*100:.1f}%")
        print(
            f"   {'Vector DB가' if avg_vector_accuracy > avg_mongo_accuracy else 'MongoDB가'} 더 정확함"
        )

    async def test_scalability(self):
        """확장성 테스트 (대량 데이터)"""
        print("\n" + "=" * 80)
        print("📈 확장성 테스트 (대량 데이터)")
        print("=" * 80)

        # 대량 데이터 생성
        print("\n📝 대량 테스트 데이터 생성 중...")
        large_messages = self.generate_test_messages(num_sessions=50, messages_per_session=100)
        print(f"✅ {len(large_messages)}개 메시지 생성 완료")

        # Vector DB 대량 저장
        print("\n1️⃣ Vector DB 대량 저장 테스트")
        vector_start = time.time()
        vector_success = 0

        batch_size = 100
        for i in range(0, len(large_messages), batch_size):
            batch = large_messages[i : i + batch_size]
            for msg in batch:
                try:
                    await self.message_vector_store.save_message(
                        session_id=msg["session_id"],
                        message_id=msg["message_id"],
                        role=msg["role"],
                        content=msg["content"],
                        model=msg["model"],
                        timestamp=msg["timestamp"],
                        metadata=msg.get("metadata"),
                    )
                    vector_success += 1
                except Exception:
                    pass

            if (i + batch_size) % 500 == 0:
                print(f"   진행: {i + batch_size}/{len(large_messages)}")

        vector_elapsed = time.time() - vector_start
        print(f"   ✅ {vector_success}개 저장 완료 ({vector_elapsed:.2f}초)")

        # 대량 검색 테스트
        print("\n2️⃣ 대량 데이터 검색 테스트")
        search_start = time.time()
        search_results = await self.message_vector_store.search_messages(query="인공지능", k=20)
        search_elapsed = time.time() - search_start
        print(f"   ✅ 검색 완료: {len(search_results)}개 결과 ({search_elapsed*1000:.2f}ms)")

        self.test_results["scalability"] = {
            "total_messages": len(large_messages),
            "storage_time": vector_elapsed,
            "storage_throughput": vector_success / vector_elapsed if vector_elapsed > 0 else 0,
            "search_time_ms": search_elapsed * 1000,
            "search_results": len(search_results),
        }

        print("\n📊 확장성 결과:")
        if vector_elapsed > 0:
            print(f"   저장 처리량: {vector_success / vector_elapsed:.1f} messages/sec")
        else:
            print("   저장 처리량: 측정 불가 (시간이 0초)")
        print(f"   검색 시간: {search_elapsed*1000:.2f}ms (대량 데이터)")

    async def test_session_retrieval(self):
        """세션별 메시지 조회 성능 테스트"""
        print("\n" + "=" * 80)
        print("📂 세션별 메시지 조회 성능 테스트")
        print("=" * 80)

        # 테스트 세션 ID
        test_session_ids = [msg["session_id"] for msg in self.test_messages[:10]]
        test_session_ids = list(set(test_session_ids))[:5]  # 중복 제거 후 5개

        # Vector DB 세션 조회
        print("\n1️⃣ Vector DB 세션 조회")
        vector_times = []
        for session_id in test_session_ids:
            start = time.time()
            try:
                messages = await self.message_vector_store.get_session_messages(
                    session_id=session_id
                )
                elapsed = time.time() - start
                vector_times.append(elapsed)
                print(f"   ✅ {session_id}: {len(messages)}개 메시지 ({elapsed*1000:.2f}ms)")
            except Exception as e:
                print(f"   ❌ {session_id}: 실패 - {e}")

        vector_avg = sum(vector_times) / len(vector_times) * 1000 if vector_times else 0
        print(f"   📈 평균: {vector_avg:.2f}ms")

        # MongoDB 세션 조회
        if self.mongodb:
            print("\n2️⃣ MongoDB 세션 조회 (비교용)")
            mongo_times = []
            for session_id in test_session_ids:
                start = time.time()
                try:
                    session = await self.mongodb.chat_sessions.find_one({"session_id": session_id})
                    messages = session.get("messages", []) if session else []
                    elapsed = time.time() - start
                    mongo_times.append(elapsed)
                    print(f"   ✅ {session_id}: {len(messages)}개 메시지 ({elapsed*1000:.2f}ms)")
                except Exception as e:
                    print(f"   ❌ {session_id}: 실패 - {e}")

            mongo_avg = sum(mongo_times) / len(mongo_times) * 1000 if mongo_times else 0
            print(f"   📈 평균: {mongo_avg:.2f}ms")
        else:
            print("\n2️⃣ MongoDB 세션 조회 (건너뜀 - MongoDB 연결 없음)")
            mongo_avg = 0

        print("\n📊 비교 결과:")
        if mongo_avg > 0:
            print(f"   {'Vector DB가' if vector_avg < mongo_avg else 'MongoDB가'} 더 빠름")
            print(f"   속도 차이: {abs(mongo_avg - vector_avg):.2f}ms")
        else:
            print("   MongoDB 비교 불가 (MongoDB 연결 없음)")
            print(f"   Vector DB 세션 조회 성능: {vector_avg:.2f}ms")

    async def cleanup(self):
        """테스트 데이터 정리"""
        print("\n" + "=" * 80)
        print("🧹 테스트 데이터 정리")
        print("=" * 80)

        # Vector DB 정리
        try:
            session_ids = list(set([msg["session_id"] for msg in self.test_messages]))
            for session_id in session_ids:
                await self.message_vector_store.delete_session_messages(session_id)
            print(f"✅ Vector DB 테스트 데이터 삭제 완료 ({len(session_ids)}개 세션)")
        except Exception as e:
            print(f"⚠️ Vector DB 정리 실패: {e}")

        # MongoDB 정리
        if self.mongodb:
            try:
                session_ids = list(set([msg["session_id"] for msg in self.test_messages]))
                await self.mongodb.chat_sessions.delete_many({"session_id": {"$in": session_ids}})
                print(f"✅ MongoDB 테스트 데이터 삭제 완료 ({len(session_ids)}개 세션)")
            except Exception as e:
                print(f"⚠️ MongoDB 정리 실패: {e}")
        else:
            print("⚠️ MongoDB 정리 건너뜀 (MongoDB 연결 없음)")

    def print_summary(self):
        """테스트 결과 요약 출력"""
        print("\n" + "=" * 80)
        print("📊 테스트 결과 요약")
        print("=" * 80)

        # 저장 성능
        if "storage_performance" in self.test_results:
            sp = self.test_results["storage_performance"]
            print("\n💾 저장 성능:")
            print(f"   Vector DB: {sp['vector_db']['avg_time_ms']:.2f}ms/메시지")
            if sp["mongodb"]["total_time"] > 0:
                print(f"   MongoDB: {sp['mongodb']['avg_time_ms']:.2f}ms/메시지")
                print(
                    f"   {'Vector DB가' if sp['comparison']['vector_db_faster'] else 'MongoDB가'} {sp['comparison']['speedup']:.2f}배 빠름"
                )
            else:
                print("   MongoDB: 비교 불가 (연결 없음)")

        # 검색 성능
        if "search_performance" in self.test_results:
            sp = self.test_results["search_performance"]
            print("\n🔍 검색 성능:")
            print(f"   Vector DB: {sp['vector_db']['avg_time_ms']:.2f}ms")
            if sp["mongodb"]["total_time"] > 0:
                print(f"   MongoDB: {sp['mongodb']['avg_time_ms']:.2f}ms")
                print(
                    f"   {'Vector DB가' if sp['comparison']['vector_db_faster'] else 'MongoDB가'} 더 빠름"
                )
            else:
                print("   MongoDB: 비교 불가 (연결 없음)")

        # 정확도
        if "accuracy" in self.test_results:
            acc = self.test_results["accuracy"]
            print("\n🎯 검색 정확도:")
            print(f"   Vector DB: {acc['vector_db']['average']*100:.1f}%")
            if len(acc["mongodb"]["individual"]) > 0 and any(
                a > 0 for a in acc["mongodb"]["individual"]
            ):
                print(f"   MongoDB: {acc['mongodb']['average']*100:.1f}%")
                print(
                    f"   {'Vector DB가' if acc['comparison']['vector_db_better'] else 'MongoDB가'} 더 정확함"
                )
            else:
                print("   MongoDB: 비교 불가 (연결 없음)")

        # 확장성
        if "scalability" in self.test_results:
            sc = self.test_results["scalability"]
            print("\n📈 확장성:")
            print(f"   처리량: {sc['storage_throughput']:.1f} messages/sec")
            print(f"   대량 검색: {sc['search_time_ms']:.2f}ms")

        # JSON 저장
        output_file = Path(__file__).parent / "vector_db_test_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, indent=2, default=str, ensure_ascii=False)
        print(f"\n💾 상세 결과 저장: {output_file}")


async def main():
    """메인 테스트 실행"""
    print("=" * 80)
    print("🚀 Vector DB vs MongoDB 성능 및 정확도 테스트")
    print("=" * 80)

    tester = VectorDBPerformanceTest()

    # 환경 설정
    if not await tester.setup():
        print("❌ 테스트 환경 설정 실패")
        return

    try:
        # 테스트 실행 (순차적으로)
        print("\n" + "=" * 80)
        print("📋 테스트 실행 순서")
        print("=" * 80)
        print("1. 저장 성능 테스트")
        print("2. 검색 성능 테스트")
        print("3. 검색 정확도 테스트")
        print("4. 확장성 테스트 (대량 데이터)")
        print("5. 세션별 조회 성능 테스트")
        print("=" * 80)

        await tester.test_storage_performance()
        await tester.test_search_performance()
        await tester.test_search_accuracy()
        await tester.test_scalability()
        await tester.test_session_retrieval()

        # 결과 요약
        tester.print_summary()

    except KeyboardInterrupt:
        print("\n⚠️ 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 정리
        try:
            await tester.cleanup()
        except Exception as e:
            print(f"⚠️ 정리 중 오류: {e}")

    print("\n✅ 모든 테스트 완료!")


if __name__ == "__main__":
    asyncio.run(main())
