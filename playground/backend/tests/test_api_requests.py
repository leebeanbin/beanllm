"""
프론트엔드 API 요청 테스트 스크립트

프론트엔드가 보내는 실제 요청을 시뮬레이션하여 모니터링이 작동하는지 테스트합니다.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import httpx

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# .env 파일 로드
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ .env 파일 로드: {env_path}")
    else:
        print(f"ℹ️  .env 파일 없음: {env_path}")
except ImportError:
    print("⚠️  python-dotenv가 설치되지 않음")

# Redis 클라이언트 (대시보드와 동일한 방식)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️  redis 패키지가 설치되지 않음")


def check_redis_metrics():
    """Redis에서 메트릭 확인"""
    if not REDIS_AVAILABLE:
        return None
    
    try:
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )
        
        # 연결 테스트
        redis_client.ping()
        
        # 메트릭 조회
        metrics_keys = redis_client.keys("metrics:*")
        request_keys = redis_client.keys("request:status:*")
        
        # 응답 시간 메트릭
        response_times = redis_client.zrange("metrics:response_time", 0, -1, withscores=True)
        
        # 요청 수 메트릭
        current_time = int(time.time())
        minute_key = f"metrics:requests:{current_time // 60}"
        request_count = redis_client.get(minute_key)
        
        # 엔드포인트 통계
        endpoint_keys = redis_client.keys("metrics:endpoint:*")
        endpoint_stats = {}
        for key in endpoint_keys[:5]:  # 처음 5개만
            stats = redis_client.hgetall(key)
            endpoint_stats[key] = stats
        
        return {
            "metrics_keys_count": len(metrics_keys),
            "request_keys_count": len(request_keys),
            "response_times_count": len(response_times),
            "request_count": request_count,
            "endpoint_stats_count": len(endpoint_keys),
            "sample_endpoint_stats": endpoint_stats,
            "sample_response_times": [(k, v) for k, v in response_times[:5]],
        }
    except Exception as e:
        print(f"❌ Redis 메트릭 확인 실패: {e}")
        return None


async def test_chat_api():
    """Chat API 테스트 (프론트엔드와 동일한 요청)"""
    print("=" * 60)
    print("Chat API 요청 테스트")
    print("=" * 60)
    
    api_url = os.getenv("API_URL", "http://localhost:8000")
    
    # 프론트엔드가 보내는 요청과 동일한 형식
    request_data = {
        "messages": [
            {"role": "user", "content": "안녕하세요! 테스트 메시지입니다."}
        ],
        "model": "qwen2.5:0.5b",  # 기본 모델
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "enable_thinking": False,
    }
    
    print(f"\n📤 요청 전송:")
    print(f"   URL: {api_url}/api/chat")
    print(f"   Method: POST")
    print(f"   Model: {request_data['model']}")
    print(f"   Messages: {len(request_data['messages'])}개")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            start_time = time.time()
            
            response = await client.post(
                f"{api_url}/api/chat",
                json=request_data,
                headers={
                    "Content-Type": "application/json",
                },
            )
            
            duration = time.time() - start_time
            
            print(f"\n📥 응답 수신:")
            print(f"   Status: {response.status_code}")
            print(f"   Duration: {duration:.2f}초")
            
            # Request ID 확인
            request_id = response.headers.get("X-Request-ID")
            response_time_header = response.headers.get("X-Response-Time")
            
            if request_id:
                print(f"   Request ID: {request_id}")
            if response_time_header:
                print(f"   Response Time: {response_time_header}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   Model: {data.get('model', 'N/A')}")
                print(f"   Provider: {data.get('provider', 'N/A')}")
                if 'usage' in data:
                    usage = data['usage']
                    print(f"   Tokens: {usage.get('input_tokens', 0)} input / {usage.get('output_tokens', 0)} output")
                print(f"   Content length: {len(data.get('content', ''))} chars")
                print(f"\n✅ 요청 성공!")
                
                # 잠시 대기 (Redis에 데이터가 저장될 시간)
                print("\n⏳ Redis에 데이터가 저장될 때까지 2초 대기...")
                await asyncio.sleep(2)
                
                # Redis 메트릭 확인
                print("\n📊 Redis 메트릭 확인:")
                metrics = check_redis_metrics()
                if metrics:
                    print(f"   ✅ 메트릭 키 개수: {metrics['metrics_keys_count']}")
                    print(f"   ✅ 요청 상태 키 개수: {metrics['request_keys_count']}")
                    print(f"   ✅ 응답 시간 메트릭 개수: {metrics['response_times_count']}")
                    print(f"   ✅ 현재 분 요청 수: {metrics['request_count']}")
                    print(f"   ✅ 엔드포인트 통계 개수: {metrics['endpoint_stats_count']}")
                    
                    if metrics['sample_response_times']:
                        print(f"\n   📈 샘플 응답 시간:")
                        for req_id, resp_time in metrics['sample_response_times']:
                            print(f"      - {req_id[:8]}... : {resp_time:.2f}ms")
                    
                    if metrics['sample_endpoint_stats']:
                        print(f"\n   📊 샘플 엔드포인트 통계:")
                        for endpoint, stats in list(metrics['sample_endpoint_stats'].items())[:3]:
                            print(f"      - {endpoint}:")
                            for key, value in stats.items():
                                print(f"        {key}: {value}")
                else:
                    print("   ❌ Redis 메트릭을 확인할 수 없습니다")
                
                return True
            else:
                error_data = response.text
                print(f"   ❌ 요청 실패: {error_data}")
                return False
                
    except httpx.TimeoutException:
        print("   ❌ 요청 타임아웃 (60초 초과)")
        return False
    except Exception as e:
        print(f"   ❌ 요청 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_multiple_requests():
    """여러 요청 테스트"""
    print("\n" + "=" * 60)
    print("여러 요청 테스트 (3개 요청)")
    print("=" * 60)
    
    api_url = os.getenv("API_URL", "http://localhost:8000")
    
    requests = [
        {"messages": [{"role": "user", "content": "첫 번째 테스트 메시지"}]},
        {"messages": [{"role": "user", "content": "두 번째 테스트 메시지"}]},
        {"messages": [{"role": "user", "content": "세 번째 테스트 메시지"}]},
    ]
    
    success_count = 0
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, request_data in enumerate(requests, 1):
            print(f"\n📤 요청 {i}/3 전송 중...")
            
            try:
                full_request = {
                    **request_data,
                    "model": "qwen2.5:0.5b",
                    "temperature": 0.7,
                    "max_tokens": 100,
                }
                
                response = await client.post(
                    f"{api_url}/api/chat",
                    json=full_request,
                    headers={"Content-Type": "application/json"},
                )
                
                if response.status_code == 200:
                    request_id = response.headers.get("X-Request-ID", "N/A")
                    print(f"   ✅ 성공 (Request ID: {request_id[:8]}...)")
                    success_count += 1
                else:
                    print(f"   ❌ 실패 (Status: {response.status_code})")
                
                # 요청 간 간격
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"   ❌ 오류: {e}")
    
    print(f"\n📊 결과: {success_count}/3 요청 성공")
    
    # 최종 메트릭 확인
    print("\n📊 최종 Redis 메트릭:")
    await asyncio.sleep(2)  # 데이터 저장 대기
    metrics = check_redis_metrics()
    if metrics:
        print(f"   ✅ 메트릭 키 개수: {metrics['metrics_keys_count']}")
        print(f"   ✅ 요청 상태 키 개수: {metrics['request_keys_count']}")
        print(f"   ✅ 응답 시간 메트릭 개수: {metrics['response_times_count']}")
        print(f"   ✅ 엔드포인트 통계 개수: {metrics['endpoint_stats_count']}")
    
    return success_count == 3


async def test_health_check():
    """백엔드 헬스 체크"""
    print("=" * 60)
    print("백엔드 헬스 체크")
    print("=" * 60)
    
    api_url = os.getenv("API_URL", "http://localhost:8000")
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{api_url}/health")
            if response.status_code == 200:
                print("✅ 백엔드가 실행 중입니다")
                return True
            else:
                print(f"⚠️  백엔드 응답: {response.status_code}")
                return False
    except httpx.ConnectError:
        print("❌ 백엔드에 연결할 수 없습니다")
        print(f"   URL: {api_url}")
        print("   백엔드가 실행 중인지 확인하세요: python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"❌ 헬스 체크 실패: {e}")
        return False


async def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("프론트엔드 API 요청 시뮬레이션 테스트")
    print("=" * 60)
    print()
    
    results = []
    
    # 1. 헬스 체크
    health_ok = await test_health_check()
    if not health_ok:
        print("\n❌ 백엔드가 실행 중이지 않습니다. 테스트를 중단합니다.")
        return
    
    # 2. 단일 요청 테스트
    results.append(("Chat API 단일 요청", await test_chat_api()))
    
    # 3. 여러 요청 테스트
    results.append(("Chat API 여러 요청", await test_multiple_requests()))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    for name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 모든 테스트 통과!")
        print("\n이제 대시보드에서 데이터를 확인할 수 있습니다:")
        print("1. Streamlit 대시보드 실행: streamlit run monitoring_dashboard.py")
        print("2. 브라우저에서 http://localhost:8501 접속")
        print("3. '디버깅 정보' 섹션에서 메트릭 확인")
    else:
        print("❌ 일부 테스트 실패")
        print("\n실패한 테스트를 확인하고 문제를 해결하세요.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
