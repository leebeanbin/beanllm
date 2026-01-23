"""
모니터링 시스템 테스트 스크립트

이 스크립트는 모니터링 시스템이 제대로 작동하는지 테스트합니다.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

async def test_redis_connection():
    """Redis 연결 테스트"""
    print("=" * 60)
    print("1. Redis 연결 테스트")
    print("=" * 60)
    
    try:
        from beanllm.infrastructure.distributed.redis.client import get_redis_client
        
        redis_client = get_redis_client()
        if redis_client:
            # ping 테스트
            result = await redis_client.ping()
            print(f"✅ Redis 연결 성공: {result}")
            print(f"   Redis 클라이언트 타입: {type(redis_client)}")
            return True
        else:
            print("❌ Redis 클라이언트가 None입니다")
            return False
    except Exception as e:
        print(f"❌ Redis 연결 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_request_monitor():
    """RequestMonitor 초기화 테스트"""
    print("\n" + "=" * 60)
    print("2. RequestMonitor 초기화 테스트")
    print("=" * 60)
    
    try:
        from beanllm.infrastructure.distributed.messaging import RequestMonitor
        
        monitor = RequestMonitor()
        if monitor and monitor.redis:
            print(f"✅ RequestMonitor 초기화 성공")
            print(f"   Redis 클라이언트 타입: {type(monitor.redis)}")
            return True
        else:
            print("❌ RequestMonitor.redis가 None입니다")
            return False
    except Exception as e:
        print(f"❌ RequestMonitor 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_redis_write():
    """Redis에 데이터 쓰기 테스트"""
    print("\n" + "=" * 60)
    print("3. Redis 데이터 쓰기 테스트")
    print("=" * 60)
    
    try:
        from beanllm.infrastructure.distributed.redis.client import get_redis_client
        
        redis_client = get_redis_client()
        if not redis_client:
            print("❌ Redis 클라이언트가 없습니다")
            return False
        
        # 테스트 데이터 쓰기
        test_key = "test:monitoring:write"
        test_value = json.dumps({"test": "data", "timestamp": 1234567890})
        
        await redis_client.setex(test_key, 60, test_value.encode('utf-8'))
        print(f"✅ 테스트 데이터 쓰기 성공: {test_key}")
        
        # 데이터 읽기
        result = await redis_client.get(test_key)
        if result:
            if isinstance(result, bytes):
                result = result.decode('utf-8')
            print(f"✅ 테스트 데이터 읽기 성공: {result}")
            await redis_client.delete(test_key)
            print(f"✅ 테스트 데이터 삭제 완료")
            return True
        else:
            print("❌ 테스트 데이터를 읽을 수 없습니다")
            return False
    except Exception as e:
        print(f"❌ Redis 데이터 쓰기 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_metrics_storage():
    """메트릭 저장 테스트"""
    print("\n" + "=" * 60)
    print("4. 메트릭 저장 테스트")
    print("=" * 60)
    
    try:
        from beanllm.infrastructure.distributed.redis.client import get_redis_client
        import time
        
        redis_client = get_redis_client()
        if not redis_client:
            print("❌ Redis 클라이언트가 없습니다")
            return False
        
        # 테스트 메트릭 저장
        request_id = "test-request-123"
        current_time = int(time.time())
        
        # 1. 요청 상태 저장
        status_data = {
            "request_id": request_id,
            "status": "completed",
            "started_at": current_time - 1,
            "completed_at": current_time,
            "duration_ms": 100.5,
            "status_code": 200,
            "path": "/api/test",
            "method": "GET",
        }
        await redis_client.setex(
            f"request:status:{request_id}",
            3600,
            json.dumps(status_data).encode('utf-8')
        )
        print(f"✅ 요청 상태 저장 성공: request:status:{request_id}")
        
        # 2. 응답 시간 메트릭 저장
        await redis_client.zadd(
            "metrics:response_time",
            {request_id: 100.5}
        )
        await redis_client.expire("metrics:response_time", 3600)
        print(f"✅ 응답 시간 메트릭 저장 성공")
        
        # 3. 요청 수 메트릭 저장
        minute_key = f"metrics:requests:{current_time // 60}"
        await redis_client.incr(minute_key)
        await redis_client.expire(minute_key, 3600)
        print(f"✅ 요청 수 메트릭 저장 성공: {minute_key}")
        
        # 4. 엔드포인트 통계 저장
        endpoint_key = "metrics:endpoint:GET:/api/test"
        await redis_client.hincrby(endpoint_key, "count", 1)
        await redis_client.hincrby(endpoint_key, "total_time_ms", 100)
        await redis_client.expire(endpoint_key, 3600)
        print(f"✅ 엔드포인트 통계 저장 성공: {endpoint_key}")
        
        # 데이터 확인
        status_check = await redis_client.get(f"request:status:{request_id}")
        response_times = await redis_client.zrange("metrics:response_time", 0, -1, withscores=True)
        request_count = await redis_client.get(minute_key)
        endpoint_stats = await redis_client.hgetall(endpoint_key)
        
        print(f"\n📊 저장된 데이터 확인:")
        print(f"   - 요청 상태: {'✅' if status_check else '❌'}")
        print(f"   - 응답 시간 메트릭: {len(response_times)}개")
        print(f"   - 요청 수 메트릭: {request_count}")
        print(f"   - 엔드포인트 통계: {endpoint_stats}")
        
        return True
    except Exception as e:
        print(f"❌ 메트릭 저장 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_dashboard_data_retrieval():
    """대시보드 데이터 조회 테스트"""
    print("\n" + "=" * 60)
    print("5. 대시보드 데이터 조회 테스트")
    print("=" * 60)
    
    try:
        import redis
        import time
        
        # 동기 Redis 클라이언트 (대시보드와 동일)
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=True,
        )
        
        # 연결 테스트
        redis_client.ping()
        print("✅ Redis 연결 성공 (동기 클라이언트)")
        
        # 메트릭 조회
        current_time = int(time.time())
        time_window_minutes = 60
        
        # 1. 응답 시간 메트릭
        response_times_raw = redis_client.zrange(
            "metrics:response_time", 0, -1, withscores=True
        )
        print(f"   - 응답 시간 메트릭: {len(response_times_raw)}개")
        
        # 2. 요청 수 메트릭
        request_counts = {}
        for minute in range(time_window_minutes):
            minute_key = f"metrics:requests:{int((current_time - minute * 60) // 60)}"
            count = redis_client.get(minute_key)
            if count:
                request_counts[minute] = int(count)
        print(f"   - 요청 수 메트릭: {len(request_counts)}개")
        
        # 3. 엔드포인트 통계
        endpoint_keys = redis_client.keys("metrics:endpoint:*")
        print(f"   - 엔드포인트 통계: {len(endpoint_keys)}개")
        
        # 4. 토큰 통계
        token_keys = redis_client.keys("metrics:tokens:*")
        print(f"   - 토큰 통계: {len(token_keys)}개")
        
        # 5. 요청 상태
        request_keys = redis_client.keys("request:status:*")
        print(f"   - 요청 상태: {len(request_keys)}개")
        
        print(f"\n📊 전체 메트릭 키 개수: {len(redis_client.keys('metrics:*'))}")
        print(f"📊 전체 요청 키 개수: {len(redis_client.keys('request:*'))}")
        
        if len(response_times_raw) == 0 and len(request_counts) == 0:
            print("\n⚠️  경고: Redis에 메트릭 데이터가 없습니다.")
            print("   백엔드가 요청을 받아야 데이터가 저장됩니다.")
            return False
        
        return True
    except Exception as e:
        print(f"❌ 대시보드 데이터 조회 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("모니터링 시스템 종합 테스트")
    print("=" * 60)
    print()
    
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
    
    results = []
    
    # 테스트 실행
    results.append(("Redis 연결", await test_redis_connection()))
    results.append(("RequestMonitor 초기화", await test_request_monitor()))
    results.append(("Redis 데이터 쓰기", await test_redis_write()))
    results.append(("메트릭 저장", await test_metrics_storage()))
    results.append(("대시보드 데이터 조회", await test_dashboard_data_retrieval()))
    
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
        print("\n대시보드가 정상적으로 작동해야 합니다.")
        print("만약 대시보드가 비어있다면, 백엔드가 실제 요청을 받아야 데이터가 저장됩니다.")
    else:
        print("❌ 일부 테스트 실패")
        print("\n실패한 테스트를 확인하고 문제를 해결하세요.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
