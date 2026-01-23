"""
프론트엔드 16개 페이지 테스트 스크립트
- Ollama 무료 모델만 사용 (qwen2.5:0.5b, phi3, phi3.5, nomic-embed-text)
"""
import asyncio
import aiohttp

FRONTEND_URL = "http://localhost:3000"
PAGES = [
    "/",
    "/chat",
    "/rag",
    "/agent",
    "/multi-agent",
    "/chain",
    "/knowledge-graph",
    "/vision-rag",
    "/audio",
    "/ocr",
    "/web-search",
    "/evaluation",
    "/finetuning",
    "/rag-debug",
    "/optimizer",
    "/orchestrator"
]

async def test_page(session, page_path):
    """페이지 로딩 테스트"""
    url = f"{FRONTEND_URL}{page_path}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            status = response.status
            html = await response.text()

            # 기본 체크
            has_title = '<title>' in html
            has_body = '<body' in html
            has_error = 'Application error' in html or 'Error:' in html

            return {
                'path': page_path,
                'status': status,
                'has_title': has_title,
                'has_body': has_body,
                'has_error': has_error,
                'size': len(html),
                'success': status == 200 and has_title and has_body and not has_error
            }
    except Exception as e:
        return {
            'path': page_path,
            'status': 'ERROR',
            'error': str(e),
            'success': False
        }

async def main():
    """모든 페이지 테스트"""
    print("=" * 60)
    print("프론트엔드 16개 페이지 UI 테스트 시작")
    print("=" * 60)
    print()

    async with aiohttp.ClientSession() as session:
        tasks = [test_page(session, page) for page in PAGES]
        results = await asyncio.gather(*tasks)

        # 결과 출력
        success_count = 0
        fail_count = 0

        for result in results:
            page = result['path']
            if result['success']:
                print(f"✅ {page:25s} - OK (Status: {result['status']}, Size: {result['size']:,} bytes)")
                success_count += 1
            else:
                print(f"❌ {page:25s} - FAIL")
                if 'error' in result:
                    print(f"   Error: {result['error']}")
                else:
                    print(f"   Status: {result['status']}, Has Error: {result['has_error']}")
                fail_count += 1

        print()
        print("=" * 60)
        print(f"테스트 완료: {success_count}/{len(PAGES)} 성공, {fail_count}/{len(PAGES)} 실패")
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
