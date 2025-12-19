"""
Web Search 실습 튜토리얼

이 튜토리얼에서는 Google, Bing, DuckDuckGo 검색 엔진 통합과
웹 스크래핑, 실시간 정보 검색을 실습합니다.

Prerequisites:
- Python 3.10+
- requests, httpx, beautifulsoup4, duckduckgo-search 설치
- (Optional) Google/Bing API keys

Install:
    pip install requests httpx beautifulsoup4 duckduckgo-search

Author: LLMKit Team
"""

import asyncio
import time
from typing import List
import os

# ============================================================================
# Part 1: DuckDuckGo Search (No API Key Required!)
# ============================================================================

print("=" * 80)
print("Part 1: DuckDuckGo Search (No API Key!)")
print("=" * 80)

from llmkit.web_search import DuckDuckGoSearch, search_web

# Example 1.1: Basic search
print("\n1.1 Basic DuckDuckGo search:")
ddg = DuckDuckGoSearch(max_results=5)

try:
    results = ddg.search("Python programming")

    print(f"Query: {results.query}")
    print(f"Engine: {results.engine}")
    print(f"Search time: {results.search_time:.2f}s")
    print(f"Total results: {len(results.results)}\n")

    for i, result in enumerate(results.results, 1):
        print(f"{i}. {result.title}")
        print(f"   URL: {result.url}")
        print(f"   Snippet: {result.snippet[:100]}...")
        print()

except Exception as e:
    print(f"Error: {e}")


# Example 1.2: Search with different regions
print("\n1.2 Region-specific search:")
ddg = DuckDuckGoSearch(max_results=3)

try:
    # US English
    results_us = ddg.search("football", region="us-en")
    print("US results (American football):")
    for result in results_us.results[:2]:
        print(f"  - {result.title[:60]}")

    # UK English
    results_uk = ddg.search("football", region="uk-en")
    print("\nUK results (Soccer):")
    for result in results_uk.results[:2]:
        print(f"  - {result.title[:60]}")

except Exception as e:
    print(f"Error: {e}")


# Example 1.3: Using the convenience function
print("\n1.3 Convenience function search_web():")
try:
    results = search_web(
        "machine learning",
        engine="duckduckgo",
        max_results=3
    )

    print(f"Found {len(results)} results:")
    for result in results:
        print(f"  • {result.title}")

except Exception as e:
    print(f"Error: {e}")


# ============================================================================
# Part 2: Google Custom Search (Requires API Key)
# ============================================================================

print("\n" + "=" * 80)
print("Part 2: Google Custom Search")
print("=" * 80)

from llmkit.web_search import GoogleSearch

# Note: You need to:
# 1. Get API key from Google Cloud Console
# 2. Create Custom Search Engine at programmablesearchengine.google.com
# 3. Get Search Engine ID

google_api_key = os.getenv("GOOGLE_API_KEY")
google_search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

if google_api_key and google_search_engine_id:
    print("\n2.1 Google search:")
    google = GoogleSearch(
        api_key=google_api_key,
        search_engine_id=google_search_engine_id,
        max_results=5
    )

    try:
        results = google.search("artificial intelligence", language="en")

        print(f"Query: {results.query}")
        print(f"Total results (estimated): {results.total_results:,}")
        print(f"Search time: {results.search_time:.2f}s\n")

        for i, result in enumerate(results.results, 1):
            print(f"{i}. {result.title}")
            print(f"   {result.url}")
            print(f"   {result.snippet[:80]}...")
            print()

    except Exception as e:
        print(f"Error: {e}")


    # Example 2.2: Language-specific search
    print("\n2.2 Korean language search:")
    try:
        results_ko = google.search("머신러닝", language="ko")

        for result in results_ko.results[:3]:
            print(f"  - {result.title}")

    except Exception as e:
        print(f"Error: {e}")

else:
    print("\n⚠️  Google API keys not configured.")
    print("   Set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables.")
    print("   See: https://developers.google.com/custom-search/v1/overview")


# ============================================================================
# Part 3: Bing Search (Requires API Key)
# ============================================================================

print("\n" + "=" * 80)
print("Part 3: Bing Search")
print("=" * 80)

from llmkit.web_search import BingSearch

bing_api_key = os.getenv("BING_API_KEY")

if bing_api_key:
    print("\n3.1 Bing search:")
    bing = BingSearch(
        api_key=bing_api_key,
        max_results=5
    )

    try:
        results = bing.search("quantum computing", market="en-US")

        print(f"Query: {results.query}")
        print(f"Total results (estimated): {results.total_results:,}")
        print(f"Search time: {results.search_time:.2f}s\n")

        for i, result in enumerate(results.results, 1):
            print(f"{i}. {result.title}")
            print(f"   {result.url}")
            print(f"   {result.snippet[:80]}...")
            if result.published_date:
                print(f"   Published: {result.published_date}")
            print()

    except Exception as e:
        print(f"Error: {e}")


    # Example 3.2: Different markets
    print("\n3.2 Market-specific search:")
    try:
        # US market
        results_us = bing.search("football", market="en-US")
        print("US market:")
        print(f"  Top result: {results_us.results[0].title[:60]}")

        # UK market
        results_uk = bing.search("football", market="en-GB")
        print("\nUK market:")
        print(f"  Top result: {results_uk.results[0].title[:60]}")

    except Exception as e:
        print(f"Error: {e}")

else:
    print("\n⚠️  Bing API key not configured.")
    print("   Set BING_API_KEY environment variable.")
    print("   Get key from Azure Portal: Bing Search v7")


# ============================================================================
# Part 4: Web Scraping
# ============================================================================

print("\n" + "=" * 80)
print("Part 4: Web Scraping")
print("=" * 80)

from llmkit.web_search import WebScraper

# Example 4.1: Scrape a webpage
print("\n4.1 Scraping a webpage:")
scraper = WebScraper()

try:
    content = scraper.scrape("https://en.wikipedia.org/wiki/Machine_learning")

    print(f"Title: {content['title']}")
    print(f"Text length: {len(content['text'])} characters")
    print(f"Number of links: {len(content['links'])}")
    print(f"\nFirst 200 characters:")
    print(content['text'][:200])

except Exception as e:
    print(f"Error: {e}")


# Example 4.2: Extract specific information
print("\n4.2 Analyzing scraped content:")
try:
    content = scraper.scrape("https://en.wikipedia.org/wiki/Python_(programming_language)")

    # Count occurrences of "Python"
    python_count = content['text'].lower().count('python')
    print(f"'Python' mentioned {python_count} times")

    # Find external links (simplified)
    external_links = [
        link for link in content['links']
        if link.startswith('http') and 'wikipedia' not in link
    ]
    print(f"External links: {len(external_links)}")

    if external_links:
        print("Sample external links:")
        for link in external_links[:3]:
            print(f"  - {link}")

except Exception as e:
    print(f"Error: {e}")


# ============================================================================
# Part 5: Search and Scrape Combined
# ============================================================================

print("\n" + "=" * 80)
print("Part 5: Search and Scrape Combined")
print("=" * 80)

from llmkit.web_search import WebSearch, SearchEngine

# Example 5.1: Search then scrape top results
print("\n5.1 Search and scrape top 2 results:")

web = WebSearch(
    google_api_key=google_api_key,
    google_search_engine_id=google_search_engine_id,
    bing_api_key=bing_api_key,
    default_engine=SearchEngine.DUCKDUCKGO,
    max_results=5
)

try:
    # Search and scrape in one call
    scraped = web.search_and_scrape(
        "transformers architecture",
        max_scrape=2
    )

    for i, item in enumerate(scraped, 1):
        search_result = item['search_result']
        content = item['content']

        print(f"\nResult {i}:")
        print(f"Title: {search_result.title}")
        print(f"URL: {search_result.url}")
        print(f"Scraped text length: {len(content['text'])} chars")
        print(f"Preview: {content['text'][:150]}...")

except Exception as e:
    print(f"Error: {e}")


# Example 5.2: Async search and scrape (faster!)
async def async_search_scrape_example():
    print("\n5.2 Async search and scrape:")

    web = WebSearch(
        default_engine=SearchEngine.DUCKDUCKGO,
        max_results=5
    )

    start_time = time.time()

    try:
        scraped = await web.search_and_scrape_async(
            "deep learning",
            max_scrape=3
        )

        elapsed = time.time() - start_time

        print(f"Scraped {len(scraped)} results in {elapsed:.2f}s")
        for i, item in enumerate(scraped, 1):
            content = item['content']
            print(f"  {i}. {item['search_result'].title[:50]}")
            print(f"     Text: {len(content['text'])} chars")

    except Exception as e:
        print(f"Error: {e}")

# asyncio.run(async_search_scrape_example())


# ============================================================================
# Part 6: Real-Time Information Retrieval
# ============================================================================

print("\n" + "=" * 80)
print("Part 6: Real-Time Information Retrieval")
print("=" * 80)

# Example 6.1: Get current information
print("\n6.1 Real-time search (current events):")
ddg = DuckDuckGoSearch(max_results=3)

queries = [
    "latest AI news",
    "stock market today",
    "weather New York"
]

for query in queries:
    try:
        results = ddg.search(query)
        print(f"\nQuery: '{query}'")
        if results.results:
            top = results.results[0]
            print(f"  Top result: {top.title[:60]}")
            print(f"  {top.snippet[:100]}...")
    except Exception as e:
        print(f"  Error: {e}")


# Example 6.2: Temporal queries
print("\n6.2 Temporal queries:")
time_sensitive_queries = [
    "python 3.12 release date",
    "current bitcoin price",
    "2024 olympics"
]

for query in time_sensitive_queries:
    try:
        results = ddg.search(query, safe_search="moderate")
        print(f"\n'{query}':")
        if results.results:
            print(f"  → {results.results[0].snippet[:80]}...")
    except Exception as e:
        print(f"  Error: {e}")


# ============================================================================
# Part 7: Integration with RAG
# ============================================================================

print("\n" + "=" * 80)
print("Part 7: Integration with RAG (Real-Time RAG)")
print("=" * 80)

# Example 7.1: Web-augmented RAG
print("\n7.1 Web-augmented RAG pattern:")

def web_augmented_rag(query: str, llm_model=None):
    """
    RAG with real-time web search

    Pattern:
    1. Search web for recent information
    2. Scrape top results
    3. Use as context for LLM
    """
    print(f"\nQuery: {query}")

    # Step 1: Search
    print("  [1] Searching web...")
    web = WebSearch(default_engine=SearchEngine.DUCKDUCKGO)
    search_results = web.search(query, max_results=3)
    print(f"      Found {len(search_results.results)} results")

    # Step 2: Scrape
    print("  [2] Scraping content...")
    scraped = web.search_and_scrape(query, max_scrape=2)

    # Step 3: Extract text
    context_parts = []
    for item in scraped:
        result = item['search_result']
        content = item['content']

        # Use title + snippet + scraped text (first 500 chars)
        context_parts.append(f"""
Source: {result.title}
URL: {result.url}
Content: {content['text'][:500]}
        """)

    context = "\n---\n".join(context_parts)

    # Step 4: Prompt for LLM
    prompt = f"""Based on the following web search results, answer the question:

Question: {query}

Search Results:
{context}

Answer:"""

    print("  [3] Context prepared for LLM")
    print(f"      Context length: {len(context)} chars")

    # In production, you would call LLM here:
    # response = llm_model.generate(prompt)

    return context

# Run example
try:
    context = web_augmented_rag("What are the latest developments in GPT-4?")
    print("\nSample context for LLM:")
    print(context[:300] + "...")
except Exception as e:
    print(f"Error: {e}")


# Example 7.2: Fact-checking with web search
print("\n\n7.2 Fact-checking pattern:")

def fact_check(claim: str) -> dict:
    """
    Verify a claim using web search

    Returns:
        {
            'claim': str,
            'evidence': List[str],
            'sources': List[str]
        }
    """
    print(f"\nClaim: \"{claim}\"")

    # Search for the claim
    web = WebSearch(default_engine=SearchEngine.DUCKDUCKGO)
    results = web.search(claim, max_results=5)

    evidence = []
    sources = []

    for result in results.results[:3]:
        evidence.append(result.snippet)
        sources.append(f"{result.title} - {result.url}")

    print(f"  Found {len(evidence)} pieces of evidence")

    return {
        'claim': claim,
        'evidence': evidence,
        'sources': sources
    }

try:
    result = fact_check("Python was created by Guido van Rossum")
    print("\nEvidence:")
    for i, ev in enumerate(result['evidence'], 1):
        print(f"  {i}. {ev[:80]}...")
except Exception as e:
    print(f"Error: {e}")


# ============================================================================
# Part 8: Advanced Search Patterns
# ============================================================================

print("\n" + "=" * 80)
print("Part 8: Advanced Search Patterns")
print("=" * 80)

# Example 8.1: Multi-engine search
print("\n8.1 Multi-engine search (ensemble):")

async def multi_engine_search(query: str):
    """Search across multiple engines and combine results"""
    print(f"\nQuery: '{query}'")

    web = WebSearch(
        google_api_key=google_api_key,
        google_search_engine_id=google_search_engine_id,
        bing_api_key=bing_api_key,
        default_engine=SearchEngine.DUCKDUCKGO
    )

    # Search multiple engines in parallel
    tasks = []

    # DuckDuckGo (always available)
    tasks.append(web.search_async(query, engine=SearchEngine.DUCKDUCKGO))

    # Google (if configured)
    if google_api_key and google_search_engine_id:
        tasks.append(web.search_async(query, engine=SearchEngine.GOOGLE))

    # Bing (if configured)
    if bing_api_key:
        tasks.append(web.search_async(query, engine=SearchEngine.BING))

    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    # Combine results
    all_results = []
    for response in results_list:
        if isinstance(response, Exception):
            continue
        all_results.extend(response.results)

    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for result in all_results:
        if result.url not in seen_urls:
            seen_urls.add(result.url)
            unique_results.append(result)

    print(f"  Total results from all engines: {len(unique_results)}")
    print(f"  Engines used: {len([r for r in results_list if not isinstance(r, Exception)])}")

    return unique_results

# asyncio.run(multi_engine_search("quantum machine learning"))


# Example 8.2: Search result ranking
print("\n8.2 Re-ranking search results:")

def rerank_by_relevance(query: str, results: List, method="keyword_count"):
    """
    Re-rank results by relevance

    Methods:
    - keyword_count: Count query terms in snippet
    - length: Prefer longer snippets (more info)
    """
    query_terms = set(query.lower().split())

    for result in results:
        if method == "keyword_count":
            # Count how many query terms appear in snippet
            snippet_lower = result.snippet.lower()
            matches = sum(1 for term in query_terms if term in snippet_lower)
            result.score = matches

        elif method == "length":
            result.score = len(result.snippet)

    # Sort by score (descending)
    ranked = sorted(results, key=lambda r: r.score, reverse=True)
    return ranked

# Example usage
try:
    query = "machine learning tutorial"
    results = ddg.search(query, max_results=5)

    print(f"\nOriginal order:")
    for i, r in enumerate(results.results[:3], 1):
        print(f"  {i}. {r.title[:50]}")

    # Re-rank
    reranked = rerank_by_relevance(query, results.results, method="keyword_count")

    print(f"\nRe-ranked by keyword count:")
    for i, r in enumerate(reranked[:3], 1):
        print(f"  {i}. {r.title[:50]} (score: {r.score})")

except Exception as e:
    print(f"Error: {e}")


# Example 8.3: Search with filters
print("\n8.3 Search with domain filter:")

def search_specific_domain(query: str, domain: str):
    """
    Search within a specific domain

    Example:
        search_specific_domain("python tutorial", "stackoverflow.com")
    """
    # Add site: operator to query
    filtered_query = f"site:{domain} {query}"

    web = WebSearch(default_engine=SearchEngine.DUCKDUCKGO)
    results = web.search(filtered_query, max_results=5)

    return results

try:
    results = search_specific_domain("numpy array", "stackoverflow.com")
    print(f"\nResults from stackoverflow.com:")
    for r in results.results[:3]:
        print(f"  - {r.title[:60]}")
except Exception as e:
    print(f"Error: {e}")


# ============================================================================
# Part 9: Performance Benchmarking
# ============================================================================

print("\n" + "=" * 80)
print("Part 9: Performance Benchmarking")
print("=" * 80)

# Example 9.1: Compare sync vs async
print("\n9.1 Sync vs Async comparison:")

async def benchmark_async_vs_sync():
    web = WebSearch(default_engine=SearchEngine.DUCKDUCKGO)
    queries = ["AI", "ML", "DL", "NLP", "CV"]

    # Sync
    start = time.time()
    for query in queries:
        web.search(query, max_results=3)
    sync_time = time.time() - start

    # Async
    start = time.time()
    tasks = [web.search_async(q, max_results=3) for q in queries]
    await asyncio.gather(*tasks)
    async_time = time.time() - start

    print(f"  Sync: {sync_time:.2f}s")
    print(f"  Async: {async_time:.2f}s")
    print(f"  Speedup: {sync_time / async_time:.1f}x")

# asyncio.run(benchmark_async_vs_sync())


# Example 9.2: Cache effectiveness
print("\n9.2 Cache effectiveness:")
ddg = DuckDuckGoSearch(max_results=5, cache_ttl=60)

query = "python programming"

# First call (cache miss)
start = time.time()
results1 = ddg.search(query)
time1 = time.time() - start

# Second call (cache hit)
start = time.time()
results2 = ddg.search(query)
time2 = time.time() - start

print(f"  First call (cache miss): {time1:.3f}s")
print(f"  Second call (cache hit): {time2:.3f}s")
print(f"  Speedup: {time1 / time2:.1f}x")


# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("TUTORIAL SUMMARY")
print("=" * 80)

summary = """
이 튜토리얼에서 다룬 내용:

1. DuckDuckGo Search
   - API 키 불필요!
   - Region-specific search
   - Privacy-focused

2. Google Custom Search
   - API 키 필요
   - High-quality results
   - Language filtering

3. Bing Search
   - Azure API 키 필요
   - Market-specific results
   - Published dates

4. Web Scraping
   - BeautifulSoup로 HTML 파싱
   - Text extraction
   - Link extraction

5. Search and Scrape Combined
   - One-step search + scrape
   - Async for performance

6. Real-Time Information
   - Current events
   - Temporal queries
   - Fresh data

7. RAG Integration
   - Web-augmented RAG
   - Fact-checking
   - Real-time context

8. Advanced Patterns
   - Multi-engine search
   - Result re-ranking
   - Domain filtering

9. Performance
   - Async speedup
   - Caching effectiveness

다음 단계:
- API 키 설정하여 Google/Bing 테스트
- 실제 RAG 시스템에 통합
- 프로덕션 배포 (rate limiting 고려)
"""

print(summary)

print("\n" + "=" * 80)
print("튜토리얼 완료!")
print("=" * 80)
