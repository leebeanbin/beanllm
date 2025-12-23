"""
Web Scraper - 웹 페이지 콘텐츠 추출기
"""

from typing import Any, Dict

import httpx
import requests
from bs4 import BeautifulSoup


class WebScraper:
    """
    웹 페이지 콘텐츠 추출기

    BeautifulSoup을 사용하여 HTML에서 텍스트 추출
    """

    @staticmethod
    def scrape(url: str, timeout: int = 10) -> Dict[str, Any]:
        """
        URL에서 콘텐츠 추출

        Args:
            url: 대상 URL
            timeout: 타임아웃 (초)

        Returns:
            {
                'title': str,
                'text': str,
                'links': List[str],
                'metadata': dict
            }
        """
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get title
            title = soup.find("title")
            title_text = title.string if title else ""

            # Get text
            text = soup.get_text(separator="\n", strip=True)

            # Get links
            links = [a.get("href") for a in soup.find_all("a", href=True)]

            return {
                "title": title_text,
                "text": text,
                "links": links,
                "metadata": {
                    "url": url,
                    "status_code": response.status_code,
                    "content_type": response.headers.get("Content-Type", ""),
                },
            }

        except Exception as e:
            return {"title": "", "text": "", "links": [], "metadata": {"error": str(e)}}

    @staticmethod
    async def scrape_async(url: str, timeout: int = 10) -> Dict[str, Any]:
        """비동기 스크래핑"""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, "html.parser")

                for script in soup(["script", "style"]):
                    script.decompose()

                title = soup.find("title")
                title_text = title.string if title else ""

                text = soup.get_text(separator="\n", strip=True)
                links = [a.get("href") for a in soup.find_all("a", href=True)]

                return {
                    "title": title_text,
                    "text": text,
                    "links": links,
                    "metadata": {
                        "url": url,
                        "status_code": response.status_code,
                        "content_type": response.headers.get("Content-Type", ""),
                    },
                }

        except Exception as e:
            return {"title": "", "text": "", "links": [], "metadata": {"error": str(e)}}
