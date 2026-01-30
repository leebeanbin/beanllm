# 오픈소스 MCP 통합

## 🎯 목표

오픈소스 MCP 서버를 활용하여 기능 확장

---

## 📊 현재 MCP 통합 상태

- ✅ FastMCP 기반 MCP 서버 구현됨
- ✅ beanllm 기능을 MCP tool로 wrapping
- ✅ 세션 관리, RAG, Multi-Agent, KG, ML tools 지원

---

## ✅ 오픈소스 MCP 서버 옵션

### 1. 파일 시스템 MCP
- **패키지**: `@modelcontextprotocol/server-filesystem`
- **용도**: 파일 읽기/쓰기, 디렉토리 탐색
- **활용**: 문서 자동 로드, 파일 관리

### 2. PostgreSQL MCP
- **패키지**: `@modelcontextprotocol/server-postgres`
- **용도**: PostgreSQL 데이터베이스 쿼리
- **활용**: pgvector 벡터 스토어 연동

### 3. GitHub MCP
- **패키지**: `@modelcontextprotocol/server-github`
- **용도**: GitHub API 연동
- **활용**: 코드 검색, 이슈 관리

### 4. 브라우저 MCP
- **패키지**: `@modelcontextprotocol/server-puppeteer`
- **용도**: 웹 브라우저 자동화
- **활용**: 웹 검색, 스크래핑

---

## ✅ 통합 전략

### 옵션 1: MCP 서버 통합 (권장)

```python
# playground/backend/services/mcp_integration_service.py
class MCPIntegrationService:
    """오픈소스 MCP 서버 통합 서비스"""
    
    async def call_mcp_server(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """MCP 서버 호출"""
        # MCP 클라이언트로 외부 서버 호출
        pass
```

### 옵션 2: 직접 통합

```python
# 파일 시스템 MCP 직접 통합
from mcp import ClientSession, StdioServerParameters

async def read_file_via_mcp(file_path: str) -> str:
    """MCP를 통해 파일 읽기"""
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/path"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            result = await session.call_tool("read_file", {"path": file_path})
            return result.content[0].text
```

---

## 📋 구현 체크리스트 및 상태

### ✅ 구현됨
- [x] FastMCP 기반 MCP 서버 (`mcp_server/`)
- [x] MCP Client Service (`mcp_client_service.py`)
- [x] beanllm 기능을 MCP tool로 wrapping (33개 tools)

### ❌ 미구현
- [ ] **MCPIntegrationService 생성**
  - **파일**: `playground/backend/services/mcp_integration_service.py` (신규 생성 필요)
  - **구현 방향**:
    1. 외부 MCP 서버와 통신하는 클라이언트
    2. 여러 MCP 서버를 관리하는 레지스트리
    3. MCP 서버별 tool 목록 조회
  - **방법**: 문서의 "옵션 1: MCP 서버 통합" 섹션 참조
  - **의존성**: `mcp` Python 패키지 추가 필요
- [ ] **파일 시스템 MCP 통합**
  - **통합 위치**: `MCPIntegrationService` 또는 별도 서비스
  - **구현 방향**:
    1. `@modelcontextprotocol/server-filesystem` 서버 실행
    2. 파일 읽기/쓰기 tool 호출
    3. 문서 자동 로드 기능에 활용
  - **방법**: 문서의 "옵션 2: 직접 통합" 섹션 참조
- [ ] **PostgreSQL MCP 통합 (선택)**
  - **통합 위치**: `MCPIntegrationService`
  - **구현 방향**: pgvector 벡터 스토어 연동
  - **의존성**: `@modelcontextprotocol/server-postgres` 패키지
- [ ] **GitHub MCP 통합 (선택)**
  - **통합 위치**: `MCPIntegrationService`
  - **구현 방향**: GitHub API를 통한 코드 검색, 이슈 관리
  - **의존성**: `@modelcontextprotocol/server-github` 패키지

---

## 🎯 우선순위

**낮음**: 기능 확장 (선택적)
