"""
Google Workspace Tools - 기존 Google API 연동을 MCP tool로 wrapping

🎯 핵심: 새로운 코드를 만들지 않고 기존 코드를 함수화!
"""

import asyncio
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

# 기존 Google API 코드 import
from beanllm.infrastructure.distributed.google_events import (
    get_google_export_stats,
    log_admin_action,
    log_google_export,
)
from mcp_server.config import MCPServerConfig

# FastMCP 인스턴스 생성
mcp = FastMCP("Google Workspace Tools")


@mcp.tool()
async def export_to_google_docs(
    title: str,
    user_id: str,
    access_token: str,
    session_id: Optional[str] = None,
    content: Optional[str] = None,  # content가 없으면 session_id에서 가져옴
) -> dict:
    """
    채팅 내역을 Google Docs로 내보내기 (세션 메시지 자동 가져오기)

    Args:
        title: 문서 제목
        user_id: 사용자 ID
        access_token: Google OAuth 2.0 액세스 토큰
        session_id: 세션 ID (content가 없으면 이 세션의 메시지를 사용)
        content: 문서 내용 (선택, session_id가 있으면 무시됨)

    Returns:
        dict: 생성된 문서 ID, URL

    Example:
        User: "이 채팅 내역을 Google Docs로 저장해줘"
        → export_to_google_docs(
            title="My Chat History",
            user_id="user123",
            access_token="ya29.a0...",
            session_id="session_abc123"  # 세션 메시지 자동 가져오기
        )
    """
    try:
        # ✅ session_id가 있으면 MongoDB에서 메시지 가져오기
        if session_id and not content:
            from mcp_server.services.session_manager import get_session_manager

            session_manager = get_session_manager()
            messages = await session_manager.get_session_messages(session_id)

            if not messages:
                return {
                    "success": False,
                    "error": f"Session {session_id} not found or has no messages",
                }

            # 메시지를 마크다운으로 변환
            content = f"# {title}\n\n"
            for msg in messages:
                role = msg.get("role", "unknown")
                msg_content = msg.get("content", "")
                timestamp = msg.get("timestamp", "")
                content += f"## {role.capitalize()}\n"
                if timestamp:
                    content += f"*{timestamp}*\n\n"
                content += f"{msg_content}\n\n"
        elif not content:
            return {
                "success": False,
                "error": "Either content or session_id must be provided",
            }

        # 🎯 기존 Google Docs API 사용!
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        credentials = Credentials(token=access_token)
        docs_service = build("docs", "v1", credentials=credentials)

        # 1. 새 문서 생성
        document = docs_service.documents().create(body={"title": title}).execute()
        doc_id = document["documentId"]

        # 2. 내용 삽입
        requests = [
            {
                "insertText": {
                    "location": {"index": 1},
                    "text": content,
                }
            }
        ]

        docs_service.documents().batchUpdate(
            documentId=doc_id, body={"requests": requests}
        ).execute()

        # 3. 이벤트 로깅 (관리자 모니터링용)
        await log_google_export(
            user_id=user_id,
            export_type="docs",
            metadata={
                "doc_id": doc_id,
                "title": title,
                "content_length": len(content),
                "message_count": len(messages) if session_id else None,
            },
            session_id=session_id,
        )

        doc_url = f"https://docs.google.com/document/d/{doc_id}/edit"

        return {
            "success": True,
            "doc_id": doc_id,
            "doc_url": doc_url,
            "title": title,
            "message_count": len(messages) if session_id else None,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def save_to_google_drive(
    filename: str,
    user_id: str,
    access_token: str,
    folder_id: Optional[str] = None,
    session_id: Optional[str] = None,
    content: Optional[str] = None,  # content가 없으면 session_id에서 가져옴
) -> dict:
    """
    채팅 내역을 Google Drive에 텍스트 파일로 저장 (세션 메시지 자동 가져오기)

    Args:
        filename: 파일명
        user_id: 사용자 ID
        access_token: Google OAuth 2.0 액세스 토큰
        folder_id: 저장할 폴더 ID (None이면 루트)
        session_id: 세션 ID (content가 없으면 이 세션의 메시지를 사용)
        content: 파일 내용 (선택, session_id가 있으면 무시됨)

    Returns:
        dict: 생성된 파일 ID, URL

    Example:
        User: "이 채팅을 Drive에 저장해줘"
        → save_to_google_drive(
            filename="chat_history.txt",
            user_id="user123",
            access_token="ya29.a0...",
            session_id="session_abc123"  # 세션 메시지 자동 가져오기
        )
    """
    try:
        # ✅ session_id가 있으면 MongoDB에서 메시지 가져오기
        if session_id and not content:
            from mcp_server.services.session_manager import get_session_manager

            session_manager = get_session_manager()
            messages = await session_manager.get_session_messages(session_id)

            if not messages:
                return {
                    "success": False,
                    "error": f"Session {session_id} not found or has no messages",
                }

            # 메시지를 텍스트로 변환
            content = "beanllm Chat History\n"
            content += f"Session ID: {session_id}\n"
            content += "=" * 60 + "\n\n"

            for msg in messages:
                role = msg.get("role", "unknown")
                msg_content = msg.get("content", "")
                timestamp = msg.get("timestamp", "")
                content += f"{role.upper()}:\n"
                if timestamp:
                    content += f"[{timestamp}]\n"
                content += f"{msg_content}\n\n"
        elif not content:
            return {
                "success": False,
                "error": "Either content or session_id must be provided",
            }

        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaInMemoryUpload

        credentials = Credentials(token=access_token)
        drive_service = build("drive", "v3", credentials=credentials)

        # 1. 파일 메타데이터
        file_metadata = {"name": filename}
        if folder_id:
            file_metadata["parents"] = [folder_id]

        # 2. 파일 업로드
        media = MediaInMemoryUpload(content.encode("utf-8"), mimetype="text/plain")

        file = (
            drive_service.files()
            .create(body=file_metadata, media_body=media, fields="id,webViewLink")
            .execute()
        )

        file_id = file["id"]
        file_url = file["webViewLink"]

        # 3. 이벤트 로깅
        await log_google_export(
            user_id=user_id,
            export_type="drive",
            metadata={
                "file_id": file_id,
                "filename": filename,
                "content_length": len(content),
                "folder_id": folder_id,
                "message_count": len(messages) if session_id else None,
            },
            session_id=session_id,
        )

        return {
            "success": True,
            "file_id": file_id,
            "file_url": file_url,
            "filename": filename,
            "message_count": len(messages) if session_id else None,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def share_via_gmail(
    recipient_email: str,
    subject: str,
    user_id: str,
    access_token: str,
    session_id: Optional[str] = None,
    content: Optional[str] = None,  # content가 없으면 session_id에서 가져옴
    message: Optional[str] = None,  # 추가 메시지 (선택)
) -> dict:
    """
    채팅 내역을 Gmail로 공유 (세션 메시지 자동 가져오기)

    Args:
        recipient_email: 수신자 이메일
        subject: 이메일 제목
        user_id: 사용자 ID
        access_token: Google OAuth 2.0 액세스 토큰
        session_id: 세션 ID (content가 없으면 이 세션의 메시지를 사용)
        content: 이메일 본문 (선택, session_id가 있으면 무시됨)
        message: 추가 메시지 (선택, 세션 메시지 앞에 추가)

    Returns:
        dict: 전송된 메시지 ID

    Example:
        User: "이 채팅을 friend@example.com에게 보내줘"
        → share_via_gmail(
            recipient_email="friend@example.com",
            subject="My Chat History",
            user_id="user123",
            access_token="ya29.a0...",
            session_id="session_abc123"  # 세션 메시지 자동 가져오기
        )
    """
    try:
        # ✅ session_id가 있으면 MongoDB에서 메시지 가져오기
        if session_id and not content:
            from mcp_server.services.session_manager import get_session_manager

            session_manager = get_session_manager()
            messages = await session_manager.get_session_messages(session_id)

            if not messages:
                return {
                    "success": False,
                    "error": f"Session {session_id} not found or has no messages",
                }

            # 메시지를 이메일 본문으로 변환
            content = message or "Here is my beanllm chat history:\n\n"
            content += "=" * 60 + "\n\n"

            for msg in messages:
                role = msg.get("role", "unknown")
                msg_content = msg.get("content", "")
                timestamp = msg.get("timestamp", "")
                content += f"{role.upper()}:\n"
                if timestamp:
                    content += f"[{timestamp}]\n"
                content += f"{msg_content}\n\n"
        elif not content:
            return {
                "success": False,
                "error": "Either content or session_id must be provided",
            }

        import base64
        from email.mime.text import MIMEText

        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        credentials = Credentials(token=access_token)
        gmail_service = build("gmail", "v1", credentials=credentials)

        # 1. 이메일 메시지 생성
        email_message = MIMEText(content)
        email_message["to"] = recipient_email
        email_message["subject"] = subject

        # 2. Base64 인코딩
        raw_message = base64.urlsafe_b64encode(email_message.as_bytes()).decode("utf-8")

        # 3. 전송
        result = (
            gmail_service.users().messages().send(userId="me", body={"raw": raw_message}).execute()
        )

        message_id = result["id"]

        # 4. 이벤트 로깅
        await log_google_export(
            user_id=user_id,
            export_type="gmail",
            metadata={
                "message_id": message_id,
                "recipient": recipient_email,
                "subject": subject,
                "content_length": len(content),
                "message_count": len(messages) if session_id else None,
            },
            session_id=session_id,
        )

        return {
            "success": True,
            "message_id": message_id,
            "recipient": recipient_email,
            "subject": subject,
            "message_count": len(messages) if session_id else None,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def get_google_export_statistics(
    hours: int = 24,
    user_id: Optional[str] = None,
) -> dict:
    """
    Google Workspace 내보내기 통계 조회 (관리자용)

    Args:
        hours: 조회 기간 (시간)
        user_id: 특정 사용자 필터 (None이면 전체)

    Returns:
        dict: 서비스별 사용량, 상위 사용자, 시간대별 패턴

    Example:
        User: "지난 24시간 Google 내보내기 통계 보여줘"
        → get_google_export_statistics(hours=24)
    """
    try:
        # 🎯 기존 google_events.py의 get_google_export_stats() 사용!
        stats = await get_google_export_stats(hours=hours, user_id=user_id)

        return {
            "success": True,
            "period_hours": hours,
            "user_filter": user_id,
            **stats,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def list_google_drive_files(
    access_token: str,
    folder_id: Optional[str] = None,
    page_size: int = 10,
) -> dict:
    """
    Google Drive 파일 목록 조회

    Args:
        access_token: Google OAuth 2.0 액세스 토큰
        folder_id: 폴더 ID (None이면 루트)
        page_size: 페이지 크기

    Returns:
        dict: 파일 목록

    Example:
        User: "내 Drive 파일 목록 보여줘"
        → list_google_drive_files(access_token="ya29.a0...")
    """
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        credentials = Credentials(token=access_token)
        drive_service = build("drive", "v3", credentials=credentials)

        # 쿼리 구성
        query = f"'{folder_id}' in parents" if folder_id else None

        # 파일 목록 조회
        results = (
            drive_service.files()
            .list(
                pageSize=page_size,
                q=query,
                fields="files(id, name, mimeType, modifiedTime, webViewLink)",
            )
            .execute()
        )

        files = results.get("files", [])

        return {
            "success": True,
            "file_count": len(files),
            "files": [
                {
                    "id": f["id"],
                    "name": f["name"],
                    "type": f["mimeType"],
                    "modified": f["modifiedTime"],
                    "url": f["webViewLink"],
                }
                for f in files
            ],
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# ===========================================
# 데이터 읽기 기능 (RAG 학습용)
# ===========================================


@mcp.tool()
async def read_google_drive_file(
    file_id: str,
    access_token: str,
) -> dict:
    """
    Google Drive에서 파일 내용 읽기 (텍스트/PDF/문서)

    지원 형식: txt, pdf, docx, csv, json, md

    Args:
        file_id: Google Drive 파일 ID
        access_token: Google OAuth 2.0 액세스 토큰

    Returns:
        dict: 파일 내용, 메타데이터

    Example:
        User: "Drive에서 이 파일 내용 읽어서 학습시켜줘"
        → read_google_drive_file(file_id="1abc...", access_token="ya29.a0...")
    """
    try:
        import io

        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        credentials = Credentials(token=access_token)
        drive_service = build("drive", "v3", credentials=credentials)

        # 1. 파일 메타데이터 조회
        file_metadata = (
            drive_service.files().get(fileId=file_id, fields="id, name, mimeType, size").execute()
        )

        mime_type = file_metadata.get("mimeType", "")
        file_name = file_metadata.get("name", "")

        # 2. 파일 내용 읽기
        content = ""

        # Google Docs/Sheets/Slides는 export로 변환
        if mime_type == "application/vnd.google-apps.document":
            # Google Docs → 텍스트
            response = drive_service.files().export(fileId=file_id, mimeType="text/plain").execute()
            content = response.decode("utf-8")

        elif mime_type == "application/vnd.google-apps.spreadsheet":
            # Google Sheets → CSV
            response = drive_service.files().export(fileId=file_id, mimeType="text/csv").execute()
            content = response.decode("utf-8")

        elif mime_type == "application/vnd.google-apps.presentation":
            # Google Slides → 텍스트
            response = drive_service.files().export(fileId=file_id, mimeType="text/plain").execute()
            content = response.decode("utf-8")

        elif mime_type in ["text/plain", "text/csv", "text/markdown", "application/json"]:
            # 텍스트 파일 직접 다운로드
            response = drive_service.files().get_media(fileId=file_id).execute()
            content = response.decode("utf-8")

        elif mime_type == "application/pdf":
            # PDF → 텍스트 (OCR 필요 시 별도 처리)
            response = drive_service.files().get_media(fileId=file_id).execute()

            # PyPDF2 또는 pdfplumber로 텍스트 추출
            try:
                import pdfplumber

                pdf_bytes = io.BytesIO(response)
                with pdfplumber.open(pdf_bytes) as pdf:
                    content = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
            except ImportError:
                # pdfplumber 없으면 PyMuPDF 시도
                try:
                    import fitz  # PyMuPDF

                    pdf_bytes = io.BytesIO(response)
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    content = "\n\n".join(page.get_text() for page in doc)
                    doc.close()
                except ImportError:
                    return {
                        "success": False,
                        "error": "PDF processing requires pdfplumber or PyMuPDF",
                    }

        else:
            return {
                "success": False,
                "error": f"Unsupported file type: {mime_type}",
                "supported_types": [
                    "text/plain",
                    "text/csv",
                    "text/markdown",
                    "application/json",
                    "application/pdf",
                    "Google Docs",
                    "Google Sheets",
                    "Google Slides",
                ],
            }

        return {
            "success": True,
            "file_id": file_id,
            "file_name": file_name,
            "mime_type": mime_type,
            "content": content,
            "content_length": len(content),
            "ready_for_rag": True,  # RAG 인덱싱 가능 표시
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def read_google_doc(
    doc_id: str,
    access_token: str,
) -> dict:
    """
    Google Docs 문서 내용 읽기

    Args:
        doc_id: Google Docs 문서 ID
        access_token: Google OAuth 2.0 액세스 토큰

    Returns:
        dict: 문서 내용, 제목, 메타데이터

    Example:
        User: "이 Google 문서 내용 읽어서 학습시켜줘"
        → read_google_doc(doc_id="1abc...", access_token="ya29.a0...")
    """
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        credentials = Credentials(token=access_token)
        docs_service = build("docs", "v1", credentials=credentials)

        # 문서 가져오기
        document = docs_service.documents().get(documentId=doc_id).execute()

        title = document.get("title", "")
        body = document.get("body", {})
        content_elements = body.get("content", [])

        # 텍스트 추출
        def extract_text(elements):
            text_parts = []
            for element in elements:
                if "paragraph" in element:
                    para = element["paragraph"]
                    for elem in para.get("elements", []):
                        if "textRun" in elem:
                            text_parts.append(elem["textRun"].get("content", ""))
                elif "table" in element:
                    # 테이블 내용 추출
                    table = element["table"]
                    for row in table.get("tableRows", []):
                        row_texts = []
                        for cell in row.get("tableCells", []):
                            cell_content = cell.get("content", [])
                            cell_text = extract_text(cell_content)
                            row_texts.append(cell_text.strip())
                        text_parts.append(" | ".join(row_texts))
            return "".join(text_parts)

        content = extract_text(content_elements)

        return {
            "success": True,
            "doc_id": doc_id,
            "title": title,
            "content": content,
            "content_length": len(content),
            "ready_for_rag": True,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def read_google_sheet(
    spreadsheet_id: str,
    access_token: str,
    sheet_name: Optional[str] = None,
    range_notation: Optional[str] = None,
) -> dict:
    """
    Google Sheets 데이터 읽기

    Args:
        spreadsheet_id: Google Sheets 스프레드시트 ID
        access_token: Google OAuth 2.0 액세스 토큰
        sheet_name: 시트 이름 (None이면 첫 번째 시트)
        range_notation: 범위 (예: "A1:D10", None이면 전체)

    Returns:
        dict: 데이터 (2D 배열), 헤더, 메타데이터

    Example:
        User: "이 스프레드시트 데이터 읽어서 학습시켜줘"
        → read_google_sheet(spreadsheet_id="1abc...", access_token="ya29.a0...")
    """
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        credentials = Credentials(token=access_token)
        sheets_service = build("sheets", "v4", credentials=credentials)

        # 스프레드시트 메타데이터 조회
        spreadsheet = sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()

        title = spreadsheet.get("properties", {}).get("title", "")
        sheets = spreadsheet.get("sheets", [])

        # 시트 선택
        if sheet_name:
            target_sheet = sheet_name
        elif sheets:
            target_sheet = sheets[0].get("properties", {}).get("title", "Sheet1")
        else:
            target_sheet = "Sheet1"

        # 범위 구성
        if range_notation:
            full_range = f"'{target_sheet}'!{range_notation}"
        else:
            full_range = target_sheet

        # 데이터 읽기
        result = (
            sheets_service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=full_range)
            .execute()
        )

        values = result.get("values", [])

        # 텍스트 변환 (RAG용)
        if values:
            headers = values[0] if values else []
            rows = values[1:] if len(values) > 1 else []

            # CSV 형식으로 변환
            content_lines = [", ".join(str(cell) for cell in headers)]
            for row in rows:
                # 각 행을 "헤더: 값" 형식으로 변환 (RAG 친화적)
                row_text = ", ".join(
                    f"{headers[i] if i < len(headers) else f'Col{i}'}: {cell}"
                    for i, cell in enumerate(row)
                )
                content_lines.append(row_text)

            content = "\n".join(content_lines)
        else:
            content = ""
            headers = []
            rows = []

        return {
            "success": True,
            "spreadsheet_id": spreadsheet_id,
            "title": title,
            "sheet_name": target_sheet,
            "headers": headers,
            "row_count": len(rows),
            "data": values,  # 원본 2D 배열
            "content": content,  # RAG용 텍스트
            "content_length": len(content),
            "ready_for_rag": True,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def import_google_data_to_rag(
    access_token: str,
    session_id: str,
    source_type: str,
    source_id: str,
    collection_name: Optional[str] = None,
    sheet_name: Optional[str] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> dict:
    """
    Google 서비스 데이터를 읽어와서 RAG에 인덱싱 (학습)

    Args:
        access_token: Google OAuth 2.0 액세스 토큰
        session_id: 세션 ID
        source_type: "drive" | "docs" | "sheets"
        source_id: 파일/문서/스프레드시트 ID
        collection_name: RAG 컬렉션 이름 (None이면 세션 ID 기반)
        sheet_name: 시트 이름 (sheets 타입일 때만)
        chunk_size: 청크 크기 (기본: 500)
        chunk_overlap: 청크 오버랩 (기본: 50)

    Returns:
        dict: 인덱싱 결과, 청크 수, 메타데이터

    Example:
        User: "내 Google Docs 문서를 학습시켜줘"
        → import_google_data_to_rag(
            access_token="ya29.a0...",
            session_id="session_abc",
            source_type="docs",
            source_id="1abc..."
        )
    """
    try:
        # 1. 데이터 읽기
        if source_type == "drive":
            read_result = await read_google_drive_file(
                file_id=source_id,
                access_token=access_token,
            )
        elif source_type == "docs":
            read_result = await read_google_doc(
                doc_id=source_id,
                access_token=access_token,
            )
        elif source_type == "sheets":
            read_result = await read_google_sheet(
                spreadsheet_id=source_id,
                access_token=access_token,
                sheet_name=sheet_name,
            )
        else:
            return {
                "success": False,
                "error": f"Unknown source_type: {source_type}",
                "supported_types": ["drive", "docs", "sheets"],
            }

        if not read_result.get("success"):
            return read_result

        content = read_result.get("content", "")
        if not content:
            return {
                "success": False,
                "error": "No content to index",
            }

        # 2. RAG에 인덱싱 (beanllm RAGChain 직접 사용)
        from beanllm.domain.loaders import Document
        from beanllm.domain.rag import RAGChain
        from mcp_server.services.session_manager import session_manager

        # 컬렉션 이름 결정
        rag_collection = collection_name or f"session_{session_id}"

        # 메타데이터 구성
        title = read_result.get("title") or read_result.get("file_name", "")
        metadata = {
            "source_type": f"google_{source_type}",
            "source_id": source_id,
            "session_id": session_id,
            "title": title,
        }

        # Document 객체 생성
        documents = [Document(page_content=content, metadata=metadata)]

        # RAG 구축
        rag = RAGChain.from_documents(
            documents=documents,
            collection_name=rag_collection,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # 세션에 RAG 인스턴스 저장
        session_manager.set_rag_instance(session_id, rag_collection, rag)

        # 청크 수 계산
        try:
            total_chunks = len(rag._vector_store._collection.get()["ids"])
        except Exception:
            total_chunks = -1  # 청크 수 계산 실패 시

        return {
            "success": True,
            "source_type": source_type,
            "source_id": source_id,
            "title": title,
            "content_length": len(content),
            "collection_name": rag_collection,
            "chunk_count": total_chunks,
            "message": f"Google {source_type} 데이터가 RAG에 학습되었습니다.",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
