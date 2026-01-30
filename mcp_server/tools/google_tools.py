"""
Google Workspace Tools - 기존 Google API 연동을 MCP tool로 wrapping

🎯 핵심: 새로운 코드를 만들지 않고 기존 코드를 함수화!
"""
import asyncio
from typing import List, Dict, Any, Optional
from fastmcp import FastMCP

# 기존 Google API 코드 import
from beanllm.infrastructure.distributed.google_events import (
    log_google_export,
    log_admin_action,
    get_google_export_stats,
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
            content = f"beanllm Chat History\n"
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
        
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from email.mime.text import MIMEText
        import base64

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
            gmail_service.users()
            .messages()
            .send(userId="me", body={"raw": raw_message})
            .execute()
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
