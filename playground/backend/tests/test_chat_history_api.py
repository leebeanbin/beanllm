"""
Test Chat History API

Simple script to test MongoDB integration and session APIs.
"""

import asyncio
from datetime import datetime

import httpx

API_URL = "http://localhost:8000"


async def test_health():
    """Test health endpoint"""
    print("\n1️⃣  Testing /health endpoint...")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/health")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"MongoDB: {data.get('services', {}).get('mongodb', False)}")
        return response.status_code == 200


async def test_create_session():
    """Test session creation"""
    print("\n2️⃣  Testing POST /api/chat/sessions...")
    async with httpx.AsyncClient() as client:
        payload = {
            "title": f"Test Session {datetime.now().strftime('%H:%M:%S')}",
            "feature_mode": "chat",
            "model": "qwen2.5:0.5b",
            "feature_options": {},
        }
        response = await client.post(f"{API_URL}/api/chat/sessions", json=payload)
        print(f"Status: {response.status_code}")

        if response.status_code == 201:
            data = response.json()
            session_id = data["session"]["session_id"]
            print(f"✅ Created session: {session_id}")
            return session_id
        else:
            print(f"❌ Failed: {response.text}")
            return None


async def test_add_message(session_id: str):
    """Test adding messages to session"""
    print(f"\n3️⃣  Testing POST /api/chat/sessions/{session_id}/messages...")

    async with httpx.AsyncClient() as client:
        # Add user message
        user_msg = {
            "role": "user",
            "content": "Hello, this is a test message!",
            "model": "qwen2.5:0.5b",
            "metadata": {"test": True},
        }
        response = await client.post(
            f"{API_URL}/api/chat/sessions/{session_id}/messages", json=user_msg
        )
        print(f"User message - Status: {response.status_code}")

        # Add assistant message
        assistant_msg = {
            "role": "assistant",
            "content": "Hello! This is a test response.",
            "model": "qwen2.5:0.5b",
            "usage": {"total_tokens": 25},
            "metadata": {"test": True},
        }
        response = await client.post(
            f"{API_URL}/api/chat/sessions/{session_id}/messages", json=assistant_msg
        )
        print(f"Assistant message - Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Message count: {data['session']['message_count']}")
            print(f"✅ Total tokens: {data['session']['total_tokens']}")
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False


async def test_list_sessions():
    """Test listing sessions"""
    print("\n4️⃣  Testing GET /api/chat/sessions...")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/api/chat/sessions?limit=10")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Total sessions: {data['total']}")
            print(f"✅ Returned: {len(data['sessions'])} sessions")

            if data["sessions"]:
                session = data["sessions"][0]
                print("\nFirst session:")
                print(f"  - Title: {session['title']}")
                print(f"  - Messages: {session.get('message_count', 0)}")
                print(f"  - Tokens: {session.get('total_tokens', 0)}")
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False


async def test_get_session(session_id: str):
    """Test getting specific session"""
    print(f"\n5️⃣  Testing GET /api/chat/sessions/{session_id}...")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/api/chat/sessions/{session_id}")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            session = data["session"]
            print(f"✅ Title: {session['title']}")
            print(f"✅ Messages: {len(session['messages'])}")

            for i, msg in enumerate(session["messages"][:2]):  # Show first 2
                print(f"\n  Message {i+1}:")
                print(f"    Role: {msg['role']}")
                print(f"    Content: {msg['content'][:50]}...")
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False


async def test_update_title(session_id: str):
    """Test updating session title"""
    print(f"\n6️⃣  Testing PATCH /api/chat/sessions/{session_id}/title...")
    new_title = f"Updated Title {datetime.now().strftime('%H:%M:%S')}"

    async with httpx.AsyncClient() as client:
        response = await client.patch(
            f"{API_URL}/api/chat/sessions/{session_id}/title?title={new_title}"
        )
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ New title: {data['title']}")
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False


async def test_delete_session(session_id: str):
    """Test deleting session"""
    print(f"\n7️⃣  Testing DELETE /api/chat/sessions/{session_id}...")
    async with httpx.AsyncClient() as client:
        response = await client.delete(f"{API_URL}/api/chat/sessions/{session_id}")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            print("✅ Session deleted")
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False


async def main():
    print("=" * 60)
    print("Chat History API Test")
    print("=" * 60)

    try:
        # Test health
        if not await test_health():
            print("\n❌ Health check failed - is MongoDB configured?")
            print("Set MONGODB_URI in .env file")
            return

        # Create session
        session_id = await test_create_session()
        if not session_id:
            print("\n❌ Could not create session")
            return

        # Add messages
        await test_add_message(session_id)

        # List sessions
        await test_list_sessions()

        # Get specific session
        await test_get_session(session_id)

        # Update title
        await test_update_title(session_id)

        # Delete session
        await test_delete_session(session_id)

        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
