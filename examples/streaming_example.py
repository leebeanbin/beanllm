"""
Real-time Streaming Example

WebSocket을 사용한 실시간 진행 상황 스트리밍 예제
"""

import asyncio
from beanllm import Client
from beanllm.facade.advanced.knowledge_graph_facade import KnowledgeGraph
from beanllm.infrastructure.streaming import (
    WebSocketServer,
    get_websocket_server,
    ProgressTracker,
)


# ============================================================================
# Server Example: Knowledge Graph Building with Progress Streaming
# ============================================================================


async def build_knowledge_graph_with_streaming(
    documents: list,
    websocket_session,
):
    """
    WebSocket 스트리밍과 함께 Knowledge Graph 구축

    Args:
        documents: 문서 리스트
        websocket_session: WebSocket 세션
    """
    # Progress tracker 생성
    tracker = ProgressTracker(
        task_id="kg_build",
        total_steps=len(documents) * 2,  # entities + relations per document
        websocket_session=websocket_session,
        stage="Building Knowledge Graph",
    )

    try:
        await tracker.start(message="Starting knowledge graph construction")

        # Initialize KG
        client = Client(provider="openai", api_key="your-api-key")
        kg = KnowledgeGraph(client=client)

        # Process each document
        for i, doc in enumerate(documents):
            # Extract entities
            await tracker.update(
                message=f"Extracting entities from document {i+1}/{len(documents)}",
                metadata={"document_id": i},
            )

            entities = await kg.extract_entities(text=doc)

            # Extract relations
            await tracker.update(
                message=f"Extracting relations from document {i+1}/{len(documents)}",
                metadata={"document_id": i, "num_entities": len(entities.entities)},
            )

            # Simulate some processing time
            await asyncio.sleep(0.5)

        # Build the graph
        await tracker.update(
            message="Building final graph...",
            metadata={"phase": "graph_construction"},
        )

        response = await kg.build_graph(
            documents=documents,
            graph_id="streaming_example",
        )

        # Complete
        await tracker.complete(
            message="Knowledge graph built successfully",
            result={
                "num_nodes": response.num_nodes,
                "num_edges": response.num_edges,
                "graph_id": response.graph_id,
            },
        )

        return response

    except Exception as e:
        await tracker.error(
            error_message=str(e),
            details={"error_type": type(e).__name__},
        )
        raise


async def run_server_example():
    """서버 예제 실행"""
    print("=" * 70)
    print("WebSocket Streaming Server Example")
    print("=" * 70)

    # Start WebSocket server
    server = get_websocket_server(host="localhost", port=8765)
    await server.start()

    print(f"\n✅ Server started: ws://localhost:8765")
    print("📡 Waiting for client connections...")
    print("\nPress Ctrl+C to stop\n")

    try:
        # Wait for connections and handle tasks
        while True:
            await asyncio.sleep(1)

            # Check for active sessions
            active_sessions = server.get_active_sessions()
            if active_sessions:
                print(f"Active sessions: {len(active_sessions)}")

            # Simulate task execution for connected clients
            for session_id in active_sessions:
                session = server.get_session(session_id)
                if session and not hasattr(session, "_task_started"):
                    # Mark as started
                    session._task_started = True

                    # Run example task
                    documents = [
                        "Apple was founded by Steve Jobs.",
                        "Microsoft was founded by Bill Gates.",
                        "Google was founded by Larry Page and Sergey Brin.",
                    ]

                    asyncio.create_task(
                        build_knowledge_graph_with_streaming(documents, session)
                    )

    except KeyboardInterrupt:
        print("\n\nStopping server...")
        await server.stop()
        print("✅ Server stopped")


# ============================================================================
# Client Example: Connecting and Receiving Updates
# ============================================================================


async def run_client_example():
    """클라이언트 예제 실행"""
    try:
        import websockets
    except ImportError:
        print("❌ websockets library required for client")
        print("   Install with: pip install websockets")
        return

    print("=" * 70)
    print("WebSocket Client Example")
    print("=" * 70)

    uri = "ws://localhost:8765"
    print(f"\n📡 Connecting to {uri}...")

    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected!\n")

            # Receive messages
            async for message in websocket:
                import json

                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "connected":
                    print(f"🔗 {data['data']['message']}")
                    print(f"   Session ID: {data['session_id']}\n")

                elif msg_type == "progress":
                    progress_data = data["data"]
                    percentage = progress_data.get("percentage", 0)
                    message_text = progress_data.get("message", "")
                    current = progress_data.get("current", 0)
                    total = progress_data.get("total", 0)

                    # Progress bar
                    bar_length = 40
                    filled = int(bar_length * percentage / 100)
                    bar = "█" * filled + "░" * (bar_length - filled)

                    print(f"\r[{bar}] {percentage:.1f}% | {message_text}", end="")

                elif msg_type == "result":
                    result_data = data["data"]
                    print(f"\n\n📊 Result received:")
                    for key, value in result_data.items():
                        print(f"   {key}: {value}")

                elif msg_type == "complete":
                    final_data = data["data"]
                    print(f"\n\n✅ Task completed!")
                    print(f"   {final_data.get('message', 'Done')}")
                    if "elapsed_time" in final_data:
                        print(f"   Elapsed time: {final_data['elapsed_time']:.2f}s")
                    break

                elif msg_type == "error":
                    error_data = data["data"]
                    print(f"\n\n❌ Error: {error_data.get('error')}")
                    break

    except websockets.exceptions.ConnectionClosed:
        print("\n\n🔌 Connection closed")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")


# ============================================================================
# Main
# ============================================================================


async def main():
    """메인 함수"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "client":
        # Run client
        await run_client_example()
    else:
        # Run server
        await run_server_example()


if __name__ == "__main__":
    print("""
    Usage:
        python streaming_example.py          # Run server
        python streaming_example.py client   # Run client (in another terminal)
    """)

    asyncio.run(main())
