"""
Kafka 클라이언트 관리

환경변수에서 Kafka 설정을 읽어 클라이언트를 생성합니다.
기존 최적화 패턴 참고: 에러 처리, 로깅, 연결 풀링
"""

import os
from typing import Any, Optional

try:
    from kafka import KafkaConsumer, KafkaProducer
    from kafka.admin import KafkaAdminClient
except ImportError:
    KafkaProducer = None
    KafkaConsumer = None
    KafkaAdminClient = None

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)

_kafka_producer: Optional[Any] = None
_kafka_consumer: Optional[Any] = None


def get_kafka_client():
    """
    Kafka 클라이언트 반환

    환경변수:
        KAFKA_BOOTSTRAP_SERVERS: Kafka 브로커 주소 (쉼표 구분, 기본: localhost:9092)
        KAFKA_CLIENT_ID: Kafka 클라이언트 ID (기본: beanllm)

    Returns:
        (KafkaProducer, KafkaConsumer) 튜플

    Raises:
        ImportError: kafka-python 패키지가 설치되지 않음
    """
    global _kafka_producer, _kafka_consumer

    if KafkaProducer is None:
        raise ImportError(
            "kafka-python package is required for distributed mode. "
            "Install it with: pip install kafka-python"
        )

    if _kafka_producer is None:
        bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(",")
        client_id = os.getenv("KAFKA_CLIENT_ID", "beanllm")
        request_timeout_ms = int(os.getenv("KAFKA_REQUEST_TIMEOUT_MS", "30000"))
        max_block_ms = int(os.getenv("KAFKA_MAX_BLOCK_MS", "60000"))

        try:
            _kafka_producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                client_id=client_id,
                value_serializer=lambda v: v.encode("utf-8") if isinstance(v, str) else v,
                key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
                request_timeout_ms=request_timeout_ms,
                max_block_ms=max_block_ms,
                retries=3,  # 재시도 횟수
                acks="all",  # 모든 replica 확인 (내구성)
            )

            _kafka_consumer = KafkaConsumer(
                bootstrap_servers=bootstrap_servers,
                client_id=client_id,
                value_deserializer=lambda v: v.decode("utf-8") if isinstance(v, bytes) else v,
                key_deserializer=lambda k: k.decode("utf-8") if isinstance(k, bytes) else k,
                consumer_timeout_ms=1000,  # 폴링 타임아웃
                enable_auto_commit=True,  # 자동 커밋
                auto_commit_interval_ms=5000,  # 5초마다 커밋
            )
            logger.info(f"Kafka client initialized: {bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka client: {e}")
            # 연결 실패해도 클라이언트는 생성 (fallback에서 처리)
            _kafka_producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                client_id=client_id,
                value_serializer=lambda v: v.encode("utf-8") if isinstance(v, str) else v,
                key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
            )
            _kafka_consumer = KafkaConsumer(
                bootstrap_servers=bootstrap_servers,
                client_id=client_id,
                value_deserializer=lambda v: v.decode("utf-8") if isinstance(v, bytes) else v,
                key_deserializer=lambda k: k.decode("utf-8") if isinstance(k, bytes) else k,
            )

    return _kafka_producer, _kafka_consumer


def close_kafka_client():
    """Kafka 클라이언트 종료"""
    global _kafka_producer, _kafka_consumer
    if _kafka_producer:
        _kafka_producer.close()
        _kafka_producer = None
    if _kafka_consumer:
        _kafka_consumer.close()
        _kafka_consumer = None
