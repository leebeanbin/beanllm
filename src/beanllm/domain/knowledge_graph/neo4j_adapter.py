"""
Neo4jAdapter - Neo4j 데이터베이스 연동 (Optional)
SOLID 원칙:
- SRP: Neo4j 연동만 담당
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx

from beanllm.utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jAdapter:
    """
    Neo4j 데이터베이스 어댑터 (Optional)

    Note:
        neo4j 패키지가 설치되어 있고, Neo4j 서버가 실행 중일 때만 사용 가능

    Example:
        ```python
        adapter = Neo4jAdapter(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )

        # Export graph to Neo4j
        adapter.export_graph(graph)

        # Query from Neo4j
        results = adapter.query("MATCH (n:Person) RETURN n LIMIT 10")

        # Import graph from Neo4j
        graph = adapter.import_graph()
        ```
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ) -> None:
        """
        Initialize Neo4j adapter

        Args:
            uri: Neo4j URI
            user: Username
            password: Password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None

        logger.info(f"Neo4jAdapter initialized (uri={uri})")

    def connect(self) -> None:
        """Neo4j에 연결"""
        try:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info("Connected to Neo4j")

        except ImportError:
            logger.error("neo4j package not installed. Install with: pip install neo4j")
            raise

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self) -> None:
        """연결 종료"""
        if self._driver:
            self._driver.close()
            logger.info("Disconnected from Neo4j")

    def export_graph(
        self,
        graph: nx.Graph,
        clear_existing: bool = False,
    ) -> None:
        """
        NetworkX 그래프를 Neo4j로 내보내기

        Args:
            graph: NetworkX 그래프
            clear_existing: 기존 데이터 삭제 여부
        """
        if not self._driver:
            self.connect()

        logger.info(f"Exporting graph to Neo4j: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        with self._driver.session() as session:
            # Clear existing data
            if clear_existing:
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Cleared existing Neo4j data")

            # Create nodes
            for node, data in graph.nodes(data=True):
                node_type = data.get("type", "Entity")
                properties = {
                    "id": node,
                    "name": data.get("name", ""),
                    "description": data.get("description", ""),
                    "confidence": data.get("confidence", 1.0),
                }

                # Merge properties
                if "properties" in data and isinstance(data["properties"], dict):
                    properties.update(data["properties"])

                # Create node
                query = f"MERGE (n:{node_type} {{id: $id}}) SET n += $properties"
                session.run(query, id=node, properties=properties)

            # Create relationships
            for u, v, data in graph.edges(data=True):
                rel_type = data.get("type", "RELATED_TO").upper().replace(" ", "_")

                properties = {
                    "description": data.get("description", ""),
                    "confidence": data.get("confidence", 1.0),
                }

                if "properties" in data and isinstance(data["properties"], dict):
                    properties.update(data["properties"])

                query = f"""
                MATCH (a {{id: $source_id}})
                MATCH (b {{id: $target_id}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r += $properties
                """

                session.run(query, source_id=u, target_id=v, properties=properties)

        logger.info("Graph exported to Neo4j successfully")

    def import_graph(
        self,
        directed: bool = True,
    ) -> nx.Graph:
        """
        Neo4j에서 그래프 가져오기

        Args:
            directed: 방향 그래프 여부

        Returns:
            nx.Graph: NetworkX 그래프
        """
        if not self._driver:
            self.connect()

        logger.info("Importing graph from Neo4j")

        if directed:
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()

        with self._driver.session() as session:
            # Import nodes
            result = session.run("MATCH (n) RETURN n")

            for record in result:
                node = record["n"]
                node_id = node.get("id")

                # Extract properties
                properties = dict(node)

                graph.add_node(node_id, **properties)

            # Import relationships
            result = session.run("MATCH (a)-[r]->(b) RETURN a, r, b, type(r) as rel_type")

            for record in result:
                source = record["a"].get("id")
                target = record["b"].get("id")
                rel_type = record["rel_type"]
                rel_properties = dict(record["r"])

                rel_properties["type"] = rel_type.lower()

                graph.add_edge(source, target, **rel_properties)

        logger.info(f"Imported graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        return graph

    def query(
        self,
        cypher_query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Cypher 쿼리 실행

        Args:
            cypher_query: Cypher 쿼리
            parameters: 쿼리 파라미터

        Returns:
            List[Dict]: 쿼리 결과
        """
        if not self._driver:
            self.connect()

        results = []

        with self._driver.session() as session:
            result = session.run(cypher_query, parameters or {})

            for record in result:
                results.append(dict(record))

        return results

    def __enter__(self):
        """Context manager enter"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
