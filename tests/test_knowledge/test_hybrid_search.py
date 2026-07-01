import numpy as np

from src.knowledge.db import get_connection, init_schema
from src.knowledge.hybrid_search import HybridSearcher


FIXED_EMBEDDING = np.ones(384, dtype=np.float32) * 0.1


def _insert_chunk(conn, chunk_id, source, content, tags="experimento"):
    conn.execute(
        """
        INSERT INTO knowledge_chunks
            (chunk_id, source, layer, version, tags, header_1, header_2, content, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chunk_id,
            source,
            "SIMULATION",
            "v1",
            tags,
            "h1",
            "",
            content,
            np.array(FIXED_EMBEDDING, dtype=np.float32).tobytes(),
        ),
    )


def test_hybrid_search_rrf(tmp_path):
    db_path = tmp_path / "hybrid.db"
    conn = get_connection(db_path)
    init_schema(conn)

    _insert_chunk(conn, "c1", "run_001.md", "accuracy do run_001 foi 0.92")
    _insert_chunk(conn, "c2", "run_002.md", "resultado do run_002")
    conn.commit()
    conn.close()

    class FakeIndexer:
        def semantic_search(self, query, filters=None, top_k=10):
            return [{"id": "c1", "score": 0.9, "content": "accuracy do run_001 foi 0.92"}]

        def get_doc(self, doc_id):
            return {"id": doc_id, "content": f"doc {doc_id}"}

    searcher = HybridSearcher(str(db_path), FakeIndexer())
    searcher.index_document(doc_id="c1", content="accuracy do run_001 foi 0.92", run_id="c1")
    searcher.index_document(doc_id="c2", content="resultado do run_002", run_id="c2")
    results = searcher.search("run_001 accuracy", top_k=5)
    assert any(r["id"] == "c1" for r in results)


def test_hybrid_search_bm25_boosts_rrf(tmp_path):
    db_path = tmp_path / "hybrid.db"
    conn = get_connection(db_path)
    init_schema(conn)

    _insert_chunk(conn, "c1", "run_001.md", "accuracy do run_001 foi 0.92")
    _insert_chunk(conn, "c2", "run_002.md", "resultado do run_002")
    conn.commit()
    conn.close()

    class FakeIndexer:
        def semantic_search(self, query, filters=None, top_k=10):
            # Vector ranking alone would pick c2.
            return [
                {"id": "c2", "score": 0.95, "content": "resultado do run_002"},
                {"id": "c1", "score": 0.5, "content": "accuracy do run_001 foi 0.92"},
            ]

        def get_doc(self, doc_id):
            content_map = {
                "c1": "accuracy do run_001 foi 0.92",
                "c2": "resultado do run_002",
            }
            return {"id": doc_id, "content": content_map.get(doc_id, "")}

    searcher = HybridSearcher(str(db_path), FakeIndexer())
    searcher.index_document(doc_id="c1", content="accuracy do run_001 foi 0.92", run_id="c1")
    searcher.index_document(doc_id="c2", content="resultado do run_002", run_id="c2")

    results = searcher.search("run_001 accuracy", top_k=5)
    ids = [r["id"] for r in results]
    assert ids[0] == "c1"
    assert "c2" in ids
