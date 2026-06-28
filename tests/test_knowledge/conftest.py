"""Fixtures compartilhados para os testes da Camada C11.

Adaptação em relação ao SDD-C11 §9.3:
- ``test_embedding`` usa um modelo fake leve e deterministico para evitar
  download de modelos e manter os testes rapidos/reprodutiveis.
- ``db_conn`` é function-scoped com caminho único para evitar "database is locked"
  quando um teste anterior falha antes de fechar a conexão.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import sqlite_vec

from src.knowledge import cli as cli_module
from src.knowledge import constants as constants_module
from src.knowledge import db as db_module
from src.knowledge import indexer as indexer_module
from src.knowledge import retriever as retriever_module
from src.knowledge.schemas import DocumentMeta


class FakeSentenceTransformer:
    """Modelo de embedding deterministico baseado em bag-of-words hash.

    Usa tokens alfanumericos hasheados para dimensoes fixas, garantindo
    que textos com palavras em comum tenham similaridade cosseno alta.
    E leve, offline e deterministico.
    """

    dim: int = constants_module.EMBEDDING_DIM

    def encode(
        self,
        sentences: str | List[str],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]

        embeddings: List[np.ndarray] = []
        for text in sentences:
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in re.findall(r"\w+", text.lower()):
                idx = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.dim
                vec[idx] += 1.0
            if normalize_embeddings:
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec = vec / norm
            embeddings.append(vec)

        return np.array(embeddings, dtype=np.float32)


@pytest.fixture(scope="session")
def test_embedding() -> FakeSentenceTransformer:
    return FakeSentenceTransformer()


@pytest.fixture
def db_conn(tmp_path: Path) -> Tuple[Any, Path]:
    """Conexao SQLite + sqlite-vec com schema inicializado e seu caminho."""
    db_path = tmp_path / "knowledge.db"
    connection = db_module.get_connection(db_path)
    db_module.init_schema(connection)
    yield connection, db_path
    connection.close()


@pytest.fixture
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Path]:
    """Isola DB, DLQ e audit trail de queries para cada teste."""
    db_path = tmp_path / "knowledge.db"
    dlq_path = tmp_path / "knowledge_rejected.jsonl"
    log_path = tmp_path / "knowledge_queries.jsonl"
    lineage_dir = tmp_path / "lineage" / "knowledge"

    monkeypatch.setattr(constants_module, "KNOWLEDGE_DB", db_path)
    monkeypatch.setattr(constants_module, "DLQ_PATH", dlq_path)
    monkeypatch.setattr(constants_module, "LOG_QUERIES", log_path)
    monkeypatch.setattr(constants_module, "LINEAGE_DIR", lineage_dir)

    return {
        "db_path": db_path,
        "dlq_path": dlq_path,
        "log_path": log_path,
        "lineage_dir": lineage_dir,
    }


@pytest.fixture
def mini_index_env(
    tmp_path: Path,
    isolated_paths: Dict[str, Path],
    test_embedding: FakeSentenceTransformer,
    monkeypatch: pytest.MonkeyPatch,
) -> Dict[str, Path]:
    """Cria um projeto mínimo e redireciona o indexador/CLI/db para testes rápidos."""
    root = tmp_path / "mini_project"
    docs = root / "docs"
    src = root / "src"
    firmware = root / "firmware" / "src"
    for d in (docs, src / "models", firmware):
        d.mkdir(parents=True, exist_ok=True)

    (docs / "Camada-04-Modelagem-v1.1.md").write_text(
        "# Modelagem\n\nO QG5 exige F1-macro maior que 30 por cento.",
        encoding="utf-8",
    )
    (docs / "Camada-08-Firmware-v1.1.md").write_text(
        "# Firmware\n\nO STM32 roda TFLM e CMSIS-NN no Cortex-M4.",
        encoding="utf-8",
    )
    (src / "models" / "stage1_binary.py").write_text(
        "def build_model():\n    return None\n",
        encoding="utf-8",
    )
    (firmware / "main.c").write_text(
        "int main(void) { while(1); }\n",
        encoding="utf-8",
    )

    db_path = isolated_paths["db_path"]
    lineage_dir = isolated_paths["lineage_dir"]

    monkeypatch.setattr(indexer_module, "PROJECT_ROOT", root)
    monkeypatch.setattr(indexer_module, "DOCS_DIR", docs)
    monkeypatch.setattr(indexer_module, "SRC_DIR", src)
    monkeypatch.setattr(indexer_module, "FIRMWARE_DIR", firmware)
    monkeypatch.setattr(indexer_module, "KNOWLEDGE_DB", db_path)
    monkeypatch.setattr(indexer_module, "LINEAGE_DIR", lineage_dir)
    monkeypatch.setattr(indexer_module, "CONFIG_PATH", root / "knowledge_v2.0.yaml")
    monkeypatch.setattr(indexer_module, "SentenceTransformer", lambda *a, **kw: test_embedding)

    monkeypatch.setattr(db_module, "KNOWLEDGE_DB", db_path)
    monkeypatch.setattr(cli_module, "KNOWLEDGE_DB", db_path)
    monkeypatch.setattr(cli_module, "LINEAGE_DIR", lineage_dir)
    monkeypatch.setattr(retriever_module, "_get_model", lambda: test_embedding)
    monkeypatch.setattr(retriever_module, "LOG_QUERIES", tmp_path / "queries.jsonl")

    return {
        "root": root,
        "db_path": db_path,
        "lineage_dir": lineage_dir,
        "dlq_path": isolated_paths["dlq_path"],
    }


@pytest.fixture
def sample_chunks() -> List[Tuple[str, str, str, List[str], str]]:
    """Documentos de exemplo com metadados 3D variados."""
    return [
        (
            "docs/Camada-04-Modelagem-v1.1.md",
            "C04",
            "v1.1",
            ["ml"],
            "O threshold F1-macro do QG5 deve ser maior que 30 por cento.",
        ),
        (
            "src/models/stage1_binary.py",
            "GENERAL",
            "unversioned",
            ["ml"],
            "def build_stage1_model(input_shape=(500, 1)): return model",
        ),
        (
            "docs/Camada-08-Firmware-v1.1.md",
            "C08",
            "v1.1",
            ["firmware"],
            "O firmware STM32 usa TFLM e CMSIS-NN no Cortex-M4.",
        ),
        (
            "docs/Camada-05-Quantizacao-Exportacao-v1.1.md",
            "C05",
            "v1.1",
            ["quantizacao"],
            "A quantizacao INT8 usa zero_point e scale no TFLite.",
        ),
    ]


def _insert_chunk(
    connection: Any,
    source: str,
    layer: str,
    version: str,
    tags: List[str],
    content: str,
    embedding_model: FakeSentenceTransformer,
) -> DocumentMeta:
    """Insere um chunk diretamente no banco de testes."""
    meta = DocumentMeta(
        source=source,
        layer=layer,
        version=version,
        tags=tags,
        filename=Path(source).name,
        chunk_id=indexer_module.deterministic_chunk_id(source, content),
        header_1=None,
        header_2=None,
    )
    emb = embedding_model.encode(f"{source}\n{content}")[0]
    connection.execute(
        """
        INSERT INTO knowledge_chunks
        (chunk_id, source, layer, version, tags, header_1, header_2, content, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            meta.chunk_id,
            meta.source,
            meta.layer,
            meta.version,
            json.dumps(meta.tags),
            meta.header_1 or "",
            meta.header_2 or "",
            content,
            sqlite_vec.serialize_float32(emb),
        ),
    )
    return meta


@pytest.fixture
def populated_db(
    db_conn: Tuple[Any, Path],
    sample_chunks: List[Tuple[str, str, str, List[str], str]],
    test_embedding: FakeSentenceTransformer,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Banco temporario populado com chunks de teste e retriever monkeypatched.

    A conexao de insercao e fechada antes do yield; sqlite-vec atualiza o
    indice KNN por conexao, entao consultas precisam de uma nova conexao.
    """
    connection, db_path = db_conn
    for source, layer, version, tags, content in sample_chunks:
        _insert_chunk(connection, source, layer, version, tags, content, test_embedding)
    connection.commit()
    connection.close()

    # Redireciona o db para o banco de teste e o retriever para o modelo fake.
    monkeypatch.setattr(db_module, "KNOWLEDGE_DB", db_path)
    monkeypatch.setattr(retriever_module, "_get_model", lambda: test_embedding)
    monkeypatch.setattr(retriever_module, "LOG_QUERIES", db_path.parent / "queries.jsonl")

    yield db_path
