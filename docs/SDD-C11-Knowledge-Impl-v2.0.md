# SDD — Project-Lewis Camada C11: Knowledge Layer (RAG + sqlite-vec + MCP)
## Documento de Implementação Extremamente Planejada — Blueprint Completo

**Documento:** SDD-C11-Knowledge-Impl-v2.0  
**Versão:** 2.0  
**Data:** 2026-06-27  
**Autor:** Douglas Souza (Arquiteto)  
**Projeto:** Project-Lewis — Pipeline ECG Edge ML  
**Status:** Aprovado para implementação imediata  
**Hardware Dev:** Lenovo IdeaPad 3 15ITL6 | Zorin OS 18.1  
**Python:** 3.12.x (system Python — veto 3.13+)  
**Stack:** uv (Astral), sentence-transformers, sqlite-vec, MCP Python SDK, argparse  

---

## ÍNDICE EXECUTIVO

| Seção | Conteúdo | Página |
|-------|----------|--------|
| 1. Contexto e Escopo | Por que RAG, por que agora, limites do contexto atual | — |
| 2. Decisões Arquiteturais (ADRs) | ADR-005 a ADR-009, todas ratificadas | — |
| 3. Stack Tecnológico Detalhado | Versões, constraints, vetos | — |
| 4. Estrutura de Diretórios | Árvore completa C11 | — |
| 5. Módulos de Implementação | Código completo: indexer, retriever, mcp_server, cli | — |
| 6. Pipeline de Indexação | Fluxo de dados, determinismo, idempotência | — |
| 7. Sistema de Retrieval | Similaridade cosseno, filtros 3D, formatação | — |
| 8. MCP Server | Protocolo stdio via SDK oficial | — |
| 9. Testes e Quality Gates | QG-C11, pirâmide 70/20/10, fixtures | — |
| 10. Segurança e LGPD | Superfície de ataque, mitigações, compliance | — |
| 11. DevOps e CI/CD | Makefile targets, pre-commit, Docker | — |
| 12. Roadmap de Implementação | Fases, entregáveis, verificação | — |
| 13. Checklist de Aceite | Binário passa/falha para merge | — |
| 14. Anexos | Referências, glossário, troubleshooting | — |

---

## 1. CONTEXTO E ESCOPO

### 1.1 Diagnóstico do Estado Atual

O Project-Lewis possui 18 documentos técnicos markdown (Camadas 01–09, SDDs, PRDs, Especificações) totalizando ~180 KB de texto puro, além de ~8.500 linhas de código Python e ~3.500 linhas de C/C++ no firmware. Os agentes de coding (Kimi Code, OpenCode) operam com **context windows limitadas** (~128k–200k tokens). Quando o agente é solicitado a implementar `src/models/stage1_binary.py`, ele não carrega automaticamente `Camada-04-Modelagem-v1.1.md` na memória — a menos que o arquivo esteja aberto no editor. Isso gera:

- **Alucinações arquiteturais:** o agente inventa thresholds de QG5 ou esquece que a classe Q foi excluída no v2.0.
- **Inconsistências de stack:** o agente sugere PyTorch ou Python 3.13, violando as Regras de Ouro.
- **Re-trabalho:** cada sessão requer re-explicação do contexto do projeto.

### 1.2 Objetivo da Camada C11

Construir um **sistema de recuperação semântica (RAG) local, persistente e versionado** que:

1. Indexe 100% da documentação técnica do Project-Lewis (markdown + código-fonte Python/C).
2. Permita queries semânticas filtradas por metadados 3D (camada, versão, tags).
3. Seja exposto como **MCP server** (`project-lewis-knowledge`) consumível por Kimi Code e OpenCode.
4. Opere **offline** no IdeaPad 3, sem dependência de serviços cloud ou GPU.
5. Respeite LGPD: nunca indexe dados de ECG, PII, ou registros de paciente.
6. **Não quebre o sistema e a estrutura já construídas** — reaproveite SQLite/SQLAlchemy, argparse, Pydantic v2 e demais convenções do projeto.

### 1.3 Escopo Inclui

- Indexação semântica de docs, src Python e firmware C.
- Vector DB local (`sqlite-vec`) sobre SQLite, CPU-only.
- MCP server stdio via SDK oficial do Python.
- CLI interno (`uv run python -m src.knowledge.cli`) para reindexação e query manual.
- Testes pytest com fixtures parametrizadas.
- Quality Gate QG-C11 com thresholds quantitativos.
- Procedimento de remoção completa de qualquer implementação anterior antes da ativação.

### 1.4 Escopo Exclui

- UI web/frontend (veto Radix UI; headless por design).
- Integração com MLflow, Weights & Biases, ou experiment trackers externos.
- Indexação de dados brutos de ECG (`.dat`, `.mat`, `.hea`) — LGPD proibitivo.
- Deploy em cloud ou servidor remoto.
- Fine-tuning de embedding models (uso de modelo pré-treinado frozen).
- Qualquer dependência de LangChain, Chroma ou typer.

---

## 2. DECISÕES ARQUITETURAIS (ADRs)

### ADR-005: Vector DB Local vs. Remoto

**Contexto:** Vector DBs cloud exigem conectividade, credenciais e custo. O Project-Lewis é desenvolvido em um IdeaPad 3 sem GPU, frequentemente offline. A versão v1.0 do SDD-C11 propunha Chroma, que introduz um segundo SQLite e dependências adicionais.

**Decisão:** Adotar **sqlite-vec** como extensão do SQLite já utilizado pelo tracking (`src/tracking/db.py`). O banco de knowledge será `data/knowledge.db`.

**Consequências:**
- (+) Zero infraestrutura extra, zero custo, 100% offline.
- (+) Backup trivial: `cp data/knowledge.db ...`.
- (+) Infraestrutura unificada: SQLAlchemy 2.0, SQLite, transações ACID.
- (+) Menor lockfile (`uv.lock`) e superfície de ataque.
- (-) Exige carregar extensão nativa (`sqlite_vec.load`).
- (-) Menos abstrações de RAG prontas do que Chroma/LangChain.

### ADR-006: Embedding Model CPU-Only

**Contexto:** O IdeaPad 3 não possui GPU dedicada. Modelos de embedding cloud exigem API key e internet.

**Decisão:** Adotar `sentence-transformers` diretamente com modelo multilíngue moderno. Opções viáveis:
- **Padrão:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensões, ~120 MB, melhor para PT-BR técnico).
- **Alternativa leve:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensões, ~80 MB).
- **Alternativa E5:** `intfloat/multilingual-e5-small` (384 dimensões, ~130 MB).

A escolha final é configurável em `config/knowledge_v2.0.yaml`. O padrão é `paraphrase-multilingual-MiniLM-L12-v2` por melhor desempenho em português técnico.

**Consequências:**
- (+) Offline, sem API key.
- (+) PT-BR técnico melhor representado do que com `all-MiniLM-L6-v2`.
- (-) ~40 MB a mais de download em relação ao `all-MiniLM-L6-v2`.

### ADR-007: MCP stdio via SDK Oficial

**Contexto:** MCP (Model Context Protocol) suporta transporte stdio (subprocesso local) ou HTTP/SSE. Kimi Code e OpenCode suportam ambos. A v1.0 do SDD-C11 implementava JSON-RPC 2.0 manualmente.

**Decisão:** Usar o **SDK oficial do MCP para Python** (`mcp.server.fastmcp.FastMCP`) com transporte stdio.

**Consequências:**
- (+) Nenhuma porta de rede exposta.
- (+) O agente gerencia o lifecycle do processo.
- (+) Schema JSON gerado automaticamente a partir de type hints.
- (-) Adiciona dependência `mcp` ao projeto.

### ADR-008: Metadados 3D (Layer, Version, Tags)

**Contexto:** Documentação do Project-Lewis é hierárquica por camada (C01–C10) e versionada (v1.1, v2.0).

**Decisão:** Cada chunk armazena metadados estruturados em 3 eixos:
- **D1 — layer:** `C01`, `C02`, ..., `C10`, `SDD`, `PRD`, `UNIFIED`, `ESPECIFICACAO`.
- **D2 — version:** `v1.1`, `v2.0`, `v1.4`, `unversioned`.
- **D3 — tags:** lista de tags semânticas extraídas heurísticamente do conteúdo.

Além disso, os metadados são persistidos como **colunas metadata** na tabela virtual do sqlite-vec, permitindo filtros diretos na query KNN.

### ADR-009: CLI Nativo argparse

**Contexto:** O projeto usa `argparse` em `src/tracking/cli.py`. A v1.0 do SDD-C11 propunha `typer`, criando inconsistência de CLI.

**Decisão:** O CLI da Camada C11 usará `argparse` com subcomandos `reindex`, `query` e `status`.

**Consequências:**
- (+) Consistência com o restante do projeto.
- (+) Zero dependências extras.
- (-) Mais verboso que typer/click, mas aceitável para 3 subcomandos.

---

## 3. STACK TECNOLÓGICO DETALHADO

### 3.1 Dependências Produtivas

| Pacote | Versão | Função | Justificativa |
|--------|--------|--------|---------------|
| `sentence-transformers` | `>=3.0.0,<4.0.0` | Embedding model | Modelo multilíngue, CPU-only, offline |
| `sqlite-vec` | `>=0.1.0,<1.0.0` | Vector search no SQLite | Reaproveita SQLite do projeto, sem Chroma |
| `mcp` | `>=1.6.0,<2.0.0` | SDK oficial MCP Python | stdio server com schema automático |
| `markdown` | `>=3.6.0,<4.0.0` | Parsing de frontmatter e headers | Já presente no projeto |
| `sqlalchemy` | `>=2.0.0` | ORM SQLite | Já presente no tracking |
| `pydantic` | `>=2.0` | Schemas | Já presente no projeto |

### 3.2 Dependências de Teste

| Pacote | Versão | Função |
|--------|--------|--------|
| `pytest` | `>=8.0.0` | Framework de testes |
| `pytest-cov` | `>=4.1.0` | Cobertura |
| `pytest-xdist` | `>=3.5.0` | Paralelismo |

### 3.3 Constraints e Vetos

| # | Constraint | Severidade | Verificação |
|---|------------|------------|-------------|
| C01 | Python 3.12.x exclusivamente | 🔴 Bloqueante | `python --version` em `check-env` |
| C02 | uv como único gerenciador | 🔴 Bloqueante | `pyproject.toml` + `uv.lock` |
| C03 | Nenhuma dependência de frontend | 🔴 Bloqueante | `bandit` + `pre-commit` scan |
| C04 | Proibido LangChain, Chroma, typer | 🔴 Bloqueante | `test_deps_compliance.py` |
| C05 | Nenhum dado de ECG ou PII no índice | 🔴 Bloqueante | Regex scan em `test_lgpd_compliance` |
| C06 | CPU-only (sem CUDA) | 🟡 Alto | `device="cpu"` no sentence-transformers |
| C07 | MCP stdio (não HTTP) | 🟡 Alto | `mcp.json` config |
| C08 | Banco SQLite único por contexto | 🟡 Alto | `data/knowledge.db` isolado de `data/lewis_metrics.db` |
| C09 | Docstrings em PT-BR | 🟢 Baixo | `flake8` + review manual |

---

## 4. ESTRUTURA DE DIRETÓRIOS — CAMADA C11

```
Project-Lewis/
├── src/
│   ├── knowledge/                    # NOVO — Camada C11
│   │   ├── __init__.py
│   │   ├── indexer.py                 # Indexação semântica
│   │   ├── retriever.py               # Retrieval com filtros 3D
│   │   ├── mcp_server.py             # MCP server stdio (SDK oficial)
│   │   ├── cli.py                     # CLI argparse (reindex, query, status)
│   │   ├── schemas.py                 # Pydantic v2 models
│   │   ├── db.py                      # SQLite/SQLAlchemy + sqlite-vec
│   │   ├── constants.py               # Paths, defaults, tag keywords
│   │   └── utils.py                   # Helpers (hash, regex PII, etc.)
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── quantization/
│   └── tracking/                      # EXISTENTE — SQLite/SQLAlchemy
├── tests/
│   ├── test_knowledge/               # NOVO — Testes C11
│   │   ├── conftest.py               # Fixtures (db, embedding, sample docs)
│   │   ├── test_indexer.py           # QG-C11-01: indexação
│   │   ├── test_retriever.py         # QG-C11-02: retrieval
│   │   ├── test_mcp_server.py        # QG-C11-03: MCP protocol
│   │   ├── test_lgpd_compliance.py   # QG-C11-04: zero PII
│   │   ├── test_integration.py       # QG-C11-05: end-to-end
│   │   └── test_deps_compliance.py   # QG-C11-09: ausência de LangChain/Chroma/typer
├── data/
│   ├── knowledge.db                  # NOVO — Vector DB SQLite (gitignored)
│   ├── lewis_metrics.db              # EXISTENTE — Tracking DB
│   ├── lineage/
│   │   └── knowledge/                # NOVO — Linhagem de indexação
│   └── .dlq/
│       └── knowledge_rejected.jsonl  # NOVO — DLQ C11
├── logs/
│   └── knowledge_queries.jsonl       # NOVO — Audit trail de queries
├── scripts/
│   └── validate_knowledge_index.py   # NOVO — Validação pós-indexação
├── config/
│   └── knowledge_v2.0.yaml           # NOVO — Config da Camada C11
├── mcp.json                          # MODIFICADO — Adicionar server
├── Makefile                          # MODIFICADO — Targets knowledge-*
└── pyproject.toml                    # MODIFICADO — Deps RAG
```

---

## 5. MÓDULOS DE IMPLEMENTAÇÃO — CÓDIGO COMPLETO

### 5.1 `src/knowledge/constants.py`

```python
"""Constantes da Camada C11 — Knowledge Layer.

Autor: Douglas Souza
Data: 2026-06-27
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"
SRC_DIR = PROJECT_ROOT / "src"
FIRMWARE_DIR = PROJECT_ROOT / "firmware" / "src"
KNOWLEDGE_DB = PROJECT_ROOT / "data" / "knowledge.db"
LINEAGE_DIR = PROJECT_ROOT / "data" / "lineage" / "knowledge"
DLQ_PATH = PROJECT_ROOT / "data" / ".dlq" / "knowledge_rejected.jsonl"
LOG_QUERIES = PROJECT_ROOT / "logs" / "knowledge_queries.jsonl"
CONFIG_PATH = PROJECT_ROOT / "config" / "knowledge_v2.0.yaml"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
COLLECTION_NAME = "project_lewis_knowledge"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
MAX_CHUNK_CHARS = 1024

TAG_KEYWORDS = {
    "quantizacao": ["quantizacao", "INT8", "PTQ", "QAT", "zero_point", "quantization"],
    "firmware": ["firmware", "STM32", "TFLM", "CMSIS-NN", "bare-metal", "Cortex-M4"],
    "dsp": ["filtro", "Butterworth", "AMPT", "R-peak", "filtfilt", "bandpass", "notch"],
    "ml": ["GroupKFold", "F1-macro", "backbone", "fine-tuning", "AAMI", "inter-patient"],
    "lgpd": ["LGPD", "PII", "anonimizacao", "consentimento", "dados pessoais"],
    "devops": ["uv", "Docker", "CI/CD", "pre-commit", "DVC", "Makefile"],
    "energia": ["energia", "mJ", "mAh", "Renode", "autonomia", "consumo"],
    "dados": ["MIT-BIH", "Chapman", "PhysioNet", "dataset", "download", "resample"],
    "modelagem": ["CNN", "1D-CNN", "softmax", "sigmoid", "backbone", "pre-treino"],
    "seguranca": ["JWT", "OAuth2", "Argon2", "bcrypt", "hash", "CSP", "XSS"],
}

LAYER_MAP = {
    "Camada-01": "C01",
    "Camada-02": "C02",
    "Camada-03": "C03",
    "Camada-04": "C04",
    "Camada-05": "C05",
    "Camada-06": "C06",
    "Camada-07": "C07",
    "Camada-08": "C08",
    "Camada-09": "C09",
    "Camada-10": "C10",
    "SDD_": "SDD",
    "PRD": "PRD",
    "UNIFIED": "UNIFIED",
    "ESPECIFICACAO": "ESPECIFICACAO",
    "SIMULATION": "SIMULATION",
    "DEBITO": "DEBITO_TECNICO",
}

PII_PATTERNS = [
    r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b",  # CPF
    r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b",  # CNPJ
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
    r"\b(?:paciente|patient|nome|name):\s*\w+",  # Nomes próprios em contexto clínico
]

FORBIDDEN_EXTENSIONS = {".dat", ".mat", ".hea", ".atr", ".xyz", ".ecg"}
FORBIDDEN_PATH_PATTERNS = [
    "raw_chapman/",
    "raw_mitbih/",
    "raw_svdb/",
    "raw_afdb/",
    "raw_incart/",
    "raw_ptbxl/",
]
```

### 5.2 `src/knowledge/schemas.py`

```python
"""Schemas Pydantic v2 para a Camada C11.

Autor: Douglas Souza
Data: 2026-06-27
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class _BaseConfig(BaseModel):
    """Base com configuração comum."""

    model_config = {"from_attributes": True}


class DocumentMeta(_BaseConfig):
    """Metadados 3D de um chunk indexado."""

    source: str = Field(..., description="Caminho relativo do arquivo fonte")
    layer: str = Field(..., description="D1: Camada arquitetural (C01-C10, SDD, PRD, etc.)")
    version: str = Field(..., description="D2: Versão do documento (v1.1, v2.0, unversioned)")
    tags: List[str] = Field(default_factory=list, description="D3: Tags semânticas extraídas")
    filename: str = Field(..., description="Nome do arquivo")
    chunk_id: str = Field(..., description="Hash determinístico do chunk")
    header_1: Optional[str] = Field(None, description="H1 do markdown, se presente")
    header_2: Optional[str] = Field(None, description="H2 do markdown, se presente")

    @field_validator("layer")
    @classmethod
    def validate_layer(cls, v: str) -> str:
        allowed = {
            "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10",
            "SDD", "PRD", "UNIFIED", "ESPECIFICACAO", "SIMULATION", "DEBITO_TECNICO", "GENERAL",
        }
        if v not in allowed:
            raise ValueError(f"Layer '{v}' não é válida. Valores permitidos: {allowed}")
        return v


class QueryRequest(_BaseConfig):
    """Request de query para o retriever."""

    query: str = Field(..., min_length=1, max_length=4096)
    layer: Optional[str] = Field(None, description="Filtro D1")
    version: Optional[str] = Field(None, description="Filtro D2")
    tags: Optional[List[str]] = Field(None, description="Filtro D3 (AND lógico)")
    k: int = Field(5, ge=1, le=20)
    fetch_k: int = Field(20, ge=5, le=100)

    @field_validator("layer")
    @classmethod
    def validate_layer_optional(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {
            "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10",
            "SDD", "PRD", "UNIFIED", "ESPECIFICACAO", "SIMULATION", "DEBITO_TECNICO", "GENERAL",
        }
        if v not in allowed:
            raise ValueError(f"Layer '{v}' não é válida.")
        return v


class QueryResult(_BaseConfig):
    """Resultado de uma query."""

    chunk_id: str
    source: str
    layer: str
    version: str
    tags: List[str]
    content: str
    score: float = Field(..., description="Score de similaridade (cosine)")
    rank: int = Field(..., ge=1)


class IndexLineage(_BaseConfig):
    """Registro de linhagem de uma indexação."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_files: int
    total_chunks: int
    layers_found: List[str]
    versions_found: List[str]
    index_duration_sec: float
    db_size_bytes: int
    config_hash: str
```

### 5.3 `src/knowledge/utils.py`

```python
"""Utilitários da Camada C11.

Autor: Douglas Souza
Data: 2026-06-27
"""

import hashlib
import json
import re
from pathlib import Path
from typing import List, Set

from .constants import (
    FORBIDDEN_EXTENSIONS,
    FORBIDDEN_PATH_PATTERNS,
    PII_PATTERNS,
    TAG_KEYWORDS,
)


def deterministic_chunk_id(source: str, content_prefix: str) -> str:
    """Gera ID determinístico de 16 chars para evitar duplicatas."""
    payload = f"{source}:{content_prefix[:200]}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def extract_tags_from_content(content: str) -> List[str]:
    """Extrai tags semânticas por heurística de keywords."""
    content_lower = content.lower()
    tags: Set[str] = set()
    for tag, keywords in TAG_KEYWORDS.items():
        if any(kw.lower() in content_lower for kw in keywords):
            tags.add(tag)
    return sorted(tags)


def detect_pii(text: str) -> List[str]:
    """Detecta padrões de PII no texto. Retorna lista de matches."""
    matches: List[str] = []
    for pattern in PII_PATTERNS:
        matches.extend(re.findall(pattern, text, re.IGNORECASE))
    return matches


def is_forbidden_path(path: Path) -> bool:
    """Verifica se o caminho contém dados brutos de ECG ou diretórios proibidos."""
    text = path.as_posix().lower()
    if path.suffix.lower() in FORBIDDEN_EXTENSIONS:
        return True
    return any(p.lower() in text for p in FORBIDDEN_PATH_PATTERNS)


def compute_config_hash(config_path: Path) -> str:
    """Hash SHA256 do arquivo de config para invalidação de cache."""
    if not config_path.exists():
        return "no-config"
    return hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]


def format_context_for_agent(docs: List[dict]) -> str:
    """Formata documentos para injeção em prompt de agente."""
    blocks = []
    for i, doc in enumerate(docs, 1):
        tags = doc.get("tags", [])
        block = (
            f"[{i}] Fonte: {doc['source']} | "
            f"Camada: {doc['layer']} | Versão: {doc['version']} | "
            f"Tags: {', '.join(tags)}\n"
            f"{doc['content']}\n"
        )
        blocks.append(block)
    return "\n---\n".join(blocks)


def parse_markdown_frontmatter(content: str) -> dict:
    """Extrai YAML frontmatter de markdown, se presente."""
    if not content.startswith("---"):
        return {}
    try:
        _, frontmatter, _ = content.split("---", 2)
        meta = {}
        for line in frontmatter.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        return meta
    except ValueError:
        return {}


def write_dlq(record: dict) -> None:
    """Escreve registro rejeitado na DLQ."""
    from .constants import DLQ_PATH

    DLQ_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DLQ_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

### 5.4 `src/knowledge/db.py`

```python
"""Banco de dados SQLite + sqlite-vec para a Camada C11.

Autor: Douglas Souza
Data: 2026-06-27
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import sqlite_vec
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .constants import EMBEDDING_DIM, KNOWLEDGE_DB


_SQL_CREATE_VEC_TABLE = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks USING vec0(
    chunk_id TEXT PRIMARY KEY,
    source TEXT,
    layer TEXT,
    version TEXT,
    tags TEXT,
    header_1 TEXT,
    header_2 TEXT,
    content TEXT,
    embedding float[{EMBEDDING_DIM}] distance_metric=cosine
);
"""

_SQL_CREATE_META_TABLE = """
CREATE TABLE IF NOT EXISTS knowledge_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Retorna conexão SQLite com extensão sqlite-vec carregada."""
    path = db_path or KNOWLEDGE_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    return conn


def get_engine(db_path: Path | None = None) -> Engine:
    """Retorna engine SQLAlchemy apontando para o banco de knowledge."""
    path = db_path or KNOWLEDGE_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", echo=False, future=True)


def init_schema(conn: sqlite3.Connection | None = None) -> None:
    """Cria tabela virtual sqlite-vec e tabela de metadados."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        conn.execute(_SQL_CREATE_VEC_TABLE)
        conn.execute(_SQL_CREATE_META_TABLE)
        conn.commit()
    finally:
        if own_conn:
            conn.close()


def count_chunks(conn: sqlite3.Connection | None = None) -> int:
    """Retorna número de chunks indexados."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        row = conn.execute("SELECT COUNT(*) FROM knowledge_chunks").fetchone()
        return row[0] if row else 0
    finally:
        if own_conn:
            conn.close()


def clear_index(conn: sqlite3.Connection | None = None) -> None:
    """Remove todos os chunks do índice."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        conn.execute("DELETE FROM knowledge_chunks")
        conn.commit()
    finally:
        if own_conn:
            conn.close()
```

### 5.5 `src/knowledge/indexer.py`

```python
"""Indexador semântico da Camada C11.

Responsabilidade: carregar documentos markdown e código-fonte,
segmentar preservando hierarquia, extrair metadados 3D,
gerar embeddings e persistir no sqlite-vec.

Restrições:
- Não indexar arquivos fora de docs/, src/, firmware/src/.
- Rejeitar chunks contendo PII (LGPD).
- Determinístico: reindexação gera IDs idênticos.
- Nunca indexar dados brutos de ECG (.dat, .mat, .hea).

Autor: Douglas Souza
Data: 2026-06-27
"""

import json
import re
import time
from pathlib import Path
from typing import List

import sqlite_vec
from sentence_transformers import SentenceTransformer

from .constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONFIG_PATH,
    DOCS_DIR,
    EMBEDDING_MODEL,
    FIRMWARE_DIR,
    KNOWLEDGE_DB,
    LAYER_MAP,
    LINEAGE_DIR,
    MAX_CHUNK_CHARS,
    PROJECT_ROOT,
    SRC_DIR,
)
from .db import get_connection, init_schema
from .schemas import DocumentMeta, IndexLineage
from .utils import (
    compute_config_hash,
    detect_pii,
    deterministic_chunk_id,
    extract_tags_from_content,
    is_forbidden_path,
    parse_markdown_frontmatter,
    write_dlq,
)


class Chunk:
    """Chunk de documento com metadados e conteúdo."""

    def __init__(self, content: str, source: str, header_1: str | None = None, header_2: str | None = None):
        self.content = content
        self.source = source
        self.header_1 = header_1
        self.header_2 = header_2


def resolve_layer(file_path: Path) -> str:
    """Resolve camada arquitetural a partir do nome do arquivo."""
    name = file_path.name
    for prefix, layer in LAYER_MAP.items():
        if prefix in name:
            return layer
    return "GENERAL"


def resolve_version(file_path: Path) -> str:
    """Extrai versão do nome do arquivo (vX.Y)."""
    match = re.search(r"v(\d+\.\d+)", file_path.name)
    return f"v{match.group(1)}" if match else "unversioned"


def load_markdown_documents() -> List[Chunk]:
    """Carrega todos os arquivos markdown do projeto."""
    chunks: List[Chunk] = []
    md_paths = list(DOCS_DIR.rglob("*.md")) + list(PROJECT_ROOT.glob("*.md"))

    for md_path in md_paths:
        if not md_path.exists() or is_forbidden_path(md_path):
            continue
        raw = md_path.read_text(encoding="utf-8")
        parse_markdown_frontmatter(raw)  # descartado por simplicidade; pode enriquecer metadados
        # Split por headers
        parts = raw.split("\n# ")
        for i, part in enumerate(parts):
            if i == 0 and not part.strip().startswith("#"):
                header = None
                body = part
            else:
                lines = part.splitlines()
                header = lines[0].lstrip("# ").strip() if lines else None
                body = "\n".join(lines[1:])
            if len(body) > MAX_CHUNK_CHARS:
                sub_chunks = split_text_recursive(body, CHUNK_SIZE, CHUNK_OVERLAP)
                for sub in sub_chunks:
                    chunks.append(Chunk(sub, md_path.relative_to(PROJECT_ROOT).as_posix(), header_1=header))
            else:
                chunks.append(Chunk(body, md_path.relative_to(PROJECT_ROOT).as_posix(), header_1=header))
    return chunks


def load_code_documents() -> List[Chunk]:
    """Carrega código-fonte Python e C como documentos."""
    chunks: List[Chunk] = []

    for py_path in SRC_DIR.rglob("*.py"):
        if "__pycache__" in str(py_path) or is_forbidden_path(py_path):
            continue
        content = py_path.read_text(encoding="utf-8")
        for sub in split_text_recursive(content, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append(Chunk(sub, py_path.relative_to(PROJECT_ROOT).as_posix()))

    for ext in ("*.c", "*.h", "*.cpp"):
        for c_path in FIRMWARE_DIR.rglob(ext):
            if is_forbidden_path(c_path):
                continue
            content = c_path.read_text(encoding="utf-8")
            for sub in split_text_recursive(content, CHUNK_SIZE, CHUNK_OVERLAP):
                chunks.append(Chunk(sub, c_path.relative_to(PROJECT_ROOT).as_posix()))

    return chunks


def split_text_recursive(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Segmentador recursivo simples por parágrafos/frases/palavras."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    result: List[str] = []
    current = ""
    for sep in separators:
        parts = text.split(sep) if sep else list(text)
        for part in parts:
            if len(current) + len(part) + len(sep) > chunk_size and current:
                result.append(current.strip())
                current = current[-overlap:] if overlap > 0 else ""
            current += part + sep
        if current.strip():
            result.append(current.strip())
        if len(result) > 1:
            return result
    if not result and text.strip():
        result.append(text.strip())
    return result


def enrich_metadata(chunks: List[Chunk]) -> List[tuple[DocumentMeta, str]]:
    """Enriquece cada chunk com metadados 3D e rejeita PII."""
    enriched: List[tuple[DocumentMeta, str]] = []

    for chunk in chunks:
        source = chunk.source
        file_path = Path(source)
        content = chunk.content

        if is_forbidden_path(file_path):
            continue

        pii_matches = detect_pii(content)
        if pii_matches:
            write_dlq({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": source,
                "reason": "PII_DETECTED",
                "matches": pii_matches,
                "action": "REJECTED",
            })
            continue

        layer = resolve_layer(file_path)
        version = resolve_version(file_path)
        tags = extract_tags_from_content(content)
        chunk_id = deterministic_chunk_id(source, content)

        meta = DocumentMeta(
            source=source,
            layer=layer,
            version=version,
            tags=tags,
            filename=file_path.name,
            chunk_id=chunk_id,
            header_1=chunk.header_1,
            header_2=chunk.header_2,
        )
        enriched.append((meta, content))

    return enriched


def build_index() -> IndexLineage:
    """Pipeline completo de indexação. Retorna linhagem."""
    start_time = time.perf_counter()
    KNOWLEDGE_DB.parent.mkdir(parents=True, exist_ok=True)
    LINEAGE_DIR.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    md_chunks = load_markdown_documents()
    code_chunks = load_code_documents()
    all_chunks = md_chunks + code_chunks

    enriched = enrich_metadata(all_chunks)

    conn = get_connection()
    init_schema(conn)
    conn.execute("DELETE FROM knowledge_chunks")

    contents = [f"{m.source}\n{c}" for m, c in enriched]
    embeddings = model.encode(contents, normalize_embeddings=True, show_progress_bar=False)

    for (meta, content), emb in zip(enriched, embeddings):
        conn.execute(
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
                meta.header_1,
                meta.header_2,
                content,
                sqlite_vec.serialize_float32(emb),
            ),
        )
    conn.commit()
    conn.close()

    duration = time.perf_counter() - start_time

    metas = [m for m, _ in enriched]
    layers = sorted({m.layer for m in metas})
    versions = sorted({m.version for m in metas})
    db_size = KNOWLEDGE_DB.stat().st_size if KNOWLEDGE_DB.exists() else 0

    lineage = IndexLineage(
        total_files=len({m.source for m in metas}),
        total_chunks=len(metas),
        layers_found=layers,
        versions_found=versions,
        index_duration_sec=duration,
        db_size_bytes=db_size,
        config_hash=compute_config_hash(CONFIG_PATH),
    )

    lineage_path = LINEAGE_DIR / f"index_{lineage.timestamp.isoformat()}.json"
    lineage_path.write_text(lineage.model_dump_json(indent=2), encoding="utf-8")

    print(f"[indexer] {len(metas)} chunks indexados em {duration:.2f}s")
    print(f"[indexer] Camadas: {layers}")
    print(f"[indexer] Versões: {versions}")
    print(f"[indexer] Tamanho DB: {db_size / 1024 / 1024:.2f} MB")

    return lineage


if __name__ == "__main__":
    build_index()
```

### 5.6 `src/knowledge/retriever.py`

```python
"""Retriever inteligente da Camada C11.

Responsabilidade: busca semântica com filtros 3D,
formatação para agentes e logging de audit trail.

Autor: Douglas Souza
Data: 2026-06-27
"""

import json
import time
from pathlib import Path
from typing import List, Optional

import sqlite_vec
from sentence_transformers import SentenceTransformer

from .constants import EMBEDDING_MODEL, KNOWLEDGE_DB, LOG_QUERIES
from .db import get_connection
from .schemas import QueryRequest, QueryResult
from .utils import format_context_for_agent


def _get_model() -> SentenceTransformer:
    """Factory do modelo de embeddings."""
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def _build_where_clause(req: QueryRequest) -> tuple[str, list]:
    """Constrói cláusula WHERE e parâmetros para filtros metadata."""
    conditions: List[str] = []
    params: List[str] = []
    if req.layer:
        conditions.append("layer = ?")
        params.append(req.layer)
    if req.version:
        conditions.append("version = ?")
        params.append(req.version)
    if req.tags:
        for tag in req.tags:
            conditions.append("tags LIKE ?")
            params.append(f"%\"{tag}\"%")
    where = " AND ".join(conditions)
    return where, params


def search(req: QueryRequest) -> List[QueryResult]:
    """Executa busca semântica com filtros e retorna top-k."""
    model = _get_model()
    query_emb = model.encode([req.query], normalize_embeddings=True, show_progress_bar=False)[0]

    conn = get_connection()
    where_clause, where_params = _build_where_clause(req)

    sql = """
        SELECT
            chunk_id,
            source,
            layer,
            version,
            tags,
            header_1,
            header_2,
            content,
            distance
        FROM knowledge_chunks
        WHERE embedding MATCH ?
    """
    params: List = [sqlite_vec.serialize_float32(query_emb)]
    if where_clause:
        sql += f" AND {where_clause}"
        params.extend(where_params)
    sql += " AND k = ? ORDER BY distance"
    params.append(req.fetch_k)

    rows = conn.execute(sql, params).fetchall()
    conn.close()

    results = []
    for i, row in enumerate(rows[: req.k], 1):
        tags = json.loads(row["tags"] or "[]")
        results.append(QueryResult(
            chunk_id=row["chunk_id"],
            source=row["source"],
            layer=row["layer"],
            version=row["version"],
            tags=tags,
            content=row["content"],
            score=1.0 - float(row["distance"]),
            rank=i,
        ))

    _log_query(req, len(results))
    return results


def _log_query(req: QueryRequest, result_count: int) -> None:
    """Registra query em audit trail JSONL."""
    LOG_QUERIES.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query": req.query,
        "layer": req.layer,
        "version": req.version,
        "tags": req.tags,
        "k": req.k,
        "results_returned": result_count,
    }
    with open(LOG_QUERIES, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_context_for_agent(req: QueryRequest) -> str:
    """Busca e formata contexto para injeção em prompt de agente."""
    results = search(req)
    docs = [r.model_dump() for r in results]
    return format_context_for_agent(docs)
```

### 5.7 `src/knowledge/mcp_server.py`

```python
"""MCP Server stdio para integração Kimi Code / OpenCode.

Usa SDK oficial do Python (FastMCP).
Tools expostas:
  - search_docs(query, layer?, version?, tags?, k?)
  - list_layers()
  - get_doc_by_source(source, k?)

Autor: Douglas Souza
Data: 2026-06-27
"""

from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from .retriever import search
from .schemas import QueryRequest

mcp = FastMCP("project-lewis-knowledge")


@mcp.tool()
def search_docs(
    query: str,
    layer: Optional[str] = None,
    version: Optional[str] = None,
    tags: Optional[List[str]] = None,
    k: int = 5,
) -> str:
    """Busca semântica na documentação do Project-Lewis."""
    req = QueryRequest(query=query, layer=layer, version=version, tags=tags, k=k)
    results = search(req)
    blocks = []
    for r in results:
        blocks.append(
            f"[{r.rank}] {r.source} (Camada {r.layer}, {r.version}, score={r.score:.4f})\n"
            f"Tags: {', '.join(r.tags)}\n{r.content}"
        )
    return "\n---\n".join(blocks)


@mcp.tool()
def list_layers() -> List[str]:
    """Lista camadas arquiteturais disponíveis."""
    return [
        "C01", "C02", "C03", "C04", "C05", "C06",
        "C07", "C08", "C09", "C10", "SDD", "PRD", "UNIFIED",
    ]


@mcp.tool()
def get_doc_by_source(source: str, k: int = 3) -> str:
    """Recupera chunks por caminho de arquivo."""
    req = QueryRequest(query=source, k=k)
    results = [r for r in search(req) if r.source == source]
    return "\n---\n".join(
        f"[{r.rank}] {r.source}\n{r.content}" for r in results
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### 5.8 `src/knowledge/cli.py`

```python
"""CLI interno da Camada C11.

Comandos:
  reindex    — Rebuild completo do índice sqlite-vec
  query      — Query via terminal
  status     — Status do índice (chunks, camadas, tamanho)

Autor: Douglas Souza
Data: 2026-06-27
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .constants import KNOWLEDGE_DB, LINEAGE_DIR
from .db import count_chunks
from .indexer import build_index
from .retriever import search
from .schemas import QueryRequest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.knowledge.cli",
        description="CLI da Camada C11 — Knowledge Layer",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("reindex", help="Rebuild completo do índice semântico")

    query_parser = subparsers.add_parser("query", help="Executa busca semântica")
    query_parser.add_argument("q", help="Pergunta ou keyword")
    query_parser.add_argument("--layer", "-l", default=None, help="Filtro camada")
    query_parser.add_argument("--version", "-v", default=None, help="Filtro versão")
    query_parser.add_argument("--k", "-k", type=int, default=5, help="Número de resultados")

    subparsers.add_parser("status", help="Exibe status do índice")

    return parser


def _cmd_reindex() -> int:
    print("Iniciando reindexação...")
    lineage = build_index()
    print(f"✅ {lineage.total_chunks} chunks indexados em {lineage.index_duration_sec:.2f}s")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    req = QueryRequest(query=args.q, layer=args.layer, version=args.version, k=args.k)
    results = search(req)
    for r in results:
        print(f"\n[{r.rank}] {r.source} | {r.layer} | {r.version}")
        print(f"Tags: {', '.join(r.tags)}")
        print(f"{r.content[:300]}...")
    return 0


def _cmd_status() -> int:
    if not KNOWLEDGE_DB.exists():
        print("❌ Índice não encontrado. Execute: uv run python -m src.knowledge.cli reindex")
        return 1

    size_mb = KNOWLEDGE_DB.stat().st_size / 1024 / 1024
    lineage_files = sorted(LINEAGE_DIR.glob("index_*.json")) if LINEAGE_DIR.exists() else []

    print(f"📦 Banco de knowledge: {KNOWLEDGE_DB}")
    print(f"📊 Tamanho: {size_mb:.2f} MB")
    print(f"📄 Chunks: {count_chunks()}")
    print(f"📄 Registros de linhagem: {len(lineage_files)}")
    if lineage_files:
        latest = lineage_files[-1]
        print(f"🕐 Última indexação: {latest.name}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "reindex":
        return _cmd_reindex()
    if args.command == "query":
        return _cmd_query(args)
    if args.command == "status":
        return _cmd_status()
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

### 5.9 `config/knowledge_v2.0.yaml`

```yaml
# Configuração da Camada C11 — Knowledge Layer
# Project-Lewis v2.0

indexer:
  embedding_model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  embedding_dim: 384
  chunk_size: 512
  chunk_overlap: 64
  max_chunk_chars: 1024
  collection_name: "project_lewis_knowledge"

retriever:
  default_k: 5
  default_fetch_k: 20
  distance_metric: "cosine"

mcp:
  transport: "stdio"
  server_name: "project-lewis-knowledge"
  server_version: "2.0.0"

lgpd:
  pii_scan_enabled: true
  reject_on_pii_match: true
  allowed_file_extensions:
    - ".md"
    - ".py"
    - ".c"
    - ".h"
    - ".cpp"
  forbidden_extensions:
    - ".dat"
    - ".mat"
    - ".hea"
    - ".atr"
  forbidden_path_patterns:
    - "raw_chapman/"
    - "raw_mitbih/"
    - "raw_svdb/"
    - "raw_afdb/"
    - "raw_incart/"
    - "raw_ptbxl/"
```

---

## 6. PIPELINE DE INDEXAÇÃO

### 6.1 Fluxo de Dados

```mermaid
flowchart LR
    A[docs/*.md] --> B[Parser markdown]
    C[src/**/*.py] --> D[Recursive splitter]
    E[firmware/src/**/*.{c,h,cpp}] --> D
    B --> F[Enrich Metadata 3D]
    D --> F
    F --> G{PII / forbidden?}
    G -->|Rejeita| H[DLQ: data/.dlq/knowledge_rejected.jsonl]
    G -->|Aprova| I[SentenceTransformer]
    I --> J[sqlite-vec: data/knowledge.db]
    J --> K[data/lineage/knowledge/index_*.json]
```

### 6.2 Determinismo e Idempotência

| Propriedade | Garantia | Mecanismo |
|-------------|----------|-----------|
| IDs determinísticos | ✅ | `SHA256(source + content[:200])[:16]` |
| Reindexação idêntica | ✅ | Mesmo input → mesmos IDs → `DELETE FROM knowledge_chunks` + reinsert |
| Cache de config | ✅ | `config_hash` na linhagem |
| Embeddings determinísticos | ✅ | `normalize_embeddings=True`, seed fixa no sentence-transformers |

### 6.3 Dead Letter Queue (DLQ) para Indexação

```json
{
  "timestamp": "2026-06-27T00:00:00Z",
  "source": "docs/Camada-04-Modelagem-v1.1.md",
  "chunk_id": "abc123def456",
  "reason": "PII_DETECTED",
  "matches": ["123.456.789-00"],
  "action": "REJECTED"
}
```

---

## 7. SISTEMA DE RETRIEVAL

### 7.1 Algoritmo de Busca

1. **Query Embedding:** `query → SentenceTransformer → vetor 384d normalizado`.
2. **Filtro Metadata:** Aplica `=` em `layer`, `version`; `LIKE` em `tags` (JSON array).
3. **KNN:** `sqlite-vec` recupera `fetch_k` vizinhos por distância cosseno.
4. **Corte:** Retorna top-`k` (default 5).

### 7.2 Formatação para Agentes

O MCP server formata o contexto como blocos numerados com metadados inline:

```
[1] docs/Camada-04-Modelagem-v1.1.md (Camada C04, v1.1, score=0.8723)
Tags: quantizacao, ml
O input shape do modelo deve ser consistente com a segmentação da Camada 2...
---
[2] src/models/backbone_1d.py (Camada GENERAL, unversioned, score=0.8510)
Tags: ml, firmware
Input shape: (500, 1) — 1000ms @ 500Hz, 1 canal...
```

---

## 8. MCP SERVER — ESPECIFICAÇÃO DO PROTOCOLO

### 8.1 Transporte

- **Tipo:** stdio (subprocesso).
- **Comando de ativação:** `uv run python -m src.knowledge.mcp_server`.
- **Lifecycle:** Gerenciado pelo cliente; inicia on-demand, morre ao fechar.
- **SDK:** `mcp.server.fastmcp.FastMCP` — schema gerado automaticamente a partir de type hints.

### 8.2 Tools Expostas

| Tool | Params | Retorno | Descrição |
|------|--------|---------|-----------|
| `search_docs` | `query`, `layer?`, `version?`, `tags?`, `k?` | Contexto formatado | Busca semântica |
| `list_layers` | — | `List[str]` | Camadas disponíveis |
| `get_doc_by_source` | `source`, `k?` | Contexto formatado | Chunks por caminho |

### 8.3 Configuração `mcp.json` (Atualizada)

```json
{
  "mcpServers": {
    "sdd-docs": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "./docs"]
    },
    "sdd-rules": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "./.kimi", "./.opencode", "./AGENTS.md"]
    },
    "project-lewis-src": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "./src", "./firmware/src", "./tests", "./scripts"]
    },
    "project-lewis-knowledge": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.knowledge.mcp_server"],
      "env": { "PYTHONPATH": "." }
    }
  }
}
```

---

## 9. TESTES E QUALITY GATES

### 9.1 Pirâmide de Testes C11

| Tipo | % | Arquivos | Tempo |
|------|---|----------|-------|
| Unit | 70% | `test_indexer.py`, `test_retriever.py`, `test_deps_compliance.py` | < 30s |
| Integration | 20% | `test_mcp_server.py`, `test_integration.py` | < 60s |
| E2E | 10% | `test_knowledge_e2e.py` | < 120s |

### 9.2 Quality Gate QG-C11

| ID | Critério | Threshold | Como Validar | Bloqueia |
|----|----------|-----------|--------------|----------|
| QG-C11-01 | Cobertura de indexação | 100% dos `.md` e `.py` indexados | `test_indexer::test_coverage` | Merge |
| QG-C11-02 | Determinismo de IDs | Reindexação gera mesmos IDs | `test_indexer::test_idempotent` | Merge |
| QG-C11-03 | Retrieval precisão | MRR@5 >= 0.80 em 10 queries ref | `test_retriever::test_precision` | Merge |
| QG-C11-04 | Filtro por layer | 100% das camadas recuperáveis | `test_retriever::test_layer_filter` | Merge |
| QG-C11-05 | LGPD — zero PII | 0 ocorrências no índice | `test_lgpd::test_zero_pii` | Merge |
| QG-C11-06 | MCP protocol | Responde a `initialize` e `tools/list` | `test_mcp::test_protocol` | Merge |
| QG-C11-07 | Tamanho do banco | < 500 MB | `test_indexer::test_size` | Merge |
| QG-C11-08 | CLI funcional | `reindex`, `query`, `status` passam | `test_integration::test_cli` | Merge |
| QG-C11-09 | Deps proibidas | Ausência de LangChain/Chroma/typer | `test_deps_compliance::test_no_forbidden_deps` | Merge |

### 9.3 Fixtures de Teste (`tests/test_knowledge/conftest.py`)

```python
import pytest
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.knowledge.db import get_connection, init_schema


@pytest.fixture(scope="session")
def test_db_path(tmp_path_factory):
    return tmp_path_factory.mktemp("knowledge") / "test.db"


@pytest.fixture(scope="session")
def test_embedding():
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",
    )


@pytest.fixture
def conn(test_db_path):
    conn = get_connection(test_db_path)
    init_schema(conn)
    yield conn
    conn.close()
```

### 9.4 Exemplo de Teste Unitário

```python
# tests/test_knowledge/test_retriever.py
import pytest
from src.knowledge.schemas import QueryRequest
from src.knowledge.retriever import search


@pytest.mark.qg_c11
class TestRetriever:
    def test_search_basic(self, populated_db):
        req = QueryRequest(query="threshold F1-macro QG5", k=3)
        results = search(req)
        assert len(results) > 0
        assert any("C04" in r.layer for r in results)

    def test_layer_filter(self, populated_db):
        req = QueryRequest(query="STM32", layer="C08", k=3)
        results = search(req)
        assert all(r.layer == "C08" for r in results)

    def test_version_filter(self, populated_db):
        req = QueryRequest(query="quantizacao", version="v1.1", k=5)
        results = search(req)
        assert all(r.version == "v1.1" for r in results)
```

---

## 10. SEGURANÇA E LGPD

### 10.1 Superfície de Ataque

| Vetor | Risco | Mitigação | Status |
|-------|-------|-----------|--------|
| **PII no índice** | Alto | Regex scan pré-indexação; rejeição automática | Implementado |
| **Dados ECG no índice** | Alto | `is_forbidden_path()` bloqueia `.dat`, `.mat`, `.hea` e `raw_*` | Implementado |
| **Acesso ao SQLite** | Médio | Banco em `data/knowledge.db` (user-only permissions) | Por design |
| **MCP stdio injection** | Baixo | SDK oficial valida schema; sem execução dinâmica | Implementado |
| **Path traversal** | Baixo | `source` sempre relativo; nunca absoluto | Implementado |
| **DDoS do retriever** | Baixo | `k` limitado a 20 | Implementado |

### 10.2 Compliance LGPD

- **Art. 7 (Base legal):** O índice contém apenas documentação técnica.
- **Art. 46 (Segurança):** Nenhum dado pessoal sensível (saúde) é indexado.
- **Art. 50 (Registro de operações):** `logs/knowledge_queries.jsonl` registra queries sem conteúdo sensível.
- **Anonimização:** PII scan rejeita qualquer chunk com CPF, CNPJ, email ou nomes em contexto clínico.

---

## 11. DEVOPS E CI/CD

### 11.1 Targets Makefile (Adições)

```makefile
# Camada C11 — Knowledge Layer
knowledge-index:
	@echo "[C11] Reindexando knowledge base..."
	$(UV) run python -m src.knowledge.cli reindex

knowledge-query:
	@read -p "Query: " q; $(UV) run python -m src.knowledge.cli query "$$q"

knowledge-status:
	$(UV) run python -m src.knowledge.cli status

knowledge-test:
	$(UV) run pytest tests/test_knowledge/ -v --tb=short

knowledge-clean:
	rm -f data/knowledge.db
	rm -rf data/lineage/knowledge/
	rm -f logs/knowledge_queries.jsonl
	rm -f data/.dlq/knowledge_rejected.jsonl

knowledge-validate:
	$(UV) run python scripts/validate_knowledge_index.py
```

### 11.2 Pre-commit Hooks (Adições)

```yaml
# .pre-commit-config.yaml (adições)
- repo: local
  hooks:
    - id: knowledge-lint
      name: Knowledge Layer Type Check
      entry: uv run mypy src/knowledge/
      language: system
      pass_filenames: false
      always_run: true
    - id: knowledge-test-smoke
      name: Knowledge Smoke Test
      entry: uv run pytest tests/test_knowledge/test_indexer.py -v
      language: system
      pass_filenames: false
      always_run: false
      stages: [pre-push]
```

### 11.3 Docker

O Dockerfile existente já cobre Python 3.12 + uv. Adicionar ao `pyproject.toml` as deps (`sentence-transformers`, `sqlite-vec`, `mcp`) é suficiente; o container reproduzirá o ambiente identicamente.

---

## 12. ROADMAP DE IMPLEMENTAÇÃO

### Fase 1: Fundação

| Task | Entregável | Verificação | Owner |
|------|-----------|-------------|-------|
| Adicionar deps ao `pyproject.toml` | `uv.lock` atualizado | `uv sync` passa | DevOps |
| Criar estrutura de diretórios C11 | `src/knowledge/` presente | `ls -R src/knowledge` | Engenheiro |
| Implementar `constants.py` + `schemas.py` + `db.py` | Tipagem completa | `mypy src/knowledge/` passa | Engenheiro |
| Implementar `utils.py` | PII scan e forbidden path funcional | `pytest tests/test_knowledge/test_lgpd.py` | Engenheiro |

### Fase 2: Indexação

| Task | Entregável | Verificação | Owner |
|------|-----------|-------------|-------|
| Implementar `indexer.py` | Indexação completa | `make knowledge-index` gera `data/knowledge.db` | Engenheiro |
| Testar com docs reais | > 100 chunks gerados | `make knowledge-status` | Engenheiro |
| Validar determinismo | Reindexação idêntica | `pytest tests/test_knowledge/test_indexer.py` | QA |

### Fase 3: Retrieval

| Task | Entregável | Verificação | Owner |
|------|-----------|-------------|-------|
| Implementar `retriever.py` | Busca com filtros | `make knowledge-query "threshold QG5"` | Engenheiro |
| Implementar `cli.py` | CLI funcional | `python -m src.knowledge.cli --help` | Engenheiro |
| Testes de precisão | MRR@5 > 0.80 | `pytest tests/test_knowledge/test_retriever.py` | QA |

### Fase 4: MCP

| Task | Entregável | Verificação | Owner |
|------|-----------|-------------|-------|
| Implementar `mcp_server.py` | Server stdio via FastMCP | Cliente MCP reconhece tools | Engenheiro |
| Atualizar `mcp.json` | Config válida | Kimi Code reconhece tool | Arquiteto |
| Teste de integração | Tool call funcional | `pytest tests/test_knowledge/test_mcp_server.py` | QA |

### Fase 5: Quality Gates e Merge

| Task | Entregável | Verificação | Owner |
|------|-----------|-------------|-------|
| Todos os QG-C11 passando | Badge verde | `make knowledge-test` | QA |
| Documentação atualizada | `SDD-C11-Knowledge-Impl-v2.0.md` | Revisão arquitetural | Arquiteto |
| Pre-commit passando | Zero falhas | `uv run pre-commit run --all-files` | DevOps |
| Merge para `main` | Commit semântico | `git log --oneline` | Arquiteto |

---

## 13. CHECKLIST DE ACEITE (Binário Passa/Falha)

```
[ ] pyproject.toml contém sentence-transformers, sqlite-vec, mcp (e NÃO contém langchain, chromadb, typer)
[ ] uv.lock atualizado e válido (uv sync passa em < 30s)
[ ] src/knowledge/ contém __init__.py, constants.py, schemas.py, utils.py, db.py, indexer.py, retriever.py, mcp_server.py, cli.py
[ ] config/knowledge_v2.0.yaml existe e é válido
[ ] make knowledge-index executa sem erro e gera data/knowledge.db
[ ] make knowledge-status reporta > 0 chunks e > 0 camadas
[ ] make knowledge-query "threshold QG5" retorna resultados da Camada C04
[ ] pytest tests/test_knowledge/ passa 100% (QG-C11-01 a QG-C11-09)
[ ] test_lgpd_compliance.py: 0 ocorrências de PII no índice
[ ] test_idempotent.py: reindexação gera IDs idênticos
[ ] test_deps_compliance.py: confirma ausência de LangChain/Chroma/typer
[ ] mcp.json atualizado com project-lewis-knowledge server
[ ] Pre-commit passa (black, isort, flake8, mypy, bandit)
[ ] Makefile contém targets knowledge-index, knowledge-query, knowledge-status, knowledge-test, knowledge-clean
[ ] Documentação técnica (este SDD) revisada e aprovada pelo arquiteto
[ ] Nenhuma dependência de frontend adicionada (veto Radix UI respeitado)
[ ] Python 3.12 exclusivo (nenhuma feature de 3.13+)
[ ] LGPD: nenhum dado de ECG ou PII no índice sqlite-vec
[ ] Implementação anterior removida sem restar nada (Task 0 executada)
```

---

## 14. ANEXOS

### Anexo A: Referências Técnicas

- sqlite-vec Documentation (2026): https://alexgarcia.xyz/sqlite-vec
- sqlite-vec Python bindings (2026): `sqlite-vec` PyPI, `serialize_float32`, `vec0` virtual tables.
- Sentence-Transformers `paraphrase-multilingual-MiniLM-L12-v2` (2026): 384d, multilíngue, CPU-optimized.
- MCP Python SDK (2026): `mcp.server.fastmcp.FastMCP`, stdio transport.
- LGPD Lei 13.709/18: Art. 7 (base legal), Art. 46 (segurança), Art. 50 (registro).

### Anexo B: Glossário

| Termo | Definição |
|-------|-----------|
| **RAG** | Retrieval-Augmented Generation — recuperação semântica para enriquecer contexto de LLMs |
| **sqlite-vec** | Extensão SQLite para busca por similaridade de vetores |
| **MCP** | Model Context Protocol — protocolo de comunicação entre agentes AI e ferramentas externas |
| **Embedding** | Representação vetorial densa de texto em espaço semântico (384d) |
| **Chunk** | Segmento de documento indexado independentemente |
| **PII** | Personally Identifiable Information — dados pessoais sensíveis (CPF, nome, email) |
| **DLQ** | Dead Letter Queue — fila de rejeição para chunks que falham na validação |
| **QG-C11** | Quality Gate da Camada C11 — verificações obrigatórias para merge |

### Anexo C: Troubleshooting

| Sintoma | Causa Provável | Solução |
|---------|---------------|---------|
| `ModuleNotFoundError: sqlite_vec` | Deps não instaladas | `uv sync` |
| `sqlite-vec` extensão não carrega | wheels incompatível | `pip install sqlite-vec --force-reinstall` |
| Banco vazio após indexação | `DOCS_DIR` não encontrado | Verificar `PROJECT_ROOT` em `constants.py` |
| Query retorna 0 resultados | Índice não construído | Executar `make knowledge-index` |
| MCP server não responde | Transporte incorreto | Verificar `mcp.json` e `PYTHONPATH` |
| PII scan muito agressivo | Regex de email captura código | Ajustar `PII_PATTERNS` em `constants.py` |
| Reindexação lenta (> 5min) | Embedding em CPU sem batch | `model.encode(..., batch_size=32)` |

---

*Documento revisado para implementação compatível com o Project-Lewis v2.0.*  
*Arquiteto: Douglas Souza | Camada SDD: C11-Knowledge | Status: Aprovado*  
*Total de seções: 14 | ADRs: 5 | Módulos de código: 9 | Quality Gates: 9 | Checklist: 20 itens*
