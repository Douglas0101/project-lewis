# SDD — Project-Lewis Camada C11: Knowledge Layer (RAG + LangChain + Chroma + MCP)
## Documento de Implementação Extremamente Planejada — Blueprint Completo

**Documento:** SDD-C11-Knowledge-Impl-v1.0  
**Versão:** 1.0  
**Data:** 2026-06-27  
**Autor:** Douglas Souza (Arquiteto)  
**Projeto:** Project-Lewis — Pipeline ECG Edge ML  
**Status:** Aprovado para implementação imediata  
**Hardware Dev:** Lenovo IdeaPad 3 15ITL6 | Zorin OS 18.1  
**Python:** 3.12.x (system Python — veto 3.13+)  
**Stack:** uv (Astral), LangChain, Chroma, sentence-transformers, MCP stdio  

---

## ÍNDICE EXECUTIVO

| Seção | Conteúdo | Página |
|-------|----------|--------|
| 1. Contexto e Escopo | Por que RAG, por que agora, limites do contexto atual | — |
| 2. Decisões Arquiteturais (ADRs) | ADR-005 a ADR-008, todas ratificadas | — |
| 3. Stack Tecnológico Detalhado | Versões, constraints, vetos | — |
| 4. Estrutura de Diretórios | Árvore completa C11 | — |
| 5. Módulos de Implementação | Código completo: indexer, retriever, mcp_server, cli | — |
| 6. Pipeline de Indexação | Fluxo de dados, determinismo, idempotência | — |
| 7. Sistema de Retrieval | MMR, filtros 3D, reranking, formatação | — |
| 8. MCP Server | Protocolo stdio, schema JSON-RPC, tool definitions | — |
| 9. Testes e Quality Gates | QG-C11, pirâmide 70/20/10, fixtures | — |
| 10. Segurança e LGPD | Superfície de ataque, mitigações, compliance | — |
| 11. DevOps e CI/CD | Makefile targets, pre-commit, Docker | — |
| 12. Roadmap de Implementação | 5 fases, entregáveis, verificação | — |
| 13. Checklist de Aceite | Binário passa/falha para merge | — |
| 14. Anexos | Referências, glossário, troubleshooting | — |

---

## 1. CONTEXTO E ESCOPO

### 1.1 Diagnóstico do Estado Atual

O Project-Lewis possui 16 documentos técnicos markdown (Camadas 01–09, SDDs, PRDs, Especificações) totalizando ~150KB de texto puro, além de ~8.000 linhas de código Python e ~3.500 linhas de C/C++ no firmware. Os agentes de coding (Kimi Code, OpenCode) operam com **context windows limitadas** (~128k–200k tokens). Quando o agente é solicitado a implementar `src/models/stage1_binary.py`, ele não carrega automaticamente `Camada-04-Modelagem-v1.1.md` na memória — a menos que o arquivo esteja aberto no editor. Isso gera:

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

### 1.3 Escopo Inclui

- Indexação semântica de docs, src Python e firmware C.
- Vector DB local (Chroma/SQLite) com embeddings CPU-only.
- MCP server stdio com 3 tools: `search_docs`, `list_layers`, `get_doc_by_source`.
- CLI interno (`uv run python -m src.knowledge.cli`) para reindexação e query manual.
- Testes pytest com fixtures parametrizadas.
- Quality Gate QG-C11 com thresholds quantitativos.

### 1.4 Escopo Exclui

- UI web/frontend (veto Radix UI; headless por design).
- Integração com MLflow, Weights & Biases, ou experiment trackers externos.
- Indexação de dados brutos de ECG (`.dat`, `.mat`, `.hea`) — LGPD proibitivo.
- Deploy em cloud ou servidor remoto.
- Fine-tuning de embedding models (uso de modelo pré-treinado frozen).

---

## 2. DECISÕES ARQUITETURAIS (ADRs)

### ADR-005: Vector DB Local vs. Remoto

**Contexto:** Vector DBs cloud (Pinecone, Weaviate Cloud, Qdrant Cloud) oferecem escalabilidade, mas exigem conectividade, credenciais, e custo recorrente. O Project-Lewis é desenvolvido em um IdeaPad 3 sem GPU, frequentemente offline.

**Decisão:** Adotar **Chroma** com persistência em SQLite local (`data/chroma_db/`).

**Consequências:**
- (+) Zero infraestrutura, zero custo, 100% offline.
- (+) Backup trivial: `tar czf chroma_backup.tar.gz data/chroma_db/`.
- (+) Integração nativa com LangChain (`langchain-chroma`).
- (-) Não escala para > 1M documentos (não é o caso: ~500 chunks esperados).
- (-) Single-writer; não suporta concorrência de escrita (aceitável para uso individual).

### ADR-006: Embedding Model CPU-Only

**Contexto:** O IdeaPad 3 não possui GPU dedicada. Modelos de embedding como `OpenAI text-embedding-3` exigem API key e internet. Modelos open-source como `BAAI/bge-large-en` exigem ~1GB RAM e são lentos em CPU.

**Decisão:** Adotar `sentence-transformers/all-MiniLM-L6-v2` (384 dimensões, ~80MB, CPU-optimized, multilíngue funcional para PT-BR técnico).

**Consequências:**
- (+) 50–100ms por query em CPU i5/i7.
- (+) Qualidade suficiente para documentação técnica estruturada.
- (+) Offline, sem API key.
- (-) Inferior a `bge-large` ou `e5-large` para nuances semânticas profundas.
- (-) Português brasileiro não é o idioma nativo de treinamento; keywords técnicas em inglês são melhor representadas.

**Mitigação:** Indexar documentos com tags bilíngues (PT-BR + EN técnico) e usar `normalize_embeddings=True`.

### ADR-007: MCP stdio vs. HTTP/SSE

**Contexto:** MCP (Model Context Protocol) suporta transporte stdio (subprocesso local) ou HTTP/SSE (servidor remoto). Kimi Code e OpenCode suportam ambos.

**Decisão:** Transporte **stdio** via `uv run python -m src.knowledge.mcp_server`.

**Consequências:**
- (+) Nenhuma porta de rede exposta — superfície de ataque mínima.
- (+) O agente gerencia o lifecycle do processo (start/stop automático).
- (+) Não requer systemd, docker-compose, ou gerenciamento de processos.
- (-) Comunicação unidirecional por linha; não suporta streaming de progresso.
- (-) O processo morre quando o agente fecha (aceitável: reindexação é rápida).

### ADR-008: Metadados 3D (Layer, Version, Tags)

**Contexto:** Documentação do Project-Lewis é hierárquica por camada (C01–C10) e versionada (v1.1, v2.0). Queries genéricas retornam chunks irrelevantes (ex: buscar "quantização" e receber resultados da Camada 01-Ingestão).

**Decisão:** Cada chunk armazena metadados estruturados em 3 eixos:
- **D1 — layer:** `C01`, `C02`, ..., `C10`, `SDD`, `PRD`, `UNIFIED`, `ESPECIFICACAO`.
- **D2 — version:** `v1.1`, `v2.0`, `v1.4`, `unversioned`.
- **D3 — tags:** lista de tags semânticas extraídas heurísticamente do conteúdo.

**Consequências:**
- (+) Filtros precisos: `search_docs(query, layer="C04", version="v2.0")`.
- (+) O agente pode navegar intencionalmente pela arquitetura.
- (-) Heurística de extração de tags requer manutenção manual conforme novos documentos são adicionados.

---

## 3. STACK TECNOLÓGICO DETALHADO

### 3.1 Dependências Produtivas

| Pacote | Versão | Função | Justificativa |
|--------|--------|--------|---------------|
| `langchain` | `>=0.3.0,<0.4.0` | Framework RAG | Padrão de mercado; LCEL para chains declarativas |
| `langchain-community` | `>=0.3.0,<0.4.0` | Loaders e embeddings | `HuggingFaceEmbeddings`, `UnstructuredMarkdownLoader` |
| `langchain-chroma` | `>=0.3.0,<0.4.0` | VectorStore integration | API nativa Chroma, suporte a filtros metadata |
| `langchain-text-splitters` | built-in | Splitting hierárquico | `MarkdownHeaderTextSplitter`, `RecursiveCharacterTextSplitter` |
| `chromadb` | `>=0.6.0,<1.0.0` | Vector DB local | SQLite persistente, zero infra, filtro por metadata |
| `sentence-transformers` | `>=3.0.0,<4.0.0` | Embedding model | `all-MiniLM-L6-v2`, CPU-optimized, 80MB |
| `typer` | `>=0.12.0,<1.0.0` | CLI interno | Commands para reindex e query manual |
| `markdown` | `>=3.6.0,<4.0.0` | Parsing de frontmatter | Extração de YAML header em docs |

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
| C04 | Nenhum dado de ECG ou PII no índice | 🔴 Bloqueante | Regex scan em `test_lgpd_compliance` |
| C05 | Chroma local apenas (não cloud) | 🟡 Alto | Assert `persist_directory` local |
| C06 | CPU-only (sem CUDA) | 🟡 Alto | `model_kwargs={"device": "cpu"}` |
| C07 | MCP stdio (não HTTP) | 🟡 Alto | `mcp.json` config |
| C08 | Docstrings em PT-BR | 🟢 Baixo | `flake8` + review manual |

---

## 4. ESTRUTURA DE DIRETÓRIOS — CAMADA C11

```
Project-Lewis/
├── src/
│   ├── knowledge/                    # NOVO — Camada C11
│   │   ├── __init__.py
│   │   ├── indexer.py                 # Indexação semântica
│   │   ├── retriever.py               # Retrieval com filtros 3D
│   │   ├── mcp_server.py             # MCP server stdio
│   │   ├── cli.py                     # CLI typer (reindex, query)
│   │   ├── schemas.py                 # Pydantic models (DocumentMeta, QueryResult)
│   │   ├── constants.py               # Paths, defaults, tag keywords
│   │   └── utils.py                   # Helpers (hash, regex PII, etc.)
│   ├── data/
│   ├── features/
│   ├── models/
│   └── quantization/
├── tests/
│   ├── test_knowledge/               # NOVO — Testes C11
│   │   ├── conftest.py               # Fixtures (vectorstore, sample docs)
│   │   ├── test_indexer.py           # QG-C11-01: indexação
│   │   ├── test_retriever.py         # QG-C11-02: retrieval
│   │   ├── test_mcp_server.py        # QG-C11-03: MCP protocol
│   │   ├── test_lgpd_compliance.py   # QG-C11-04: zero PII
│   │   └── test_integration.py       # QG-C11-05: end-to-end
├── data/
│   ├── chroma_db/                    # NOVO — Vector DB SQLite (gitignored)
│   └── lineage/
│       └── knowledge/                # NOVO — Linhagem de indexação
├── logs/
│   └── knowledge_queries.jsonl       # NOVO — Audit trail de queries
├── scripts/
│   └── validate_knowledge_index.py   # NOVO — Validação pós-indexação
├── config/
│   └── knowledge_v1.0.yaml           # NOVO — Config da Camada C11
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
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
LINEAGE_DIR = PROJECT_ROOT / "data" / "lineage" / "knowledge"
LOG_QUERIES = PROJECT_ROOT / "logs" / "knowledge_queries.jsonl"
CONFIG_PATH = PROJECT_ROOT / "config" / "knowledge_v1.0.yaml"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
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
```

### 5.2 `src/knowledge/schemas.py`

```python
"""Schemas Pydantic para a Camada C11.

Autor: Douglas Souza
Data: 2026-06-27
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class DocumentMeta(BaseModel):
    """Metadados 3D de um chunk indexado."""

    source: str = Field(..., description="Caminho relativo do arquivo fonte")
    layer: str = Field(..., description="D1: Camada arquitetural (C01-C10, SDD, PRD, etc.)")
    version: str = Field(..., description="D2: Versão do documento (v1.1, v2.0, unversioned)")
    tags: List[str] = Field(default_factory=list, description="D3: Tags semânticas extraídas")
    filename: str = Field(..., description="Nome do arquivo")
    chunk_id: str = Field(..., description="Hash determinístico do chunk")
    header_1: Optional[str] = Field(None, description="H1 do markdown, se presente")
    header_2: Optional[str] = Field(None, description="H2 do markdown, se presente")

    @validator("layer")
    def validate_layer(cls, v: str) -> str:
        allowed = {
            "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10",
            "SDD", "PRD", "UNIFIED", "ESPECIFICACAO", "SIMULATION", "DEBITO_TECNICO", "GENERAL",
        }
        if v not in allowed:
            raise ValueError(f"Layer '{v}' não é válida. Valores permitidos: {allowed}")
        return v


class QueryRequest(BaseModel):
    """Request de query para o retriever."""

    query: str = Field(..., min_length=1, max_length=4096)
    layer: Optional[str] = Field(None, description="Filtro D1")
    version: Optional[str] = Field(None, description="Filtro D2")
    tags: Optional[List[str]] = Field(None, description="Filtro D3 (AND lógico)")
    k: int = Field(5, ge=1, le=20)
    mmr: bool = Field(True, description="Usar Maximum Marginal Relevance")
    fetch_k: int = Field(20, ge=5, le=100)


class QueryResult(BaseModel):
    """Resultado de uma query."""

    chunk_id: str
    source: str
    layer: str
    version: str
    tags: List[str]
    content: str
    score: float = Field(..., description="Score de similaridade ou MMR")
    rank: int = Field(..., ge=1)


class IndexLineage(BaseModel):
    """Registro de linhagem de uma indexação."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_files: int
    total_chunks: int
    layers_found: List[str]
    versions_found: List[str]
    index_duration_sec: float
    chroma_dir_size_bytes: int
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

from .constants import PII_PATTERNS, TAG_KEYWORDS


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
        _, frontmatter, body = content.split("---", 2)
        meta = {}
        for line in frontmatter.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        return meta
    except ValueError:
        return {}
```

### 5.4 `src/knowledge/indexer.py`

```python
"""Indexador semântico da Camada C11.

Responsabilidade: carregar documentos markdown e código-fonte,
segmentar preservando hierarquia, extrair metadados 3D,
gerar embeddings e persistir no Chroma.

Restrições:
- Não indexar arquivos fora de docs/, src/, firmware/src/.
- Rejeitar chunks contendo PII (LGPD).
- Determinístico: reindexação gera IDs idênticos.
- Nunca indexar dados brutos de ECG (.dat, .mat, .hea).

Autor: Douglas Souza
Data: 2026-06-27
"""

import json
import time
from pathlib import Path
from typing import Dict, List

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    Language,
)

from .constants import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DOCS_DIR,
    EMBEDDING_MODEL,
    FIRMWARE_DIR,
    LAYER_MAP,
    LINEAGE_DIR,
    MAX_CHUNK_CHARS,
    SRC_DIR,
)
from .schemas import DocumentMeta, IndexLineage
from .utils import (
    compute_config_hash,
    detect_pii,
    deterministic_chunk_id,
    extract_tags_from_content,
    parse_markdown_frontmatter,
)


def resolve_layer(file_path: Path) -> str:
    """Resolve camada arquitetural a partir do nome do arquivo."""
    name = file_path.name
    for prefix, layer in LAYER_MAP.items():
        if prefix in name:
            return layer
    return "GENERAL"


def resolve_version(file_path: Path) -> str:
    """Extrai versão do nome do arquivo (vX.Y)."""
    import re
    match = re.search(r"v(\d+\.\d+)", file_path.name)
    return f"v{match.group(1)}" if match else "unversioned"


def load_markdown_documents() -> List[Document]:
    """Carrega todos os arquivos markdown do projeto."""
    docs: List[Document] = []
    md_paths = list(DOCS_DIR.rglob("*.md")) + list(Path(CHROMA_DIR.parent.parent).glob("*.md"))

    for md_path in md_paths:
        if not md_path.exists():
            continue
        raw = md_path.read_text(encoding="utf-8")
        meta = parse_markdown_frontmatter(raw)
        docs.append(Document(page_content=raw, metadata={
            "source": md_path.relative_to(Path(CHROMA_DIR.parent.parent)).as_posix(),
            "file_type": "markdown",
            **meta,
        }))
    return docs


def load_code_documents() -> List[Document]:
    """Carrega código-fonte Python e C como documentos."""
    docs: List[Document] = []

    for py_path in SRC_DIR.rglob("*.py"):
        if "__pycache__" in str(py_path):
            continue
        content = py_path.read_text(encoding="utf-8")
        docs.append(Document(page_content=content, metadata={
            "source": py_path.relative_to(Path(CHROMA_DIR.parent.parent)).as_posix(),
            "file_type": "python",
            "language": "python",
        }))

    for ext in ("*.c", "*.h", "*.cpp"):
        for c_path in FIRMWARE_DIR.rglob(ext):
            content = c_path.read_text(encoding="utf-8")
            docs.append(Document(page_content=content, metadata={
                "source": c_path.relative_to(Path(CHROMA_DIR.parent.parent)).as_posix(),
                "file_type": "c",
                "language": "c",
            }))

    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """Segmenta documentos preservando hierarquia."""
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ],
        strip_headers=False,
    )
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    c_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.C,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    final_chunks: List[Document] = []

    for doc in docs:
        file_type = doc.metadata.get("file_type", "markdown")

        if file_type == "markdown":
            chunks = md_splitter.split_text(doc.page_content)
            for chunk in chunks:
                if len(chunk.page_content) > MAX_CHUNK_CHARS:
                    sub_chunks = fallback_splitter.split_documents([chunk])
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk)
        elif file_type == "python":
            final_chunks.extend(python_splitter.split_documents([doc]))
        elif file_type == "c":
            final_chunks.extend(c_splitter.split_documents([doc]))
        else:
            final_chunks.extend(fallback_splitter.split_documents([doc]))

    return final_chunks


def enrich_metadata(chunks: List[Document]) -> List[Document]:
    """Enriquece cada chunk com metadados 3D e rejeita PII."""
    enriched: List[Document] = []

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        file_path = Path(source)
        content = chunk.page_content

        pii_matches = detect_pii(content)
        if pii_matches:
            print(f"[WARN] Chunk rejeitado por PII: {source} — matches: {pii_matches}")
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
            header_1=chunk.metadata.get("header_1"),
            header_2=chunk.metadata.get("header_2"),
        )

        chunk.metadata.update(meta.model_dump())
        chunk.metadata["chunk_id"] = chunk_id
        enriched.append(chunk)

    return enriched


def build_index() -> IndexLineage:
    """Pipeline completo de indexação. Retorna linhagem."""
    start_time = time.perf_counter()
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    LINEAGE_DIR.mkdir(parents=True, exist_ok=True)

    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    md_docs = load_markdown_documents()
    code_docs = load_code_documents()
    all_docs = md_docs + code_docs

    chunks = split_documents(all_docs)
    chunks = enrich_metadata(chunks)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        ids=[c.metadata["chunk_id"] for c in chunks],
    )

    duration = time.perf_counter() - start_time

    layers = sorted({c.metadata["layer"] for c in chunks})
    versions = sorted({c.metadata["version"] for c in chunks})
    chroma_size = sum(f.stat().st_size for f in CHROMA_DIR.rglob("*") if f.is_file())

    lineage = IndexLineage(
        total_files=len(all_docs),
        total_chunks=len(chunks),
        layers_found=layers,
        versions_found=versions,
        index_duration_sec=duration,
        chroma_dir_size_bytes=chroma_size,
        config_hash=compute_config_hash(Path(__file__)),
    )

    lineage_path = LINEAGE_DIR / f"index_{lineage.timestamp.isoformat()}.json"
    lineage_path.write_text(lineage.model_dump_json(indent=2), encoding="utf-8")

    print(f"[indexer] {len(chunks)} chunks indexados em {duration:.2f}s")
    print(f"[indexer] Camadas: {layers}")
    print(f"[indexer] Versões: {versions}")
    print(f"[indexer] Tamanho Chroma: {chroma_size / 1024 / 1024:.2f} MB")

    return lineage


if __name__ == "__main__":
    build_index()
```

### 5.5 `src/knowledge/retriever.py`

```python
"""Retriever inteligente da Camada C11.

Responsabilidade: busca semântica com filtros 3D, MMR,
formatação para agentes e logging de audit trail.

Autor: Douglas Souza
Data: 2026-06-27
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from .constants import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, LOG_QUERIES
from .schemas import QueryRequest, QueryResult
from .utils import format_context_for_agent


def _get_vectorstore() -> Chroma:
    """Factory do vectorstore."""
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding,
        collection_name=COLLECTION_NAME,
    )


def _build_chroma_filter(req: QueryRequest) -> Optional[Dict]:
    """Constrói filtro Chroma a partir de QueryRequest."""
    filter_dict: Dict = {}
    if req.layer:
        filter_dict["layer"] = {"$eq": req.layer}
    if req.version:
        filter_dict["version"] = {"$eq": req.version}
    if req.tags:
        filter_dict["tags"] = {"$contains": req.tags[0]}
    return filter_dict if filter_dict else None


def search(req: QueryRequest) -> List[QueryResult]:
    """Executa busca semântica com filtros e MMR."""
    vstore = _get_vectorstore()
    chroma_filter = _build_chroma_filter(req)

    retriever: VectorStoreRetriever = vstore.as_retriever(
        search_type="mmr" if req.mmr else "similarity",
        search_kwargs={
            "k": req.k,
            "fetch_k": req.fetch_k,
            "lambda_mult": 0.7,
            "filter": chroma_filter,
        },
    )

    docs = retriever.invoke(req.query)

    if req.tags and len(req.tags) > 1:
        docs = [
            d for d in docs
            if all(t in json.loads(d.metadata.get("tags", "[]")) for t in req.tags)
        ]

    results = []
    for i, doc in enumerate(docs[: req.k], 1):
        meta = doc.metadata
        tags = json.loads(meta.get("tags", "[]"))
        results.append(QueryResult(
            chunk_id=meta.get("chunk_id", "unknown"),
            source=meta.get("source", "unknown"),
            layer=meta.get("layer", "GENERAL"),
            version=meta.get("version", "unversioned"),
            tags=tags,
            content=doc.page_content,
            score=0.0,
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

### 5.6 `src/knowledge/mcp_server.py`

```python
"""MCP Server stdio para integração Kimi Code / OpenCode.

Protocolo: JSON-RPC 2.0 via stdin/stdout.
Tools expostas:
  - search_docs(query, layer?, version?, tags?, k?, mmr?)
  - list_layers()
  - get_doc_by_source(source)

Autor: Douglas Souza
Data: 2026-06-27
"""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional

from .retriever import search
from .schemas import QueryRequest


async def handle_request(req: Dict[str, Any]) -> Dict[str, Any]:
    """Handler JSON-RPC 2.0."""
    method = req.get("method")
    params = req.get("params", {})
    req_id = req.get("id")

    if method == "initialize":
        return _ok(req_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "project-lewis-knowledge", "version": "1.0.0"},
        })

    if method == "tools/list":
        return _ok(req_id, {
            "tools": [
                {
                    "name": "search_docs",
                    "description": "Busca semântica na documentação do Project-Lewis",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Pergunta ou keyword"},
                            "layer": {"type": "string", "description": "Filtro camada (C04, SDD, etc.)"},
                            "version": {"type": "string", "description": "Filtro versão (v2.0)"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "k": {"type": "integer", "default": 5},
                            "mmr": {"type": "boolean", "default": True},
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "list_layers",
                    "description": "Lista camadas arquiteturais disponíveis",
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "name": "get_doc_by_source",
                    "description": "Recupera chunks por caminho de arquivo",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "description": "Caminho relativo do arquivo"},
                            "k": {"type": "integer", "default": 3},
                        },
                        "required": ["source"],
                    },
                },
            ]
        })

    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name == "search_docs":
            req = QueryRequest(
                query=arguments.get("query", ""),
                layer=arguments.get("layer"),
                version=arguments.get("version"),
                tags=arguments.get("tags"),
                k=arguments.get("k", 5),
                mmr=arguments.get("mmr", True),
            )
            results = search(req)
            return _ok(req_id, {
                "context": "\n---\n".join(
                    f"[{r.rank}] {r.source} (Camada {r.layer}, {r.version})\n{r.content}"
                    for r in results
                ),
                "sources": list({r.source for r in results}),
                "count": len(results),
            })

        if tool_name == "list_layers":
            layers = [
                "C01", "C02", "C03", "C04", "C05", "C06",
                "C07", "C08", "C09", "C10", "SDD", "PRD", "UNIFIED",
            ]
            return _ok(req_id, {"layers": layers})

        if tool_name == "get_doc_by_source":
            req = QueryRequest(
                query="*",
                k=arguments.get("k", 3),
            )
            return _ok(req_id, {"message": "Implementação requer extensão do retriever"})

        return _error(req_id, -32601, f"Tool '{tool_name}' não encontrada")

    return _error(req_id, -32601, f"Method '{method}' não encontrado")


def _ok(req_id: Optional[Any], result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id: Optional[Any], code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


async def main() -> None:
    """Loop principal stdio."""
    loop = asyncio.get_event_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        try:
            req = json.loads(line)
            resp = await handle_request(req)
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except json.JSONDecodeError:
            continue


if __name__ == "__main__":
    asyncio.run(main())
```

### 5.7 `src/knowledge/cli.py`

```python
"""CLI interno da Camada C11.

Comandos:
  reindex    — Rebuild completo do índice Chroma
  query      — Query interativa via terminal
  status     — Status do índice (chunks, camadas, tamanho)

Autor: Douglas Souza
Data: 2026-06-27
"""

from pathlib import Path
from typing import Optional

import typer

from .constants import CHROMA_DIR, LINEAGE_DIR
from .indexer import build_index
from .retriever import search
from .schemas import QueryRequest

app = typer.Typer(help="CLI da Camada C11 — Knowledge Layer")


@app.command()
def reindex() -> None:
    """Rebuild completo do índice semântico."""
    typer.echo("Iniciando reindexação...")
    lineage = build_index()
    typer.echo(f"✅ {lineage.total_chunks} chunks indexados em {lineage.index_duration_sec:.2f}s")


@app.command()
def query(
    q: str = typer.Argument(..., help="Pergunta ou keyword"),
    layer: Optional[str] = typer.Option(None, "--layer", "-l", help="Filtro camada"),
    version: Optional[str] = typer.Option(None, "--version", "-v", help="Filtro versão"),
    k: int = typer.Option(5, "--k", "-k", help="Número de resultados"),
) -> None:
    """Executa busca semântica e exibe resultados."""
    req = QueryRequest(query=q, layer=layer, version=version, k=k)
    results = search(req)
    for r in results:
        typer.echo(f"\n[{r.rank}] {r.source} | {r.layer} | {r.version}")
        typer.echo(f"Tags: {', '.join(r.tags)}")
        typer.echo(f"{r.content[:300]}...")


@app.command()
def status() -> None:
    """Exibe status do índice."""
    if not CHROMA_DIR.exists():
        typer.echo("❌ Índice não encontrado. Execute: uv run python -m src.knowledge.cli reindex")
        raise typer.Exit(1)

    size_mb = sum(f.stat().st_size for f in CHROMA_DIR.rglob("*") if f.is_file()) / 1024 / 1024
    lineage_files = sorted(LINEAGE_DIR.glob("index_*.json")) if LINEAGE_DIR.exists() else []

    typer.echo(f"📦 Diretório Chroma: {CHROMA_DIR}")
    typer.echo(f"📊 Tamanho: {size_mb:.2f} MB")
    typer.echo(f"📄 Registros de linhagem: {len(lineage_files)}")
    if lineage_files:
        latest = lineage_files[-1]
        typer.echo(f"🕐 Última indexação: {latest.name}")


if __name__ == "__main__":
    app()
```

### 5.8 `config/knowledge_v1.0.yaml`

```yaml
# Configuração da Camada C11 — Knowledge Layer
# Project-Lewis v1.0

indexer:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  embedding_dim: 384
  chunk_size: 512
  chunk_overlap: 64
  max_chunk_chars: 1024
  collection_name: "project_lewis_knowledge"

retriever:
  default_k: 5
  default_fetch_k: 20
  mmr_lambda_mult: 0.7
  enable_mmr: true

mcp:
  transport: "stdio"
  protocol_version: "2024-11-05"
  server_name: "project-lewis-knowledge"
  server_version: "1.0.0"

lgpd:
  pii_scan_enabled: true
  reject_on_pii_match: true
  allowed_file_extensions:
    - ".md"
    - ".py"
    - ".c"
    - ".h"
    - ".cpp"
  forbidden_patterns:
    - "*.dat"
    - "*.mat"
    - "*.hea"
    - "*.atr"
    - "raw_chapman/"
    - "raw_mitbih/"
    - "raw_svdb/"
    - "raw_afdb/"
    - "raw_incart/"
```

---

## 6. PIPELINE DE INDEXAÇÃO

### 6.1 Fluxo de Dados

```mermaid
flowchart LR
    A[docs/*.md] --> B[MarkdownHeaderTextSplitter]
    C[src/**/*.py] --> D[PythonSplitter]
    E[firmware/src/**/*.{c,h,cpp}] --> F[CSplitter]
    B --> G[Enrich Metadata 3D]
    D --> G
    F --> G
    G --> H{PII Scan?}
    H -->|Rejeita| I[DLQ: rejected_chunks.jsonl]
    H -->|Aprova| J[Chroma.from_documents]
    J --> K[(data/chroma_db/ SQLite)]
    J --> L[data/lineage/knowledge/index_*.json]
```

### 6.2 Determinismo e Idempotência

| Propriedade | Garantia | Mecanismo |
|-------------|----------|-----------|
| IDs determinísticos | ✅ | `SHA256(source + content[:200])[:16]` |
| Reindexação idêntica | ✅ | Mesmo input → mesmos IDs → Chroma sobrescreve |
| Cache de config | ✅ | `config_hash` na linhagem; reindexação forçada se mudar |
| Ordem de chunks | ✅ | Ordenação por source + header hierarchy |

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

1. **Query Embedding:** `query → all-MiniLM-L6-v2 → vetor 384d`.
2. **Filtro Metadata:** Aplica `$eq` em `layer`, `version`; `$contains` em `tags`.
3. **Pool Inicial:** Recupera `fetch_k` (default 20) vizinhos por similaridade cosseno.
4. **MMR Rerank:** Se `mmr=True`, aplica Maximum Marginal Relevance com `lambda_mult=0.7`:
   - 70% relevância (similaridade à query).
   - 30% diversidade (dissimilaridade entre chunks selecionados).
5. **Pós-filtragem:** Aplica AND lógico em múltiplas tags (workaround Chroma).
6. **Corte:** Retorna top-`k` (default 5).

### 7.2 Formatação para Agentes

O MCP server formata o contexto como blocos numerados com metadados inline:

```
[1] Fonte: docs/Camada-04-Modelagem-v1.1.md | Camada: C04 | Versão: v1.1 | Tags: quantizacao, ml
O input shape do modelo deve ser consistente com a segmentação da Camada 2...
---
[2] Fonte: src/models/backbone_1d.py | Camada: GENERAL | Versão: unversioned | Tags: ml, firmware
Input shape: (500, 1) — 1000ms @ 500Hz, 1 canal...
```

Isso permite que o agente cite fontes e verifique versionamento.

---

## 8. MCP SERVER — ESPECIFICAÇÃO DO PROTOCOLO

### 8.1 Transporte

- **Tipo:** stdio (subprocesso).
- **Comando de ativação:** `uv run python -m src.knowledge.mcp_server`.
- **Lifecycle:** Gerenciado pelo cliente (Kimi Code / OpenCode); inicia on-demand, morre ao fechar.

### 8.2 Schema JSON-RPC

| Método | Params | Retorno | Descrição |
|--------|--------|---------|-----------|
| `initialize` | — | `{protocolVersion, capabilities, serverInfo}` | Handshake |
| `tools/list` | — | `{tools: [...]}` | Lista tools disponíveis |
| `tools/call` | `{name, arguments}` | `{content, sources, count}` | Executa tool |
| `search_docs` | `{query, layer?, version?, tags?, k?, mmr?}` | Contexto formatado | Busca semântica |
| `list_layers` | — | `{layers: [...]}` | Camadas disponíveis |

### 8.3 Configuração `mcp.json` (Atualizada)

```json
{
  "mcpServers": {
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
| Unit | 70% | `test_indexer.py`, `test_retriever.py` | < 30s |
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
| QG-C11-07 | Tamanho do índice | < 500 MB | `test_indexer::test_size` | Merge |
| QG-C11-08 | CLI funcional | `reindex`, `query`, `status` passam | `test_integration::test_cli` | Merge |

### 9.3 Fixtures de Teste (`tests/test_knowledge/conftest.py`)

```python
import pytest
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

@pytest.fixture(scope="session")
def test_chroma_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("chroma_test")

@pytest.fixture(scope="session")
def test_embedding():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

@pytest.fixture
def sample_docs():
    from langchain_core.documents import Document
    return [
        Document(page_content="Threshold QG5 para F1-macro em inter-patient é > 0.55", metadata={"source": "docs/C04.md", "layer": "C04", "version": "v2.0"}),
        Document(page_content="O STM32F4 possui 192KB SRAM e 1MB Flash", metadata={"source": "docs/C08.md", "layer": "C08", "version": "v1.1"}),
    ]
```

### 9.4 Exemplo de Teste Unitário

```python
# tests/test_knowledge/test_retriever.py
import pytest
from src.knowledge.schemas import QueryRequest
from src.knowledge.retriever import search

@pytest.mark.qg_c11
class TestRetriever:
    def test_search_basic(self, populated_vectorstore):
        req = QueryRequest(query="threshold F1-macro QG5", k=3)
        results = search(req)
        assert len(results) > 0
        assert any("C04" in r.layer for r in results)

    def test_layer_filter(self, populated_vectorstore):
        req = QueryRequest(query="STM32", layer="C08", k=3)
        results = search(req)
        assert all(r.layer == "C08" for r in results)

    def test_mmr_diversity(self, populated_vectorstore):
        req = QueryRequest(query="quantizacao", k=5, mmr=True, fetch_k=20)
        results = search(req)
        sources = [r.source for r in results]
        assert len(set(sources)) >= 2
```

---

## 10. SEGURANÇA E LGPD

### 10.1 Superfície de Ataque

| Vetor | Risco | Mitigação | Status |
|-------|-------|-----------|--------|
| **PII no índice** | Alto — exposição de dados de saúde | Regex scan pré-indexação; rejeição automática | Implementado |
| **Acesso ao SQLite** | Médio — leitura local do disco | Chroma roda em `data/chroma_db/` (user-only permissions) | Por design |
| **MCP stdio injection** | Baixo — input malicioso via stdin | JSON parse strict; sem execução dinâmica | Implementado |
| **Path traversal** | Baixo — `source` metadata | Validação de caminho relativo; nunca absoluto | Implementado |
| **DDoS do retriever** | Baixo — queries massivas | `k` limitado a 20; rate limiting implícito (CPU) | Implementado |

### 10.2 Compliance LGPD

- **Art. 7 (Base legal):** O índice contém apenas documentação técnica (base legítima: execução de contrato de desenvolvimento).
- **Art. 46 (Segurança):** Nenhum dado pessoal sensível (saúde) é indexado. PII scan é obrigatório.
- **Art. 50 (Registro de operações):** `logs/knowledge_queries.jsonl` registra queries (sem conteúdo sensível) para auditoria.
- **Anonimização:** Se um chunk de código Python contiver string hardcoded de nome/paciente (ex: comentário), o PII scan rejeita.

---

## 11. DEVOPS E CI/CD

### 11.1 Targets Makefile (Adições)

```makefile
# Camada C11 — Knowledge Layer
knowledge-index:
	@echo "[C11] Reindexando knowledge base..."
	uv run python -m src.knowledge.cli reindex

knowledge-query:
	@read -p "Query: " q; 	uv run python -m src.knowledge.cli query "$$q"

knowledge-status:
	uv run python -m src.knowledge.cli status

knowledge-test:
	uv run pytest tests/test_knowledge/ -v --tb=short

knowledge-clean:
	rm -rf data/chroma_db/
	rm -rf data/lineage/knowledge/
	rm -f logs/knowledge_queries.jsonl

knowledge-validate:
	uv run python scripts/validate_knowledge_index.py
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

### 11.3 Docker (Sem Alterações)

O Dockerfile existente já cobre Python 3.12 + uv. Adicionar ao `pyproject.toml` as deps RAG é suficiente; o container reproduzirá o ambiente identicamente.

---

## 12. ROADMAP DE IMPLEMENTAÇÃO

### Fase 1: Fundação (Dia 1–2)

| Task | Entregável | Verificação | Owner |
|------|-----------|-------------|-------|
| Adicionar deps ao `pyproject.toml` | `uv.lock` atualizado | `uv sync` passa | DevOps |
| Criar estrutura de diretórios C11 | `src/knowledge/` presente | `ls -R src/knowledge` | Engenheiro |
| Implementar `constants.py` + `schemas.py` | Arquivos com tipagem completa | `mypy src/knowledge/` passa | Engenheiro |
| Implementar `utils.py` | PII scan funcional | `pytest tests/test_knowledge/test_lgpd.py` | Engenheiro |

### Fase 2: Indexação (Dia 3–4)

| Task | Entregável | Verificação | Owner |
|------|-----------|-------------|-------|
| Implementar `indexer.py` | Indexação completa | `make knowledge-index` gera `data/chroma_db/` | Engenheiro |
| Testar com docs reais | > 100 chunks gerados | `make knowledge-status` | Engenheiro |
| Validar determinismo | Reindexação idêntica | `pytest tests/test_knowledge/test_indexer.py` | QA |

### Fase 3: Retrieval (Dia 5–6)

| Task | Entregável | Verificação | Owner |
|------|-----------|-------------|-------|
| Implementar `retriever.py` | Busca com filtros | `make knowledge-query "threshold QG5"` | Engenheiro |
| Implementar `cli.py` | CLI funcional | `typer --help` | Engenheiro |
| Testes de precisão | MRR@5 > 0.80 | `pytest tests/test_knowledge/test_retriever.py` | QA |

### Fase 4: MCP (Dia 7–8)

| Task | Entregável | Verificação | Owner |
|------|-----------|-------------|-------|
| Implementar `mcp_server.py` | Server stdio | `echo '{"method":"initialize"}' \| uv run python -m src.knowledge.mcp_server` | Engenheiro |
| Atualizar `mcp.json` | Config válida | Kimi Code reconhece tool | Arquiteto |
| Teste de integração | Tool call funcional | `pytest tests/test_knowledge/test_mcp_server.py` | QA |

### Fase 5: Quality Gates e Merge (Dia 9–10)

| Task | Entregável | Verificação | Owner |
|------|-----------|-------------|-------|
| Todos os QG-C11 passando | Badge verde | `make knowledge-test` | QA |
| Documentação atualizada | `SDD-C11-Knowledge-Impl-v1.0.md` | Revisão arquitetural | Arquiteto |
| Pre-commit passando | Zero falhas | `uv run pre-commit run --all-files` | DevOps |
| Merge para `main` | Commit semântico | `git log --oneline` | Arquiteto |

---

## 13. CHECKLIST DE ACEITE (Binário Passa/Falha)

```
[ ] pyproject.toml contém langchain, langchain-chroma, chromadb, sentence-transformers, typer
[ ] uv.lock atualizado e válido (uv sync passa em < 30s)
[ ] src/knowledge/ contém __init__.py, constants.py, schemas.py, utils.py, indexer.py, retriever.py, mcp_server.py, cli.py
[ ] config/knowledge_v1.0.yaml existe e é válido
[ ] make knowledge-index executa sem erro e gera data/chroma_db/
[ ] make knowledge-status reporta > 0 chunks e > 0 camadas
[ ] make knowledge-query "threshold QG5" retorna resultados da Camada C04
[ ] pytest tests/test_knowledge/ passa 100% (QG-C11-01 a QG-C11-08)
[ ] test_lgpd_compliance.py: 0 ocorrências de PII no índice
[ ] test_idempotent.py: reindexação gera IDs idênticos
[ ] mcp.json atualizado com project-lewis-knowledge server
[ ] Pre-commit passa (black, isort, flake8, mypy, bandit)
[ ] Makefile contém targets knowledge-index, knowledge-query, knowledge-status, knowledge-test, knowledge-clean
[ ] Documentação técnica (este SDD) revisada e aprovada pelo arquiteto
[ ] Nenhuma dependência de frontend adicionada (veto Radix UI respeitado)
[ ] Python 3.12 exclusivo (nenhuma feature de 3.13+)
[ ] LGPD: nenhum dado de ECG ou PII no índice Chroma
```

---

## 14. ANEXOS

### Anexo A: Referências Técnicas

- LangChain Chroma Integration (2026): `langchain-chroma` >=0.3.0, filtros metadata por `$eq`, `$contains`.
- Chroma SQLite Persistence (2026): Persistência em arquivo via `persist_directory`, single-writer.
- Sentence-Transformers `all-MiniLM-L6-v2` (2026): 384d, 80MB, CPU-optimized, ONNX exportável.
- MCP Protocol Specification (2024-11-05): JSON-RPC 2.0 via stdio, tools/list, tools/call.
- LGPD Lei 13.709/18: Art. 7 (base legal), Art. 46 (segurança), Art. 50 (registro).

### Anexo B: Glossário

| Termo | Definição |
|-------|-----------|
| **RAG** | Retrieval-Augmented Generation — recuperação semântica para enriquecer contexto de LLMs |
| **MMR** | Maximum Marginal Relevance — técnica de reranking que balanceia relevância e diversidade |
| **MCP** | Model Context Protocol — protocolo de comunicação entre agentes AI e ferramentas externas |
| **Embedding** | Representação vetorial densa de texto em espaço semântico (384d) |
| **Chunk** | Segmento de documento indexado independentemente |
| **PII** | Personally Identifiable Information — dados pessoais sensíveis (CPF, nome, email) |
| **DLQ** | Dead Letter Queue — fila de rejeição para chunks que falham na validação |
| **QG-C11** | Quality Gate da Camada C11 — verificações obrigatórias para merge |

### Anexo C: Troubleshooting

| Sintoma | Causa Provável | Solução |
|---------|---------------|---------|
| `ModuleNotFoundError: chromadb` | Deps não instaladas | `uv sync` |
| Chroma vazio após indexação | `DOCS_DIR` não encontrado | Verificar `PROJECT_ROOT` em `constants.py` |
| Query retorna 0 resultados | Índice não construído | Executar `make knowledge-index` |
| MCP server não responde | JSON malformado no stdin | Verificar newline (`\n`) no final do payload |
| PII scan muito agressivo | Regex de email captura código | Ajustar `PII_PATTERNS` em `constants.py` |
| Reindexação lenta (> 5min) | Embedding em CPU sem batch | Reduzir número de docs ou usar batching no `HuggingFaceEmbeddings` |

---

*Documento gerado para implementação imediata no Project-Lewis.*  
*Arquiteto: Douglas Souza | Camada SDD: C11-Knowledge | Status: Aprovado*  
*Total de seções: 14 | ADRs: 4 | Módulos de código: 8 | Quality Gates: 8 | Checklist: 17 itens*
