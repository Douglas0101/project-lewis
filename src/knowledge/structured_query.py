"""Ponte 2: consulta natural → SQL estruturado sobre lewis_metrics.db.

Abordagem híbrida determinística + validação:
- Catálogo de queries comuns mapeia perguntas por padrão para SQL parametrizado.
- SQL customizado passa por parser allowlist antes da execução.
- Execução sempre em conexão read-only do SQLite.
- Audit trail de toda consulta.

Autor: Douglas Souza
Data: 2026-06-30
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sqlparse
from pydantic import BaseModel, Field
from sqlalchemy import Engine, create_engine, text

from src.observability.metrics import LatencyTracker
from src.security.rls import RowLevelSecurity
from src.tracking.db import get_db_path

logger = logging.getLogger(__name__)

_AUDIT_LOG = Path(__file__).resolve().parents[2] / "logs" / "knowledge_structured_queries.jsonl"
_MAX_ROWS = 1000
_ALLOWED_TABLES = {
    "experiment",
    "run",
    "metric",
    "alert",
    "artifact",
    "hardware_snapshot",
    "experiment_timeline",
}
_FORBIDDEN_KEYWORDS = {
    "drop",
    "delete",
    "insert",
    "update",
    "replace",
    "truncate",
    "attach",
    "detach",
    "pragma",
}


class NaturalQueryRequest(BaseModel):
    """Request de consulta em linguagem natural."""

    question: str = Field(..., min_length=1, max_length=4096)
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    allow_custom_sql: bool = False
    max_rows: int = Field(default=_MAX_ROWS, ge=1, le=_MAX_ROWS)


class StructuredQueryResult(BaseModel):
    """Resultado de uma consulta estruturada."""

    question: str
    sql: str
    params: Dict[str, Any]
    columns: List[str]
    rows: List[Dict[str, Any]]
    row_count: int
    truncated: bool
    execution_time_ms: float
    source: str = Field(..., description="catalog | custom_sql | schema_help")
    audit_id: str


class SchemaColumn(BaseModel):
    """Descrição de uma coluna para o contexto de schema."""

    name: str
    type: str
    nullable: bool
    default: Optional[str] = None


class SchemaTable(BaseModel):
    """Descrição de uma tabela para o contexto de schema."""

    name: str
    description: str
    columns: List[SchemaColumn]
    sample_sql: str


class SchemaContext(BaseModel):
    """Contexto de schema completo para geração/validação de SQL."""

    tables: Dict[str, SchemaTable]
    relationships: List[str]
    business_rules: List[str]


@dataclass
class QueryPattern:
    """Padrão de pergunta natural mapeado para SQL parametrizado."""

    name: str
    description: str
    patterns: Sequence[str]
    sql: str
    param_schema: Dict[str, Any] = field(default_factory=dict)
    required_params: Sequence[str] = field(default_factory=list)


_QUERY_CATALOG: List[QueryPattern] = [
    QueryPattern(
        name="last_failed_runs",
        description="Últimas runs que falharam",
        patterns=[
            r"run.*falh",
            r"falhas recentes",
            r"treinamentos.*falh",
            r"last failed runs",
            r"runs?.*failed",
        ],
        sql="""
            SELECT r.id AS run_id, e.name AS experiment_name,
                   r.run_type, r.status, r.start_time, r.end_time
            FROM run r
            JOIN experiment e ON e.id = r.experiment_id
            WHERE r.status = 'failed'
            ORDER BY r.start_time DESC
            LIMIT :max_rows
        """,
    ),
    QueryPattern(
        name="runs_by_stage",
        description="Runs de um estágio específico",
        patterns=[
            r"runs? do est[áa]gio (\w+)",
            r"est[áa]gio (\w+) runs?",
            r"stage (\w+) runs?",
        ],
        sql="""
            SELECT r.id AS run_id, e.name AS experiment_name,
                   e.stage, r.run_type, r.status, r.start_time
            FROM run r
            JOIN experiment e ON e.id = r.experiment_id
            WHERE e.stage = :stage
            ORDER BY r.start_time DESC
            LIMIT :max_rows
        """,
        param_schema={"stage": "str"},
        required_params=["stage"],
    ),
    QueryPattern(
        name="alerts_by_severity",
        description="Alertas por severidade",
        patterns=[
            r"alertas (cr[ií]ticos|warning|info)",
            r"(cr[ií]ticos|warning|info) alertas?",
        ],
        sql="""
            SELECT a.id, a.severity, a.category, a.message,
                   r.id AS run_id, e.name AS experiment_name
            FROM alert a
            LEFT JOIN run r ON r.id = a.run_id
            LEFT JOIN experiment e ON e.id = a.experiment_id
            WHERE a.severity = :severity
            ORDER BY a.recorded_at DESC
            LIMIT :max_rows
        """,
        param_schema={"severity": "str"},
        required_params=["severity"],
    ),
    QueryPattern(
        name="timeline",
        description="Linha do tempo consolidada de experimentos",
        patterns=[
            r"timeline",
            r"linha do tempo",
            r"resumo dos experimentos",
        ],
        sql="""
            SELECT *
            FROM experiment_timeline
            ORDER BY run_start DESC
            LIMIT :max_rows
        """,
    ),
    QueryPattern(
        name="timeline_health",
        description="Runs por status de saúde",
        patterns=[
            r"(HEALTHY|UNSTABLE|REGRESSION|FAILED)",
            r"status de sa[uú]de",
            r"runs (saud[áa]veis|inst[áa]veis|regress[ãa]o|falhas)",
        ],
        sql="""
            SELECT *
            FROM experiment_timeline
            WHERE health_status = :health_status
            ORDER BY run_start DESC
            LIMIT :max_rows
        """,
        param_schema={"health_status": "str"},
        required_params=["health_status"],
    ),
    QueryPattern(
        name="metrics_for_run",
        description="Métricas de uma run específica",
        patterns=[
            r"m[ée]tricas da run (\d+)",
            r"m[ée]tricas do run (\d+)",
            r"run (\d+) m[ée]tricas",
        ],
        sql="""
            SELECT m.namespace, m.name, m.value, m.class_name, m.step, m.recorded_at
            FROM metric m
            JOIN run r ON r.id = m.run_id
            WHERE m.run_id = :run_id
            ORDER BY m.namespace, m.name, m.step
            LIMIT :max_rows
        """,
        param_schema={"run_id": "int"},
        required_params=["run_id"],
    ),
]


def _audit_id() -> str:
    """Gera ID simples de auditoria baseado em timestamp."""
    return f"{time.time():.6f}"


def _log_audit(entry: Dict[str, Any], user_id: str | None = None) -> None:
    """Registra entrada de audit trail."""
    _AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if user_id is not None:
        entry["user_id"] = user_id
    with open(_AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def _get_readonly_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Retorna conexão SQLite em modo read-only (URI)."""
    path = (db_path or get_db_path()).resolve()
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def build_schema_context(engine: Engine | None = None) -> SchemaContext:
    """Extrai schema das tabelas permitidas via introspecção SQLAlchemy.

    Como as tabelas são gerenciadas pelo SQLAlchemy ORM, usamos PRAGMA do SQLite
    para obter informações portáteis sem depender de reflection pesado.
    """
    target = engine or create_engine(f"sqlite:///{get_db_path()}", future=True)
    tables: Dict[str, SchemaTable] = {}

    with target.connect() as conn:
        for table_name in _ALLOWED_TABLES:
            try:
                rows = (
                    conn.execute(
                        text("PRAGMA table_info(:table_name)"),
                        {"table_name": table_name},
                    )
                    .mappings()
                    .all()
                )
            except Exception:
                rows = []

            if not rows:
                # Fallback direto com interpolação controlada (PRAGMA não aceita bind no nome)
                rows = conn.execute(text(f"PRAGMA table_info({table_name})")).mappings().all()

            columns = [
                SchemaColumn(
                    name=str(row["name"]),
                    type=str(row["type"]),
                    nullable=bool(row["notnull"] == 0),
                    default=str(row["dflt_value"]) if row["dflt_value"] is not None else None,
                )
                for row in rows
            ]

            if columns:
                tables[table_name] = SchemaTable(
                    name=table_name,
                    description=_table_description(table_name),
                    columns=columns,
                    sample_sql=f"SELECT * FROM {table_name} LIMIT 3",  # nosec B608
                )

    return SchemaContext(
        tables=tables,
        relationships=[
            "metric.run_id -> run.id",
            "alert.run_id -> run.id",
            "alert.experiment_id -> experiment.id",
            "run.experiment_id -> experiment.id",
            "artifact.run_id -> run.id",
            "hardware_snapshot.run_id -> run.id",
        ],
        business_rules=[
            "run.status IN ('running', 'completed', 'failed')",
            "alert.severity IN ('info', 'warning', 'critical')",
            (
                "experiment.stage IN "
                "('pretrain', 'stage1', 'stage2', 'finetune', 'two_stage', 'inference')"
            ),
        ],
    )


def _table_description(table_name: str) -> str:
    """Retorna descrição legível da tabela."""
    descriptions = {
        "experiment": "Agrupa runs de um treinamento/avaliação",
        "run": "Instância de execução (train/val/test/inference)",
        "metric": "Métricas numéricas globais, por classe ou históricas",
        "alert": "Alertas de queda de performance, falha de QG ou anomalia",
        "artifact": "Artefatos versionados (modelos, scalers, relatórios)",
        "hardware_snapshot": "Snapshot de uso de CPU/RAM/GPU durante a run",
        "experiment_timeline": "View consolidada de experimentos, runs, métricas e alertas",
    }
    return descriptions.get(table_name, f"Tabela {table_name}")


def validate_sql(sql: str, allowed_tables: Optional[set[str]] = None) -> Tuple[bool, str]:
    """Valida SQL customizado contra regras de segurança.

    Aceita aliases de tabela (ex.: ``FROM run r``) desde que o nome da
    tabela subjacente esteja na allowlist.

    Returns
    -------
    tuple[bool, str]
        (ok, motivo_rejeicao)
    """
    allowed_tables = allowed_tables or _ALLOWED_TABLES
    allowed_lower = {t.lower() for t in allowed_tables}

    if not sql or not sql.strip():
        return False, "SQL vazio"

    parsed = sqlparse.parse(sql)
    if not parsed:
        return False, "Não foi possível fazer parse do SQL"

    for statement in parsed:
        tokens = [str(t) for t in statement.tokens if not t.is_whitespace]
        first_keyword = tokens[0].lower() if tokens else ""

        if first_keyword not in {"select", "with", "explain"}:
            return False, f"Comando não permitido: {tokens[0] if tokens else '<vazio>'}"

        flat_lower = " ".join(t.lower() for t in tokens)
        for forbidden in _FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{forbidden}\b", flat_lower):
                return False, f"Palavra-chave proibida detectada: {forbidden}"

        if re.search(r"\bunion\b", flat_lower):
            return False, "Palavra-chave proibida detectada: union"

    # Extrai referências de tabela de FROM/JOIN, ignorando aliases opcionais.
    for match in re.finditer(
        r"\b(from|join)\s+(\w+)(?:\s+(?:as\s+)?(\w+))?",
        flat_lower,
    ):
        table_name = match.group(2)
        if table_name.lower() not in allowed_lower:
            return False, f"Tabela não permitida: {table_name}"

    return True, ""


def _extract_param_from_question(question: str, param_name: str) -> Optional[str]:
    """Extrai parâmetro simples da pergunta por regex."""
    q_lower = question.lower()
    if param_name == "stage":
        m = re.search(r"est[áa]gio\s+(\w+)", q_lower) or re.search(r"stage\s+(\w+)", q_lower)
        return m.group(1) if m else None
    if param_name == "severity":
        for sev in ["critical", "crítico", "warning", "info"]:
            if sev in q_lower:
                return "critical" if sev in {"critical", "crítico"} else sev
    if param_name == "health_status":
        for hs in ["HEALTHY", "UNSTABLE", "REGRESSION", "FAILED"]:
            if re.search(rf"\b{hs}\b", q_lower, re.IGNORECASE):
                return hs
    if param_name == "run_id":
        m = re.search(r"run\s+(\d+)", question, re.IGNORECASE)
        return m.group(1) if m else None
    return None


def _match_catalog(question: str) -> Optional[Tuple[QueryPattern, Dict[str, Any]]]:
    """Tenta encontrar padrão no catálogo e extrair parâmetros."""
    q_lower = question.lower()

    for pattern in _QUERY_CATALOG:
        matched = False
        for regex in pattern.patterns:
            if re.search(regex, q_lower, re.IGNORECASE):
                matched = True
                break

        if not matched:
            continue

        params: Dict[str, Any] = {}
        for param_name in pattern.required_params:
            value = _extract_param_from_question(question, param_name)
            if value is None:
                matched = False
                break
            expected_type = pattern.param_schema.get(param_name, "str")
            if expected_type == "int":
                try:
                    params[param_name] = int(value)
                except ValueError:
                    matched = False
                    break
            else:
                params[param_name] = value

        if matched:
            return pattern, params

    return None


def _catalog_owner_alias(pattern_name: str) -> str | None:
    """Retorna o alias de tabela para aplicar RLS em queries do catálogo."""
    return {
        "last_failed_runs": "e",
        "runs_by_stage": "e",
        "alerts_by_severity": "e",
        "metrics_for_run": "r",
        "timeline": None,
        "timeline_health": None,
    }.get(pattern_name)


def _merge_rls_params(
    params: Dict[str, Any] | Tuple[Any, ...] | None,
    rls_params: Dict[str, Any] | Tuple[str, ...],
) -> Dict[str, Any] | Tuple[Any, ...]:
    """Mescla parâmetros originais com os gerados pelo RLS.

    Preserva o estilo de bind: quando o RLS gera binds posicionais (?),
    o resultado é uma tupla; quando gera binds nomeados, o resultado é dict.
    Para binds posicionais, os parâmetros RLS vêm primeiro porque o filtro
    owner_id é injetado no início da cláusula WHERE.
    """
    if params is None:
        return rls_params
    if isinstance(rls_params, tuple):
        if isinstance(params, tuple):
            return rls_params + params
        return rls_params + tuple(params.values())
    if isinstance(params, dict):
        return {**params, **rls_params}
    # rls_params é dict e params é tuple
    return {**rls_params, **{f"param_{i}": v for i, v in enumerate(params)}}


def _execute_sql(
    sql: str,
    params: Dict[str, Any] | Tuple[Any, ...],
    max_rows: int,
    db_path: Path | None = None,
) -> Tuple[List[str], List[Dict[str, Any]], int, float]:
    """Executa SQL validado em conexão read-only do SQLite."""
    conn = _get_readonly_connection(db_path)
    start = time.perf_counter()

    try:
        if isinstance(params, dict):
            merged: Dict[str, Any] | Tuple[Any, ...] = {**params, "max_rows": max_rows + 1}
            exec_sql = sql
        else:
            merged = tuple(params)
            if ":max_rows" in sql:
                exec_sql = sql.replace(":max_rows", "?")
            else:
                exec_sql = f"{sql.rstrip(';').strip()} LIMIT ?"
            merged = merged + (max_rows + 1,)
        cursor = conn.execute(exec_sql, merged)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        all_rows = [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

    elapsed = (time.perf_counter() - start) * 1000
    rows = all_rows[:max_rows]
    return columns, rows, len(all_rows), elapsed


def answer_question(
    req: NaturalQueryRequest,
    db_path: Path | None = None,
    user_id: str | None = None,
    roles: Sequence[str] = (),
) -> StructuredQueryResult:
    """Responde pergunta em linguagem natural via catálogo determinístico.

    Se ``user_id`` for informado e o usuário não for admin, aplica filtro
    RLS baseado no dono do experimento/run.

    Se nenhum padrão for encontrado, retorna ajuda do schema em vez de
    executar SQL não validado.
    """
    audit_id = _audit_id()
    matched = _match_catalog(req.question)

    if matched is None:
        context = build_schema_context(
            create_engine(f"sqlite:///{db_path or get_db_path()}", future=True)
        )
        _log_audit(
            {
                "audit_id": audit_id,
                "question": req.question,
                "sql": None,
                "params": req.params,
                "row_count": 0,
                "source": "schema_help",
                "allowed_custom_sql": req.allow_custom_sql,
            },
            user_id=user_id,
        )
        return StructuredQueryResult(
            question=req.question,
            sql="",
            params={},
            columns=["schema_context"],
            rows=[{"schema_context": context.model_dump()}],
            row_count=0,
            truncated=False,
            execution_time_ms=0.0,
            source="schema_help",
            audit_id=audit_id,
        )

    pattern, extracted_params = matched
    params: Dict[str, Any] = {**extracted_params, **(req.params or {}), "max_rows": req.max_rows}
    sql = pattern.sql

    ok, reason = validate_sql(sql)
    if not ok:
        raise ValueError(f"SQL do catálogo rejeitado pela validação: {reason}")

    if user_id is not None and "admin" not in roles:
        alias = _catalog_owner_alias(pattern.name)
        sql, rls_params = RowLevelSecurity().apply_filter(sql, user_id, roles, table_alias=alias)
        if rls_params:
            params = _merge_rls_params(params, rls_params)  # type: ignore[assignment]

    with LatencyTracker("sql", "answer_question"):
        columns, rows, total, elapsed = _execute_sql(sql, params, req.max_rows, db_path)

    _log_audit(
        {
            "audit_id": audit_id,
            "question": req.question,
            "sql": sql,
            "params": params,
            "row_count": len(rows),
            "source": "catalog",
            "pattern": pattern.name,
        },
        user_id=user_id,
    )

    return StructuredQueryResult(
        question=req.question,
        sql=sql,
        params=params,
        columns=columns,
        rows=rows,
        row_count=len(rows),
        truncated=total > req.max_rows,
        execution_time_ms=elapsed,
        source=f"catalog:{pattern.name}",
        audit_id=audit_id,
    )


def _extract_owner_alias(sql: str) -> str | None:
    """Extrai o alias de tabela usado em FROM run/experiment/experiment_timeline."""
    sql_lower = sql.lower()
    sql_keywords = {
        "where",
        "group",
        "order",
        "having",
        "limit",
        "join",
        "union",
        "select",
        "from",
        "on",
        "and",
        "or",
        "not",
        "in",
        "is",
        "null",
        "as",
        "by",
        "for",
        "with",
    }
    match = re.search(
        r"\bfrom\s+(run|experiment|experiment_timeline)\b(?:\s+as\b)?\s+(\w+)",
        sql_lower,
    )
    if match:
        alias = match.group(2)
        if alias not in sql_keywords:
            return alias
    for table_name in ("experiment_timeline", "experiment", "run"):
        if re.search(rf"\bfrom\s+{table_name}\b", sql_lower):
            return table_name
    return None


def execute_custom_sql(
    sql: str,
    params: Optional[Dict[str, Any]] = None,
    max_rows: int = _MAX_ROWS,
    db_path: Path | None = None,
    user_id: str | None = None,
    roles: Sequence[str] = (),
) -> StructuredQueryResult:
    """Executa SQL customizado após validação rigorosa.

    Deve ser usado apenas quando ``allow_custom_sql=True``; o MCP server
    expõe essa funcionalidade de forma controlada.
    """
    audit_id = _audit_id()
    ok, reason = validate_sql(sql)
    if not ok:
        _log_audit(
            {
                "audit_id": audit_id,
                "sql": sql,
                "params": params,
                "row_count": 0,
                "source": "custom_sql_rejected",
                "reason": reason,
            },
            user_id=user_id,
        )
        raise ValueError(f"SQL rejeitado: {reason}")

    user_params = params or {}
    is_positional = isinstance(user_params, (tuple, list))

    if is_positional:
        exec_params: Dict[str, Any] | Tuple[Any, ...] = tuple(user_params)
        result_params: Dict[str, Any] = {f"param_{i}": v for i, v in enumerate(user_params)}
    else:
        exec_params = dict(user_params)
        result_params = {**user_params, "max_rows": max_rows}

    if user_id is not None and "admin" not in roles:
        alias = _extract_owner_alias(sql)
        sql, rls_params = RowLevelSecurity().apply_filter(sql, user_id, roles, table_alias=alias)
        if rls_params:
            exec_params = _merge_rls_params(exec_params, rls_params)
            if isinstance(rls_params, dict):
                result_params["owner_id"] = rls_params["owner_id"]
            else:
                result_params["owner_id"] = rls_params[0]

    with LatencyTracker("sql", "execute_custom_sql"):
        columns, rows, total, elapsed = _execute_sql(sql, exec_params, max_rows, db_path)

    _log_audit(
        {
            "audit_id": audit_id,
            "sql": sql,
            "params": result_params,
            "row_count": len(rows),
            "source": "custom_sql",
        },
        user_id=user_id,
    )

    return StructuredQueryResult(
        question="",
        sql=sql,
        params=result_params,
        columns=columns,
        rows=rows,
        row_count=len(rows),
        truncated=total > max_rows,
        execution_time_ms=elapsed,
        source="custom_sql",
        audit_id=audit_id,
    )
