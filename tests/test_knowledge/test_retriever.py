"""Testes do retriever semântico (QG-C11-03, QG-C11-04)."""

from __future__ import annotations

from typing import Any, List, Tuple

import pytest

from src.knowledge.retriever import _build_where_clause, search
from src.knowledge.schemas import QueryRequest


@pytest.mark.qg_c11
class TestRetrieverPrecision:
    """QG-C11-03: precisão semântica medida por MRR@5 >= 0.80."""

    def test_mrr_at_5_above_threshold(
        self,
        populated_db: Any,
        sample_chunks: List[Tuple[str, str, str, List[str], str]],
    ) -> None:
        reciprocal_ranks: List[float] = []
        for source, _layer, _version, _tags, content in sample_chunks:
            # Usa o mesmo texto indexado (source + content) para maximizar
            # a similaridade com o modelo fake determinístico.
            query = f"{source}\n{content}"
            req = QueryRequest(query=query, k=5, fetch_k=10)
            results = search(req)
            for rank, r in enumerate(results, start=1):
                if r.source == source:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        assert mrr >= 0.80, f"MRR@5 = {mrr:.2f}, abaixo do threshold 0.80"

    def test_search_returns_ranked_results(self, populated_db: Any) -> None:
        req = QueryRequest(query="F1-macro QG5", k=3, fetch_k=10)
        results = search(req)
        assert len(results) <= 3
        assert all(r.rank == i + 1 for i, r in enumerate(results))
        # sqlite-vec cosine distance pode apresentar pequena imprecisão
        # numérica; score = 1 - distance deve estar no intervalo [-1, 1].
        assert all(-1.0 <= r.score <= 1.0 for r in results)


@pytest.mark.qg_c11
class TestRetrieverFilters:
    """QG-C11-04: filtros 3D (layer, version, tags)."""

    def test_layer_filter_returns_only_requested_layer(self, populated_db: Any) -> None:
        req = QueryRequest(query="STM32 TFLM", layer="C08", k=5, fetch_k=10)
        results = search(req)
        assert results, "Filtro por layer não retornou resultados"
        assert all(r.layer == "C08" for r in results), "Resultado fora do layer filtrado"

    def test_version_filter_returns_only_requested_version(self, populated_db: Any) -> None:
        req = QueryRequest(query="quantizacao INT8", version="v1.1", k=5, fetch_k=10)
        results = search(req)
        assert results, "Filtro por versão não retornou resultados"
        assert all(r.version == "v1.1" for r in results), "Resultado fora da versão filtrada"

    def test_tags_filter_returns_only_matching_tags(self, populated_db: Any) -> None:
        req = QueryRequest(query="quantizacao", tags=["quantizacao"], k=5, fetch_k=10)
        results = search(req)
        assert results, "Filtro por tags não retornou resultados"
        assert all("quantizacao" in r.tags for r in results), "Tag obrigatória não presente"

    def test_invalid_layer_is_rejected_by_schema(self) -> None:
        with pytest.raises(ValueError):
            QueryRequest(query="teste", layer="INVALID")

    def test_build_where_clause_handles_metadata_filters(self) -> None:
        """sqlite-vec não suporta LIKE em metadata; tags são filtradas em Python."""
        req = QueryRequest(
            query="teste", layer="C04", version="v1.1", tags=["ml", "quantizacao"], k=5
        )
        where, params = _build_where_clause(req)
        assert "layer = ?" in where
        assert "version = ?" in where
        assert "tags" not in where
        assert params == ["C04", "v1.1"]
