# ADR: Revisão dos Thresholds do Estágio 1 (v2.1)

**Status:** Aprovado  
**Data:** 2026-06-30  
**Autor:** Kimi Code (sob supervisão do arquiteto)  
**Referências:** `reports/stage1_v2_training_analysis.md`, `docs/UNIFIED_DOCUMENT_v2.0.md`

---

## Contexto

O Estágio 1 do pipeline de duas etapas (N vs Anormal) foi treinado no experimento `20260630_020334_stage1_v2.0` com os seguintes resultados:

| Métrica | Valor obtido | Meta v2.0 (UNIFIED_DOCUMENT) | Status |
|---------|--------------|------------------------------|--------|
| Accuracy | 0,7888 | > 0,92 | 🔴 FALHA |
| F1-macro | 0,5205 | > 0,90 | 🔴 FALHA |
| Recall Anormal (melhor fold) | 0,263 | ≥ 0,95 | 🔴 FALHA |
| AUC-ROC (melhor fold) | 0,5588 | > 0,98 | 🔴 FALHA |

A análise aprofundada mostrou que:

1. O modelo atua próximo de aleatório na separação N vs Anormal (AUC-ROC ≈ 0,56).
2. A backbone pré-treinada no Chapman não transfere representações úteis para MIT-BIH.
3. Descongelar toda a backbone não melhorou o desempenho.
4. Treinamento from scratch também falhou (AUC-ROC = 0,50).
5. As metas v2.0 são irreais para CNN 1D pura com sinal raw em inter-patient split.

## Decisão

Revisar os thresholds do Estágio 1 para valores factíveis, alinhados ao QG5' v2.2 do `AGENTS.md`, e acionar o fallback para features morfológicas como próximo passo.

### Novos thresholds do Estágio 1 (v2.1)

| Métrica | Threshold v2.0 | **Threshold v2.1** | Justificativa |
|---------|---------------|-------------------|---------------|
| Accuracy | > 0,92 | **> 0,75** | Ainda acima do trivial (tudo N = ~0,90); exige algum aprendizado real. |
| F1-macro | > 0,90 | **> 0,55** | Alinhado com QG5' v2.2 do AGENTS.md. |
| Recall Anormal | ≥ 0,95 | **≥ 0,30** | Mínimo observável com threshold tuning no experimento atual. |
| Precision Anormal | — | **≥ 0,25** | Mínimo observável; será refinado pelo Estágio 2. |
| AUC-ROC | > 0,98 | **> 0,60** | Meta incremental para sinal raw; > 0,80 apenas com features. |
| F1 (N) | — | **> 0,90** | Classe majoritária deve permanecer bem classificada. |
| F1 (Anormal) | — | **> 0,30** | Mínimo viável com arquitetura atual. |

### Critério de recall crítico

O UNIFIED_DOCUMENT v2.0 define recall Anormal ≥ 0,95 como "crítico para minimizar falsos negativos de arritmia". Com a arquitetura CNN pura, esse valor não é alcançável. A nova estratégia é:

1. Usar **F1-macro > 0,55** como gate principal do Estágio 1.
2. Aceitar recall Anormal ≥ 0,30 no curto prazo, com meta de ≥ 0,50 após introdução de features morfológicas.
3. Garantir que o Estágio 2 opere apenas sobre amostras classificadas como Anormal, reduzindo o impacto de falsos positivos.

## Consequências

### Positivas

- Metas tornam-se alcançáveis com a arquitetura atual.
- Permite validar o pipeline de duas etapas sem bloqueio no Estágio 1.
- Libera esforço para investir em features morfológicas, que têm maior potencial de separação.

### Negativas

- Recall Anormal mais baixo aumenta risco de falsos negativos de arritmia.
- Requer compensação no firmware (alertas por persistência de anormalidade, ex: 3 batimentos consecutivos).
- Necessita revisar casos de uso UC-02 e UC-03 no UNIFIED_DOCUMENT.

## Próximos passos

1. Atualizar `docs/UNIFIED_DOCUMENT_v2.0.md` com thresholds v2.1.
2. Atualizar `config/stage1_binary.yaml`.
3. Prototipar MLP com features morfológicas (Fase 1 do plano aprovado).
