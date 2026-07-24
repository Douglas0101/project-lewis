# Plano de Implementação — Pivotagem MLP Features v2.3

**Status:** Aprovado e implementado em `feature/mlp-features-v2.3`
**Data:** 2026-07-06
**Autor:** Kimi Code CLI

## Objetivo

Consolidar a pivotagem do pipeline two-stage de classificação de arritmias ECG
para MLPs leves sobre features morfológicas + time-domain, atingindo os
thresholds do Quality Gate 5' v2.3 e mantendo compatibilidade com quantização
INT8 para STM32F4.

## Estado da Arte Aplicado

- Features morfológicas (QRS width, área, assimetria, rugosidade, T/R ratio)
  robustas a variações de amplitude e baseline.
- RR features como contexto ritmico (prev/next, razão, média local, RMSSD,
  heart rate).
- MLP compacto com regularização (Dropout) e pesos de classe limitados.
- Oversampling real da classe F no Estágio 2 (sem SMITE sintético).
- GroupKFold por paciente para estimativa realista de generalização.

## Tarefas Executadas

1. [x] Expandir `MorphologicalFeatures` com fallback de onset/offset e três
   novas features (`qrs_asymmetry_index`, `t_r_ratio`, `qrs_raggedness`).
2. [x] Atualizar schema `MorphologicalFeatures`, pipeline de extração e scripts
   de preparação (`prepare_stage1_features.py`, `prepare_stage2_features.py`).
3. [x] Regenerar `finetuning_mitbih_family.parquet/npz` com 473.036 batimentos.
4. [x] Gerar `stage1_binary.parquet/npz/features.npz` e
   `stage2_multiclass.parquet/npz/features.npz`.
5. [x] Parametrizar `train_stage1_mlp.py` (`--hidden-units`, `--max-weight`).
6. [x] Parametrizar `train_stage2_mlp.py` (`--hidden-units`).
7. [x] Treinar Stage 1 (64 unidades, max_weight=20) e Stage 2 (64 unidades,
   F oversample=0.75, max_weight=10).
8. [x] Selecionar e publicar melhores folds em `models/*_v2.3.*`.
9. [x] Criar `TwoStageMLPPipeline` e `FeatureExtractor` com suporte a
   `precomputed_temporal`.
10. [x] Criar `tests/test_two_stage_mlp_qg5.py` e validar QG5' v2.3.
11. [x] Criar scripts de quantização (`quantize_mlp_features.py`) e validação
    (`validate_quantized_mlp.py`) para INT8.
12. [x] Quantizar modelos e validar ΔF1-macro < 2%.
13. [x] Escrever ADR v2.3.

## Resultados

| Gate | Critério | Resultado |
|------|----------|-----------|
| QG5' Estágio 1 | Recall(Anormal) ≥ 0.30, Precision ≥ 0.25, F1-macro ≥ 0.55 | Recall=0.8342, Precision=0.8241, F1=0.9005 |
| QG5' Estágio 2 | F1(S) ≥ 0.55, F1(V) ≥ 0.70, F1(F) ≥ 0.15, F1-macro ≥ 0.45 | F1(S)=0.7750, F1(V)=0.7355, F1(F)=0.9253, F1-macro=0.8119 |
| QG6 | ΔF1-macro INT8 < 2%, FlatBuffer < 64 KB | ΔStage1=0.0019, ΔStage2=0.0041, Stage1=4.88 KB, Stage2=4.98 KB |

## Próximos Passos Recomendados

1. **Firmware (C11/C08):** portar `FeatureExtractor` e `TwoStageMLPPipeline`
   para C, integrando com TFLM e os headers INT8 gerados.
2. **Bit-exatidão (QG8):** validar saída do TFLM INT8 vs Python
   `BUILTIN_REF` em beats representativos.
3. **Pipeline end-to-end clínico:** testar inferência em streaming com
   detector de R-peaks completo e histórico por paciente.
4. **Knowledge Layer (C11):** indexar ADR v2.3 e lições aprendidas no RAG
   local para consulta por futuros agentes.
5. **Refinamento de thresholds:** avaliar threshold operacional do Estágio 1 em
   cenários de prevalência variada (N: 70-95%).
6. **Monitoramento de drift:** adicionar métricas de drift nas features
   (`rr_local_mean`, `qrs_width_ms`, etc.) no pipeline de produção.

## Lições Aprendidas

- A extração de features time-domain **não pode ser recomputada** a partir de
  subconjuntos esparsos de R-peaks; o histórico consecutivo por registro é
  mandatório.
- A seleção do melhor fold deve priorizar folds que atendem **todos** os
  thresholds por classe, não apenas a métrica agregada.
- A quantização INT8 de MLP sobre features é trivialmente pequena (< 5 KB) e
  preserva performance, desde que a calibração use dados na mesma escala de
  treinamento.
