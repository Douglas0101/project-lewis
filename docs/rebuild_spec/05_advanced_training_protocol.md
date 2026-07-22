# 05 — Protocolo Avançado de Treinamento

**Status:** PROPOSTO — aguardando ratificação humana
**Data:** 2026-07-18
**Entregável:** `advanced_training_protocol` (6)
**Corrige:** DQ-06 (parcial), DQ-11, DQ-14 (via protocolo), modelo `passes_qg5=false` promovido

---

## 1. Pré-condições bloqueantes (nenhuma célula roda sem todas)

1. Ontologia ratificada (01) e regeneração v3 concluída com `TEMPORAL_ALIGNMENT_FAILURE`
  ausente (02).
2. Manifests de split congelados (paciente–fold–seed) com hash registrado — encerra DQ-11.
3. Contrato de features v3.0.0 com relatório R-F1 publicado (03).
4. Nenhum artefato legado em qualquer caminho de treino/avaliação.

## 2. Matriz experimental

```text
4 famílias (04 §4: a, b, c, d)
× 5 outer folds (GroupKFold por paciente, congelados)
× 5 seeds {17, 29, 43, 71, 101}
= 100 células
```

Dentro de **cada** outer fold:

- inner GroupKFold por paciente sobre o treino externo;
- feature selection, scaler, imputer, sampler, templates — **fit somente no inner-train**;
- busca de hiperparâmetros (incl. λ's, β, γ, estratégia de balanceamento) **somente no inner loop**;
- early stopping **somente** na validação interna;
- threshold candidato escolhido **somente** na validação interna e congelado;
- calibrador ajustado em partição independente do treino do modelo-base (ver 07 §2);
- **outer test avaliado uma única vez**, depois de tudo congelado.

Todos os candidatos recebem **exatamente** os mesmos folds e seeds (mesmos manifests).
Nenhuma intervenção de balanceamento antes do split.

## 3. Registro obrigatório por célula

```json
{
  "patient_ids": ["..."], "record_ids": ["..."], "dataset_ids": ["..."],
  "fold": 0, "seed": 17, "model_family": "a|b|c|d",
  "config_hash": "sha256…", "data_hash": "sha256…",
  "ontology_hash": "sha256…", "preprocessing_hash": "sha256…"
}
```

## 4. Desbalanceamento (somente inner loop; decisão por pacientes-por-classe)

| # | Estratégia | Notas |
|---|---|---|
| 1 | baseline sem intervenção | referência |
| 2 | pesos inversos | w_c = N/(K·n_c) |
| 3 | pesos por número efetivo | w_c = (1−β)/(1−β^{n_c}), β ∈ {0,99; 0,999; 0,9999} |
| 4 | batches balanceados | amostrador por época |
| 5 | focal loss | −α_c(1−p_t)^γ log p_t, γ ∈ {1, 2} |
| 6 | oversampling | somente treino do fold |
| 7 | undersampling | somente treino do fold |
| 8 | combinação hierárquica | por nível da ontologia |

Suporte de referência (auditoria): FUSION 1.044 beats/45 pacientes (top-5 = 82%); AFIB depende de
D3; S 141 pacientes; V 174; Q_OR_UNKNOWN 83. A escolha registra efeito sobre: recall, precision,
especificidade, PR-AUC, ROC-AUC, MCC, F1 por classe, calibração, variabilidade entre seeds,
desempenho por paciente e por dataset.

| Parâmetro | Fórmula | Unidade | Intervalo candidato | Método de seleção | Dados permitidos | Risco | Status |
|---|---|---|---|---|---|---|---|
| β (nº efetivo) | w_c=(1−β)/(1−β^{n_c}) | — | {0,99; 0,999; 0,9999} | inner loop | inner-train | baixo | PROPOSED_REQUIRES_RATIFICATION |
| γ (focal) | (1−p_t)^γ | — | {1, 2} | inner loop | inner-train | baixo | PROPOSED_REQUIRES_RATIFICATION |
| outer folds | GroupKFold | — | {5} | protocolo | — | baixo | PROJECT_EXISTING |
| seeds | — | — | {17,29,43,71,101} | protocolo E06.5 existente | — | baixo | PROJECT_EXISTING |

## 5. Métricas (por tarefa, classe, dataset, paciente e fold)

Matriz de confusão; sensibilidade/recall; especificidade; precision; F1; F1-macro; MCC;
balanced accuracy; ROC-AUC; PR-AUC; average precision; log-loss; Brier; calibration intercept;
calibration slope; ECE global e por classe; **pior fold**; **percentil 10 fold–seed**; desvio
entre seeds. Denominadores sempre declarados: pacientes / registros / batimentos. Proibido
reportar apenas média global.

### 5.1 Incerteza e comparação

```text
bootstrap agrupado por paciente: 10.000 repetições, IC 95%, cluster = patient_id
ΔM = M_A − M_B (pareado por fold–seed)
superioridade:      LCB_95%(ΔM) > 0
não-inferioridade:  LCB_95%(ΔF1_macro) > −0,02
```

| Parâmetro | Fórmula | Unidade | Intervalo candidato | Método de seleção | Dados permitidos | Risco | Status |
|---|---|---|---|---|---|---|---|
| repetições bootstrap | 10.000 | — | {10.000} | convenção bioestatística | outer (somente leitura) | baixo | STANDARD_DERIVED |
| margem superioridade | LCB95(ΔM)>0 | — | {0} | ratificação | outer | médio | PROPOSED_REQUIRES_RATIFICATION |
| margem não-inferioridade | LCB95(ΔF1)>−0,02 | F1 | {−0,01; −0,02; −0,05} | ratificação clínica | outer | alto | PROPOSED_REQUIRES_RATIFICATION |

## 6. Gates de falsificação (sanidade, não tuning)

- **G-F1:** primeira célula (família a, fold 1, seed 17) com ROC-AUC < 0,6 na tarefa de
  batimento → interromper a matriz inteira: pipeline ainda quebrado; retornar
  `INSUFFICIENT_EVIDENCE` com relatório.
- **G-F2:** probe de dataset (06 §2) acima do limiar → `DATASET_SHORTCUT_LEARNING`, matriz
  interrompida.
- **G-F3:** variância anômala entre seeds (P10 fold–seed abaixo do piso ratificado) →
  `REVIEW_REQUIRED`.
- **G-F4:** qualquer violação de fit-fora-do-treino detectada em auditoria → célula invalidada e
  investigação.

## 7. Rastreabilidade

Cada célula grava: métricas completas, hashes de config/dados/ontologia/preprocessing,
manifest de split, orçamento consumido. Orçamento total: ~100 células CPU; critérios de
interrupção: G-F1…G-F4 ou orçamento excedido sem decisão. **Nenhuma célula autoriza promoção** —
promoção exige bundle + attestation + gates (08/09/11) e quorum humano.
