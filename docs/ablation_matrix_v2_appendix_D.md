# Adendo D — Trilha Teacher/Destilação (T10.2.1)

**Versão:** v1.0.0 · **Data:** 2026-08-01 · **Branch:** `develop` @ `70153cf`
**Task:** T10.2.1 (adendo à matriz v2, SDD-LEWIS ML Protocol v2)
**Status:** normativo para T10.3 (pilotos) e T11 (pré-treinos oficiais)
**Decisão de origem:** Caminho A (teacher + destilação) escolhido pelo owner
**Referências:** `docs/ablation_matrix_v2.md` · `docs/algorithm_engineering_audit_v1.md` ·
`reports/ml_protocol_v2/pretrain_reconciliation.md`

> Nenhum treino foi executado na produção deste documento.
> O teacher não é modelo de produção — é ferramenta experimental de treino.
> O student permanece dentro das restrições TinyML (QG6/QG9 inalterados).
> Custos são ESTIMATIVAS ancoradas na base medida local (32k params/30 épocas ≈ 33–40 min CPU).

---

## 1. Hipótese central — H8: o gargalo CD/HYP é limitação de capacidade

**Prioridade:** P0 (ao lado de H7, H1, H2 da matriz v2)

**Evidência:**

- CD PR-AUC 0,5561 [IC95 0,5450; 0,5670] · HYP 0,5078 [0,4957; 0,5176] — gargalo sistemático
  (T9.3, `evaluation_v2/` + Tabela 2 do relatório de reconciliação).
- RF do backbone A1 = 188 ms < QT (~400 ms) — derivado na auditoria §5.1.
- −7 p.p. AUC vs Strodthoff 2021 (0,5–8M params, 12 leads) — `docs/pretrain_benchmark_comparison.md`
  (com ressalvas de comparabilidade já registradas).
- A variável **capacidade nunca foi isolada**: todos os backbones treinados têm ≤ 32k params
  (auditoria §2, inventário).

**Critério de refutação:** se o teacher de 1M+ params (D2) não melhorar CD PR-AUC em ≥ 5 p.p.
sobre o A2-full (≥ 0,606 vs referência 0,5561), a hipótese de capacidade é refutada e o gargalo
passa a ser tratado como dados/formulação (redirecionar para trilhas S/P da matriz v2).

**Critério de validação parcial:** se o teacher melhorar CD/HYP mas o student destilado não
reter ≥ 50% do ganho do teacher, o gargalo é de **destilação** (não de capacidade pura) —
escalar para D5 (student 64k) antes de concluir.

---

## 2. Células experimentais

| Célula | Teacher | Params (alvo) | Leads | Student | Objetivo |
|---|---|---:|---:|---|---|
| D0 | A2-full (self) | 32k | 1 | A2-full 32k | controle de destilação (self-distillation) |
| D1 | ResNet1D-leve | ~500k | 1 | A2-full 32k | capacidade moderada, mesmo input |
| D2 | ResNet1D-padrão | ~1M | 1 | A2-full 32k | capacidade alta, mesmo input — **teste central de H8** |
| D3 | ResNet1D-padrão | ~1M | 12 | A2-full 32k (1 lead) | informação de 12 leads é destilável? |
| D4 | Inception1D | ~5M | 1 | A2-full 32k | teto de capacidade, mesmo input |
| D5 | ResNet1D-padrão (reusa D2) | ~1M | 1 | **A1 expandido 64k** | student maior retém mais? |

Regras:

- **D0 é obrigatório**: sem o controle de self-distillation, um ganho em D1–D5 pode ser efeito
  do protocolo KD (regularização por soft targets), não do teacher maior.
- D1–D2 isolam capacidade com input idêntico (1 lead, 500 amostras).
- D3 isola informação de leads (12 → 1 via destilação) — tem custo de pipeline (seção 9).
- D4 é teto superior: se D4 não melhora CD, capacidade não é o gargalo.
- **Ordem condicional anti-desperdício:** D0 → D1 → D2 → {D3, D5} → D4 **somente se** D2 ≥ +3
  p.p. em CD PR-AUC.

---

## 3. Arquiteturas do teacher (especificação mínima)

```yaml
teacher_archetypes:
  resnet1d_leve:
    blocks: [16, 32, 64, 128]
    kernel_sizes: [7, 5, 5, 3]
    params_target: ~500k
    params_derived_estimate: "~125k (ver nota abaixo)"
    residual: true
    normalization: GroupNorm   # BatchNorm problemático com batches pequenos; TFLM não se aplica
    activation: relu
    dropout: 0.2
    head: linear 5 (sigmoid)

  resnet1d_padrao:
    blocks: [32, 64, 128, 256]
    kernel_sizes: [15, 7, 5, 3]   # RF maior: coerente com H3 (cobrir QRS largo → QT)
    params_target: ~1M
    params_derived_estimate: "~460k (ver nota abaixo)"
    residual: true
    normalization: GroupNorm
    activation: relu
    dropout: 0.2
    head: linear 5 (sigmoid)

  inception1d:
    branches: [[32, k3], [32, k7], [32, k15], [16, k31]]
    blocks: 4
    params_target: ~5M
    params_derived_estimate: "~1–5M conforme crescimento de canais (pinado em T9.4)"
    residual: true
    normalization: GroupNorm
    activation: relu
    dropout: 0.3
    head: linear 5 (sigmoid)
```

**Nota de honestidade paramétrica (derivada nesta task):** os blocos do prompt original rendem
menos params que os alvos declarados (ex.: [16,32,64,128] com 2 convs/bloco + projeções ≈ 125k,
não 500k; [32,64,128,256] ≈ 460k, não 1M). Para atingir os alvos, as larguras devem subir um
nível (ex.: [32,64,128,256] → ~500k; [64,128,256,512] → ~1,8M). **O que é normativo é o ALVO
de capacidade** (500k/1M/5M); a configuração exata (larguras/blocos) será pinada em T9.4 com
teste de orçamento de params no padrão de `tests/test_backbone_budget.py`.

Regra: o teacher **não** tem restrições de FlatBuffer/SRAM/latência. É treinado e avaliado
offline em Python. Nunca vai para o firmware, `models/` ou headers C.

---

## 4. Protocolo de destilação (KD)

```yaml
distillation_protocol:
  loss:
    type: KD + task
    formula_prompt_original: "L = α · KL_div(softmax(z_t/τ), softmax(z_s/τ)) + (1−α) · task_loss"
    formula_normativa_corrigida: >
      L = α · KD + (1−α) · focal_BCE(y, z_s),  com
      KD = mean_c BCE_with_logits(z_s,c / τ, sigmoid(z_t,c / τ)) · τ²
    correcao_justificativa: >
      A tarefa é multi-label sigmoid (SCP-ECG, 1,70 rótulos/registro, co-ocorrência
      NORM 54–59% — auditoria §3.1 e T9.3 Tabela 4). Softmax-KD impõe exclusão mútua
      entre classes e contradiria a formulação; a forma correta é KD binário por
      rótulo sobre logits com temperatura (sigmoid), com fator τ² de escala do
      gradiente (Hinton 2015). A fórmula do prompt fica registrada acima e a
      correção é a forma normativa.
    alpha: 0.7        # varrer {0.5, 0.7, 0.9} na célula de tuning
    tau: 4.0          # varrer {2.0, 4.0, 6.0}
    task_loss: focal γ=2.0 (mesma do student original; ou γ vencedor da trilha F, se já corrida)

  training:
    teacher_frozen: true
    student_init: "pesos do A2-full (backbone_pretrained.keras) via
      load_backbone_weights_from_pretrained (src/models/backbone_1d.py:377) — nunca from-scratch"
    optimizer: adamw (weight_decay 1e-4)
    lr: 1e-3
    schedule: cosine
    warmup_epochs: 2
    max_epochs: 50
    early_stopping:
      metric: val_macro_pr_auc    # protocolo v2 — nunca val_loss cru
      patience: 10

  evaluation:
    evaluator: canonical v2 (src/evaluation/canonical_evaluator.py)
    protocol_status: PROSPECTIVE obrigatório para claims (fit de T/thresholds só em calibration)
    metrics: [macro_pr_auc, macro_auroc, per_class_pr_auc (IC95), ece_post_calibration,
              ece_norm0, brier_post_calibration, bce_post_temperature]
    calibration:
      method: temperature_scaling
      split: calibration (separado de validation e test — split pareado v2, matriz v2 §9)
```

## 5. Métricas de sucesso por célula (pré-registradas)

| Célula | Métrica primária | Threshold de sucesso | Critério de refutação |
|---|---|---|---|
| D0 | macro PR-AUC | ≥ 0,7008 (A2-full) | self-distillation piora → KD mal configurado (corrigir antes de D1+) |
| D1 | CD PR-AUC | ≥ 0,58 (+2 p.p.) | < 0,56 → capacidade moderada insuficiente |
| D2 | CD PR-AUC | ≥ 0,60 (+5 p.p.) | < 0,58 → **H8 refutada** (capacidade não é o gargalo) |
| D3 | CD PR-AUC | ≥ 0,60 (+5 p.p.) | < 0,58 → informação de 12 leads não destilável |
| D4 | CD PR-AUC | ≥ 0,62 (+7 p.p.) | < 0,60 → teto de capacidade confirma refutação de H8 |
| D5 | CD PR-AUC | ≥ 0,60 (+5 p.p.) | < 0,58 → student 64k não retém mais que 32k |

Métricas transversais (todas as células): macro PR-AUC ≥ 0,70 · ECE pós-calibração < 0,025 ·
ECE NORM=0 < 0,10 · Δ INT8 macro PR-AUC < 0,01 (após PTQ do student) · FlatBuffer < 64 KB ·
latência Renode < 200 ms.

## 6. Critérios de promoção student → candidato (T11)

Um student destilado só vira candidato se satisfazer **todos**:

1. macro PR-AUC ≥ 0,70;
2. CD PR-AUC ≥ 0,60 (≥ +5 p.p. sobre A2-full);
3. HYP PR-AUC ≥ 0,55 (≥ +4 p.p.);
4. ECE pós-calibração < 0,025;
5. ECE NORM=0 < 0,10;
6. Δ INT8 macro PR-AUC < 0,01;
7. FlatBuffer < 64 KB;
8. latência Renode < 200 ms;
9. SRAM total < 128 KB;
10. bit-exatidão atol ≤ 1 LSB (QG8);
11. reconciliação com A2-full registrada em `evaluation_v2/`;
12. revisão humana aprovada + freeze de hash (ML Protocol v2 §11).

## 7. Orçamento experimental (ESTIMATIVA — não executar)

Âncora medida neste host (CPU-only): 32k params / 30 épocas ≈ 33–40 min (runs A0/A2 existentes).
Escala aproximada ~ linear em MACs (teacher 1M ≈ 15–30× MACs do A2-full por configuração).

| Célula | Teacher train | Student distill | Total ~ (CPU) | Total ~ (GPU, se disponível) |
|---|---|---|---|---|
| D0 | — | ~40 min | 0,7 h | ~0,2 h |
| D1 | ~1 h | ~40 min | 1,7 h | ~0,5 h |
| D2 | ~2 h | ~40 min | 2,7 h | ~0,8 h |
| D3 | ~2,5 h (12 leads + pipeline novo) | ~40 min | 3,2 h + pipeline C02 | ~1 h |
| D4 | ~8 h | ~40 min | 8,7 h | ~2,5 h |
| D5 | (reusa D2) | ~50 min | 0,9 h | ~0,3 h |
| **Total** | | | **≈ 18–20 h CPU** | **≈ 5–6 h GPU** |

Regra anti-desperdício: D4 (5M) só executa se D2 ≥ +3 p.p. em CD; D3 só agenda pipeline 12-lead
após aprovação (custo de C02 novo + revisão de QG1/QG2 equivalentes para o novo processamento).

## 8. Impacto nos QGs e no firmware

Nenhum QG é alterado por esta trilha:

- **QG6** (FlatBuffer < 64 KB): incide sobre o **student**, nunca sobre o teacher.
- **QG9** (latência < 200 ms, SRAM, arena): idem — student apenas.
- **QG4** (AUC/BCE): incide sobre o student após destilação; o braço BCE segue FAIL até a RFC
  T9.5 — a trilha D não o afrouxa nem o contorna.

O teacher é avaliado offline com o avaliador canônico v2, mas **não está sujeito a QGs de
firmware** e não gera artefatos de produção. Nenhum header C, nenhuma entrada em `models/`,
nenhuma alteração em `firmware/src/ml/inference.cpp`.

## 9. Dependências

| Dependência | Bloqueia | Estado |
|---|---|---|
| T9.4 — configs v2 (incluir `teacher_resnet1d.yaml`, `teacher_inception1d.yaml`, `distillation_kd.yaml` + pino de params com teste de orçamento) | todas as células D | pendente — próxima task |
| Split pareado v2 (matriz v2 §9 — geração + freeze) | todas as células D | pendente |
| Avaliador: extensão `ece_norm0` + IC por classe em artefato (G4) | claims de ECE NORM=0 | desejável antes de T10.3 |
| G6 — hotfix `reconcile_with_legacy` | T9.5 (não D diretamente) | aberto |
| **Pipeline C02 para 12 leads** (raw `g1–g12` existe; processado atual é 1 lead `*_II.npy`) | D3 | pendente — custo próprio + revisão |
| Aprovação humana (compute alto) | D4 | bloqueado por governança |
| Student A1 expandido 64k (design + teste de orçamento) | D5 | pendente em T9.4 |

## 10. Declarações finais

> Nenhum treino foi executado na produção deste documento.
> Nenhum Quality Gate foi alterado.
> Nenhum artefato em `models/` foi modificado.
> O teacher não é modelo de produção e nunca vai para o firmware.
> A promoção de qualquer student a candidato exige revisão humana (T11).
> O QG4-BCE permanece FAIL; a revisão é escopo da RFC T9.5.
