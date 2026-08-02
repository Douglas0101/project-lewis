# Matriz de Hipóteses e Ablações v2 — Gate experimental de pré-treino (T10.2)

**Versão:** v1.0.0 · **Data:** 2026-08-01 · **Branch:** `develop` @ `9fa668b`
**Task:** T10.2 (SDD-LEWIS ML Protocol v2, prompt SDD-LEWIS-CLI-T10.2-001)
**Status:** normativo para T10.3 (pilotos) e T11 (pré-treinos oficiais)
**Referências:** `docs/algorithm_engineering_audit_v1.md` · `docs/ml_protocol_v2.md` ·
`reports/ml_protocol_v2/pretrain_reconciliation.md` · `experiments/*/evaluation_v2/`

> Nenhum treino foi executado na produção deste documento. Todas as células aqui definidas são
> experimentos planejados, não resultados. Todos os números citados vêm dos artefatos
> `evaluation_v2/` (T9.3) ou da auditoria T10.1 — nada é estimativa não marcada; custos da
> seção 12 são ESTIMATIVAS sinalizadas.

---

## 0. Escopo

Este documento transforma os achados de T10.1 (auditoria algorítmica) e T9.3 (reavaliação
canônica) em um plano experimental auditável. **Nenhuma célula experimental pode ser executada
sem estar registrada aqui** (regra de gate para T10.3 e T11). Fora de escopo: treinos, pilotos,
inferências, alteração de Quality Gates, geração de splits.

### 0.1 Fatos de T9.3 que informam esta matriz (exatos, não aproximados)

| Run | AUC v2 | BCE cru | BCE pós-T | ECE pré | ECE pós-T | T | Fonte |
|---|---:|---:|---:|---:|---:|---:|---|
| A0 histórico `20260728_033533` | 0,8334 | 0,3907 | 0,3905 | 0,0191 | 0,0185 | 0,9722 (fit, RETRO) | evaluation_v2/metrics.json |
| A0 novo `20260729_042301` | 0,8365 | 0,3880 | 0,3869 | 0,0251 | 0,0206 | 0,9130 (congelada) | evaluation_v2/metrics.json |
| A2-full `20260728_053011` | 0,8639394700 | 0,4317 | **0,3417** | 0,1508 | **0,0152** | **0,3741** (congelada) | evaluation_v2/metrics.json |

Achados estratificados (relatório T9.3, Tabelas 2–4):

- ECE pós-T no **estrato NORM=0**: NORM **0,2167** · STTC **0,1685** (marginal 0,0152 esconde o
  estrato patológico).
- F1@0.5 → F1@tuned: **+12 a +14 p.p.** em CD/HYP (RETROSPECTIVE — teto otimista a revalidar
  prospectivamente).
- CD PR-AUC 0,5561 [IC95 0,5450; 0,5670] · HYP 0,5078 [0,4957; 0,5176] — gargalo **sistemático**.
- P(NORM=1 | classe) = 54–59% em CD/MI/HYP — NORM não é "ausência de doença".
- A0 × A2-full: **NON_COMPARABLE** por `split_id` (seed 42 ≠ 13); A0 hist × A0 novo COMPARABLE
  (ΔAUC 0,0031, ICs sobrepostos — strict = ruído).
- RF do backbone A1: **188 ms** < QT (~400 ms) — derivado na auditoria §5.1.

---

## 1. Hipóteses priorizadas (H1–H7)

| ID | Hipótese | Prioridade | Evidência (fonte) | Critério de refutação |
|---|---|---|---|---|
| **H7** | O ganho do A2-full não é atribuível: `A1+BCE` nunca existiu e os splits não são pareados | **P0** | NON_COMPARABLE por `split_id` (T9.3 Tabela 3); auditoria §11.1 | Célula C1 (A1+BCE, split pareado) com IC95 de macro PR-AUC **sobreposto** ao do A2-full |
| **H1** | Focal γ=2,0 está alto demais: comprime probabilidades globalmente | **P0** | ECE pré 0,13–0,17 uniforme em todas as classes; T=0,3741 ≪ 1 (T9.3 Tabela 2) | F2 (γ=1,0) com ECE pós-T **≥** ECE pós-T de F3 (γ=2,0) e macro PR-AUC não menor |
| **H2** | Calibração global (T única) é insuficiente para o estrato patológico | **P0** | ECE NORM=0 = 0,2167 (NORM) / 0,1685 (STTC) pós-T (T9.3 Tabela 4) | K1/K2 com ECE NORM=0 **≥ 0,15** (melhora < 30% vs T global) |
| **H3** | CD é gargalo de receptive field (188 ms < QRS largo/QT) | **P1** | CD PR-AUC 0,5561 IC estreito; RF derivado (auditoria §5.1) | R1–R4 (RF ≥ 300 ms) sem ganho ≥ 3 p.p. de PR-AUC em CD |
| **H4** | Threshold tuning recupera desempenho de decisão real | **P1** | F1@tuned +12–14 p.p. em CD/HYP (RETROSPECTIVE, T9.3 Tabela 2) | T1 com fit em calibration separado rendendo < 4 p.p. sobre T0 |
| **H5** | O A0 estava sub-treinado (30 épocas não bastaram para BCE) | **P1** | val_loss do A0 novo estritamente decrescente até a época 29; ES nunca disparou (auditoria §7) | B1 (100 épocas, ES por `val_macro_pr_auc`) sem ganho ≥ 1 p.p. de macro PR-AUC no A0 |
| **H6** | NORM não é "ausência de doença" — formulação distorce métricas | **P2** | P(NORM\|classe) = 54–59% (T9.3 Tabela 4); auditoria §3.2 | P1 (4 classes sem NORM) alterando macro PR-AUC das 4 classes < 1 p.p. |
| **H8** | O gargalo CD/HYP é limitação de **capacidade** (≤ 32k params em todos os backbones) | **P0** | CD/HYP PR-AUC com IC estreito (T9.3); −7 p.p. AUC vs Strodthoff (0,5–8M params); capacidade nunca isolada (auditoria §2) | D2/D4 (teacher 1M+/5M) sem ganho ≥ 5 p.p. de PR-AUC em CD — ver adendo `docs/ablation_matrix_v2_appendix_D.md` |

### 1.1 Mapa de rastreabilidade de IDs (esta matriz ↔ auditoria T10.1)

| Matriz v2 | Auditoria T10.1 | Síntese |
|---|---|---|
| H1 | H1 | focal comprime probabilidades |
| H2 | H5 + Tabela 4 (T9.3) | T global insuficiente por estrato/classe |
| H3 | H3 | receptive field curto |
| H4 | §6.4 (threshold `analysis_only`) | ganho de threshold tuning |
| H5 | §7 (otimização) | A0 sub-treinado |
| H6 | H10 | semântica de NORM |
| H7 | H7 + H8 | atribuição arch×loss×seed + splits não pareados |

---

## 2. Células de controle obrigatórias (H7) — pré-requisito de tudo

| Célula | Arch | Loss | γ | Seed | Split | Objetivo |
|---|---|---|---|---:|---|---|
| C0 | A0 | BCE | — | 13 | **pareado v2** | baseline pareado |
| C1 | A1 | BCE | — | 13 | pareado v2 | **isolar arquitetura** (a célula que nunca existiu) |
| C2 | A1 | focal | 2,0 | 13 | pareado v2 | reproduzir A2-full sob protocolo (sanidade ±IC) |
| C3 | A0 | focal | 2,0 | 13 | pareado v2 | isolar loss no backbone A0 |

**Regras duras:** sem C1, nenhuma conclusão sobre arquitetura é válida; sem C3, nenhuma conclusão
sobre loss é válida. As 4 células compartilham o mesmo split pareado v2 (seção 9), seed 13 e
orçamento idêntico (30 épocas, batch 64, Adam 1e-3, ES por `val_macro_pr_auc`).

Leitura esperada (pré-registro): se C1 ≈ C2 (ICs sobrepostos), o ganho do A2-full é
**arquitetura**; se C1 ≈ C0 e C3 ≈ C2, o ganho é **loss**; padrões intermediários = efeito misto.

---

## 3. Ablação de focal γ (H1)

| Célula | γ | α/weight | Métrica de sucesso (pré-registrada) |
|---|---:|---|---|
| F0 | 0,0 (= BCE) | nenhum | baseline probabilístico (esperado: ECE baixo, decisão fraca) |
| F1 | 0,5 | nenhum | ECE pós-T < 0,10 |
| F2 | 1,0 | nenhum | ECE pós-T < 0,08 |
| F3 | 2,0 | nenhum | referência atual (≡ C2) |
| F4 | 1,0 | `pos_weight` por prevalência (maquinaria existente em `pretrain_losses.py`) | ECE pós-T < 0,08 **e** macro PR-AUC ≥ F2 |
| F5 | 1,0 | effective number (Cui 2019) | ECE pós-T < 0,08 **e** PR-AUC CD/HYP ≥ F2 |

Leitura dose-resposta: se ECE cru cair monotonicamente com γ (F0<F1<F2<F3), H1 é confirmada e o
γ operacional é o menor que preserva macro PR-AUC. Dependência: roda sobre a arch vencedora de
C0–C3 (não antes).

---

## 4. Ablação de calibração (H2)

| Célula | Método | Métrica de sucesso |
|---|---|---|
| K0 | T global (referência — já existe: T=0,3741, ECE 0,0152, NORM=0 0,2167) | ECE global < 0,025 |
| K1 | Platt por classe (5 × {a, b}) | **ECE NORM=0 < 0,10** em todas as classes |
| K2 | T por estrato NORM (2 × T: NORM=1 / NORM=0) | ECE NORM=0 < 0,10 |
| K3 | Vector scaling (matriz diagonal 5×5 sobre logits) | ECE global < 0,020 **e** ECE NORM=0 < 0,12 |

**Restrição dura:** macro PR-AUC e macro AUROC inalteradas (Δ < 1e-4) — calibração não pode
mudar ranking (monotonia verificada por teste no avaliador). Custo baixo: K0–K3 operam sobre
predições existentes + split de calibração (sem retreino) quando aplicável ao melhor modelo das
células C/F.

---

## 5. Ablação de receptive field (H3)

| Célula | Modificação | RF estimado (derivado) | Métrica de sucesso |
|---|---|---:|---|
| R0 | A1 atual | 188 ms | CD PR-AUC referência (0,5561) |
| R1 | stem k7 → k15 | ~300 ms | CD PR-AUC ≥ +2 p.p. |
| R2 | stem k7 → k31 | ~500 ms | CD PR-AUC ≥ +3 p.p. |
| R3 | blocos multi-scale (k3+k7+k15 paralelos) | variável (~500 ms) | CD PR-AUC ≥ +3 p.p. e HYP ≥ +2 p.p. |
| R4 | janela 2000 ms (input 1000 amostras) | ~376 ms (mesma arquitetura) | CD/MI PR-AUC ≥ +3 p.p. |

**Restrições TinyML obrigatórias por célula:** params < 64k · FlatBuffer < 64 KB · latência
Renode < 200 ms · SRAM total < 128 KB. R4 dobra o custo de entrada — medir latência antes de
qualquer promoção (A2-full hoje: 73 ms; orçamento folga ~2,7×). Params estimados em design review
antes da execução (T10.3).

---

## 6. Ablação de threshold (H4)

| Célula | Política | Split de fit | Métrica de sucesso |
|---|---|---|---|
| T0 | `fixed_0.5` | — | macro F1@0.5 (referência: 0,6089 no A2-full) |
| T1 | `max_f1_per_class` | **calibration separado** | macro F1@tuned ≥ +4 p.p. sobre T0 (prospectivo) |
| T2 | `min_sensitivity_per_class` (recall ≥ 0,30) | calibration separado | sensibilidade mínima por classe com F1 ≥ T0 |
| T3 | `cost_sensitive` (FN > FP; custos a definir em T9.4) | calibration separado | custo clínico esperado mínimo |

**Regra dura:** T1–T3 **nunca** usam test set para fit. O avaliador canônico v2 já separa
fit/apply (`src/evaluation/thresholding.py`) e carimba `protocol_status` — toda célula T deve
produzir artefatos PROSPECTIVE (fit em calibration, apply em test). Os números RETROSPECTIVE de
T9.3 (+12–14 p.p.) são teto otimista, não evidência.

---

## 7. Ablação de orçamento de treino (H5)

| Célula | max_epochs | patience | Early stopping | Métrica de sucesso |
|---|---:|---:|---|---|
| B0 | 30 | 5 | `val_loss` (referência legada) | baseline A0 |
| B1 | 100 | 10 | **`val_macro_pr_auc`** | macro PR-AUC ≥ +1 p.p. sobre B0 |
| B2 | 100 | 10 | `val_macro_pr_auc` + EMA de pesos | macro PR-AUC ≥ +1 p.p. e ECE ≤ B1 |

Aplicado ao backbone A0 (teste direto de H5) e, se H5 confirmar, repetido no arch vencedor de
C0–C3. Protocolo v2 obrigatório: ES por métrica equalizada, nunca por BCE/focal cru.

---

## 8. Ablação de formulação (H6)

| Célula | Formulação | Métrica de sucesso |
|---|---|---|
| P0 | Multi-label 5 classes (atual) | referência |
| P1 | Multi-label 4 classes (NORM = ausência das 4) | macro PR-AUC das 4 classes patológicas (Δ vs P0) |
| P2 | Multi-label 5 classes + prior de co-ocorrência na loss | ECE NORM=0 < P0 com macro PR-AUC ≥ P0 |

**Dependência de governança:** P1 muda a semântica da ontologia — exige RFC de ontologia antes da
execução (não é apenas célula experimental). P0/P2 não requerem RFC.

---

## 9. Definição do split pareado v2 (normativa — nada é gerado nesta task)

```yaml
split_id: chapman-record-disjoint-paired-v2
derivation: chapman-record-disjoint-val0.1-seed13   # mesmo universo do A2-full
purpose: comparação pareada entre todas as células (C/F/R/B/P)
partitions:
  train: 80% dos registros (seed 13, record-disjoint)
  validation: 10% (early stopping + seleção de época)
  calibration: 5% (fit de T e thresholds — NUNCA em validation/test)
  test: 5% (congelado; somente avaliação final de célula)
rules:
  - mesmo train/val/calibration/test para TODAS as células
  - estratificação aproximada por superclasse (desvio de prevalência < 1 p.p. por partição)
  - suporte por classe por partição registrado em manifesto com SHA-256
  - geração única, write-once, com freeze de hash (mesma disciplina dos splits v3/v4)
immutability: true
```

Regra: qualquer célula executada fora deste split (ou de sucessores versionados dele) recebe
`NON_COMPARABLE` do avaliador canônico (`src/evaluation/schema.py`). A **geração** deste split é
task de T9.4/T10.3 com governança de splits — este documento apenas a normatiza.

---

## 10. Métricas de sucesso transversais (toda célula, sem exceção)

```text
macro_pr_auc                    # primária (early stopping + seleção)
macro_auroc                     # comparabilidade externa
macro_f1_at_0.5
macro_f1_tuned                  # thresholds fit SOMENTE em calibration
ece_post_calibration            # global, n_bins=15
ece_post_calibration_norm0      # estrato NORM=0 (novo — H2)
brier_post_calibration
bce_post_temperature
per_class_pr_auc com IC95 bootstrap (n≥200)
delta_int8_macro_pr_auc         # se a célula for quantizada
protocol_status                 # PROSPECTIVE obrigatório para claims; RETROSPECTIVE = análise
```

Artefatos por célula: `experiments/<run>/evaluation_v2/` completo (schema 2.0) + entrada de
reconciliação contra a referência da sua seção (C0, F3, K0, R0, T0, B0, P0 conforme o caso).

## 11. Critérios de promoção piloto → candidato (T10.3 → T11)

Um piloto só vira candidato se satisfazer **todos**:

1. macro PR-AUC ≥ 0,70 (piso do A2-full: 0,7065);
2. ECE pós-calibração global < 0,025;
3. ECE NORM=0 < 0,10 (todas as classes);
4. CD PR-AUC ≥ 0,58 (≥ +3 p.p. sobre a referência 0,5561);
5. Δ INT8 macro PR-AUC < 0,01;
6. FlatBuffer < 64 KB;
7. latência Renode < 200 ms;
8. SRAM total < 128 KB;
9. bit-exatidão atol ≤ 1 LSB (QG8);
10. reconciliação contra A2-full registrada no `evaluation_v2/` da célula.

Promoção exige adicionalmente: revisão humana + freeze de hash (ML Protocol v2 §11).

## 12. Orçamento experimental estimado (ESTIMATIVA — não executar nada aqui)

Base observável (host CPU-only, medido nos runs existentes): treino de 30 épocas ≈ **0,5–0,7
CPU-h** (A0 hist 32 min, A0 novo 40 min, A2-full 33 min — mtimes dos artefatos); avaliação
canônica completa ≈ 0,1 CPU-h (T9.3: ~4 min/run).

| Seção | Células | Treino/célula | Subtotal (ESTIMATIVA) |
|---|---:|---|---:|
| 2 — Controle C | 4 | 0,6 CPU-h | ≈ 2,4 CPU-h |
| 3 — Focal F | 5 novas (F3≡C2) | 0,6 CPU-h | ≈ 3,0 CPU-h |
| 4 — Calibração K | 3 novas | sem retreino (predições + calibration split) | ≈ 0,3 CPU-h |
| 5 — RF R | 4 novas | 0,6–1,2 CPU-h (R4 dobra input) | ≈ 3,2 CPU-h |
| 6 — Threshold T | 3 novas | sem retreino | ≈ 0,2 CPU-h |
| 7 — Orçamento B | 2 novas | ≈ 2,0 CPU-h (100 épocas) | ≈ 4,0 CPU-h |
| 8 — Formulação P | 2 novas | 0,6 CPU-h | ≈ 1,2 CPU-h |
| **Total** | **23 execuções** | — | **≈ 14–17 CPU-h + avaliações ≈ 2,5 CPU-h** |

Sequência de execução (ordem obrigatória): **C → F → {K, T} → R → B → P** (K/T são baratas e
podem rodar logo após C; R/B/P dependem dos vencedores anteriores). Execução incremental:
relatório parcial após cada seção — nunca o pacote inteiro de uma vez.

## 13. Dependências e bloqueios

| Dependência | Bloqueia | Estado |
|---|---|---|
| T9.4 — configs de treino v2 (YAML por task profile, split pareado referenciado) | T10.3 | pendente — **próxima task** |
| G6 — `reconcile_with_legacy` ler schema aninhado do `calibration.json` legado | T9.5 (RFC precisa reconciliar A0 novo sem adapter) | aberto — hotfix pequeno no avaliador |
| Split pareado v2 — geração + freeze (seção 9) | células C0–C3 e todas as seguintes | pendente (task de T9.4/T10.3 com governança) |
| RFC de ontologia (H6/P1) | célula P1 | pendente — decisão humana |
| Aprovação humana explícita para treinos | T10.3 (pilotos) e T11 (oficiais) | bloqueado por governança |
| Extensão do avaliador: IC por classe em artefato (G4) + métrica `ece_norm0` | claims das células K/T | desejável antes de T10.3 — registrar em T9.4 |
| **Trilha D (teacher/destilação)** — adendo T10.2.1: `docs/ablation_matrix_v2_appendix_D.md` (6 células D0–D5, hipótese H8, protocolo KD, ~18–20 h CPU) | H8 (capacidade) | definida; ordem de execução D0 → D1 → D2 → {D3, D5} → D4 (condicional a D2 ≥ +3 p.p. CD) |

## 14. Declarações finais

> Nenhum treino foi executado na produção deste documento.
> Nenhum Quality Gate foi alterado.
> Nenhum artefato em `models/` foi modificado.
> Nenhum split existente foi re-gerado; o split pareado v2 é definição normativa.
> A promoção de qualquer célula a candidato exige revisão humana (T11).
> O QG4-BCE permanece FAIL; sua revisão é escopo da RFC T9.5.
