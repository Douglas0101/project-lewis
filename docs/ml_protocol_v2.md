# ML Protocol v2 — Pipeline ML com Métricas Equalizadas e Governança Experimental

**Versão:** v2.0.0 · **Data:** 2026-08-01 · **Branch:** `develop` @ `b0ca382`
**Prompt de origem:** SDD-LEWIS-CLI-ML-PROTOCOL-V2-001 · **Status:** normativo (vigente para novas avaliações)

> Este documento é **normativo, não executável**: define o protocolo único de métricas, calibração,
> thresholds, comparação e treino do pipeline ML do Project-Lewis. **Nenhum código de Quality Gate
> foi alterado por este documento.** Mudanças de QG exigem RFC, governança e revisão humana
> (regras 2.1/2.3 do prompt de origem). Os valores do run A2-full citados como exemplos foram
> verificados contra `experiments/20260728_053011_pretrain_chapman/` (`calibration.json`,
> `metrics_per_class.json`, `quantized/`).

---

## 1. Escopo

O ML Protocol v2 cobre três perfis de tarefa (`task_profile`), cada um com contrato próprio de
split, ontologia, métricas e thresholds:

| `task_profile` | Tarefa | Datasets | Estado |
|---|---|---|---|
| `pretrain_scp_ecg_multilabel` | Pré-treino multi-rótulo de superclasses SCP-ECG (NORM, CD, MI, HYP, STTC) | Chapman-Shaoxing | Ativo (backbone A2-full) |
| `beat_classification_aami` | Classificação de batimentos AAMI em dois estágios (N vs Anormal → S/V/F) | MIT-BIH, SVDB, INCART | Ativo (produção Stage 1/2 v2.0) |
| `rhythm_afib_afl` | Classificação de ritmo AFIB/AFL em escopo de episódio | AFDB | Futuro/contexto (decisão D3: fora do classificador de batimentos) |

Fora de escopo: promoção de modelos (regida por governança, seção 11), alteração de QGs em código,
e qualquer treino — este documento apenas os normatiza.

---

## 2. Princípios

1. **Fonte única de verdade para métricas** — toda métrica reportada deriva do avaliador canônico
   (`evaluator_version`), nunca de logs ad hoc.
2. **Avaliação offline canônica** — a métrica válida é a offline, recomputada de
   `y_true` × `y_score`; log de treino Keras é telemetria, não evidência (ex.: AUC Keras 0,8596 vs
   AUC offline 0,8639 no A2-full).
3. **Métricas equalizadas por classe** — macro-médias não ponderadas; classes raras (CD, HYP, F)
   não podem ser escondidas por agregados dominados pela classe majoritária.
4. **Separação train/validation/calibration/test** — quatro papéis distintos; threshold e
   temperatura são aprendidos fora do teste.
5. **Prevenção de leakage por paciente/registro** — splits patient-disjoint (batimentos) ou
   record-disjoint (pré-treino Chapman), conforme o `task_profile`; `split_id` é parte do contrato.
6. **Calibração obrigatória** — nenhum modelo é avaliado ou promovido sem calibração medida antes
   e depois da quantização (seção 5).
7. **Comparação somente sob mesmo protocolo** — fora do contrato de comparabilidade, o veredito é
   `NON_COMPARABLE` (seção 7).
8. **Reprodutibilidade** — toda run carrega seed, config YAML versionado e manifesto de
   proveniência com SHA-256.

---

## 3. Dicionário de métricas

### 3.1 Métricas primárias (pré-treino SCP-ECG multi-rótulo)

| Métrica | Definição | Papel |
|---|---|---|
| `macro_pr_auc` | Média não ponderada da área sob a curva precision–recall por classe | Métrica principal sob desbalanceio; critério de early stopping do pré-treino |
| `macro_auroc` | Média não ponderada da AUC-ROC por classe | Comparabilidade externa (literatura reporta AUROC) |
| `ece_post_calibration` | Expected Calibration Error após temperature scaling, `n_bins=15` | Evidência de calibração utilizável |
| `brier_mean` | Brier score médio (multi-rótulo: média sobre classes e amostras) | Probabilidade bem ordenada em magnitude |
| `delta_quantization_macro_pr_auc` | `macro_pr_auc_float − macro_pr_auc_int8` | Guarda de fidelidade da quantização |

### 3.2 Métricas secundárias

| Métrica | Definição |
|---|---|
| `per_class_pr_auc` / `per_class_auroc` | PR-AUC / AUROC por classe, com suporte |
| `per_class_f1` / `per_class_precision` / `per_class_recall` | Métricas de decisão por classe no threshold vigente |
| `macro_f1_at_0.5` | Macro-F1 com threshold fixo 0,5 |
| `macro_f1_tuned` | Macro-F1 com thresholds tunados (política da seção 6) |
| `bce` | Binary cross-entropy média (pré-calibração) |
| `bce_post_temperature` | BCE após `logits / T → sigmoid` |
| `nll` / `nll_post_temperature` | Negative log-likelihood pré/pós-temperatura |
| `mce` | Maximum Calibration Error (pior bin) |
| `rejection_rate` | Fração roteada para `ABSTAIN_*` / `Q_OR_UNKNOWN` |

### 3.3 Métricas de guarda (edge/firmware)

| Métrica | Definição | Referência A2-full (verificada) |
|---|---|---|
| `model_size_int8` | Tamanho do FlatBuffer INT8 | 54,77 KB |
| `latency_renode` | Latência por inferência no Renode (via `lewis_hal_millis`/TIM2) | 73 ms |
| `sram_total` | SRAM total consumida | 52,4 KB |
| `arena_used` | Arena TFLM utilizada | 22.820 / 49.152 B |
| `bitexact_atol_1_lsb` | Bit-exatidão int8 vs Python (atol 1 LSB) | PASS (QG8) |
| `cosine_fidelity` | Similaridade cosseno vs ground-truth | 1,000000 (QG10) |
| `saturation_int8` | Fração de ativações saturadas na quantização | reportar por run |
| `sha256_provenance` | SHA-256 dos artefatos no manifesto de proveniência | `provenance.json` |

---

## 4. Regras de equalização

1. **Macro-média não ponderada por suporte** — cada classe pesa 1, independentemente de NORM
   dominar o suporte.
2. **Suporte sempre reportado** — toda tabela por classe inclui `support`; métrica sem suporte é
   inválida.
3. **PR-AUC é a métrica preferencial** para classes desbalanceadas; AUROC é mantida para
   comparabilidade externa.
4. **F1@0.5 e F1@tuned são sempre reportados juntos** — nunca apenas um deles.
5. **Thresholds são tunados apenas em validation/calibration, nunca em test** — o teste recebe
   thresholds congelados.
6. **`n_bins=15` é o padrão** de ECE/MCE/reliability; desvios devem ser justificados e marcados.
7. **ECE/MCE/Brier/BCE são sempre acompanhados** de `reliability.json` (reliability diagram em
   JSON), nunca reportados como escalar isolado.
8. **BCE absoluto não é métrica única de calibração** — BCE mistura discriminação e calibração;
   sua leitura exige par pré/pós-temperatura e Brier/ECE ao lado (seção 5).

---

## 5. Protocolo de calibração

Sequência obrigatória (temperature scaling como parte do pipeline, não pós-ajuste opcional):

```text
1. Treinar modelo.
2. Gerar logits/probabilidades no validation set.
3. Usar calibration set separado.
4. Aprender temperature T (minimizando NLL no calibration set).
5. Aplicar logits / T → sigmoid.
6. Recalcular ECE, Brier, BCE, NLL.
7. Congelar T (artefato calibration.json, com hash).
8. Validar T pós-quantização (ECE pós-PTQ com T fixo).
9. Validar T no firmware, quando aplicável (inferência: logits int8 → dequant → /T → sigmoid).
```

Leitura obrigatória dos resultados:

- `T < 1` indica **underconfidence** (probabilidades comprimidas; T as afia). Exemplo verificado:
  A2-full com `T = 0,3741`, ECE 0,1508 → 0,0152, NLL 0,4317 → 0,3417.
- `T > 1` indica **overconfidence**.
- **ECE mede magnitude, não direção** — a direção vem do reliability diagram e de `T − 1`.
- Temperature scaling é monotônico nos logits: **AUROC e PR-AUC são invariantes a T**; o que muda é
  BCE/NLL/Brier/ECE. Divergência de AUROC pré/pós-T é defeito do avaliador.

---

## 6. Política de thresholds

Catálogo de políticas (`threshold_policy`):

| Política | Descrição | Uso |
|---|---|---|
| `fixed_0.5` | Threshold fixo 0,5 em todas as classes | Baseline; sempre reportado |
| `max_f1_per_class` | Threshold por classe maximizando F1 no calibration set | **Padrão `pretrain_scp_ecg_multilabel`** |
| `cost_sensitive` | Threshold por classe minimizando custo clínico (FN > FP) | **Padrão `beat_classification_aami`** (alternativa: `max_f1_per_class`) |
| `min_sensitivity_per_class` | Maior threshold que atinge sensibilidade mínima por classe | Cenários com piso de recall (ex.: QG5′) |
| `rejection_aware` | Threshold combinado à política de abstenção `Q_OR_UNKNOWN` | Quando rejeição estiver ativa |

Regras:

1. Thresholds são aprendidos **somente** em validation/calibration e aplicados **congelados** ao
   teste (`fit_split: calibration`, `apply_to_test: frozen`).
2. Todo `metrics.json` referencia o `thresholds.json` vigente; F1 sem política declarada é inválido.
3. Mudança de `threshold_policy` quebra comparabilidade (seção 7).

---

## 7. Protocolo de comparação

Dois modelos/avaliações só são comparáveis se compartilharem **todos** os campos do contrato:

```text
evaluator_version      (ex.: v2.0)
task_profile           (ex.: pretrain_scp_ecg_multilabel)
split_id               (ex.: chapman-record-disjoint-val0.1-seed13)
ontology_version       (ex.: v3)
n_bins                 (padrão 15)
calibration_method     (ex.: temperature_scaling, ou none)
threshold_policy       (seção 6)
preprocessing_version  (ex.: v1.0)
```

Qualquer divergência ⇒ veredito obrigatório:

```text
status = NON_COMPARABLE
```

Notas:

- Tarefas diferentes (pré-treino SCP-ECG vs batimentos AAMI vs ritmo AFIB/AFL) são sempre
  `NON_COMPARABLE` entre si.
- Métricas de log Keras vs offline são `NON_COMPARABLE` por construção (avaliadores diferentes).
- Comparações externas (literatura) devem declarar as ressalvas estruturais (1 lead vs 12 leads,
  orçamento de parâmetros, dataset) e marcar lacunas como `N/A — não reportado`.

---

## 8. Protocolo de treino reestruturado

1. **Early stopping por métrica equalizada**:
   - pré-treino: `val_macro_pr_auc` (mode `max`);
   - batimentos: `val_macro_f1` (mode `max`).
2. **BCE bruto nunca é critério único** de seleção ou parada.
3. **Catálogo de losses** (`loss_catalog`): `bce`, `focal`, `focal_class_weighted`,
   `class_balanced_focal`, `asymmetric_loss` (ASL).
4. **Catálogo de sampling** (`sampling_catalog`): `random`, `class_balanced`, `oversample_rare`,
   `hard_example_mining`.
5. **Calibração obrigatória** ao final de todo treino (seção 5).
6. **SMOTE apenas em treino, nunca em validation/test** — e somente no espaço de features
   (Regra de Ouro 6). `smote_on_validation`, `smote_on_test` e `test_threshold_tuning` são
   proibidos por contrato.
7. **Augmentação apenas no treino** e somente com transformações fisiologicamente válidas para ECG.
8. **Toda run tem config YAML versionado** (task profile, seed, split, loss, sampling, métricas).
9. **Proibições específicas de split**: o split `v4.0-patient-disjoint` (Stage 2) não é usado para
   pré-treino Chapman; splits existentes são imutáveis.

---

## 9. Estrutura de artefatos

Toda avaliação protocolo v2 escreve em diretório novo, sem modificar artefatos originais da run:

```text
experiments/<run>/evaluation_v2/
  metrics.json                # agregados, schema 2.0 (seção 10)
  metrics_per_class.json      # por classe, com suporte
  calibration.json            # T, método, split de ajuste, ECE/NLL/Brier pré/pós
  thresholds.json             # política, thresholds por classe, fit_split
  reliability.json            # reliability diagram (bins, conf, acc, count)
  confidence_intervals.json   # ICs (bootstrap) das primárias
  reconciliation.json         # reconciliação com métricas legadas da run
```

Se as predições não existirem, o avaliador as gera em modo read-only sobre o checkpoint e grava em
`evaluation_v2/predictions/`, sem tocar nos artefatos originais.

---

## 10. Schema mínimo do `metrics.json`

```json
{
  "schema_version": "2.0",
  "run_id": "...",
  "task_profile": "pretrain_scp_ecg_multilabel",
  "split_id": "chapman-record-disjoint-val0.1-seed13",
  "ontology_version": "v3",
  "evaluator_version": "v2.0",
  "n_samples": 45040,
  "metrics": {
    "macro_pr_auc": null,
    "macro_auroc": null,
    "macro_f1_at_0.5": null,
    "macro_f1_tuned": null,
    "bce": null,
    "bce_post_temperature": null,
    "nll": null,
    "nll_post_temperature": null,
    "brier_mean": null,
    "ece_pre_calibration": null,
    "ece_post_calibration": null,
    "mce_post_calibration": null,
    "temperature": null
  },
  "per_class": {},
  "thresholds": {},
  "provenance": {}
}
```

Campos nulos são explícitos: ausência de `calibration.json` na run origem ⇒ `temperature = null` e
apenas métricas pré-calibração são preenchidas.

---

## 11. Critérios de promoção

Cadeia de estados obrigatória:

```text
CANDIDATE → EVALUATED → CALIBRATED → QUANTIZED → SIMULATED → PROMOTED → FROZEN
```

Promoção (transição para `PROMOTED`) exige **todas** as condições:

1. avaliação sob protocolo v2 (`evaluator_version` vigente);
2. métricas primárias aprovadas nos thresholds de QG vigentes;
3. calibração aprovada (ECE pós-T dentro do limiar governado);
4. quantização aprovada (`delta_quantization_macro_pr_auc` e guardas INT8);
5. firmware/simulação aprovados (bit-exatidão, latência, arena);
6. **revisão humana registrada**;
7. **freeze de hash** dos artefatos promovidos.

Nenhuma transição é feita por ferramenta de automação sem governança; `PROMOTED` sem revisão humana
é inválido por construção.

---

## Declaração de não-alteração

Este documento **não altera nenhum código de Quality Gate, threshold de QG, split, modelo em
`models/`, artefato do freeze E07R ou firmware**. A reavaliação do QG4-BCE é tratada em RFC própria
(task T9.5), com decisão humana. Publicação permanece em `HOLD` conforme governança E07R-PD.
