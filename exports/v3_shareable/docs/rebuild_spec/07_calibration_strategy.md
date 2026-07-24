# 07 — Estratégia de Calibração (D6), Calibração Hierárquica e Máquina de Estados (D7)

**Status:** PROPOSTO — aguardando ratificação humana
**Data:** 2026-07-18
**Entregáveis:** `calibration_strategy` (8), `hierarchical_calibration_specification` (9),
`autonomous_calibration_state_machine` (10)

---

## 1. Princípio

**Calibração nunca corrige ausência de discriminação.** Evidência do estado legado: ROC-AUC
0,5546 e calibração não monotônica no subset focal — calibrar isso produziria confiança
organizada em torno do acaso. Calibração só começa após as pré-condições da §2.

## 2. Pré-condições duras (todas obrigatórias)

1. Ranking útil: ROC/PR-AUC acima do baseline com IC 95% (bootstrap por paciente) excluindo o
   aleatório, por tarefa e classe crítica.
2. Estabilidade entre folds e seeds (P10 fold–seed dentro do piso ratificado).
3. Suporte suficiente por classe (pacientes, não só batimentos — ver 05 §4).
4. Ausência de leakage (06 aprovado; splits verificados).
5. Dados de calibração **independentes** do treino do modelo-base (partição própria por
   paciente, congelada no manifest).
6. Threshold clínico **não** é escolhido na calibração nem no teste: vem da validação interna,
   é co-produzido com modelo+scaler+calibrador e vinculado por hash (08).

## 3. Métodos candidatos

| Escopo | Métodos |
|---|---|
| binário / multilabel | temperature scaling; Platt/logistic; beta calibration; isotonic (somente com suporte suficiente) |
| multiclasse | temperature; vector scaling; matrix scaling; Dirichlet calibration |

Seleção no inner loop da partição de calibração, por tarefa. Comparação obrigatória com:

```math
NLL = -\frac{1}{n}\sum_i \log p_{i,y_i}
\qquad
Brier = \frac{1}{n}\sum_i\sum_c (p_{ic}-y_{ic})^2
\]
```math
ECE = \sum_{b=1}^{B} \frac{|S_b|}{n}\,|acc(S_b)-conf(S_b)|
\quad (+ \text{classwise-ECE e reliability diagrams})
```

ECE sozinho é proibido como critério. Ganho de calibração não pode degradar métricas clínicas
além da margem ratificada (§6, estado C).

| Parâmetro | Fórmula | Unidade | Intervalo candidato | Método de seleção | Dados permitidos | Risco | Status |
|---|---|---|---|---|---|---|---|
| B (bins ECE) | — | bins | {10, 15} | convenção + suporte | calibração | baixo | PROJECT_EXISTING |
| método por tarefa | — | — | {temperature, vector, Dirichlet, beta, Platt} | menor NLL na partição de calibração | calibração | médio | PROPOSED_REQUIRES_RATIFICATION |
| isotonic mínimo | pacientes/classe | — | {≥ 50 pacientes/classe} | ratificação | calibração | alto | PROPOSED_REQUIRES_RATIFICATION |

## 4. Matriz de decisão — D6 (topologia do calibrador)

| Opção | Descrição | Uso recomendado |
|---|---|---|
| global | um calibrador por modelo | baseline obrigatório |
| **por tarefa** | quality/beat/rhythm/diagnosis separados | **recomendado** |
| por classe | um por classe (one-vs-rest) | somente classes com suporte; senão instável |
| **hierárquico** | pooling parcial entre classes/subgrupos (§5) | **recomendado para classes raras (FUSION, AFIB)** |
| condicionado ao domínio | por dataset | **proibido** quando compensar não-generalização; permitido só se domínio for legítimo, declarado e ratificado (→ `REVIEW_REQUIRED` por padrão) |
| política de abstenção | rejeição por confiança/entropia/OOD | **obrigatória** (§7) |

## 5. Calibração hierárquica entre classes e subgrupos

Para classe c e subgrupo/domínio d:

```math
\operatorname{logit}(p'_{c,d}) = \alpha_{c,d} + \beta_{c,d}\,\operatorname{logit}(p_c)
\]
```math
\alpha_{c,d} \sim \mathcal N(\mu_{\alpha,c}, \sigma_{\alpha,c}^{2}),
\qquad
\log \beta_{c,d} \sim \mathcal N(\mu_{\beta,c}, \sigma_{\beta,c}^{2})
\]

Objetivos: compartilhar evidência entre classes relacionadas (FUSION↔V; AFIB↔AFL); impedir
calibradores instáveis em classes raras (shrinkage para a média da classe); preservar diferenças
clínicas reais (hiperpriors frouxos o bastante); **quantificar a incerteza** de α e β e publicá-la
por classe. Se o IC de β_{c,d} inclui 0 com suporte insuficiente, o calibrador da classe regride
ao global da tarefa — nunca inventa precisão.

## 6. Máquina de estados — calibração autônoma limitada (D7)

### Estado A — Monitoramento

```text
CALIBRATION_STABLE | CALIBRATION_DRIFT_SUSPECTED | INSUFFICIENT_FEEDBACK_LABELS | DOMAIN_SHIFT_DETECTED
```

### Estado B — Treinamento em shadow (permitido somente quando TODOS)

- labels pós-uso revisados por humano;
- janela temporal congelada e versionada;
- política de inclusão versionada;
- número efetivo de **pacientes** suficiente;
- ontologia inalterada; modelo-base imutável; preprocessing inalterado;
- nenhum dado de teste regulatório utilizado.

### Estado C — Avaliação pareada (mesmos pacientes, atual × candidato)

```math
\Delta Brier,\ \Delta NLL,\ \Delta ECE_c,\ \Delta sensitivity_c,\ \Delta specificity_c
```

Aprovação automática **proibida**; qualquer degradação clínica além da margem ratificada →
rejeição; ganho apenas em ECE com perda de sensibilidade crítica → rejeição explícita.

### Estado D — Bundle candidato

O sistema pode emitir somente `CALIBRATION_CANDIDATE_READY`. **Nunca** `CALIBRATION_ACTIVATED`
sem quorum humano e assinatura válida (09). Nível máximo permitido agora:

```text
D7 = LEVEL_1_SHADOW_RECALIBRATION
```

`LEVEL_2_SIGNED_CANDIDATE_GENERATION` só com attestation funcional (09) ratificada;
`LEVEL_3_HUMAN_AUTHORIZED_ACTIVATION` só com quorum + revisão clínica. **Proibido** atualizar
autonomamente: pesos, arquitetura, ontologia, mapeamento de classes, preprocessing, threshold
clínico, critérios de exclusão, população-alvo, finalidade prevista.

## 7. Abstenção e revisão humana

Política de rejeição por: baixa confiança; alta entropia `H(p) = −Σ p_c log p_c`; discordância
entre modelos; qualidade insuficiente; OOD; inconsistência hierárquica; calibração não confiável.

```text
PREDICTION_ACCEPTED_FOR_RESEARCH | ABSTAIN_LOW_CONFIDENCE | ABSTAIN_POOR_SIGNAL |
ABSTAIN_OUT_OF_DISTRIBUTION | REVIEW_REQUIRED
```

Avaliação obrigatória com **curvas risco–cobertura** por tarefa e classe crítica; limiares de
abstenção escolhidos na validação interna e congelados (nunca no teste).

| Parâmetro | Fórmula | Unidade | Intervalo candidato | Método de seleção | Dados permitidos | Risco | Status |
|---|---|---|---|---|---|---|---|
| limiar de abstenção | H(p) > h ou max p < τ | — | {validação interna} | curva risco–cobertura | inner-val | alto | PROPOSED_REQUIRES_RATIFICATION |
| margem degradação clínica (estado C) | ΔSe_c, ΔSpe_c | — | {0,00–0,02} | ratificação clínica | pareado shadow | alto | PROPOSED_REQUIRES_RATIFICATION |
