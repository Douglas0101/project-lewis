# 06 — Auditoria de Atalhos de Domínio

**Status:** PROPOSTO — aguardando ratificação humana
**Data:** 2026-07-18
**Entregável:** `domain_shortcut_audit` (7)
**Corrige:** DQ-14 (e valida a correção de DQ-01/DQ-02/DQ-08)

---

## 1. Pergunta central

O modelo aprendeu morfologia e dinâmica cardíaca — ou a origem do arquivo? Baseline do defeito
(medido): ROC-AUC 0,5546 global; recall @0,54 por dataset svdb 0,467 / incart 0,119 / mitdb
0,104; features com |corr(dataset)| até 0,758. Esta auditoria é **obrigatória** para qualquer
candidato antes de promoção.

## 2. Probe de identidade do dataset

Treinar, **somente para auditoria**, um classificador `dataset_id` a partir das representações:

| Representação | Expectativa pós-correção | Estado se falhar |
|---|---|---|
| embeddings brutos | probe ≈ chance (1/3) | `DATASET_SHORTCUT_LEARNING` |
| embeddings após harmonização (normalização/λ_inv) | probe ≈ chance | `DATASET_SHORTCUT_LEARNING` |
| embeddings condicionados à classe verdadeira | probe ≈ chance dentro de cada classe | `DATASET_SHORTCUT_LEARNING` |
| embeddings agregados por paciente | probe ≈ chance | `DATASET_SHORTCUT_LEARNING` |

Limiar de decisão: probe com balanced accuracy acima de `chance + δ` com IC 95% excluindo chance
→ atalho material. Se a identidade do dataset continuar recuperável **e** explicar a
classificação clínica (ablation: remover direção do probe degrada a tarefa), a matriz de
treinamento é interrompida (G-F2, 05 §6).

| Parâmetro | Fórmula | Unidade | Intervalo candidato | Método de seleção | Dados permitidos | Risco | Status |
|---|---|---|---|---|---|---|---|
| δ (atalho material) | probe_acc > 1/3 + δ | acc | {0,05; 0,10} | ratificação | inner/outer leitura | médio | PROPOSED_REQUIRES_RATIFICATION |

## 3. Leave-one-dataset-out (LODO)

Para cada domínio d ∈ {mitdb, svdb, incart} (+ afdb no nível ritmo, se D3 ratificado):

```math
D_{\mathrm{train}} = D \setminus D_d, \qquad D_{\mathrm{test}} = D_d
```

- Folds LODO também separados por paciente; mesma família vencedora da matriz (05).
- Reportar degradação por dataset vs validação interna (ΔF1_macro, ΔAUC, ΔECE), por classe.
- LODO não substitui validação externa prospectiva; é o teste mínimo de transporte.

## 4. Desempenho condicional (nenhuma média global esconde colapso)

```math
M(\text{classe}=c,\ \text{dataset}=d),\quad
M(\text{classe}=c,\ \text{paciente}=p),\quad
M(\text{classe}=c,\ \text{qualidade}=q)
```

- Tabela completa por candidato, com denominadores (pacientes/registros/batimentos).
- Regra: se alguma célula (c, d) fica abaixo do piso ratificado → `REVIEW_REQUIRED`, mesmo que a
  média passe. Pisos por classe são definidos em 11 com as margens ratificadas.

## 5. Testes contrafactuais

A predição clínica **não pode mudar materialmente** quando apenas fatores não fisiológicos mudam
(expectativa de invariância); e **deve** mudar quando a fisiologia relevante muda (sensibilidade
desejada):

| Intervenção | Invariância esperada? | Tolerância candidata |
|---|---|---|
| escala global de amplitude (×0,5–2,0) | sim (normalização por registro/dataset) | ΔP < 0,05 |
| reamostragem 500↔250↔500 | sim | ΔP < 0,05 |
| padding (edge) | sim | ΔP < 0,02 |
| offset DC / baseline wander leve | sim | ΔP < 0,05 |
| ruído gaussiano leve (SNR ≥ 20 dB) | degradação suave, não abrupta | relatório de curva |
| troca de nome/origem do arquivo | sim (bit-identical) | ΔP = 0 |
| lead (MLII↔II em registros com ambos) | degradação limitada | relatório |
| normalização por registro vs global | sim (contrato único) | ΔP < 0,02 |
| inversão de polaridade | documentar comportamento | relatório |
| deslocamento de R (±50 ms) | degradação suave | relatório |

| Parâmetro | Fórmula | Unidade | Intervalo candidato | Método de seleção | Dados permitidos | Risco | Status |
|---|---|---|---|---|---|---|---|
| tolerância contrafactual | ΔP máx | probabilidade | {0,02–0,05} | ratificação | inner/outer leitura | médio | PROPOSED_REQUIRES_RATIFICATION |

## 6. Estados de saída

```text
DATASET_SHORTCUT_LEARNING      atalho material confirmado → bloqueio
INSUFFICIENT_EVIDENCE          probes/testes não conclusivos → bloqueio
REVIEW_REQUIRED                resultado misto ou célula condicional abaixo do piso
```

Nenhum estado de aprovação é emitido por esta auditoria isoladamente; ela alimenta os gates de
promoção (11).
