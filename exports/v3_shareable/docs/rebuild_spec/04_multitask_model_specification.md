# 04 — Especificação do Modelo Multitarefa (D5)

**Status:** PROPOSTO — aguardando ratificação humana
**Data:** 2026-07-18
**Entregável:** `multitask_model_specification` (5)

---

## 1. Variáveis-alvo (alinhadas à ontologia 01)

```math
y^{(q)} = \text{qualidade (softmax: VALID / NOISY / CLIPPED / FLATLINE / ...)}
```
```math
y^{(b)} = \text{morfologia do batimento (softmax: N / S / V / FUSION / Q\_OR\_UNKNOWN)}
```
```math
y^{(r)} = \text{ritmo (softmax em escopo de episódio: SINUS / AFIB / AFL / ...)}
```
```math
\mathbf{y}^{(d)} = \text{diagnósticos multilabel (sigmoides independentes, SCP-ECG)}
```

- Cabeça de **qualidade e abstenção antes** da classificação clínica: qualidade insuficiente →
  `ABSTAIN_POOR_SIGNAL`, nunca predição clínica.
- Ritmo exige contexto temporal; para janela de batimento isolado a cabeça de ritmo retorna
  `INSUFFICIENT_TEMPORAL_CONTEXT`.
- Diagnóstico só onde existem statements (chapman/ptbxl); nunca inferido de classe de batimento.

## 2. Função de perda candidata

```math
\mathcal L =
\lambda_q \mathcal L_q +
\lambda_b \mathcal L_b +
\lambda_r \mathcal L_r +
\lambda_d \mathcal L_d +
\lambda_{cal} \mathcal L_{cal} +
\lambda_{cons} \mathcal L_{cons} +
\lambda_{inv} \mathcal L_{inv}
```

| Componente | Papel | Seleção |
|---|---|---|
| L_q | qualidade/abstenção | cross-entropy |
| L_b | batimento | cross-entropy (ou focal — inner loop, ver 05 §4) |
| L_r | ritmo (episódio) | cross-entropy |
| L_d | diagnóstico multilabel | binary cross-entropy por label |
| L_cal | regularização probabilística (ex.: penalidade de entropia/extremos não suportados) | inner loop |
| L_cons | consistência hierárquica (§3) | inner loop |
| L_inv | penalidade anti-atalho de domínio (gradient reversal ou penalidade de probe — ver 06) | inner loop |

**Todos os λ são selecionados exclusivamente no inner loop da validação aninhada** e congelados
antes de qualquer outer test.

| Parâmetro | Fórmula | Unidade | Intervalo candidato | Método de seleção | Dados permitidos | Risco | Status |
|---|---|---|---|---|---|---|---|
| λ_q, λ_b, λ_r, λ_d | pesos escalares | — | {0,25–2,0} | busca inner loop | inner-train | médio | PROPOSED_REQUIRES_RATIFICATION |
| λ_cal | escalar | — | {0, 0,01–0,1} | inner loop | inner-train | médio | EXPERIMENTAL |
| λ_cons | escalar | — | {0, 0,01–0,5} | inner loop | inner-train | médio | EXPERIMENTAL |
| λ_inv | escalar | — | {0, 0,01–0,5} | inner loop | inner-train | alto | EXPERIMENTAL |

## 3. Consistência hierárquica (restrições duras)

```math
P(\text{AFIB episódio}) \not\equiv P(\text{FUSION batimento})
```
```math
P(\text{diagnóstico} \mid x) \neq P(\text{batimento anormal} \mid x)
```
```math
P(\text{diagnóstico}) \le P(\text{evidência clínica disponível})
```
```math
\text{janela sem contexto} \Rightarrow \texttt{INSUFFICIENT\_TEMPORAL\_CONTEXT}
```

`L_cons` penaliza violações (ex.: coerência entre agregação de batimentos e rótulo de episódio;
monotonicidade qualidade→abstenção). Violador estrutural das restrições → hard reject no gate
de promoção (11).

## 4. Matriz de decisão — D5 (arquitetura)

| Família | Descrição | Prós | Contras | Papel na matriz 4×5×5 (05) |
|---|---|---|---|---|
| (a) CNN-1D waveform | backbone atual, somente sinal | baseline congelado; já cabe no edge | não usa features clínicas; foi a que falhou (sob dados defeituosos) | candidata (baseline) |
| (b) MLP features | features clínicas (03) | barato; interpretável; rápido no edge | perde morfologia fina | candidata |
| (c) Fusão CNN+features | waveform + features clínicas | combina morfologia e dinâmica RR | maior custo; risco de atalho via features (mitigado por 03/06) | candidata |
| (d) Multitarefa CNN | cabeças quality/beat/rhythm (+diagnosis onde houver) | implementa a ontologia D2 nativamente; abstenção integrada | complexidade; dados de ritmo limitados (AFDB) | candidata |
| hierárquico 2 estágios | N-vs-Anormal → subclasses | forma de deployment atual; simples de auditar | erro de triagem propaga (medido: 93,4% dos anormais mortos no S1 sob dados defeituosos) | mantido como **forma de deployment**; candidato interno à família (a)–(d) |
| multitarefa total / multilabel | cabeças simultâneas sem cascata | sem propagação de erro de triagem | exige rebalanceamento conjunto | dentro de (d) |
| mixture-of-experts | especialistas por domínio/classe | capacidade | risco de especializar em dataset (DQ-14) | **EXPERIMENTAL** — fora da matriz principal |

**Recomendação:** executar as 4 famílias (a)–(d) na matriz 100 células com folds/seeds idênticos;
a cascata de dois estágios permanece como empacotamento de deployment da família vencedora, com
gate de triagem medido separadamente (ver 05 §6). A decisão final de arquitetura sai dos dados,
não desta especificação.

## 5. Restrições de edge (hard, de AGENTS.md/QG)

- FlatBuffer TFLM < 64 KB; arena TFLM ≤ 48/64 KB; inferência < 200 ms/batimento no Cortex-M4F;
  input shape (500, 1); quantização INT8 com ΔF1-macro < 2% (QG6); bit-exatidão QG8 contra
  BUILTIN_REF. Famílias que não couberem são descartadas para deployment independentemente de
  desempenho em pesquisa (podem seguir como referência científica).

## 6. Critérios de aceite

1. Toda cabeça com nível ontológico declarado e escopo temporal explícito.
2. Abstenção/INSUFFICIENT_TEMPORAL_CONTEXT testados em contrato (unidade).
3. Nenhuma saída de batimento apresentada como diagnóstico; nenhum ritmo de batimento único.
4. Orçamento edge verificado por build antes de qualquer promoção (QG6/QG7/QG9/QG12).
