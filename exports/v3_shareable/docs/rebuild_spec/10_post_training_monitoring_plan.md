# 10 — Plano de Monitoramento Pós-Treinamento

**Status:** PROPOSTO — aguardando ratificação humana; **não ativar nesta etapa**
**Data:** 2026-07-18
**Entregável:** `post_training_monitoring_plan` (13)

---

## 1. Princípios

- Baseline de referência **congelado no bundle** (08): distribuições, métricas e calibração da
  geração promovida. Toda comparação é contra esse baseline versionado.
- **Nenhum dado pós-produção é incorporado automaticamente ao treinamento** — alimenta somente
  os estados A→D da calibração (07 §6) após revisão humana.
- Drift não supervisionado **nunca** dispara recalibração direta: exige labels revisados e
  análise causal.

## 2. Famílias de drift

### 2.1 Drift de entrada
amplitude; fs; lead; qualidade do sinal; duração; missingness; distribuição de features.
Métricas: **MMD**, **Wasserstein**, **PSI** (auxiliar). Por feature do contrato (03) e por
subgrupo (lead/dispositivo/site).

### 2.2 Drift clínico
prevalência; composição de doenças e ritmos; proporção de classes; desempenho por doença e por
população; **falsos-negativos** (amostras rotuladas de auditoria revisadas por humano).

### 2.3 Drift probabilístico
NLL; Brier; calibration intercept; calibration slope; classwise-ECE; curvas de confiabilidade
contra o baseline.

### 2.4 Drift operacional
versão de firmware/modelo/calibrador/threshold/preprocessing; sensor; taxa de abstenção;
latência; memória; falhas de inferência; divergência entre versões; **integridade dos artefatos**
(hash do bundle carregado × manifest).

## 3. Quadro operacional

| Item | Especificação |
|---|---|
| baseline | manifest do bundle promovido (imutável) |
| janelas | deslizante semanal + acumulada mensal |
| frequência | contínua (inferência) + consolidação semanal |
| severidades | INFO / WARN / CRITICAL |
| INFO | desvio dentro dos limites; log |
| WARN | PSI ou distância acima do limiar em 1 janela; revisão na próxima rotina |
| CRITICAL | hash divergente; recall de triagem abaixo do piso; calibração fora dos limites em 2 janelas consecutivas; taxa de abstenção fora da faixa |
| ação CRITICAL | suspensão da saída clínica (fail-closed), rollback para o bundle anterior íntegro, revisão humana obrigatória |
| suspensão | automática em divergência de hash ou integridade; demais casos por decisão humana |

| Parâmetro | Fórmula | Unidade | Intervalo candidato | Método de seleção | Dados permitidos | Risco | Status |
|---|---|---|---|---|---|---|---|
| limiar PSI | Σ (a−e)·ln(a/e) | — | {0,1 WARN / 0,25 CRITICAL} | convenção + piloto | produção | médio | PROPOSED_REQUIRES_RATIFICATION |
| piso recall triagem | gate ratificado | recall | {definir com dados v3} | ratificação clínica | produção rotulada | alto | PROPOSED_REQUIRES_RATIFICATION |
| faixa de abstenção | taxa esperada ± margem | fração | {±50% relativo} | ratificação | produção | médio | PROPOSED_REQUIRES_RATIFICATION |

## 4. Proibições

- Recalibração ou atualização de pesos por drift não supervisionado.
- Incorporação automática de dados pós-produção a qualquer fit.
- Mudança autônoma de threshold, ontologia, preprocessing, população-alvo ou finalidade (07 §6).
- Apresentação de métricas de monitoramento como evidência clínica sem labels revisados.

## 5. Critérios de aceite

1. Tabela de limites por métrica publicada e versionada no bundle.
2. Simulação de drift (shadow) demonstrando transições INFO→WARN→CRITICAL e rollback.
3. Runbook de revisão humana por severidade, com prazos e responsáveis.
