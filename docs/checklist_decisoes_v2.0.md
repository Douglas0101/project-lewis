# Checklist de Decisões v2.0

**Documento:** `checklist_decisoes_v2.0.md`  
**Versão:** 2.0  
**Data:** 2026-06-30  
**Autor:** Douglas Souza  
**Status:** Aprovado — Fase 5 do plano v2.0  
**Referências:**
- `docs/UNIFIED_DOCUMENT_v2.0.md` (Seção S11)
- `docs/SDD_Project-Lewis_v3.md`
- `AGENTS.md`
- Artefatos v2.0: `models/stage1_float32_v2.0.keras`, `models/stage2_float32_v2.0.keras`, `models/quantized/stage1_int8_v2.0.tflite`, `models/quantized/stage2_int8_v2.0.tflite`

---

## Resumo

Este checklist consolida as decisões arquiteturais do pipeline MIT-BIH/AAMI v2.0, conforme levantado na Seção S11 do `UNIFIED_DOCUMENT_v2.0.md`. Cada item indica se a decisão foi efetivada (`[x]`) ou permanece pendente/deferida (`[ ]`), com uma breve justificativa baseada nos artefatos existentes e nos testes já executados.

---

## Decisão 1: Excluir classe Q do escopo

- [x] **Aprovado e implementado.**

**Justificativa:** A classe Q (batimentos paced/não classificáveis) apresenta apenas 8 amostras no treino DS1 do MIT-BIH, resultando em bound de generalização `E_gen ≈ 157` para o modelo de ~19.933 parâmetros — estatisticamente não aprendível. A exclusão está refletida em:
- Redefinição dos thresholds QG5' para 4 classes (N, S, V, F) no `UNIFIED_DOCUMENT_v2.0.md`;
- Datasets de fine-tuning gerados sem a classe Q;
- Modelos v2.0 (`stage1_float32_v2.0.keras`, `stage2_float32_v2.0.keras`) treinados para as classes remanescentes;
- Regra de ouro no `AGENTS.md`: "Classe Q (paced/unclassifiable) excluída da classificação final a partir de v2.0 — tratada como 'Anormal' no Estágio 1".

---

## Decisão 2: Revisar thresholds QG5

- [x] **Aprovado e implementado.**

**Justificativa:** A meta v1.1 (`F1-macro > 0,85` em 5 classes AAMI com inter-patient split) é matematicamente inatingível para um modelo < 20k parâmetros. Os thresholds foram revisados para o QG5' v2.0:

| Métrica | Threshold v1.1 | Threshold v2.0 |
|---------|---------------|----------------|
| Accuracy | > 0,93 | > 0,88 |
| F1-macro | > 0,85 | > 0,55 |
| F1 (N) | — | > 0,90 |
| F1 (S) | — | > 0,45 |
| F1 (V) | — | > 0,70 |
| F1 (F) | — | > 0,30 |

Os thresholds estão codificados nos testes (`tests/test_two_stage_qg5.py`) e validados contra os modelos v2.0. O relatório `reports/two_stage_evaluation_v2.0.json` indica que Estágio 1 (`F1-macro = 0,5927`) e Estágio 2 (`F1-macro = 0,5185`) atendem individualmente os novos critérios.

---

## Decisão 3: Implementar pipeline de duas etapas

- [x] **Aprovado e implementado.**

**Justificativa:** O pipeline divide o problema em (i) detecção binária de anormalidade e (ii) subtipificação condicional (S vs. V vs. F), reduzindo a complexidade por estágio e permitindo otimizações dedicadas:
- `src/inference/two_stage_pipeline.py`: implementação Python do pipeline;
- `src/inference/quantized_runner.py`: execução dos modelos quantizados;
- Modelos treinados e versionados: `models/stage1_float32_v2.0.keras`, `models/stage2_float32_v2.0.keras`;
- Modelos quantizados: `models/quantized/stage1_int8_v2.0.tflite`, `models/quantized/stage2_int8_v2.0.tflite`;
- Scalers dedicados por estágio: `models/input_scaler_stage1_v2.0.pkl`, `models/input_scaler_stage2_v2.0.pkl`.

> **Nota:** A integração completa dos dois estágios ainda apresenta F1-macro inferior ao esperado no caminho integrado (`reports/two_stage_evaluation_v2.0.json` → `integrated.F1_macro = 0,3162`). O ajuste de thresholds e calibração entre estágios está em andamento, mas a decisão arquitetural de adotar duas etapas está efetivada.

---

## Decisão 4: Definir hardware alvo (STM32F4 vs STM32L4)

- [x] **Aprovado: STM32F4 (STM32F407VG).**

**Justificativa:** Embora o `UNIFIED_DOCUMENT_v2.0.md` tenha levantado STM32L4 e ESP32-S3 como alternativas, o `AGENTS.md` — documento de governaça do projeto — estabelece o hardware alvo como **STM32F407VG** (Cortex-M4F, 168 MHz, 192 KB SRAM, 1 MB Flash), com aceleração via CMSIS-DSP/CMSIS-NN. A simulação no Renode 1.15.3 usa a placa STM32F4 Discovery, conforme evidenciado em `reports/firmware_simulation_report.json`:
- `firmware_bin`: `firmware/build/stm32f4/lewis.bin`;
- Latência por batimento: ~16 ms;
- Arena usada: ~19.972 bytes.

A decisão de manter STM32F4 como plataforma de referência garante compatibilidade com a pilha TFLM, CMSIS-NN e os quality gates QG7–QG13 já instrumentados.

---

## Decisão 5: Adaptive inference skipping

- [x] **Aprovado e implementado.**

**Justificativa:** A técnica permite pular inferências quando o ritmo cardíaco está estável, economizando energia e reduzindo wear do MCU:
- Implementação em C: `firmware/src/dsp/adaptive_skipping.c`, `firmware/src/dsp/adaptive_skipping.h`;
- Teste de simulação: `tests/test_adaptive_skipping_sim.py`;
- A lógica monitora a variação do intervalo RR e mantém a última classe conhecida durante os ciclos estáveis, com verificação periódica para não perder transientes.

A meta de economia de ~70% em ritmo sinusal estável é projetada, mas ainda depende de benchmark energético completo (QG19).

---

## Decisão 6: Filtro digital em firmware

- [x] **Aprovado e implementado.**

**Justificativa:** Os filtros digitais em C garantem equivalência funcional com o pré-processamento Python e mitigam interferência eletromagnética:
- `firmware/src/dsp/ecg_filter.c`, `firmware/src/dsp/ecg_filter.h`;
- Filtros: passa-alta 0,5 Hz, passa-baixa 40 Hz, notch 50/60 Hz;
- Testes de equivalência: `tests/test_firmware_filters_python.py`, `tests/test_dsp_filters.py`, `tests/test_dsp_fidelity.py`;
- Quality gate QG16: RMSE < 1e-6 entre implementações C e Python.

---

## Decisão 7: Fallback para features morfológicas

- [ ] **Deferido / não implementado como fallback ativo.**

**Justificativa:** O `UNIFIED_DOCUMENT_v2.0.md` previa a avaliação de features morfológicas + MLP como fallback caso o Estágio 2 não atingisse `F1-macro > 0,45`. Atualmente:
- O Estágio 2 isolado atinge `F1-macro = 0,5185` (`reports/two_stage_evaluation_v2.0.json`), superando o threshold mínimo;
- O extractor morfológico existe em `src/features/`, mas não foi integrado como fallback do pipeline v2.0;
- A decisão permanece aberta para a Fase 6 caso a integração completa (Estágio 1 → Estágio 2) não alcance estabilidade.

---

## Decisão 8: Pruning estruturado + QAT/PTQ

- [x] **Aprovado e implementado com fallback PTQ INT8.**

**Justificativa:** A compactação dos modelos é mandatória para o deployment em TFLM:
- Implementação: `src/models/pruning_qat.py`, `scripts/apply_pruning_qat.py`;
- Testes: `tests/test_pruning_qat.py`, `tests/test_quantization_degradation.py`;
- Modelos quantizados gerados: `models/quantized/stage1_int8_v2.0.tflite` (~54,36 KB) e `models/quantized/stage2_int8_v2.0.tflite` (~54,47 KB);
- Resumo em `models/quantized/quantization_summary_v2.0.json`.

**Observação importante:** O QAT (Quantization-Aware Training) não é suportado pelo ambiente `tf-keras` atual (`tfmot` requer instâncias de `keras.layers.Layer` legadas; o TensorFlow 2.21 com `tf-keras` levanta `ValueError`). O fallback para PTQ INT8 full-integer está automatizado e documentado no ADR `docs/adr_qat_ptq_v2.0.md` (ver Fase 5 / `decision_checklist == adr_qat_ptq`). A degradação observada nos testes de quantização está dentro da tolerância de 2% do QG6.

---

## Decisão 9: Callbacks de instrumentação

- [x] **Aprovado e implementado.**

**Justificativa:** Os callbacks fornecem visibilidade da dinâmica de treinamento sem alterar a arquitetura dos modelos:
- `src/callbacks/gradient_monitor.py`: normas de gradiente, razões e detecção de vanishing/exploding;
- `src/callbacks/calibration_monitor.py`: ECE, MCE, Brier Score e reliability diagram;
- `src/callbacks/f1_macro_checkpoint.py`: seleção de melhores pesos por F1-macro;
- Testes: `tests/test_callbacks.py`, `tests/test_callbacks_integration.py`.

Os callbacks são desacopláveis e não adicionam dependências além de TensorFlow, NumPy e SciPy.

---

## Decisão 10: Análise dinâmica pós-treinamento

- [x] **Aprovado e implementada.**

**Justificativa:** A análise correlacional entre gradientes, calibração e métricas F1 permite diagnóstico acionável após cada treinamento:
- Script: `scripts/analyze_training_dynamics.py`;
- Testes: `tests/test_analyze_training_dynamics.py`;
- Saídas geradas: heatmap de correlação, gráfico ECE vs F1-macro, reliability diagram e relatório Markdown (`logs/training_dynamics_analysis.md`).

A ferramenta será executada formalmente assim que o treinamento do Estágio 1 em andamento for concluído; o código e os testes já estão validados.

---

## Consolidação

| # | Decisão | Status | Artefato Principal |
|---|---------|--------|-------------------|
| 1 | Excluir classe Q | [x] | Modelos v2.0, AGENTS.md |
| 2 | Revisar thresholds QG5 | [x] | `UNIFIED_DOCUMENT_v2.0.md`, `tests/test_two_stage_qg5.py` |
| 3 | Pipeline duas etapas | [x] | `src/inference/two_stage_pipeline.py`, modelos v2.0 |
| 4 | Hardware alvo STM32F4 | [x] | `AGENTS.md`, `reports/firmware_simulation_report.json` |
| 5 | Adaptive inference skipping | [x] | `firmware/src/dsp/adaptive_skipping.c/h` |
| 6 | Filtro digital em firmware | [x] | `firmware/src/dsp/ecg_filter.c/h` |
| 7 | Fallback morfológico | [ ] | `src/features/` (não integrado) |
| 8 | Pruning + PTQ INT8 | [x] | `models/quantized/stage1_int8_v2.0.tflite`, `stage2_int8_v2.0.tflite` |
| 9 | Callbacks de instrumentação | [x] | `src/callbacks/` |
| 10 | Análise dinâmica | [x] | `scripts/analyze_training_dynamics.py` |

---

## Pendências e Riscos

1. **Integração do pipeline duas etapas:** o caminho integrado apresenta `F1-macro = 0,3162` no relatório atual, abaixo da meta de 0,55. O ajuste de thresholds entre estágios e a calibração de confiança são os próximos passos.
2. **Fallback morfológico:** permanece como opção de contingência para a Fase 6, caso o ajuste da integração não seja suficiente.
3. **Resultados numéricos do treinamento em andamento:** assim que o Estágio 1 concluir, o relatório `reports/two_stage_evaluation_v2.0.json` deve ser atualizado com as métricas finais.
4. **QAT:** permanece como melhoria futura caso a compatibilidade entre `tfmot` e `tf-keras` seja resolvida; o fallback PTQ INT8 cobre o requisito de deployment.

---

## Histórico de Revisões

| Versão | Data | Autor | Mudanças |
|--------|------|-------|----------|
| 2.0 | 2026-06-30 | Douglas Souza | Checklist consolidado da Fase 5, com base nos artefatos v2.0 e testes executados. |

---

*Documento gerado para registro arquitetural do Project-Lewis. Manter sincronizado com `UNIFIED_DOCUMENT_v2.0.md` e `AGENTS.md`.*
