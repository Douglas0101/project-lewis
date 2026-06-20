# Design — Correções Finais no SDD + Aprimoramento de Harness

**Data:** 2026-06-20  
**Escopo:** `docs/SDD_Project-Lewis_v3.md` + harness de firmware/simulação  
**Autor:** Kimi Code (SDD Agent)  

---

## 1. Contexto e Objetivo

O `docs/SDD_Project-Lewis_v3.md` é o documento mestre de arquitetura para agentes de coding AI. Após uma passada de pente fino com inspeção do projeto real, verificação via WebSearch de fatos técnicos e leitura das camadas de docs, identificaram-se inconsistências entre o SDD e o estado atual do repositório, além de gaps de documentação sobre o harness de testes de firmware/simulação.

**Objetivo desta tarefa:**
1. Corrigir o SDD para que ele reflita fielmente o código, configurações e quality gates do projeto.
2. Adicionar uma seção dedicada ao **Test Harness de Firmware e Simulação**, hoje inexistente (`firmware/tests/` está vazio, exceto `.gitkeep`).
3. Criar/validar os artefatos de suporte mencionados no SDD (`mcp.json`, contextos `.kimi/`, `.opencode/`, `AGENTS.md`, `PLAN.md`).

---

## 2. Problemas Encontrados no SDD v3

### 2.1 Inconsistências de Fase
- **Seção 1.1 (tabela):** C08 e C09 estão listados como `Fase 1`. No restante do documento e na organização do projeto, C08 (Firmware) e C09 (Simulação/Energia) são claramente **Fase 2**.

### 2.2 Inconsistências nas Regras de Ouro
- **Seção 1.3** lista 15 regras corretas (fonte da verdade).
- **Seção 4.1 (Kimi Code context)** diverge nas regras 13–15: inclui restrições de modelo/firmware que deveriam estar nas regras 1–12 ou serem regras à parte, omitindo "Senhas hasheadas" e "LGPD: nenhum PII em logs".
- **Seção 4.2 (OpenCode context)** lista apenas 13 regras, faltando as regras 14 e 15 da seção 1.3.

### 2.3 Inconsistências de Versões e Dependências
- **Seção 1.2 / 4.1 / 4.2:** mencionam TensorFlow `>=2.16,<2.17`, mas `pyproject.toml` usa `tensorflow>=2.21.0`.
- **Seção 1.2 / 4.1:** mencionam `pytest-cov`, `pytest-xdist` e schema tests com `pandera`, mas nenhum consta em `pyproject.toml`.
- **Seção 1.2:** classifica OpenCode como "Open Source", o que é impreciso (é ferramenta da All Hands AI).
- **`pyproject.toml` real:** classificador inclui `Programming Language :: Python :: 3.13`, contradizendo o veto 3.13+ do SDD.

### 2.4 Inconsistências de Estrutura de Diretórios
- **Seção 2** lista arquivos que **não existem** no repositório:
  - `.kimi/sdd-context.md`
  - `.opencode/sdd-context.md`
  - `AGENTS.md`
  - `PLAN.md`
  - `mcp.json`
  - `config/pretrain_v1.0.yaml`
  - `config/power_model_v1.4.yaml` (existe em `firmware/config/`)
  - `src/quantization/export_tflite.py` (o real é `src/models/export_tflm.py`)
  - `firmware/src/ml/inference.c` (o real é `firmware/src/ml/inference.cpp`)
  - `firmware/src/features/features_config.h`
  - `firmware/src/features/normalization_params.h`
  - `firmware/src/features/filter_coeffs_q31.h`
- **Seção 2** omite arquivos/scripts reais:
  - `scripts/generate_filter_coeffs.py`
  - `scripts/run_hard_gates.py`
  - `scripts/install_firmware_tools.sh`
  - `firmware/scripts/install_deps.sh`
  - `firmware/renode/*.robot`, `FidelityKeywords.py`, `dummy_spi_device.py`

### 2.5 Inconsistências de Testes
- **Seção 2** lista `tests/test_integration.py`, mas o arquivo real é `tests/test_pipeline.py`.
- **Seção 2** omite dezenas de testes reais: `test_arena_limits.py`, `test_dsp_*.py`, `test_fault_injection.py`, `test_firmware_qg.py`, `test_native_tflm.py`, `test_renode_runner.py`, `test_r_peak_firmware.py`, `test_tflm_bitexact.py`, `test_watchdog.py`, etc.

### 2.6 Inconsistências de Quality Gates
- **Seção 3.8** lista QG7, QG8, QG9, QG10, QG13, QG16, QG17, QG18 — pulando QG11, QG12, QG14, QG15.
- **Seção 5 (AGENTS.md embarcado)** lista QG0–QG19, mas a tabela também omite QG11, QG12, QG14, QG15.
- `pyproject.toml` define markers `qg11` e `qg12`, confirmando que esses QGs existem no projeto.

### 2.7 Inconsistências de Pipeline e Artefatos de Firmware
- **Seção 3.8** menciona `inference.c/h`; deve ser `inference.cpp/h`.
- **Seção 3.9** não reflete a pluralidade de scripts Renode (`.robot` para fault injection, watchdog, arena 48k, fidelity, shutdown).
- **Seção 3.8 / 3.9** não documentam o harness de testes de firmware.

### 2.8 Configuração MCP
- **Seção 6** descreve `mcp.json`, mas o arquivo não existe na raiz.

---

## 3. Abordagens Consideradas

| Abordagem | Descrição | Prós | Contras | Recomendação |
|-----------|-----------|------|---------|--------------|
| **A — Correção mínima** | Corrigir apenas erros óbvios de digitação e paths. | Rápido, baixo risco. | Não resolve gaps estruturais (harness, MCP, regras divergentes). | Não recomendada |
| **B — Correção + Sincronização completa** | Atualizar SDD para refletir o projeto real (paths, versões, QGs, regras) e criar artefatos faltantes (`mcp.json`, `.kimi/`, `.opencode/`, `AGENTS.md`). | Documento confiável, agentes alinhados. | Requer mais edições. | **Recomendada** |
| **C — Refactor do SDD + harness novo** | Além da correção, adicionar seção dedicada de Test Harness e criar esqueleto de harness em `firmware/tests/`. | Resolve a demanda explícita de "aprimoramento de harness". | Maior escopo, mas justificado. | **Adotar como complemento à B** |

**Decisão:** adotar a abordagem **B + C** — sincronizar o SDD e, na sequência, implementar o harness de firmware/simulação documentado.

---

## 4. Design das Correções no SDD

### 4.1 Fases e Escopo
- Corrigir tabela da seção 1.1: C01–C07 = Fase 1; C08–C09 = Fase 2.

### 4.2 Regras de Ouro
- Padronizar regras 1–15 em **1.3**, **4.1** e **4.2** para serem idênticas.
- Incluir explicitamente:
  - 13. Senhas hasheadas com Argon2id/bcrypt (se houver auth)
  - 14. LGPD: nenhum PII em logs
  - 15. Revisão humana obrigatória para código crítico
- As restrições de modelo/firmware (LSTM, BatchNorm, printf/semihosting) já estão no corpo das camadas e não precisam ocupar as regras 13–15.

### 4.3 Stack Técnica
- Atualizar TensorFlow para `>=2.21.0` (refletir `pyproject.toml`) ou manter `2.16` e corrigir `pyproject.toml`. **Decisão:** o código já roda com 2.21; atualizar SDD.
- Remover `pytest-cov`, `pytest-xdist`, `pandera` do SDD ou adicioná-los ao `pyproject.toml`. **Decisão:** adicionar `pytest-cov` e `pytest-xdist` ao `pyproject.toml` (são úteis); `pandera` será removido do SDD por ora (não está em uso).
- Corrigir classificação do OpenCode.
- Adicionar nota de que `pyproject.toml` não deve declarar Python 3.13.

### 4.4 Estrutura de Diretórios
- Atualizar seção 2 para refletir paths reais.
- Adicionar arquivos faltantes (`mcp.json`, `.kimi/sdd-context.md`, `.opencode/sdd-context.md`, `AGENTS.md`, `PLAN.md`) como artefatos a serem criados/verificados.
- Mover `power_model_v1.4.yaml` para `firmware/config/`.
- Corrigir `src/quantization/export_tflite.py` → `src/models/export_tflm.py`.
- Corrigir `firmware/src/ml/inference.c/h` → `inference.cpp/h`.
- Atualizar lista de testes.

### 4.5 Quality Gates
- Completar tabelas com QG11, QG12, QG14, QG15, QG19.
- Definições sugeridas:
  - **QG11:** Fault injection SPI/UART
  - **QG12:** Limites de arena RAM
  - **QG14:** (reservado — ex: segurança/LGPD no firmware)
  - **QG15:** (reservado — ex: OTA/update seguro)
  - **QG19:** Consumo energético Renode

### 4.6 Configuração MCP
- Criar `mcp.json` na raiz com os servidores descritos na seção 6.
- Criar `.kimi/sdd-context.md` e `.opencode/sdd-context.md` com o contexto do SDD.

---

## 5. Design do Aprimoramento de Harness

### 5.1 Objetivo do Harness
Fornecer uma infraestrutura de testes para o firmware C/C++ e para a simulação Renode, permitindo:
- Compilar e rodar testes unitários no host (`native`) e no target emulado (`renode`).
- Injetar sinais de ECG sintéticos/dormente no `adc_stub.c`.
- Comparar saídas do firmware com referências Python (bit-exatidão, fidelidade DSP).
- Gerar relatório JSON de harness (`firmware/test_harness_report.json`).

### 5.2 Componentes

| Componente | Path | Função |
|------------|------|--------|
| Harness runner | `firmware/scripts/run_harness.py` | Orquestra build native/renode e coleta resultados |
| Testes C/C++ | `firmware/tests/test_dsp.c`, `test_inference.cpp`, `test_r_peak.c` | Unitários de DSP, TFLM e detector R-peak |
| Fixtures | `firmware/tests/fixtures/` | Arrays de entrada e expected outputs gerados por Python |
| Makefile target | `firmware/Makefile` | `harness`, `harness-native`, `harness-renode` |
| CI step | `.github/workflows/quality-gates.yml` | Executar harness após QG7 |

### 5.3 Interfaces
- `test_harness_init()`: inicializa HAL, UART stub e watchdog mock.
- `test_harness_run_all()`: executa suite e imprime resultados no formato `HARNESS <suite> <test> <PASS|FAIL> <detail>`.
- `assert_int8_equal()`, `assert_float_close()`: helpers para comparação com tolerância.

### 5.4 Acceptance Criteria
- [ ] `make -C firmware harness-native` compila e executa sem erro.
- [ ] `make -C firmware harness-renode` executa no Renode headless.
- [ ] Saída UART parseável por `firmware/scripts/run_harness.py`.
- [ ] Relatório `firmware/test_harness_report.json` gerado.
- [ ] ≥ 3 suites cobertas: DSP filters, TFLM inference, R-peak detector.
- [ ] Nenhum regression nos QGs existentes (QG7–QG18).

---

## 6. Plano de Implementação

1. **Correções no SDD**
   1.1. Fases (C08/C09 → Fase 2)  
   1.2. Regras de ouro (sincronizar 1.3, 4.1, 4.2)  
   1.3. Stack técnica (TensorFlow, pytest-cov, pytest-xdist, OpenCode)  
   1.4. Estrutura de diretórios  
   1.5. Lista de testes  
   1.6. Quality gates completos  
   1.7. Inference.cpp e paths de scripts/config  

2. **Artefatos de Suporte**
   2.1. Criar `mcp.json`  
   2.2. Criar `.kimi/sdd-context.md` e `.opencode/sdd-context.md`  
   2.3. Criar `AGENTS.md` e `PLAN.md` (ou stubs)  
   2.4. Atualizar `pyproject.toml` (pytest-cov, pytest-xdist, remover classificador 3.13)  

3. **Harness de Firmware**
   3.1. Criar `firmware/scripts/run_harness.py`  
   3.2. Criar `firmware/tests/test_dsp.c`, `test_inference.cpp`, `test_r_peak.c`  
   3.3. Adicionar targets no `firmware/Makefile`  
   3.4. Gerar fixtures iniciais via scripts Python  
   3.5. Validar com `make harness-native`  

---

## 7. Riscos e Mitigações

| Risco | Mitigação |
|-------|-----------|
| Quebra de build ao alterar `pyproject.toml` | Manter `uv.lock` consistente; rodar `uv sync` após mudança |
| Harness native depender de `libm` no bare-metal | Usar lookup tables e stubs para funções trigonométricas |
| Divergência futura entre SDD e código | Adicionar checklist de sincronização no `PLAN.md` |

---

## 8. Notas de Validação Externa

- STM32F407VG: 168 MHz, 1024 kB Flash, 192 kB SRAM confirmados via WebSearch.
- AAMI EC57: tolerância de 150 ms para detecção de QRS confirmada via WebSearch.
- CMSIS-NN/TFLM: Conv2D, DepthwiseConv2D, FullyConnected, Softmax, Pooling otimizados confirmados via WebSearch.
