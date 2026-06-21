# Conferência em Engenharia Aplicada — SLHA vs SDD v2.0

**Data:** 2026-06-21  
**Autor:** equipe Project-Lewis  
**Documento de referência:** `docs/SDD_Sistema_Leitura_Hardware_Automatico.md`  
**Versão do SLHA avaliada:** implementação em `src/models/slha/` (branch `feature/slha-hardening`)

## 1. Objetivo

Revisar a implementação do Sistema de Leitura de Hardware Automático (SLHA) sob a ótica da engenharia aplicada, verificando aderência ao SDD, qualidade técnica, trade-offs e riscos operacionais.

## 2. Sumário Executivo

A implementação do SLHA no Project-Lewis está **conforme ao SDD v2.0** para todos os requisitos funcionais e não-funcionais críticos. Os principais pontos fortes são:

- Arquitetura modular e bem isolada (Discovery → Warmup → Decision → Monitor).
- Fallback robusto para CPU-only.
- Integração opt-in que não altera o comportamento padrão dos scripts de treino.
- Logs estruturados em todas as fases quando habilitados.

Os principais gaps são:

- RNF-02 (overhead de warmup < 5%) não validado automaticamente.
- Dependência de `pynvml` opcional limita métricas GPU em alguns ambientes.

**Veredito:** O SLHA está tecnicamente maduro para uso em equipe, com melhorias incrementais documentadas para versões futuras.

## 3. Rastreabilidade de Requisitos

### Requisitos Funcionais

| ID | Requisito | Implementação | Status | Evidência / Observação |
|----|-----------|---------------|--------|------------------------|
| RF-01 | Detectar CPU, cores, frequência, flags SIMD | `src/models/slha/discovery.py:_read_cpu` | ✅ | `psutil.cpu_count`, `/proc/cpuinfo` |
| RF-02 | Detectar GPU, VRAM, compute capability | `src/models/slha/discovery.py:_read_gpu` | ✅ | `tf.config.list_physical_devices` + `get_device_details` |
| RF-03 | Detectar RAM total e disponível | `src/models/slha/discovery.py:_read_ram` | ✅ | `psutil.virtual_memory` |
| RF-04 | Calcular batch size máxima estimada | `src/models/slha/decision.py:decide_training_config` | ✅ | Heurística de memória com fator 0.75 |
| RF-05 | Selecionar accelerator | `src/models/slha/decision.py:decide_training_config` | ✅ | `gpu` se disponível, senão `cpu` |
| RF-06 | Selecionar precision | `src/models/slha/decision.py:_supports_mixed_precision` | ✅ | `mixed_float16` apenas se compute capability ≥ 7 |
| RF-07 | Executar warmup sem modificar código do treino | `src/models/slha/warmup.py:warmup_model` | ✅ | `tf.GradientTape` não treinável, `training=False` |
| RF-08 | Monitorar CPU/RAM em tempo real | `src/models/slha/monitor.py:ResourceMonitor` | ✅ | Callback Keras `on_epoch_end` |
| RF-09 | Emitir alertas quando recursos críticos | `src/models/slha/monitor.py:ResourceMonitor` | ✅ | CPU, RAM e GPU memory > 95% |
| RF-10 | Registrar logs estruturados de todas as fases | `src/models/slha/*.py` + integração | ✅ | JSON/JSONL quando `log_dir` fornecido |
| RF-11 | Fallback robusto para CPU-only | `src/models/slha/discovery.py:_read_gpu` | ✅ | Graceful degradation para `gpu.available=False` |
| RF-12 | Opt-in nos scripts de treino | `pretrain_chapman`, `finetune_mitbih` | ✅ | `use_slha=False` por padrão |

### Requisitos Não-Funcionais

| ID | Requisito | Status | Evidência / Observação |
|----|-----------|--------|------------------------|
| RNF-01 | Discovery < 2s | ✅ | `test_discovery_runs_under_two_seconds` passa |
| RNF-02 | Overhead de Warmup < 5% | ⚠️ | Não validado automaticamente; depende de benchmark manual |
| RNF-03 | Compatibilidade com Python 3.12 | ✅ | Código compatível; ambiente agora restringido a `<3.13` |
| RNF-04 | Zero dependência de interface gráfica | ✅ | Apenas bibliotecas headless |
| RNF-05 | Logs em JSON estruturado | ✅ | `ResourceMonitor` grava JSONL; discovery/warmup/decision JSON |
| RNF-06 | Isolamento de falhas | ✅ | Try/except nos callbacks e discovery |
| RNF-07 | Compatível com WSL2/Linux/Docker | ✅ | `psutil` + `tf.config` multi-plataforma |
| RNF-08 | Sem PyTorch/Lightning, Nsight, etc. | ✅ | Apenas TensorFlow/Keras, psutil, pydantic |

## 4. Análise por Camada

### 4.1 Discovery

**Pontos fortes:**
- Uso de `psutil` e `tf.config` sem dependências extras.
- Leitura de flags SIMD via `/proc/cpuinfo` é leve e headless.
- Fallback para CPU-only é automático e sem exceções.

**Observações técnicas:**
- `tf.config.experimental.get_device_details` pode retornar `memory_limit` igual à memória total alocada pelo TF, não necessariamente VRAM física. Isso é aceitável para heurística de batch size, mas deve ser documentado.
- Em WSL2 sem passthrough GPU, `tf.config.list_physical_devices("GPU")` geralmente retorna lista vazia, ativando o fallback corretamente.

**Decisão de design recomendada:** manter a abordagem; considerar adição de `pynvml` como fallback para VRAM detalhada no futuro (já previsto como opcional no SDD).

### 4.2 Warmup

**Pontos fortes:**
- Execução com `tf.GradientTape(watch_accessed_variables=False)` e `training=False` garante que pesos não sejam alterados.
- Limite de `max_batches=2` e `timeout_seconds=30` protege contra loops longos.

**Observações técnicas:**
- A estimativa de memória por amostra é baseada no delta de RSS do processo. Em ambientes com garbage collection ativo do TF, o pico pode subestimar a memória real das ativações.
- O warmup roda no accelerator selecionado (GPU se disponível), o que é fiel ao uso real.

**Decisão de design recomendada:** manter; adicionar comentário documentando que a estimativa é conservadora e pode ser ajustada via fator de segurança.

### 4.3 Decision

**Pontos fortes:**
- Heurística simples e previsível: 75% da memória disponível, batch size limitado ao reference.
- Seleção de `mixed_float16` apenas quando compute capability ≥ 7.0 evita instabilidade numérica em hardware antigo.

**Observações técnicas:**
- O cálculo `usable_memory = total_memory * 0.75` é global; não considera outros processos concorrentes. Para máquinas compartilhadas, isso pode ser otimista.
- `num_workers` fixado em `min(4, logical_cores)` é razoável para ECG (I/O leve), mas pode ser insuficiente para datasets maiores.

**Decisão de design recomendada:** manter heurística por simplicidade; documentar que o fator 0.75 pode ser exposto como parâmetro futuro.

### 4.4 Monitor

**Pontos fortes:**
- Callback Keras nativo, sem modificar o loop de treino.
- Isolamento de falhas: exceções no monitor não interrompem o treino.
- Alertas de CPU, RAM e GPU memory cobrem os recursos críticos.

**Observações técnicas:**
- `cpu_percent` é normalizado para [0, 100], o que mascara uso multi-core acima de 100% em processos multiprocessados. Para o SLHA (single-process TF), isso é aceitável.
- GPU utilization requer `pynvml`; sem ele, apenas memória total é reportada.

**Decisão de design recomendada:** manter; considerar log de `cpu_percent` bruto em modo debug no futuro.

### 4.5 Integração com Scripts de Treino

**Pontos fortes:**
- Opt-in via `use_slha=True` preserva comportamento padrão.
- Logs persistidos em `experiment_dir/slha/` permitem auditoria por experimento.

**Observações técnicas:**
- O sample usado no warmup (`X_train[:8]` / `next(gen)[:8]`) assume que as primeiras amostras são representativas. Para datasets muito desbalanceados, isso pode afetar a estimativa de memória.
- O `ResourceMonitor` é adicionado sem `log_path` se `use_slha=False`, o que é correto.

**Decisão de design recomendada:** manter; documentar que amostras devem ser representativas.

## 5. Decisões de Design e Trade-offs

| Decisão | Alternativa não escolhida | Trade-off | Avaliação |
|---------|---------------------------|-----------|-----------|
| Heurística simples de batch size | Grid search de batch size | Menor custo computacional, mas menos ótimo | ✅ Adequado para edge/embedded |
| `mixed_float16` apenas CC ≥ 7.0 | Sempre `mixed_float16` | Estabilidade numérica vs velocidade | ✅ Conservador e seguro |
| Warmup com 2 batches | Perfil completo de época | Baixo overhead, mas estimativa aproximada | ✅ Compatível com RNF-02 |
| Logs JSON/JSONL opcionais | Logging estruturado obrigatório | Flexibilidade vs auditoria sempre ativa | ✅ Opt-in alinhado ao RF-12 |
| `pynvml` opcional | Dependência obrigatória | Menos setup em CPU-only vs métricas GPU pobres | ✅ Alinhado ao fallback CPU-only |

## 6. Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação atual | Recomendação |
|-------|---------------|---------|-----------------|--------------|
| GPU indisponível | Alta em laptops/CPU-only | Média | Fallback para `cpu`/`float32` | ✅ Adequada |
| Estimativa de memória otimista | Média | Média | Fator de segurança 0.75 | Considerar fator configurável |
| Falha no SLHA quebra treino | Baixa | Alto | Try/except no monitor e discovery | ✅ Adequada |
| Ambiente Python 3.13+ | Média | Alto | Restrição em `pyproject.toml` | ✅ Resolvido na frente de onboarding |
| Overhead de warmup > 5% | Baixa | Média | Timeout 30s, max 2 batches | Medir em benchmark futuro |

## 7. Recomendações e Ações

### Ações imediatas (alto retorno)
1. **Documentar o fator de segurança de memória** em `docs/Camada-04-Modelagem-v1.1.md` ou SDD.
2. **Adicionar benchmark de overhead de warmup** para validar RNF-02.
3. **Garantir que CI execute testes SLHA** em ambiente Python 3.12.

### Ações futuras (médio retorno)
4. Expor `memory_safety_factor` como parâmetro de `decide_training_config`.
5. Adicionar testes de stress para monitor em alta carga de CPU/RAM.
6. Avaliar `pynvml` como dependência dev opcional para métricas GPU completas.

### Não fazer
- Não introduzir PyTorch Lightning, Nsight, DLProf, LOTUS ou JAX (proibido pelo RNF-08).
- Não tornar SLHA obrigatório (violaria RF-12).
