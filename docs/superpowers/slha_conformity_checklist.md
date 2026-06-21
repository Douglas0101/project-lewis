# SLHA Conformity Checklist

Mapeamento dos requisitos do `docs/SDD_Sistema_Leitura_Hardware_Automatico.md` para os pontos de implementação reais do módulo SLHA.

---

## RF — Requisitos Funcionais

| ID | Requisito | Implementação | Status |
|----|-----------|---------------|--------|
| RF-01 | Detectar CPU, cores, frequência e flags SIMD | `src/models/slha/discovery.py:_read_cpu` (invocado por `discover_hardware`) | ✅ |
| RF-02 | Detectar GPU(s), VRAM e compute capability via TensorFlow | `src/models/slha/discovery.py:_read_gpu` (`tf.config.list_physical_devices`, `tf.config.experimental.get_device_details`) | ✅ |
| RF-03 | Detectar RAM total e disponível | `src/models/slha/discovery.py:_read_ram` (`psutil.virtual_memory`) | ✅ |
| RF-04 | Calcular batch size máxima estimada | `src/models/slha/decision.py:decide_training_config` | ✅ |
| RF-05 | Selecionar accelerator (`cpu`/`gpu`) | `src/models/slha/decision.py:decide_training_config` | ✅ |
| RF-05 / QG 4.2 | Persistência JSONL do monitor | `ResourceMonitor` grava JSONL quando `log_path` é fornecido. `pretrain_chapman` e `finetune_mitbih` passam `log_path=experiment_dir / "slha" / "resource_logs.jsonl"` quando `use_slha=True`. | ✅ |
| RF-06 | Selecionar precision (`float32`/`mixed_float16`) | `src/models/slha/decision.py:decide_training_config` + `src/models/slha/decision.py:_supports_mixed_precision` | ✅ |
| RF-07 | Executar warmup sem modificar código do treino | `src/models/slha/warmup.py:warmup_model` | ✅ |
| RF-08 | Monitorar CPU/RAM durante o treino em tempo real | `src/models/slha/monitor.py:ResourceMonitor.on_epoch_end` + `ResourceMonitor._build_log` | ✅ |
| RF-09 | Emitir alertas quando recursos estiverem críticos | `ResourceMonitor._build_log` emite alertas para CPU (`cpu_percent`), RAM do sistema (`ram_percent`) e, quando `pynvml` está disponível, memória GPU > 95%. | ✅ |
| RF-10 | Registrar logs estruturados de todas as fases | Todas as fases persistem JSON/JSONL quando `log_dir` é fornecido: `hardware_specs.json`, `warmup_result.json`, `training_config.json` via `auto_configure_training`, e `resource_logs.jsonl` via `ResourceMonitor`. | ✅ |
| RF-11 | Fallback robusto para CPU-only | `src/models/slha/discovery.py:_read_gpu` + `src/models/slha/decision.py:decide_training_config` | ✅ |
| RF-12 | Ser opt-in (não alterar comportamento padrão) | `src/models/pretrain_chapman.py:pretrain_chapman(use_slha=False)` + `src/models/finetune_mitbih.py:finetune_mitbih(use_slha=False)` | ✅ |

---

## RNF — Requisitos Não-Funcionais

| ID | Requisito | Implementação / Evidência | Status |
|----|-----------|---------------------------|--------|
| RNF-01 | Discovery < 2 segundos | `tests/test_slha_discovery.py:test_discovery_runs_under_two_seconds` | ✅ |
| RNF-02 | Overhead de Warmup < 5% do tempo de treino | Não medido automaticamente; depende de benchmark manual | ⚠️ |
| RNF-03 | Compatibilidade com Python 3.12 | Código usa anotações modernas (PEP 604 `\|`, `from __future__ import annotations`) compatíveis com Python 3.10+, mas o `.venv` atual executa Python 3.13.9, fora da stack aprovada 3.12.x do Project-Lewis. | ⚠️ |
| RNF-04 | Zero dependência de interface gráfica | Apenas `psutil`, `tensorflow` e bibliotecas padrão; sem GUI | ✅ |
| RNF-05 | Logs em JSON estruturado | `src/models/slha/monitor.py:ResourceMonitor.on_epoch_end` grava `model_dump_json()` em `.jsonl` | ✅ |
| RNF-06 | Isolamento de falhas (falha no SLHA não quebra treino) | Try/except em `ResourceMonitor.on_epoch_end`, `discover_hardware` e `warmup_model` | ✅ |
| RNF-07 | Compatível com WSL2, Linux bare-metal e Docker | Uso de `psutil` + `tf.config` (multi-plataforma); sem testes específicos por SO | ⚠️ |
| RNF-08 | Não adicionar PyTorch/Lightning, Nsight, DLProf, LOTUS, JAX, structlog | Nenhum import proibido encontrado no módulo SLHA | ✅ |

---

## Quality Gates SLHA

### Fase 1 — Discovery

| # | Verificação | Implementação | Status |
|---|-------------|---------------|--------|
| 1.1 | Coleta de CPU | `src/models/slha/discovery.py:_read_cpu` | ✅ |
| 1.2 | Coleta de GPU | `src/models/slha/discovery.py:_read_gpu` | ✅ |
| 1.3 | Coleta de RAM | `src/models/slha/discovery.py:_read_ram` | ✅ |
| 1.4 | Fallback CPU-only | `src/models/slha/discovery.py:_read_gpu` + `tests/test_slha_discovery.py:test_cpu_only_fallback_never_raises` | ✅ |
| 1.5 | Performance < 2s | `tests/test_slha_discovery.py:test_discovery_runs_under_two_seconds` | ✅ |

### Fase 2 — Warmup

| # | Verificação | Implementação | Status |
|---|-------------|---------------|--------|
| 2.1 | Warmup sem erro | `src/models/slha/warmup.py:warmup_model` + `tests/test_slha_warmup.py:test_warmup_returns_memory_estimate` | ✅ |
| 2.2 | Sem alteração de pesos | Garantido por `training=False` + `tf.stop_gradient`; coberto por `tests/test_slha_warmup.py:test_warmup_does_not_change_model_weights` | ✅ |
| 2.3 | Estimativa de memória ≥ 0 | `tests/test_slha_warmup.py:test_warmup_returns_memory_estimate` | ✅ |

### Fase 3 — Decision

| # | Verificação | Implementação | Status |
|---|-------------|---------------|--------|
| 3.1 | Consistência com specs | `src/models/slha/schemas.py:TrainingConfig.devices_consistent` (validador Pydantic) | ✅ |
| 3.2 | Batch size ≥ 1 | `tests/test_slha_decision.py:test_batch_size_never_below_one` | ✅ |
| 3.3 | Precision válida | `src/models/slha/decision.py:_supports_mixed_precision` + `tests/test_slha_decision.py:test_gpu_config_uses_mixed_precision_only_when_available` | ✅ |

### Fase 4 — Monitor

| # | Verificação | Implementação | Status |
|---|-------------|---------------|--------|
| 4.1 | Monitoramento ativo por epoch | `src/models/slha/monitor.py:ResourceMonitor.on_epoch_end` + `tests/test_slha_monitor.py:test_monitor_writes_resource_logs` | ✅ |
| 4.2 | Persistência de logs em disco | `ResourceMonitor` grava JSONL quando `log_path` é fornecido; os scripts de treino ativam a persistência em `experiment_dir / "slha" / "resource_logs.jsonl"` quando `use_slha=True`. Coberto por `tests/test_slha_monitor.py:test_monitor_writes_resource_logs`. | ✅ |
| 4.3 | Graceful degradation | Try/except em `on_epoch_end`; coberto por `tests/test_slha_monitor.py:test_monitor_failure_does_not_stop_training` | ✅ |

---

## Integration — Pontos de Integração com o Circuito de Treinamento

| Script | Função | Uso do SLHA |
|--------|--------|-------------|
| `src/models/pretrain_chapman.py` | `pretrain_chapman` | `use_slha=False` por padrão; quando `True`, chama `src/models/slha/__init__.py:auto_configure_training` e anexa `ResourceMonitor` aos callbacks. |
| `src/models/finetune_mitbih.py` | `finetune_mitbih` | Mesmo padrão opt-in: `use_slha=False` padrão, `auto_configure_training` + `ResourceMonitor` quando `True`. |
| `src/models/slha/__init__.py` | `auto_configure_training` | Orquestra `discover_hardware` → `warmup_model` → `decide_training_config`. Cobertura de integração em `tests/test_slha_integration.py` e `tests/test_slha_training_integration.py`. |

---

## Gaps Conhecidos

1. **RNF-02 não validado automaticamente**: o overhead de warmup em relação ao treino real ainda requer benchmark manual.
2. **RNF-03 / ambiente**: o `.venv` atual está rodando Python 3.13.9, enquanto a stack aprovada do Project-Lewis fixa Python 3.12.x.
3. **RNF-07 sem testes específicos por SO**: o código é multi-plataforma (`psutil` + `tf.config`), mas não há testes automatizados para WSL2, Linux bare-metal ou Docker.

---

*Gerado em conformidade com o SDD `docs/SDD_Sistema_Leitura_Hardware_Automatico.md` v2.0.*
