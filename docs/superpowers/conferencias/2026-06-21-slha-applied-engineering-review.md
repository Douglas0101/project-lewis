# Conferência em Engenharia Aplicada — SLHA vs SDD v2.0

**Data:** 2026-06-21  
**Autor:** equipe Project-Lewis  
**Documento de referência:** `docs/SDD_Sistema_Leitura_Hardware_Automatico.md`  
**Versão do SLHA avaliada:** implementação em `src/models/slha/` (branch `feature/slha-hardening`)

## 1. Objetivo

Revisar a implementação do Sistema de Leitura de Hardware Automático (SLHA) sob a ótica da engenharia aplicada, verificando aderência ao SDD, qualidade técnica, trade-offs e riscos operacionais.

## 2. Sumário Executivo

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

## 5. Decisões de Design e Trade-offs

## 6. Riscos e Mitigações

## 7. Recomendações e Ações
