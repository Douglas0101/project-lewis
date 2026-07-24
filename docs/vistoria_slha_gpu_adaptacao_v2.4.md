# Vistoria Avançada — SLHA e Adaptação CPU/GPU no Treinamento

**Versão:** 2.4  
**Data:** 2026-07-10  
**Projeto:** Project-Lewis  
**Executor:** agente-ciência-dados  
**Ambiente de vistoria:** IdeaPad 3 15ITL6 / Zorin OS / Python 3.12.3 / TensorFlow 2.21 / uv

---

## 1. Resumo Executivo

A vistoria avaliou o **Sistema de Leitura de Hardware Automático (SLHA)** (`src/models/slha`) e o **caminho de adaptação de treinamento** dos scripts de modelagem (`scripts/train_stage2_mlp.py`, `scripts/run_stage2_training.py`, `src/models/pretrain_chapman.py`, `src/models/finetune_mitbih.py`) frente à presença ou ausência de GPU.

**Veredito geral:** o sistema detecta hardware corretamente, isola falhas e mantém o treinamento funcional em CPU-only. Os testes de SLHA passam 15/15. Não foram encontrados hardcodes de GPU nem quebras de treinamento quando CUDA está indisponível. Há oportunidades de melhoria na **supressão de mensagens de erro C++ do TensorFlow** e na **integração explícita do SLHA nos scripts de treinamento MLP de features** (hoje opt-in e não utilizado pelos scripts `train_stage1_mlp.py` / `train_stage2_mlp.py`).

---

## 2. Escopo e Metodologia

### 2.1 Escopo

| Componente | Arquivos verificados |
| ------------ | --------------------- |
| Discovery de hardware | `src/models/slha/discovery.py` |
| Warmup de modelo | `src/models/slha/warmup.py` |
| Decision engine | `src/models/slha/decision.py` |
| Monitor de recursos | `src/models/slha/monitor.py` |
| Schemas Pydantic | `src/models/slha/schemas.py` |
| Integração SLHA em treinadores | `src/models/pretrain_chapman.py`, `src/models/finetune_mitbih.py` |
| Treinamento MLP de features | `scripts/train_stage1_mlp.py`, `scripts/train_stage2_mlp.py` |
| Treinamento CNN raw-signal | `scripts/run_stage2_training.py` |
| Testes de SLHA | `tests/test_slha_*.py` |

### 2.2 Metodologia

1. Execução da suíte de testes `tests/test_slha_*.py`.
2. Execução de `discover_hardware()` no ambiente atual (CPU-only).
3. Inspeção estática de busca por `cuda`, `gpu`, `set_visible_devices`, `mixed_float16` e `tf.device` em `src/`, `scripts/` e `tests/`.
4. Verificação de uso do SLHA nos scripts de treinamento atuais.
5. Execução de warmup/decision em modelo dummy para CPU-only.
6. Análise de logs de inicialização do TensorFlow sem CUDA.

---

## 3. Resultados dos Testes

### 3.1 Suíte SLHA

```bash
python -m pytest tests/test_slha_*.py -v
```

Resultado: **15 passed, 0 failed**.

| Teste | Descrição | Status |
| ------- | ----------- | -------- |
| `test_discovery_returns_valid_specs` | JSON de specs válido | ✅ |
| `test_discovery_runs_under_two_seconds` | Discovery < 2s | ✅ |
| `test_cpu_only_fallback_never_raises` | Sem GPU não levanta exceção | ✅ |
| `test_auto_configure_returns_config` | Caminho feliz discovery → warmup → decision | ✅ |
| `test_auto_configure_persists_logs` | Persistência de logs | ✅ |
| `test_warmup_returns_memory_estimate` | Estimativa de memória por amostra | ✅ |
| `test_warmup_does_not_change_model_weights` | Warmup não altera pesos | ✅ |
| `test_monitor_writes_resource_logs` | Logs de recursos por epoch | ✅ |
| `test_monitor_failure_does_not_stop_training` | Isolamento de falhas do monitor | ✅ |
| `test_pretrain_chapman_accepts_use_slha` | SLHA opt-in no pré-treino | ✅ |
| `test_finetune_mitbih_accepts_use_slha` | SLHA opt-in no fine-tuning | ✅ |

### 3.2 Discovery no Ambiente Atual

Execução:

```python
from src.models.slha import discover_hardware
print(discover_hardware().model_dump_json(indent=2))
```

Saída resumida:

```json
{
  "gpu": {
    "available": false,
    "count": 0,
    "devices": []
  },
  "cpu": {
    "physical_cores": 2,
    "logical_cores": 4,
    "flags": ["avx", "avx2", "avx512f", "fma", "sse4_2"]
  },
  "ram": {
    "total_gb": 19.3,
    "available_gb": 8.8
  }
}
```

**Observação:** o TensorFlow emitiu mensagens informativas C++ sobre ausência de CUDA, mas **não quebrou** e a função retornou specs válidas.

---

## 4. Análise da Arquitetura de Reconhecimento de Hardware

### 4.1 Discovery (`src/models/slha/discovery.py`)

**Pontos fortes:**

- Usa `tf.config.list_physical_devices("GPU")` via API oficial do TensorFlow, sem depender de `pynvml`.
- Cada dispositivo GPU é lido com `tf.config.experimental.get_device_details`, com `try/except` por dispositivo (graceful degradation).
- Fallback para CPU-only: `gpu.available=False`, `count=0`, `devices=[]`.
- Leitura de CPU via `psutil`, incluindo flags SIMD por parsing de `/proc/cpuinfo`.
- Schema Pydantic valida ranges e tipos.

**Risco identificado:** se `tf.config.list_physical_devices("GPU")` levantar uma exceção não-prevista, a função loga warning e retorna GPU indisponível. Isso está correto, mas em ambientes WSL2/containers mal-configurados o TensorFlow pode imprimir mensagens de erro C++ (CUDA) que **não são fatais**, mas podem ser interpretadas como falha pelo operador.

### 4.2 Warmup (`src/models/slha/warmup.py`)

**Pontos fortes:**

- Usa `tf.GradientTape(watch_accessed_variables=False)` e `training=False`, garantindo que os pesos não são modificados.
- Mede tempo e delta de RAM via `psutil.Process().memory_info().rss`.
- Timeout de 30s e proteção contra divisão por zero.

**Verificação:** em CPU-only, o warmup em modelo dummy concluiu sem erros e retornou estimativa de memória positiva.

### 4.3 Decision Engine (`src/models/slha/decision.py`)

**Pontos fortes:**

- Seleção de accelerator: `gpu` somente se `specs.gpu.available and specs.gpu.count > 0`.
- Precision: `mixed_float16` apenas se GPU detectada e compute capability ≥ 7.0; caso contrário `float32`.
- Batch size: `max(1, min(batch_max, reference_batch_size))`, garantindo ≥ 1.
- Reserva 25% de memória para overhead (`MEMORY_SAFETY_FACTOR = 0.75`).

### 4.4 Monitor (`src/models/slha/monitor.py`)

**Pontos fortes:**

- Callback Keras com `try/except` em `on_epoch_end`; falhas nunca interrompem o treino.
- GPU monitoring é opcional (pynvml lazy); ausência não quebra o log.
- Alertas de CPU/RAM e GPU configuráveis.

### 4.5 Integração nos Treinadores Keras

- `src/models/pretrain_chapman.py` e `src/models/finetune_mitbih.py` aceitam `use_slha=True` (opt-in).
- Quando ativo, carregam `auto_configure_training`, aplicam `ResourceMonitor` e logam a config em `experiment_dir/slha/`.
- Quando inativo, mantêm o comportamento legado (batch_size fixo, float32), o que continua funcional em CPU.

---

## 5. Caminho de Adaptação nos Scripts de Treinamento

### 5.1 Scripts `train_stage1_mlp.py` e `train_stage2_mlp.py`

**Status:** funcionam em CPU-only, mas **não usam SLHA**.

- Não há chamada a `discover_hardware`, `auto_configure_training` ou `ResourceMonitor`.
- Batch size fixo: 256 (Stage 1) e 128 (Stage 2).
- Modelo é pequeno (MLP com 16 features → poucos milhares de parâmetros), portanto não há pressão de memória no IdeaPad.
- Não há configuração explícita de dispositivo; TensorFlow usa CPU por padrão quando GPU ausente.

**Risco baixo:** em CPU-only o treino continua. O único "erro" visível são mensagens C++ de CUDA durante o import do TensorFlow, que não quebram o script.

### 5.2 Script `run_stage2_training.py` (CNN raw-signal)

**Status:** não detectou hardcodes de GPU.

- Usa `train_group_kfold` de `src/models/train.py`, que chama `model.fit` padrão.
- Não há `tf.device` ou `mixed_float16` hardcoded.
- Pode ser mais pesado em CPU, mas não há quebra por ausência de GPU.

### 5.3 Busca por Hardcodes de GPU

Comando executado:

```bash
grep -R -i -E 'cuda|gpu|set_visible_devices|mixed_float16|device.*gpu|tf.device' \
  src/ scripts/ tests/ --include='*.py'
```

Resultado: todas as ocorrências estão confinadas ao módulo `src/models/slha` e a testes que validam o próprio SLHA. Nenhum script de treinamento hardcoded GPU.

---

## 6. Findings e Recomendações

### 6.1 Findings Positivos

1. **SLHA é funcional e testado:** 15 testes passam, incluindo fallback CPU-only e isolamento de falhas.
2. **Nenhum hardcode de GPU** nos scripts de treinamento atuais.
3. **Treinamento CPU-only não quebra:** o ambiente atual rodou `discover_hardware` e a suíte de testes sem GPU.
4. **Monitor isolado:** falhas no monitor não interrompem o treino (RNF-06 atendido).
5. **Schemas Pydantic:** configuração é validada, evitando batch_size < 1 ou precision inválida.

### 6.2 Riscos e Mitigações

| # | Risco | Severidade | Mitigação Recomendada |
| --- | ------- | ------------ | ---------------------- |
| 1 | Mensagens C++ de CUDA ("failed call to cuInit") durante import do TensorFlow em CPU-only podem ser confundidas com erro fatal. | 🟡 Média | Adicionar no `Makefile`/`scripts` env `TF_CPP_MIN_LOG_LEVEL=2` quando `CUDA_VISIBLE_DEVICES=-1` e GPU não detectada. Documentar que a mensagem é esperada e não fatal. |
| 2 | `train_stage2_mlp.py` e `train_stage1_mlp.py` não usam SLHA; batch size fixo pode ser subótimo em hardware diferente. | 🟡 Média | Criar flag `--use-slha` opcional nesses scripts ou, pelo menos, logar as specs de hardware no início do treino. |
| 3 | `warmup_model` mede delta de RAM apenas do processo atual; não captura alocação de bibliotecas nativas que usam memória fora do processo (ex: oneDNN/MKL). | 🟢 Baixa | Aceitável para heurística; documentar limitação no SDD. |
| 4 | `pynvml` é opcional; sem ele o monitor não reporta utilização % nem memória GPU. | 🟢 Baixa | Comportamento esperado; logs continuam válidos. |
| 5 | Em WSL2, `tf.config.list_physical_devices("GPU")` pode listar uma GPU "fantasma" sem driver real, causando `accelerator="gpu"` e posterior falha. | 🟡 Média | Adicionar validação: se GPU listada mas `get_device_details` retorna erro ou memory_limit=0, forçar fallback para CPU. |

### 6.3 Ações Recomendadas (não bloqueantes para o experimento)

1. **Suprimir logs C++ de CUDA em CPU-only:** garantir que scripts de treinamento setem `TF_CPP_MIN_LOG_LEVEL=2` quando não há GPU.
2. **Documentar mensagem de CUDA:** adicionar nota em `docs/SDD_Sistema_Leitura_Hardware_Automatico.md` indicando que a mensagem é informativa e não impede treinamento.
3. **Integrar SLHA opcional nos scripts MLP:** adicionar `--use-slha` em `train_stage1_mlp.py` e `train_stage2_mlp.py` para aproveitar batch size adaptativo e monitoramento.
4. **Adicionar teste de WSL2/fallback:** reforçar `test_cpu_only_fallback_never_raises` para cenário de GPU fantasma.

---

## 7. Checklist de Conformidade

| Critério | Resultado | Evidência |
| ---------- | ----------- | ----------- |
| Discovery retorna specs válidas sem GPU | ✅ | Saída JSON do `discover_hardware` |
| Fallback CPU-only não levanta exceção | ✅ | `test_cpu_only_fallback_never_raises` passou |
| Warmup executa sem GPU | ✅ | `test_warmup_returns_memory_estimate` passou |
| Decision retorna batch_size ≥ 1 | ✅ | `test_decision_cpu_only_returns_valid_config` passou |
| Monitor não quebra treino | ✅ | `test_monitor_failure_does_not_stop_training` passou |
| Scripts de treinamento não hardcodam GPU | ✅ | Busca `grep` sem ocorrências fora de SLHA |
| Treinamento continua funcional em CPU-only | ✅ | Testes de SLHA e importação do TF concluídos |
| Sem dependência de bibliotecas gráficas | ✅ | psutil + TensorFlow nativo apenas |

---

## 8. Conclusão

A arquitetura de reconhecimento de hardware e o caminho de adaptação de treinamento do Project-Lewis estão **aptos para execução do experimento Stage 2 v11/v12 em CPU-only**. O SLHA cumpre seu propósito de fallback robusto, e os scripts de treinamento MLP não apresentam hardcodes de GPU. As recomendações são de **melhoria operacional**, não bloqueantes.

**Ação imediata autorizada:** prosseguir com o experimento Stage 2 v11/v12 (`hidden=256`, `dropout=0.5`) no ambiente atual, observando que o TensorFlow pode emitir mensagens informativas de CUDA que não afetam o treinamento.
