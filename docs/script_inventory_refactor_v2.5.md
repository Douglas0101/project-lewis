# Inventário de scripts para refatoração v2.5

Data: 2026-07-11

## Critérios

- **Manter como CLI**: entrypoint operacional, documentado, coberto por testes ou chamado pelo Makefile.
- **Promover lógica para `src/`**: script ainda útil, mas contém algoritmo reutilizável ou lógica de domínio extensa.
- **Arquivar**: experimento reproduzível, resultado negativo ou utilitário de versão encerrada; preservar evidência, mas retirar do caminho operacional.
- **Excluir somente após revisão humana**: duplicata comprovada e sem valor de reprodutibilidade. Nenhum arquivo será excluído nesta fase.

## Resultado

Foram avaliados 40 scripts: 12 sem referência direta em testes/documentação/Makefile, 9 explicitamente experimentais/versionados e 8 com 300 ou mais linhas.

### Manter como CLI operacional

| Script | Motivo |
| -------- | -------- |
| `audit_training_data.py` | Quality Gate de dados; chamado pelo Makefile. Refatoração interna necessária. |
| `check_environment.py` | Diagnóstico de ambiente; chamado pelo Makefile. |
| `eval_hybrid.py` | Avaliação operacional; chamado pelo Makefile. |
| `generate_filter_coeffs.py` | Geração de coeficientes firmware; documentado no SDD. |
| `generate_quality_report.py` | Consolidação histórica de QGs; deve convergir com `run_quality_gates.py`. |
| `memory_commit.py` | Integração da memória; chamado pelo Makefile. |
| `prepare_stage1_features.py` | Pipeline oficial de features do Estágio 1. |
| `prepare_stage2_features.py` | Pipeline oficial de features do Estágio 2. |
| `prepare_two_stage_datasets.py` | Preparação oficial dos datasets em duas etapas. |
| `quantize_mlp_features.py` | Quantização do pipeline atual v2.3. |
| `run_hard_gates.py` | Execução de gates firmware. |
| `run_stage1_training.py` | Orquestração oficial do Estágio 1. |
| `run_stage2_training.py` | Orquestração oficial do Estágio 2. |
| `run_two_stage_pipeline.py` | Execução do pipeline completo. |
| `select_best_mlp_fold.py` | Publication guard e seleção auditável. |
| `train_stage1_mlp.py` | Treinamento MLP atual. |
| `train_stage2_mlp.py` | Treinamento MLP atual; precisa extrair lógica para `src/models/`. |
| `validate_firmware_deliverables.py` | Validação dos entregáveis C08. |
| `validate_knowledge_index.py` | Validação C11; chamado pelo Makefile. |
| `validate_quantized_mlp.py` | Validação INT8; chamado pelo Makefile. |

### Promover lógica para `src/`

| Script | Destino sugerido |
| -------- | ------------------ |
| `analyze_training_dynamics.py` | `src/models/training_analysis.py` |
| `apply_pruning_qat.py` | `src/models/quantization/pruning_qat.py` |
| `audit_stage2_feature_separability.py` | `src/models/stage2_analysis.py` |
| `audit_stage2_patient_distribution.py` | `src/data/patient_distribution.py` |
| `audit_stage2_split_protocol.py` | Consolidar em `src/models/split_protocol.py` |
| `audit_stage2_labels_e07.py` | `src/features/label_audit.py` |
| `engineer_stage2_features_for_class_f.py` | `src/features/stage2_class_f.py` |
| `resample_f_by_patient.py` | `src/data/patient_resampling.py` |
| `train_stage2_baseline_enhanced.py` | Consolidar runner em `src/models/` |
| `train_stage2_mlp.py` | Extrair dataset, loss, threshold e treinamento para módulos testáveis. |

Os scripts permanecem como wrappers CLI finos após a promoção.

### Arquivar como evidência experimental

Destino sugerido: `experiments/stage2_v2.4_research/scripts/` ou `archive/scripts/` com manifest de origem.

- `_smoke_stage1.py`
- `analyze_stage1_v2_training.py`
- `analyze_stage2_thresholds_v11_v13.py`
- `optimize_stage1_threshold.py`
- `quantize_finetuned_v1.1.py`
- `quantize_two_stage_v2.0.py`
- `retrain_stage1_mlp_best_fold.py`
- `run_finetune_groupkfold.py`
- `run_qg5_v2.4_gates.py`
- `smoke_test_stage1_scratch.py`
- `smoke_test_stage1_unfrozen.py`

### Candidatos a consolidação

- Unificar `_smoke_stage1.py`, `smoke_test_stage1_scratch.py` e `smoke_test_stage1_unfrozen.py` em um único CLI parametrizado.
- Unificar `generate_quality_report.py`, `run_hard_gates.py` e o futuro `run_quality_gates.py` sob um contrato comum de resultados.
- Eliminar versões numéricas no nome do código operacional; manter versão no manifest/configuração.
- Fazer `scripts/` um pacote consistente ou configurar mypy com `explicit_package_bases`, evitando o erro de módulo duplicado.

## Política de preservação

Resultados negativos e scripts de pesquisa não serão apagados. O arquivamento deve preservar:

1. hash SHA-256 do script original;
2. commit de origem;
3. caminhos dos artefatos produzidos;
4. dependências/configuração usadas;
5. decisão PASS, FAIL ou `PASS_HYPOTHESIS_REJECTED`.

## Ordem de refatoração

1. Qualificar e enxugar `audit_training_data.py`.
2. Criar runner unificado de Quality Gates.
3. Extrair lógica de `train_stage2_mlp.py` para `src/models/`.
4. Promover módulos analíticos da research branch.
5. Arquivar wrappers versionados e smoke tests duplicados após revisão humana.
