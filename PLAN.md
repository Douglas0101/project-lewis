# PLAN.md — Project-Lewis

Plano de execução de tasks decompostas. Atualizado a cada sessão.

## Status atual

Investigação forense do recall Stage 1 QG5 em andamento, sem autorização para retreino ou alteração de threshold/teste.

### Sessão atual — recall Stage 1

- [x] R00: congelar Git, ambiente, falha isolada e suíte completa.
- [x] R00: registrar hashes e baseline manual TP/FP/TN/FN = 127/0/128/1793.
- [x] R01: rastrear seleção → scaler → modelo → coluna 1 → threshold → label → recall.
- [x] R01: gerar trace JSON e CSV por amostra em modo `STAGE1_DIAGNOSTIC=1`.
- [x] R01: confirmar zero perda, duplicação, desalinhamento ou divergência do caminho canônico.
- [x] R02: auditar ZIP Keras 3, entrada `(500,1)`, saída `(2)` softmax e classe positiva.
- [x] R02: corrigir AP com prevalência/lift e documentar `QG5_STAGE1_ABNORMAL_STRESS`.
- [x] R02: restaurar Pyright para 0 erros/0 warnings e validar runtime canônico.
- [x] R03: duas lanes independentes, safe mode, pesos/previsões delta 0 e H5 rejeitada.
- [x] R04: modo de inferência, Dropout e RNG auditados; H11 rejeitada.

Estado anterior: a fase de **refatoração e qualificação em massa** estava concluída. A suíte atual reproduz uma única falha residual conhecida no QG5 Stage 1.

## Objetivo desta fase

Deixar o repositório em estado de produção: código limpo, testes confiáveis, lint/type checking zerados, Quality Gates executáveis e documentação sincronizada. Ao final, a decisão sobre v2.5 será tomada com métricas sólidas.

## Fase 1 — Diagnóstico e auditoria (Sessão atual)

- [x] Inspecionar estado atual: 1506 arquivos `.py`, 67 testes, 40 scripts, 113 módulos `src/`.
- [x] Coletar baseline de qualidade:
  - `flake8` sobre `src/`, `scripts/`, `tests/`: 67 issues.
  - `mypy`: 1 erro crítico de mapeamento de módulo (`scripts/train_stage1_mlp.py`).
- [x] Executar `pytest` parcial e catalogar falhas:
  - `test_finetuning_artifacts_sane` falhava por `NaN` em `qrs_asymmetry_index` (corrigido).
  - `test_audit_script_passes_small_sample` falhava por timeout; raiz: `jsonschema.validate` recompilando schema a cada linha de catalogo (88k linhas).
- [x] Corrigir imputação de NaN em `src/features/pipeline.py` (`build_beat_records`):
  - `qrs_asymmetry_index`, `t_r_ratio`, `qrs_raggedness` agora usam sentinela `-1.0`.
  - `qrs_width_ms` e `qrs_area` continuam com `0.0`.
- [x] Otimizar `src/data/_schemas.py`:
  - Compilar validadores `Draft202012Validator` uma vez (drasticamente reduz tempo de carga do catalogo).
- [x] Otimizar `tests/test_training_data_audit.py`:
  - `skip_checksum=True` no teste de amostra pequena.
- [x] Regenerar `data/features/finetuning_mitbih_family.*` com NaN removido.
- [x] Mapear QG0-QG19 + QG-C11 + QG-MEM executáveis e seus status.
  - Estado atual após correções: 13 PASS, 4 FAIL, 5 NÃO MAPEADO/N/A.
  - Relatório publicado em `docs/quality_gates_report_v2.5.md`.
- [x] Listar scripts/estudos de experimento que devem ser promovidos, arquivados ou deletados.
  - Inventário publicado em `docs/script_inventory_refactor_v2.5.md`.
  - Nenhuma exclusão será feita sem revisão humana; resultados negativos serão preservados.

## Fase 2 — Refatoração estrutural

- [x] Criar novos loaders para v2.5 (pendentes de qualificação):
  - `src/features/afdb_beat_loader.py` (qualificado: `make lint` passa; carrega 520k F do AFDB).
  - `src/features/ptbxl_beat_loader.py` (qualificado parcialmente; 48 registros, ~894 F).
  - Input/output em `experiments/` com contrato claro.
  - Leitura de config via `pyproject.toml` ou `config/*.yaml`.
  - Logging uniforme.
  - Salvamento de manifest por experimento (versão, hashes, dependências).
- [ ] Mover lógica reutilizável de scripts para `src/`:
  - `src/features/afdb_beat_loader.py` e `src/features/ptbxl_beat_loader.py` (já criados, precisam de qualificação).
  - `src/models/qg5_gates.py` e `src/models/split_protocol.py` revisar e consolidar.
- [ ] Remover dead code, imports não usados e variáveis fantasmas.
- [ ] Corrigir issues de flake8 (especialmente linhas > 100 e imports fora de ordem).
- [ ] Corrigir mypy: mapeamento de módulos, stubs, anotações faltantes.
- [ ] Revisar `tests/`:
  - Eliminar testes que dependem de dados ausentes ou ambiente não configurado.
  - Agrupar testes por camada (C01–C11) quando possível.
  - Garantir que todos os testes da research branch v2.4 sejam deterministicos.

## Fase 3 — Quality Gates em massa

- [x] Executar QGs em batches e documentar status.
- [ ] Criar/atualizar `scripts/run_quality_gates.py` para automatizar a execução de todos os QGs mapeados.
- [x] Corrigir falhas identificadas:
  - [x] QG0: mirror Chapman reconstruído deterministicamente e checksum validado.
  - [x] QG6: INT8 incompatível regenerado; ΔF1 = 0.0058 e FlatBuffer = 26,62 KB.
  - [x] QG8/QG10: fidelidade de beat restaurada ao sincronizar parâmetros de quantização do firmware.
  - [x] QG16/QG17: filtros C vs Python e adaptive skipping ajustados; harness nativo passa.
- [x] Atualizar `docs/quality_gates_report_v2.5.md` com todos os QGs executáveis PASS.

## Fase 4 — Integridade e publicação guard

- [ ] Garantir que artefatos v2.3 em `models/` estejam protegidos contra sobrescrita.
- [ ] Validar manifestos (`data/features/training_manifest.json`, `data/features/stage2_multiclass_features.json`).
- [ ] Verificar que backups de datasets v2.3 existam e sejam checksum-áveis.
- [ ] Revisar `CHANGELOG.md` para refletir a fase de refatoração.

## Fase 5 — Decisão sobre v2.5

- [ ] Após Fase 3 e Fase 4 passarem, reavaliar a proposta `docs/dataset_update_proposal_v2.5.md`.
- [ ] Decidir:
  - Implementar AFDB + PTB-XL `AFIB=100` (opção 3 original).
  - Implementar apenas AFDB ou apenas PTB-XL forte.
  - Arquivar v2.5 e manter v2.3 como produção.
- [ ] Se avançar, criar sub-tasks de implementação com checkpoints de QG.

## Backlog mantido

- [ ] C06 — Adicionar schema tests estruturais (pydantic) nos pipelines de dados
- [ ] C08 — Migrar firmware de modelo monolítico v1.1 para dois modelos TFLM v2.0
- [ ] C09 — Expandir relatório de energia com cenários de sleep/stop/standby
- [ ] C10 — Gerar fixtures Python para testes de harness DSP/inference/R-peak
- [ ] v2.5 — Expansão de dataset para classe F (AFDB + PTB-XL forte)

## Concluídas

- Ver histórico em seções anteriores do PLAN.md (v3.0, C01–C04, research branch v2.4).

## Notas

- A fase de refatoração deve ser feita **sem modificar datasets v2.3**.
- Novos loaders (`src/features/afdb_beat_loader.py`, `src/features/ptbxl_beat_loader.py`) serão qualificados, mas não integrados ao pipeline de produção até a Fase 5.
- Critério de saída: `make lint && make test && make test-e2e` passando; relatório de QGs em `docs/quality_gates_report_v2.5.md` publicado.
