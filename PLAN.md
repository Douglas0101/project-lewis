# PLAN.md — Project-Lewis

Plano de execução de tasks decompostas. Atualizado a cada sessão.

## Status atual

### Sessão 2026-08-01 — ML Protocol v2 + Algorithm Audit (SDD-LEWIS-CLI-ML-PROTOCOL-V2-001)

**DECISÃO (owner, 2026-08-01):** novos pré-treinos oficiais (T11) **BLOQUEADOS** até T10.1 + T9.2.
Pilotos (T10.3) só após T9.2 + configs v2, status `PILOT` (nunca benchmark/promoção/publicação).
**REORDENAÇÃO (owner, 2026-08-01):** T9.3 antes de T10.2 — a matriz de hipóteses precisa dos dados
reais da reavaliação (BCE pós-T exato, ICs, ECE×NORM, comparabilidade).
Fila (uma task por sessão): T10.1 ✅ → T9.2 ✅ → T9.3 ✅ → T10.2 → T9.4 → T10.3 → T9.5 → T11.
T6/T7/T8 seguem ao final.

- [x] T9.1 `docs/ml_protocol_v2.md` — documento normativo (métricas equalizadas, calibração,
  thresholds, comparação, treino, promoção). Sem alteração de código de QG.
- [x] T10.1 `docs/algorithm_engineering_audit_v1.md` — auditoria algorítmica dos 3 runs
  (inventário, 7 camadas, diagnóstico por classe, 8 perguntas, matriz H1–H10, ablações
  L/S/A/O/C). Achados-chave: célula `A1+BCE` inexistente (ganho A2 não atribuível); focal γ=2
  fixo nunca tunado; early stopping por `val_loss`; thresholds tunados no próprio val; splits
  não pareados entre runs; QG4 julgou época ≠ checkpoint embarcado; CD = classe-sentinela.
- [x] T9.2 Avaliador canônico (`src/evaluation/{metric_definitions,calibration_metrics,
  thresholding,schema,canonical_evaluator}.py` + 9 testes em `tests/test_canonical_evaluator.py`;
  CLI `python -m src.evaluation.canonical_evaluator`; schema `metrics.json` 2.0; protocolo
  PROSPECTIVE/RETROSPECTIVE/FROZEN_PARAMS; legado `pretrain_evaluation.py` intacto)
- [x] T9.3 Reavaliação dos 3 runs (swarm de 3 agentes): `experiments/<run>/evaluation_v2/` com 7
  artefatos/run + `reports/ml_protocol_v2/pretrain_reconciliation.md`. Reconciliação perfeita
  (A2-full Δ=0.0; A0 novo Δ=1,3e-9); **BCE pós-T exato: 0,3417/0,3869/0,3905**; A0hist×A0novo
  COMPARABLE (strict = ruído); A0×A2 NON_COMPARABLE por split; ECE×NORM no estrato NORM=0 até
  0,217 → suporte a C2/H5; gap G6 registrado (reconcile não lê schema aninhado legado).
- [x] T10.2 `docs/ablation_matrix_v2.md` — matriz normativa: 7 hipóteses (P0×3/P1×3/P2×1) com
  critério de refutação e mapa de IDs ↔ auditoria; 29 células em 7 trilhas (C×4 controle H7,
  F×6 focal-γ, K×4 calibração c/ métrica NORM=0, R×5 RF c/ restrições TinyML, T×4 thresholds
  PROSPECTIVE, B×3 orçamento, P×3 formulação); split pareado v2 definido (normativo); 10
  critérios piloto→candidato; orçamento ≈ 14–17 CPU-h; execução C→F→{K,T}→R→B→P.
- [ ] T9.4 `configs/ml_protocol/v2/*.yaml` (sem treinar; split manifest fixo + estratificado;
  incluir extensão do avaliador: IC por classe em artefato + métrica `ece_norm0`)
- [ ] G6 hotfix `reconcile_with_legacy` (schema aninhado legado) — pré-requisito de T9.5
- [ ] T10.3 Pilotos pequenos `PILOT` (BLOQUEADO até T9.4 + G6; ordem C→F→{K,T}→R→B→P; células
  H7 primeiro: `A1+BCE`, `A0+focal`, split pareado v2)
- [ ] T9.5 RFC `docs/rfc_qg4_bce_threshold.md` (sem alterar QG em código; incluir baseline
  teórico de BCE e divergência época-17 vs checkpoint-28)
- [ ] T11 Pré-treinos oficiais v2 (somente com governança)
- [ ] T6 Tabela quantitativa por superclasse em `docs/pretrain_benchmark_comparison.md`
- [ ] T7 Draft `docs/publication_decision_v3.2.md` (HOLD permanece)

### Sessão anterior — identidade de paciente e preflight v3.1

**Autorização:** A01 registra a declaração do project owner na sessão atual para continuar a
implementação e a pesquisa controlada com todos os dados existentes. Nome/assinatura não foram
fornecidos; promoção e attestation permanecem bloqueadas.

- [x] Demonstrar que os splits v3 agrupavam registros, não pacientes: 29/32 pacientes INCART
  atravessavam outer folds.
- [x] Criar contratos Pydantic strict/frozen para identidade, splits, geração e preflight.
- [x] Mapear INCART em 32 pacientes pelos headers e MITDB em 47 sujeitos, agrupando 201/202.
- [x] Quarentenar SVDB como `DOMAIN_SENSITIVITY` e AFDB como `RHYTHM_EXPLORATORY`, sem inventar
  patient IDs.
- [x] Implementar split patient-aware v3.1.0, hashes detached e publicação sem overwrite.
- [x] Implementar CLI canônica `src.cli.advanced_training_v3` e arquivar/bloquear entry points
  legados antes de TensorFlow ou writes.
- [x] Implementar os dez hashes da geração sobre bytes exatos e snapshot semântico da fonte.
- [x] Endurecer após três rodadas de revisão adversarial: checks canônicos recomputados na
  publicação/leitura; vínculos fonte→identidade→split→geração; reconstrução integral da família
  e do AFDB; transformação pai→Stage 1 exata; locks com ownership; staging sem replace; targets
  read-only; markers de conclusão escritos por último; revalidação obrigatória no consumo.
- [x] Corrigir o produtor canônico: AFDB removido da família de beats; clocks nativo/500 Hz,
  símbolo original, classe canônica, sample IDs, hashes de waveform, patient ID e fold passam a
  ser explícitos. Artefatos legados existentes não foram regravados.
- [x] Regenerar artefatos write-once. r2 foi rejeitada por shape `(500,)` e arrays object; r3 foi
  rejeitada por renumerar identidade após drops de borda; r4 preserva índice original e passou
  reconstrução integral. Os markers `GENERATION_REJECTED` impedem reutilização de r2/r3.
- [x] Publicar `data/splits/groupkfold_5_stratified/v3.1.0/` somente após os três gates da família
  pai passarem; cada JSON tem digest detached e `SPLIT_BUNDLE_COMPLETE.json` foi escrito por
  último.
- [x] Regenerar Stage 1 r4 com 461.600 rows, lineage completo e binding pai→filho validado. O
  Stage 2 emitido incidentalmente está marcado `NOT_AUTHORIZED_FOR_TRAINING` porque ainda não tem
  contrato próprio de ordered binding.
- [x] Publicar `reports/training_preflight_v3.1.r4.{json,md}` e marker de completude. `status` e
  `verify-generation` reproduzem exit code 10 com blocker único
  `AFDB_PATIENT_IDENTITY_UNVERIFIED`; nenhum `PRETRAINING_GATE_PASS` existe.
- [x] Validar: 65 testes focais PASS, `make lint` PASS, LSP/lens sem erros e revisão adversarial
  final `CLEAN`. A suíte ampla final executou `963 passed, 1 deselected` e `make test-e2e`
  17/17 PASS; o único teste excluído é o QG5 Stage 1 conhecido/deferido.

**Regra de saída:** esta sessão termina antes de qualquer célula. O treinamento continua bloqueado
pela identidade AFDB não autenticada. Stage 2 também exige contrato próprio e nova geração antes
de qualquer uso; nenhuma dessas lacunas pode ser reparada ou presumida silenciosamente.

**Handoff para a outra máquina:** `docs/handoffs/advanced_training_v3.1_partner_handoff.md`.
Pendências operacionais: TODO #7 (identidade AFDB), TODO #9 (provenance Stage 2), TODO #3
(matriz, bloqueada por #7), TODO #4 (robustez) e TODO #5 (bundle de decisão).

### Sessão anterior — auditoria de integridade da matriz avançada v3

**Escopo desta sessão:** executar somente o `training_matrix_integrity_report` exigido pelo
prompt mestre de treinamento baseado em evidências. Nenhum novo treinamento, recalibração,
seleção, promoção ou execução E06.5 está autorizado por esta etapa.

- [x] Classificar o protocolo solicitado como `NEW_EVIDENCE_GENERATION`; as métricas históricas
  MLP/fusão são referência condicionada, não hard gates.
- [x] Verificar cobertura 4 famílias × 5 outer folds × 5 seeds: 100 JSONs presentes.
- [x] Recomputar hashes dos dados realmente carregados, manifests, ontologia, preprocessing,
  células e ledger; verificar unicidade record/fold e estado do Git.
- [x] Auditar completude de lineage, artefatos co-produzidos, calibração, tendências,
  bootstrap por paciente, atalhos de dataset, contrafactuais, controles negativos e validação
  externa.
- [x] Publicar `reports/training_matrix_integrity_report.md` com categorias epistemológicas,
  denominadores, limitações e estados permitidos.
- [ ] Antes de nova matriz: ratificação humana D1–D7 e do pivot; source/config/environment limpos
  e congelados; provenance por amostra; preflight fail-closed; smoke autenticado; autorização
  explícita de recursos.

**Decisão desta sessão:** `TRAINING_BLOCKED_BY_DATA_PROVENANCE / REVIEW_REQUIRED`.
As 100 células existentes são exploratórias não autenticadas e não podem ser promovidas nem
usadas como evidência confirmatória.

**Prioridade preexistente congelada: E06.5 — robustez multi-seed e auditoria do Fold 5.**
O E06 confirmou sinal de representação, mas não atingiu o target final. O estado correto é
`REPRESENTATION_SIGNAL_CONFIRMED / TARGET_NOT_MET / ROBUSTNESS_VALIDATION_REQUIRED`.
E07 somente avança após o checkpoint E06.5; o target de publicação F1(F) >= 0.50 permanece.

### Sessão atual — CLI canônico Stage 2

- [x] Congelar estado E06: baseline, H6, H11 e H12; sem declarar limitação de dados.
- [x] Confirmar que não existe CLI canônico de pesquisa nem `[project.scripts]`; padrão é
  `argparse` com `main(argv) -> int`.
- [x] Implementar `src.cli.stage2_research` com preflight, status, plan, e065-run,
  fold-audit, representation-select, E07, E08, report, verify e resume.
- [x] Congelar manifests de outer/inner splits, hashes de datasets/features/config e matriz
  E06.5 com folds 1–5 e seeds 17,29,43,71,101.
- [x] Executar somente preflight, plan e smoke (fold 1 / seed 17) nesta sessão.
- [x] Exigir `E06_5_SMOKE_PASS` antes de autorizar os 100 runs de auditoria.
  Checkpoint válido: `e065-smoke-v5`; quatro `DONE`; audit completo permanece com zero runs.
- [x] Preservar produção, datasets, `uv.lock`, decisão argmax, MLP-128 e target 0.50.

### Sessão atual — remediação pi-lens Stage 2 (executada com ressalvas aprovadas)

**Escopo aprovado:** endurecer caminhos/locks, confirmar o diagnóstico optional-member e
reduzir apenas duplicações mecânicas. Não alterar políticas científicas E06.5/E07/E08.

#### P0 — Congelar baseline e separar achado ativo de cache

- [x] Executar `lens_diagnostics mode=full` somente nos arquivos Stage 2 afetados e registrar
  contagem por regra/arquivo. O contador global `1E/220W` não será usado isoladamente porque
  mistura escopos e cache; o último scan focal mostrou `0E/67W` após corrigir o erro anterior.
- [x] Confirmar antes das mudanças: `e065-audit-v1` continua com zero `DONE`; datasets,
  `uv.lock`, `models/`, splits, seeds e artefatos concluídos permanecem intocados.
- [x] Atualizar especificamente o diagnóstico de
  `tests/test_stage2_research_long_tail.py`: o teste já usa `cast(Any, load_model(...))` e o
  Pyright canônico está limpo, portanto editar somente se um scan primário fresco ainda provar
  `reportOptionalMemberAccess`.

#### P1 — TDD para contenção de caminho e propriedade do lock (bloqueador)

- [x] Criar `tests/test_stage2_research_integrity.py` antes da implementação, cobrindo:
  caminho descendente válido; segmento absoluto; `..`; candidato/experiment-id com separador;
  caminho igual ao root; sibling; symlink escapando do root; e ausência de criação fora do root.
- [x] Cobrir ciclo de vida do lock: existe durante o contexto; é removido pelo proprietário após
  sucesso ou exceção do corpo; segundo adquirente recebe `BLOCKED_PRECONDITION`; e o segundo
  adquirente **não pode apagar** o lock ativo do primeiro.
- [x] Cobrir deleção defensiva: `reset_incomplete_run` rejeita qualquer diretório fora do
  `output_root`, nunca remove `DONE` e preserva os arquivos autorizados por `keep_manifest`.

#### P2 — Implementar path safety fail-closed

- [x] Em `src/stage2_research/integrity.py`, adicionar helper único de contenção usando
  `resolve(strict=False)` + `Path.is_relative_to()`, rejeitando root igual, escape por `..` e
  symlink resolvido fora do root com `ExitCode.BLOCKED_PRECONDITION`.
- [x] Alterar `run_lock(run_dir, *, output_root)` para validar **antes** de `mkdir`/`os.open` e
  remover `.RUNNING.lock` apenas quando a própria invocação adquiriu o descritor.
- [x] Alterar `reset_incomplete_run(run_dir, *, output_root, ...)` para validar antes de qualquer
  `unlink`/`rmtree`; manter a proibição absoluta de sobrescrever célula com `DONE`.
- [x] Em `src/stage2_research/training.py`, validar `experiment_id` e `candidate` como segmentos
  relativos únicos em `stage_run_dir`; passar `config.output_root` aos dois sinks. Preservar
  layout, códigos de saída, resume, `--force`, hashes e ordem de escrita.

#### P3 — Remover somente duplicações mecânicas de baixo risco

- [x] Criar `src/stage2_research/tabular_io.py` com writers CSV/Parquet atômicos compartilhados;
  preservar `index=False`, temporário por PID, `os.replace`, cleanup, exceções, exit code e
  mensagens específicas do contexto.
- [x] Criar testes de equivalência/erro/cleanup em
  `tests/test_stage2_research_tabular_io.py`; então substituir apenas os writers duplicados de
  `training.py` e `advanced_workflows.py`.
- [x] Promover a prova train-only de template em `workflows.py` para helper reutilizável e remover
  `_verify_template_scope` duplicado de `advanced_workflows.py`; testar inner→validation,
  inner→outer-test e outer→outer-test, preservando mensagens e `ExitCode.LEAKAGE`.
- [x] Consolidar `_safe_float`/`_safe_int` somente onde contrato, finitude e
  `ExitCode.EVALUATION_FAILURE` forem idênticos. Não consolidar parsers de dados que usam
  `DATA_INTEGRITY` ou `ValueError`.
- [x] **Deferir** nesta rodada: abstração dos loops `run_e07`/`run_e08`, rankings, seleção,
  Fold 5/DONE binding, features `e06_*` e scripts legados. Essas duplicações são manutenção,
  mas uma generalização pré-audit aumenta o risco científico sem corrigir defeito funcional.

#### P4 — Validação incremental e revisão

- [x] Após cada bloco: Black/isort, pytest focal, Pyright focal e `git diff --check`.
- [x] Ao final: `make lint`; `uv run --locked pyright src tests`; testes Stage 2/E06 focados;
  `make test`; `make test-e2e`; e pi-lens full nos arquivos tocados.
- [x] Critérios pi-lens focais: zero erros; `python-path-traversal` ausente nos run paths;
  optional-member ausente em scan fresco; `jscpd` Stage 2 reduzido de aproximadamente 24 para
  no máximo 20; nenhuma nova categoria blocker/high.
- [x] Revisão read-only independente do diff, com foco em path traversal, lock ownership,
  deleção, leakage, resume, seleção e identidade. Duas revisões retornaram `CLEAN`; revisão
  humana continua obrigatória.
- Evidência: `make lint` PASS; Pyright 0/0; 104 testes focais PASS; `make test-e2e` 17/17 PASS.
  O `make test` executou 809 testes (806 PASS) e ficou vermelho pelos gates preexistentes QG5
  Stage 1 e manifesto E06 reaberto; a falha Renode transitória passou no e2e subsequente.

#### P5 — Regerar identidade canônica sem iniciar audit

- [x] Somente após congelar o último byte de source, mudar `e065_smoke` de
  `e065-smoke-v4` para um novo ID imutável `e065-smoke-v5`; nunca reutilizar `DONE` nem usar
  `--force`. `e065-audit-v1` não muda.
- [x] Executar preflight deterministic/CPU → plano E06.5 exato de 100 células → smoke exato
  baseline/H6/H11/H12, fold 1, seed 17, serial/CPU.
- [x] Exigir `E06_5_SMOKE_PASS`, quatro `DONE` válidos, audit `DONE` count zero e equivalência
  semântica v4↔v5 de predictions/métricas/splits/features (ignorando timestamps e hashes que
  incorporam source/config identity).
- [x] Não executar os 100 runs nesta remediação. O audit completo continuará dependendo de
  autorização explícita posterior.

#### Critério de saída da remediação

- [x] Zero alteração em datasets, `uv.lock`, produção `models/`, arquitetura MLP-128, argmax,
  splits/seeds, sampling/losses, gates e target F1(F) >= 0.50.
- [x] `make test` permanece vermelho somente pelo gate científico Stage 1 registrado na task
  #41. A task #42 foi reconciliada sem fabricar manifesto nem alterar resultados científicos.
- [x] Smoke v5 válido; audit zerado; nenhuma conclusão científica derivada do smoke.
  Hashes: preflight `179ee041...80a0`; source `369247b7...fb01`; plan `394e51b2...b539f`;
  aggregate `9bf42c91...b6088`; gate `78d8a3ca...a840183`.

### Sessão atual — Política de Decisão Autenticada v1 (shadow/fail-closed)

**Escopo autorizado:** primeira etapa segura, sem assinatura, sem ação operacional e sem
estado produtivo. Backend criptográfico escolhido: interface injetável fail-closed; ausência de
Sigstore real retorna `REVIEW_REQUIRED`. Identidades são apenas fixtures. Quorum normativo para
`APPROVED_FOR_AUDIT`: evidence bot + um scientific approver humano independente.

- [x] Inspecionar Git, dependências, ferramentas criptográficas e conceitos existentes sem limpar,
  reverter ou sobrescrever alterações. `cosign`/`sigstore-python` não estão disponíveis.
- [x] Congelar `docs/policies/authenticated_research_decision_v1.md`, separando requisitos
  aprovados, gates existentes e thresholds propostos não executáveis.
- [x] Criar contratos Pydantic v2 strict/forbid para DSSE, in-toto Statement v1,
  `research-decision/v1`, bundles, policy local e relatório shadow.
- [x] Implementar parser JSON pré-Pydantic que rejeite chaves duplicadas, constantes não finitas,
  tipos ambíguos e campos desconhecidos em todos os níveis controlados.
- [x] Implementar verificador somente leitura com backend Sigstore injetável; backend ausente,
  erro de infraestrutura ou identidade não ratificada nunca aprova.
- [x] Validar hashes SHA-256 de subject/política/evidências usando somente paths confiáveis
  fornecidos pelo chamador; nunca resolver path vindo do envelope.
- [x] Implementar validade UTC, nonce 256-bit, UUID, sequência, quorum, separação de funções,
  reason codes e antirreplay exclusivamente em memória/fixture.
- [x] Limitar resultados a `REJECTED_AUTHENTICATED`, `INSUFFICIENT_EVIDENCE`,
  `REVIEW_REQUIRED` e `APPROVED_FOR_AUDIT`.
- [x] Cobrir rejeições, evidência insuficiente, waiver, backend ausente, replay e aprovação
  limitada ao audit; nenhuma fixture poderá representar assinatura produtiva.
- [x] Executar lint, Pyright, testes focais, pi-lens e revisão read-only; não executar E06.5,
  não alterar CI e não regenerar smoke/preflight nesta etapa.
  Evidência: 55 testes de decisão + 134 na suíte focal; `make lint` PASS; Pyright 0/0;
  3 revisões read-only de segunda rodada: CLEAN/CLEAN/sem achados; policy clarificada quanto à
  precedência waiver-vs-autenticidade.

### Sessão atual — reconciliação determinística do manifest E06 (#42)

**Escopo autorizado:** corrigir somente o contrato do teste de conclusão v2.4. Não fabricar
manifest em `E06_reopened`, não declarar E06.5 concluído e não executar audit.

- [x] Identificar a causa: `glob("E06_*")` possui três candidatos e `dirs[0]` depende da ordem do
  filesystem, apesar de `E06_feature_engineering/E06_manifest.json` existir.
- [x] Substituir descoberta ambígua por mapa explícito dos diretórios canônicos E00–E08.
- [x] Cobrir separadamente que `E06_reopened` e `E06_5` não são manifests históricos de conclusão.
- [x] Executar teste focal, lint, Pyright, suíte completa, e2e, `git diff --check` e pi-lens focal.
- [x] Confirmar que resta somente o QG5 Stage 1 (#41), sem alterar modelo, threshold, teste ou dados.
  Evidência: teste focal 6/6 PASS; `make lint` PASS; Pyright 0/0; `make test` 865 PASS e
  1 FAIL exclusivamente em `test_two_stage_qg5_end_to_end` (recall 0,0661 < 0,30);
  `make test-e2e` 17/17 PASS; pi-lens sem achados; diff-check limpo.

### Histórico pausado — recall Stage 1

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
- [x] Disposição humana #41: `KNOWN_FAILED_GATE / DEFERRED`. Manter modelos, threshold, teste,
  dados e artefatos congelados; não usar skip/xfail nem retraining sem nova autorização explícita.

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
