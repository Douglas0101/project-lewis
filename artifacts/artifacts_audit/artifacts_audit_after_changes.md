# Auditoria de artefatos — DEPOIS das alterações

**Data:** 2026-08-02 · **Repo:** `Project-Lewis`, branch `develop` @ `db827d0` + alterações locais (não commitadas)
**Base:** `artifacts_audit_before_changes.md` (mesma sessão) · **Método:** `scripts/inspect_training_artifacts.py` (novo, read-only)

---

## 1. Matriz dos runs (históricos — inalterados)

| run | célula | arch | loss | seed | perfil | checkpoint_epoch | qg4_epoch | macro PR-AUC | macro AUROC | ECE pós | T | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260802_035117 | c0 | a0 | bce | 13 | n/a (pré-perfil) | 29 | 29 | 0,6458 | 0,8241 | 0,0159 | 0,939 | PILOT · QG4 FAIL · **test.npz sem freeze** |
| 20260802_161312 | c0 | a0 | bce | 13 | fast (rótulo "strict") | 20 | 25 | 0,6595 | 0,8247 | 0,0184 | 1,001 | PILOT · QG4 FAIL |
| 20260802_043748 | c1 | a1 | bce | 13 | n/a | 29 | 29 | 0,6749 | 0,8502 | 0,0187 | 0,915 | PILOT · QG4 FAIL · **test.npz sem freeze** |
| 20260802_163129 | c1 | a1 | bce | 13 | fast ("strict") | 26 | 28 | 0,6956 | 0,8551 | 0,0190 | 0,961 | PILOT · QG4 FAIL |
| 20260802_051138 | c2 | a1 | focal | 13 | n/a | 29 | 16 | 0,6748 | 0,8521 | 0,0217 | 0,340 | PILOT · QG4 FAIL · **test.npz sem freeze** |
| 20260802_165458 | c2 | a1 | focal | 13 | fast ("strict") | 30 | 29 | 0,6964 | 0,8592 | 0,0222 | 0,358 | PILOT · QG4 FAIL |
| 20260802_054320 | c3 | a0 | focal | 13 | n/a | 29 | 16 | 0,6507 | 0,8280 | 0,0149 | 0,344 | PILOT · QG4 FAIL · **test.npz sem freeze** |
| 20260802_171757 | c3 | a0 | focal | 13 | fast ("strict") | 30 | 13 | 0,6723 | 0,8345 | 0,0185 | 0,363 | PILOT · QG4 FAIL |
| 20260728_053011 | A2-full | a1 | focal | 13 | n/a | 29 | (sem QG4) | 0,7065 | 0,8639 | 0,0152 | 0,374 | congelado (git-tracked) |

Runs históricos **não foram modificados** — as divergências acima (época QG4 ≠ checkpoint, rótulo
"strict", `test.npz`) permanecem como evidência registrada, por design (§14 do protocolo).

## 2. Novos artefatos (pós-correção, smoke 1 época)

| run | célula | loss | perfil registrado | checkpoint_epoch | qg4_epoch | braços do QG4 | issues do inspetor |
|---|---|---|---|---|---|---|---|
| 20260802_210322 | c1 (smoke) | bce | `fast` ✔ coerente | 1 | 1 | `val_auc_roc`, `val_loss` | **0** |
| 20260802_210707 | c2 (smoke) | focal | `fast` ✔ coerente | 1 | 1 | `val_auc_roc`, **`val_bce_monitor`** | **0** |

Evidência direta nos artefatos novos:
- `provenance.json`: `deterministic_mode="fast"` (antes: `"strict"` falso) + bloco `runtime`
  `{"profile":"fast","onednn":true,"deterministic_ops":false,"intra_threads":"2","inter_threads":"1","cuda_visible_devices":"-1"}`;
- `provenance.dataset.split_policy`: `"paired manifest chapman-record-disjoint-paired-v2
  (train=0.8/validation=0.1/calibration=0.05/test=0.05)"` (antes: texto legado fixo);
- `provenance.hashes.split_manifest_sha256 = 988d78ae…211f` == sha256 real de
  `data/splits/chapman_paired_v2/manifest.json` ✔;
- `qg4_result.json` focal: braços `val_auc_roc` + **`val_bce_monitor`**, `decision_rule:
  "val_auc_roc > min AND val_bce_monitor < max at checkpoint epoch (best val_auc_pr)"`;
- `run_status.qg4`: `gate_loss_metric` + `checkpoint_monitor` explícitos; `known_issues: []`
  (ressalva focal tornou-se desnecessária — o artefato agora diz a verdade);
- `evaluation_v2/predictions/`: somente `validation.npz` + `calibration.npz` (teste bloqueado ✔);
  IDs presentes; 10 segmentos/registro; overlap ∅.

## 3. Linhagem de hashes (verificada — todas as referências válidas)

| artefato | referência | verificação |
|---|---|---|
| `data/splits/chapman_paired_v2/manifest.json` | `988d78ae…211f` | == `provenance.hashes.split_manifest_sha256` (runs novos) ✔ |
| `config.json` | declarado em `provenance.hashes.config_sha256` | == calculado, 11/11 runs históricos + 2 novos ✔ |
| `history.json` | `provenance.hashes.history_sha256` | == calculado ✔ |
| `backbone_pretrained.keras` | `provenance.hashes.model_sha256` e `validation_meta.sha256_model` | == calculado ✔ |
| `metrics_per_class.json` | `provenance.hashes.metrics_per_class_sha256` | == calculado ✔ |
| `evaluation_v2/*` | produzidos pós-treino pelo orquestrador | sem hash na proveniência (lacuna registrada — por design do orquestrador) |
| `qg4_result.json` / `run_status.json` | artefatos terminais | `qg4.pass` consistente entre os dois e com `provenance.qg4.pass` ✔ |

## 4. Comparação antes × depois (por inconsistência)

| # | Inconsistência | Estado anterior (evidência) | Correção | Estado posterior (evidência) |
|---|---|---|---|---|
| D1 | `test.npz` sem freeze | 5 runs históricos (`4c88def`/`489993e`) com `test.npz` | nenhum patch — trava `2cdb3b1` já vigente; inspetor agora falha (exit 3) nesse caso | runs novos sem `test.npz`; inspetor reproduz exit 3 nos históricos |
| D2 | runtime solicitado × registrado | `runtime_profile: fast` × `deterministic_mode: "strict"` + oneDNN ativo (runs da tarde) | F1: wrapper propaga `LEWIS_RUNTIME_PROFILE`; treino resolve o modo pelo perfil; proveniência grava bloco `runtime` efetivo | `deterministic_mode: "fast"` + `runtime.profile: "fast"` coerentes (run 210322/210707) |
| D3 | QG4 julga época ≠ checkpoint | 4/4 runs da tarde (ex.: C3 QG4=ép.13, checkpoint=ép.30); C2-manhã: braço AUC FAIL (0,8459) vs checkpoint PASS (0,8576) | F2: `_checkpoint_epoch_metrics` seleciona a época do monitor ES/MC e lê o gate nessa época | `qg4_epoch == checkpoint_epoch` nos runs novos; log `QG4 \| checkpoint_epoch=… (val_auc_pr)` |
| D4 | braço "val_loss" enganoso (focal) | `qg4_result.arms.val_loss` carregando `val_bce_monitor` (C2/C3) | F2: braços rotulados pela métrica real + `decision_rule` explícita | `arms: [val_auc_roc, val_bce_monitor]` no run focal 210707 |
| D5 | `split_policy` estático, manifesto sem hash | texto "record_disjoint (val_ratio=0.1…)" em runs com manifesto pareado; sem `split_manifest_sha256` | F3: policy descreve o manifesto + ratios; hash do manifesto na proveniência | `split_policy: "paired manifest … (80/10/5/5)"` + hash `988d78ae…` verificado |
| D6 | batches manhã × tarde divergentes | explicada (código + runtime diferentes) — sem ação | — | batch canônico: tarde; PRD cita números da manhã (risco registrado) |
| D7 | `.npz` sem IDs (manhã) | histórica — exporter atual inclui IDs | enforcement via inspetor (aviso) | runs novos com IDs completos + overlap ∅ |
| D8 | TensorBoard nunca emitido | config declara, callbacks não instanciam | sem patch (registrado) | — |
| D9 | QG4 `val_loss<0,15` inalcançável | pré-registrada (RFC T9.5) | **fora de escopo** (governança) | thresholds inalterados (pinados por `tests/test_qg4.py`) |

## 5. Correções de código (diff resumido)

- `scripts/pretrain_wrapper.py` (+3): `LEWIS_RUNTIME_PROFILE` no env do subprocesso.
- `src/models/pretrain_chapman.py`: `_resolve_runtime_profile()` (perfil governa o determinismo);
  `_checkpoint_epoch_metrics()` (substitui `_best_epoch_metrics`); QG4 chamado com
  `checkpoint_monitor` + `gate_loss_key`; manifesto gera `split_manifest_sha256`/`split_policy`.
- `src/models/pretrain_provenance.py`: `runtime_env_snapshot()`; `build_provenance` com
  `runtime` + `split_policy`; `write_gate_and_status` com braços rotulados pela métrica real e
  campos `gate_loss_metric`/`checkpoint_monitor`; `write_provenance_and_metrics` propaga tudo.
- `scripts/inspect_training_artifacts.py` (novo, ~500 linhas): inspetor read-only com
  `--run-dir/--experiments-dir/--runs/--cell/--latest/--verify-hashes/--inspect-model/
  --load-model/--inspect-predictions/--compare-runs/--output/--format`; exit 3 em graves.
- Testes: `tests/test_inspect_training_artifacts.py` (17), F1 em `test_pretrain_pipeline.py` (5),
  F2/F3 em `test_pretrain_artifacts.py` (4), `_checkpoint_epoch_metrics` em
  `test_chapman_dataset.py` (4), docstring de `test_qg4.py` atualizada.
- `Makefile` (`pretrain-check` cobre o inspetor) e `AGENTS.md` (nota datada) atualizados.

## 6. Prova de preservação

- Baseline: `preservation_baseline.tsv` — 9.460 arquivos (sha256 do arquivo:
  `7edd921647f6a83df2efec974c58da0f2576b72c26cf4b5e887208ea68a81ec9`).
- Pós: `preservation_after.tsv` — 9.504 arquivos (sha256:
  `9f0ff13126a548d39f155ee1843401990d8c993aab92554a73103cee22f342c4`).
- Diff: **0 arquivos históricos alterados ou removidos**; **+44 arquivos, todos dentro dos dois
  runs smoke novos** (`20260802_210322`, `20260802_210707`).
- Findings do inspetor sobre os 11 runs históricos: **idênticos antes × depois** (issues e
  hash_checks). Nenhum artefato histórico foi reescrito, recalibrado ou reavaliado.

## 7. Verificação

- `flake8` (escopo pretrain-check + inspetor): OK · `mypy` nos módulos alterados: OK.
- `pytest` escopo `pretrain-check` + isolamento de piloto: **115 passed** (1 slow desmarcado).
- Smoke c1 + c2 pós-correção: execução OK, artefatos válidos (`validate_run_dir` strict),
  inspetor exit 0, zero issues.
- Não executado (fora de escopo): `make fw-build` (advertência AGENTS.md), re-treinos completos,
  avaliação em teste, commits (aguardando autorização explícita).

## 8. Critérios de aceite — estado final

- [x] runs inventariados; artefatos críticos localizados; hashes declarados × reais comparados
- [x] `.keras` inspecionado sem edição; nenhum modelo carregado sem `safe_mode=True`
- [x] `.npz` com `allow_pickle=False`; chaves/shapes/dtypes/IDs registrados
- [x] presença/ausência de `test.npz` comprovada por run e atribuída a commit
- [x] linhagem config → checkpoint → predições → QG4 construída
- [x] época do checkpoint confirmada por artefatos; causa do QG4 divergente comprovada e corrigida
- [x] runtime efetivo confirmado por artefatos e registrado fielmente nos novos
- [x] artefatos históricos inalterados (prova sha256 antes × depois)
- [x] `artifacts_audit_before_changes.md` e `artifacts_audit_after_changes.md` produzidos
- [x] `scripts/inspect_training_artifacts.py` + testes (17/17) implementados
- [x] correções limitadas às divergências comprovadas (F1–F3), com 115 testes verdes
