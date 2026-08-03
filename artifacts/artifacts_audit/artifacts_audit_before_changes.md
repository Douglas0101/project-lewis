# Auditoria de artefatos — ANTES das alterações

**Data:** 2026-08-02 · **Repo:** `/home/douglas-souza/PycharmProjects/Project-Lewis`
**Branch:** `develop` @ `db827d05f924e7411e1373617c3fad9b7e122de7` · **Modo:** somente leitura sobre `experiments/`
**Método:** inspeção programática ad-hoc (scripts em `/tmp`, stdlib + numpy; `.keras` como ZIP; `.npz` com `allow_pickle=False`; hashes SHA-256 em streaming). Nenhum artefato histórico foi modificado.

---

## 1. Inventário

- **221 diretórios de run** em `experiments/` (9.460 arquivos, 1,3 GB). Baseline de preservação:
  `artifacts/artifacts_audit/preservation_baseline.tsv` (9.460 linhas; sha256 do arquivo:
  `7edd921647f6a83df2efec974c58da0f2576b72c26cf4b5e887208ea68a81ec9`).
- **Distribuição:** `finetune` ×83 · `pretrain_chapman` legado (sem `pilot_status`) ×48 ·
  `stage2_research` ×35 · `outros` ×28 · `groupkfold` ×16 · **piloto T10.3 ×11**
  (c0×2, c1×2 + 3 smokes, c2×2, c3×2).
- **JSONs corrompidos:** 0 (config/pilot_status/provenance/qg4/run_status validados em todos os 221 runs).
- **TensorBoard:** nenhum `events.out.tfevents.*` em todo `experiments/` (ver D8).
- **Runs sem `qg4_result.json`/`run_status.json`:** 183 (anteriores ao contrato 10.4/10.6 — esperado; inclui A2-full).
- **`environment.json` / `checkpoint_metadata.json`:** inexistem em todos os runs (schema não implementado — registrado, não é erro isolado de um run).

### Runs selecionados

| Run | Papel | Célula | Git | oneDNN | `deterministic_mode` registrado |
|---|---|---|---|---|---|
| `20260802_161023` | smoke mais recente | c1 (smoke) | `db827d0` | true | strict |
| `20260802_033236` | smoke c/ `test.npz` | c1 (smoke) | `4c88def` | false | strict |
| `20260802_035117` | C0 manhã (PRD) | c0 | `489993e` | false | strict |
| `20260802_161312` | C0 tarde (HEAD) | c0 | `db827d0` | true | strict |
| `20260802_043748` | C1 manhã (PRD) | c1 | `489993e` | false | strict |
| `20260802_163129` | C1 tarde (HEAD) | c1 | `db827d0` | true | strict |
| `20260802_051138` | C2 manhã (PRD) | c2 | `489993e` | false | strict |
| `20260802_165458` | C2 tarde (HEAD) | c2 | `db827d0` | true | strict |
| `20260802_054320` | C3 manhã (PRD) | c3 | `489993e` | false | strict |
| `20260802_171757` | C3 tarde (HEAD) | c3 | `db827d0` | true | strict |
| `20260728_053011` | A2-full histórico (referência T9.3, congelado) | — | `48931a7` | true | strict |

Linha do tempo dos commits: `e427b17` (orquestrador+exporter, **sem** trava de teste) → `4c88def` →
`489993e` (T10.3-ENABLE) → `04b7c66` (perfis runtime strict/fast) → `2cdb3b1` (**trava de teste**,
repeat dos datasets pareados, gates em validation) → `db827d0` (HEAD; gate off em smoke).

## 2. Matriz dos runs

| Run | Cél. | Arch | Loss | Seed | Perfil solicitado | Épocas (conc/solic) | Época checkpoint (argmax `val_auc_pr`) | Época QG4 | QG4 | macro PR-AUC | macro AUROC | ECE pós | T | Protocolo | Teste |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `035117` | c0 | a0 | bce | 13 | n/a (pré-perfil) | 30/30 | 29 | 29 | FAIL | 0,6458 | 0,8241 | 0,0159 | 0,939 | PROSP. | **materializado** |
| `161312` | c0 | a0 | bce | 13 | fast | 25/30 (ES) | **20** | **25** | FAIL | 0,6595 | 0,8247 | 0,0184 | 1,001 | PROSP. | ausente ✓ |
| `043748` | c1 | a1 | bce | 13 | n/a | 30/30 | 29 | 29 | FAIL | 0,6749 | 0,8502 | 0,0187 | 0,915 | PROSP. | **materializado** |
| `163129` | c1 | a1 | bce | 13 | fast | 30/30 | **26** | **28** | FAIL | 0,6956 | 0,8551 | 0,0190 | 0,961 | PROSP. | ausente ✓ |
| `051138` | c2 | a1 | focal | 13 | n/a | 30/30 | 29 | 16 | FAIL | 0,6748 | 0,8521 | 0,0217 | 0,340 | PROSP. | **materializado** |
| `165458` | c2 | a1 | focal | 13 | fast | 30/30 | **30** | **29** | FAIL | 0,6964 | 0,8592 | 0,0222 | 0,358 | PROSP. | ausente ✓ |
| `054320` | c3 | a0 | focal | 13 | n/a | 30/30 | 29 | 16 | FAIL | 0,6507 | 0,8280 | 0,0149 | 0,344 | PROSP. | **materializado** |
| `171757` | c3 | a0 | focal | 13 | fast | 30/30 | **30** | **13** | FAIL | 0,6723 | 0,8345 | 0,0185 | 0,363 | PROSP. | ausente ✓ |
| `053011` | A2-full | a1 | focal | 13 | n/a | 30/30 | 29 | 18 | (sem QG4) | 0,7065 | 0,8639 | 0,0152 | 0,374 | RETRO. | n/a |

Época do checkpoint inferida por matching das métricas de `evaluation_v2` com `history.json`
(diferença ≤ 2×10⁻⁴ em macro PR-AUC/AUROC vs a época argmax `val_auc_pr` em 4/4 runs da tarde) +
código (§4, D3). Todos os runs pilotos: split `chapman-record-disjoint-paired-v2`, ES por
`val_auc_pr`, `status=PILOT`, `promotion_eligible=false` (QG4 FAIL + sem `--promote`).

## 3. Linhagem

Cadeia por run (ex. `20260802_161312`, C0 tarde) — **todos os hashes declarados conferem com os calculados**:

```text
data/splits/chapman_paired_v2/manifest.json  sha256=988d78ae…  (referenciado só por NOME: split_id)
  → config.json            sha256=ff7bffdb… == provenance.hashes.config_sha256 ✓
  → history.json           sha256=9586cff6… == provenance.hashes.history_sha256 ✓
  → backbone_pretrained.keras sha256=30595678… == provenance.hashes.model_sha256
                                               == validation_meta.sha256_model ✓
  → metrics_per_class.json sha256=88536311… == provenance.hashes.metrics_per_class_sha256 ✓
  → evaluation_v2/predictions/{validation,calibration}.npz  (exportadas pós-treino pelo orquestrador;
                                               sem hash registrado na proveniência)
  → evaluation_v2/{metrics,metrics_per_class,calibration,thresholds,reliability,
                   confidence_intervals,reconciliation}.json
  → qg4_result.json / run_status.json (terminais; hash não registrado em lugar nenhum)
  → pilot_status.json (orquestrador)
```

**Lacunas de linhagem:** (i) hash do manifesto do split não é registrado em nenhum artefato do run
(vínculo apenas por `split_id`); (ii) artefatos `evaluation_v2/` não são cobertos por hash na
proveniência; (iii) runs do batch da manhã têm `validation_meta.json` sem `n_records`/
`segments_per_record` e `.npz` sem IDs (schema pré-`2cdb3b1`).

**Órfãos/anomalias:** `test.npz` sem `model_freeze.json` em 5 runs (D1); 3 runs com
`predictions.npz` de schema legado (`20260728_033533`, `20260728_053011` A2-full, `20260729_042301`);
183 runs pré-contrato sem QG4/run_status (esperado); A2-full com avaliação RETROSPECTIVE e thresholds
fit em `"evaluation"` (protocolo anterior — diferença registrada, não erro).

## 4. Divergências confirmadas

### D1 — `test.npz` materializado antes do freeze (violação RF-DATA-005) — HISTÓRICA, já corrigida no código
- **Artefatos:** `evaluation_v2/predictions/test.npz` presente em `20260802_033236` (smoke), `035117`,
  `043748`, `051138`, `054320` (todo o batch da manhã) — nenhum com `model_freeze.json`;
  `test_status: "locked_until_model_freeze"` declarado no `pilot_status.json` do próprio batch posterior.
- **Código/git:** trava `is_test_authorized()` introduzida em `2cdb3b1`
  (`scripts/export_pilot_predictions.py:90`); os 5 runs são de `4c88def`/`489993e`, anteriores.
- **Estado:** código atual (`db827d0`) bloqueia; batch da tarde não tem `test.npz`. ✔ resolvida
  (evidência histórica preservada; enforcement automatizado passa ao script inspetor — F4).

### D2 — Runtime solicitado × runtime registrado/efetivo — ATIVA
- **Artefatos:** `pilot_status.json` (tarde) registra `runtime_profile: "fast"`; `provenance.json`
  dos MESMOS runs registra `deterministic_mode: "strict"` com `onednn_enabled: true`. O perfil
  `configs/runtime/strict.yaml` exige `onednn: false` — ou o rótulo "strict" está errado, ou o
  oneDNN está. Os dois canais coexistem: wrapper `--runtime-profile fast` só ajusta **env**
  (`scripts/pretrain_wrapper.py:132-142`); o subprocesso de treino lê `deterministic.mode: strict`
  de `config/pretrain_v1.0.yaml:32-33` e chama `apply_deterministic_mode("strict")`
  (`src/models/pretrain_chapman.py:492-496`), gravando `"strict"` na proveniência
  (`src/models/pretrain_provenance.py:144`). O perfil solicitado nunca chega ao treino.
- **Consequência:** proveniência afirma determinismo estrito que não vigorava (oneDNN ativo,
  `TF_DETERMINISTIC_OPS` ausente no perfil fast); batch manhã (realmente strict: oneDNN off) ×
  tarde (misto) produziram métricas diferentes para mesma célula/seed/split (C0 0,6458×0,6595).
- **Causa-raiz:** dois canais de configuração de runtime não reconciliados; ausência de
  `environment_report()` (RF-PROV-003, existe em `src/runtime/cpu_policy.py:95` mas não é usado).
- **Confiança:** alta. **Severidade:** média-alta (proveniência incorreta; reprodutibilidade não auditável).

### D3 — Checkpoint salvo × época julgada pelo QG4 — ATIVA (principal achado)
- **Código:** `EarlyStopping` **e** `ModelCheckpoint` usam o MESMO `monitor=early_stopping_metric`
  (`src/models/pretrain_chapman.py:55-77`; pilotos: `val_auc_pr`, mode=max) → o
  `backbone_pretrained.keras` salvo contém os pesos da época **argmax `val_auc_pr`**. O QG4, porém,
  seleciona a época **argmin `val_loss`/`val_bce_monitor`** (`_best_epoch_metrics`,
  `pretrain_chapman.py:150-167,598-599`), apesar do comentário em `:153` afirmar que "QG4 must judge
  the best checkpoint (restored by EarlyStopping and saved by ModelCheckpoint)".
- **Artefatos (prova por matching):** métricas de `evaluation_v2` (modelo efetivamente avaliado)
  batem com `history.json` na época argmax `val_auc_pr`, não na época QG4:
  C0-tarde avaliado=ép.20 vs QG4=ép.25 · C1-tarde 26 vs 28 · C2-tarde 30 vs 29 · C3-tarde 30 vs 13.
- **Impacto:** `qg4_result.json`/`run_status.json`/`provenance.metrics` reportam métricas de uma
  época que NÃO é a do artefato implantável. Caso material: C2-manhã — QG4 reporta
  `val_auc_roc=0,8459` (braço FAIL) na época 16, mas o checkpoint avaliado (ép.29) tem
  `val_auc_roc=0,8576` (braço PASS). `metrics_per_class.json` (legado) é calculado com o modelo em
  memória (pesos restaurados pelo ES = argmax `val_auc_pr`) — internamente inconsistente com
  `provenance.metrics`. `known_issues` não menciona o descompasso de época.
- **Causa-raiz:** critério de seleção de "melhor época" do QG4 desalinhado do monitor de
  checkpoint introduzido pelo protocolo v2 (`--early-stopping-metric val_auc_pr`).
- **Confiança:** alta (artefatos + código). **Severidade:** alta (gate julga artefato errado).

### D4 — Rótulo da métrica de perda do QG4 em runs focais — ATIVA
- **Artefatos:** `qg4_result.json` de C2/C3 rotula `"metric": "val_loss"` com
  `decision_rule: "… at best epoch (min val_loss)"`, mas o valor observado é o `val_bce_monitor`
  (ex.: C3-tarde observado 0,4683 == `val_bce_monitor` ép.13; o `val_loss` focal nessa época é 0,1065).
- **Código:** `write_gate_and_status` nomeia os braços literalmente `"val_loss"`
  (`src/models/pretrain_provenance.py:222-239`); `_best_epoch_metrics` devolve o valor do
  `bce_monitor` sob a chave `"val_loss"`; a ressalva existe só em `run_status.known_issues`
  (`pretrain_chapman.py:647`), ausente do `qg4_result.json`.
- **Confiança:** alta. **Severidade:** média (artefato enganoso para o leitor; decisão em si é a pretendida).

### D5 — `split_policy` estático e manifesto sem hash na proveniência — ATIVA
- **Artefatos:** `provenance.dataset.split_policy` = `"record_disjoint (val_ratio=0.1, seeded shuffle)"`
  em TODOS os runs pilotos, embora o split efetivo seja o manifesto pareado v2
  (44.986 registros; 80/10/5/5; `n_records` bate: 35.989/4.499/2.249/2.249).
- **Código:** texto hardcoded em `src/models/pretrain_provenance.py:150`; `main()` tem o manifesto
  em mãos (`pretrain_chapman.py:519-533`) mas não propaga hash/ratios.
- **Confiança:** alta. **Severidade:** baixa-média (linhagem do split só por nome).

### D6 — Dois batches, mesma célula/seed/split, métricas diferentes — EXPLICADA
Batch manhã (`489993e`, env realmente strict, exportação pré-repeat/sem IDs, com `test.npz`) ×
batch tarde (`db827d0`, env fast + API op-determinism, exportação com IDs/repeat, sem `test.npz`).
`reconciliation.json` da manhã mostra deltas legacy×v2 ≠ 0; tarde = 0,0 (exporter convergiu).
**Risco registrado:** o documento `docs/PRD + SDD — Otimização CPU-First…md` cita os números do
batch da MANHÃ (C1 0,6749 etc.), produzidos com a exportação pré-`2cdb3b1` e com teste materializado.
Batch canônico para comparações futuras: **tarde** (HEAD, schema atual, teste ausente).

### D7 — `.npz` do batch da manhã sem IDs — HISTÓRICA, corrigida no exporter atual
`validation/calibration/test.npz` da manhã têm apenas `y_score,y_true` → impossível verificar
sobreposição e 10 segmentos/registro nesses runs (**limitação declarada**). Batch tarde:
`patient_ids/record_ids/segment_ids` presentes, 4.499/2.249 registros × exatamente 10 segmentos,
`record_ids` validation∩calibration = ∅. ✔ resolvida em `2cdb3b1`.

### D8 — TensorBoard configurado mas nunca emitido — MENOR
`config/pretrain_v1.0.yaml:49-50` declara `tensorboard.log_dir`, mas `_make_callbacks`
(`pretrain_chapman.py:56-83`) não instancia o callback. Nenhum `tfevents` em `experiments/`.
Ausência registrada (o protocolo não a considera erro).

### D9 — QG4 `val_loss < 0,15` inalcançável — PRÉ-REGISTRADA (fora de escopo)
Todos os pilotos FAIL no braço de perda (observado 0,35–0,47). Documentado como decisão pendente
da RFC T9.5 (`docs/t10_3_pilot_execution_plan.md` §5/§10). **Não é alvo de correção nesta tarefa.**

## 5. Diagnóstico por alerta (síntese)

| Alerta | Evidência nos artefatos | Evidência no código | Causa-raiz | Confiança |
|---|---|---|---|---|
| Teste acessado pré-freeze | `test.npz` ×5 runs, sem `model_freeze.json` | guard só a partir de `2cdb3b1` | código anterior à trava | alta |
| Runtime "strict" falso | `deterministic_mode` vs `onednn_enabled` vs `runtime_profile` | dois canais (yaml × env); `environment_report` não usado | perfil não propagado ao treino | alta |
| QG4 julga época errada | eval_v2 == argmax `val_auc_pr`; QG4 == argmin perda | MC/ES=`val_auc_pr` vs `_best_epoch_metrics` | seleção de época do gate desalinhada do checkpoint | alta |
| Braço "val_loss" enganoso (focal) | observado == `val_bce_monitor`, rótulo "val_loss" | braços/decision_rule hardcoded | rotulagem estática | alta |
| Split policy estático | texto idêntico em runs com manifesto | hardcoded `pretrain_provenance.py:150` | manifesto não propagado | alta |
| PRD cita batch pré-fix | números do PRD == batch manhã | — | citação anterior ao `2cdb3b1` | alta |

## 6. Isolamento do teste e validações de predição (batch tarde — canônico)

- `test.npz` **ausente** nos 4 pilotos + smoke recente ✔; `evaluation_split: "validation"`;
  T/thresholds fit na **calibration** (`thresholds.json.fit_split`).
- Cardinalidade: validation 44.990 = 4.499×10 ✔ · calibration 22.490 = 2.249×10 ✔;
  segmentos/registro = 10 exatos (min/med/max); `record_ids` validation∩calibration = ∅ ✔.
- Sem NaN/Inf; `y_score ∈ [0,1]`; `y_true` binário; shapes/dtypes consistentes (`float32`, 5 classes).
- Prevalências validation: NORM 0,7488 · CD 0,1616 · MI 0,2927 · HYP 0,2194 · STTC 0,2732.

## 7. Verificações íntegras (sem divergência)

- Hashes declarados == calculados (modelo/config/history/metrics_per_class/`validation_meta`) nos 11 runs.
- `config` × `provenance` (seed/arch/loss) ✔ · `qg4_result.pass` == `run_status.qg4.pass` ==
  `provenance.qg4.pass` ✔ · `execution_success=true` com QG4 FAIL (política SDD DEF-010) ✔.
- `.keras` = ZIP Keras v3 válido (`metadata.json`/`config.json`/`model.weights.h5`) em todos;
  nenhum modelo carregado durante a auditoria (inspeção somente-ZIP).
- **C1 × C2 (per-class, batch tarde):** focal NÃO melhora classes raras — Δ PR-AUC: NORM +0,0006,
  CD ±0,0000, MI −0,0020, HYP +0,0067, STTC −0,0010 (macro +0,0009). Efeito dominante da focal é o
  perfil de calibração (ECE pré 0,16 vs 0,019; T≈0,36 vs 0,96), corrigido pela temperature
  (ECE pós 0,022 vs 0,019). Sem regressão grave; thresholds estáveis (0,23–0,60). A2-full segue
  à frente (macro 0,7065, CD 0,5561) — lacuna de capacidade/dados, não de loss.

## 8. Limitações declaradas

1. Batch da manhã sem IDs nos `.npz` → sobreposição e segmentação não verificáveis nesses runs.
2. Época do checkpoint inferida por matching de métricas (≤2×10⁻⁴) + leitura de código;
   o `.keras` não carrega metadado de época.
3. 183 runs legados sem contrato QG4/run_status não foram inspecionados individualmente (fora do escopo T10.3).
4. Vínculo do split é por nome (`split_id`); sem hash de manifesto nos artefatos (D5).
5. Determinismo bit-a-bit intra-batch não foi re-executado (re-treinos estão fora do modo read-only).

## 9. Correções propostas (derivadas das evidências acima)

| Fix | Divergência | Mudança | Arquivos |
|---|---|---|---|
| F1 | D2 | Propagar perfil de runtime ao subprocesso de treino (env `LEWIS_RUNTIME_PROFILE`); treino usa o perfil para o modo determinístico; proveniência registra perfil solicitado + ambiente efetivo (oneDNN/det-ops/threads) | `scripts/pretrain_wrapper.py`, `src/models/pretrain_chapman.py`, `src/models/pretrain_provenance.py` |
| F2 | D3+D4 | QG4 julga a época do checkpoint salvo (monitor ES/MC), com braços rotulados pela métrica real (`val_bce_monitor` em focal) + `decision_rule`/`known_issues` coerentes; `provenance.metrics` na mesma época | `src/models/pretrain_chapman.py`, `src/models/pretrain_provenance.py` |
| F3 | D5 | `provenance.dataset` reflete o split efetivo (split_id + ratios do manifesto) + `hashes.split_manifest_sha256` | `src/models/pretrain_chapman.py`, `src/models/pretrain_provenance.py` |
| F4 | D1/D7 | Sem patch (trava `2cdb3b1` vigente; IDs já exportados) — enforcement passa ao inspetor: exit≠0 se `test.npz` sem freeze; flag para `.npz` sem IDs | `scripts/inspect_training_artifacts.py` (novo) |
| F5 | D8 | Sem patch (registrado; TensorBoard opcional) | — |

Fora de escopo: thresholds do QG4 (RFC T9.5 — governança); alteração de qualquer artefato histórico.

## 10. Critérios de aceite (estado pré-alteração)

- [x] runs inventariados (221) · artefatos críticos localizados · hashes declarados × reais comparados
- [x] `.keras` inspecionado sem edição · nenhum modelo carregado (somente ZIP)
- [x] `.npz` com `allow_pickle=False` · chaves/shapes/dtypes/IDs registrados
- [x] presença/ausência de `test.npz` comprovada por run e atribuída a commit
- [x] linhagem config → checkpoint → predições → QG4 construída
- [x] época do checkpoint confirmada por artefatos · causa do QG4 divergente comprovada
- [x] runtime efetivo confirmado por artefatos · relatório pré-alteração produzido
