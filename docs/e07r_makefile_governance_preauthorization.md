# E07R + Makefile — preautorização de governança autônoma

- **Data:** 2026-07-26
- **Modo:** `AUTONOMOUS_PREAUTHORIZED`
- **Responsável automatizado:** `AUTONOMOUS_GOVERNANCE_PREAUTH`
- **Intervenção humana exigida nesta simulação:** não
- **Origem:** missão consolidada de remediação científica E07R-PD + refactor do Makefile

## 1. Declaração

Esta execução representa uma simulação realista de governança experimental sem intervenção humana. Este documento registra — sem ampliar — as autorizações antecipadas concedidas pela instrução corrente do usuário para (A) remediação científica do bloqueio fatal de leakage por paciente do E07 e (B) refactor de UX do Makefile. As autorizações **não** cobrem publicação externa, promoção de modelos, afrouxamento de gates nem produção artificial de resultados.

Referências-base:

- Bloqueio E07 por leakage: `docs/e07_blocked_report.md`, `experiments/stage2_v2.4_research/E07_blocked_20260726.json` (SHA-256 `d8a684c0…49f63`).
- Fonte oficial de identidade: <https://physionet.org/physiobank/database/html/mitdbdir/intro.htm> — "Records 201 and 202 came from the same male subject." (`docs/physionet_mitdb_patient_statement.md`).
- Preautorização científica original: `docs/e07r_governance_preauthorization.md`.
- Refactor do Makefile: seção 13 da instrução consolidada (FASE 7).

## 2. Autorizações concedidas

### Científicas

1. Mapeamento autenticado `record_id → patient_id` (201/202 unificados por evidência oficial).
2. Quarentena `QUARANTINED_NOT_DELETED` dos splits record-disjoint.
3. Geração de splits patient-disjoint versionados `v4.0`.
4. Reexecução do E06.5-PD (100 células) e — somente com candidato H*-PD válido — do E07-PD (150 células).
5. Congelamento de H*-PD pela política congelada; relatórios e checkpoints aditivos assinados `AUTONOMOUS_GOVERNANCE_PREAUTH`.

### Integridade

1. Freeze manifest + validação SHA-256.
2. Permissões somente leitura nos pins quando possível; proteção lógica (preflight + guarda de escrita + testes) sempre.
3. Revalidação de testes herdados e adição de testes de regressão.

### Makefile

1. Refactor para clareza: reduzir alvos públicos, padronizar domínios (`data-*`, `mlp-*`, `e07r-*`, `fw-*`, `kb-*`, `rag-*`, `obs-*`, `gates-*`), documentar todos os alvos públicos com `##`, seções `##@` no help.
2. Aliases legados com aviso `DEPRECATED` — nenhum alvo antigo removido sem alias.
3. Mover receitas longas para `scripts/`/CLI.
4. Flags padronizadas: `DRY_RUN=1`, `FORCE=1`, `RUN_ID=...`, `STAGE=...`, `JSON=1`.
5. Atualizar `make help` e documentação de uso.

## 3. Limites (permanecem proibidos)

- publicação externa; promoção/escrita em `models/`; alteração de `backup_v2.3/` ou da quarentena v3.1;
- afrouxamento de QG5' ou do target F1(F) ≥ 0,50;
- uso de teste/validação para fitting de sampler, pesos, thresholds ou seleção;
- remoção/mascaramento de folds problemáticos; cherry-picking; declaração artificial de sucesso;
- sobrescrita de artefatos congelados; alteração de hashes herdados;
- remoção de alvos make antigos sem alias compatível; alteração de comportamento científico durante o refactor.

## 4. Estado científico pré-existente adotado

A remediação científica E07R-PD executada nesta data é adotada como estado válido: mapping autenticado (119 records/76 pacientes), splits `v4.0-patient-disjoint` congelados, freeze de 101 pins com preflight 9/9, E06.5-PD 100/100 células com seleção `NO_VALID_CANDIDATE` (H6 − baseline = −0,1601; IC95 [−0,398; +0,153]) e E07-PD formalmente bloqueado (0/150) conforme o pré-registro. Esta missão formaliza os artefatos de governança/relatórios remanescentes e executa o refactor do Makefile sem alterar nenhum resultado científico.

## 5. Assinatura

```text
responsible = AUTONOMOUS_GOVERNANCE_PREAUTH
governance_mode = AUTONOMOUS_PREAUTHORIZED
date = 2026-07-26
publication_authority = NONE
model_promotion_authority = NONE
gate_relaxation_authority = NONE
```
