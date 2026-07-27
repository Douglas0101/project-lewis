# E07R + Makefile — plano de execução consolidado

- **Data:** 2026-07-26
- **Governança:** `AUTONOMOUS_PREAUTHORIZED` (`docs/e07r_makefile_governance_preauthorization.md`)
- **Regra de execução:** integridade > ausência de leakage > preservação > validade > clareza operacional > estética

## 1. Contexto científico

E07 estava `BLOCKED` por leakage de paciente (MIT-BIH 201/202 no mesmo indivíduo, comprovado pela fonte oficial PhysioNet). A remediação E07R-PD já executada produziu: mapping autenticado, Stage 2 r5 com custódia ordenada, splits `v4.0-patient-disjoint` (5 outer + 20 inner, zero overlap), quarentena documental dos splits legados, freeze de 101 pins com preflight fail-closed, E06.5-PD 100/100 células com seleção `NO_VALID_CANDIDATE` (H6 não supera baseline: −0,1601; IC95 [−0,398; +0,153]) e bloqueio formal do E07-PD (0/150) exigido pelo pré-registro. Publicação permanece `HOLD`.

## 2. Contexto de build/Makefile

O Makefile tinha 89 alvos, com nomenclatura inconsistente, alvos auxiliares expostos, receitas longas embutidas e `make help` sem seções. A FASE 7 padroniza domínios, reduz o help aos alvos públicos canônicos, cria aliases `DEPRECATED` (nenhum alvo removido), move lógica longa para scripts e introduz flags padronizadas.

## 3. Ordem de execução

1. **FASE 0** — governança, manifesto de preautorização, snapshot lógico. *(esta fase)*
2. **FASE 1** — *(concluída)* freeze manifest, 101 pins, proteção (0444 + guarda lógica), preflight 9/9, `docs/e07r_integrity_report.md` (formalização aqui).
3. **FASE 2** — *(concluída)* mapping `record_id → patient_id` autenticado.
4. **FASE 3** — *(concluída)* splits v4.0 + leakage report + quarentena; formalização do relatório.
5. **FASE 4** — *(concluída)* E06.5-PD 100/100; seleção `NO_VALID_CANDIDATE`; relatório formal.
6. **FASE 5** — *(bloqueio formal registrado)* E07-PD 0/150 por ausência de candidato H*-PD; relatórios formais com classificação honesta.
7. **FASE 6** — relatórios e checkpoints científicos consolidados.
8. **FASE 7** — refactor do Makefile + validação integral.

## 4. Congelamentos em vigor

- `e07r_freeze_manifest.json` (`ba1c4aa1…`), 101 pins, preflight 9/9 PASS.
- r5 `v3.1.0-r5-stage2-pd`, splits v4.0, mapping v4.0, quarentena v2.3-era, `models/`, `backup_v2.3/`, quarentena v3.1.
- Makefile **não** é pin científico (fora de `PD_SOURCE_FILES`): o refactor não invalida hashes nem exige re-freeze — verificado por preflight pós-refactor.

## 5. Critérios de aceite

### Científicos (já satisfeitos; revalidar ao final)

- zero overlap patient/record outer+inner; 201/202 juntos; SVDB fora do confirmatório;
- `models_untouched=true`, `gates_relaxed=false`, `publication_ready=false`;
- preflight final 9/9 PASS; suíte e focados verdes.

### Makefile

- todo alvo público com `##`; help por seções `##@`;
- nenhum alvo interno/alias no help;
- todos os ~31 alvos legados funcionam via alias com aviso `DEPRECATED`;
- flags `DRY_RUN/FORCE/RUN_ID/STAGE/JSON` funcionais nos alvos E07R;
- comportamento idêntico: `make -n` dos alvos legados expande para as mesmas receitas canônicas;
- `make lint`, focados E07R e preflight verdes após o refactor.

## 6. Riscos e mitigações

| Risco | Mitigação |
| --- | --- |
| Alias alterar comportamento | Aliases chamam o alvo canônico via sub-make; validação `make -n` + execução read-only |
| Flags mudarem semântica dos treinos | `DRY_RUN/FORCE` só controlam `--dry-run`/arquivamento+run-id; célula intocada |
| Help estourar 50 alvos | Aliases/auxiliares sem `##` ficam ocultos; apenas os 58 canônicos documentados (lista explícita da missão) |
| Refactor tocar artefato crítico | Nenhuma receita nova escreve em paths pinados; preflight pós-refactor prova integridade |

## 7. Checklist de integridade

- [x] hashes herdados válidos (stage2 npz/parquet, finetuning npz/parquet, checkpoint BLOCKED)
- [x] models/ intocado; backup_v2.3/ intacto; quarentena v3.1 preservada
- [x] splits legados quarentenados e inativos
- [x] splits ativos patient-disjoint (outer+inner)
- [x] preflight fail-closed publicado e verde
- [ ] preflight pós-refactor verde (revalidar)
- [ ] `new_artifact_hashes` do manifesto preenchido ao final

## 8. Plano de refactor (resumo)

- 58 alvos públicos canônicos com `##` nas seções `##@`: Geral, Dados, MLP, E07R, Firmware/Gates, Knowledge/RAG, Observabilidade.
- ~31 aliases legados ocultos com aviso `DEPRECATED` (sub-make ao canônico).
- Alvos auxiliares internos (`mlp-logs-dir`, `format`, `type-check`, `docker-*`, `stress-test-p1..3`, etc.) permanecem funcionais, fora do help.
- Flags: `DRY_RUN=1` (dry-run), `FORCE=1` (fresh com arquivamento), `RUN_ID=...` (override por alvo), `STAGE=e065|e07` (watch), `JSON=1` (saída JSON onde aplicável).

## 9. Plano de validação

1. `make help` renderiza por seção; contagem de públicos conferida.
2. `make -n` de cada alias legado → expansão para o canônico.
3. Execução real read-only: `make check`, `make e07r-status`, `make e07r-e065 DRY_RUN=1`, `make e07r-e07` (BLOCKED tolerado), `make e07r-watch EXTRA="--once"`.
4. `make lint`, pyright, focados E07R (19 testes), preflight FREEZE_VALIDATION.
5. Não executar `e07r-e065 FORCE=1` real nem `e07r-e065-fresh` (re-treino) — validados estaticamente.
