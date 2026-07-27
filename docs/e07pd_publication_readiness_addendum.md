# E07R/E07-PD — adendo de prontidão de publicação

**Data:** 2026-07-26 | **Decisão:** `HOLD — publication_ready=false`

## 1. Estado dos gates de publicação

| Gate | Threshold | Resultado | Estado |
| --- | --- | --- | --- |
| Target principal F1(F) | ≥ 0,50 | baseline 0,6325 (OOF PD); H6 0,4724 | baseline supera, mas é baseline — não há candidato de representação |
| QG5' F1(F) | ≥ 0,15 | H6 0,4724 | OK em métrica, falho em ganho robusto |
| Ganho robusto H6 vs baseline | IC95 exclui 0, Δ > 0 | Δ = −0,1601; IC95 [−0,398; +0,153] | **NOT MET** |
| E07-PD sampling | evidência de braço válido | 0/150 células | NOT RUN |
| Integridade | preflight 9/9 | PASS | OK |
| models/ | intocado | true | OK |

## 2. Declaração

Mesmo com F1(F) do baseline acima de 0,50 na avaliação OOF patient-disjoint, **não há candidato de representação válido** e nenhuma evidência de que sampling melhore o estado da arte (E07-PD não executado). A publicação permanece bloqueada; nenhum artefato foi promovido; nenhum gate foi afrouxado.

## 3. Fatos que sustentam o HOLD

1. O ranking record-era (H6 > baseline) **não se reproduz** patient-disjoint — evidência de que o ganho anterior era inflado por leakage de paciente.
2. H6 regride F1(F) em −0,1601 vs baseline (IC95 inclui zero e ponto negativo).
3. E07-PD bloqueado por pré-registro — sampling não avaliado; qualquer afirmação sobre sampling seria artificial.
4. Fold 5 classificado `PARTIALLY_RECOVERABLE` com evidência (`docs/e07pd_fold5_diagnostics.md`).

## 4. Próximos passos possíveis (fora desta missão)

- Nova hipótese de representação com ganho robusto patient-disjoint → novo freeze + pré-registro.
- Ampliação de identidade de pacientes (SVDB permanece `IDENTITY_UNVERIFIED`).
- `next_authorized_stage = NONE` — nenhuma etapa de publicação autorizada.
