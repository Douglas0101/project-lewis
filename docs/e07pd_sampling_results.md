# E07-PD — resultados da auditoria de sampling (FASE 5)

**Data:** 2026-07-26 | **Status:** `BLOCKED — NOT_RUN (0/150 células)`

## 1. Decisão

A auditoria E07-PD de sampling **não foi executada**. O pré-registro exige um candidato H*-PD válido como entrada; o E06.5-PD terminou com `NO_VALID_CANDIDATE` (H6 não supera baseline de forma robusta: ΔF1(F) = −0,1601; IC95 [−0,3983; +0,1534]). O workflow fail-closed bloqueou formalmente:

```text
$ make e07r-e07
{
  "error_type": "E07RIntegrityError",
  "reason": "E07-PD blocked: no valid H*-PD candidate",
  "status": "BLOCKED"
}   # exit 10
```

Checkpoint: `experiments/stage2_v2.4_research/E07_PD_blocked_20260726.json`.

## 2. Braços pré-registrados (não executados)

| Braço | Semântica | Células |
| --- | --- | ---: |
| S0 | dados naturais (controle) | 0/25 |
| S1 | slots iguais por classe até `max(class_count_train)` | 0/25 |
| S2 | bootstrap por patient ID + linha | 0/25 |
| S3 | bootstrap ponderado 4,0×F | 0/25 |
| S4 | slots por classe + patient uniforme | 0/25 |
| S5 | bootstrap uniforme (controle de orçamento) | 0/25 |

## 3. Conformidade

- Nenhum sampler foi ajustado com dados de teste/validação (nenhum sampler foi executado).
- Nenhuma métrica E07-PD foi produzida — nenhum resultado artificial.
- Threshold, representação e gates inalterados.
- A interpretação pré-registrada se aplica: ausência de candidato ⇒ evidência insuficiente de representação; sampling não pode ser avaliado validamente sobre uma base inválida.

## 4. O que desbloquearia (fora desta missão)

Nova evidência de representação (novo candidato/método) com ganho robusto sobre baseline em splits patient-disjoint, seguida de novo freeze e pré-registro. Isso é trabalho futuro de pesquisa, não de build.
