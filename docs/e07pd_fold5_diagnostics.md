# E07-PD — diagnóstico do fold 5 (base E06.5-PD)

**Data:** 2026-07-26 | **Classificação:** `PARTIALLY_RECOVERABLE` | **Base de evidência:** E06.5-PD (100 células); E07-PD não executado

## 1. Estrutura do fold 5 (v4.0-patient-disjoint)

- Teste: 18 pacientes / 28 records / 6.925 amostras; **F: 95 beats de 7 pacientes**; S: 1.399; V: 5.431.
- Comparativo: folds 1–2 concentram 373–374 beats F de apenas 2–3 pacientes (escopo 208/213); folds 3–5 dispersam 86–95 beats F em 7–9 pacientes.

## 2. F1(F) no fold 5 por seed

| Candidato | s17 | s29 | s43 | s71 | s101 | média |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0,041 | 0,041 | 0,041 | 0,041 | 0,041 | 0,041 |
| H6 | 0,038 | 0,103 | 0,064 | 0,109 | 0,053 | 0,073 |
| H11 | 0,020 | 0,000 | 0,019 | 0,000 | 0,000 | 0,008 |
| H12 | 0,019 | 0,000 | 0,019 | 0,000 | 0,000 | 0,008 |

Zero-F1 runs no fold 5: baseline 0/5; H6 0/5; H11 3/5; H12 2/5.

## 3. Leitura

- O fold 5 é difícil para **todos** os candidatos, inclusive o baseline (0,041) — não é colapso específico de uma representação.
- H6 (0,073) supera baseline no fold 5, mas com suporte baixo (95 F) e margens apertadas — evidência insuficiente para reverter o veredito global (Δ OOF −0,1601).
- A dispersão de F em 7 pacientes (vs 2–3 nos folds 1–2) sugere componente de diversidade morfológica/OOD parcial, não apenas escassez.

## 4. Classificação

```text
PARTIALLY_RECOVERABLE
```

Justificativa: há sinal não-zero em todos os braços do E06.5-PD (baseline/H6) e suporte F real (95 beats, 7 pacientes) — não é `INSUFFICIENT_DATA` nem `STRUCTURALLY_OOD` absoluto; mas nenhuma representação atual recupera F1(F) ≥ 0,15 (QG5') neste fold. O fold 5 **permanece na matriz** (nunca removido/mascarado); sua dificuldade é registrada como estrutural com evidência.

## 5. Pendente (depende de E07-PD)

F1(F) por braço S0–S5, margens de decisão F, taxa de margens negativas e confiança em erros F no fold 5: **não computados** — E07-PD bloqueado por ausência de candidato H*-PD (`docs/e07pd_sampling_results.md`).
