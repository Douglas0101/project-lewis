# 01 — Decisão de Ontologia Clínica (D2, D3, D4) e Mapeamento Dataset–Doença

**Status:** PROPOSTO — aguardando ratificação humana
**Data:** 2026-07-18
**Entregáveis:** `clinical_ontology_decision` (1), `dataset_disease_mapping` (2)

---

## 1. Princípio

Quatro níveis semânticos **obrigatoriamente separados**. Nenhum nível pode ser apresentado como
outro. Proibições duras (do prompt mestre e dos achados DQ-03/DQ-05/DQ-12):

- `F` não pode significar simultaneamente fusão de batimentos e fibrilação atrial;
- uma classe de batimento não é diagnóstico de doença;
- ritmo AFIB não pode ser inferido de um único batimento;
- Q não entra no Stage 1 sem destino explícito;
- "Anormal" não é uma doença homogênea;
- nenhum mapeamento desconhecido cai automaticamente em Q ou Anormal (encerra a política atual de
  `aami_mapper.py:96`, que manda desconhecidos → Q).

## 2. Nível 1 — Qualidade do sinal (`quality`, escopo: janela/batimento)

```text
VALID | NOISY | CLIPPED | FLATLINE | LEAD_MISSING | RPEAK_UNCERTAIN | OUT_OF_DISTRIBUTION
```

Origem: medidas objetivas do pipeline (std da janela, saturação, SNR estimado, flag do detector).
Semântica: gate de entrada; qualidade insuficiente → abstenção, nunca classificação clínica.
Baseline medido na auditoria: flatline 0,52% global / 1,38% INCART; clip ±10σ em 0,14% do svdb.

## 3. Nível 2 — Morfologia do batimento (`beat`, escopo: batimento)

```text
N | S | V | FUSION | Q_OR_UNKNOWN
```

Mapeamento AAMI EC57 a partir de anotações beat-level (mitdb, svdb, incart):

| Símbolo WFDB | Classe canônica | Observação |
|---|---|---|
| `N`, `L`, `R`, `e`, `j` | `N` | normal, BBE/BBD, escapes |
| `A`, `a`, `J`, `S` | `S` | ectopia supraventricular |
| `V`, `E` | `V` | ectopia ventricular |
| `F` | `FUSION` | **somente fusão V+N; nunca fibrilação atrial** |
| `/`, `f`, `Q` | `Q_OR_UNKNOWN` | paced, fusão paced, não-classificável — classe de rejeição (ver D4) |
| `\|`, `x`, `~`, `+` e qualquer outro | **não mapeado** | excluído com registro; `reviewStatus=excluded`; nunca → Q |

## 4. Nível 3 — Ritmo (`rhythm`, escopo: episódio/registro)

```text
SINUS | AFIB | AFL | JUNCTIONAL | OTHER_RHYTHM | UNKNOWN_RHYTHM
```

- Somente inferível com contexto temporal (episódio); janela de 1 batimento retorna
  `INSUFFICIENT_TEMPORAL_CONTEXT`.
- Fonte: AFDB (`.atr` com anotações de ritmo `(AFIB`, `(AFL`, `(N`, `(J`) via
  `src/features/afdb_beat_loader.py` (existe, não integrado); ritmo sinusal nos demais datasets
  por ausência de anotação de arritmia sustentada.
- `P(AFIB episode) ≢ P(FUSION beat)` — restrição de consistência obrigatória (ver 04).

## 5. Nível 4 — Declaração diagnóstica (`diagnosis`, multilabel, escopo: registro/paciente)

Derivada de tabela clínica versionada (SCP-ECG para Chapman/PTB-XL):

```text
myocardial_infarction | conduction_disturbance | hypertrophy |
ischemic_st_t_change | rhythm_disorder | normal_ecg | other_diagnostic_statement
```

`P(diagnóstico | x) ≠ P(batimento anormal | x)` e `P(diagnóstico) ≤ P(evidência clínica
disponível)`. Registros sem statement anotado → `reviewStatus=review_required`, nunca imputado.

### 5.1 `dataset_disease_mapping`

| Dataset | Nível quality | Nível beat | Nível rhythm | Nível diagnosis | Lacuna |
|---|---|---|---|---|---|
| mitdb (48 reg.) | computável | N/S/V/FUSION/Q | SINUS (presumido) | — | sem statements |
| svdb (78 reg.) | computável | N/S/V/FUSION/Q | SINUS/SVT (presumido) | — | sem statements |
| incart (75 reg.) | computável | N/S/V/FUSION/Q | SINUS (presumido) | — | sem statements |
| afdb (23 utilizáveis) | computável | — (sem beat labels AAMI) | **AFIB/AFL/SINUS/JUNCTIONAL** | — | 00735/03665 sem `.dat` |
| chapman (45.152) | computável | — | — | SCP-ECG superclasses | 10 s, pré-treino |
| ptbxl (43.598) | computável | — | — | SCP-ECG superclasses | fallback pré-treino |

## 6. Contrato de label (obrigatório por instância)

```json
{
  "ontologyVersion": "3.0.0",
  "sourceDataset": "mitdb",
  "sourceLabel": "F",
  "canonicalCode": "FUSION",
  "semanticLevel": "beat",
  "temporalScope": "beat",
  "mappingRule": "AAMI-EC57 beat-level v3",
  "mappingAuthority": "tabela clínica versionada (única)",
  "ambiguity": "none|symbol_conflict|border_region|rhythm_overlap",
  "reviewStatus": "approved|review_required|excluded"
}
```

Regras: ontologia imutável por versão; qualquer mudança = nova versão + hash novo; as três
cópias divergentes do mapa AAMI (`aami_mapper.py`, `annotations.py`, `loader.py`) são
substituídas por **uma tabela única versionada**; duplicatas com labels conflitantes (DQ-04)
recebem `ambiguity=symbol_conflict` e `reviewStatus=review_required` até decisão humana.

## 7. Matriz de decisão — D2 (significado clínico das saídas)

| Opção | Descrição | Consequência | Recomendação |
|---|---|---|---|
| D2-a | 4 níveis separados (quality/beat/rhythm/diagnosis) | pipeline e modelo multitarefa (04); elimina ambiguidade F/Q | **Recomendada** |
| D2-b | Manter N/S/V/F/Q plano | perpetua DQ-03/DQ-05/DQ-12 | REJEITADA |

## 8. Matriz de decisão — D3 (AFDB)

| Opção | Descrição | Prós | Contras | Status |
|---|---|---|---|---|
| 1+3 | Integrar AFIB/AFL como **tarefa de ritmo** com cabeça temporal independente | dá semântica real a AFIB; usa os 23 registros; coerente com D2 | exige cabeça temporal e dados de episódio; AFDB sozinho não sustenta validação externa de ritmo | **Recomendada (PROPOSED_REQUIRES_RATIFICATION)** |
| 2 | Manter AFDB fora do classificador de batimentos | simples; honesto | F clínico (AFIB) continua ausente; meta F1(F)≥0,50 mede só fusão | alternativa conservadora |
| 4 | Excluir formalmente | fecha escopo | perde a única fonte de AFIB | REJEITADA salvo falha de integração |

## 9. Matriz de decisão — D4 (classe Q)

| Opção | Descrição | Consequência | Status |
|---|---|---|---|
| rejeição/abstenção | `Q_OR_UNKNOWN` visível no nível beat, fora dos alvos clínicos, roteada para `ABSTAIN_*` | coerente com D2; paced/unclassifiable não contamina N nem Anormal | **Recomendada** |
| inclusão apenas em Stage 1 | status quo v2.x | recria DQ-05 (17,2% do Anormal sem destino) | REJEITADA |
| exclusão justificada | remove Q do treino | perde a função "não confundir paced com N"; exige justificativa clínica formal | alternativa, requer ratificação |
| remapeamento clínico | Q→outra classe | sem base AAMI | REJEITADA |

## 10. Critérios de aceite desta decisão

1. Tabela de mapeamento única, versionada, com hash registrado no bundle (08).
2. Zero labels sem `canonicalCode`; zero desconhecidos → Q.
3. Contagens por nível/classe/dataset/paciente publicadas no manifest de dados.
4. `reviewStatus` presente em 100% das instâncias.
