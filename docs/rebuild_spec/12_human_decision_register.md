# 12 — Registro de Decisões Humanas (D1–D7)

**Status:** TODAS as decisões `PENDING_RATIFICATION`
**Data:** 2026-07-18
**Entregável:** `human_decision_register` (15)

---

## Ordem de dependência

```text
D2–D4 (ontologia)  →  D1 (relógio)  →  regeneração v3  →  D5 (arquitetura, via matriz 05)
   →  D6 (calibração)  →  bundle/attestation (08/09)  →  D7 (nível de autonomia pós-treino)
```

Nenhuma decisão posterior é executável sem as anteriores ratificadas.

## D1 — Relógio e reamostragem

| Campo | Conteúdo |
| --- | --- |
| Pergunta | frequência de trabalho para todos os datasets |
| Opções | nativo por dataset / canônico 500 Hz / canônico 250 Hz / multirresolução |
| Evidência | 02 §3 (Nyquist, antialiasing, custo STM32F4, erro de interpolação, equivalência de domínios) |
| Recomendação | **canônico 500 Hz** (re-derivado, não herdado) |
| Implicação | todos os derivados v3 a 500 Hz; firmware/TFLM inalterado em shape (500,1) |
| Status | **PENDING_RATIFICATION** |

## D2 — Significado clínico das saídas

| Campo | Conteúdo |
| --- | --- |
| Pergunta | separação quality / beat / rhythm / diagnosis |
| Opções | 4 níveis separados (D2-a) / esquema plano atual (D2-b) |
| Evidência | 01 §§1–5; DQ-03/DQ-05/DQ-12 |
| Recomendação | **D2-a** (4 níveis) |
| Implicação | ontologia v3.0.0; modelo multitarefa (04); fim de F=AFIB e de desconhecido→Q |
| Status | **PENDING_RATIFICATION** |

## D3 — AFDB

| Campo | Conteúdo |
| --- | --- |
| Pergunta | destino do AFDB e da classe AFIB |
| Opções | 1+3 (ritmo + cabeça temporal) / 2 (fora do classificador de batimentos) / 4 (excluir) |
| Evidência | 01 §8; AFDB hoje contribui 0 beats (DQ-03); `afdb_beat_loader.py` existe e não está integrado |
| Recomendação | **1+3** |
| Implicação | nova tarefa de ritmo; AFDB sozinho não sustenta validação externa de ritmo (limitação declarada) |
| Status | **PENDING_RATIFICATION** |

## D4 — Classe Q

| Campo | Conteúdo |
| --- | --- |
| Pergunta | papel de Q/paced/unclassifiable |
| Opções | rejeição/abstenção explícita / inclusão só no Stage 1 (status quo) / exclusão justificada / remapeamento |
| Evidência | 01 §9; DQ-05 (Q=17,2% do Anormal sem destino; modelo de produção treinou sem Q) |
| Recomendação | **rejeição/abstenção explícita (`Q_OR_UNKNOWN`)** |
| Implicação | Q fora dos alvos clínicos; roteada para ABSTAIN_*; mapa AAMI unificado |
| Status | **PENDING_RATIFICATION** |

## D5 — Arquitetura

| Campo | Conteúdo |
| --- | --- |
| Pergunta | família de modelos |
| Opções | (a) CNN-1D / (b) MLP features / (c) fusão / (d) multitarefa; hierárquico 2 estágios como deployment; MoE EXPERIMENTAL |
| Evidência | 04 §4; Stage 2 condicional funcional (F1 0,642) vs triagem falha sob dados defeituosos |
| Recomendação | decidir **por dados** na matriz 4×5×5 (05); cascata 2 estágios mantida como forma de deployment |
| Implicação | 100 células controladas; nenhuma decisão antecipada por intuição |
| Status | **PENDING_RATIFICATION** (do protocolo; a escolha da família é resultado do protocolo) |

## D6 — Calibração

| Campo | Conteúdo |
| --- | --- |
| Pergunta | topologia do calibrador |
| Opções | global / por tarefa / por classe / hierárquico / por domínio / abstenção |
| Evidência | 07 §4; classes raras (FUSION 45 pacientes) inviabilizam calibrador por classe puro |
| Recomendação | **por tarefa + hierárquico (pooling parcial) para classes raras + política de abstenção**; calibrador por domínio proibido quando esconder não-generalização |
| Implicação | partição de calibração independente congelada no manifest |
| Status | **PENDING_RATIFICATION** |

## D7 — Atualização pós-treinamento

| Campo | Conteúdo |
| --- | --- |
| Pergunta | nível máximo de autonomia |
| Opções | LEVEL_0_MONITOR_ONLY / LEVEL_1_SHADOW_RECALIBRATION / LEVEL_2_SIGNED_CANDIDATE_GENERATION / LEVEL_3_HUMAN_AUTHORIZED_ACTIVATION |
| Evidência | 07 §6; attestation ainda shadow (09); backend Sigstore não ratificado |
| Recomendação | **LEVEL_1 agora**; LEVEL_2 somente com attestation funcional ratificada; LEVEL_3 somente com quorum + revisão clínica |
| Implicação | proibida qualquer atualização autônoma de pesos/classes/ontologia/threshold |
| Status | **PENDING_RATIFICATION** |

## Como ratificar

Cada decisão registrada neste arquivo exige: aprovador humano nomeado, data UTC, hash deste
documento, e resultado (`APPROVED` / `AMENDED` / `REJECTED`) anexado em novo registro
(imutável). Ratificação parcial desbloqueia somente a etapa correspondente da cadeia de
dependência.

## Eventos de autorização append-only

### A01 — Continuação com dados existentes

| Campo | Conteúdo |
| --- | --- |
| Data UTC | `2026-07-21T03:53:18Z` |
| Hash do registro antes do anexo | `3ae2318ae5df12c020df755e4216187a0b7626188a748321c3cf80fa146bde42` |
| Declaração literal | “Com todos os dados que o projeto já possui, eu concedo aprovações antecipadas, pode continuar” |
| Ator | `UNNAMED_PROJECT_OWNER_CURRENT_SESSION` |
| Identidade | `UNVERIFIED` — nome e assinatura não fornecidos |
| Escopo | implementação, preflight e pesquisa controlada com dados existentes |
| Exclusões | fabricar evidência; enfraquecer gates; sobrescrever geração; promoção clínica |
| Resultado | `AUTHORIZATION_RECORDED / APPROVED_FOR_IMPLEMENTATION_AND_PREFLIGHT` |
| Categoria epistemológica | `OBSERVED` |

Este evento autoriza a continuação técnica da pesquisa, mas **não** transforma identidade de
paciente ausente, lineage incompleto ou validação externa ausente em evidência observada. D1–D7
continuam pendentes de identidade formal conforme o contrato acima; D7 e attestation permanecem
bloqueadores absolutos para qualquer promoção ou ação operacional.
