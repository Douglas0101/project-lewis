# Relatório da Research Branch v2.4 — Classe F em Stage 2

## Resumo Executivo

A research branch v2.4 investigou a baixa performance inter-paciente da classe F
(fibrilação atrial/flutter) no Stage 2 do Project-Lewis. Após nove etapas
auditorias e experimentais (E00–E08), a **publicação v2.4 não foi autorizada**.
A melhor configuração alcançou **F1(F)=0.45 ± 0.08** em validação cruzada
inter-paciente, abaixo do gate de publicação `F1(F) >= 0.50`.

## Linha do Tempo e Checkpoints

| Etapa | Nome | Checkpoint | Resultado Chave |
| ------- | ------ | ------------ | ----------------- |
| E00 | Snapshot forense do baseline v14 | PASS | v14 reproduzido delta 0.0 |
| E01 | Distribuição de F por registro | PASS | 70% de F concentrado em 208/213 |
| E02 | Manifesto imutável dataset/features | PASS | Manifests e validador criados |
| E03 | Protocolo de split | PASS | StratifiedGroupKFold selecionado |
| E04 | Quality Gates QG5 | PASS | Status: `RESEARCH_CANDIDATE_NOT_PUBLICATION_READY` |
| E05 | Separabilidade das 16 features | PASS | Top features RR-dominadas; F não generaliza fora de 208/213 |
| E06 | Engenharia de features para F | PASS_HYPOTHESIS_REJECTED | 33 features não melhoraram F1(F) |
| E07 | Reescrita de rótulos / reamostragem | PASS | Reescrita não justificada; reamostragem por paciente melhorou baseline para F1(F)=0.47 |
| E08 | MLP + focal loss + class-weight | PASS | F1(F)=0.45, F1-macro=0.61; target 0.50 não atingido |
| E09 | Publicação guard / documentação | PASS | Nenhum artefato v2.4 publicado em `models/` |

## Diagnóstico Principal

A classe F apresenta **escassez e concentração**:

- Apenas **1.044 batimentos F** entre 55.161 amostras (~1.9%).
- **70%** de todos os F estão nos records **208 e 213**.
- F aparece em **45 records** distintos, mas geralmente como batimentos isolados
  (burst médio ≈ 1 batimento).
- As **features de RR** são as mais informativas, mas a separabilidade F vs resto
  permanece fraca (mutual information < 0.03).
- Treinar sem o record 208 e testar nele resulta em **F1(F)=0.02**.
- Treinar sem o record 213 e testar nele resulta em **F1(F)=0.40**.

A conclusão é que o problema é **predominantemente de dados**, não de modelagem.
A classe F carece de sinal generalizável suficiente nas 16 features atuais para
atingir o gate de publicação.

## Resultados Experimentais

### Baseline v14 (publicado, v2.3)

- F1-macro: ~0.545
- F1(F): ~0.214

### Baseline MLP minimal sobre features originais

- F1-macro: 0.495
- F1(F): 0.210

### Baseline MLP minimal sobre features enhanced (33 dimensões)

- F1-macro: 0.489
- F1(F): 0.170

### Baseline MLP minimal sobre F reamostrado por paciente

- F1-macro: 0.566
- F1(F): 0.465

### MLP 256 + focal loss + class-weight sobre F reamostrado

- F1-macro: 0.607 ± 0.039
- F1(F): 0.453 ± 0.082

## Decisão de Publicação

- **Não publicar** v2.4.
- Manter v2.3 como linha de produção.
- A research branch v2.4 fica arquivada em `experiments/stage2_v2.4_research/`.
- A próxima investigação deve priorizar:
  1. Coleta/criação de mais dados F diversos (ex: AFDB, registros adicionais).
  2. Features espectrais ou de morfologia de onda P/ausência de onda P.
  3. Arquitetura de two-stage mais robusta ou detector dedicado F-vs-rest.

## Quality Gates Aplicáveis

| Gate | Resultado | Observação |
| ------ | ----------- | ------------ |
| QG5 smoke balanced | PASS | F1(F)=0.94 em subconjunto balanceado (diagnóstico) |
| QG5 patientwise | FAIL | F1(F) inter-paciente < 0.50 |
| QG5 publication | FAIL | Sistema classificado como research candidate |

## Reprodutibilidade

Todos os scripts, testes e manifests estão em:

- `scripts/audit_stage2_*.py`
- `scripts/engineer_stage2_features_for_class_f.py`
- `scripts/train_stage2_baseline_enhanced.py`
- `scripts/resample_f_by_patient.py`
- `tests/test_*_e0*.py`
- `experiments/stage2_v2.4_research/E*/E*_manifest.json`

## Segurança e Compliance

- Nenhum PII foi exposto.
- Nenhum artefato v2.4 foi publicado em `models/`.
- Os artefatos v2.3 originais foram restaurados após a research branch.
- LGPD: dados apenas de registros fisiológicos anonimizados (PhysioNet).
