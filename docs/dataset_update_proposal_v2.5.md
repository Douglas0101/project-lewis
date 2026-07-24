# Proposta de Atualização de Dataset v2.5 — Classe F (Fibrilação Atrial/Flutter)

## Status

Documento de planejamento gerado a partir de consulta a:

- Dados locais (`data/catalog/dataset_catalog.jsonl`, `data/features/stage2_multiclass.parquet`, `data/raw_afdb/`, `data/processed/afdb/`).
- Documentação PhysioNet (AFDB, SVDB, INCART, PTB-XL) via web search + indexação.
- MCP/local tools para inspeção de estrutura de arquivos.
- Resultado da investigação PTB-XL em `docs/ptbxl_afib_investigation_report.md`.

## Diagnóstico do gap atual

O Stage 2 atual contém **55.161 batimentos** classificados em S/V/F, provenientes de:

| Dataset | Registros | Batimentos F | % do total F |
| --------- | ----------- | -------------- | -------------- |
| MIT-BIH (`mitdb`) | 48 | 802 | 76.8% |
| INCART (`incart`) | 75 | 219 | 21.0% |
| SVDB (`svdb`) | 78 | 23 | 2.2% |
| **AFDB** (`afdb`) | **23** | **0** | **0%** |
| **PTB-XL** (`ptbxl`) | **21.799** | **0** | **0%** |
| **Total** | — | **1.044** | **100%** |

Apesar de AFDB e PTB-XL estarem catalogados e pré-processados, **nenhum batimento F** entrou no dataset Stage 2. O motivo principal:

- **AFDB**: os arquivos `.atr` são anotações de **ritmo** (`AFIB`, `AFL`, `N`, `J`), não de batimento. A função `src/features/pipeline.py::_load_raw_annotations` carrega `.atr` e mapeia símbolos de batimento; como AFDB não possui símbolos AAMI de batimento, nenhum batimento é extraído.
- **PTB-XL**: possui apenas rótulos de **diagnóstico global** (SCP-ECG) por registro de 10 s. Não há anotações de batimento AAMI.

## Oportunidade de expansão

### 1. AFDB — beat-level a partir de rhythm annotations

O MIT-BIH Atrial Fibrillation Database contém 23 gravações longas (10 h cada) com:

- `.qrs`: localizações de R-peak (automáticas, não corrigidas);
- `.atr`: anotações de ritmo manuais (`AFIB`, `AFL`, `J`, `N`).

A abordagem recomendada na literatura e documentação PhysioNet é:

1. Carregar `.qrs` para obter posições de R-peak.
2. Carregar `.atr` para obter intervalos de ritmo.
3. Para cada batimento, verificar em qual intervalo de ritmo ele cai.
4. Atribuir label F aos batimentos dentro de intervalos `AFIB` ou `AFL`.
5. Atribuir label N/S/V conforme possível aos demais (ou excluir do Stage 2).

**Estimativa conservadora**: cada gravação de 10 h a ~250 Hz com AF paroxismal pode conter dezenas de milhares de batimentos. Mesmo que apenas uma fração seja F, a adição pode ser **2.000–5.000 batimentos F**, triplicando ou quintuplicando a classe F.

**Riscos e mitigações**:

| Risco | Mitigação |
| ------- | ----------- |
| `.qrs` automáticos com falsos positivos/negativos | Filtrar batimentos por qualidade e consistência de RR; comparar com taxa de amostragem resample para 500 Hz. |
| Ritmo misto dentro de AF (batimentos normais no meio de AF) | Manter label F apenas para batimentos dentro de segmentos AFIB/AFL anotados; documentar que é rótulo de segmento. |
| Vazamento de informação do outer test | Extrair rhythm intervals e labels usando apenas o arquivo de anotação; não usar o sinal para derivar labels. |
| Diferença de frequência de amostragem (AFDB 250 Hz) | Resample para 500 Hz no pipeline existente (`data/processed/afdb` já processado). |

### 2. PTB-XL — registros com diagnóstico AFIB

PTB-XL local possui **21.799 registros** de 10 s a 500 Hz. A investigação detalhada está em `docs/ptbxl_afib_investigation_report.md`. Resumo:

| Cenário | Registros F | Batimentos F estimados | Rótulo |
| --------- | ------------ | ------------------------ | -------- |
| `AFIB=100` em `scp_codes` | 48 | ~1.056 | Forte |
| Texto "atrial fibrillation" + `validated_by_human=True` | 961 | ~21.142 | Médio |
| Texto "atrial fibrillation" sem validação | 1.481 | ~32.582 | Fraco |

A distribuição por `strat_fold` oficial é balanceada (~143–153 por fold), favorecendo validação cruzada sem vazamento.

**Riscos e mitigações**:

| Risco | Mitigação |
| ------- | ----------- |
| Rótulo de diagnóstico global, não beat-level | Tratar como rótulo fraco; documentar na manifest; usar apenas como dados adicionais de F. |
| Presença de ritmo sinusal no meio de ECGs com AFIB | Aplicar filtro de variabilidade de RR para manter apenas batimentos com RR irregular sugestivo de AF. |
| `AFIB: 0.0` no `scp_codes` enquanto `report` diz AFIB | Usar `scp_codes` como primário; investigar discordância antes de usar report. |
| Diferença de população (alemã, 12 derivações) | Usar lead II; manter como dataset separado no manifest para auditoria de generalização. |

### 3. Outras fontes identificadas

- **VITALDB Arrhythmia Database**: publicado em 2026, com anotações de arritmia. Requer avaliação de licença e formato.
- **PhysioNet Challenge 2017**: contém AFib em registros de ECG de curta duração; pode ser útil para validação cruzada.

## Impacto esperado na classe F

| Fonte | F adicionais estimados | Nota |
| ------- | ------------------------ | ------ |
| AFDB (beat-level via .atr+.qrs) | 2.000–5.000 | Rótulo de segmento |
| PTB-XL (`AFIB=100`, rótulo forte) | ~1.000 | Rótulo forte, volume pequeno |
| PTB-XL (AFIB texto + validado, filtro RR) | 10.000–20.000 | Rótulo médio, volume grande |
| **Total líquido estimado (AFDB + PTB-XL forte)** | 3.000–6.000 | Classe F passa de ~1% para ~5–10% |
| **Total máximo (AFDB + PTB-XL texto validado)** | 12.000–25.000 | Classe F ~15–30% dos dados |

## Proposta de implementação

### Fase 1: Integração AFDB (prioridade alta)

1. Criar `src/features/afdb_beat_loader.py` para:
   - Carregar `.qrs` e `.atr` do AFDB.
   - Mapear cada R-peak para intervalo de ritmo.
   - Retornar `(r_peaks, aami_labels)` compatível com `src/features/pipeline.py`.
2. Criar script de validação/teste para verificar:
   - Ausência de batimentos sem ritmo atribuído;
   - Distribuição de F por registro AFDB;
   - Consistência de cardinalidade.
3. Reexecutar `src/features/pipeline.py::build_finetuning_dataset` incluindo `afdb`.
4. Reexecutar `scripts/prepare_two_stage_datasets.py` e `scripts/prepare_stage2_features.py`.
5. Treinar baseline E06/E07 sobre novo dataset e medir F1(F) inter-paciente.

### Fase 2: Integração PTB-XL (prioridade média)

1. Criar mapeador SCP-ECG → AFIB.
2. Selecionar registros AFIB e detectar R-peaks.
3. Aplicar filtro de RR irregular para manter batimentos prováveis de AF.
4. Adicionar como dataset separado no manifest.
5. Avaliar ganho de generalização em grupos não-AFDB.

### Fase 3: Reexecução da research branch

Após expansão do dataset, reexecutar:

- E05 (separabilidade das features atualizadas);
- E06 (context features, se ainda justificado);
- E07 (sampling);
- E08 (long-tail loss);
- E09 (decision/calibration);
- E10–E13 (se candidato promissor emergir).

## Requisitos de compliance

- Novo dataset deve ter `dataset_manifest_hash` e `feature_schema_hash` atualizados.
- Não usar informação do outer test para selecionar registros ou derivar labels.
- Manter GroupKFold/StratifiedGroupKFold por paciente.
- Documentar rótulos fracos vs fortes no manifest.
- Preservar artefatos v2.3 em `models/`.

## Recomendação

A integração do **AFDB** é a ação de maior impacto e menor risco para a research branch. Recomenda-se iniciar por ela antes de PTB-XL, pois fornece rótulos de segmento de ritmo mais confiáveis do que rótulos globais de diagnóstico.

## Próxima decisão

Decidir se a equipe avança para:

1. **Implementação Fase 1 (AFDB)** — criar loader e regenerar datasets.
2. **Integração PTB-XL (`AFIB=100`)** — começar pelo rótulo forte de 48 registros.
3. **Parada da research branch** — manter v2.3 e arquivar a proposta para futura sprint.
