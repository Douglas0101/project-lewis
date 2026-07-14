# Investigação PTB-XL — Registros com Fibrilação Atrial (AFIB)

## Fonte de dados

- `data/raw_ptbxl/ptbxl_database.csv`
- `data/raw_ptbxl/scp_statements.csv`
- `data/processed/ptbxl/*.npy` (lead II, 500 Hz, 10 s)

## Resumo da população PTB-XL local

| Métrica | Valor |
| --------- | ------- |
| Total de registros no CSV | 21.799 |
| Registros processados (`.npy`) | 43.598 |
| Duração por registro | 10 s |
| Frequência de amostragem | 500 Hz |
| Amostras por registro | 5.000 |

## Códigos SCP-ECG relacionados

| Código | Descrição | Tipo | Observação |
| -------- | ----------- | ------ | ------------ |
| `AFIB` | atrial fibrillation | rhythm | Statement principal de interesse |
| `AFL` | atrial flutter | rhythm | **Não presente** nos dados locais |
| `AFLT` | atrial flutter | ? | Aparece em alguns registros, mas não é AFL puro |

## Contagem de registros AFIB

### Por `scp_codes` estruturado (confiança > 0)

| Critério | Registros |
| ---------- | ----------- |
| `AFIB` > 0 | 48 |
| `AFIB` = 100 | 48 |
| `AFIB` > 0 e `validated_by_human` | 48 |

### Por texto do relatório (`report`)

| Critério | Registros |
|----------|-----------|
| Texto contém "atrial fibrillation", "vorhofflimmern" etc. | 1.481 |
| Texto AFIB + `validated_by_human=True` | 961 |

### Distribuição por strat_fold

Os registros com AFIB por texto estão bem distribuídos pelos 10 folds oficiais do PTB-XL (143–153 por fold), o que favorece validação cruzada sem vazamento.

## Estimativa de batimentos F

Usando detecção simples de R-peaks (`scipy.signal.find_peaks`) nos 48 registros com `AFIB=100`:

| ECG ID | Batimentos detectados |
| -------- | ---------------------- |
| 351 | 25 |
| 4117 | 23 |
| 4401 | 23 |
| 4423 | 22 |
| 4531 | 20 |
| 4532 | 20 |
| 4761 | 18 |
| 5634 | 23 |
| 5776 | 21 |
| 7215 | 22 |

**Média:** ~22 batimentos por registro de 10 s de AFIB.

| Cenário | Registros F | Batimentos F estimados |
| --------- | ------------ | ------------------------ |
| `AFIB=100` (rótulo forte) | 48 | ~1.056 |
| Texto AFIB + validado (rótulo médio) | 961 | ~21.142 |
| Texto AFIB sem validação (rótulo fraco) | 1.481 | ~32.582 |

## Comparativo com dataset Stage 2 atual

| Dataset | F atual | F potencial (PTB-XL) |
| --------- | --------- | ---------------------- |
| MIT-BIH | 802 | — |
| INCART | 219 | — |
| SVDB | 23 | — |
| PTB-XL | 0 | 1.056–32.582 |
| **Total** | **1.044** | **2.100–33.682** |

Mesmo apenas os 48 registros `AFIB=100` dobrariam a classe F. Com os 961 registros validados por texto, a classe F cresceria 20×.

## Riscos do rótulo PTB-XL

| Risco | Severidade | Mitigação |
| ------- | ------------ | ----------- |
| Rótulo de diagnóstico global (10 s), não beat-level | Alto | Tratar como rótulo fraco; documentar no manifest; usar apenas para aumentar exposição da classe F |
| Presença de ritmo sinusal dentro de AFIB | Médio | Filtrar batimentos por RR irregular (ex: RR local std / mean > threshold) |
| `AFIB: 0.0` no `scp_codes` enquanto report diz AFIB | Alto | Investigar; usar scp_codes como primário e report como confirmação secundária |
| Diferença de população (alemã, 12 derivações, hospitalar) | Médio | Manter como dataset separado no manifest; medir generalização cruzada por dataset |
| Relatório multilíngue | Baixo | Busca textual limitada a termos conhecidos; preferir scp_codes estruturados |

## Conclusão

A integração do **PTB-XL** pode fornecer volume substancial de batimentos F. A estratégia recomendada é:

1. **Começar pelos 48 registros `AFIB=100`** como rótulo forte e de baixo risco.
2. **Em seguida, adicionar os 961 registros validados** com texto AFIB, aplicando filtro de RR irregular para aumentar pureza.
3. **Evitar** usar os 1.481 registros não validados como rótulo F até auditoria humana adicional.

## Recomendação para research branch

Priorizar a integração **AFDB** antes ou junto com PTB-XL, porque:

- AFDB fornece rótulos de **segmento de ritmo** (`.atr` manual) cruzados com batimentos (`.qrs`), gerando rótulo mais forte que PTB-XL.
- PTB-XL oferece **volume**, mas com rótulo de diagnóstico global.

A combinação AFDB + PTB-XL (`AFIB=100`) pode adicionar **3.000–6.000 batimentos F** de forma relativamente controlada, potencialmente suficiente para atingir F1(F) ≥ 0.50.
