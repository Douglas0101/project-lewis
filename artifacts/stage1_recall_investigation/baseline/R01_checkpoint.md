# R01 — rastreamento completo do fluxo de inferência

## Hipótese

O caminho canônico mantém cardinalidade e ordem entre amostras, targets, outputs e decisões.

## Alterações

- `scripts/diagnose_stage1_qg5.py`: diagnóstico opt-in, validado por Pydantic e sem sinais ECG persistidos.
- `docs/stage1_inference_contract.md`: mapa e contrato observados.
- `stage1_qg5_trace.json`: estatísticas, identidades, métricas e checks de integridade.
- `stage1_qg5_samples.csv`: rastreio técnico por amostra.

Nenhum teste, threshold, scaler, modelo ou dataset foi alterado.

## Verificações

- [PASS] IDs estáveis;
- [PASS] nenhuma duplicação no subset selecionado;
- [PASS] nenhuma amostra perdida;
- [PASS] ordem preservada;
- [PASS] 2.048 targets, outputs e predições;
- [PASS] zero NaN/Inf nas transformações;
- [PASS] reprodução direta e `pipeline.predict` têm delta máximo 0.0;
- [PASS] zero divergência de decisão;
- [PASS] threshold efetivo idêntico;
- [PASS] métrica reproduzida manualmente;
- [PASS] hashes de modelo, scaler e threshold permanecem idênticos ao R00;
- [PASS] `make lint` e flake8 focado no script diagnóstico;
- [FAIL] Pyright: `0 errors, 2 warnings` preexistentes fora do script novo;
- [PASS] teste QG5 relacionado reproduz a mesma falha e as mesmas métricas;
- [FAIL] `record_id` e `group_id` não existem nos NPZ usados pelo QG5;
- [NOT RUN] semântica arquitetural da coluna 1 (R02/R07);
- [NOT RUN] compatibilidade histórica modelo–scaler–threshold (R05/R10).

## Métricas antes e depois

Não houve correção nem mudança de decisão.

| Métrica | Antes | Depois |
| --- | ---: | ---: |
| Recall | 0.0661458333 | 0.0661458333 |
| Precision | 1.0 | 1.0 |
| F1 Anormal | 0.1240840254 | 0.1240840254 |
| Specificity | 1.0 | 1.0 |
| Balanced Accuracy | 0.5330729167 | 0.5330729167 |
| AP | 0.9136421077 | 0.9136421077 |
| Positive Prevalence | 0.9375 | 0.9375 |
| AP Lift | -0.0238578923 | -0.0238578923 |
| TP/FP/TN/FN | 127/0/128/1793 | 127/0/128/1793 |
| taxa predita Anormal | 0.06201171875 | 0.06201171875 |

## Hipótese

`SUPPORTED`: alinhamento e cardinalidade estão íntegros para o subset observável.

## Checkpoint

`PASS_WITH_WARNINGS`: a proveniência de paciente/registro não pode ser provada a partir dos NPZ.

Próxima etapa autorizada: `R02 — auditoria do artefato Stage 1`.
