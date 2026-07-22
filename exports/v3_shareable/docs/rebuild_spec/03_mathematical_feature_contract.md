# 03 — Contrato Matemático de Features

**Status:** PROPOSTO — aguardando ratificação humana
**Data:** 2026-07-18
**Entregável:** `mathematical_feature_contract` (4)
**Corrige:** DQ-02, DQ-08, DQ-10, DQ-14 (componente de features)

---

## 1. Grupos de features

```text
morphology        r_amplitude, q_depth, t_amplitude, qrs_width_ms, qrs_area, st_slope_mV_s,
                  j_point, qrs_asymmetry_index, t_r_ratio, qrs_raggedness
timing            rr_prev, rr_next, rr_ratio, rr_local_mean, rr_local_std, rmssd, heart_rate
rhythm            agregados de episódio (rr_cv, runs ectópicos, irregularidade) — nível 3
spectral          potência em bandas fisiológicas (VLF/LF/HF) por janela ou episódio
signal_quality    std da janela, fração saturada, flatline flag, SNR estimado, deriva
metadata          lead, fs_hz, unidade — somente contrato/auditoria
dataset_provenance dataset_id, record_id, patient_id — **PROIBIDO como preditor clínico**
```

`dataset_provenance` existe para auditoria, splits e probes (06); nunca entra em qualquer
`fit`/`transform` de modelo clínico.

## 2. Schema obrigatório por feature

```json
{
  "name": "rr_prev",
  "group": "timing",
  "definition": "intervalo desde o batimento anterior",
  "formula": "RR_i = 1000 * (t_i - t_{i-1}) / f_t",
  "unit": "ms",
  "clock": "TARGET_FS (posições reescalonadas, ver 02)",
  "window": "beat",
  "version": "3.0.0",
  "plausible_range": [150, 3000],
  "out_of_range_policy": "flag RPEAK_UNCERTAIN + revisão",
  "missing_policy": "sem sentinela silenciosa; missing explícito + política por grupo",
  "fit_data": "inner-train do fold corrente apenas"
}
```

Regras duras:

- **Unidade e relógio explícitos** em todas as features temporais (encerra DQ-02).
- **Sem sentinelas silenciosas**: o padrão atual (`qrs_asymmetry_index` com −1,0 em 27,96% das
  linhas) é substituído por missing explícito + política documentada por grupo.
- **Intervalos plausíveis fisiológicos** declarados; violação gera flag de qualidade, não
  classificação.
- Features calculadas **somente com informação causal disponível na janela/episódio**; `rr_next`
  é permitido por ser informação de sinal já adquirida em inferência offline, mas deve ser
  declarado como tal e avaliado para o modo de operação alvo (edge).

## 3. Bateria de testes por feature (inner-train apenas)

| Teste | Pergunta | Regra |
|---|---|---|
| missingness | quanto falta? | > 10% → política explícita obrigatória |
| variância/estabilidade | degenerada? | var ≈ 0 → rejeitar |
| correlação com classe | sinal clínico? | reportar por dataset |
| correlação com dataset | atalho? | ver regra R-F1 |
| correlação parcial (classe | dataset) | sinal residual honesto? | reportar |
| informação mútua (classe) vs MI (dataset) | atalho? | ver regra R-F1 |
| estabilidade por paciente | artefato de poucos pacientes? | IC por bootstrap de pacientes |
| sensibilidade ao pré-processamento | muda com fs/filtro/normalização? | contrafactual (06 §5) |
| importância por permutação | contribui de fato? | somente inner loop |
| SHAP | interpretação complementar | nunca como seleção primária |
| degradação LODO | transporta entre domínios? | ver 06 §3 |

**R-F1 (regra de rejeição):** feature cuja associação com `dataset_id` excede sua associação
clínica condicionada é rejeitada ou devolvida para revisão. Baseline medido na auditoria
(defeito DQ-02 ativo): rr_local_mean 0,758, rr_prev/rr_next 0,711, heart_rate 0,675,
t_amplitude 0,593, qrs_raggedness 0,506 — após a correção de relógio (02), espera-se colapso
das correlações de origem temporal para valores fisiológicos; a regra R-F1 é aplicada de novo e
publicada por feature no manifest.

## 4. Tabela de parâmetros

| Parâmetro | Fórmula | Unidade | Intervalo candidato | Método de seleção | Dados permitidos | Risco | Status |
|---|---|---|---|---|---|---|---|
| janela RR local | 5 batimentos (i−2…i+2) | batimentos | {5, 7} | literatura HRV + inner loop | inner-train | baixo | PROJECT_EXISTING |
| janela RMSSD | 5 intervalos | intervalos | {5} | literatura HRV | inner-train | baixo | PROJECT_EXISTING |
| bandas espectrais | VLF/LF/HF padrão | Hz | {0,003–0,04 / 0,04–0,15 / 0,15–0,40} | padrão HRV | inner-train | baixo | STANDARD_DERIVED |
| limiar R-F1 | assoc(dataset) > assoc(classe\|cond) | — | fixo | regra de auditoria | inner-train | médio | PROPOSED_REQUIRES_RATIFICATION |
| missing máximo | 10% | fração | {0,05–0,10} | ratificação | inner-train | baixo | PROPOSED_REQUIRES_RATIFICATION |

## 5. Critérios de aceite

1. 100% das features com schema completo e `version=3.0.0`.
2. Relatório R-F1 publicado por feature (inner-train), sem uso de outer test.
3. Zero features temporais sem unidade/relógio; zero sentinelas silenciosas.
4. Features de ritmo existem apenas em escopo de episódio (coerente com 01 §4).
