# 02 — Especificação de Sincronização Temporal (D1) e Contrato de Alinhamento

**Status:** PROPOSTO — aguardando ratificação humana
**Data:** 2026-07-18
**Entregável:** `temporal_alignment_specification` (3)
**Corrige:** DQ-01, DQ-02, DQ-17

---

## 1. Defeito corrigido por esta especificação

Índices de anotação `s_i` em frequência nativa `f_d` (360/128/257 Hz) foram aplicados
diretamente ao sinal reamostrado a 500 Hz (`pipeline.py:146-155`, `segmenter.py:154-167`,
`time_domain.py:68`), produzindo janelas desalinhadas dos rótulos (drift linear até ~22 min no
svdb) e features RR erradas por fator `f_d/500` (0,72/0,256/0,514 — medido com exatidão contra
as anotações brutas). Esta especificação define o relógio único e os gates que impedem
recorrência.

## 2. Transformação canônica

Para cada anotação nativa `s_i` (amostras, relógio `f_d`) e frequência-alvo `f_t`:

```math
t_i = round\!\left(s_i \cdot \frac{f_t}{f_d}\right)
\qquad
\tau_i = \frac{s_i}{f_d}\ \text{(tempo contínuo, segundos)}
```

Equivalência obrigatória por índice:

```math
\left|\frac{t_i}{f_t} - \tau_i\right| \;\le\; \frac{0{,}5}{f_t} + \varepsilon_{\mathrm{num}}
```

Intervalos RR calculados **somente** em relógio correto, pelas duas vias, com coincidência
obrigatória dentro da tolerância de arredondamento (≤ 1/f_t + ε_num):

```math
RR_i^{ms} = 1000\,\frac{s_i - s_{i-1}}{f_d}
\qquad\text{ou}\qquad
RR_i^{ms} = 1000\,\frac{t_i - t_{i-1}}{f_t}
```

Toda feature temporal declara `unidade` e `relógio` no schema (ver 03). Nenhuma feature temporal
existe sem unidade explícita. Nenhum índice nativo toca sinal reamostrado.

## 3. Decisão D1 — frequência de trabalho

### 3.1 Matriz de avaliação

| Critério | Nativo por dataset | **Canônico f_t = 500 Hz** | Canônico 250 Hz | Multirresolução |
|---|---|---|---|---|
| Nyquist vs conteúdo clínico | varia; svdb 128 Hz fica no limite do filtro 0,5–40 Hz do projeto | folga >6× sobre a banda de 40 Hz; >1,6× sobre 150 Hz (banda diagnóstica AAMI/IEC) | folga >3× sobre 40 Hz | equivalente ao canônico |
| Filtro antialiasing | n/a | `resample_poly` polifásico (existente, validado Welch >40 dB) | idem | idem |
| Erro de interpolação | zero | documentado por registro na linhagem C02 | idem | idem |
| Precisão de R-peaks | nativa (0,5 amostra nativa ≈ 1,4–3,9 ms) | ≤ 1 ms após reamostragem (erro ≤ 0,5 amostra @500 Hz) | ≤ 2 ms | equivalente |
| Custo STM32F4 (168 MHz, 192 KB SRAM) | formas heterogêneas — inviável para contrato único de firmware | janela 500×1 float32 = 2 KB; inferência CNN < 200 ms/batimento (QG9) viável | janela 250×1 = 1 KB; ~50% do custo computacional | maior custo de engenharia |
| Equivalência entre domínios | **ausente** (raiz de DQ-02/DQ-14) | total | total | total |
| Complexidade de contrato | alta | baixa | baixa | alta |

### 3.2 Recomendação

```text
TARGET_FS = 500 Hz   (canônico para todos os datasets)
```

Justificativa: herança **não** é o argumento — o valor é re-derivado: (i) o filtro do projeto
(0,5–40 Hz) e o conteúdo QRS (<40 Hz) ficam com folga >6× de Nyquist; (ii) a janela de 1 s
(500 amostras) satisfaz o contrato de input (500,1) já usado pelo firmware e pelo TFLM;
(iii) 250 Hz é candidato legítimo de redução de custo, mas muda o shape de entrada e exigiria
revalidação de QG2 (AMPT @500 Hz usa banda 5–15 Hz com tolerância 150 ms) e do firmware — fora
de escopo nesta reconstrução. Picos de marcapasso (~2 ms) **não** são preservados a 500 Hz;
paced beats são `Q_OR_UNKNOWN` (nível de rejeição), logo fora dos alvos clínicos — limitação
declarada, não oculta.

| Parâmetro | Fórmula | Unidade | Intervalo candidato | Método de seleção | Dados permitidos | Risco | Status |
|---|---|---|---|---|---|---|---|
| TARGET_FS | — | Hz | {250, **500**} | análise espectral + custo edge (acima) | todos os brutos | médio | PROPOSED_REQUIRES_RATIFICATION |
| tol alinhamento | 0,5/f_t + ε_num | s | fixo por f_t | derivação de arredondamento | — | baixo | STANDARD_DERIVED |
| tol RR dual-clock | 1/f_t + ε_num | s | fixo por f_t | derivação de arredondamento | — | baixo | STANDARD_DERIVED |
| janela | 1000 ms = f_t amostras | ms | {600–1000} | contrato de input existente + QG3 | treino | baixo | PROJECT_EXISTING |
| posição do R | centro (f_t/2) | amostras | {f_t/2} | contrato existente | treino | baixo | PROJECT_EXISTING |

## 4. Gates obrigatórios (todos bloqueantes)

1. **G-T1** — R-peak dentro da região central prevista da janela (|argmax−centro| ≤ tolerância
   declarada) para ≥ 99% das janelas por dataset. Medida atual (defeituosa): 12,9% dentro de
   ±25 — evidência de DQ-01; pós-correção deve ser ≥ 99%.
2. **G-T2** — erro temporal máximo por índice documentado: `|t_i/f_t − τ_i| ≤ 0,5/f_t + ε_num`,
   com ε_num registrado no manifest.
3. **G-T3** — zero ocorrências de índice nativo aplicado a sinal reamostrado (teste estático +
   teste de correlação janela×sinal na posição correta ≥ 0,99 por amostragem, contra ≥ 0 na
   posição nativa — hoje o padrão é o inverso).
4. **G-T4** — nenhuma feature temporal sem `unidade` e `relógio` no schema.
5. **G-T5** — fs e unidade incorporados ao schema de dados (`fs_hz`, `time_unit`) e ao manifest.
6. **G-T6** — transformação testada em todos os datasets (mitdb 360, svdb 128, afdb 250,
   incart 257, chapman 500, ptbxl 100/500).
7. **G-T7** — teste de ida e volta amostra↔tempo: `s_i → t_i → s'_i` com |s'_i − s_i| ≤ tolerância
   declarada por dataset (≤ f_d/f_t amostras nativas + 1).
8. **G-T8** — comparação visual e estatística de janelas antes/depois (amostra por classe e
   dataset; correlação RR dual-clock ≥ 1 − tol).

Qualquer divergência material retorna:

```text
TEMPORAL_ALIGNMENT_FAILURE
```

## 5. Impacto de aceitação

A regeneração v3 (fora de escopo aqui) produz novos `finetuning_*`/`stage*_(npz|parquet)` com
hashes novos, `preprocessing_version=3.0.0` e manifests novos. **Todos os artefatos legados
(modelo, scaler, threshold, calibrador) permanecem inválidos** — ver 08/11.
