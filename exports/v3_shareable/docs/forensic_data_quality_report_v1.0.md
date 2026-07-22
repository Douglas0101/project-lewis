# Relatório Forense Consolidado — Qualidade de Dados e Falha QG5' Stage 1

**Projeto:** Project-Lewis (classificação de arritmias ECG em edge, STM32F4)
**Versão:** 1.0 — 2026-07-18
**Natureza:** síntese interligada das investigações #41 (forense do teste QG5) e da auditoria de
dados ponta a ponta (Fases 0–18). **Documento para leitura e decisão humana.**
**Nenhuma correção, re-treino, promoção ou alteração de threshold foi executada ou é autorizada
por este relatório.**

---

## 1. Sumário executivo — a cadeia causal completa

Os dados brutos estão íntegros e o pré-processamento de sinal é bem documentado. O sistema falha
em um único ponto estrutural, do qual quase todos os sintomas derivam:

> **Os índices de R-peak das anotações WFDB (relógio nativo de cada dataset: 360/128/257 Hz)
> são aplicados diretamente sobre o sinal reamostrado a 500 Hz, sem reescalonamento, tanto na
> segmentação das janelas quanto no cálculo das features temporais.**

Em cadeia:

1. **Janelas desalinhadas dos rótulos (DQ-01).** Cada janela de 500 amostras é extraída na
   posição `t × (fs_nativo/500)` em vez de `t`. O deslocamento cresce linearmente dentro de cada
   registro: até ~8,4 min no mitdb, ~14,6 min no INCART e **~22 min no svdb** no final dos
   registros. O rótulo descreve um batimento; a janela mostra outro. Classes majoritárias (N)
   sobrevivem por dominância estatística; classes anormais são destruídas.
2. **Features RR erradas por fator fixo por dataset (DQ-02).** `rr_ms = rr_samples / 500 × 1000`
   recebe amostras em relógio nativo: os intervalos RR ficam multiplicados por 0,72 (mitdb),
   0,256 (svdb) e 0,514 (INCART). `heart_rate` fica inflado 1,39–3,9× (mediana global de
   190 bpm — fisiologicamente impossível). Como o fator é diferente por dataset, as features
   temporais **codificam a identidade do dataset**, não fisiologia.
3. **O modelo aprende o dataset, não a doença (DQ-14).** Somam-se duas confusões: features com
   escala dependente de fs_nativo e composição de classes por dataset (S≈svdb, V≈incart,
   F≈mitdb rec. 208/213, Q≈mitdb rec. 102). Resultado medido: ROC-AUC global 0,5546 (quase
   aleatório); recall de Anormal por dataset @0,54: svdb 0,467, incart 0,119, mitdb 0,104.
4. **A governança amplifica em vez de conter (DQ-06).** Um modelo cuja run declarou
   `passes_qg5=false` (MCC 0,0305) foi promovido a `models/`; o pipeline de inferência e o teste
   carregam um threshold (0,58) de uma geração anterior à do modelo (que co-produziu 0,54); a
   run produtora usou 2 folds ~50/50 em vez do GroupKFold-5 do protocolo; nenhum manifest de
   split foi congelado.
5. **O teste QG5 mede e expõe o colapso, com lupa distorcida (DQ-16).** O subset focal é 100%
   mitdb (o pior domínio do modelo), 93,75% anormais, 28 pacientes, N de um único registro.
   Recall = 0,0661 (IC95% agrupado [0,0135; 0,1176]). O gate 0,30 falharia em qualquer subset:
   no dataset completo o recall é 0,118 @0,58 e 0,242 @0,54; na validação do próprio fold de
   treino, 0,229 @0,54.
6. **Achados independentes agravam o quadro:** AFDB contribui zero batimentos (a classe F é
   somente fusão V+N — não há fibrilação atrial no dataset); 7 pares de janelas idênticas com
   rótulos conflitantes no svdb; a política da classe Q mudou silenciosamente entre o treino do
   modelo (sem Q) e a avaliação atual (com Q).

**Conclusão:** a falha do QG5 não é um problema de threshold, de teste, de runtime ou de
arquitetura de modelo. É a expressão mensurável de um defeito de dados (DQ-01/02), combinado com
falhas de representatividade (AFDB/F), de contrato de classes (Q), de proveniência de splits e
de governança de artefatos. **Qualquer correção de DQ-01/DQ-02 regenera os datasets e invalida
todos os baselines, modelos, scalers e thresholds existentes.**

---

## 2. Linha do tempo dos artefatos — onde a correspondência se quebra

| Data | Evento | Artefatos (hash curto) | Quebra de correspondência |
|---|---|---|---|
| jun 22 | Avaliação `two_stage_evaluation_v2.0.json` reporta Recall 0,3254, `passes_qg5: true` | modelo/scaler da época (não mais presentes) | **Inválida hoje**: modelo, dados e contexto de threshold diferentes dos atuais |
| jun 25–26 | Processed `.npy` gerados (C02) — **estáveis desde então** | 201 registros, linhagem JSON por registro | — |
| jun 26 02:13 | Modelo Stage 2 + `stage1_threshold_v2.0.json` (**0,58**) | `afd20c4a…`, `b75e14ab…` | o 0,58 pertence a esta geração, não à seguinte |
| jul 1 | Teste `test_two_stage_qg5.py` criado (e28d0ad) — contrato inalterado desde então | `edf58f71…` | — |
| jul 4 00:43:52 | Modelo Stage 1 + scaler + `stage1_threshold.json` (**0,54**) co-produzidos no worktree (run 2-fold, `passes_qg5=false`) | `cd5e2474…`, `80f5fabf…`, `77b5a410…` | modelo promovido apesar do QG reprovado; threshold 0,54 **não** é o que o teste carrega |
| jul 7 | Mudanças C03 (QRS onset/offset, features novas); pivot MLP v2.3 | commits `4eb2b2f`, `7078865` | — |
| jul 8 01:13 | `stage1_binary.npz` / `stage2_multiclass.npz` regenerados (agora **com Q**) | `23adba21…`, `68fb0a8e…` | dados de avaliação ≠ dados de treino do modelo (461.614 vs 473.036 beats) |
| jul 11 16:45 | `finetuning_mitbih_family.npz` + `training_manifest.json` (declara afdb com **0 beats**) | — | stage NPZs ficam defasados da própria fonte |

Leitura da tabela: nenhum par (modelo, threshold, dados, avaliação) atualmente em uso foi
produzido na mesma geração. O teste QG5 amarra modelo de jul 4 + threshold de jun 26 + dados de
jul 8 (fonte de jul 11).

---

## 3. O defeito central (DQ-01/DQ-02): prova e cascata

### 3.1 Código implicado

- `src/features/pipeline.py:118-120` — `_load_raw_annotations` retorna `ann.sample` **em fs
  nativo**, sem conversão;
- `src/features/pipeline.py:146-155` — `build_beat_records` fixa `fs = 500.0` e repassa os
  índices nativos ao segmentador e ao extrator temporal;
- `src/data/segmenter.py:154-167` — `r_idx = int(r_peaks[i])` indexa diretamente o sinal de
  500 Hz; a decisão `rr_ms < 600 ms` usa o RR já errado;
- `src/features/time_domain.py:67-68` — `rr_samples = np.diff(r_peaks)`; `rr_ms = rr_samples /
  fs * 1000.0` com `fs=500` sobre amostras nativas.

### 3.2 Prova empírica (somente leitura, reproduzível)

**(a) Posição das janelas.** Para cada registro, correlacionou-se a janela armazenada no NPZ com
o sinal processado fatiado (i) na posição nativa e (ii) na posição corretamente reescalada
`r×500/fs_nat` (anotações filtradas como na produção, sem `~`/`+`/`x`):

| Registro | fs nativo | corr(posição nativa) | corr(posição correta) |
|---|---|---|---|
| mitdb/100 | 360 | 0,361 / 0,972 / 0,967 / 0,947 / **1,000** / 0,971 | −0,013…0,081 |
| svdb/800 | 128 | 0,320 / 0,192 / 0,468 / 0,593 / **0,909** / 0,830 | −0,256…0,194 |
| incart/I01 | 257 | 0,563 / 0,943 / 0,958 / 0,813 / 0,678 / **0,887** (até 0,971 no teste ampliado) | −0,194…0,240 |

As janelas provêm das posições nativas — defeito confirmado nos três domínios. Valores <1,0
refletem o edge-padding das janelas de 600 ms e arredondamento. Consequência visível nos dados:
`r_peak_in_segment` tem apenas 12,9% das janelas com argmax dentro de 250±25 (esperado ~100% se
centradas no R; ~10% se aleatório).

**(b) Unidade das features RR.** Razão `rr_prev_parquet / RR_real` medida contra as anotações
brutas: mitdb 586,0/813,9 ms = **0,72** = 360/500; svdb 334/1304,7 = **0,256** = 128/500;
INCART 330/642 = **0,514** = 257/500. Exato em todos os casos testados. `heart_rate =
60000/rr_prev` herda o erro (mediana global 189,9 bpm; p99 = 625 bpm).

### 3.3 Cascata de consequências

| Efeito | Mecanismo | Evidência |
|---|---|---|
| CNN Stage 1 não aprende morfologia | rótulo e waveform divergem (exceto perto do início dos registros) | AUC 0,5546 (dataset completo); MCC 0,0305 na validação do fold |
| N "funciona" por acidente | dominância de N: janela errada geralmente mostra outro N | especificidade 1,0 no subset; precisão 1,0 com 6,2% de encaminhamento |
| Anormais colapsam | janela rotulada S/V/F geralmente contém um N de outro instante | recall 0,066–0,242 conforme subset/threshold |
| Resíduo de sinal (AUC > 0,5) | início dos registros quase alinhado + artefatos por dataset | recall svdb 4,5× mitdb @0,54 |
| Features MLP v2.x contaminadas | escala fs_nat/500 nas RR-features; amplitude por domínio | \|corr\| com dataset: rr_local_mean 0,758, rr_prev 0,711, heart_rate 0,675 |
| Janelas 600 ms + edge-pad massivas | decisão `rr<600` usa RR errado | svdb/INCART ~100% em fallback |
| Stage 2 condicional sobrevive | S/V/F morfologia grosseira (V largo) ainda parcialmente presente; métrica condicional ignora a triagem | F1-macro 0,642 sobre anormais verdadeiros |

---

## 4. Mapa de interdependência dos achados

| ID | Severidade | Causa raiz | Efeito medido | Interligações |
|---|---|---|---|---|
| DQ-01 | **CRÍTICA** | índices `.atr` nativos sem reescalonamento (`pipeline.py`, `segmenter.py`) | janela↔rótulo desalinhados (drift até ~22 min) | causa de DQ-02, DQ-09 (parcial), DQ-17; explica #41 |
| DQ-02 | **CRÍTICA** | `time_domain.py:68` com fs=500 sobre clock nativo | RR-features e heart_rate errados por fator fixo por dataset | causa de DQ-14 (features); alimenta MLP v2.x |
| DQ-03 | ALTA | `afdb_beat_loader.py` não integrado; anotações de ritmo dropadas | AFDB = 0 beats; F = 1.044 fusion beats, 45 pacientes, top-5=82% | define a ontologia de F (DQ-05 adjacente); agrava DQ-14 |
| DQ-04 | ALTA | anotações duplicadas adjacentes no svdb | 7 pares de janelas idênticas com y=0 e y=1 | ruído de label supervisionado direto |
| DQ-05 | ALTA | build jul-4 sem Q; build jul-8 com Q (`--exclude-q` / política do mapper) | modelo de produção nunca viu Q; avaliação inclui 11.422 Q (17,2% do Anormal) | liga DQ-06 (governança) a DQ-12 (mapper) |
| DQ-06 | ALTA | promoção fora de bundle; run 2-fold; threshold órfão | modelo QG-failed em produção; teste usa 0,58 em vez de 0,54 co-produzido | amplifica DQ-01 (recall 0,1448→0,0661 no subset) |
| DQ-07 | MÉDIA | ausência de gate de qualidade de sinal | flatline 0,52% global / 1,38% incart; clip svdb 0,14% | degrada janelas já comprometidas |
| DQ-08 | MÉDIA | variância de incart dominada por ruído/deriva | p2p mediano 0,46 vs 3,9 (mitdb/svdb) | alimenta DQ-14 (amplitude) |
| DQ-09 | MÉDIA | janela 1s vs RR mediano 0,63s | 76,9% das janelas sobrepostas; 38,1% >50% | suporte efetivo ≪ 473k; potencializado por DQ-01 |
| DQ-10 | MÉDIA | falha de detecção QRS em 27,96% | `qrs_asymmetry_index` com sentinela −1,0 não contratada | afeta MLP v2.x |
| DQ-11 | ALTA | dados regenerados sem congelar manifests; run 2-fold | associação paciente–fold do modelo atual **irrecuperável**; `data/splits/` vazio | `SPLIT_PROVENANCE_UNVERIFIABLE` |
| DQ-12 | MÉDIA | mapa AAMI triplicado com políticas divergentes (`\|`/desconhecidos) | contagem e semântica de Q instáveis | alimenta DQ-05 |
| DQ-13 | BAIXA | distribuição PhysioNet | AFDB 00735/03665 sem `.dat` | reduz AFDB para 23 registros |
| DQ-14 | ALTA | DQ-02 + composição (S≈svdb, V≈incart, F≈mitdb) | features e scores codificam dataset; recall por dataset 0,10–0,47 | efeito de DQ-02, DQ-03, DQ-08 |
| DQ-15 | MÉDIA | stats z-score calculadas no bruto pré-filtro, aplicadas pós-filtro | pequeno desvio de contrato de normalização | documentação |
| DQ-16 | MÉDIA | construção do subset do teste | teste focal 100% mitdb, 28 pacientes, N de 1 registro (100), F de 2 registros (94%) | lupa sobre DQ-01; não representa a população |
| DQ-17 | INFO | DQ-01 | `r_peak_in_segment` ~uniforme (12,9% ±25) | confirmação independente de DQ-01 |

---

## 5. Achados secundários — detalhe

### 5.1 Duplicatas com rótulos conflitantes (DQ-04)
7 pares de janelas byte-idênticas no `stage1_binary.npz`, todos no svdb, cada par com `beat_idx`
adjacentes e labels 0 e 1: registros 804 (1749/1750), 823 (1087/1088), 848 (2397/2398) e mais 4
grupos. Mesma waveform, duas classes — ruído de label na forma mais direta possível.

### 5.2 AFDB e a semântica de F (DQ-03, DQ-13)
AFDB tem anotações de **ritmo** (`+` com aux_note `(AFIB`/`(AFL`/`(N`/`(J`), não de batimento.
O pipeline de produção lê apenas `.atr`, dropa `+` e gera zero beats. O loader correto
(`src/features/afdb_beat_loader.py`, AFIB/AFL→F) existe mas não é importado por nenhum caminho
produtivo. O `training_manifest.json` declara `afdb: n_records=23, n_beats=0` — perda silenciosa
documentada retroativamente. Consequência: **não há nenhum batimento de fibrilação atrial no
dataset**; a meta de publicação `F1(F) ≥ 0,50` mede fusão V+N, majoritariamente dos registros
mitdb 208/213.

### 5.3 Classe Q (DQ-05, DQ-12)
Q = 11.422 beats (17,2% do Anormal do Stage 1), 83 pacientes, top-5 = 74,8% (registro 102 =
2.083). Não tem destino no Stage 2 (excluída da classificação final desde v2.0). O modelo de
produção treinou **sem Q** (461.614 = 473.036 − 11.422, exato); a avaliação atual inclui Q.
Três cópias do mapa AAMI divergem sobre `|` e desconhecidos (produção: →Q; outras: drop; doc:
excluir).

### 5.4 Qualidade fisiológica (DQ-07, DQ-08)
Flatlines (std<0,01) retidos com labels: 0,52% global, 1,375% no INCART (~2.418 beats). Clipping
irreversível em dois pontos: ±5 mV (pré-normalização) e ±10σ (pós), atingindo 0,14% dos beats
do svdb. Baseline |mean| até 4,08σ no svdb. INCART tem amplitude mediana 8,5× menor que
mitdb/svdb (variância dominada por não-QRS).

### 5.5 Splits (DQ-11)
`data/splits/groupkfold_5_stratified/` contém apenas `.gitkeep`. A run produtora do modelo
(jul 4) usou **2 folds ~50/50** (val 231.134 e 230.480 beats), não o GroupKFold-5 configurado.
A associação paciente–fold de jul 4 é irrecuperável porque a regeneração de jul 8 alterou as
contagens por registro (fold-1 histórico: 230.480 beats vs 95.723 na reconstrução com dados
atuais). O mecanismo de split em si é correto (GroupKFold determinístico por `record_id`).

### 5.6 Governança de artefatos (DQ-06)
Modelo promovido com `passes_qg5=false` (linhagem: worktree
`20260704_033953_stage1_v2.0/fold_1`, hashes idênticos aos de produção). O teste e o pipeline
canônico carregam `stage1_threshold_v2.0.json` (0,58, jun 26) — artefato da geração anterior; o
threshold co-produzido com o modelo é 0,54 (`stage1_threshold.json`). Impacto do desencontro no
subset: recall 0,1448→0,0661; não altera o veredito do gate (ambos < 0,30, no subset e no
dataset completo).

### 5.7 O teste focal (DQ-16)
Subset determinístico: 128 N (primeiras do stage1_npz — todas do registro 100) + 640/640/640
S/V/F (primeiras do stage2_npz — V: registro 106 com 520; F: registros 208+213 com 600). 28
pacientes, 100% mitdb, prevalência 93,75%. IC95% do recall (bootstrap por paciente): [0,0135;
0,1176] — a falha é robusta para o subset, mas o subset não estima a população.

---

## 6. O que está são (evidências negativas verificadas)

- Zero NaN/Inf nos dois NPZs; dtypes e shapes consistentes ((n,500,1) float32).
- Brutos completos conforme QG0 (48/78/25/75 registros; 45.152 Chapman; 43.598 PTB-XL).
- Sem colisões de `record_id` entre datasets; sem `(dataset,record,beat_idx)` duplicado.
- Inferência determinística: 3 execuções bit-idênticas (delta 0,0); modo de inferência correto
  (R04); loader Keras 3 equivalente (R03); pesos imutáveis.
- Scaler de produção comprovadamente fit **somente** no treino do fold (n=231.134 beats, exato).
- Segmentação usa edge-pad, nunca zeros (regra de ouro #5 honrada); sem SMOTE fora do treino;
  oversampling do Stage 2 apenas no X_train do fold.
- Linhagem C02 por registro com checksums do raw; mapa AAMI semanticamente estável desde a9d15a1.
- Estágio 2 condicional funcional (F1-macro 0,642; F 0,747; S 0,717; V 0,463).
- Nenhuma evidência de vazamento de validação/teste em fits (o defeito é de relógio de dados,
  não de protocolo de fit).

---

## 7. Registro de evidências (reproduzível)

**Ambiente:** Python 3.12.3 / TF 2.21.0 / Keras 3.14.1 / NumPy 2.4.6 / sklearn 1.9.0 / wfdb
4.3.1; CPU-only (cuInit falha); git HEAD `886f1b3` (develop).

**Hashes SHA-256 (artefatos em uso):** teste `edf58f71…93dc`; pipeline `b9d57ee1…2bb`; modelo S1
`cd5e2474…c347`; scaler S1 `80f5fabf…acb3`; threshold usado (0,58) `b75e14ab…6746`; threshold
co-produzido (0,54) `77b5a410…8526`; modelo S2 `afd20c4a…abc2`; `stage1_binary.npz`
`23adba21…2b35`; `stage2_multiclass.npz` `68fb0a8e…b04a`; X(stage1) `81be6c90…` (16 primeiros);
y(stage1) `412fff16…`.

**Números-chave:**
- Teste focal: TP/FP/TN/FN = 127/0/128/1793; recall 0,0661458333; ROC-AUC 0,2297; PR-AUC 0,9136
  (prevalência 0,9375); bootstrap por paciente IC95% [0,0135; 0,1176].
- Dataset completo (473.036): @0,54 recall 0,2423 / prec 0,1795 / esp 0,8186; @0,58 recall
  0,1180; ROC-AUC 0,5546; AP 0,1641. Por dataset @0,54→@0,58: mitdb 0,1035→0,0416; incart
  0,1193→0,0153; svdb 0,4672→0,2734.
- Validação do próprio fold produtor: Se 0,2287 @0,54, PPV 0,1565, MCC 0,0305,
  `passes_qg5=false`; run com 2 folds (231.134/230.480).
- Prova DQ-02: razões 0,72 / 0,256 / 0,514 exatas vs anotações brutas (mitdb/svdb/incart).
- Classes (stage1): N 406.453/200 pac.; S 16.934/141 (72% svdb); V 37.183/174 (54% incart); F
  1.044/45 (top-5=82%); Q 11.422/83 (top-5=74,8%).

**Comandos (todos somente leitura):** `pytest tests/test_two_stage_qg5.py::test_two_stage_qg5_end_to_end
-p no:cacheprovider -v -s` (falha reproduzida); scripts de análise em `/tmp/qg5_forensics/` e
`/tmp/data_forensics/` (nada escrito no projeto).

---

## 8. Pontos de decisão humana

> Aviso transversal: corrigir DQ-01/DQ-02 regenera os datasets com novos hashes e **invalida
> todos os modelos, scalers, thresholds, baselines e relatórios anteriores**. Nada abaixo foi
> executado.

1. **Correção do relógio de anotações (DQ-01/DQ-02).** Reescalonar índices para 500 Hz antes da
   segmentação e das features; regenerar NPZs v3 versionados. Desbloqueia: qualquer treinamento
   cientificamente válido. Opções: corrigir e reprocessar tudo (recomendado) vs manter dados
   atuais congelados como contrafactual documentado.
2. **AFDB e a classe F (DQ-03/DQ-13).** Integrar `afdb_beat_loader` (F passa a incluir AFIB/AFL
   reais, ganhando ~23 registros longitudinais) vs excluir AFDB formalmente e redefinir F como
   "fusão" apenas. Impacta a meta `F1(F) ≥ 0,50` e o significado clínico da classe.
3. **Política da classe Q (DQ-05/DQ-12).** Manter Q em Anormal sem destino no Stage 2 (atual) vs
   excluir Q do Stage 1 vs criar destino explícito. Exige unificar as três cópias do mapa AAMI
   em tabela única versionada.
4. **Governança e promoção (DQ-06).** Ratificar a política de bundle imutável (bloquear
   `passes_qg5=false`, threshold não co-produzido, split não verificável) e decidir o destino do
   artefato atual em `models/` (manter como baseline congelado documentado vs aposentar).
5. **Splits e protocolo (DQ-11).** Adotar manifests congelados fold–seed–hash por run; proibir
   execuções fora do protocolo (a run 2-fold não seguiu o GroupKFold-5 configurado).
6. **Qualidade de sinal e deduplicação (DQ-04/DQ-07).** Aprovar regras de exclusão (flatline,
   clip) com registro de cada exclusão (id, motivo, regra, etapa, classe, paciente, impacto na
   prevalência) e a regra de deduplicação para os 7 pares conflitantes.
7. **Redesenho do subset do teste QG5 (DQ-16).** Substituir "primeiros-N-por-classe" por
   estimativa agrupada por paciente com IC, ou aceitar o subset atual como stress test
   explicitamente rotulado como tal (documentando que não estima recall populacional).

---

## 9. Estado final

- **Classificação da investigação:** `MULTIPLE_FAILURES_REVIEW_REQUIRED`
  (PREPROCESSING_DRIFT + LABEL_QUALITY_FAILURE + FEATURE_LEAKAGE +
  SPLIT_PROVENANCE_UNVERIFIABLE + INSUFFICIENT_CLASS_SUPPORT (F) + ARTIFACT_LINEAGE_FAILURE +
  POST_TRAINING_GOVERNANCE_FAILURE + DATASET_REPRESENTATION_FAILURE).
- **Decisão:** `REVIEW_REQUIRED`.
- Nenhum gate de prontidão passou (integridade, labels, splits, pré-processamento,
  representatividade, estatístico). Nada neste relatório autoriza treinamento, correção,
  promoção ou deployment. Aguardando resposta humana aos 7 pontos da seção 8.
