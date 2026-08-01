# Auditoria de Engenharia de Algoritmos v1 — Gate pré-pré-treino (T10.1)

**Versão:** v1.0.0 · **Data:** 2026-08-01 · **Branch:** `develop` · **Task:** T10.1 (SDD-LEWIS ML Protocol v2)
**Escopo:** auditoria matemático-algorítmica dos 3 runs de pré-treino Chapman (A0 histórico, A0 novo,
A2-full). **Nenhum treino, piloto ou inferência foi executado; nenhum código de QG foi alterado.**
Referência normativa: `docs/ml_protocol_v2.md`.

> Todos os números deste documento foram extraídos de artefatos versionados em `experiments/` ou
> pinados em código-fonte (`arquivo:linha`). Derivados analíticos (receptive field, MACs) são
> marcados como *(derivado)*. O que não existe em artefato está marcado **HIPÓTESE ABERTA** ou
> **PENDÊNCIA** — nada aqui é estimativa não marcada.

---

## 1. Resumo executivo e decisão

O A2-full é o melhor modelo do projeto (AUC offline 0,8639), mas a auditoria mostra que **o
experimento que o coroou não permite atribuir a causa do ganho**: arquitetura (A0→A1), loss
(BCE→focal) e seed (42→13) mudaram simultaneamente, e a célula de controle `A1 + BCE` nunca foi
treinada. Além disso, o pipeline atual viola três regras do ML Protocol v2:

1. **Early stopping por `val_loss`** (focal, no A2) em vez de métrica equalizada
   (`src/models/pretrain_chapman.py:50-55`) — o checkpoint embarcado é o de melhor focal, não o de
   melhor discriminação/calibração.
2. **Thresholds tunados no mesmo split de validação** usado para early stopping e avaliação
   (`evaluation_report.json › best_f1_thresholds_analysis_only`) — protocolo v2 exige split de
   calibração separado.
3. **Splits distintos entre runs** (seed 42 vs 13 → registros de validação diferentes, suportes
   divergentes) — comparações A0×A2 são indicativas, não pareadas (ML Protocol v2, seção 7).

**DECISÃO (registrada conforme owner):**

```text
Novos pré-treinos oficiais (T11): BLOQUEADOS até T10.1 + T9.2 concluídos.
Pilotos experimentais (T10.3): PERMITIDOS só após T9.2 + configs v2, status PILOT
  (nunca benchmark, promoção ou publicação).
Promoção/publicação: BLOQUEADAS por governança (HOLD E07R permanece).
```

---

## 2. Inventário dos modelos (fonte por célula)

| Campo | A0 histórico `20260728_033533` | A0 novo `20260729_042301` | A2-full `20260728_053011` | Fonte |
|---|---|---|---|---|
| Backbone | A0 (`a0_baseline`) | A0 (`a0_baseline`) | A1 (`a1_stable`, arch="a2") | config.json de cada run |
| Parâmetros | 19.933 | 19.933 | 32.005 | config.json; `tests/test_backbone_budget.py` |
| MACs/inferência *(derivado)* | ≈ 2,06 M | ≈ 2,06 M | ≈ 2,99 M | shapes em model_summary.txt |
| Receptive field *(derivado)* | 30 amostras ≈ **60 ms** | 60 ms | 94 amostras ≈ **188 ms** | `backbone_1d.py:56-143`; `a1_stable.py:38-55` |
| Normalização | nenhuma (sem BN/LN/GN — restrição TFLM) | idem | idem | `backbone_1d.py:12-14`; `a1_stable.py:1-11` |
| Dropout | 0,3 (apenas pós-GAP) | 0,3 (pós-GAP) | 0,3 (pós-GAP) | `backbones/spec.py:15` |
| Head | Dense(80,relu)→Dense(5,sigmoid) | idem A0 | GAP→Dropout→Dense(5,sigmoid) (linear) | `backbone_1d.py:117-131`; `a1_stable.py:52-54` |
| Loss | BCE | BCE | **focal γ=2,0 fixo, sem α, sem pos_weight** | `pretrain_losses.py:74` |
| Otimizador | Adam(lr=1e-3), **sem weight decay, sem clipping, sem warmup** | idem | idem | `pretrain_chapman.py:264` |
| Schedule | ReduceLROnPlateau(0,5, patience 3, min 1e-6) | idem (1ª queda ép. ~28) | idem (1ª queda ép. 15) | `pretrain_chapman.py:56-62`; history.json |
| Early stopping | `val_loss`, patience 5, restore best | idem — **não disparou** (val_loss ainda caindo na ép. 29) | idem — **sobre focal**, best ép. 28 | `pretrain_chapman.py:50-55`; history.json |
| Batch / épocas | 64 / 30 | 64 / 30 | 64 / 30 | config.json |
| Seed / modo | 42 / pré-strict | 42 / strict (oneDNN off) | **13** / strict | config.json; `docs/pretrain_benchmark_comparison.md` |
| Split | record-disjoint val 0,1, **não estratificado** | idem (seed 42) | idem (seed 13 — **registros diferentes do A0**) | `chapman_dataset.py:157-175`; calibration.json |
| Métricas de treino | AUC ROC/PR multi-label | idem + bce_monitor n/a | idem + `bce_monitor` (BCE auxiliar p/ QG4) | `pretrain_chapman.py:257-262` |
| Artefatos | esparsos (sem history/per-class/calibração) | completos | completos + `quantized/` | listagem dos diretórios |

**Lacuna de inventário:** o A0 histórico não tem `history.json`, `metrics_per_class.json` nem
`calibration.json` — reconciliação detalhada dele fica limitada ao `metrics.json` (PENDÊNCIA T9.3).

---

## 3. Camada 1 — Formulação do problema

### 3.1 Multi-rótulo confirmado pelos dados

Computado nesta auditoria sobre `data/catalog/dataset_catalog.jsonl` (44.986 registros Chapman com
diagnóstico; 166 sem rótulo, excluídos; mapeamento `src/data/chapman_labels.py`):

| Classe | Prevalência | P(NORM\|classe) | P(MI\|classe) | P(STTC\|classe) |
|---|---:|---:|---:|---:|
| NORM | 75,1% (33.768) | — | 0,21 | 0,06 |
| CD | 16,2% (7.291) | 0,57 | 0,41 | 0,47 |
| MI | 29,3% (13.180) | 0,54 | — | 0,50 |
| HYP | 21,8% (9.805) | 0,58 | 0,45 | 0,47 |
| STTC | 27,3% (12.259) | 0,18 | 0,53 | — |

Média de **1,70 rótulos por registro** → a formulação multi-rótulo (sigmoid) está correta.
MI↔STTC co-ocorrem em ~50% dos casos (clinicamente esperado: isquemia ⇄ alterações ST-T).

### 3.2 Achado: NORM não é "ausência de doença"

**57–58% dos registros CD/MI/HYP também carregam o rótulo NORM.** NORM se comporta como
"ritmo sinusal presente", não como "registro saudável" — a premissa implícita NORM ⊥ patologia é
falsa nos dados. Consequências:

- métricas de NORM (F1 0,956) medem uma tarefa mais fácil que "detectar ausência de patologia";
- macro-métricas são infladas por NORM (já mitigado pelo protocolo v2 via per-class + PR-AUC);
- **HIPÓTESE ABERTA H-FORM-1:** redefinir `NORM := ausência das outras 4` mudaria a semântica da
  classe e possivelmente a dificuldade da tarefa. Decisão ontológica — exige governança (C03/C04)
  antes de qualquer experimento.

### 3.3 Achado: split record-disjoint **não estratificado**

`chapman_split_record_sets` faz shuffle seeded e corte 90/10 sobre registros, sem estratificação por
classe (`chapman_dataset.py:167-174`). Com CD a 16,2%, a composição do val varia com a seed — o que
agrava a não-paridade A0 (seed 42) × A2 (seed 13): suportes de val divergem (NORM 33.740 vs
33.750; CD 7.410 vs 7.350). **Correção entra na matriz de ablações (S5) e nos configs v2 (T9.4).**

### 3.4 Janela de 1000 ms

Ver Camada 3 (receptive field). **HIPÓTESE ABERTA H-SIG-1:** janela maior (2000 ms) pode ajudar
MI/STTC/HYP, que dependem de contexto ST-T e ritmo; custo: 2× latência/SRAM — avaliar contra
QG9 (< 200 ms; A2-full hoje: 73 ms, `reports/firmware_simulation_report_a2_full.json`).

---

## 4. Camada 2 — Representação do sinal

Contrato vigente (C02): 500 Hz, lead única, 1000 ms = 500 amostras, bandpass 0,5–40 Hz, detrend,
Z-score global (`config/preprocess_v1.0.yaml`; AGENTS.md Regra 8).

| Pergunta | Estado da evidência |
|---|---|
| Filtro 0,5–40 Hz remove informação de ST/HYP? | **HIPÓTESE ABERTA** — o corte em 0,5 Hz atenua componentes de deriva lenta do segmento ST; nenhuma ablação de filtro foi feita. Análise pendente: espectro por classe (T10.2). |
| Z-score global vs por janela? | **HIPÓTESE ABERTA** — sem ablação. Z-score global é Regra de Ouro 8; mudar exige RFC. |
| Lead única é teto para CD/HYP? | Provável mas não quantificável com os artefatos atuais — **limitação estrutural** (contrato 1 lead). Benchmark externo (12 leads) já marcado NON_COMPARABLE em `docs/pretrain_benchmark_comparison.md`. |
| Features DSP explícitas ajudariam? | **HIPÓTESE ABERTA** — `build_backbone_1d_with_features` já existe (`backbone_1d.py:146-232`) e nunca foi usado no pré-treino. Candidato barato de ablação (A7). |

---

## 5. Camada 3 — Arquitetura

### 5.1 Receptive field *(derivado)*

Fórmula padrão RF_out = RF_in + (k−1)·J_in (J = jump acumulado), sobre as definições em código:

- **A0: RF = 30 amostras ≈ 60 ms** (conv 7/5/3, 3 pools stride 2).
- **A1: RF = 94 amostras ≈ 188 ms** (stem k7; blocos residuais k5/k5/k3; pools ×2×2×2).

Referências fisiológicas a 500 Hz: QRS largo (bloqueios, CD) ~120–200 ms = 60–100 amostras;
intervalo QT ~350–450 ms = 175–225 amostras; ciclo cardíaco completo ~1000 ms. **Ambos os
backbones têm RF menor que o QT e comparável/maior que o QRS** — o GAP agrega features locais
sobre a janela, mas nenhuma unidade individual "vê" um complexo QT inteiro.

**HIPÓTESE ABERTA H-ARCH-1:** RF curto limita CD (condução: QRS/PR prolongados) e HYP/STRain
(padrões de voltagem + ST-T de longa duração). É a explicação arquitetural candidata para o
gargalo CD/HYP — concorre com a explicação por dados (prevalência) e por loss (Camada 4).

### 5.2 Eficiência

| Modelo | Params | MACs *(derivado)* | FlatBuffer INT8 | Observação |
|---|---:|---:|---:|---|
| A0 | 19.933 | ≈ 2,06 M | ~26 KB (est. docstring) | 2 blocos densos (embedding 80) |
| A1/A2 | 32.005 | ≈ 2,99 M | 54,77 KB medido | residual; head linear direta |

A1 entrega +61% de params por +45% de MACs e ganhou +2,3 p.p. AUC — custo/benefício favorável,
mas a atribuição arch-vs-loss é indeterminada (ver pergunta 1, seção 11).

### 5.3 Ausência deliberada de normalização

Sem BatchNorm/LayerNorm/GroupNorm por restrição TFLM (`backbone_1d.py:12-14`). Estabilidade vem só
das conexões residuais (A1). **HIPÓTESE ABERTA H-ARCH-2:** sem normalização, a escala das
ativações depende inteiramente da init + LR — pode contribuir para a compressão de logits
(underconfidence) observada na Camada 4. Testável via O-ablações (LR/warmup) antes de mexer em
arquitetura.

### 5.4 Regularização estrutural

Dropout 0,3 **apenas após o GAP**; nenhum `kernel_regularizer` usado (parâmetro existe e não é
passado pelos builders A0/A1); nenhuma augmentação no pré-treino. Head do A1 é linear pura
(GAP→dropout→sigmoid) — capacidade decisória mínima.

---

## 6. Camada 4 — Loss e calibração (camada crítica)

### 6.1 Fatos pinados

- Focal = `BinaryFocalCrossentropy(gamma=2.0)` — **γ fixo, nunca varrido; sem α; sem class
  weights** (`pretrain_losses.py:74`). A maquinaria de `pos_weight` (treino-only, clip [1,10])
  existe e só é usada por `bce_weighted` (`pretrain_losses.py:21-62`) — **o A2 não a usou**.
- Early stopping/checkpoint monitoram `val_loss` — para o A2, **focal** (`pretrain_chapman.py:50-68`).
- Focal comprime probabilidades: p_t alto nos exemplos fáceis (NORM, 75% dos registros) recebe
  fator (1−p_t)² → gradiente suprimido globalmente, não só nas raras.

### 6.2 Evidência de underconfidence (A2-full)

| Métrica | Pré-T | Pós-T (T=0,3741) | Fonte |
|---|---:|---:|---|
| ECE macro (n_bins=15) | 0,1508 | **0,0152** | calibration.json |
| ECE por classe | NORM 0,129 · CD 0,169 · MI 0,141 · HYP 0,147 · STTC 0,168 | todas ≤ 0,019 | calibration.json › before/after |
| Brier macro | 0,1327 | 0,1062 | calibration.json |
| NLL (= BCE offline) | 0,4317 | **0,3417** | calibration.json › temperature_scaling |
| AUROC macro | 0,8639394699 | 0,8639394700 (**Δ ≈ 8e-11** — invariância verificada) | idem |

A descalibração é **global** (ECE 0,13–0,17 em todas as classes, incluindo NORM) — consistente com
focal γ=2 comprimindo p_t em toda a distribuição, não um efeito restrito às raras. T = 0,3741 ≪ 1
afia as probabilidades e praticamente zera o ECE. Leitura de direção: T−1 = −0,63 (underconfidence),
coerente com o reliability diagram (pred 0,86 → obs 0,99 em NORM, `docs/pretrain_benchmark_comparison.md`).

### 6.3 Achado: divergência BCE × focal durante o treino (history.json do A2-full)

```text
val_bce_monitor: mínimo 0,4226 na época 17 → sobe para 0,4554 na época 29
val_loss (focal): mínimo 0,0898 na época 28  → checkpoint salvo/restaurado = época 28
val_auc_roc:      máximo 0,8639 na época 28
```

Ou seja: **a partir da época 17 o modelo melhora discriminação e piora BCE** — a compressão de
probabilidade se aprofunda com o treino focal. Dois desdobramentos:

1. O QG4 reportou BCE = 0,4226 (mínimo do monitor, ép. 17), mas o checkpoint embarcado (ép. 28)
   tem BCE offline ≈ 0,4317 (NLL pré-T do calibration.json). **QG4 julgou uma época que não é o
   artefato** — item de reconciliação obrigatório para a RFC T9.5.
2. **bce_post_temperature ≈ 0,3417** (NLL pós-T, mesma redução média-por-elemento — confirmação
   formal da equivalência NLL≡BCE fica para o avaliador canônico, T9.2/T9.3). Mesmo calibrado,
   0,3417 ≫ 0,15: o braço BCE do QG4 mede algo que nem a calibração perfeita (ECE 0,015) alcança —
   evidência central para a RFC: **0,15 é inatingível sob prevalências reais** (BCE de referência
   do prior de NORM já ≈ 0,56 por classe-amostra; baseline teórico entra na T9.5).

### 6.4 Achado: threshold tuning fora do protocolo

`evaluation_report.json › best_f1_thresholds_analysis_only` (A2-full): thresholds 0,4/0,4/0,4/0,45
para CD/MI/HYP/STTC elevam macro-F1 de **0,6089 → 0,6821** (+7,3 p.p.; CD +12,8 p.p., HYP +13,6
p.p.) — mas foram computados **no mesmo val usado para early stopping e avaliação** (o próprio
artefato marca `analysis_only`). Protocolo v2 seção 6 exige `fit_split: calibration`. O ganho é
real como potencial; como evidência é otimista. **T9.2/T9.3 devem refazer com split de calibração.**

### 6.5 Respostas parciais de calibração

- T global sobrevive ao INT8: ECE pós-PTQ pré-T 0,1636 → pós-T 0,0207 (`post_quant_calibration.json`).
- ECE pós-T por classe ≤ 0,019 em float ⇒ **T global é suficiente em magnitude**; Platt/vector
  scaling por classe (C2/C3) só se justifica se ECE por classe importar abaixo de 0,02 — prioridade
  baixa. MCE por classe permanece alto pós-T (HYP 0,178) — bins de cauda; monitorar, não otimizar.

---

## 7. Camada 5 — Otimização

| Item | Estado | Fonte |
|---|---|---|
| Otimizador | Adam puro (β padrão), lr 1e-3 | `pretrain_chapman.py:264` |
| Weight decay | **ausente** (não é AdamW) | idem |
| Warmup / clipping / EMA / SWA | **ausentes** | `_make_callbacks`, `pretrain_chapman.py:43-74` |
| Schedule | ReduceLROnPlateau ×0,5, patience 3 | `pretrain_chapman.py:56-62` |
| LR finder / tuning de lr | **nunca executado** | ausência em código e artefatos |
| Early stopping | `val_loss` (não equalizado) | `pretrain_chapman.py:50-55` |

Evidência nos histories:

- **A0 novo (BCE) não convergiu em 30 épocas:** val_loss estritamente decrescente até a época 29
  (min = última época), ES nunca disparou, LR só caiu na ép. ~28. O A0 está **sub-treinado** —
  parte do gap A0×A2 pode ser orçamento de treino, não arquitetura. Confundidor adicional para a
  pergunta 1.
- A2-full: LR caiu na ép. 15 (5e-4); melhor val_bce na ép. 17; melhor focal na ép. 28.
- Gaps train/val pequenos em ambos (A0 novo: loss 0,3947 vs val 0,3880 — val abaixo do treino,
  típico de dropout ativo no treino; A2: 0,0942 vs 0,0914, mesma assinatura) ⇒ **sem overfitting
  aparente**; a regularização atual é suficiente para este orçamento — e possivelmente excessiva
  para o A0 sub-treinado.

---

## 8. Camada 6 — Regularização e generalização

- Única regularização: dropout 0,3 pós-GAP + early stopping. Sem augmentação (Regra 7 permite só
  no fine-tuning; pré-treino roda sem), sem weight decay, sem label smoothing.
- Split record-disjoint evita leakage de segmentos do mesmo registro (`chapman_dataset.py:162-165`)
  — correto para Chapman (1 registro ≈ 1 paciente, 10 s).
- **PENDÊNCIAS** (não existem em artefato): curvas por classe ao longo do treino, análise de erro
  por paciente/registro, clusters de erro, robustez a ruído, saliency/Grad-CAM. Todas dependem do
  avaliador canônico (T9.2) ou de análises dedicadas (T10.2).

---

## 9. Camada 7 — Quantização e inferência (A2-full)

| Item | Valor | Fonte |
|---|---:|---|
| ΔAUC float→INT8 | 0,0027 | quant_report.json |
| ΔF1-macro float→INT8 | 0,0024 (INT8 levemente maior) | quant_report.json |
| Saturação logits INT8 | 0,0315% (rails: 14 baixo + 57 alto) | quant_report.json |
| ECE pós-PTQ: pré-T → pós-T | 0,1636 → 0,0207 (float: 0,1508 → 0,0152) | post_quant_calibration.json |
| T aplicado | float32 após dequant (não amplifica saturação INT8) | quant_report.json › note |

- T global **sobrevive à quantização** (Δ ECE-pós-T = +0,0055 vs float) — PTQ + T fixo é suficiente
  hoje; QAT (C-ablação futura) só se justifica se Δs crescerem em modelos maiores.
- ECE pós-PTQ por classe pré-T: NORM 0,152, CD 0,178, MI 0,148, HYP 0,157, STTC 0,180 — a
  quantização degrada calibração quase uniformemente (sem concentração nas raras).
- Saturação concentrada nos rails dos **logits** crus (bounds −3,83/+4,82) — fração desprezível;
  monitorar em modelos futuros com logits mais largos (focal γ maior tende a ampliar range).

---

## 10. Diagnóstico por classe (A0 novo BCE × A2-full focal)

> **Ressalva de paridade:** splits de val diferentes (seeds 42/13) — deltas são indicativos, não
> pareados. Suportes: A0N / A2F.

| Classe | Sup. A0N/A2F | PR-AUC A0N→A2F (Δ) | F1@0.5 A0N→A2F (Δ) | Recall A0N→A2F | Hipótese de falha dominante |
|---|---|---:|---:|---:|---|
| NORM | 33.740/33.750 | 0,9763→0,9887 (+0,012) | 0,9258→0,9557 (+0,030) | 0,944→0,966 | — (resolvida) |
| CD | 7.410/7.350 | 0,5451→0,5561 (+0,011) | 0,3673→0,3881 (+0,021) | 0,240→0,260 | **Ranking quase inalterado** ⇒ gargalo de representação/dados, não de loss |
| MI | 13.220/13.060 | 0,6062→0,6252 (+0,019) | 0,5047→0,4931 (**−0,012**) | 0,429→0,389 | Ranking melhora, decisão piora ⇒ threshold 0,5 inadequado (tuned: F1 0,584) |
| HYP | 10.100/9.930 | 0,4902→0,5078 (+0,018) | 0,3213→0,4208 (**+0,099**) | 0,221→0,337 | Ganho quase todo na região de decisão ⇒ efeito focal; ranking continua o pior do modelo |
| STTC | 12.530/12.380 | 0,7745→0,8548 (**+0,080**) | 0,6923→0,7867 (+0,094) | 0,629→0,733 | Única classe com ganho grande de ranking ⇒ representação (residual) + focal |

**Gargalo nomeado: CD e HYP** (PR-AUC 0,556/0,508 vs NORM 0,989). O ganho do A2 concentrou-se em
STTC (ranking) e HYP (decisão); **CD praticamente não melhorou** — é a classe mais resistente e a
candidata nº 1 a gargalo de dados/representação (RF 188 ms ≈ QRS largo no limite; prevalência
mínima 16,2%).

---

## 11. Diagnóstico matemático — as 8 perguntas do owner

1. **O ganho veio da arquitetura ou do focal?** **INDETERMINADO.** A0→A2 mudou arch+loss+seed
   juntos; a célula `A1+BCE` (e idealmente `A0+focal`) não existe. Evidência indireta: STTC ganhou
   ranking (compatível com residual) e HYP ganhou decisão sem ranking (compatível com focal).
   Resolver com a matriz de ablação L×A (seção 13) — **é a pergunta nº 1 do T10.3.**
2. **STTC/HYP: representação ou gradiente?** Misto (evidência acima): STTC = representação;
   HYP = redistribuição de gradiente do focal; **CD não se moveu** — nenhuma das duas alavancas
   atuou nela.
3. **γ do focal adequado?** **NUNCA TUNADO** — γ=2,0 fixo, sem α, sem pos_weight
   (`pretrain_losses.py:74`). A underconfidence global (ECE ~0,15 em todas as classes) é a
   assinatura esperada de γ alto demais para 75% de exemplos fáceis. Ablação L0–L6 obrigatória.
4. **Desbalanceamento tratado de forma ótima?** **NÃO** — focal puro sem pesos de classe, split
   não estratificado, sampling random único. As três alavancas (pesos, sampling, loss) nunca foram
   combinadas nem abladas.
5. **Underconfidence corrigível no treino?** Provavelmente sim: γ menor (L1), BCE+calibração (L0),
   class-balanced focal (L5), ou head com temperatura aprendida regularizada. Evidência: BCE puro
   (A0) sai quase calibrado (T=0,913, ECE 0,025) com mesmo pipeline ⇒ a compressão vem da loss,
   não dos dados nem do otimizador.
6. **Otimização tunada?** **NÃO** — Adam default, lr 1e-3 de catálogo, sem LR finder, weight decay,
   warmup, clipping ou EMA. A0 nem sequer convergiu em 30 épocas. Ablações O0–O4 pendentes.
7. **Janela/kernel/RF/normalização adequados?** RF A1 = 188 ms < QT (~400 ms); sem nenhuma
   normalização (restrição TFLM); kernels mono-escala por bloco. **HIPÓTESE ABERTA** com
   candidatos concretos: multi-scale kernels (A3), janela 2000 ms (custo QG9), normalização por
   janela (RFC — Regra 8).
8. **CD/HYP: dados, arquitetura ou loss?** **HIPÓTESE ABERTA com hierarquia de suspeitos:** (i)
   representação/RF (CD não respondeu a loss nem a residual); (ii) prevalência (CD 16,2%, HYP
   21,8% — mas NORM com 75% não explica sozinho: STTC com 27% tem PR-AUC 0,855); (iii) loss (HYP
   respondeu ao focal na decisão). Desempate exige ablações S (sampling) × A (multi-scale) — CD é
   a classe-sentinela: se S2 (oversample CD) não mover o PR-AUC de CD, o gargalo é representação.

---

## 12. Matriz de hipóteses (consolidada)

| ID | Hipótese | Intervenção | Métrica de sucesso | Risco | Origem |
|---|---|---|---|---|---|
| H1 | Focal γ=2 comprime probabilidades globalmente | L0–L3 (γ sweep + BCE) | ECE pré-T ↓, BCE pós-T ↓, macro PR-AUC ≥ A2 | perder ganho de decisão em HYP | owner + §6.2 |
| H2 | CD/HYP têm poucos exemplos efetivos | S1–S4 (sampling) | PR-AUC CD/HYP ↑ | overfit/overfit em NORM | owner |
| H3 | RF curto limita CD/MI/STTC | A3 multi-scale / janela 2000 ms | PR-AUC CD/MI/STTC ↑ | latência/SRAM (QG9) | §5.1 |
| H4 | Otimização sub-ótima (sem wd/warmup; A0 sub-treinado) | O1–O4 + mais épocas | convergência estável, val PR-AUC ↑ | custo de tuning | §7 |
| H5 | T global insuficiente por classe | C2 Platt / C3 vector | ECE por classe ↓ | complexidade; prioridade **baixa** (ECE já ≤0,019) | §6.5 |
| H6 | Janela 1000 ms curta para MI/STTC | H-SIG-1: 2000 ms | PR-AUC MI/STTC ↑ | 2× latência/SRAM | owner |
| H7 | Ganho A2 é confundido arch×loss×seed | células A1+BCE, A0+focal (mesma seed/split) | atribuição causal | — (pré-requisito de tudo) | §11.1 |
| H8 | Splits divergentes entre runs viciam deltas | split manifest fixo + estratificado (S5) | comparabilidade | — | §3.3 |
| H9 | Sem normalização, escala de ativação instável | O-ablações (warmup/LR) antes de BN-alternativos | estabilidade de loss | restrição TFLM proíbe BN | §5.3 |
| H10 | NORM≠saudável distorce macro | redefinição ontológica (governança) | clareza de métricas | mudança de ontologia = RFC | §3.2 |

---

## 13. Plano de ablações (proposta para T10.2 — **nenhuma execução nesta task**)

Restrições de orçamento para toda variante A*: `params < 64k`, latência < 200 ms, SRAM < 128 KB.
Todas as ablações rodam sob ML Protocol v2: mesmo split manifest, seed fixa, avaliador canônico
(T9.2), early stopping por `val_macro_pr_auc`, thresholds só em calibration.

### 13.1 Loss (prioridade máxima — desempata H1/H7)

| Run | Loss | Hipótese |
|---|---|---|
| L0 | BCE | baseline probabilístico (esperado: ECE baixo, decisão ruim nas raras) |
| L1 | focal γ=0,5 | compressão mínima |
| L2 | focal γ=1 | intermediário |
| L3 | focal γ=2 (A2 replica) | ponto atual |
| L4 | focal γ=3 | compressão extrema (teste de dose-resposta de H1) |
| L5 | focal + pos_weight (maquinaria já existe) | balanceamento barato |
| L6 | class-balanced focal (effective number) | balanceamento teórico |
| L7 | ASL (asymmetric loss multi-label) | suprime fáceis negativos sem comprimir positivos |

### 13.2 Sampling

| Run | Sampling | Hipótese |
|---|---|---|
| S0 | random (atual) | baseline |
| S1 | class-balanced batch | reduz domínio de NORM |
| S2 | oversample CD/HYP | teste direto de H2 (CD = sentinela) |
| S3 | hard example mining | foca erros |
| S4 | effective number weights | H2 variante teórica |
| S5 | split estratificado por classe (mesma seed) | H8 — pré-requisito de comparabilidade |

### 13.3 Arquitetura

| Run | Variante | Hipótese |
|---|---|---|
| A0/A1/A2 | existentes | referência |
| A3 | multi-scale kernels (3/7/15/31 paralelos) | H3 |
| A4 | depthwise separable (se TFLM permitir — verificar) | eficiência |
| A5 | squeeze-excite leve | recalibração de canais |
| A6 | atenção temporal leve (se orçamento permitir) | H3/H6 |
| A7 | A1 + features DSP (`build_backbone_1d_with_features` já existe) | Camada 2 |

### 13.4 Otimização

| Run | Otimizador/schedule | Hipótese |
|---|---|---|
| O0 | Adam + plateau (atual) | baseline |
| O1 | AdamW (wd 1e-4–1e-2) + cosine | H4 |
| O2 | AdamW + warmup + cosine | H4/H9 |
| O3 | mais épocas (60) + ES por val_macro_pr_auc | A0 sub-treinado |
| O4 | EMA de pesos | suavização |

### 13.5 Calibração (aplicada pós-treino, sobre o melhor L×A)

| Run | Método | Hipótese |
|---|---|---|
| C0 | sem calibração | baseline |
| C1 | temperature global (atual) | ponto atual |
| C2 | Platt por classe | H5 (prioridade baixa) |
| C3 | vector scaling | H5 |
| C4 | isotonic (se suporte permitir) | não paramétrico |
| C5 | temperatura aprendida no head durante treino | H1 estrutural |

---

## 14. Critérios para liberar pré-treinos (T11)

1. T9.2 (avaliador canônico) implementado e testado.
2. T10.2 (matriz de hipóteses) aprovada por governança.
3. Configs v2 (T9.4) versionados com split manifest fixo + estratificado.
4. Células de controle H7 (`A1+BCE`, `A0+focal`) executadas como pilotos (T10.3) — sem elas,
   nenhum pré-treino novo tem interpretação causal.
5. Toda run com `evaluation_v2/metrics.json` (protocolo v2); nenhuma promoção automática.
6. Treinos pesados só com autorização explícita; pilotos com status `PILOT` (nunca CANDIDATE).

---

## 15. Pendências registradas

| Pendência | Destino |
|---|---|
| `bce_post_temperature` formal (equivalência NLL≡BCE na redução usada) | T9.2/T9.3 |
| Threshold tuning em split de calibração (refazer o `analysis_only`) | T9.2/T9.3 |
| A0 histórico sem history/per-class/calibração — reconciliação limitada | T9.3 |
| Espectro de potência por classe; saliency/Grad-CAM; erro por paciente | T10.2 |
| Baseline teórico de BCE sob prevalências reais (entrada da RFC QG4) | T9.5 |
| Verificação de suporte TFLM a depthwise/atenção para A4/A6 | T10.2 |

## Fontes

- Runs: `experiments/20260728_033533_pretrain_chapman/` (metrics.json), `…/20260729_042301_pretrain_chapman/`
  (metrics.json, metrics_per_class.json, history.json, calibration.json, qg4_result.json,
  run_status.json), `…/20260728_053011_pretrain_chapman/` (config.json, history.json,
  metrics_per_class.json, evaluation_report.json, calibration.json, quantized/quant_report.json,
  quantized/post_quant_calibration.json).
- Código: `src/models/pretrain_losses.py:74` (focal γ=2), `src/models/pretrain_chapman.py:50-68,264`
  (callbacks/optimizer), `src/models/backbones/a1_stable.py:38-55`, `src/models/backbone_1d.py:56-143`,
  `src/models/backbones/spec.py:15`, `src/models/chapman_dataset.py:157-175`.
- Dados: `data/catalog/dataset_catalog.jsonl` + `src/data/chapman_labels.py` (prevalência e
  co-ocorrência computadas nesta auditoria, seção 3.1).
- Externo/benchmark: `docs/pretrain_benchmark_comparison.md`; normativo: `docs/ml_protocol_v2.md`.
