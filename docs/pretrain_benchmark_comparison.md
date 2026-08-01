# Comparativo — backbones pré-treinados (Chapman) vs. benchmarks externos

Data: 2026-07-30 | Fontes locais: `experiments/*_pretrain_chapman` (artefatos com SHA-256) |
Fontes externas: URLs citadas inline (todas acessadas em 2026-07-30)

> Escopo: análise dos últimos modelos pré-treinados do projeto e posicionamento
> frente à literatura. Nenhum número externo sem URL é afirmado como fato;
> divergências de protocolo são marcadas explicitamente.

---

## 1. Resumo executivo

| Run | Arch | Params | Seed / Det. | val_auc_roc | val_auc_pr | val_loss (BCE) | ECE→(T) | QG4 |
|---|---|---|---|---|---|---|---|---|
| A0 histórico `20260728_033533` | A0 | 19.933 | 42 / pré-strict | 0.8333 | 0.6734 | 0.3907 | 0.055→0.023 (0.761)* | FAIL |
| A0 novo `20260729_042301` | A0 | 19.933 | 42 / **strict** | **0.8365** | 0.6784 | **0.3880** | 0.025→0.020 (0.913) | FAIL |
| **A2-full `20260728_053011`** | A2 (A1+focal) | 32.005 | 13 / strict | **0.8596** | **0.7008** | 0.4226 | 0.151→0.0152 (0.374) † | FAIL (AUC PASS) |

\* ECE do run E0-triagem proxy (mesma arquitetura/seed, 5 épocas).

† A2-full: ECE oficial com **n_bins=15** (pós-T1, `calibration.json` versionado); valores
A0 calculados com n_bins=10. AUC do A2-full: 0.8596 é a métrica de log Keras
(batch-averaged, treino); a métrica de avaliação **offline** (macro por classe,
`metrics_per_class.json` pinado no provenance) é **0.8639**.

Leituras centrais:

1. **A2-full é o melhor modelo já produzido** pelo projeto (AUC 0.8596 / PR-AUC 0.7008),
   superando o A0 em +2,3 p.p. AUC e +2,2 p.p. PR-AUC — e é o primeiro a passar o braço
   AUC do QG4 (> 0.85). O braço BCE (< 0.15) permanece fora de alcance para todas as variantes.
2. **Reprodutibilidade strict funciona**: o A0 novo (strict, oneDNN off) reproduz o histórico
   (pré-strict) com delta de apenas +0,3 p.p. AUC (0.8365 × 0.8333) e loss −0,003 —
   variação compatível com a reordenação numérica do oneDNN, não com mudança de pipeline.
3. **Calibração**: o A0 (BCE) é naturalmente bem calibrado (ECE 0.025); o A2 (focal) sai
   **sub-confiante** (underconfident; ECE 0.151; reliability diagram: pred 0.86 → obs 0.99
   em NORM). T = 0.3741 (< 1) afia as probabilidades e corrige o descalibramento
   (ECE → **0.0152**, n_bins=15) — consistente com Mukhoti 2020 (focal loss reduz
   overconfidence e pode inclinar para underconfidence em datasets desbalanceados) e com
   a prática pós-hoc de temperature scaling (Guo 2017).

---

## 2. Reprodutibilidade (A0 novo × A0 histórico)

Mesma arquitetura (A0, 19.933 params), mesma seed (42), mesmo split (40.637/4.515),
mesma config; diferença: `deterministic.mode=strict` (TF_ENABLE_ONEDNN_OPTS=0) no novo.

| Métrica (best epoch) | Histórico (e28) | Novo (e30) | Δ |
|---|---|---|---|
| val_auc_roc | 0.8333 | 0.8365 | +0,0032 |
| val_auc_pr | 0.6734 | 0.6784 | +0,0050 |
| val_loss | 0.3907 | 0.3880 | −0,0027 |

O novo run ainda melhorava na época 30 (best_epoch=30, sem early stop) — tendência
monotônica em ambos. Conclusão: a política strict introduz determinismo **sem** custo
mensurável de desempenho; o delta é ruído numérico esperado da troca de backend (oneDNN
custom ops → Eigen/CPU padrão), conforme o próprio aviso do TF sobre "slightly different
numerical results".

---

## 3. Comparativo interno por classe (PR-AUC / F1@0.5)

| Classe | A0 novo (PR / F1) | A2-full (PR / F1) | Δ PR (A2−A0) |
|---|---|---|---|
| NORM (n≈33,7k) | 0.976 / 0.926 | 0.989 / 0.956 | +0.013 |
| CD (n≈7,4k) | 0.545 / 0.367 | 0.556 / 0.388 | +0.011 |
| MI (n≈13,2k) | 0.606 / 0.505 | 0.625 / 0.493 | +0.019 |
| HYP (n≈10k) | 0.490 / 0.321 | 0.508 / 0.421 | +0.018 |
| STTC (n≈12,5k) | 0.775 / 0.692 | 0.855 / 0.787 | **+0.080** |

- O ganho do A2 concentra-se onde importa: **STTC** (+8,0 p.p. PR-AUC) e **HYP** (F1
  0.321→0.421, recall 0.221→0.337) — classes tipicamente mais difíceis e desbalanceadas.
- **CD e HYP seguem como gargalo absoluto** (PR-AUC ≤ 0.56 em todas as variantes) —
  limitação arquitetural/dados, não de treino (coerente com o gap para o QG4-BCE).

---

## 4. Comparativo externo (com ressalvas de protocolo)

### 4.1 Chapman-Shaoxing — paper original (Zheng et al., 2020)

Fonte: [Sci Data 7, 48 (2020)](https://www.nature.com/articles/s41597-020-0386-x)

- O paper valida o dataset com **XGBoost sobre 230 features manuais** (não deep learning
  em sinal bruto), reportando **F1 = 0.97** — mas em **4 grupos de ritmo agregados**
  (SB/AFIB/GSVT/SR), 10-fold CV, features de 12 leads.
- **Ressalva**: tarefa diferente da nossa (superclasses SCP-ECG multi-label em 5 classes
  sobre sinal bruto de 1 lead). O F1=0.97 demonstra a qualidade dos rótulos, não um teto
  comparável ao nosso cenário. Nosso F1 macro ponderado por classe (A2-full): NORM 0.956 /
  CD 0.388 / MI 0.493 / HYP 0.421 / STTC 0.787 — lacuna esperada dado que o baseline do
  paper usa features clínicas explícitas e 12 leads.

### 4.2 PTB-XL / SCP-ECG superclasses — Strodthoff et al., 2021

Fontes: [arXiv:2004.13701](https://arxiv.org/abs/2004.13701) |
[IEEE JBHI 25(5):1519](https://www.medsci.cn/sci/show_paper.asp?id=bb93e12a8c2013c5)

- Mesmo esquema de 5 superclasses SCP-ECG (NORM/CD/MI/HYP/STTC), métrica primária
  **macro-AUC term-centric** (mesma definição do nosso `multi_label=True`).
- Modelos resnet/inception atingem **macro-AUC ~0.93** nas categorias diagnósticas
  (diag./sub-diag./super-diag.) — com 12 leads, modelos de ~0,5–8 M params, fs=100 Hz.
- **Nosso A2-full: 0.8596** com **1 lead**, **32k params** (~0,4–6 % do tamanho), fs=500 Hz.
  A distância de ~7 p.p. para a referência PTB-XL é plausívelmente decomposta em
  (i) leads (1 × 12), (ii) orçamento de modelo (32k × milhões), (iii) dataset.
  Registrado como **referência indireta**, não como meta.
- O resultado de transferência do paper (pré-treino ajuda em regime de poucos dados)
  sustenta a estratégia do projeto: pré-treino Chapman → fine-tuning MIT-BIH.

### 4.3 PhysioNet/CinC Challenge 2021

Fontes: [página oficial do Challenge](https://moody-challenge.physionet.org/2021/) |
[resultados oficiais](https://moody-challenge.physionet.org/2021/results/) |
[Reyna et al., CinC 2021](https://ieeexplore.ieee.org/abstract/document/9662687/)

- 131.155 registros de 7 instituições (inclui **Chapman-Shaoxing + Ningbo** como dados
  públicos de treino); 68 equipes, 39 classificadas; métrica oficial = score customizado
  da competição (não AUC), com AUROC/AUPRC/F-measure reportados em CSVs oficiais.
- Vencedor: **ISIBrno-AIMT** (ensemble de ResNets com atenção, 12 leads) — reforça a
  escolha arquitetural do projeto (blocos residuais da A1) e indica o teto da competição
  com ensembles grandes.
- Nosso cenário (single-lead, TinyML, 32k params) não disputa a mesma liga — a comparação
  útil é: o Challenge usa **os mesmos dados públicos** que nosso pré-treino, e as tarefas
  análogas (multi-label, classes desbalanceadas) enfrentam os mesmos gargalos de classes
  raras (nosso CD/HYP).

### 4.4 TinyML / ECG em MCU

Fonte MCP-arXiv: [Family-FL Tiny, arXiv:2605.18862](https://arxiv.org/pdf/2605.18862v1)

- Trabalhos recentes de ECG em microcontrolador operam de **centenas** (669 params /
  4,65 KB Flash, acurácia 91,9 %, F1-macro 0,483 no MIT-BIH) a dezenas de milhares de
  parâmetros. Nosso orçamento (19,9k–32k params, 25–40 KB INT8) fica na faixa média —
  compatível com a restrição do projeto (Flash < 512 KB, arena < 64 KB), com AUC macro
  0.86/0.83 em tarefa de 5 superclasses (mais rica que o binário/ritmo típico de MCU).

### 4.5 Calibração — focal × BCE

Fontes: [Mukhoti et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/aeb7b30ef1d024a76f21a1d40e30c302-Abstract.html) (via MCP-scholar) |
Guo et al. 2017 (temperature scaling)

- Mukhoti: focal loss tende a produzir modelos **melhor calibrados** que cross-entropy e,
  combinada com temperature scaling, atinge calibração estado-da-arte.
- Observado aqui: A2-focal sai **sub-confiante cru** (ECE 0.151, T=0.374 < 1 ⇒
  underconfidence — reliability diagram: pred 0.86 → obs 0.99 em NORM) e, após
  temperature scaling, atinge **ECE 0.0152** (n_bins=15) — melhor que o A0-BCE calibrado
  (0.020). Ou seja: cru, o resultado é **consistente** com Mukhoti 2020 (focal loss reduz
  overconfidence e pode inclinar para underconfidence em datasets desbalanceados);
  pós-T, confirma o ganho prático da combinação focal+T.

> **Nota (T1.5):** ECE mede a *magnitude* do descalibramento, não a *direção*. A direção
> (over vs. under) é determinada pelo reliability diagram e pelo sinal de T − 1
> (T < 1 ⇒ underconfidence; T > 1 ⇒ overconfidence). Versões anteriores deste doc
> liam o ECE 0.151 como "superconfiante" — leitura incorreta da magnitude como direção.

---

## 5. Posição consolidada

| Dimensão | Estado | Evidência |
|---|---|---|
| Melhor modelo | **A2-full** (AUC 0.8596, PR 0.7008) | `experiments/20260728_053011_pretrain_chapman` (provenance + hashes) |
| Reprodutibilidade | strict OK (Δ ≈ 0,3 p.p.) | A0 novo × histórico |
| vs. Zheng 2020 | tarefa diferente; qualidade dos rótulos confirmada | Sci Data 7:48 |
| vs. Strodthoff PTB-XL | −7 p.p. AUC com 1 lead × 12, 0,4–6 % do orçamento | arXiv:2004.13701 |
| vs. CinC 2021 | liga diferente; mesmos dados públicos; gargalos coincidentes | physionet.org results |
| Orçamento TinyML | na faixa (25–40 KB INT8) | arXiv:2605.18862 |
| QG4 | FAIL honesto (AUC PASS, BCE FAIL) | `docs/qg4_analysis.md` |

## 6. Limitações desta análise

1. Métricas internas calculadas **apenas** no split de validação record-disjoint
   (sem teste externo independente nesta sessão).
2. Comparações externas têm protocolos diferentes (leads, dataset, métrica, orçamento) —
   todas marcadas como referências indiretas.
3. Score exato da equipe vencedora do CinC 2021 não foi re-verificado numericamente aqui
   (CSVs oficiais estão na página de resultados citada); optou-se por não citar número
   não verificado.
4. Threshold QG4-BCE (0.15) permanece inalterado e não atingido — revisão somente via
   governança.
