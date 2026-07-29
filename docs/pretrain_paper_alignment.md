# Alinhamento bibliográfico — pré-treino Chapman / TinyML ECG

Data: 2026-07-28

## Referências-base

1. **Zheng et al., 2020** — Chapman-Shaoxing 12-lead ECG database
   (*Scientific Data* 7:36 / PhysioNet Challenge 2021). Fonte do dataset de
   pré-treino; 45.152 registros de 10 s @ 500 Hz (Chapman-Shaoxing + Ningbo).
2. **Wagner et al., 2020** — PTB-XL (*Scientific Data* 7:154). Define as 5
   superclasses SCP-ECG (NORM, CD, MI, HYP, STTC) usadas como alvo multi-label
   e valida o protocolo de pré-treino por superclasses.
3. **Strodthoff et al., 2021** — *Deep learning for ECG analysis: benchmarks
   and insights from PTB-XL* (IEEE JBHI). Baselines de CNN residual para
   superclasses SCP-ECG; motiva a variante residual A1 e o uso de PR-AUC em
   cenário desbalanceado.
4. **Saito & Rehmsmeier, 2015** — *The Precision–Recall plot is more
   informative than the ROC plot...* (PLOS ONE). Justifica PR-AUC como métrica
   primária de triagem em classes raras (CD ≈ 16 %).
5. **Guo et al., 2017** — *On Calibration of Modern Neural Networks* (ICML).
   Temperature scaling escalar pós-treino; ECE como métrica de calibração
   (FASE 8).
6. **Lin et al., 2017** — *Focal Loss for Dense Object Detection* (ICCV).
   Base da variante de loss `focal` (A2) para desbalanceamento.
7. **AAMI EC57** — prática de splits por paciente/grupo. Aplicável ao
   fine-tuning beat-level (MIT-BIH); para o pré-treino Chapman o split é
   record-disjoint (cada registro = um exame de 10 s; `patient_id` não existe
   no catálogo → `patient_disjoint: null` registrado na proveniência).
8. **Documentação TFLite/TFLM** — ops built-in suportadas (Conv1D via Conv2D
   lowering, Add, ReLU, MaxPool, GAP, Dense) e orçamento de FlatBuffer
   (QG6/QG7 do projeto).

## Mapa decisão → evidência

| Decisão | Referência |
|---|---|
| Superclasses SCP-ECG multi-label | Wagner 2020; Zheng 2020 |
| Backbone residual (A1) | Strodthoff 2021 |
| PR-AUC como métrica primária | Saito & Rehmsmeier 2015 |
| pos_weight / focal (A2) | Lin 2017 |
| Temperature scaling (A3) | Guo 2017 |
| Sem BatchNorm (restrição TFLM) | TFLM docs + `backbone_1d.py` |
| Split record-disjoint | AAMI EC57 (analogia; patient_id ausente) |

## Lacuna conhecida

O catálogo Chapman não carrega `patient_id` (registros são exames únicos na
fonte pública); portanto o split é por **registro**, não por paciente. Isso é
registrado como `patient_disjoint: null` em `provenance.json` e não afeta o
fine-tuning (MIT-BIH usa GroupKFold por paciente — regra de ouro 4).
