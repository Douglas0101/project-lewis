# PROMPT MESTRE DE ENGENHARIA — STAGE 2 RESEARCH BRANCH v2.4

Atue como engenheiro sênior de machine learning, pesquisador de ECG e responsável técnico pela confiabilidade experimental deste repositório.

Sua missão é investigar e corrigir, de forma científica, incremental, reprodutível e auditável, o baixo desempenho inter-paciente da classe AAMI F no Stage 2.

Não faça tuning cego.

Não altere múltiplas causas conceituais simultaneamente.

Não tente simplesmente aumentar `alpha`, `gamma`, `class_weight`, hidden size ou intensidade de SMOTE.

Não considere um experimento bem-sucedido apenas porque seu melhor fold melhorou.

Não publique automaticamente o melhor fold.

Não use o outer test para escolher arquitetura, hiperparâmetro, threshold, calibrador ou estratégia de sampling.

Trabalhe sequencialmente.

Após CADA implementação, execute obrigatoriamente uma checagem pós-operatória.

Uma etapa só poderá liberar a próxima quando seu checkpoint estiver classificado como:

`PASS`

ou, quando a hipótese tiver sido corretamente testada mas refutada:

`PASS_HYPOTHESIS_REJECTED`

Qualquer falha de integridade deve produzir:

`BLOCKED`

Qualquer regressão técnica corrigível deve produzir:

`ROLLBACK_REQUIRED`

Preserve todos os resultados negativos. Um experimento negativo corretamente executado é evidência científica e não deve ser apagado.

---

# 0. ESTADO CONHECIDO DO PROJETO

Considere inicialmente o seguinte estado, mas valide tudo no repositório antes de modificar código:

* Stage 1 está funcional.
* Adaptação CPU/GPU passou nos testes SLHA existentes.
* `discover_hardware()` funciona em ambiente sem GPU.
* Não foi identificado hardcode obrigatório de GPU nos scripts de treinamento.
* Experimentos Stage 2 v11–v16 utilizaram principalmente mudanças em focal loss, pesos de classe, SMOTE, arquitetura MLP e thresholds.
* Melhor resultado geral inter-paciente observado até o momento: v14.
* v14:

  * `F1-macro ≈ 0.545`
  * `F1(S) ≈ 0.625`
  * `F1(V) ≈ 0.796`
  * `F1(F) ≈ 0.214`
* v15 alcançou aproximadamente `F1(F)=0.24`.
* v16 com class weight F=8.0 regrediu para aproximadamente `F1(F)=0.21`.
* QG5 balanceado apresenta aproximadamente:

  * Stage 1 Recall Anormal = 0.8352
  * Stage 1 Precision = 0.8286
  * Stage 1 F1-macro = 0.9021
  * Stage 2 F1-macro = 0.8654
  * Stage 2 F1(S) = 0.8544
  * Stage 2 F1(V) = 0.8038
  * Stage 2 F1(F) = 0.9379
* O subset QG5 atual amostra até aproximadamente 683 exemplos por classe.
* O threshold publicado do Stage 2 está registrado como:

  * S = 0.5
  * V = 0.5
  * F = 0.8
* Artefatos v2.3 existentes não devem ser sobrescritos.

Antes de iniciar, localize e inspecione especialmente:

* `tests/test_two_stage_mlp_qg5.py`
* `scripts/select_best_mlp_fold.py`
* scripts de preparação do dataset
* scripts de feature extraction
* scripts Stage 2 v11–v16
* implementação de focal loss
* implementação de SMOTE/resampling
* função real de inferência multiclass do Stage 2
* tratamento dos thresholds S/V/F
* arquivos de scaler
* código de split por paciente/grupo
* manifests existentes
* logs/resultados dos folds v11–v16
* modelos e artefatos publicados v2.3

Não presuma nomes adicionais de arquivos. Descubra a estrutura real do repositório.

---

# REGRA GLOBAL DE EXECUÇÃO

Para cada etapa `E00`, `E01`, `E02` etc., seguir exatamente:

## A. PRÉ-CHECAGEM

1. Registrar estado Git atual.
2. Registrar arquivos que serão modificados.
3. Confirmar que artefatos v2.3 não serão sobrescritos.
4. Executar testes relacionados existentes.
5. Registrar baseline de métricas relevante.
6. Criar ou atualizar um log experimental estruturado.

Formato mínimo:

```json
{
  "experiment_id": "E00",
  "status_before": "...",
  "git_commit_or_tree": "...",
  "dataset_manifest_hash": "...",
  "split_manifest_hash": "...",
  "feature_schema_hash": "...",
  "seed": null,
  "files_modified": [],
  "baseline_metrics": {},
  "started_at": "..."
}
```

## B. IMPLEMENTAÇÃO

Implementar somente o objetivo conceitual da etapa.

Não incluir otimizações oportunistas.

Não “aproveitar” para alterar dropout, batch size, optimizer, arquitetura e loss sem autorização da etapa.

## C. CHECAGEM PÓS-OPERATÓRIA

Após a implementação:

1. executar testes unitários;
2. executar testes de integração relacionados;
3. executar `flake8` nos arquivos modificados;
4. executar `mypy` nos arquivos modificados/tipados;
5. validar ausência de NaN e Inf;
6. validar shapes;
7. validar cardinalidade das classes;
8. validar grupos/pacientes;
9. comparar com baseline;
10. gerar artefato de resultado;
11. registrar decisão.

Formato:

```json
{
  "experiment_id": "...",
  "implementation_status": "PASS|FAIL",
  "tests": {},
  "data_integrity": {},
  "metrics": {},
  "regressions": [],
  "warnings": [],
  "hypothesis_result": "SUPPORTED|REJECTED|INCONCLUSIVE",
  "checkpoint": "PASS|PASS_HYPOTHESIS_REJECTED|BLOCKED|ROLLBACK_REQUIRED",
  "next_authorized_step": "..."
}
```

## D. REGRA DE AVANÇO

Não executar a próxima otimização quando:

* houver leakage detectado;
* train e test compartilharem grupos;
* scaler tiver sido ajustado usando outer test;
* feature selection tiver visto outer test;
* thresholds tiverem sido escolhidos pelo outer test;
* qualquer classe esperada tiver desaparecido silenciosamente;
* artefato v2.3 tiver sido sobrescrito;
* dataset manifest não corresponder ao dataset usado;
* split real não corresponder ao split manifest;
* NaN/Inf estiverem presentes;
* resultado não puder ser reproduzido com o mesmo manifest e seed em modo auditável.

Nesses casos, interromper a sequência experimental, corrigir a causa e repetir a checagem da própria etapa.

---

# E00 — SNAPSHOT FORENSE E CONGELAMENTO DO BASELINE

## Objetivo

Congelar o Stage 2 v2.3/v14 como baseline legado imutável e reconstruir exatamente o estado experimental.

## Implementação

Criar uma estrutura equivalente a:

```text
experiments/
  stage2_v2.4_research/
    E00_baseline_snapshot/
```

Criar:

```text
baseline_manifest.json
baseline_metrics.json
baseline_artifacts.json
environment_manifest.json
```

O environment manifest deve registrar, quando disponíveis:

* Python
* TensorFlow
* Keras
* NumPy
* pandas
* scikit-learn
* imbalanced-learn
* CUDA
* cuDNN
* sistema operacional
* CPU
* GPU
* RAM disponível
* dispositivo escolhido pelo pipeline

Calcular SHA-256 dos seguintes artefatos existentes:

* Stage 1 model
* Stage 2 model
* Stage 1 scaler
* Stage 2 scaler
* Stage 1 threshold
* Stage 2 thresholds

Não modificar esses arquivos.

## Reproduzir baseline

Reexecutar, quando tecnicamente possível, a avaliação de v14 utilizando os mesmos:

* dados;
* features;
* grupos;
* folds;
* scaler policy;
* seed;
* loss;
* arquitetura;
* decisão multiclass.

Não retreinar se os metadados necessários para reprodução exata não existirem sem registrar essa limitação.

Nesse caso classificar:

`BASELINE_REPRODUCIBILITY_INCOMPLETE`

e documentar exatamente o dado faltante.

## Checagem pós-operatória E00

Confirmar:

* todos os hashes foram registrados;
* nenhum artefato v2.3 foi alterado;
* todos os paths existem;
* métricas históricas foram importadas corretamente;
* diferenças entre resultado histórico e reprodução foram quantificadas;
* ambiente foi registrado.

Criar teste automatizado que falha caso um comando da research branch tente publicar usando paths v2.3 sem flag explícita de migração/publicação.

## Critério de aprovação

`PASS` quando o baseline estiver congelado e rastreável.

Não exigir reprodução bitwise se o ambiente histórico não puder ser reconstruído.

Entretanto, registrar claramente:

```text
historical_result
reproduced_result
absolute_delta
environment_difference
```

---

# E01 — AUDITORIA DA DISTRIBUIÇÃO DA CLASSE F POR REGISTRO/PACIENTE

## Objetivo

Quantificar a concentração real da classe F antes de qualquer novo treinamento.

## Hipótese

A maior parte da classe F está concentrada nos registros 208 e 213.

Valores conhecidos da documentação oficial devem ser tratados somente como referência de auditoria:

```text
record 208: F ≈ 373
record 213: F ≈ 362
```

Não codificar esses números como resultado.

Recalcular diretamente a partir da fonte local realmente utilizada pelo pipeline.

## Implementação

Criar script equivalente a:

```text
scripts/audit_stage2_patient_distribution.py
```

O script deve gerar:

```text
patient_class_distribution.csv
patient_class_distribution.json
f_concentration_report.json
f_concentration_report.md
```

Para cada grupo/registro:

```text
group_id
N_count
S_count
V_count
F_count
total_stage2
F_percentage_within_group
percentage_of_all_F
cumulative_F_percentage
```

Calcular:

```text
total_F
number_of_groups_with_F
number_of_groups_without_F
top1_F_concentration
top2_F_concentration
top3_F_concentration
Herfindahl-like concentration diagnostic
median_F_per_F_group
mean_F_per_F_group
std_F_per_F_group
```

O índice de concentração adicional é diagnóstico interno e não deve ser apresentado como métrica clínica.

Ordenar grupos por `F_count`.

## Auditorias obrigatórias

Comparar:

1. annotation symbol original;
2. AAMI mapping;
3. target final do Stage 2.

Gerar uma tabela de rastreabilidade:

```text
original_annotation
mapped_aami_class
stage1_target
stage2_target
count
```

Verificar especialmente o mapeamento do símbolo de fusion beat para F.

## Testes obrigatórios

Criar testes que garantam:

* nenhum grupo possui ID nulo;
* a soma por grupo corresponde à soma global;
* a soma de S/V/F corresponde à cardinalidade Stage 2;
* cada batimento possui exatamente uma classe Stage 2;
* não existem labels Stage 2 desconhecidos;
* registros 208 e 213, caso presentes no dataset local, tenham suas contagens reportadas explicitamente.

## Checagem pós-operatória E01

Emitir:

```text
TOP F GROUPS
TOP-1 CONCENTRATION
TOP-2 CONCENTRATION
GROUPS WITH F
GROUPS WITHOUT F
```

Comparar o resultado local com a expectativa documental.

Se o resultado local divergir materialmente, NÃO corrigir números manualmente.

Investigar:

* exclusão dos primeiros cinco minutos;
* filtro de beats;
* janelas descartadas;
* preprocessing;
* mapping AAMI;
* exclusão por sinal inválido;
* diferença entre record completo e test period;
* filtro Stage 1;
* perda de janelas nas bordas.

## Critério de aprovação

`PASS` se a concentração for reproduzida e explicada.

`BLOCKED` se a origem dos labels F não puder ser rastreada até a anotação original.

---

# E02 — MANIFESTO IMUTÁVEL DO DATASET E DAS FEATURES

## Objetivo

Impedir que experimentos distintos usem datasets ou esquemas de features silenciosamente diferentes.

## Implementação

Criar:

```text
dataset_manifest_v2.4.json
feature_manifest_v2.4.json
```

O dataset manifest deve registrar:

```text
dataset_name
dataset_version
source
record_ids
excluded_record_ids
annotation_source
annotation_mapping_version
sampling_rate
channel_policy
beat_window_policy
edge_window_policy
preprocessing_version
total_examples
class_counts
group_counts
groups_with_F
source_file_hashes
```

O feature manifest deve registrar, para cada feature:

```text
feature_name
position
dtype
unit_if_known
definition
source
requires_future_context
requires_previous_context
normalization_policy
```

Gerar:

```text
feature_schema_hash
```

a partir de uma serialização canônica dos nomes, ordem e definições.

## Regra crítica

Modelos, scalers e resultados devem carregar:

```text
dataset_manifest_hash
feature_schema_hash
```

A inferência deve rejeitar scaler/model incompatível quando o hash do esquema não corresponder.

Não apenas emitir warning.

Falhar explicitamente.

## Checagem pós-operatória E02

Testar:

* troca proposital da ordem de duas features;
* remoção proposital de uma feature;
* adição de feature desconhecida;
* scaler com schema antigo.

Todos esses cenários devem ser detectados.

## Critério de aprovação

`PASS` somente quando incompatibilidade de schema causar falha explícita.

---

# E03 — AUDITORIA E REDESENHO DO PROTOCOLO DE SPLIT

## Objetivo

Garantir validação inter-paciente real e impedir seleção pelo outer test.

## Primeira tarefa

Inspecionar o código atual de split.

Responder por evidência de código:

```text
Qual é o group_id?
Record ID ou patient ID?

Os registros 201 e 202 são tratados corretamente em relação ao indivíduo correspondente?

Existe alguma forma de o mesmo indivíduo entrar em train e test?

O scaler é fitted onde?

O SMOTE é aplicado antes ou depois do split?

Feature selection usa quais dados?

Threshold selection usa quais dados?

O melhor fold é escolhido usando qual métrica e qual partição?
```

Não assumir respostas.

Documentar paths, funções e linhas relevantes.

## Implementação

Criar abstração de split versionada.

Preferência inicial:

```text
outer CV:
StratifiedGroupKFold
n_splits=5
groups=patient_or_verified_group_id

inner CV:
StratifiedGroupKFold
groups=outer_train_groups
```

Entretanto, antes de usar `StratifiedGroupKFold`, comparar formalmente:

```text
GroupKFold
vs
StratifiedGroupKFold
```

Sem treinar modelo.

Para cada splitter gerar:

```text
fold
train_groups
test_groups
train_S
train_V
train_F
test_S
test_V
test_F
F_test_percentage
contains_208
contains_213
```

## Métrica de qualidade do split

Criar um `split_diagnostic_score` somente para diagnóstico, considerando:

* ausência de overlap de grupo como requisito absoluto;
* presença das classes;
* dispersão das proporções por fold;
* concentração F.

Não permitir que o score relaxe a restrição de grupos não sobrepostos.

## Regra obrigatória de nested CV

Outer test:

```text
somente avaliação final do fold
```

Inner CV/OOF:

```text
arquitetura
loss
sampling
hyperparameters
early-stopping policy
calibration
decision rule
threshold
```

Nenhuma informação do outer test pode voltar para seleção.

## Artefatos

Criar:

```text
split_manifest_outer_v2.4.json
split_diagnostics_v2.4.csv
split_diagnostics_v2.4.md
```

Cada fold deve ter hashes dos arrays de índices.

## Testes obrigatórios

Para cada outer fold:

```python
assert train_groups.isdisjoint(test_groups)
```

Para cada inner fold:

```python
assert inner_train_groups.isdisjoint(inner_val_groups)
assert inner_train_groups.issubset(outer_train_groups)
assert inner_val_groups.issubset(outer_train_groups)
assert inner_val_groups.isdisjoint(outer_test_groups)
```

O scaler deve ser fitted exclusivamente no train correspondente.

O resampler deve operar exclusivamente no train correspondente.

## Checagem pós-operatória E03

Gerar relatório comparando GroupKFold e StratifiedGroupKFold.

Não selecionar automaticamente StratifiedGroupKFold apenas porque seu nome contém “Stratified”.

Selecionar o protocolo com:

1. zero leakage;
2. cobertura de classes tecnicamente possível;
3. melhor estabilidade distributiva;
4. justificativa documentada.

Quando a concentração estrutural de F tornar estratificação perfeita impossível, registrar essa impossibilidade.

## Critério de aprovação

`PASS` se o split for patient-wise/group-wise, imutável e auditado.

`BLOCKED` se o verdadeiro identificador de paciente não puder ser determinado.

---

# E04 — REDESENHO DOS QUALITY GATES QG5

## Objetivo

Separar teste diagnóstico balanceado de gate real de publicação.

## Implementação

Substituir conceitualmente o QG5 único por:

```text
QG5_SMOKE_BALANCED
QG5_PATIENTWISE
QG5_STABILITY
QG5_CALIBRATION
QG5_REPRODUCIBILITY
```

## QG5_SMOKE_BALANCED

Objetivo:

* detectar modelo quebrado;
* incompatibilidade de scaler;
* classe perdida;
* mapping incorreto;
* regressão grosseira.

Pode continuar utilizando subset controlado/balanceado.

Deve ser marcado:

```text
diagnostic_only = true
publication_metric = false
```

Nunca usar sua F1(F) como prova de generalização inter-paciente.

## QG5_PATIENTWISE

Usar somente outer OOF/test predictions do protocolo E03.

Registrar:

```text
F1_macro_mean
F1_macro_std

F1_S_mean
F1_V_mean
F1_F_mean

F1_F_std
F1_F_min
F1_F_max

precision_F
recall_F
AP_F

per_fold_confusion_matrix
aggregated_confusion_matrix
```

Não criar um target novo arbitrário apenas para fazer o teste passar.

Manter `mean_F1_F >= 0.50` como target de pesquisa/publicação se este é requisito formal do projeto.

Entretanto, o gate deve FALHAR honestamente enquanto a métrica real estiver abaixo dele.

## QG5_STABILITY

Inicialmente registrar, sem bloquear publicação até o baseline ser medido:

```text
F1_F_std
prediction_disagreement_rate
mean_abs_probability_delta
worst_fold_F1_F
worst_group_F1_F
```

Após E10, definir limites de estabilidade com base na distribuição experimental real.

## QG5_CALIBRATION

Registrar:

```text
log_loss
multiclass_brier_score
per_class_reliability_data
ECE_if_implemented_with_explicit_definition
```

Não tratar Brier isoladamente como medida pura de calibração.

Gerar reliability data/curve.

## QG5_REPRODUCIBILITY

Validar:

```text
dataset manifest
split manifest
feature schema
seed
environment
model configuration
```

## Alterar seleção de publicação

Remover qualquer lógica equivalente a:

```python
publish = best_fold.f1_f >= 0.50
```

Implementar conceito equivalente a:

```python
publish = (
    patientwise_gate_passed
    and stability_gate_passed
    and calibration_gate_passed
    and reproducibility_gate_passed
)
```

Durante a research branch, permitir:

```text
RESEARCH_CANDIDATE
```

para modelos ainda abaixo do target de publicação.

## Checagem pós-operatória E04

Rodar o modelo v14 atual nos novos gates.

Resultado esperado:

* smoke pode passar;
* patientwise provavelmente deve falhar no requisito F>=0.50.

Se o novo QG5 declarar v14 como publication PASS apesar de F1(F) inter-paciente próximo de 0.21, considerar a implementação incorreta.

## Critério de aprovação

`PASS` quando QG5 distinguir claramente diagnóstico balanceado e generalização real.

---

# E05 — AUDITORIA DE SEPARABILIDADE DAS 16 FEATURES ATUAIS

## Objetivo

Determinar se o espaço de 16 features contém informação generalizável suficiente para distinguir F.

Nenhuma nova loss nesta etapa.

Nenhuma nova arquitetura nesta etapa.

## Implementação

Criar:

```text
scripts/audit_stage2_feature_separability.py
```

Executar análises:

```text
F vs S
F vs V
F vs non-F
S vs V
```

Em quatro regimes:

```text
global
record/group 208
record/group 213
all groups excluding 208 and 213
```

Quando 208/213 não existirem sob esse ID exato no group schema, usar os registros correspondentes e documentar.

## Análises obrigatórias

Para cada feature:

* count;
* missing;
* mean;
* std;
* median;
* IQR;
* quantis;
* min/max;
* outlier rate segundo regra explicitada;
* mutual information estimada;
* efeito univariado apropriado;
* distribuição por classe;
* distribuição por grupo.

Executar:

```text
permutation importance
leave-one-feature-out
```

em baseline simples dentro do protocolo correto.

Pode usar PCA e UMAP exclusivamente como diagnóstico exploratório.

Marcar explicitamente:

```text
NOT PERFORMANCE EVIDENCE
```

Não selecionar features usando o dataset inteiro e depois reportar outer CV como se a seleção fosse independente.

Qualquer seleção baseada em dados deverá ser refeita dentro do inner loop.

## Experimentos leave-group-out críticos

Executar:

```text
train without 208 -> evaluate 208
train without 213 -> evaluate 213
train without 208 and 213 -> evaluate remaining F groups
```

Quando tecnicamente apropriado, realizar também o inverso diagnóstico:

```text
train dominated by 208/213
evaluate other F groups
```

Não usar o resultado inverso como estimativa final de generalização.

## Relatório obrigatório

Responder:

```text
1. Quais features realmente carregam sinal para F?
2. O sinal permanece fora de 208/213?
3. Quais features apresentam patient signature?
4. F se sobrepõe principalmente com S ou V?
5. A fronteira muda entre grupos?
6. Existe evidência de feature leakage?
7. Existe evidência de feature saturation?
```

## Critério de decisão

Classificar:

```text
REPRESENTATION_SUFFICIENT
REPRESENTATION_WEAK
REPRESENTATION_PATIENT_DEPENDENT
INCONCLUSIVE
```

## Checagem pós-operatória E05

Garantir que:

* nenhuma feature nova foi adicionada;
* nenhum hyperparameter Stage 2 foi tunado;
* outer test não participou de feature selection;
* relatórios são reproduzíveis.

## Critério de aprovação

`PASS` mesmo quando a hipótese “16 features são insuficientes” for rejeitada.

A finalidade é obter diagnóstico confiável.

---

# E06 — CONTEXT FEATURES v2.4 + BASELINE CROSS-ENTROPY NATURAL

Executar somente depois de E05.

## Objetivo

Aumentar informação temporal/morfológica antes de voltar a modificar loss.

## Criar novo schema

Não alterar silenciosamente as 16 features v2.3.

Criar schema v2.4.

Avaliar a viabilidade das seguintes features contextuais:

```text
RR_prev_1
RR_prev_2
RR_prev_3
RR_prev_4
RR_next_1

RR_local_mean
RR_local_std
RR_local_cv

RR_current_over_local_mean
RR_prev_ratio
RR_next_ratio

delta_RR_prev
delta_RR_next

tachycardia_flag

crest_factor_180
crest_factor_400
```

Avaliar também, quando deriváveis corretamente do pipeline:

```text
QRS_width
QRS_area
R_amplitude
S_amplitude
R_S_ratio
beat_energy
beat_skewness
beat_kurtosis
```

Features de correlação com templates devem ser implementadas somente como grupo experimental separado:

```text
correlation_normal_template
correlation_ventricular_template
```

## Regra contra leakage temporal

Para cada feature, documentar:

```text
requires_previous_context
requires_future_context
```

`RR_next_1` não pode ser incorporado silenciosamente em cenários de inferência online/causal.

Criar dois schemas se necessário:

```text
context_offline
context_causal
```

Não misturar a alegação de performance offline com capacidade causal em produção.

## Template correlation

Se implementada:

* templates devem ser construídos somente a partir do train do fold;
* nunca do dataset inteiro;
* outer test não pode contribuir para templates;
* normal template e ventricular template devem ser versionados por fold;
* provar por teste que os IDs do outer test não contribuíram.

## Baseline E06

Treinar:

```text
representation = context v2.4
sampling = natural distribution
loss = standard multiclass cross entropy
architecture = minimal MLP baseline
decision = argmax raw softmax
```

Evitar arquitetura excessivamente maior.

Começar com arquitetura simples e documentada.

Não usar:

* focal;
* SMOTE;
* class weight;
* Logit Adjustment;
* Balanced Softmax;
* LDAM;
* calibrated threshold.

Objetivo:

```text
medir exclusivamente o ganho da representação
```

## Checagem pós-operatória E06

Comparar E06 com v14/v15:

```text
delta_mean_F1_F
delta_std_F1_F
delta_min_fold_F1_F
delta_macro_F1
delta_F1_S
delta_F1_V
```

Também comparar:

```text
groups improved
groups regressed
208 performance
213 performance
remaining F groups performance
```

## Critério científico

Se:

```text
F1_F improves materially outside 208/213
```

classificar:

```text
REPRESENTATION_HYPOTHESIS_SUPPORTED
```

Se o ganho existir somente em 208/213:

```text
PATIENT_SPECIFIC_GAIN_WARNING
```

Se não houver ganho:

```text
CONTEXT_REPRESENTATION_NOT_SUFFICIENT
```

Não declarar sucesso somente por aumento inferior à variabilidade multi-seed ainda desconhecida.

Marcar ganho pequeno como:

```text
PROVISIONAL
```

até E10.

---

# E07 — AUDITORIA DE SAMPLING

## Objetivo

Comparar estratégias de amostragem sem alterar representação, arquitetura, optimizer ou loss.

Fixar o melhor baseline honesto de E06.

Executar:

```text
E07A natural distribution
E07B random oversampling
E07C SMOTE
E07D patient-aware sampling
```

## Patient-aware sampling

Implementar um sampler cuja lógica conceitual seja:

```text
selecionar classe
selecionar grupo/paciente elegível dentro da classe
selecionar exemplo do grupo
```

Evitar que um grupo domine automaticamente os gradientes apenas pelo número bruto de batimentos F.

Documentar a probabilidade efetiva de sampling por:

```text
class
group
example
```

## Regra SMOTE

Aplicar somente no train do fold interno/correspondente.

Nunca antes do split.

Registrar:

```text
real_examples_per_class
synthetic_examples_per_class
SMOTE_neighbors
sampling_strategy
```

Criar campo:

```text
is_synthetic
```

nos dados experimentais rastreáveis quando tecnicamente viável.

## Auditoria de geometria SMOTE

Para exemplos sintéticos F, medir distância aos:

* F reais;
* S reais;
* V reais.

Investigar exemplos sintéticos próximos da fronteira de outra classe.

Não concluir causalidade somente pela distância.

Gerar diagnóstico.

## Checagem pós-operatória E07

Uma tabela obrigatória:

```text
experiment
sampling
mean_F1_F
std_F1_F
min_F1_F
macro_F1
precision_F
recall_F
remaining_F_groups_F1
```

## Critério

Não selecionar automaticamente o maior `mean_F1_F`.

Usar inner selection policy congelada.

Registrar se a estratégia aumenta recall destruindo precision ou vice-versa.

Classificar cada estratégia:

```text
BENEFICIAL
NEUTRAL
UNSTABLE
HARMFUL
```

---

# E08 — LONG-TAIL LOSS / CLASSIFIER ABLATION

Executar somente após fixar representação e estratégia de sampling.

## Objetivo

Comparar métodos conceitualmente distintos de long-tail learning.

Uma estratégia por experimento.

Executar:

```text
E08A cross entropy natural baseline
E08B Logit Adjustment
E08C Balanced Softmax
E08D LDAM + Deferred Re-Weighting
E08E decoupled representation + classifier retraining
E08F focal legacy
```

SMOTE não deve ser combinado inicialmente com E08B–E08E.

## E08B — Logit Adjustment

Implementar frequências de classe derivadas exclusivamente do train correspondente.

Registrar:

```text
class_counts
class_priors
tau_or_equivalent_parameter
adjustment_vector
```

Não calcular prior no dataset completo.

## E08C — Balanced Softmax

Criar implementação isolada e testes numéricos.

Testar:

* logits shape;
* class counts;
* classes ausentes;
* estabilidade numérica;
* loss finita;
* gradientes finitos.

## E08D — LDAM + DRW

Separar explicitamente:

```text
margin schedule
reweighting start epoch
class counts
```

DRW deve possuir mudança de regime registrada no training log.

Criar teste comprovando que reweighting não está ativo antes da época configurada.

## E08E — Decoupled classifier

Fluxo:

```text
1. treinar representação sob distribuição selecionada;
2. salvar encoder/feature extractor;
3. congelar representação;
4. reinicializar ou substituir classifier head segundo protocolo;
5. retreinar somente classifier;
6. registrar trainable variables antes e depois.
```

Criar teste que falha se pesos da representação mudarem durante classifier retraining.

## E08F — Focal legacy

Reproduzir melhor política focal anterior como controle.

Não retunar alpha/gamma nesta etapa.

## Checagem pós-operatória E08

Para cada método:

```text
same dataset manifest?
same feature manifest?
same split manifest?
same architecture?
same optimizer policy?
same epoch policy?
same seed set?
only intended conceptual variable changed?
```

Gerar `ablation_integrity_report.json`.

Se mais de uma variável não autorizada mudar:

`EXPERIMENT_INVALID`

Não comparar sua métrica na tabela principal.

## Critério de decisão

Priorizar ganho consistente entre folds.

Relatar:

```text
fold wins
fold losses
mean delta
median delta
worst-fold delta
F groups improved
F groups regressed
```

Não eleger método apenas pelo melhor fold.

---

# E09 — AUDITORIA DA REGRA DE DECISÃO E CALIBRAÇÃO

## Objetivo

Determinar se thresholds independentes S=0.5, V=0.5 e F=0.8 estão introduzindo fragilidade.

## Primeira etapa: inspecionar inferência real

Documentar exatamente o comportamento para:

```text
P(S)=0.60 P(V)=0.55 P(F)=0.81
P(S)=0.49 P(V)=0.49 P(F)=0.79
P(S)=0.51 P(V)=0.10 P(F)=0.82
P(S)=0.34 P(V)=0.33 P(F)=0.33
```

Responder:

```text
Como múltiplos thresholds são resolvidos?
Qual é o tie-breaker?
O threshold é aplicado a softmax?
A probabilidades independentes?
A logits?
Qual é o fallback?
```

Criar testes unitários para essas decisões.

## Comparar quatro políticas

```text
E09A raw softmax + argmax
E09B temperature scaling + argmax
E09C calibrated probabilities + frozen cost-sensitive decision policy
E09D legacy independent thresholds
```

## Regra de calibração

Calibration fit somente com:

```text
inner OOF predictions
```

ou conjunto de calibração explicitamente separado do treino do estimador.

Nunca usar:

```text
training predictions do próprio modelo
```

como base principal de calibração.

Nunca usar outer test.

## Temperature scaling

Fit `T` usando inner OOF/calibration data.

Registrar por outer fold:

```text
temperature
inner_log_loss_before
inner_log_loss_after
```

Congelar `T`.

Aplicar ao outer test somente depois.

## Threshold/cost decision

Qualquer parâmetro decisório deve ser escolhido no inner loop.

Registrar:

```text
selection_metric
candidate_policy
selected_parameters
inner_score
```

Após congelamento, avaliar uma única vez no outer test.

## Checagem pós-operatória E09

Comparar:

```text
F1_F
precision_F
recall_F
macro_F1
log_loss
Brier
reliability
decision_disagreement_rate
```

entre E09A–D.

Verificar se calibration muda probabilidades de forma útil.

Não alegar melhor calibração apenas porque F1 aumentou.

Não alegar pior calibração apenas porque accuracy/F1 ficou constante.

## Critério

A política selecionada deve possuir:

* seleção inner-only;
* regra determinística;
* sem regiões de decisão não documentadas;
* métricas probabilísticas reportadas;
* artefato serializável.

---

# E10 — ESTABILIDADE MULTI-SEED E INSTABILIDADE DE PREDIÇÃO

## Objetivo

Determinar se o ganho observado é maior que a variabilidade do treinamento.

## Selecionar finalistas

Escolher no máximo:

```text
baseline
best representation candidate
best long-tail candidate
```

segundo resultados inner/outer corretamente estruturados.

## Execução

Triagem:

```text
5 outer folds × 3 seeds
```

Finalistas:

```text
5 outer folds × 5 seeds
```

Seeds devem ser explícitas e versionadas.

Exemplo:

```text
[17, 29, 43, 71, 101]
```

Não escolher seeds depois de observar resultados.

## Métricas

Registrar:

```text
F1_F_mean
F1_F_std
F1_F_min
F1_F_max

F1_macro_mean
F1_macro_std

precision_F_mean
recall_F_mean
AP_F_mean

worst_fold_F1_F
worst_group_F1_F
```

## Prediction Instability Index

Implementar como métrica experimental interna, claramente identificada como métrica do projeto.

Para cada exemplo presente nas avaliações comparáveis:

```text
PII_F_example =
max(P_F across seeds)
-
min(P_F across seeds)
```

Agregar:

```text
PII_F_mean
PII_F_median
PII_F_p90
PII_F_p95
PII_F_max
```

Também calcular:

```text
class_prediction_disagreement_rate
```

por exemplo entre seeds.

Gerar análise por grupo.

## Critério

Um ganho deve ser classificado:

```text
ROBUST_GAIN
```

somente quando for consistente frente à variabilidade observada.

Se a melhoria média for menor ou semelhante à oscilação multi-seed:

```text
GAIN_WITHIN_TRAINING_VARIANCE
```

Não promover o modelo por esse ganho.

## Checagem pós-operatória E10

Confirmar:

* mesmas seeds entre finalistas;
* mesmos folds;
* mesmo manifest;
* mesmas políticas de avaliação;
* ausência de runs faltantes;
* nenhum run descartado por métrica ruim.

Runs podem ser descartados somente por falha técnica documentada.

Registrar o erro e repetir com a MESMA seed.

---

# E11 — AUDITORIA DETERMINÍSTICA CPU/GPU

## Objetivo

Separar compatibilidade de hardware de estabilidade numérica/reprodutibilidade.

## Criar dois modos

```text
TRAIN_MODE=performance
AUDIT_MODE=deterministic
```

## AUDIT_MODE

Configurar, de forma compatível com a versão instalada do TensorFlow:

```python
tf.keras.utils.set_random_seed(seed)
tf.config.experimental.enable_op_determinism()
```

Auditar também:

* Python random;
* NumPy;
* uso de `numpy.random.default_rng`;
* `tf.data`;
* map paralelo;
* prefetch;
* generators;
* multiprocessing;
* workers;
* augmentations aleatórias;
* ordem do dataset.

Quando `default_rng` for usado, fornecer seed explicitamente.

## Experimento

Com:

```text
same dataset manifest
same split manifest
same feature manifest
same seed
same hyperparameters
same software environment
```

Executar:

```text
CPU A
CPU B

GPU A
GPU B
```

Quando GPU estiver disponível.

## Comparação intra-hardware

Comparar:

```text
CPU A vs CPU B
GPU A vs GPU B
```

Registrar:

```text
model_weight_hash
prediction_hash
max_abs_logit_delta
mean_abs_probability_delta
class_disagreement_rate
```

No mesmo hardware e ambiente, AUDIT_MODE deve buscar reprodução exata.

Se não obtida, localizar a primeira etapa não determinística.

## Comparação CPU vs GPU

Não exigir automaticamente igualdade bitwise.

Medir:

```text
max_abs_logit_delta
mean_abs_logit_delta
max_abs_probability_delta
mean_abs_probability_delta
class_disagreement_rate
metric_delta
```

Criar tolerâncias somente depois de observar e justificar o comportamento.

Não definir tolerâncias frouxas apenas para o teste passar.

## Checagem pós-operatória E11

Confirmar:

```text
execution_compatibility
intra_cpu_reproducibility
intra_gpu_reproducibility
cross_device_numerical_delta
cross_device_decision_delta
```

Classificar separadamente.

---

# E12 — ARQUITETURA HÍBRIDA Conv1D + CONTEXT FEATURES

NÃO executar automaticamente.

## Condição de entrada

Executar somente se, após E06–E10:

```text
mean inter-patient F1_F remains < 0.40
```

e a auditoria E05/E06 indicar gargalo morfológico ou representação insuficiente.

## Objetivo

Permitir ao modelo observar diretamente a morfologia bruta do ECG sem abandonar contexto tabular.

## Arquitetura baseline proposta

Branch tabular:

```text
context features
-> Dense 64
-> normalization if justified
-> GELU
```

Branch ECG:

```text
beat window
-> Conv1D 32
-> Conv1D 64
-> Conv1D 128
-> GlobalAveragePooling1D
```

Fusion:

```text
concatenate
-> Dense 128
-> GELU
-> Dropout 0.25
-> 3 logits
```

A arquitetura exata deve ser implementada como baseline minimalista.

Não adicionar simultaneamente:

* attention;
* Transformer;
* bidirectional LSTM;
* GRU;
* residual tower complexa;
* focal nova;
* SMOTE novo;
* novo calibrador.

## Primeiro experimento híbrido

```text
loss = standard CE
sampling = melhor política já validada ou natural baseline claramente identificada
decision = política validada em E09
```

Idealmente executar primeiro com natural sampling para medir o ganho morfológico isoladamente.

## Integridade da janela

Validar:

```text
sample rate
window left
window right
R-peak alignment
padding
edge beats
normalization
channel
```

Gerar visualização de amostras aleatórias S/V/F de train e test para auditoria humana.

Não usar visualização para escolher exemplos favoráveis.

## Checagem pós-operatória E12

Comparar com melhor tabular:

```text
mean_F1_F
std_F1_F
worst_fold
remaining_F_groups
PII_F
parameter_count
training_time
inference_time
```

Se a arquitetura híbrida melhorar somente 208/213:

```text
MORPHOLOGY_PATIENT_MEMORIZATION_WARNING
```

Se melhorar grupos F restantes consistentemente:

```text
RAW_MORPHOLOGY_HYPOTHESIS_SUPPORTED
```

---

# E13 — SELEÇÃO FINAL E PUBLICAÇÃO CANDIDATA v2.4

## Objetivo

Selecionar o modelo usando política previamente congelada.

## Proibição

Não publicar:

```text
o melhor fold
```

como modelo final simplesmente porque teve maior F1(F).

## Seleção

Usar resultados do protocolo completo.

O relatório final deve apresentar:

```text
candidate_id
representation
architecture
sampling
loss
calibration
decision_policy

mean_F1_F
std_F1_F
min_F1_F
mean_macro_F1

precision_F
recall_F
AP_F

worst_group_F1_F
PII_F_mean
prediction_disagreement_rate

QG5_SMOKE_BALANCED
QG5_PATIENTWISE
QG5_STABILITY
QG5_CALIBRATION
QG5_REPRODUCIBILITY
```

## Política de publicação

Somente publicar como modelo aprovado quando os gates formais passarem.

Se o target:

```text
mean_F1_F >= 0.50
```

não for alcançado, NÃO reduzir silenciosamente o target.

Classificar o melhor modelo como:

```text
RESEARCH_CANDIDATE_NOT_PUBLICATION_READY
```

Pode publicar artefatos na pasta experimental, mas não substituir os artefatos oficiais de produção.

## Treino do artefato final

Após a seleção da estratégia, definir protocolo explícito de refit.

O refit final deve usar:

```text
selected architecture
selected feature schema
selected sampling
selected loss
selected calibration protocol
selected decision policy
```

Não reutilizar diretamente “pesos do melhor outer fold” como modelo final por conveniência.

Treinar segundo protocolo de refit previamente documentado.

Registrar:

```text
final_training_manifest.json
```

## Artefatos v2.4

Somente após aprovação:

```text
models/stage2_float32_v2.4.keras
models/input_scaler_stage2_v2.4.pkl
models/stage2_calibration_v2.4.*
models/stage2_decision_policy_v2.4.json

models/dataset_manifest_v2.4.json
models/feature_manifest_v2.4.json
models/training_manifest_v2.4.json
models/evaluation_manifest_v2.4.json
```

Não sobrescrever v2.3.

## Validação pós-publicação

Carregar os artefatos a partir do disco em processo limpo.

Não avaliar usando objetos ainda residentes da sessão de treinamento.

Executar:

```text
load model
load scaler
load feature manifest
load calibration
load decision policy
validate hashes
run patientwise evaluation
run smoke test
run inference contract tests
```

Confirmar que métricas do modelo recarregado correspondem às métricas pré-serialização dentro da tolerância definida.

---

# TESTES TRANSVERSAIS OBRIGATÓRIOS

Criar ou reforçar testes para:

## DATA LEAKAGE

```text
test_no_outer_group_in_train
test_no_outer_group_in_inner_cv
test_scaler_fit_train_only
test_resampler_train_only
test_template_train_only
test_calibrator_inner_oof_only
test_threshold_selection_no_outer_test
```

## DATASET

```text
test_dataset_manifest_matches_data
test_class_count_consistency
test_group_count_consistency
test_aami_mapping_traceability
test_f_distribution_report_consistency
```

## FEATURES

```text
test_feature_order_contract
test_feature_schema_hash
test_model_rejects_wrong_feature_schema
test_scaler_rejects_wrong_feature_schema
test_context_feature_nan_inf
test_context_feature_shapes
```

## MODEL

```text
test_stage2_logits_shape
test_stage2_probability_sum
test_stage2_no_nan_logits
test_stage2_no_nan_probabilities
```

## DECISION POLICY

```text
test_multiclass_decision_overlap
test_multiclass_decision_no_threshold_hit
test_multiclass_tie_break
test_decision_policy_serialization
```

## REPRODUCIBILITY

```text
test_seed_manifest
test_split_manifest_hash
test_same_seed_deterministic_mode
```

## PUBLICATION

```text
test_balanced_smoke_cannot_authorize_publication_alone
test_best_fold_cannot_authorize_publication
test_patientwise_gate_required
test_stability_gate_required
test_calibration_gate_required
test_reproducibility_gate_required
```

---

# FORMATO OBRIGATÓRIO DO RELATÓRIO APÓS CADA ETAPA

Após cada experimento, responda com:

## 1. ETAPA EXECUTADA

```text
E0X — nome
```

## 2. HIPÓTESE

Uma frase falsificável.

## 3. ALTERAÇÕES IMPLEMENTADAS

Listar arquivos modificados e propósito.

## 4. VARIÁVEL CONCEITUAL ALTERADA

Deve existir preferencialmente apenas uma.

Exemplo:

```text
representation
```

ou:

```text
sampling policy
```

ou:

```text
loss
```

## 5. CHECAGEM PÓS-OPERATÓRIA

Usar:

```text
[PASS] testes
[PASS] flake8
[PASS] mypy
[PASS] shapes
[PASS] NaN/Inf
[PASS] dataset manifest
[PASS] feature schema
[PASS] split integrity
[PASS] no group overlap
[PASS] artifacts isolated
```

Não marcar PASS sem executar.

Usar:

```text
[NOT RUN]
```

quando não executado.

## 6. RESULTADOS

Apresentar baseline e candidato lado a lado.

Nunca apresentar apenas o melhor número novo.

## 7. ANÁLISE POR FOLD

Mostrar todos os folds.

## 8. ANÁLISE DA CLASSE F

Obrigatoriamente separar:

```text
208
213
remaining F groups
```

quando aplicável.

## 9. REGRESSÕES

Listar toda regressão detectada.

## 10. HIPÓTESE

Classificar:

```text
SUPPORTED
REJECTED
INCONCLUSIVE
```

## 11. CHECKPOINT

Classificar:

```text
PASS
PASS_HYPOTHESIS_REJECTED
BLOCKED
ROLLBACK_REQUIRED
```

## 12. PRÓXIMA ETAPA AUTORIZADA

Indicar exatamente uma próxima etapa.

---

# REGRAS DE INTERPRETAÇÃO

Não escrever:

```text
“O modelo melhorou”
```

quando apenas um fold melhorou.

Escrever:

```text
“O F1(F) médio aumentou X; Y de 5 folds melhoraram; o pior fold mudou de A para B; a variabilidade mudou de C para D.”
```

Não escrever:

```text
“SMOTE resolveu o desbalanceamento”
```

sem demonstrar generalização fora dos grupos dominantes.

Não escrever:

```text
“calibração melhorou”
```

usando somente F1.

Não escrever:

```text
“GPU é instável”
```

por causa de logs informativos CUDA.

Não escrever:

```text
“16 features são insuficientes”
```

antes de E05.

Não escrever:

```text
“Focal não funciona”
```

apenas porque uma configuração de alpha/gamma falhou.

Não promover arquitetura mais complexa quando uma modificação simples e estável produzir resultado equivalente.

Sempre diferenciar:

```text
discrimination
calibration
decision policy
training stability
patient generalization
```

---

# REGRA FINAL DE ENGENHARIA

O objetivo desta research branch não é produzir rapidamente um número F1(F) maior.

O objetivo é determinar qual destas causas domina o Stage 2:

```text
A. concentração da classe F por paciente/registro;
B. representação insuficiente das 16 features;
C. viés do classificador long-tail;
D. sampling inadequado;
E. regra de decisão/threshold frágil;
F. má calibração;
G. instabilidade de otimização;
H. insuficiência de modelagem morfológica;
I. combinação comprovada de causas.
```

Cada experimento deve eliminar, apoiar ou refinar uma hipótese.

Comece agora por `E00`.

Após concluir `E00`, execute a checagem pós-operatória completa.

Só avance para `E01` se o checkpoint de E00 autorizar.

Continue sequencialmente, sem solicitar confirmação entre etapas, desde que o checkpoint autorize o avanço.

Se encontrar uma falha crítica de integridade experimental, não contorne o problema para continuar.

Corrija a causa raiz, valide a correção, repita a checagem da etapa e registre a ocorrência.

Ao final, gere:

```text
docs/stage2_v2.4_root_cause_report.md
docs/stage2_v2.4_experiment_matrix.md
docs/stage2_v2.4_stability_report.md
docs/stage2_v2.4_publication_readiness.md
```

O relatório final deve responder objetivamente:

```text
1. Por que v11–v16 ficaram no platô de F1(F)?
2. Quanto da dificuldade é explicada pela concentração 208/213?
3. As 16 features contêm fronteira generalizável para F?
4. Context features melhoram pacientes não dominantes?
5. Qual estratégia long-tail é realmente consistente?
6. SMOTE ajuda ou prejudica neste dataset?
7. Os thresholds legados criam decisões frágeis?
8. Qual é a variabilidade entre seeds?
9. CPU/GPU alteram decisões do modelo de forma material?
10. O target inter-paciente F1(F)>=0.50 foi realmente atingido?
11. O candidato v2.4 está pronto para publicação ou permanece research-only?
```

Não suavize resultados negativos.

Não faça o QG5 passar artificialmente.

Não altere target para acomodar o modelo.

Priorize rastreabilidade, isolamento experimental, reprodutibilidade e generalização inter-paciente.
