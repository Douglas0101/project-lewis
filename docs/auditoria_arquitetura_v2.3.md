# Auditoria de Arquitetura — Project-Lewis v2.3 (Stage 2 MLP)

## Escopo

Esta auditoria examina os algoritmos de treinamento e inferência do pipeline
MLP two-stage v2.3 sob a ótica da estabilidade matemática da matriz de
confusão. O foco é o Estágio 2 (S vs V vs F), onde o desbalanceamento
extremo da classe F (Fusion) e a sobreposição morfológica com S criam uma
fronteira de decisão instável.

## 1. Distribuição de classes e natureza do problema

Distribuição observada no dataset `data/features/stage2_multiclass_features.npz`:

| Classe | AAMI | Amostras | Proporção |
|--------|------|----------|-----------|
| 0      | S    | 16.934   | 30,7 %    |
| 1      | V    | 37.183   | 67,4 %    |
| 2      | F    | 1.044    | 1,9 %     |

A classe F representa menos de 2 % dos batimentos, enquanto V domina.
Algebricamente, um classificador que simplesmente minimiza a entropia cruzada
sobre essa distribuição tenderá a:

1. Posicionar a fronteira de decisão de forma a proteger a classe
   majoritária V, porque o gradiente esperado de V pesa ~67 % do batch.
2. Colapsar a região de F no centro de massa de S/V, já que os poucos
   exemplos de F têm contribuição pequena no gradiente total.

Portanto, a otimização exige uma perda que repondera dinamicamente as classes
**sem** distorcer a geometria da fronteira (contrário a `class_weight` fixo
massivo) e uma estratégia de amostragem que aumente a densidade de F no
espaço de treino sem criar exemplos fora da variedade fisiológica.

## 2. Análise das técnicas aplicadas

### 2.1 Focal Loss (`tf.keras.losses.CategoricalFocalCrossentropy`)

A função implementada é

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

onde `p_t` é a probabilidade atribuída à classe verdadeira.

Escolha de hiperparâmetros:

- `alpha = [0.60, 0.40, 1.00]` (S, V, F):
  - `alpha` atua como multiplicador escalar da perda quando a amostra
    pertence à classe `t`. Não altera a direção do gradiente, apenas seu
    módulo.
  - `alpha_V = 0.40` reduz o peso da classe majoritária, evitando que o
    gradiente de V domine a atualização e empurre os centros de S/F.
  - `alpha_S = 0.60` e `alpha_F = 1.00` dão ênfase crescente às
    minoritárias, especialmente F, sem a distorção abrupta causada por
    pesos estáticos do tipo `class_weight`.

- `gamma = 2.0`:
  - Para um exemplo bem classificado (`p_t = 0.9`), o fator de modulação
    é `(0.1)^2 = 0.01`, reduzindo drasticamente sua contribuição.
  - Para um exemplo de fronteira (`p_t = 0.5`), o fator é `0.25`,
    mantendo a perda próxima da entropia cruzada padrão.
  - Isso concentra o gradiente nos exemplos difíceis (fronteira S/F/V),
    onde a matriz de confusão é mais instável.

Fragilidade: a escolha de `alpha` e `gamma` é feita manualmente. O espaço
não é explorado automaticamente; valores muito altos de `alpha_F` podem
inverter o domínio do gradiente e criar um efeito gangorra oposto
(melhora F, prejudica S).

### 2.2 SMOTE tabular

`imblearn.over_sampling.SMOTE` é aplicado **apenas no fold de treino**, antes
do `StandardScaler`. Isso respeita a integridade do conjunto de validação.
A estratégia `sampling_strategy` eleva as classes selecionadas até uma fração
da classe majoritária (default `target_ratio=0.5`), evitando igualdade total
que tende a overfitar na fronteira S/F.

Fragilidades:

1. SMOTE linear interpola features tabulares sem respeitar restrições
   fisiológicas. Por exemplo, pode gerar combinações inconsistentes entre
   `rr_prev`, `rr_next` e `rr_ratio`, ou entre `qrs_width_ms` e `qrs_area`.
2. A classe F tem apenas 1.044 amostras reais. Mesmo com `target_ratio=1.0`,
   a maioria dos exemplos de F vistos pelo modelo é sintética, o que pode
   amplificar ruído se a variedade real de F for pequena.
3. O parâmetro `k_neighbors=5` pode incluir vizinhos de S ou V quando F é
   muito escassa, borrando a fronteira real.

### 2.3 CosineDecayRestarts

O learning rate segue uma curva cosseno com reinícios:

```
lr(t) = alpha*lr_0 + 0.5*lr_0*(1-alpha)*(cos(pi*t_mod/T)+1)*m_mul^i
```

Configuração padrão:

- `initial_learning_rate = 1e-3`
- `first_decay_steps = 10` épocas
- `t_mul = 1.0`
- `m_mul = 0.9`
- `alpha = 0.1`

A ideia é permitir que o otimizador escape de mínimos locais na fronteira
S/F através dos reinícios, enquanto `m_mul=0.9` reduz a amplitude de
exploração a cada ciclo. Observações empíricas indicaram que, para este
conjunto, um LR fixo (`--no-lr-schedule`) convergiu ligeiramente melhor; a
flag é mantida para diagnóstico.

Fragilidade: `first_decay_steps=10` pode ser curto demais se a paisagem de
perda for ruidosa, causando oscilações. A flag `--no-lr-schedule` permite
isolar esse efeito.

### 2.4 Threshold tuning por Youden's J

Para cada classe `k`, define-se um problema binário one-vs-rest e busca-se o
threshold `t_k` que maximiza:

```
J_k(t) = TPR_k(t) + TNR_k(t) - 1
```

A estatística J é invariante sob prevalência e equaliza sensibilidade e
especificidade. Isso evita thresholds fixos (ex: 0.5) que favorecem a classe
majoritária V.

Na inferência, aplica-se a regra one-vs-rest com fallback para `argmax`
quando nenhuma classe supera seu limiar.

Fragilidades:

1. Os thresholds são otimizados **classe por classe**, não conjuntamente.
   A combinação pode não maximizar o F1-macro global.
2. O fallback para `argmax` pode introduzir viés para V quando todos os
   scores estiverem abaixo dos limiares.
3. Thresholds baixos para F (observados entre 0.11 e 0.43) aumentam o
   recall de F, mas também aumentam os falsos positivos, limitando o F1(F).

## 3. Lacunas encontradas no código

### 3.1 Inconsistência de versão e scaling no Estágio 1

Durante a auditoria detectou-se que o experimento
`experiments/stage1_mlp_features_v2.3/summary.json` foi gerado em
2026-07-08, enquanto o arquivo de features
`data/features/stage1_binary_features.npz` foi regenerado em 2026-07-09.
Consequentemente, os modelos Stage 1 publicados não correspondem ao NPZ de
features atual, o que explica AUC ~0,58 e recall ~0,10 no dataset completo.
O modelo foi retreinado em `experiments/stage1_mlp_features_v2.3_retrain`.

Adicionalmente, `scripts/prepare_stage1_features.py` salva features **já
escaladas** no NPZ:

```python
x_features_scaled = scale_features(x_features, scaler)
...
np.savez(output_path, X=x_features_scaled, ...)
```

Entretanto, `scripts/train_stage1_mlp.py` aplica `StandardScaler`
novamente sobre esses dados já escalados:

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
```

Isso causa uma **dupla normalização** no Estágio 1. O modelo treinado e o
scaler publicado refletem essa dupla escala, então o teste atual passa por
consistência interna, mas a convenção difere do Estágio 2 (onde o NPZ contém
features RAW e o scaler é fit no treino). Recomendação: alinhar
`prepare_stage1_features.py` para salvar RAW, como `prepare_stage2_features.py`.

### 3.2 Teste QG5' Stage 2 ignorava thresholds otimizados

O teste `tests/test_two_stage_mlp_qg5.py` avaliava o Estágio 2 usando
`np.argmax` puro, ignorando os thresholds otimizados por Youden publicados
em `models/stage2_thresholds_v2.3.json`. Isso validava o modelo em um ponto
de operação diferente do pipeline real (`TwoStageMLPPipeline`). Foi corrigido
para aplicar os thresholds via `_apply_stage2_thresholds`.

### 3.3 Pouca capacidade de modelo para fronteira S/F

O MLP do Estágio 2 tem uma única camada oculta:

```
16 → hidden_units → Dropout(0.3) → 3 (softmax)
```

Para uma fronteira não-linear entre S, V e F, essa capacidade pode ser
insuficiente. Experimentos empíricos (`hidden_units=128`) melhoraram as
métricas, mas o F1(F) permaneceu abaixo do desejável no `argmax`.

### 3.4 Amostragem do teste pode ser volátil

O teste amostra até 683 exemplos por classe (`STAGE2_MAX_SAMPLES_PER_CLASS`).
Essa amostra pode ter alta variância para F, cuja população total é pequena,
 especialmente se os 683 forem sorteados de poucos registros.

## 4. Recomendações futuras

1. **Corrigir o scaling do Estágio 1** para salvar RAW no NPZ e retreinar o
   modelo. Isso simplifica a portabilidade para firmware (C) e evita
   dependência de dupla normalização.
2. **Explorar `target_ratio` de SMOTE** de forma mais ampla (0.5, 0.75, 1.0)
   e monitorar overfit na classe F via validação por paciente (GroupKFold).
3. **Investigar perdas alternativas** que reponderam por frequência efetiva,
   como Class-Balanced Focal Loss (`(1-beta)/(1-beta^n`) ou LDAM (Label-
   Distribution-Aware Margin), que introduzem margens geométricas explícitas
   para minoritárias.
4. **Otimizar thresholds conjuntamente** para F1-macro, não apenas por
   Youden independente, ou calibrar o modelo com temperature scaling antes
   da busca de thresholds.
5. **Adicionar validação fisiológica pós-SMOTE** para descartar amostras
   sintéticas fora da variedade real (ex.: `rr_ratio` inconsistente,
   `qrs_width_ms` fora de [20, 180] ms).
6. **Aumentar a capacidade do MLP** do Estágio 2 (mais camadas ou camada
   intermediária maior) se o overfit permanecer controlado, ou introduzir um
   classificador específico para F em cascata.

## 5. Resultados após correções

| Métrica | Valor | Threshold QG5' |
|---------|-------|----------------|
| Estágio 1 Recall(Anormal) | 0.8352 | ≥ 0.30 |
| Estágio 1 Precision(Anormal) | 0.8286 | ≥ 0.25 |
| Estágio 1 F1-macro | 0.9021 | ≥ 0.55 |
| Estágio 2 F1-macro | 0.5493 | ≥ 0.45 |
| Estágio 2 F1(S) | 0.6532 | ≥ 0.55 |
| Estágio 2 F1(V) | 0.7946 | ≥ 0.70 |
| Estágio 2 F1(F) | 0.2000 | ≥ 0.15 |

Os thresholds publicados são:

- Stage 1: 0.6900
- Stage 2: `{'S': 0.42, 'V': 0.49, 'F': 0.27}`

`pytest tests/test_two_stage_mlp_qg5.py -v` passou com 3/3 testes.

Artefatos publicados:

- `models/stage1_float32_v2.3.keras`
- `models/input_scaler_stage1_v2.3.pkl`
- `models/stage1_threshold_v2.3.json`
- `models/stage2_float32_v2.3.keras`
- `models/input_scaler_stage2_v2.3.pkl`
- `models/stage2_thresholds_v2.3.json`

Experimentos utilizados:

- Stage 1: `experiments/stage1_mlp_features_v2.3_retrain` (hidden_units=128)
- Stage 2: `experiments/stage2_mlp_features_v2.3_focal_smote_v8`
  (hidden_units=128, alpha=[0.50, 0.30, 1.00], gamma=2.0,
  SMOTE classes [0, 2] ratio=0.5, LR fixo).

## 6. Estado atual após otimizações

- `scripts/train_stage2_mlp.py`: Focal Loss dinâmica, SMOTE apenas no fold de
  treino, CosineDecayRestarts, EarlyStopping em `val_f1_macro`, thresholds
  Youden salvos por fold.
- `src/inference/two_stage_mlp_pipeline.py`: aplica thresholds one-vs-rest com
  fallback para `argmax`.
- `scripts/select_best_mlp_fold.py`: agrega thresholds Stage 2 por mediana e
  publica `stage2_thresholds_v2.3.json`. Corrigido para aceitar
  `--stage2-features`.
- `tests/test_two_stage_mlp_qg5.py`: aplica o scaler do Stage 1 e valida o
  Estágio 2 com os thresholds publicados, refletindo o ponto de operação real.
- `docs/auditoria_arquitetura_v2.3.md`: documenta lacunas e recomendações.
