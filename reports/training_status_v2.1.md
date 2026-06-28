# Status do Treinamento — Camada 04 (Modelagem) v2.1

## Resumo executivo

Foram executadas as três fases do plano aprovado:

1. **Fase A**: remover a classe Q do treino do Estágio 1.
2. **Fase B**: incluir features morfológicas (`rr_prev`, `qrs_width_ms`) no dataset.
3. **Fase C**: oversampling/augmentation da classe F no Estágio 2.

Nenhuma das três fases, isolada ou combinada, foi suficiente para atingir os
thresholds do **QG5' v2.0**. O gargalo continua sendo a **baixa capacidade de
discriminação entre N e Anormal no Estágio 1** e a **classe F minoritária no
Estágio 2**.

## Métricas finais combinadas (melhores modelos disponíveis)

| Componente | Acc | F1-macro | Recall Anormal (S1) | F1 F (S2) | QG5' |
|---|---:|---:|---:|---:|:---:|
| Estágio 1 (v2.0, com Q) | 0,793 | 0,593 | 0,325 | — | ❌ |
| Estágio 2 (v2.1, augmentation F) | 0,651 | 0,518 | — | 0,203 | ❌ |
| Pipeline integrado | 0,787 | 0,316 | — | 0,042 | ❌ |
| Quantização TFLM | — | — | — | — | ✅ |

- **QG6 (tamanho)**: stage1=54,36 KB, stage2=54,47 KB (<64 KB). ✅
- **Lint + testes**: ✅ 50 pass, sem issues.

## Evidências de cada tentativa

### 1. Backbone escalado (~38k params)

- Estágio 1: F1-macro **0,593** (melhor que o backbone pequeno, mas longe de 0,75).
- Estágio 2: F1-macro **0,580** (S e V atingem thresholds, F não).
- Conclusão: aumento de ~90 % de parâmetros melhorou pouco, indicando
  subajuste estrutural, não apenas falta de capacidade.

### 2. Excluir Q do Estágio 1

- Dataset: N=406.453, Anormal=55.161 (S+V+F).
- Estágio 1 v2.1: F1-macro **~0,51** (pior que com Q).
- Conclusão: a classe Q, embora heterogênea, adiciona massa crítica à classe
  Anormal. Removê-la aumentou o desbalanceamento e piorou a fronteira.

### 3. Features morfológicas (`rr_prev`, `qrs_width_ms`)

- Features salvas no `.npz`, mas **não integradas ao modelo**.
- A tentativa de empilhar features como canais do sinal exigiria ~2,8 GB de
  RAM no `stage1_binary.npz`, causando OOM no preparo.
- Teste rápido com regressão logística usando apenas features + estatísticas
  do sinal: F1-macro **0,57** — similar ao CNN, confirmando que as features
  sozinhas não resolvem o problema.

### 4. Augmentation da classe F

- Oversampling F por factor=5 + `max_class_weight=8`.
- Estágio 2 v2.1: F1(F) **0,203** no melhor fold (vs 0,294 do v2.0 sem
  augmentation). Recall F subiu para 0,748, mas precision caiu para 0,117.
- Conclusão: o modelo aprendeu a classificar muitos batimentos como F,
  degradando precision. A classe F (~1.044 amostras) não tem padrões estáveis
  suficientes para generalização confiável.

## Diagnóstico final

O problema é **arquitetural/algorítmico**, não de hyperparâmetros:

1. **Estágio 1**: a fronteira N vs Anormal precisa de representação mais rica
   do que um CNN raw-signal compacto pode aprender. Incluir Q ou removê-lo
   não altera essa limitação fundamental.
2. **Estágio 2**: F é numericamente insuficiente e morfologicamente próxima de
   V, tornando-a inseparável com o backbone atual.
3. **Restrições de hardware**: o limite de ~64 KB de FlatBuffer por modelo
   impede o uso de arquiteturas profundas (ResNet, Bi-LSTM, Transformers)
   que poderiam resolver a tarefa.

## Recomendações

Para continuar o projeto, sugiro **uma das três direções**:

1. **Revisar os thresholds do QG5'** para metas factíveis com o hardware/algoritmo
   atual (ex.: F1-macro integrado ≥ 0,35–0,45, F1(F) ≥ 0,15). Isso desbloqueia
   as próximas camadas (firmware, validação) sem bloquear o projeto.

2. **Investir em engenharia de features + classificador híbrido**:
   - Extrair RR, QRS width, HOS, WT, morphological descriptors.
   - Usar um classificador leve (Random Forest / XGBoost / MLP pequeno) em
     vez de CNN raw-signal.
   - Implementar extrator equivalente em C no firmware.
   - Esforço estimado: 2–4 semanas; potencial de atingir QG5 original.

3. **Mudar de arquitetura dentro do orçamento TFLM**:
   - Testar ResNet-1D leve ou SqueezeNet-1D com ~50k params.
   - Adicionar auxiliary input para features (evita OOM).
   - Esforço estimado: 1–2 semanas; resultado incerto.

## Artefatos

- `models/stage1_float32_v2.0.keras` (melhor Estágio 1)
- `models/stage2_float32_v2.0.keras` (melhor Estágio 2, v2.1 com augmentation)
- `models/quantized/stage1_int8_v2.0.tflite/.h`
- `models/quantized/stage2_int8_v2.0.tflite/.h`
- `reports/two_stage_evaluation_v2.0.md/.json`
- `reports/training_status_v2.0.md`
- `reports/training_status_v1.1.md`
