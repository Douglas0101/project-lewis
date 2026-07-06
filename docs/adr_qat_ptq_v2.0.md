# ADR-005: Estratégia de Quantização — PTQ INT8 com Tentativa de QAT e Fallback Automático

**Status:** Aprovado  
**Data:** 2026-06-30  
**Versão:** 2.0  
**Autor:** Douglas Souza  
**Relacionado a:** `docs/UNIFIED_DOCUMENT_v2.0.md`, `docs/SDD_Project-Lewis_v3.md`, `AGENTS.md`

---

## 1. Contexto

O `UNIFIED DOCUMENT v2.0` e o `SDD_Project-Lewis_v3.md` definem, para a Camada C05 (Quantização/Exportação), o uso de **Quantization-Aware Training (QAT)** como técnica preferencial de compactação dos modelos do Project-Lewis, combinada a pruning estruturado de 30% nos filtros `Conv1D` do Estágio 1.

No entanto, o ambiente de treinamento atual utiliza:

| Componente | Versão |
|------------|--------|
| TensorFlow | 2.21.0 |
| Keras      | 3.14.1 |
| `tensorflow-model-optimization` | 0.8.x (presente em `pyproject.toml`, mas incompatível em runtime) |

Quando `src/models/pruning_qat.py::apply_qat()` tenta envolver o modelo com `tfmot.quantization.keras.quantize_model()`, o wrapper de QAT rejeita as camadas construídas pelo Keras 3.x. O erro manifestado em execuções do pipeline inclui mensagens como:

```text
ValueError: to_annotate can only be a keras.layers.Layer instance
ValueError: to_quantize can only either be a keras Sequential or Functional model
```

Isso torna o QAT **não utilizável de forma confiável** no ambiente `tf-keras` atual, sem quebrar as restrições do projeto:

- `AGENTS.md` / `UNIFIED_DOCUMENT` proíbem a adição de novas dependências (a não ser as já declaradas).
- Não é viável realizar downgrade do TensorFlow para uma versão com Keras 2.x, pois o projeto está fixado em Python 3.12.x e TensorFlow 2.21.
- Os modelos v2.0 já existem (`models/stage1_float32_v2.0.keras`, `models/stage2_float32_v2.0.keras`) e o treinamento do Estágio 1 está em andamento; a quantização não pode ficar bloqueada pela indisponibilidade do QAT.

---

## 2. Decisão

**Manter a Post-Training Quantization (PTQ) full-integer INT8 como padrão do pipeline de quantização v2.0**, preservando a tentativa de QAT como caminho opcional com fallback automático para PTQ quando o QAT falhar.

Concretamente:

1. **Padrão de quantização:** PTQ INT8 full-integer per-channel, conforme já implementado em `src/quantization/ptq.py`, com dataset representativo estratificado (`representative_dataset_stratified`) e calibração por amostragem das classes AAMI.
2. **Tentativa de QAT:** `src/models/pruning_qat.py::apply_qat()` continua tentando carregar `tensorflow_model_optimization` e aplicar `quantize_model`. Se a aplicação levantar `ValueError`/`RuntimeError`, o pipeline registra um warning e prossegue com o modelo float32 original, que será quantizado via PTQ.
3. **Pruning estruturado:** permanece como etapa independente, executada antes da quantização, via `src/models/pruning_qat.py::apply_structured_pruning()`. A redução de 30% dos filtros `Conv1D` é mantida como meta para o Estágio 1, quando houver dados de treino/validação disponíveis.
4. **Nenhuma dependência nova:** não se adiciona pacotes alternativos de QAT (ex.: `ai-edge-torch`, `onnxruntime`) nem se quebra a pinagem de versão do TensorFlow.

---

## 3. Implementação de Referência

| Arquivo | Papel |
|---------|-------|
| `src/models/pruning_qat.py` | Pruning estruturado de canais, tentativa de QAT com fallback silencioso e conversão INT8. |
| `scripts/apply_pruning_qat.py` | CLI que orquestra o pipeline completo (pruning → fine-tune → QAT/PTQ → `.tflite`). |
| `src/quantization/ptq.py` | PTQ INT8 full-integer, datasets representativos (aleatório/estratificado) e validação de I/O int8. |
| `tests/test_quantization_degradation.py` | Valida QG6: compara float32 vs INT8 e garante `ΔF1-macro < 2%`. |
| `models/quantized/stage1_int8_v2.0.tflite` | Modelo quantizado do Estágio 1 (PTQ), 54,36 KB. |
| `models/quantized/stage2_int8_v2.0.tflite` | Modelo quantizado do Estágio 2 (PTQ), 54,47 KB. |

---

## 4. Consequências

### 4.1 Positivas

- **Compatibilidade garantida:** a PTQ INT8 funciona com o ambiente `tf-keras` atual e gera FlatBuffers executáveis pelo TFLM no STM32F4.
- **Simplicidade operacional:** não é necessário treinar com QAT, reduzindo o tempo de experimentação e a superfície de erros.
- **Manutenção do QG6:** os modelos quantizados v2.0 existentes respeitam o limite de 64 KB por FlatBuffer (QG6 do `AGENTS.md`).
- **Fallback resiliente:** o código não falha se `tensorflow_model_optimization` estiver ausente ou incompatível; o pipeline continua e entrega um modelo INT8.

### 4.2 Negativas / Riscos

- **Possível perda relativa ao QAT ideal:** em cenários com distribuições de ativação extremas, a PTQ pode perder até ~1–2 p.p. de F1-macro em relação a um QAT bem calibrado. O threshold `ΔF1-macro < 2%` do QG6 cobre essa degradação.
- **Calibração depende do dataset representativo:** a qualidade da PTQ está atrelada à amostragem das classes. Classes raras (S, F) devem estar representadas; a função `representative_dataset_stratified` mitiga isso.
- **QAT não pode ser reabilitado automaticamente:** quando a incompatibilidade do `tfmot` for corrigida (Keras 2.x compatível ou nova versão do `tfmot`), será necessário alterar `apply_qat()` para remover o fallback.

---

## 5. Alternativas Consideradas

| Alternativa | Avaliação | Veredicto |
|-------------|-----------|-----------|
| **A. Forçar QAT com `tensorflow_model_optimization` atual** | Gera `ValueError` em runtime com Keras 3.x. | ❌ Rejeitada — bloqueia o pipeline. |
| **B. Fazer downgrade para TensorFlow 2.15 + Keras 2.x** | Quebra a stack aprovada (Python 3.12.x, TF 2.21) e requer revalidação de todo o pipeline. | ❌ Rejeitada — viola `AGENTS.md`. |
| **C. Adicionar framework alternativo de QAT** | Introduziria dependências não previstas e aumentaria a complexidade de build/exportação. | ❌ Rejeitada — proibido adicionar dependências. |
| **D. Manter PTQ INT8 como padrão, com tentativa/fallback de QAT** | Compatível, simples, atende QG6 e mantém a porta aberta para QAT futuro. | ✅ Aprovada. |

---

## 6. Dados de Referência Atuais (v2.0)

| Métrica | Estágio 1 | Estágio 2 | Fonte |
|---------|-----------|-----------|-------|
| Modelo float32 | `models/stage1_float32_v2.0.keras` | `models/stage2_float32_v2.0.keras` | Artefatos existentes |
| Modelo INT8 | `models/quantized/stage1_int8_v2.0.tflite` | `models/quantized/stage2_int8_v2.0.tflite` | PTQ gerada |
| Tamanho INT8 | 54,36 KB | 54,47 KB | `models/quantized/quantization_summary_v2.0.json` |
| Limite QG6 | < 64 KB | < 64 KB | `AGENTS.md` |
| F1-macro float32 (DS2) | 0,5927 | 0,5185 | `reports/two_stage_evaluation_v2.0.json` |
| Degradação PTQ vs float32 | < 2 p.p. (a confirmar nos testes em andamento) | < 2 p.p. (a confirmar nos testes em andamento) | `tests/test_quantization_degradation.py` |

> **Nota:** Os valores de degradação exatos dos modelos v2.0 serão preenchidos assim que o treinamento em background do Estágio 1 for concluído e os testes de quantização forem reexecutados. Até lá, usa-se o threshold do QG6 (`ΔF1-macro < 2%`) como meta.

---

## 7. Conformidade

- `AGENTS.md` — QG6 (FlatBuffer < 64 KB, arena < 64 KB) mantido.
- `docs/UNIFIED_DOCUMENT_v2.0.md` — Decisão 8 (Pruning + QAT) parcialmente atendida: pruning mantido; QAT convertido para tentativa com fallback PTQ.
- `docs/SDD_Project-Lewis_v3.md` — Camada C05 (Quantização/Exportação) atualizada para refletir PTQ INT8 como estratégia padrão.
- Nenhuma dependência foi adicionada.

---

## 8. Referências

- `docs/UNIFIED_DOCUMENT_v2.0.md` — Seção S6 / RF-04 / ADR-004.
- `docs/SDD_Project-Lewis_v3.md` — Camada C05 e Regras de Ouro.
- `src/models/pruning_qat.py`
- `src/quantization/ptq.py`
- `tests/test_quantization_degradation.py`
- TensorFlow Model Optimization: https://www.tensorflow.org/model_optimization
