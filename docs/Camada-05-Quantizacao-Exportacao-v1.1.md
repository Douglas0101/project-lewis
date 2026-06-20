# Project-Lewis — Camada 5: Quantização e Exportação para Firmware
## Responsável: ML Engineering / Firmware Interface

**Versão:** 1.1 | **Data:** 2026-06-09 | **Arquiteto:** Douglas Souza

---

## 5.1 Objetivo
Converter o modelo float32 treinado para formato INT8 quantizado (PTQ), validar degradação de performance por classe AAMI, e exportar como header C (`model_data.h`) compatível com TensorFlow Lite Micro (TFLM) em STM32F4, garantindo ativação de CMSIS-NN, rastreabilidade de parâmetros de quantização e conformidade com restrições de memória do dispositivo alvo.

---

## 5.2 Decisão Arquitetural: Per-Channel Quantization

O TFLite Converter utiliza **per-channel (per-axis) quantization** como padrão para camadas Conv2D/DepthwiseConv2D desde TensorFlow 2.x.citeweb_search:20#4 Isso significa que cada canal de saída possui seu próprio `scale` e `zero_point`, reduzindo o erro de quantização em ~30–50% comparado a per-tensor. Para o Project-Lewis, isso é crítico: as camadas Conv1D do backbone são mapeadas internamente como Conv2D (batch=1, height=1, width=seq_len, channels=filters) no TFLite, beneficiando-se automaticamente da per-channel quantization.

| Granularidade | Scale/Zero-Point | Precisão | Overhead | TFLite Default? |
| :--- | :--- | :--- | :--- | :--- |
| Per-tensor | 1 por tensor | Baixa | Nenhum | Não |
| **Per-channel** | 1 por canal de saída | **Alta** | ~N_channels × 4 bytes | **Sim** |
| Per-block | 1 por bloco | Muito alta | Alto | Não (TFLite) |

> **Nota:** Não é necessário configurar explicitamente per-channel no converter. O TFLite Converter aplica automaticamente quando `target_spec.supported_ops = [TFLITE_BUILTINS_INT8]`.citeweb_search:20#4

---

## 5.3 Quantização Pós-Treino (PTQ)

### Correção Crítica: Representative Dataset

O documento v1.0 sugeria "~1000 amostras aleatórias estratificadas". **Isso está incompleto.** A literatura de quantização indica que:

- **Tamanho mínimo:** 128–1024 amostras são suficientes para PTQ INT8, desde que diversas.citeweb_search:20#0citeweb_search:20#6
- **Qualidade > Quantidade:** Um conjunto pequeno mas diverso (cobrindo todas as classes e padrões de ativação) supera um conjunto grande e homogêneo.citeweb_search:20#0
- **Estratificação obrigatória:** Deve cobrir todas as 5 classes AAMI (N, V, S, F, Q) proporcionalmente, incluindo batimentos com ruído, baseline wander e artefatos para calibrar corretamente os ranges de ativação.
- **Sem labels:** O representative dataset não precisa de labels; apenas forward pass para observar min/max dos tensores intermediários.citeweb_search:19#3

### src/quantization/ptq.py
```python
def quantize_ptq(model, representative_data: np.ndarray, 
                 num_calibration_samples: int = 512) -> Tuple[bytes, dict]:
    """Quantização pós-treino para INT8 com per-channel quantization.

    1. Selecionar representative dataset estratificado:
       a. Amostrar de cada classe AAMI proporcionalmente
       b. Incluir batimentos de baixa/high amplitude (para calibrar ranges extremos)
       c. Total: num_calibration_samples (default 512)
    2. Criar generator:
       def representative_dataset():
           for i in range(0, len(X_repr), batch_size):
               yield [X_repr[i:i+batch_size].astype(np.float32)]
    3. Configurar converter:
       converter = tf.lite.TFLiteConverter.from_keras_model(model)
       converter.optimizations = [tf.lite.Optimize.DEFAULT]
       converter.representative_dataset = representative_dataset
       converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
       converter.inference_input_type = tf.int8    # input int8
       converter.inference_output_type = tf.int8   # output int8
    4. Converter: tflite_model = converter.convert()
    5. Extrair parâmetros de quantização:
       interpreter = tf.lite.Interpreter(model_content=tflite_model)
       in_det = interpreter.get_input_details()[0]
       out_det = interpreter.get_output_details()[0]
       input_scale, input_zero_point = in_det['quantization']
       output_scale, output_zero_point = out_det['quantization']
    6. Validar degradação:
       - Avaliar float32 e INT8 no MESMO conjunto de teste (inter-patient)
       - Registrar ΔAcc, ΔF1-macro, ΔSens por classe
       - Se ΔF1-macro > 2%: aumentar calibration samples ou usar QAT
    7. Retornar: (tflite_bytes, quantization_params)
    """
```

> **Cuidado com degradação:** Em modelos multi-task complexos (12-lead, 75 classes), PTQ INT8 pode degradar drasticamente (AUC de 0.893 → 0.513).citeweb_search:19#8 Para 1D-CNN simples (single-lead, 5 classes), a degradação típica é 0–1% se calibrado corretamente.citeweb_search:19#1 Se ΔF1-macro > 2%, considerar QAT (Quantization Aware Training) ou aumentar calibration samples para 1024+.

---

## 5.4 Extração de Parâmetros de Quantização para Firmware

O firmware C precisa saber como converter o sinal ADC (float32/mV) para int8 e como interpretar o output int8 como probabilidades float32.

### src/quantization/extract_params.py
```python
def extract_quantization_params(tflite_model: bytes) -> dict:
    """Extrair scale e zero_point de input e output para uso no firmware.

    Fórmulas de conversão:
    - Float → INT8:  q = round(f / scale + zero_point)
    - INT8 → Float:  f = (q - zero_point) * scale

    Retorna:
    {
        "input": {"scale": float, "zero_point": int, "dtype": "int8"},
        "output": {"scale": float, "zero_point": int, "dtype": "int8"},
        "interpreter": "tflite"
    }
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    return {
        "input": {
            "scale": float(in_det["quantization_parameters"]["scales"][0]),
            "zero_point": int(in_det["quantization_parameters"]["zero_points"][0]),
            "shape": in_det["shape"].tolist(),
            "dtype": str(in_det["dtype"]).replace("<class '", "").replace("'>", ""),
        },
        "output": {
            "scale": float(out_det["quantization_parameters"]["scales"][0]),
            "zero_point": int(out_det["quantization_parameters"]["zero_points"][0]),
            "shape": out_det["shape"].tolist(),
            "dtype": str(out_det["dtype"]).replace("<class '", "").replace("'>", ""),
        },
    }
```

**Uso no firmware C:**
```c
// Parâmetros extraídos do JSON e compilados como macros
#define INPUT_SCALE   0.0039215689f   // 1/255
#define INPUT_ZP      0
#define OUTPUT_SCALE  0.0039215689f
#define OUTPUT_ZP     0

// Conversão float (mV) → int8 para TFLM
int8_t float_to_int8(float val) {
    int32_t q = (int32_t)roundf(val / INPUT_SCALE + INPUT_ZP);
    if (q < -128) q = -128;
    if (q > 127)  q = 127;
    return (int8_t)q;
}

// Conversão int8 (logits) → float (probabilidade)
float int8_to_float(int8_t q) {
    return (q - OUTPUT_ZP) * OUTPUT_SCALE;
}
```

---

## 5.5 Cálculo da Arena TFLM

O TFLM requer uma arena de memória contígua para:
1. **Tensor arena:** tensores intermediários (ativações) alocados pelo memory planner.
2. **Scratch buffer:** buffer temporário para kernels otimizados (CMSIS-NN).
3. **Interpreter overhead:** estruturas internas do TFLM (~1–2KB).

**Estimativa para o Project-Lewis (backbone ~13K params, input 500×1):**

| Componente | Tamanho Estimado | Cálculo |
| :--- | :--- | :--- |
| Pesos INT8 | ~13KB | ~13K params × 1 byte |
| Bias INT32 | ~2KB | ~200 biases × 4 bytes |
| Scale/Zero-Point per-channel | ~1KB | ~64 canais × 2 × 4 bytes |
| Tensor arena (ativações) | ~15–25KB | Máximo tensor: 500×64 = 32KB (float32) → 8KB (INT8) + overhead |
| Scratch buffer (CMSIS-NN) | ~4–8KB | Buffer para matmul otimizado |
| Interpreter overhead | ~2KB | Estruturas internas |
| **Total Arena** | **~40–50KB** | **< 64KB (limite STM32F4)** |

> **Validação:** O tamanho exato da arena só pode ser determinado em runtime via `Interpreter::arena_used_bytes()` ou `RecordingMicroInterpreter`. O valor acima é uma estimativa conservadora. Para garantir, alocar 64KB e monitorar `arena_used_bytes` no teste de integração.

---

## 5.6 Exportação para Header C

O documento v1.0 usava `xxd -i`, uma ferramenta externa dependente de ambiente Unix. **Isso é frágil.** O pipeline deve usar um script Python nativo, cross-platform e determinístico.

### src/quantization/export_tflite.py
```python
def tflite_to_header(tflite_bytes: bytes, output_path: Path, 
                     var_name: str = "g_ecg_model_data",
                     alignment: int = 16) -> Path:
    """Exportar .tflite para header C puro, sem dependências externas.

    1. Gerar array C a partir dos bytes do FlatBuffer
    2. Aplicar alinhamento: alignas(16) para ARM Cortex-M4
       (16 bytes é o padrão seguro; cache line do M4 é 32 bytes,
        mas nem todos os STM32F4 têm D-Cache habilitado)
    3. Incluir metadata em comentários:
       - SHA256 do FlatBuffer
       - Tamanho em bytes
       - Data de geração
       - Versão do modelo
    4. Salvar: output_path (ex: firmware/src/ml/model_data.h)
    5. Validar: compilação com arm-none-eabi-gcc -c
    """

    header = f"""#ifndef ECG_MODEL_DATA_H
#define ECG_MODEL_DATA_H

#include <stdint.h>
#include <stdalign.h>

/* Model: Project-Lewis ECG Classifier
 * Generated: {datetime.now().isoformat()}
 * Size: {len(tflite_bytes)} bytes
 * SHA256: {sha256(tflite_bytes).hexdigest()}
 * Alignment: {alignment} bytes
 */

alignas({alignment}) const unsigned char {var_name}[] = {{
"""

    # Gerar bytes em formato C (16 por linha)
    for i in range(0, len(tflite_bytes), 16):
        chunk = tflite_bytes[i:i+16]
        hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
        header += f"    {hex_str},
"

    header += f"""}};

const int {var_name}_len = {len(tflite_bytes)};

#endif /* ECG_MODEL_DATA_H */
"""

    output_path.write_text(header, encoding="utf-8")
    return output_path
```

**Validação de compilação:**
```bash
arm-none-eabi-gcc -c -mcpu=cortex-m4 -mthumb -O3   -Wall -Werror   firmware/src/ml/model_data.h -o /tmp/model_data.o

arm-none-eabi-size /tmp/model_data.o
# Esperado: .text = tflite_bytes_len, .data = 0, .bss = 0
```

---

## 5.7 Integração CMSIS-NN

Para maximizar performance no Cortex-M4 (DSP extensions, 1 MAC/cycle), o TFLM deve ser compilado com kernels otimizados do CMSIS-NN.citeweb_search:19#4

### Build TFLM com CMSIS-NN
```bash
# Clone do tflite-micro
make -f tensorflow/lite/micro/tools/make/Makefile   TARGET=stm32f4   OPTIMIZED_KERNEL_DIR=cmsis_nn   person_detection_int8_bin

# Ou via CMake/PlatformIO
# Adicionar -DOPTIMIZED_KERNEL_DIR=cmsis_nn ao build
```

**Requisitos:**
- Incluir `CMSIS-DSP` e `CMSIS-NN` no projeto.
- Ativar `-DARM_MATH_CM4` e `-D__FPU_PRESENT=1`.
- Usar `-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard` para hardware FPU.

> **Performance:** CMSIS-NN acelera Conv2D/DepthwiseConv2D em ~4.6× no Cortex-M4 comparado a kernels de referência.citeweb_search:19#4 Para o Project-Lewis, isso significa inferência < 50ms/beat @ 64MHz (estimativa conservadora).

---

## 5.8 Linhagem de Quantização

Cada exportação gera um registro de linhagem em `data/lineage/quantization/{timestamp}.json`:

```json
{
  "model_source": "models/finetuned_float32_v1.0.keras",
  "model_source_sha256": "abc123...",
  "preprocess_config": "config/preprocess_v1.0.yaml",
  "quantization": {
    "type": "PTQ",
    "precision": "INT8",
    "granularity": "per-channel",
    "calibration_samples": 512,
    "calibration_strategy": "stratified_aami",
    "input_type": "int8",
    "output_type": "int8"
  },
  "degradation": {
    "delta_acc_global": -0.003,
    "delta_f1_macro": -0.008,
    "delta_sens_N": 0.0,
    "delta_sens_V": -0.015,
    "delta_sens_S": -0.022,
    "delta_sens_F": -0.031,
    "delta_sens_Q": -0.010
  },
  "output": {
    "tflite_path": "models/model_int8_v1.0.tflite",
    "header_path": "firmware/src/ml/model_data.h",
    "flatbuffer_size_bytes": 24576,
    "quantization_params": {
      "input": {"scale": 0.00392, "zero_point": 0},
      "output": {"scale": 0.00392, "zero_point": 0}
    }
  },
  "tflm": {
    "target": "STM32F4",
    "arena_size_kb": 64,
    "cmsis_nn": true,
    "estimated_inference_ms": 45
  },
  "timestamp": "2026-06-09T22:15:00Z"
}
```

---

## 5.9 Dead Letter Queue (DLQ) para Quantização

Falhas de PTQ são logadas em `data/.dlq/quantization_failures.jsonl`:

```json
{
  "model_source": "models/finetuned_float32_v1.0.keras",
  "error": "Some ops are not supported by the TFLite runtime",
  "unsupported_ops": ["BatchNormalization"],
  "traceback": "...",
  "calibration_samples": 512,
  "timestamp": "2026-06-09T22:15:00Z"
}
```

**Regras:**
- Op não suportado → remover op do modelo (ex: BatchNorm) e re-treinar.
- Degradação excessiva (ΔF1-macro > 2%) → aumentar calibration samples para 1024, ou usar QAT, ou revisar arquitetura (remover camadas sensíveis a quantização).
- FlatBuffer > 64KB → reduzir embedding_dim ou número de filtros Conv1D.

---

## 5.10 Quality Gate QG6

| Critério | Valor | Como Validar |
| :--- | :--- | :--- |
| Quantização | Per-channel INT8 | `interpreter.get_input_details()[0]['quantization_parameters']['scales']` tem len > 1 para Conv |
| Calibration samples | 512–1024, estratificado AAMI | `len(representative_data) >= 512` com distribuição proporcional |
| Degradação ΔAcc global | < 1% | `evaluate.py` float32 vs INT8 no mesmo teste inter-patient |
| Degradação ΔF1-macro | < 2% | `evaluate.py` float32 vs INT8 |
| Degradação ΔSens N | < 0.5% | `evaluate.py` por classe |
| Degradação ΔSens V,S,F,Q | < 3% | `evaluate.py` por classe |
| Tamanho FlatBuffer | < 64KB | `os.path.getsize("model_int8.tflite") < 65536` |
| Arena TFLM estimada | < 64KB | `RecordingMicroInterpreter` ou estimativa por tensor analysis |
| Header compilável | Sim | `arm-none-eabi-gcc -c -Werror` |
| Alinhamento | 16 bytes | `alignas(16)` presente no header |
| CMSIS-NN | Ativado | Build com `OPTIMIZED_KERNEL_DIR=cmsis_nn` |
| Parâmetros de quantização extraídos | Sim | `quantization_params.json` existe e válido |
| Linhagem | Completa | `data/lineage/quantization/*.json` existe |
| DLQ vazia | 0 falhas | `data/.dlq/quantization_failures.jsonl` vazio |

**Teste:** `pytest tests/test_quantization.py -v`

---

## 5.11 Pipeline de Exportação Completo

```bash
# 1. Quantizar
python -m src.quantization.ptq   --model models/finetuned_float32_v1.0.keras   --calibration-data data/processed/mitbih_calibration.npy   --output models/model_int8_v1.0.tflite

# 2. Extrair parâmetros
python -m src.quantization.extract_params   --tflite models/model_int8_v1.0.tflite   --output models/quantization_params_v1.0.json

# 3. Exportar header C
python -m src.quantization.export_tflite   --tflite models/model_int8_v1.0.tflite   --output firmware/src/ml/model_data.h   --var-name g_ecg_model_data

# 4. Validar compilação
arm-none-eabi-gcc -c -mcpu=cortex-m4 -mthumb -O3 -Werror   firmware/src/ml/model_data.h -o /tmp/model_data.o

# 5. Teste de integração TFLM (simulador ou hardware)
pytest tests/test_tflm_integration.py -v
```

---

## 5.12 Referências Verificadas

- TensorFlow Lite Post-Training Integer Quantization (input/output int8, representative dataset): https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant — `converter.inference_input_type = tf.uint8`, `converter.inference_output_type = tf.uint8`, representative dataset generator.citeweb_search:19#0
- PTQ INT8 Degradação em ECG (multi-task 12-lead): Rajotte et al., IEEE — AUC caiu de 0.893 (float32) para 0.513 (INT8) em modelo complexo; DRQ manteve 0.893.citeweb_search:19#8
- PTQ INT8 Meta-Análise (modelos médicos): MDPI 2026 — "PTQ achieves the highest Accuracy and Specificity (≈90–94%)... INT8 quantization is generally regarded as the practical sweet spot between accuracy and efficiency... minimal clinical variation (Δclinical≈0–1pp)".citeweb_search:19#1
- Calibration Data Size (128–1024 samples, diversidade > quantidade): https://apxml.com/courses/quantized-llm-deployment/chapter-1-advanced-llm-quantization-fundamentals/calibration-data-selection — "Common sizes range from 128 to 1024 samples. Focus on diversity rather than sheer quantity."citeweb_search:20#6
- Calibration Data Curation (COLA framework): arXiv 2510.10618 — "calibration data is extremely limited in size (often <1K samples)... quality over quantity".citeweb_search:20#0
- Per-Channel Quantization (TFLite default): https://developer.axis.com/computer-vision/on-device/quantization/ — "Per-channel quantization is the process of quantizing the weights of the model using a different scale for each channel... This is also the default quantization method in TensorFlow 2."citeweb_search:20#4
- CMSIS-NN + TFLM Integration (Cortex-M4, 4.6× speedup): https://blog.tensorflow.org/2021/02/accelerated-inference-on-arm-microcontrollers-with-tensorflow-lite.html — "optimized versions of the TensorFlow Lite kernels that use CMSIS-NN... 4.6x performance uplift".citeweb_search:19#4
- TFLM INT8 Issue (input_scale, input_zero_point extraction): https://github.com/tensorflow/tensorflow/issues/99052 — exemplo de extração de `input_scale`, `input_zero_point`, `output_scale`, `output_zero_point` para inference.citeweb_search:19#2
- QAT per-axis (per-channel) default: https://www.tensorflow.org/model_optimization/guide/quantization/training — "Currently only supports per-axis quantization for convolutional layers, not per-tensor quantization."citeweb_search:20#8
