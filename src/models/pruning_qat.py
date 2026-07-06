"""Pruning estruturado de canais e Quantization-Aware Training (QAT).

Este módulo implementa otimizações avançadas para modelos Keras 1D-CNN do
Project-Lewis:

1. **Pruning estruturado de canais**: remove filtros inteiros das camadas
   ``Conv1D`` com base na norma L1 dos kernels, reduzindo a largura da rede
   sem esparsidade aleatória.
2. **Fine-tuning pós-pruning**: retreina a rede podada para recuperar
   acurácia.
3. **QAT**: aplica ``tfmot.quantization.keras.quantize_model`` para simular
   erros de quantização durante o treinamento.
4. **Conversão INT8**: converte para TFLite full-integer com dataset
   representativo, compatível com TFLM.
5. **Exportação de parâmetros de quantização**: gera JSON com escalas e
   zero-points de entrada/saída.

Restrições:
- Não adiciona dependências além das já existentes.
- Compatível com modelos do tipo backbone 1D-CNN do Project-Lewis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import tensorflow as tf

LOGGER = logging.getLogger("lewis.camada04.pruning_qat")

# ---------------------------------------------------------------------------
# Importação lazy do TensorFlow Model Optimization
# ---------------------------------------------------------------------------
_tfmot: Optional[Any] = None


def _get_tfmot() -> Any:
    """Importa tensorflow_model_optimization sob demanda.

    Returns
    -------
    Any
        Módulo ``tensorflow_model_optimization`` ou ``None`` caso não esteja
        instalado.
    """
    global _tfmot  # noqa: PLW0603
    if _tfmot is None:
        try:
            import tensorflow_model_optimization as tfmot  # type: ignore[import]
        except ImportError:
            LOGGER.warning("tensorflow_model_optimization não instalado; QAT será desabilitado.")
            return None
        _tfmot = tfmot
    return _tfmot


# ---------------------------------------------------------------------------
# Pruning estruturado de canais
# ---------------------------------------------------------------------------


def get_conv_filter_norms(model: tf.keras.Model) -> Dict[str, np.ndarray]:
    """Computa a norma L1 de cada filtro para todas as camadas Conv1D.

    Parameters
    ----------
    model : tf.keras.Model
        Modelo Keras contendo camadas ``Conv1D``.

    Returns
    -------
    dict[str, np.ndarray]
        Dicionário ``{nome_da_camada: normas_por_filtro}``.
    """
    norms: Dict[str, np.ndarray] = {}
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D):
            kernel = layer.get_weights()[0]  # shape: (kernel_size, in_channels, out_channels)
            filter_norms = np.sum(np.abs(kernel), axis=(0, 1))
            norms[layer.name] = filter_norms
    return norms


def _compute_keep_indices(
    filter_norms: np.ndarray,
    target_sparsity: float,
) -> np.ndarray:
    """Seleciona índices de filtros a serem mantidos.

    Mantém os filtros com maior norma L1 (mais importantes).

    Parameters
    ----------
    filter_norms : np.ndarray
        Norma L1 de cada filtro.
    target_sparsity : float
        Fração de filtros a remover (ex.: 0.30 remove 30%).

    Returns
    -------
    np.ndarray
        Índices dos filtros mantidos, em ordem crescente.
    """
    n_filters = len(filter_norms)
    n_keep = max(1, int(np.floor(n_filters * (1.0 - target_sparsity))))
    if n_keep >= n_filters:
        return np.arange(n_filters, dtype=np.int64)
    top_indices = np.argsort(filter_norms)[-n_keep:]
    return np.sort(top_indices)


def _clone_layer(layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
    """Clona uma camada Keras a partir de sua configuração."""
    return layer.__class__.from_config(layer.get_config())


def _rebuild_pruned_model(
    model: tf.keras.Model,
    keep_indices: Dict[str, np.ndarray],
) -> tf.keras.Model:
    """Reconstrói o modelo removendo canais inteiros das Conv1D.

    A reconstrução é feita camada a camada no modo funcional. Camadas
    ``Conv1D`` são substituídas por versões com menos filtros; a primeira
    camada ``Dense`` após ``GlobalAveragePooling1D`` ou ``Flatten`` tem sua
    dimensão de entrada ajustada para refletir a última torre convolucional
    podada.

    Parameters
    ----------
    model : tf.keras.Model
        Modelo original.
    keep_indices : dict[str, np.ndarray]
        Índices de filtros mantidos por camada Conv1D.

    Returns
    -------
    tf.keras.Model
        Modelo podado com pesos copiados.
    """
    layer_inputs: Dict[str, tf.Tensor] = {}
    x: Optional[tf.Tensor] = None
    inputs: Optional[tf.keras.Input] = None

    prev_conv_name: Optional[str] = None
    conv_input_keep: Dict[str, Optional[np.ndarray]] = {}
    last_conv_keep: Optional[np.ndarray] = None

    in_dense_head = False
    dense_input_keep: Dict[str, np.ndarray] = {}
    dense_head_first_dense_seen = False

    for layer in model.layers:
        layer_type = layer.__class__.__name__

        if isinstance(layer, tf.keras.layers.InputLayer):
            input_shape = model.input_shape[1:]
            inputs = tf.keras.Input(shape=input_shape, name=layer.name)
            x = inputs
            layer_inputs[layer.name] = inputs
            continue

        if isinstance(layer, tf.keras.layers.Conv1D):
            if prev_conv_name is not None:
                conv_input_keep[layer.name] = keep_indices[prev_conv_name]
            else:
                conv_input_keep[layer.name] = None

            out_keep = keep_indices[layer.name]
            new_layer = tf.keras.layers.Conv1D(
                filters=len(out_keep),
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                activation=layer.activation,
                use_bias=layer.use_bias,
                kernel_initializer=layer.kernel_initializer,
                bias_initializer=layer.bias_initializer,
                kernel_regularizer=layer.kernel_regularizer,
                bias_regularizer=layer.bias_regularizer,
                activity_regularizer=layer.activity_regularizer,
                kernel_constraint=layer.kernel_constraint,
                bias_constraint=layer.bias_constraint,
                name=layer.name,
            )
            x = new_layer(x)
            layer_inputs[layer.name] = x

            prev_conv_name = layer.name
            last_conv_keep = out_keep
            in_dense_head = False
            continue

        if layer_type in ("MaxPooling1D", "MaxPooling2D", "AveragePooling1D"):
            new_layer = _clone_layer(layer)
            x = new_layer(x)
            layer_inputs[layer.name] = x
            continue

        if isinstance(layer, (tf.keras.layers.GlobalAveragePooling1D, tf.keras.layers.Flatten)):
            new_layer = _clone_layer(layer)
            x = new_layer(x)
            layer_inputs[layer.name] = x
            in_dense_head = True
            continue

        if isinstance(layer, tf.keras.layers.Dropout):
            new_layer = _clone_layer(layer)
            x = new_layer(x)
            layer_inputs[layer.name] = x
            continue

        if isinstance(layer, tf.keras.layers.Dense):
            if in_dense_head and not dense_head_first_dense_seen and last_conv_keep is not None:
                dense_input_keep[layer.name] = last_conv_keep
                dense_head_first_dense_seen = True

            new_layer = tf.keras.layers.Dense(
                units=layer.units,
                activation=layer.activation,
                use_bias=layer.use_bias,
                kernel_initializer=layer.kernel_initializer,
                bias_initializer=layer.bias_initializer,
                kernel_regularizer=layer.kernel_regularizer,
                bias_regularizer=layer.bias_regularizer,
                activity_regularizer=layer.activity_regularizer,
                kernel_constraint=layer.kernel_constraint,
                bias_constraint=layer.bias_constraint,
                name=layer.name,
            )
            x = new_layer(x)
            layer_inputs[layer.name] = x
            continue

        # Fallback genérico para outras camadas.
        new_layer = _clone_layer(layer)
        x = new_layer(x)
        layer_inputs[layer.name] = x

    if inputs is None or x is None:
        raise ValueError("Modelo não possui InputLayer ou não foi possível reconstruir a saída.")

    new_model = tf.keras.Model(inputs=inputs, outputs=x, name=f"{model.name}_pruned")

    # Copia pesos ajustando dimensões de entrada e saída.
    for new_layer in new_model.layers:
        name = new_layer.name
        old_layer = model.get_layer(name)

        if isinstance(new_layer, tf.keras.layers.Conv1D):
            out_keep = keep_indices[name]
            in_keep = conv_input_keep[name]
            old_kernel, *old_bias = old_layer.get_weights()
            new_kernel = old_kernel if in_keep is None else old_kernel[:, in_keep, :]
            new_kernel = new_kernel[:, :, out_keep]
            new_weights: list[np.ndarray] = [new_kernel]
            if old_layer.use_bias:
                new_weights.append(old_bias[0][out_keep])
            new_layer.set_weights(new_weights)

        elif isinstance(new_layer, tf.keras.layers.Dense) and name in dense_input_keep:
            in_keep = dense_input_keep[name]
            old_kernel, *old_bias = old_layer.get_weights()
            new_kernel = old_kernel[in_keep, :]
            new_weights = [new_kernel]
            if old_layer.use_bias:
                new_weights.append(old_bias[0])
            new_layer.set_weights(new_weights)

        elif old_layer.get_weights():
            # Outras camadas com pesos: copia diretamente (tamanhos iguais).
            new_layer.set_weights(old_layer.get_weights())

    return new_model


def apply_structured_pruning(
    model: tf.keras.Model,
    target_sparsity: float = 0.30,
    layer_sparsity: Optional[Dict[str, float]] = None,
) -> tf.keras.Model:
    """Aplica pruning estruturado de canais em camadas Conv1D.

    Parameters
    ----------
    model : tf.keras.Model
        Modelo Keras a ser podado.
    target_sparsity : float
        Fração de filtros a remover em cada Conv1D (default 0.30).
    layer_sparsity : dict[str, float], optional
        Esparsidade por camada. Se None, usa ``target_sparsity`` para todas.

    Returns
    -------
    tf.keras.Model
        Novo modelo com canais inteiros removidos.
    """
    if not 0.0 <= target_sparsity < 1.0:
        raise ValueError("target_sparsity deve estar em [0.0, 1.0)")

    norms = get_conv_filter_norms(model)
    if not norms:
        raise ValueError("Modelo não possui camadas Conv1D para pruning.")

    layer_sparsity = layer_sparsity or {}
    keep_indices: Dict[str, np.ndarray] = {}
    for layer_name, filter_norms in norms.items():
        sparsity = layer_sparsity.get(layer_name, target_sparsity)
        keep_indices[layer_name] = _compute_keep_indices(filter_norms, sparsity)
        LOGGER.info(
            "Pruning %s | mantidos %d/%d filtros",
            layer_name,
            len(keep_indices[layer_name]),
            len(filter_norms),
        )

    return _rebuild_pruned_model(model, keep_indices)


# ---------------------------------------------------------------------------
# Fine-tuning pós-pruning
# ---------------------------------------------------------------------------


def fine_tune_pruned_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    class_weight: Optional[Dict[int, float]] = None,
) -> tuple[tf.keras.Model, Dict[str, list]]:
    """Fine-tuning rápido de modelo podado.

    Parameters
    ----------
    model : tf.keras.Model
        Modelo podado.
    X_train, y_train : np.ndarray
        Dados de treino.
    X_val, y_val : np.ndarray
        Dados de validação.
    epochs : int
        Épocas de fine-tuning.
    batch_size : int
        Tamanho do batch.
    learning_rate : float
        Taxa de aprendizado.
    class_weight : dict[int, float], optional
        Pesos por classe.

    Returns
    -------
    tuple
        ``(model, history_dict)``.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        verbose=0,
    )
    return model, history.history


# ---------------------------------------------------------------------------
# Quantization-Aware Training (QAT) — opcional
# ---------------------------------------------------------------------------


def apply_qat(model: tf.keras.Model) -> tuple[tf.keras.Model, bool]:
    """Tenta envolver o modelo no pipeline de QAT do TF Model Optimization.

    O QAT depende do pacote ``tensorflow_model_optimization`` e de
    compatibilidade com a versão do ``tf-keras`` usada pelo TensorFlow. Em
    ambientes onde ``tfmot.quantization.keras.quantize_model`` rejeita as
    camadas ``Conv1D``/``Dense`` (ex.: ``ValueError`` sobre instância de
    camada), a função registra um warning e retorna o modelo original,
    sinalizando que o QAT não foi aplicado. O pipeline principal deve então
    cair no fallback de Post-Training Quantization (PTQ) full-integer INT8.

    Parameters
    ----------
    model : tf.keras.Model
        Modelo Keras (podado ou não).

    Returns
    -------
    tuple[tf.keras.Model, bool]
        ``(modelo, qat_aplicado)``. O modelo pode ser o mesmo de entrada caso
        o QAT falhe.
    """
    tfmot = _get_tfmot()
    if tfmot is None:
        LOGGER.warning("QAT desabilitado: tensorflow_model_optimization não instalado.")
        return model, False

    try:
        qat_model = tfmot.quantization.keras.quantize_model(model)
        return qat_model, True
    except (RuntimeError, ValueError) as exc:
        LOGGER.warning(
            "quantize_model falhou (%s); QAT será desabilitado. "
            "O pipeline usará PTQ full-integer INT8 como fallback.",
            exc,
        )
        return model, False


def strip_qat_wrappers(qat_model: tf.keras.Model) -> tf.keras.Model:
    """Retorna o modelo QAT (os wrappers são consumidos pelo TFLite converter).

    Em combinações recentes de ``tf-keras``/``tfmot`` o salvamento e
    recarregamento de wrappers de QAT pode falhar por incompatibilidade de
    variáveis. Como o conversor TFLite interpreta os wrappers de QAT
    diretamente, esta função apenas devolve o modelo recebido.

    Parameters
    ----------
    qat_model : tf.keras.Model
        Modelo retornado por ``apply_qat``.

    Returns
    -------
    tf.keras.Model
        Modelo QAT pronto para conversão TFLite.
    """
    return qat_model


# ---------------------------------------------------------------------------
# Conversão TFLite INT8
# ---------------------------------------------------------------------------

RepresentativeDataset = Callable[[], Iterable[list[np.ndarray]]]


def convert_to_tflite_int8(
    model: tf.keras.Model,
    representative_dataset: RepresentativeDataset,
) -> bytes:
    """Converte modelo Keras para TFLite full-integer INT8.

    Parameters
    ----------
    model : tf.keras.Model
        Modelo Keras (idealmente após QAT).
    representative_dataset : callable
        Gerador que yielda batches de entrada.

    Returns
    -------
    bytes
        FlatBuffer do modelo TFLite quantizado.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def export_quantization_params(tflite_model: bytes) -> Dict[str, Any]:
    """Extrai escalas e zero-points de entrada e saída do modelo TFLite.

    Parameters
    ----------
    tflite_model : bytes
        FlatBuffer TFLite.

    Returns
    -------
    dict[str, Any]
        Parâmetros de quantização (escalas, zero-points, shapes).
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    def _quant_params(details: Dict[str, Any]) -> Dict[str, Any]:
        qparams = details.get("quantization_parameters", {})
        scales = qparams.get("scales", np.array([]))
        zero_points = qparams.get("zero_points", np.array([]))
        return {
            "scale": float(scales[0]) if len(scales) else 1.0,
            "zero_point": int(zero_points[0]) if len(zero_points) else 0,
            "dtype": str(details["dtype"]),
        }

    input_params = _quant_params(input_details)
    output_params = _quant_params(output_details)

    return {
        "input_scale": input_params["scale"],
        "input_zero_point": input_params["zero_point"],
        "input_dtype": input_params["dtype"],
        "input_shape": [int(s) for s in input_details["shape"]],
        "output_scale": output_params["scale"],
        "output_zero_point": output_params["zero_point"],
        "output_dtype": output_params["dtype"],
        "output_shape": [int(s) for s in output_details["shape"]],
    }


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------


def prune_qat_pipeline(
    model_path: Path | str,
    output_dir: Path | str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_sparsity: float = 0.30,
    fine_tune_epochs: int = 5,
    qat_epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    class_weight: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """Pipeline completo: pruning estruturado + fine-tune + QAT/PTQ + INT8.

    O QAT é opcional: se ``tensorflow_model_optimization`` não estiver
    instalado ou se rejeitar as camadas Keras por incompatibilidade de
    versão, o pipeline cai automaticamente no fallback de Post-Training
    Quantization (PTQ) full-integer INT8.

    Parameters
    ----------
    model_path : Path | str
        Caminho para o modelo ``.keras`` original.
    output_dir : Path | str
        Diretório onde salvar ``.tflite`` e ``.json``.
    X_train, y_train : np.ndarray
        Dados de treino.
    X_val, y_val : np.ndarray
        Dados de validação / representativos.
    target_sparsity : float
        Fração de filtros Conv1D a remover.
    fine_tune_epochs : int
        Épocas de fine-tuning após pruning.
    qat_epochs : int
        Épocas de fine-tuning durante QAT (somente se QAT for aplicado).
    batch_size : int
        Tamanho do batch.
    learning_rate : float
        Taxa de aprendizado.
    class_weight : dict[int, float], optional
        Pesos por classe.

    Returns
    -------
    dict[str, Any]
        Metadados do pipeline incluindo caminhos dos artefatos e a chave
        ``qat_applied`` indicando se o QAT foi aplicado.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carrega modelo original.
    model = tf.keras.models.load_model(str(model_path), compile=False)
    original_params = int(model.count_params())
    LOGGER.info("Modelo original carregado | params=%d", original_params)

    # 2. Pruning estruturado.
    pruned = apply_structured_pruning(model, target_sparsity=target_sparsity)
    pruned_params = int(pruned.count_params())
    LOGGER.info(
        "Modelo podado | params=%d | redução=%.1f%%",
        pruned_params,
        100 * (1 - pruned_params / original_params),
    )

    # 3. Fine-tuning pós-pruning.
    pruned, _ = fine_tune_pruned_model(
        pruned,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=fine_tune_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        class_weight=class_weight,
    )

    # 4. QAT (opcional; fallback para PTQ se tfmot estiver indisponível).
    qat_model, qat_applied = apply_qat(pruned)
    if qat_applied:
        qat_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate / 2.0),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        qat_model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=qat_epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            verbose=0,
        )
        LOGGER.info("QAT aplicado com sucesso.")
    else:
        LOGGER.warning("QAT não aplicado; prosseguindo com Post-Training Quantization (PTQ) INT8.")

    # 5. Conversão INT8 (funciona tanto para QAT quanto para PTQ).
    def representative_dataset():
        for sample in X_val:
            yield [np.expand_dims(sample, axis=0)]

    tflite_bytes = convert_to_tflite_int8(qat_model, representative_dataset)

    # 6. Salva artefatos.
    tflite_path = output_dir / "pruned_qat_int8.tflite"
    tflite_path.write_bytes(tflite_bytes)

    params = export_quantization_params(tflite_bytes)
    params["original_params"] = original_params
    params["pruned_params"] = pruned_params
    params["reduction_pct"] = round(100 * (1 - pruned_params / original_params), 2)

    params_path = output_dir / "quantization_params.json"
    with params_path.open("w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2, ensure_ascii=False)

    LOGGER.info("Artefatos salvos | tflite=%s | params=%s", tflite_path, params_path)

    return {
        "tflite_path": str(tflite_path),
        "params_path": str(params_path),
        "original_params": original_params,
        "pruned_params": pruned_params,
        "reduction_pct": params["reduction_pct"],
        "qat_applied": qat_applied,
    }
