"""Modelos multitarefa v3 (docs/rebuild_spec/04).

Família (d) da matriz 4×5×5:

- ``build_multitask_beat_model``: tronco CNN compartilhado (mesma topologia da
  backbone v3) + cabeça de batimento (softmax N/Anormal) + cabeça de qualidade
  (sigmoide multi-rótulo: qf_flatline, qf_clip, qf_off_center).
- ``build_rhythm_model``: CNN pequena para episódios de 10 s (5000 amostras)
  com saída softmax de ritmo (SINUS/AFIB/AFL/JUNCTIONAL) — tarefa de nível 3,
  escopo de episódio (D3). Janelas de batimento isolado NUNCA recebem saída de
  ritmo (INSUFFICIENT_TEMPORAL_CONTEXT).
"""

from __future__ import annotations

import logging

import tensorflow as tf

LOGGER = logging.getLogger("lewis.camada04.multitask_v3")

QUALITY_HEAD_NAMES = ["qf_flatline", "qf_clip", "qf_off_center"]
RHYTHM_CLASSES = ["SINUS", "AFIB", "AFL", "JUNCTIONAL"]


def weighted_sparse_ce(class_weight: dict[int, float]) -> tf.keras.losses.Loss:
    """Sparse CE com pesos por classe embutidos na loss.

    Contorna a recusa do Keras 3 de ``class_weight``/``sample_weight`` em
    modelos multi-saída (usado pela família (d) da matriz v3).
    """
    classes = sorted(class_weight)
    w_table = tf.constant([class_weight[c] for c in classes], dtype=tf.float32)
    base = tf.keras.losses.SparseCategoricalCrossentropy(reduction="none")

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_int = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        w = tf.gather(w_table, y_int)
        return base(y_int, y_pred) * w

    return loss


def _cnn_trunk(
    x: tf.keras.Tensor,
    conv_filters: tuple[int, ...],
    conv_kernels: tuple[int, ...],
    prefix: str,
) -> tf.keras.Tensor:
    for idx, (filters, kernel_size) in enumerate(zip(conv_filters, conv_kernels), start=1):
        x = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            name=f"{prefix}_conv1d_{idx}",
        )(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, name=f"{prefix}_maxpool_{idx}")(x)
    return tf.keras.layers.GlobalAveragePooling1D(name=f"{prefix}_gap")(x)


def build_multitask_beat_model(
    input_len: int = 500,
    embedding_dim: int = 64,
    dropout_rate: float = 0.3,
    conv_filters: tuple[int, int, int] = (16, 32, 64),
    conv_kernels: tuple[int, int, int] = (7, 5, 3),
    quality_weight: float = 0.25,
    name: str = "lewis_multitask_beat_v3",
) -> tf.keras.Model:
    """Cabeça de batimento (2 classes) + cabeça de qualidade (3 flags).

    Parameters
    ----------
    quality_weight : float
        Peso fixo a priori da perda de qualidade (λ_q=0.25, EXPERIMENTAL —
        ver docs/rebuild_spec/04 §2; não é ajustado no outer test).
    """
    signal_input = tf.keras.Input(shape=(input_len, 1), name="input")
    x = _cnn_trunk(signal_input, conv_filters, conv_kernels, prefix="beat")
    x = tf.keras.layers.Dense(embedding_dim, activation="relu", name="embedding")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)

    beat_output = tf.keras.layers.Dense(2, activation="softmax", name="beat")(x)
    quality_output = tf.keras.layers.Dense(
        len(QUALITY_HEAD_NAMES), activation="sigmoid", name="quality"
    )(x)

    model = tf.keras.Model(inputs=signal_input, outputs=[beat_output, quality_output], name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={"beat": "sparse_categorical_crossentropy", "quality": "binary_crossentropy"},
        loss_weights={"beat": 1.0, "quality": quality_weight},
        metrics={"beat": "accuracy"},
    )
    # Metadado de contrato (Keras 3 não expõe loss_weights após o compile)
    model.quality_weight = quality_weight  # type: ignore[attr-defined]
    LOGGER.info(
        "MultitaskBeat v3 | params=%d | quality_weight=%.3f",
        model.count_params(),
        quality_weight,
    )
    return model


def build_rhythm_model(
    input_len: int = 5000,
    embedding_dim: int = 64,
    dropout_rate: float = 0.3,
    conv_filters: tuple[int, ...] = (16, 32, 64, 64),
    conv_kernels: tuple[int, ...] = (9, 7, 5, 3),
    name: str = "lewis_rhythm_v3",
) -> tf.keras.Model:
    """CNN de ritmo para episódios (5000 amostras @ 500 Hz = 10 s)."""
    episode_input = tf.keras.Input(shape=(input_len, 1), name="episode")
    x = _cnn_trunk(episode_input, conv_filters, conv_kernels, prefix="rhythm")
    x = tf.keras.layers.Dense(embedding_dim, activation="relu", name="rhythm_embedding")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="rhythm_dropout")(x)
    rhythm_output = tf.keras.layers.Dense(
        len(RHYTHM_CLASSES), activation="softmax", name="rhythm"
    )(x)

    model = tf.keras.Model(inputs=episode_input, outputs=rhythm_output, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    LOGGER.info("Rhythm v3 | params=%d | classes=%s", model.count_params(), RHYTHM_CLASSES)
    return model
