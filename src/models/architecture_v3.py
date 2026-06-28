"""ECGClassifierV3 — arquitetura CNN + RNN de duplo estágio.

O modelo processa sinais de ECG com shape (500, 1) através de um
empilhamento convolucional 1D, seguido por camadas recorrentes
bidirecionais (LSTM ou GRU) e classificação densa. A saída possui
dimensão 2 para o estágio 1 (N vs Anormal) e 3 para o estágio 2
(S vs V vs F).
"""

from __future__ import annotations

from tensorflow import keras


def build_model(stage: int, config: dict) -> keras.Model:
    """Constrói uma instância do ECGClassifierV3.

    Parameters
    ----------
    stage:
        Etapa do classificador. ``1`` produz saída binária (N vs Anormal)
        e ``2`` produz saída ternária (S vs V vs F).
    config:
        Dicionário de hiperparâmetros. Chaves reconhecidas:
        ``cnn_filters``, ``cnn_kernels``, ``use_gru``, ``dense_units``,
        ``dropout``, ``learning_rate``, ``loss``.

    Returns
    -------
    keras.Model
        Modelo compilado do ECGClassifierV3.
    """
    inputs = keras.Input(shape=(500, 1), name="ecg_input")
    x = inputs
    for filters, kernel in zip(
        config.get("cnn_filters", [32, 64, 96]),
        config.get("cnn_kernels", [7, 5, 3]),
    ):
        x = keras.layers.Conv1D(filters, kernel, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling1D(2)(x)

    if config.get("use_gru"):
        x = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True))(x)
        x = keras.layers.Bidirectional(keras.layers.GRU(32))(x)
    else:
        x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(32))(x)

    for units, drop in zip(
        config.get("dense_units", [128, 64]),
        config.get("dropout", [0.5, 0.3]),
    ):
        x = keras.layers.Dense(units, activation="relu")(x)
        x = keras.layers.Dropout(drop)(x)
        x = keras.layers.BatchNormalization()(x)

    outputs = keras.layers.Dense(
        2 if stage == 1 else 3,
        activation="softmax",
        name=f"stage{stage}_output",
    )(x)
    model = keras.Model(inputs, outputs, name=f"ECGClassifierV3_stage{stage}")
    model.compile(
        optimizer=keras.optimizers.Adam(config.get("learning_rate", 1e-3)),
        loss=config.get("loss", "categorical_crossentropy"),
        metrics=[
            keras.metrics.F1Score(average="macro", name="f1_macro"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model
