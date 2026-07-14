"""Treina MLP leve sobre features morfológicas/time-domain para Estágio 2.

Classificador S vs V vs F usando as mesmas 16 features do Estágio 1.
Otimizações aplicadas (v2.3.1):
  - Focal Loss dinâmica em vez de class_weight estático;
  - SMOTE no espaço tabular para minoritárias F e S (apenas no fold de treino);
  - CosineDecayRestarts para learning rate;
  - Thresholds otimizados por Youden's J no fold de validação.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import cast

import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.evaluate import evaluate_fold  # noqa: E402

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("train_stage2_mlp")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_output_dir(output_dir: str) -> Path:
    """Resolve output directory and ensure it stays inside PROJECT_ROOT."""
    target = Path(output_dir)
    if not target.is_absolute():
        target = PROJECT_ROOT / output_dir
    resolved = target.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Output directory escapes project root: {output_dir!r}") from exc
    return resolved


def build_mlp(
    input_dim: int,
    num_classes: int = 3,
    hidden_units: int = 32,
    dropout_rate: float = 0.3,
    hidden_units_2: int = 0,
    dropout_rate_2: float = 0.3,
) -> tf.keras.Model:
    """MLP leve para classificação multiclasse com features."""
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = tf.keras.layers.Dense(hidden_units, activation="relu", name="dense_1")(inputs)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    if hidden_units_2 > 0:
        x = tf.keras.layers.Dense(hidden_units_2, activation="relu", name="dense_2")(x)
        x = tf.keras.layers.Dropout(dropout_rate_2, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="stage2_mlp")
    return model


def smote_oversample(
    X: np.ndarray,
    y: np.ndarray,
    target_classes: list[int],
    target_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Aplica SMOTE apenas nas classes minoritárias especificadas.

    O oversampling é realizado ANTES do scaling, exclusivamente no fold de
    treino. A estratégia ``sampling_strategy`` eleva cada classe em
    ``target_classes`` até ``target_ratio`` da classe majoritária, sem
    inflá-la à igualdade total (evita overfitting na fronteira S/F).

    Parameters
    ----------
    X : np.ndarray
        Features de treino (não escaladas).
    y : np.ndarray
        Labels inteiros de treino.
    target_classes : list[int]
        Classes a serem reamostradas (ex.: [0, 2] para S e F).
    target_ratio : float
        Razão do tamanho da classe majoritária alvo para cada minoritária.
    seed : int
        Seed para reproducibilidade.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X e y reamostrados.
    """
    try:
        classes, counts = np.unique(y, return_counts=True)
        max_count = int(counts.max())
        target_count = min(int(max_count * target_ratio), max_count)

        strategy: dict[int, int] = {}
        for cls in target_classes:
            if cls in classes:
                current = int(counts[classes == cls][0])
                strategy[int(cls)] = max(current, target_count)

        if not strategy or all(strategy[cls] <= int(counts[classes == cls][0]) for cls in strategy):
            LOGGER.info("SMOTE: nenhuma classe precisa de oversampling")
            return X, y
    except Exception as exc:
        raise ValueError(f"Falha ao calcular estratégia SMOTE: {exc}") from exc

    LOGGER.info("SMOTE strategy: %s", strategy)
    smote = SMOTE(  # type: ignore[arg-type]
        sampling_strategy=strategy,  # type: ignore[arg-type]
        random_state=seed,
        k_neighbors=5,
    )
    X_res, y_res = smote.fit_resample(X, y)  # type: ignore[assignment]
    return X_res, y_res  # type: ignore[return-value]


def focal_loss(alpha: list[float], gamma: float) -> tf.keras.losses.Loss:
    """Retorna CategoricalFocalCrossentropy do Keras com alpha e gamma.

    A função de perda focal atenua a contribuição de exemplos já bem
    classificados e amplifica o gradiente dos exemplos de fronteira. Para
    one-hot ``y`` e probabilidade ``p`` da classe verdadeira, o termo por
    amostra é:

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    onde ``p_t = p`` se a amostra pertence à classe e ``p_t = 1-p`` no caso
    binário; no cenário multiclasse com softmax, ``p_t`` é simplesmente a
    probabilidade atribuída à classe verdadeira. O fator ``(1-p_t)^gamma``
    reduz a perda para exemplos com alta confiança (p_t → 1) e a mantém
    próxima de ``-log(p_t)`` para exemplos difíceis (p_t → 0).

    Escolha dos hiperparâmetros para Project-Lewis v2.3 (S, V, F):

    - alpha = [0.60, 0.40, 1.00]:
      * ``alpha`` atua como ponderação da classe verdadeira. Não modifica a
        geometria da fronteira de decisão, apenas o módulo do gradiente
        propagado por classe.
      * V é a classe majoritária e apresenta separabilidade razoável no
        espaço morfológico; alpha_V=0.40 evita que o gradiente de V domine
        a atualização e empurre os centros de S/F.
      * S e F compartilham fronteira no espaço de features (ambas dependem de
        morfologia QRS/T e ritmo); alpha_S=0.60 e alpha_F=1.00 dão ênfase
        crescente às minoritárias sem recorrer a pesos estáticos massivos do
        tipo ``class_weight`` (que distorcem a estimativa do gradiente total).
    - gamma = 2.0:
      * Com gamma=2, um exemplo com p_t=0.9 tem perda reduzida pelo fator
        (0.1)^2 = 0.01, enquanto um exemplo com p_t=0.5 tem fator 0.25.
        Isso força o otimizador a concentrar-se nos exemplos de fronteira
        S/F/V sem alterar o valor esperado da perda quando o modelo está
        calibrado.
    """

    return tf.keras.losses.CategoricalFocalCrossentropy(
        alpha=alpha,
        gamma=gamma,
        from_logits=False,
        label_smoothing=0.0,
    )


class F1MacroCallback(tf.keras.callbacks.Callback):
    """Computa F1-macro no conjunto de validação ao final de cada época.

    Permite monitorar a métrica de negócio diretamente no EarlyStopping,
    evitando parada prematura quando val_loss estagna mas a fronteira S/F
    ainda está evoluindo.
    """

    def __init__(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:  # type: ignore[override]
        try:
            logs = logs or {}
            y_pred = np.argmax(self.model.predict(self.X_val, verbose=0), axis=1)
            f1_macro = f1_score(
                self.y_val, y_pred, average="macro", zero_division=0  # type: ignore[arg-type]
            )
            logs["val_f1_macro"] = float(f1_macro)
        except Exception as exc:
            LOGGER.warning("F1MacroCallback falhou no epoch %d: %s", epoch, exc)


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fold_idx: int,
    output_dir: Path,
    scaler=None,
    hidden_units: int = 32,
    dropout_rate: float = 0.3,
    hidden_units_2: int = 0,
    dropout_rate_2: float = 0.3,
    focal_alpha: list[float] | None = None,
    focal_gamma: float = 2.0,
    no_lr_schedule: bool = False,
    patience: int = 15,
    class_weight: dict[int, float] | None = None,
    optimize_thresholds: bool = False,
    threshold_metric: str = "F1_macro",
) -> dict:
    """Treina um único fold."""
    if focal_alpha is None:
        focal_alpha = [0.60, 0.40, 1.00]  # S, V, F

    num_classes = 3
    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)

    model = build_mlp(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
        hidden_units_2=hidden_units_2,
        dropout_rate_2=dropout_rate_2,
    )

    # Cosine annealing com restarts para escapar de mínimos locais na
    # fronteira S/F. A curva segue:
    #
    #   lr(t) = alpha * lr_0 + 0.5 * lr_0 * (1-alpha) *
    #           (cos(pi * t_mod / T) + 1) * m_mul^i
    #
    # onde T é first_decay_steps, t_mod é a posição dentro do ciclo atual e
    # i é o índice do restart. Configuração solicitada pela especificação:
    # - first_decay_steps=10 (épocas): ciclo curto o suficiente para permitir
    #   múltiplos reinícios dentro do EarlyStopping (patience=15);
    # - t_mul=1.0: comprimento do ciclo não cresce entre restarts;
    # - m_mul=0.9: amplitude do LR decai 10% a cada restart, suavizando a
    #   exploração à medida que o modelo se aproxima de um vale;
    # - alpha=0.1: LR mínimo = 1e-4, mantendo gradiente vivo no vale do cosseno.
    # Pode ser desabilitado via --no-lr-schedule para diagnóstico.
    initial_lr = 1e-3
    if no_lr_schedule:
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    else:
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr,
            first_decay_steps=10,
            t_mul=1.0,
            m_mul=0.9,
            alpha=0.1,
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss=focal_loss(alpha=focal_alpha, gamma=focal_gamma),
        metrics=["accuracy"],
    )

    callbacks = [
        F1MacroCallback(X_val=X_val, y_val=y_val),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_f1_macro",
            patience=patience,
            mode="max",
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=100,
        batch_size=128,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )

    if optimize_thresholds:
        eval_result = evaluate_fold(
            model,
            X_val,
            y_val,
            class_names=["S", "V", "F"],
            optimize_thresholds=True,
        )
    else:
        eval_result = evaluate_fold(
            model,
            X_val,
            y_val,
            class_names=["S", "V", "F"],
            optimize_youden=True,
        )

    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(fold_dir / "model.keras"))
    if scaler is not None:
        import joblib

        joblib.dump(scaler, fold_dir / "input_scaler.pkl")

    # Salva thresholds otimizados por Youden para este fold
    youden_thresholds = eval_result.get("thresholds", {"S": 0.5, "V": 0.5, "F": 0.5})
    youden_j = eval_result.get("best_j_per_class", {})
    thresholds_path = fold_dir / "stage2_thresholds.json"
    thresholds_path.write_text(
        json.dumps(
            {
                "thresholds": youden_thresholds,
                "best_j_per_class": youden_j,
                "source": "youden_j_validation",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    LOGGER.info(
        "Fold %d | Thresholds Youden: %s | J per class: %s",
        fold_idx,
        youden_thresholds,
        youden_j,
    )

    LOGGER.info(
        "Fold %d | F1_macro=%.4f | Acc=%.4f | per_class F1=%s | epochs=%d",
        fold_idx,
        eval_result["global"]["F1_macro"],
        eval_result["global"]["Acc"],
        {k: v["F1"] for k, v in eval_result["per_class"].items()},
        len(history.history["loss"]),
    )

    return {
        "fold": fold_idx,
        "eval_result": eval_result,
        "epochs_trained": len(history.history["loss"]),
        "thresholds": youden_thresholds,
        "best_j_per_class": youden_j,
    }


def main() -> int:
    # Limpa scalers v2.3 antigos para garantir que nenhum scaler de 13 features
    # seja carregado por engano durante a publicação de artefatos.
    for stale_scaler in (
        PROJECT_ROOT / "models" / "input_scaler_stage1_v2.3.pkl",
        PROJECT_ROOT / "models" / "input_scaler_stage2_v2.3.pkl",
    ):
        if stale_scaler.exists():
            stale_scaler.unlink()
            LOGGER.info("Removido scaler antigo: %s", stale_scaler)

    parser = argparse.ArgumentParser(description="Treina Estágio 2 MLP sobre features.")
    parser.add_argument(
        "--smote-target-classes",
        type=int,
        nargs="+",
        default=[0, 2],
        help="Classes a serem reamostradas por SMOTE (S=0, V=1, F=2).",
    )
    parser.add_argument(
        "--smote-target-ratio",
        type=float,
        default=1.0,
        help="Razão do tamanho da classe majoritária alvo para SMOTE. 1.0 iguala à majoritária.",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        nargs="+",
        default=[0.60, 0.40, 1.00],
        help="Pesos alpha da Focal Loss por classe (S, V, F).",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Fator gamma da Focal Loss.",
    )
    parser.add_argument(
        "--no-lr-schedule",
        action="store_true",
        help="Desabilita CosineDecayRestarts e usa Adam com LR fixo (diagnóstico).",
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        default=32,
        help="Número de unidades na camada oculta.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Taxa de dropout após a camada oculta.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Paciência do EarlyStopping em épocas.",
    )
    parser.add_argument(
        "--class-weight",
        type=float,
        nargs="+",
        default=None,
        help="Pesos de classe para Keras (S, V, F). Se omitido, usa Focal Loss sem class_weight.",
    )
    parser.add_argument(
        "--optimize-thresholds",
        action="store_true",
        help=(
            "Otimiza thresholds one-vs-rest para maximizar F1-macro "
            "(ou --threshold-metric) em vez de Youden."
        ),
    )
    parser.add_argument(
        "--threshold-metric",
        type=str,
        default="F1_macro",
        help="Métrica a maximizar na busca de thresholds (quando --optimize-thresholds).",
    )
    parser.add_argument(
        "--hidden-units-2",
        type=int,
        default=0,
        help="Número de unidades na segunda camada oculta (0 = desabilitada).",
    )
    parser.add_argument(
        "--dropout-2",
        type=float,
        default=0.3,
        help="Taxa de dropout após a segunda camada oculta.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/stage2_mlp_features_v2.3",
        help="Directory to save fold models and summary.",
    )
    args = parser.parse_args()

    class_weight_dict: dict[int, float] | None = None
    if args.class_weight is not None:
        if len(args.class_weight) != 3:
            raise ValueError("--class-weight deve conter exatamente 3 valores (S, V, F)")
        class_weight_dict = {
            0: args.class_weight[0],
            1: args.class_weight[1],
            2: args.class_weight[2],
        }

    npz = np.load("data/features/stage2_multiclass_features.npz")
    X = cast(np.ndarray, np.asarray(npz["X"], dtype=np.float32))
    y = cast(np.ndarray, np.asarray(npz["y"], dtype=np.int64))
    groups = cast(np.ndarray, np.asarray(npz["groups"]))
    try:
        feature_names = json.loads(
            Path("data/features/stage2_multiclass_features.json").read_text(encoding="utf-8")
        )["feature_names"]
    except Exception as exc:
        LOGGER.error("Falha ao carregar feature names: %s", exc)
        raise

    LOGGER.info("Dataset: X=%s, y=%s", X.shape, y.shape)
    LOGGER.info("Features: %s", feature_names)
    LOGGER.info(
        "Optimization config: smote_target_classes=%s, smote_target_ratio=%s, "
        "focal_alpha=%s, focal_gamma=%s, hidden_units=%s, dropout=%s, patience=%s",
        args.smote_target_classes,
        args.smote_target_ratio,
        args.focal_alpha,
        args.focal_gamma,
        args.hidden_units,
        args.dropout,
        args.patience,
    )

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        LOGGER.info("=== Fold %d/%d ===", fold_idx + 1, n_splits)
        X_train = cast(np.ndarray, np.asarray(X[train_idx], dtype=np.float32))
        X_val = cast(np.ndarray, np.asarray(X[val_idx], dtype=np.float32))
        y_train = cast(np.ndarray, np.asarray(y[train_idx], dtype=np.int64))
        y_val = cast(np.ndarray, np.asarray(y[val_idx], dtype=np.int64))

        # SMOTE sintético APENAS no treino, antes do scaling.
        X_train, y_train = smote_oversample(
            X_train,
            y_train,
            target_classes=args.smote_target_classes,
            target_ratio=args.smote_target_ratio,
            seed=42 + fold_idx,
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        result = train_fold(
            X_train=X_train,
            y_train=y_train,
            X_val=cast(np.ndarray, X_val),
            y_val=y_val,
            fold_idx=fold_idx,
            output_dir=output_dir,
            scaler=scaler,
            hidden_units=args.hidden_units,
            dropout_rate=args.dropout,
            hidden_units_2=args.hidden_units_2,
            dropout_rate_2=args.dropout_2,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            no_lr_schedule=args.no_lr_schedule,
            patience=args.patience,
            class_weight=class_weight_dict,
            optimize_thresholds=args.optimize_thresholds,
            threshold_metric=args.threshold_metric,
        )
        fold_results.append(result)

    try:
        # Agregação
        f1_macros = [r["eval_result"]["global"]["F1_macro"] for r in fold_results]
        accs = [r["eval_result"]["global"]["Acc"] for r in fold_results]
        per_class_f1 = {
            cls: [r["eval_result"]["per_class"][cls]["F1"] for r in fold_results]
            for cls in ("S", "V", "F")
        }

        # Agrega thresholds por mediana entre folds (robusto a outliers)
        aggregated_thresholds = {
            cls: float(np.median([r["thresholds"][cls] for r in fold_results]))
            for cls in ("S", "V", "F")
        }

        summary = {
            "experiment": "stage2_mlp_features_v2.3",
            "feature_names": feature_names,
            "hidden_units": args.hidden_units,
            "dropout_rate": args.dropout,
            "patience": args.patience,
            "focal_alpha": args.focal_alpha,
            "focal_gamma": args.focal_gamma,
            "smote_target_classes": args.smote_target_classes,
            "smote_target_ratio": args.smote_target_ratio,
            "hidden_units_2": args.hidden_units_2,
            "dropout_rate_2": args.dropout_2,
            "class_weight": args.class_weight,
            "optimize_thresholds": args.optimize_thresholds,
            "threshold_metric": args.threshold_metric,
            "folds": fold_results,
            "aggregated_thresholds": aggregated_thresholds,
            "mean": {
                "Acc": float(np.mean(accs)),
                "F1_macro": float(np.mean(f1_macros)),
                "F1_S": float(np.mean(per_class_f1["S"])),
                "F1_V": float(np.mean(per_class_f1["V"])),
                "F1_F": float(np.mean(per_class_f1["F"])),
            },
            "std": {
                "Acc": float(np.std(accs)),
                "F1_macro": float(np.std(f1_macros)),
                "F1_S": float(np.std(per_class_f1["S"])),
                "F1_V": float(np.std(per_class_f1["V"])),
                "F1_F": float(np.std(per_class_f1["F"])),
            },
        }

        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        (output_dir / "stage2_thresholds.json").write_text(
            json.dumps(
                {"thresholds": aggregated_thresholds, "source": "youden_median_across_folds"},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        LOGGER.info(
            "=== Resultado agregado === Acc=%.4f±%.4f | F1_macro=%.4f±%.4f | "
            "F1(S)=%.4f±%.4f | F1(V)=%.4f±%.4f | F1(F)=%.4f±%.4f",
            summary["mean"]["Acc"],
            summary["std"]["Acc"],
            summary["mean"]["F1_macro"],
            summary["std"]["F1_macro"],
            summary["mean"]["F1_S"],
            summary["std"]["F1_S"],
            summary["mean"]["F1_V"],
            summary["std"]["F1_V"],
            summary["mean"]["F1_F"],
            summary["std"]["F1_F"],
        )
        LOGGER.info("Thresholds agregados (mediana): %s", aggregated_thresholds)
    except Exception as exc:
        LOGGER.error("Falha durante agregação ou escrita de resultados: %s", exc)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
