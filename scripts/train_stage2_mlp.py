"""Treina MLP leve sobre features morfológicas/time-domain para Estágio 2.

Classificador S vs V vs F usando as mesmas 13 features do Estágio 1.
Usa SMOTE apenas no fold de treino, Focal Loss e CosineDecayRestarts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.threshold_decision import predict_with_thresholds  # noqa: E402
from src.models.evaluate import evaluate_aami  # noqa: E402

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("train_stage2_mlp")

# Focal Loss: FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
# α_t repondera o gradiente da classe verdadeira; γ comprime o peso dos exemplos fáceis.
# No Stage 2 (S=0, V=1, F=2), V é a classe majoritária, F é a mais rara e S é a fronteira
# crítica que está sendo destruída. Queremos:
#   - α_V < α_S < α_F  => gradiente relativo à classe V cresce para S e F
#   - manter α_t ∈ (0,1) para não distorcer a magnitude global do loss
#   - γ=2.0: fator padrão da Focal Loss; (1-p_t)^2 reduz o peso dos exemplos com p_t>0.5
#            em até 4×, forçando o modelo a aprender exemplos difíceis na fronteira S/F.
# A escolha α=[0.5, 0.25, 0.8] produz razões de gradiente:
#   ∂L/∂z_S  : ∂L/∂z_V  = 0.5/0.25 = 2.0
#   ∂L/∂z_F  : ∂L/∂z_V  = 0.8/0.25 = 3.2
# ou seja, exemplos de F recebem ~3× mais atenção que V e S ~2×, sem o colapso
# causado por pesos manuais de 15×.
FOCAL_ALPHA = np.array([0.5, 0.25, 0.8], dtype=np.float32)
FOCAL_GAMMA = 2.0

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


def build_mlp(input_dim: int, num_classes: int = 3, hidden_units: int = 32) -> tf.keras.Model:
    """MLP leve para classificação multiclasse com features."""
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = tf.keras.layers.Dense(hidden_units, activation="relu", name="dense_1")(inputs)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="output"
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="stage2_mlp")
    return model


def _build_smote_strategy(
    y_train: np.ndarray,
    minority_classes: tuple[int, ...],
    target_ratio: float,
) -> dict[int, int]:
    """Monta dict {classe: contagem_alvo} para SMOTE.

    O alvo é ``target_ratio`` vezes a contagem da classe majoritária. Classes que
    já possuem contagem maior ou igual ao alvo são omitidas do dict, fazendo o
    SMOTE mantê-las inalteradas.
    """
    classes, counts = np.unique(y_train, return_counts=True)
    count_map = {int(c): int(n) for c, n in zip(classes, counts)}
    majority_count = int(counts.max())
    target = max(1, int(majority_count * target_ratio))

    strategy: dict[int, int] = {}
    for cls in minority_classes:
        if cls in count_map and count_map[cls] < target:
            strategy[cls] = target
    return strategy


def _youden_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    n_thresholds: int = 100,
) -> dict[str, float]:
    """Otimiza thresholds one-vs-rest maximizando a estatística J de Youden.

    Para cada classe c:
        J(τ) = TPR_c(τ) - FPR_c(τ) = Sens_c(τ) - (1 - Spec_c(τ))
    O threshold τ_c* é escolhido como o que maximiza J(τ), buscando o ponto que
    melhor separa a classe c de todas as outras sem favorecer arbitrariamente
    precision ou recall.
    """
    thresholds: dict[str, float] = {}
    for i, name in enumerate(class_names):
        binary_true = (y_true == i).astype(np.int64)
        candidate_taus = np.linspace(0.0, 1.0, n_thresholds + 1)
        best_j = -np.inf
        best_tau = 0.5
        for tau in candidate_taus:
            pred_pos = (y_score[:, i] >= tau).astype(np.int64)
            tp = int(((pred_pos == 1) & (binary_true == 1)).sum())
            fp = int(((pred_pos == 1) & (binary_true == 0)).sum())
            fn = int(((pred_pos == 0) & (binary_true == 1)).sum())
            tn = int(((pred_pos == 0) & (binary_true == 0)).sum())
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            j = sens - (1.0 - spec)
            if j > best_j:
                best_j = j
                best_tau = float(tau)
        thresholds[name] = best_tau
    return thresholds


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fold_idx: int,
    output_dir: Path,
    scaler=None,
    hidden_units: int = 32,
) -> dict:
    """Treina um único fold com Focal Loss e CosineDecayRestarts."""
    n_classes = 3
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=n_classes)

    model = build_mlp(input_dim=X_train.shape[1], num_classes=n_classes, hidden_units=hidden_units)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=10,
        t_mul=1.0,
        m_mul=0.9,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(
            alpha=FOCAL_ALPHA,
            gamma=FOCAL_GAMMA,
            from_logits=False,
        ),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=50,
        batch_size=256,
        callbacks=callbacks,
        verbose=2,
    )

    y_proba = model.predict(X_val, batch_size=1024, verbose=0)

    # Otimização de threshold via Youden J sobre validação do fold
    optimized_thresholds = _youden_thresholds(y_val, y_proba, class_names=["S", "V", "F"])

    # Métricas de validação devem refletir o ponto de operação de produção
    # (thresholds otimizados), não o argmax padrão.
    y_pred_val = predict_with_thresholds(
        y_proba,
        optimized_thresholds,
        class_names=["S", "V", "F"],
        fallback_class=1,
    )
    eval_result = evaluate_aami(y_val, y_pred_val, class_names=["S", "V", "F"])
    eval_result["y_pred"] = y_pred_val.tolist()
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    threshold_path = fold_dir / "stage2_threshold.json"
    threshold_path.write_text(
        json.dumps({
            "thresholds": optimized_thresholds,
            "source": "youden_j_statistic",
            "fold": fold_idx,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    LOGGER.info(
        "Fold %d | Youden thresholds: %s",
        fold_idx,
        optimized_thresholds,
    )

    model.save(str(fold_dir / "model.keras"), save_format="keras")
    if scaler is not None:
        import joblib
        joblib.dump(scaler, fold_dir / "input_scaler.pkl")

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
        "y_val": y_val,
        "y_proba": y_proba,
        "optimized_thresholds": optimized_thresholds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Treina Estágio 2 MLP sobre features.")
    parser.add_argument(
        "--hidden-units",
        type=int,
        default=32,
        help="Número de unidades na camada oculta.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/stage2_mlp_features_v2.3",
        help="Directory to save fold models and summary.",
    )
    args = parser.parse_args()

    npz = np.load("data/features/stage2_multiclass_features.npz")
    X, y, groups = npz["X"], npz["y"], npz["groups"]
    feature_names = json.loads(
        Path("data/features/stage2_multiclass_features.json").read_text(encoding="utf-8")
    )["feature_names"]

    LOGGER.info("Dataset: X=%s, y=%s", X.shape, y.shape)
    LOGGER.info("Features: %s", feature_names)
    LOGGER.info(
        "Mitigation config: hidden_units=%s",
        args.hidden_units,
    )

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        LOGGER.info("=== Fold %d/%d ===", fold_idx + 1, n_splits)
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # SMOTE sintético APENAS no treino; validação/teste permanecem reais.
        smote_strategy = _build_smote_strategy(
            y_train,
            minority_classes=(0, 2),  # S e F
            target_ratio=0.5,         # até 50% da classe majoritária
        )
        if smote_strategy:
            LOGGER.info("SMOTE strategy: %s", smote_strategy)
            smote = SMOTE(sampling_strategy=smote_strategy, random_state=42 + fold_idx)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            LOGGER.info("After SMOTE: %s", dict(zip(*np.unique(y_train, return_counts=True))))
        else:
            LOGGER.info("SMOTE skipped: minority classes already above target")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        result = train_fold(
            X_train, y_train, X_val, y_val, fold_idx, output_dir,
            scaler=scaler,
            hidden_units=args.hidden_units,
        )
        fold_results.append(result)

    # Agregação
    f1_macros = [r["eval_result"]["global"]["F1_macro"] for r in fold_results]
    accs = [r["eval_result"]["global"]["Acc"] for r in fold_results]
    per_class_f1 = {
        cls: [r["eval_result"]["per_class"][cls]["F1"] for r in fold_results]
        for cls in ("S", "V", "F")
    }

    summary = {
        "experiment": "stage2_mlp_features_v2.3",
        "feature_names": feature_names,
        "hidden_units": args.hidden_units,
        "focal_alpha": FOCAL_ALPHA.tolist(),
        "focal_gamma": FOCAL_GAMMA,
        "smote_target_ratio": 0.5,
        "optimized_thresholds_per_fold": {
            r["fold"]: r["optimized_thresholds"] for r in fold_results
        },
        "folds": fold_results,
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

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
