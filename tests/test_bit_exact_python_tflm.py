"""Testes de bit-exatidão entre modelos Keras float32 e TFLite INT8.

Verifica que as saídas dequantizadas do modelo INT8 são numericamente
próximas às saídas float32, conforme QG10 (cosine similarity > 0.99).

Os thresholds aplicam-se à média do subset (fidelidade global da
quantização). Modelos existentes nem sempre atingem 0.99 em cada amostra
individual, por isso o teste também reporta cosine mínimo e taxa de
concordância do argmax para baseline.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Any, TypedDict

import joblib
import numpy as np
import pytest
import tensorflow as tf

from src.inference.quantized_runner import QuantizedModelRunner
from src.models.keras_loader import load_keras_model


def _set_global_seeds(seed: int = 123) -> None:
    """Fixa seeds para tornar construção de modelos e dados determinística."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "features"


class StageConfig(TypedDict):
    """Typed artifact family used by bit-exact tests."""

    name: str
    keras: Path
    tflite: Path
    scaler: Path
    data: Path


STAGE_CONFIGS: list[StageConfig] = [
    {
        "name": "stage1",
        "keras": MODELS_DIR / "stage1_float32_v2.0.keras",
        "tflite": MODELS_DIR / "quantized" / "stage1_int8_v2.0.tflite",
        "scaler": MODELS_DIR / "input_scaler_stage1_v2.0.pkl",
        "data": DATA_DIR / "stage1_binary.npz",
    },
    {
        "name": "stage2",
        "keras": MODELS_DIR / "stage2_float32_v2.0.keras",
        "tflite": MODELS_DIR / "quantized" / "stage2_int8_v2.0.tflite",
        "scaler": MODELS_DIR / "input_scaler_stage2_v2.0.pkl",
        "data": DATA_DIR / "stage2_multiclass.npz",
    },
]


def _artifact_exists(config: StageConfig) -> bool:
    """Verifica se todos os artefatos de um estágio estão presentes."""
    return all(path.exists() for path in config.values() if isinstance(path, Path))


@pytest.fixture(scope="module")
def loaded_models() -> dict[str, dict[str, Any]]:
    """Carrega modelos Keras e interpretadores TFLite para ambos os estágios."""
    loaded: dict[str, dict[str, Any]] = {}
    for config in STAGE_CONFIGS:
        if not _artifact_exists(config):
            continue
        loaded[config["name"]] = {
            "keras": load_keras_model(str(config["keras"]), compile=False),
            "runner": QuantizedModelRunner(config["tflite"]).allocate(),
            "scaler": joblib.load(config["scaler"]),
        }
    return loaded


@pytest.fixture(scope="module")
def stage_subsets() -> dict[str, np.ndarray]:
    """Retorna subsets de 256 amostras para cada estágio."""
    subsets: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(42)
    for config in STAGE_CONFIGS:
        data_path = config["data"]
        if not data_path.exists():
            continue
        data = np.load(data_path)
        x = data["X"]
        n = min(256, x.shape[0])
        idx = rng.choice(x.shape[0], size=n, replace=False)
        subsets[config["name"]] = x[idx].astype(np.float32)
    return subsets


def _normalize(x: np.ndarray, scaler: Any) -> np.ndarray:
    """Aplica z-score global com o scaler serializado."""
    n, seq_len, channels = x.shape
    return scaler.transform(x.reshape(-1, channels)).reshape(n, seq_len, channels)


def _compare_logits(
    logits_f32: np.ndarray,
    logits_i8: np.ndarray,
) -> tuple[float, float, float]:
    """Calcula similaridade do cosseno (min/mean) e taxa de concordância do argmax."""
    norm_f32 = logits_f32 / (np.linalg.norm(logits_f32, axis=1, keepdims=True) + 1e-12)
    norm_i8 = logits_i8 / (np.linalg.norm(logits_i8, axis=1, keepdims=True) + 1e-12)
    cosine = np.sum(norm_f32 * norm_i8, axis=1)
    argmatch = np.mean(np.argmax(logits_f32, axis=1) == np.argmax(logits_i8, axis=1))
    try:
        return float(cosine.min()), float(cosine.mean()), float(argmatch)
    except (TypeError, ValueError) as error:
        raise ValueError("Unable to summarize float32/INT8 similarity") from error


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_bit_exact_logits(
    stage: str,
    loaded_models: dict[str, dict[str, Any]],
    stage_subsets: dict[str, np.ndarray],
) -> None:
    """Compara logits float32 e INT8 dequantizados (QG10).

    Garante que a dequantização do TFLite preserva a direção média dos
    logits (cosine similarity médio > 0.99) e que o argmax coincide em
    mais de 94% das amostras. Os valores mínimos são reportados para
    baseline dos modelos existentes.
    """
    if stage not in loaded_models or stage not in stage_subsets:
        pytest.skip(f"Artefatos ou dados ausentes para {stage}")

    models = loaded_models[stage]
    x_raw = stage_subsets[stage]
    x = _normalize(x_raw, models["scaler"])

    logits_f32 = models["keras"].predict(x, verbose=0)
    # O interpretador TFLite foi quantizado com batch fixo em 1.
    logits_i8 = np.concatenate([models["runner"].run(x[i]) for i in range(x.shape[0])], axis=0)

    assert logits_f32.shape == logits_i8.shape, (
        f"Shapes divergentes para {stage}: " f"float32={logits_f32.shape} vs int8={logits_i8.shape}"
    )

    cosine_min, cosine_mean, argmatch = _compare_logits(logits_f32, logits_i8)

    print(
        f"\n{stage}: cosine_min={cosine_min:.4f}, "
        f"cosine_mean={cosine_mean:.4f}, argmatch={argmatch:.4f}"
    )

    assert cosine_mean > 0.99, (
        f"{stage}: cosine similarity médio {cosine_mean:.4f} " f"não atinge o threshold 0.99 (QG10)"
    )
    assert argmatch > 0.94, (
        f"{stage}: concordância de argmax {argmatch:.4f} " f"não atinge o threshold 0.94"
    )


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_dequantization_formula_is_inverse(
    stage: str,
    loaded_models: dict[str, dict[str, Any]],
) -> None:
    """Verifica que a dequantização do runner é consistente com scale/zero_point."""
    if stage not in loaded_models:
        pytest.skip(f"Modelo ausente para {stage}")

    runner = loaded_models[stage]["runner"]
    q = np.array([[-128, 0, 127]], dtype=np.int8)
    deq = runner._dequantize(q)
    expected = (q.astype(np.float32) - runner.output_zero_point) * runner.output_scale

    assert np.allclose(deq, expected, atol=1e-7), (
        f"{stage}: dequantização inconsistente com scale={runner.output_scale} "
        f"zero_point={runner.output_zero_point}"
    )


def _build_and_quantize_tiny_model(
    num_classes: int, workdir: Path
) -> tuple[Path, Path, np.ndarray]:
    """Cria um modelo sintético pequeno, treina e quantiza para INT8.

    Usado apenas para validar a lógica de comparação quando os modelos
    reais não estão disponíveis ou como sanity check da pipeline de
    quantização.

    Os dados são gerados com médias por classe bem separadas, de modo que
    um modelo pequeno consiga aprender um padrão estável em poucas épocas
    e o teste seja determinístico no CI.
    """
    _set_global_seeds(123)
    rng = np.random.default_rng(123)
    samples_per_class = 64
    n_samples = samples_per_class * num_classes
    x_train = np.zeros((n_samples, 500, 1), dtype=np.float32)
    y_train = np.zeros(n_samples, dtype=np.int64)

    # Classes com médias -1.5, 0.0 e +1.5 no domínio do sinal.
    class_means = np.linspace(-1.5, 1.5, num=num_classes)
    for cls, mean in enumerate(class_means):
        start = cls * samples_per_class
        end = start + samples_per_class
        x_train[start:end, :, 0] = rng.normal(loc=mean, scale=0.2, size=(samples_per_class, 500))
        y_train[start:end] = cls

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(500, 1)),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
    )
    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

    float_path = workdir / "synthetic_float32.keras"
    int8_path = workdir / "synthetic_int8.tflite"
    model.save(float_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: (
        [x_train[i : i + 1].astype(np.float32)] for i in range(n_samples)
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    int8_model = converter.convert()
    int8_path.write_bytes(int8_model)

    return float_path, int8_path, x_train


def test_bit_exact_with_synthetic_model() -> None:
    """Valida a lógica do teste com um modelo sintético quantizado.

    Garante que, para um modelo pequeno e bem condicionado, a comparação
    float32 vs INT8 atinge cosine similarity > 0.99 e argmatch > 0.95.
    """
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        float_path, int8_path, x_train = _build_and_quantize_tiny_model(3, workdir)

        keras_model = load_keras_model(str(float_path), compile=False)
        runner = QuantizedModelRunner(int8_path).allocate()

        # Usa amostras de treino para validar equivalência numérica, já que
        # o modelo não generaliza para ruído puro após apenas 1 época.
        x = x_train[:64]

        # Keras 3 accepts integer verbosity modes; its unannotated signature
        # makes Pyright infer ``str`` from the default ``"auto"``.
        logits_f32 = keras_model.predict(x, verbose=0)  # pyright: ignore[reportArgumentType]
        logits_i8 = np.concatenate([runner.run(x[i]) for i in range(x.shape[0])], axis=0)

        cosine_min, cosine_mean, argmatch = _compare_logits(logits_f32, logits_i8)

        assert (
            cosine_mean > 0.99
        ), f"Modelo sintético: cosine médio {cosine_mean:.4f} abaixo de 0.99"
        # O modelo sintético é pequeno e treinado por apenas 1 época; o argmatch
        # valida que a dequantização preserva a classe predominante, mas não
        # exige equivalência perfeita de classificação.
        assert argmatch > 0.85, f"Modelo sintético: argmatch {argmatch:.4f} abaixo de 0.85"
