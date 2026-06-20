# Project-Lewis — Requirements.txt Detalhado
## Especificação de Dependências e Compatibilidade

**Versão:** 1.0 | **Data:** 2026-06-09 | **Arquiteto:** Douglas Souza

---

## 1. Visão Geral

Este documento detalha cada dependência do `requirements.txt` do Project-Lewis, justificando a versão escolhida, a função no pipeline, e as restrições de compatibilidade com o hardware de desenvolvimento (Lenovo IdeaPad 3 15ITL6, Zorin OS 18.1 / Ubuntu 24.04.3 LTS, Python 3.12.x).

---

## 2. Infraestrutura de Pacotes

| Pacote | Versão | Função | Por que esta versão |
| :--- | :--- | :--- | :--- |
| **setuptools** | 70.0.0 | Build backend para pacotes com extensões C (NumPy, SciPy, TensorFlow). | Ubuntu 24.04 shipa 68.1.2+. Versão 70 garante compatibilidade com PEP 660 (editable installs). |
| **wheel** | 0.43.0 | Formato de distribuição binária. Evita recompilação de C-extensions em cada install. | Última estável da série 0.43. Compatível com Python 3.12. |

**Nota:** Instalar primeiro com `pip install --upgrade pip setuptools wheel` antes do `requirements.txt`.

---

## 3. Leitura de Dados Biomédicos

### wfdb 4.1.2

**Função:** Leitura e download de registros de ECG no formato WFDB (Waveform Database) do PhysioNet.  
**Uso no projeto:** Download MIT-BIH Arrhythmia, SVDB, AFDB, INCART. Leitura de `.dat` (amostras), `.hea` (metadados), `.atr` (anotações AAMI).  
**APIs críticas:** `wfdb.io.dl_database()`, `wfdb.rdrecord()`, `wfdb.rdann()`.

**Por que 4.1.2:**
- Última versão da série 4.1 testada com Python 3.12.
- A versão 4.2.x (lançada em 2024-Q2) alterou a API de `dl_database()` (parâmetros de cache e retry), quebrando scripts legados.
- Não há benefício funcional da 4.2.x para o nosso uso (apenas melhorias de UI e logging).

**Risco de upgrade:** Quebra de `download_mitbih.py` se `dl_database()` mudar assinatura. **Bloqueado até validação manual.**

---

## 4. Computação Numérica

### NumPy 1.26.4

**Função:** Arrays N-dimensionais, operações vetorizadas, FFT, álgebra linear. Base de todo o pipeline.  
**Uso no projeto:** Manipulação de sinais ECG (arrays de milhões de amostras), FFT para validação de resample, operações de feature extraction.

**Por que 1.26.4 (e NÃO 2.0+):**
- **NumPy 2.0** (lançado em junho 2024) introduziu breaking changes na API C (ABI incompatível).
- **TensorFlow 2.16.1 não é compatível com NumPy 2.0**. Tentar importar TF com NumPy 2.0 resulta em `AttributeError` ou `ImportError`.
- NumPy 1.26.4 é a última release da série 1.x, com suporte de segurança ativo.

**Risco de upgrade:** Incompatibilidade total com TensorFlow. **Bloqueado até TF 2.17+ validado.**

### SciPy 1.11.4

**Função:** Filtros digitais, resample, detrend, processamento de sinais.  
**Uso no projeto:** `scipy.signal.butter()` (filtro Butterworth), `scipy.signal.filtfilt()` (zero-phase filtering), `scipy.signal.resample_poly()` (resample 360→500 Hz), `scipy.signal.find_peaks()` (AMPT), `scipy.signal.detrend()` (remoção de baseline).

**Por que 1.11.4:**
- Compatível com NumPy 1.26.x (testado pela equipe SciPy).
- Fornece `resample_poly` com filtro anti-aliasing FIR integrado (necessário para resample 360→500 Hz).
- Versão 1.12+ adicionou features de IA generativa (diffusion models) que não usamos, mas aumenta o tamanho do pacote.

---

## 5. Manipulação de Dados

### Pandas 2.1.4

**Função:** DataFrames para organização de features, anotações e metadados.  
**Uso no projeto:** `all_features.csv` (features temporais + morfológicas + labels AAMI), groupby por paciente para GroupKFold, merge de datasets.

**Por que 2.1.4:**
- Compatível com NumPy 1.26.x (Pandas 2.2+ requer NumPy 1.26.4+ mas pode ter regressões).
- Série 2.1 é LTS implícita (recebe patches de segurança).
- Pandas 2.2 introduziu `pyarrow` como backend padrão, que pode quebrar código legado sem `dtype_backend` explícito.

### PyArrow 14.0.1

**Função:** Backend de leitura/escrita Parquet (colunar, comprimido).  
**Uso no projeto:** Cache de datasets resampleados em formato Parquet (opcional, mas 10x mais rápido que CSV para leitura em massa).  
**Por que 14.0.1:** Última estável antes da série 15, que alterou a API de tipos de dados (Arrow DType system). Pandas 2.1.4 é compatível com PyArrow 14.x.

---

## 6. Visualização e EDA

### Matplotlib 3.8.2 + Seaborn 0.13.0

**Função:** Plotagem de sinais ECG, espectros de frequência, distribuições de features, matrizes de confusão.  
**Uso no projeto:** Apenas em notebooks EDA (`notebooks/01_eda_mitbih.ipynb`, etc.). **Não entram no pipeline de produção.**  
**Por que estas versões:** Compatíveis com NumPy 1.26.x. Matplotlib 3.9+ requer Python 3.10+ (OK) mas pode ter regressões de backend em Wayland (Zorin OS usa Wayland).

**Nota:** Se o simulador LVGL+SDL2 precisar de screenshots, Matplotlib não é usado — o SDL2 salva bitmaps nativamente.

---

## 7. Machine Learning — TensorFlow / Keras

### TensorFlow 2.16.1

**Função:** Framework de deep learning. Treinamento de 1D-CNN, conversão TFLite, quantização PTQ.  
**Uso no projeto:**
- `tf.keras.Sequential` (backbone 1D-CNN)
- `tf.data.Dataset` (generator para Chapman, 8 GB)
- `tf.lite.TFLiteConverter` (PTQ INT8)
- `tf.keras.callbacks` (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)

**Por que 2.16.1 (e NÃO 2.17+):**
- **Última versão estável com Keras 3 integrado e TFLite converter robusto.**
- TensorFlow 2.17 (lançado em julho 2024) migrou para Keras 3.3, que alterou a API de `TFLiteConverter` (representative dataset requer novo formato).
- TensorFlow 2.15.x usa Keras 2 (API diferente). Não queremos Keras 2.
- **TFLM (TensorFlow Lite Micro) é testado principalmente contra TF 2.14–2.16.** Usar 2.17+ pode gerar FlatBuffers incompatíveis com o runtime TFLM do STM32F4.

**Dependências transitivas críticas:**
- `h5py` (3.11.0): salvar/ler modelos `.h5`. Instalado automaticamente.
- `protobuf` (4.25.3): serialização de grafos TF. **Não atualizar manualmente** — TF 2.16.1 requer protobuf < 5.0.
- `wrapt` (1.16.0): decorators internos do TF.
- `typing-extensions` (4.12.2): tipos genéricos para compatibilidade.

**Risco de upgrade:** Incompatibilidade TFLM, quebra de TFLite converter, conflito protobuf. **Bloqueado até TFLM validar TF 2.17+.**

---

## 8. Machine Learning Auxiliar

### scikit-learn 1.4.0

**Função:** Validação cruzada, métricas, pré-processamento.  
**Uso no projeto:**
- `sklearn.model_selection.GroupKFold` (obrigatório — evita data leakage por paciente)
- `sklearn.metrics.classification_report`, `confusion_matrix`, `f1_score`, `recall_score`
- `sklearn.utils.class_weight.compute_class_weight` (desbalanceamento AAMI)

**Por que 1.4.0:** Compatível com NumPy 1.26.x e Pandas 2.1.x. Série 1.4 é estável e madura.

### imbalanced-learn 0.12.0

**Função:** Técnicas de oversampling para classes minoritárias.  
**Uso no projeto:** `imblearn.over_sampling.SMOTE` aplicado no espaço de features (não no sinal bruto) para classes V, S, F, Q do AAMI.  
**Por que 0.12.0:** Compatível com scikit-learn 1.4.x. SMOTE é a técnica padrão para oversampling de dados tabulares.

---

## 9. Download de Datasets

### kagglehub (latest)

**Função:** Download do dataset Chapman-Shaoxing via API oficial da Kaggle.  
**Uso no projeto:** `kagglehub.dataset_download()` para baixar os 45.152 registros de ECG (~8.2 GB).  
**Por que não versionado:** A API Kaggle evolui rapidamente. A versão `latest` garante acesso ao endpoint atual. O dataset baixado é validado por checksum (QG0), não pela versão da biblioteca.

**Pré-requisito:** Arquivo `~/.kaggle/kaggle.json` com credenciais da conta Kaggle (token API).  
**Alternativa:** Download manual via Figshare (wget) se Kaggle API falhar.

---

## 10. Testes e Qualidade

### pytest 8.0.0

**Função:** Framework de testes para os 7 Quality Gates (QG0–QG6).  
**Uso no projeto:** `tests/test_*.py` — validação de download, loader, resampler, AMPT, features, tamanho do modelo.  
**Por que 8.0.0:** Compatível com Python 3.12. Série 8.x é a atual estável. pytest 7.x ainda funciona, mas 8.x tem melhorias de performance em fixtures.

**Plugins não incluídos (opcionais):**
- `pytest-cov` (cobertura de código) — adicionar se necessário para TCC.
- `pytest-xdist` (execução paralela de testes) — adicionar se CI ficar lento (> 10 min).

---

## 11. Utilitários

### joblib 1.3.2

**Função:** Paralelismo e cache de objetos Python.  
**Uso no projeto:** `joblib.Parallel` + `joblib.delayed` para processamento paralelo de features em múltiplos registros MIT-BIH+. `joblib.Memory` para cache de datasets resampleados (evita reprocessar).  
**Por que 1.3.2:** Compatível com Python 3.12. Série 1.3 é estável. joblib 1.4+ alterou a API de `Memory` (location parameter).

---

## 12. Matriz de Compatibilidade

| Pacote | Versão | Python | NumPy | TF | Ubuntu 24.04 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Python | 3.12.x | — | — | — | Nativo | ✅ |
| pip | 24.0+ | — | — | — | Nativo | ✅ |
| setuptools | 70.0.0 | 3.12 | — | — | Upgrade | ✅ |
| wheel | 0.43.0 | 3.12 | — | — | Upgrade | ✅ |
| wfdb | 4.1.2 | 3.12 | 1.26 | — | Install | ✅ |
| NumPy | 1.26.4 | 3.12 | — | 2.16 | Install | ✅ |
| SciPy | 1.11.4 | 3.12 | 1.26 | — | Install | ✅ |
| Pandas | 2.1.4 | 3.12 | 1.26 | — | Install | ✅ |
| PyArrow | 14.0.1 | 3.12 | — | — | Install | ✅ |
| Matplotlib | 3.8.2 | 3.12 | 1.26 | — | Install | ✅ |
| Seaborn | 0.13.0 | 3.12 | 1.26 | — | Install | ✅ |
| TensorFlow | 2.16.1 | 3.12 | 1.26 | — | Install | ✅ |
| scikit-learn | 1.4.0 | 3.12 | 1.26 | — | Install | ✅ |
| imbalanced-learn | 0.12.0 | 3.12 | 1.26 | — | Install | ✅ |
| kagglehub | latest | 3.12 | — | — | Install | ✅ |
| pytest | 8.0.0 | 3.12 | — | — | Install | ✅ |
| joblib | 1.3.2 | 3.12 | — | — | Install | ✅ |

---

## 13. Pacotes Rejeitados (Não Adicionar)

| Pacote | Motivo da Rejeição |
| :--- | :--- |
| **torch / pytorch** | Overkill para 1D-CNN simples. Não suporta TFLite nativo. Exportação para ONNX → TFLM é pipeline extra com risco de incompatibilidade. |
| **tensorflow-gpu** | Deprecated desde TF 2.11. TF 2.16 usa GPU automaticamente se CUDA disponível. IdeaPad 3 (Intel UHD) não tem CUDA — irrelevante. |
| **keras** (standalone) | Conflita com Keras 3 embutido no TF 2.16. `pip install keras` instala Keras 3.3 que pode sobrescrever o Keras do TF. |
| **neurokit2 / biosppy** | Abstraem detecção QRS, HRV, etc. Impedem validação cruzada com firmware C (AMPT deve ser implementado do zero). |
| **heartpy** | Mesmo problema — oculta a lógica de detecção de picos. |
| **tensorflow-model-optimization** | Fornece QAT (Quantization Aware Training). Overkill para TCC — PTQ é suficiente e mais simples. |
| **mlflow / wandb** | Experiment tracking. Não necessário para TCC offline. Adicionam dependências de rede e servidores externos. |
| **dask / ray** | Distributed computing. Overkill para 20GB RAM — Pandas + NumPy são suficientes. |
| **opencv-python** | Visão computacional. Não usamos imagens — sinais 1D. |
| **jupyter / ipykernel** | Opcional para notebooks EDA. Não incluído no requirements.txt principal para manter leveza. Instalar separadamente se necessário: `pip install jupyter`. |

---

## 14. Instruções de Instalação

```bash
# 1. Criar venv (OBRIGATÓRIO — nunca instalar no system Python)
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip + infraestrutura
pip install --upgrade pip setuptools wheel

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Congelar lockfile (versionar no git)
pip freeze > requirements.lock

# 5. Verificar instalação
python -c "import tensorflow; print('TF:', tensorflow.__version__)"
python -c "import wfdb; print('WFDB:', wfdb.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import scipy; print('SciPy:', scipy.__version__)"
python -m pytest --version
```

---

## 15. Troubleshooting

### Problema: `pip install tensorflow==2.16.1` falha com erro de compilação
**Causa:** pip tentando compilar TensorFlow do source (wheel não disponível para a plataforma).  
**Solução:** `pip install --only-binary :all: tensorflow==2.16.1` ou usar `pip install tensorflow-cpu==2.16.1` (versão CPU-only, menor).

### Problema: `ImportError: numpy.core.multiarray failed to import` no TF
**Causa:** NumPy 2.0 instalado (conflito com TF 2.16).  
**Solução:** `pip install numpy==1.26.4 --force-reinstall`.

### Problema: `kagglehub` falha com "403 Forbidden"
**Causa:** Credenciais Kaggle não configuradas ou token expirado.  
**Solução:** Criar `~/.kaggle/kaggle.json` com `{"username":"...","key":"..."}` do perfil Kaggle.

### Problema: `pytest` não encontra testes
**Causa:** `tests/__init__.py` ausente ou nome de arquivo não começa com `test_`.  
**Solução:** Verificar estrutura: `tests/test_*.py` + `tests/__init__.py`.

---

## 16. CI/CD (GitHub Actions)

```yaml
- name: Setup Python
  uses: actions/setup-python@v5
  with:
    python-version: '3.12'
    cache: 'pip'
    cache-dependency-path: |
      requirements.txt
      requirements.lock

- name: Install dependencies
  run: |
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
```

---

## 17. Referências

- Ubuntu 24.04 Python packages: https://packages.ubuntu.com/noble/python3
- TensorFlow 2.16 release notes: https://github.com/tensorflow/tensorflow/releases/tag/v2.16.1
- NumPy 1.26 release notes: https://numpy.org/doc/stable/release/1.26.4-notes.html
- WFDB Python: https://github.com/MIT-LCP/wfdb-python
- PyArrow compatibility: https://arrow.apache.org/docs/python/install.html
