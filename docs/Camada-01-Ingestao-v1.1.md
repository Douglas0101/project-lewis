# Project-Lewis — Camada 1: Ingestão e Aquisição de Dados
## Responsável: Engenharia de Dados

**Versão:** 1.1 | **Data:** 2026-06-09 | **Arquiteto:** Douglas Souza

---

## 1.1 Objetivo
Ingerir, validar e armazenar localmente todos os datasets necessários para o pipeline de ML do Project-Lewis, garantindo disponibilidade offline, integridade dos dados brutos, rastreabilidade de proveniência e conformidade com políticas de governança de dados aplicáveis (LGPD/GDPR).

---

## 1.2 Datasets

| Dataset | Registros | Tamanho (Zip) | Tamanho (Raw) | Fs | Lead | Formato | Função | Acesso |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Chapman-Shaoxing** | 45.152 | ~2.3 GB | ~5.1 GB | 500 Hz | 12-lead | WFDB (.mat/.hea) / CSV | Pré-treino backbone | PhysioNet / Figshare / Kaggle (público) |
| **MIT-BIH Arrhythmia** | 48 | ~73.5 MB | ~104.3 MB | 360 Hz | 2-lead (MLII + V1/V2/V4/V5) | WFDB (.dat/.hea/.atr) | Fine-tuning + teste | PhysioNet (público) |
| **MIT-BIH SVDB** | 78 | ~52.0 MB | ~75 MB | **250 Hz** | 2-lead (ECG1 + ECG2) | WFDB (.dat/.hea/.atr) | Fine-tuning | PhysioNet (público) |
| **MIT-BIH AFDB** | 25 (23 c/ sinais) | ~439.7 MB | ~605.9 MB | 250 Hz | 2-lead | WFDB (.dat/.hea/.atr/.qrs) | Fine-tuning | PhysioNet (público) |
| **INCART** | 75 | ~563.5 MB | ~794.5 MB | 257 Hz | 12-lead | WFDB (.dat/.hea/.atr) | Fine-tuning | PhysioNet (público) |

> **Nota de proveniência:** O Chapman-Shaoxing foi inicialmente publicado no Figshare (10.646 pacientes, ~1.04 GB) e posteriormente expandido para 45.152 registros no PhysioNet (v1.0.0, 2022). Os dados do MIT-BIH foram coletados entre 1975–1979 no Beth Israel Hospital (agora Beth Israel Deaconess Medical Center) e digitados a 360 Hz com 11-bit resolution. O INCART foi contribuído pelo St.-Petersburg Institute of Cardiological Technics, com 75 gravações extraídas de 32 Holters de 30 minutos cada.

### 1.2.1 Decisão: Contagem e IDs do SVDB

A contagem autoritativa do MIT-BIH SVDB é **78 registros**, obtida do arquivo
`RECORDS` da PhysioNet v1.0.0. Os identificadores **não formam um range contínuo**:

```
800-812 (13), 820-829 (10), 840-894 (55) → total 78
```

Portanto, o download não pode confiar em `range(800, 879)` (79 números, inclui
IDs inexistentes e omite IDs reais) nem em `range(800, 854)` (54 números,
subconta). A lista exata deve ser hardcoded conforme o `RECORDS` oficial.

**Decisão arquitetural:** manter `expected_count = 78` e atualizar
`RECORDS_SVDB` para a enumeração autoritativa acima.

---

## 1.3 Estratégia de Download Multi-Modal

A PhysioNet disponibiliza três mecanismos de acesso: (a) `wget` recursivo direto, (b) pacote ZIP consolidado, e (c) API Python `wfdb.io.dl_database`. Para o Project-Lewis, adota-se a seguinte ordem de preferência:

1. **ZIP consolidado via `wget`** — mais rápido para CI e mirror inicial (ex: `https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip`).
2. **`wfdb-python` (`dl_database` / `dl_files`)** — idempotente por registro, ideal para atualizações incrementais e validação de integridade via header.
3. **Mirror local (`tar.gz`)** — fallback obrigatório para ambientes offline e execuções em GitHub Actions sem acesso à internet.

### 1.3.1 Chapman-Shaoxing

```python
def download_chapman(raw_dir: Path, mirror_path: Path = None) -> None:
    """Download Chapman-Shaoxing via PhysioNet WFDB (preferencial), Kaggle API ou mirror.

    1. Tentar kagglehub.dataset_download('erarayamorenzomuten/chapmanshaoxing-12lead-ecg-database',
       output_dir=raw_dir) com credenciais Kaggle (kaggle.json ou env vars KAGGLE_USERNAME/KAGGLE_KEY)
    2. Fallback: wget/curl do ZIP PhysioNet direto
       (https://physionet.org/static/published-projects/challenge-2021/1.0.3/training/chapman_shaoxing/)
       ou do tarball WFDB_ChapmanShaoxing.tar.gz via pipeline API do PhysioNet Challenge
    3. Fallback: extrair mirror_path (tar.gz) se download falhar
    4. Validar: ~45k arquivos (.mat + .hea) ou CSVs
    5. Verificar SHA256SUMS.txt se disponível
    6. Salvar em raw_dir/ com estrutura preservada (subpastas g1, g2... se do Challenge)
    """
```

### 1.3.2 MIT-BIH Family (MITDB + SVDB + AFDB + INCART)

```python
RECORDS_MITBIH = ["100","101","102","103","104","105","106","107","108",
                  "109","111","112","113","114","115","116","117","118",
                  "119","121","122","123","124","200","201","202","203",
                  "205","207","208","209","210","212","213","214","215",
                  "217","219","220","221","222","223","228","230","231",
                  "232","233","234"]
RECORDS_SVDB = [
    # Lista autoritativa do PhysioNet svdb/1.0.0/RECORDS (78 registros).
    # NÃO é um range contínuo: faltam 813-819 e 830-839.
    f"{i:03d}" for i in (
        list(range(800, 813)) +   # 800-812  (13)
        list(range(820, 830)) +   # 820-829  (10)
        list(range(840, 895))     # 840-894  (55)
    )
]
RECORDS_AFDB = [
    "04015", "04043", "04048", "04126", "04746", "04908", "04936", "05091",
    "05121", "05261", "06426", "06453", "06995", "07162", "07859",
    "07879", "07910", "08215", "08219", "08378", "08405", "08434", "08455",
]
AFDB_ANNOTATIONS_ONLY = {"00735", "03665"}  # .hea + .atr/.qrs, sem .dat
RECORDS_INCART = [f"I{i:02d}" for i in range(1, 76)]  # I01-I75

DATASET_CONFIG = {
    "mitdb": {"records": RECORDS_MITBIH, "pn_dir": "mitdb/1.0.0", "zip_url": "https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip"},
    "svdb": {"records": RECORDS_SVDB, "pn_dir": "svdb/1.0.0", "zip_url": "https://physionet.org/static/published-projects/svdb/mit-bih-supraventricular-arrhythmia-database-1.0.0.zip"},
    "afdb": {"records": RECORDS_AFDB, "pn_dir": "afdb/1.0.0", "zip_url": "https://physionet.org/static/published-projects/afdb/mit-bih-atrial-fibrillation-database-1.0.0.zip"},
    "incartdb": {"records": RECORDS_INCART, "pn_dir": "incartdb/1.0.0", "zip_url": "https://physionet.org/static/published-projects/incartdb/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0.zip"},
}

def download_mitbih_family(raw_dir: Path, datasets: dict, mirror_path: Path = None) -> None:
    """Download MIT-BIH + SVDB + AFDB + INCART via wfdb ou wget ZIP.

    1. Para cada dataset em datasets:
       a. Verificar se ZIP já existe em raw_dir/.cache/; se sim, extrair e pular
       b. Tentar wfdb.io.dl_database(dataset_name, record, raw_dir) com retry exponencial
       c. Se falhar: wget/curl do ZIP consolidado da PhysioNet
       d. Se falhar e mirror_path existe: tar xzf mirror_path
    2. Validar: contagem de registros por dataset (ver tabela 1.2)
    3. Validar: cada registro tem .hea + .dat (exceto AFDB 00735 e 03665, que são .hea + .atr + .qrs)
    4. Validar: checksum SHA256 dos arquivos ZIP se disponível
    5. Extrair metadados dos .hea (Fs, n_leads, gain, age, sex, diagnosis) para catalog.json
    """
```

---

## 1.4 Estratégia de Mirror Local e Cache

```bash
# 1. Download inicial (preferir ZIPs para velocidade)
mkdir -p data/.cache/zips
wget -q -O data/.cache/zips/chapman.zip https://physionet.org/static/published-projects/challenge-2021/1.0.3/training/chapman_shaoxing.zip
wget -q -O data/.cache/zips/mitdb.zip https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip
wget -q -O data/.cache/zips/svdb.zip https://physionet.org/static/published-projects/svdb/mit-bih-supraventricular-arrhythmia-database-1.0.0.zip
wget -q -O data/.cache/zips/afdb.zip https://physionet.org/static/published-projects/afdb/mit-bih-atrial-fibrillation-database-1.0.0.zip
wget -q -O data/.cache/zips/incartdb.zip https://physionet.org/static/published-projects/incartdb/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0.zip

# 2. Criar mirror para backup/offline/CI
tar czf data/mirrors/chapman_mirror.tar.gz -C data/raw_chapman .
tar czf data/mirrors/mitbih_family_mirror.tar.gz -C data/raw_mitbih . -C ../data/raw_svdb . -C ../data/raw_afdb . -C ../data/raw_incart .

# 3. No CI ou máquina sem internet
tar xzf data/mirrors/chapman_mirror.tar.gz -C data/raw_chapman/
tar xzf data/mirrors/mitbih_family_mirror.tar.gz
```

**Regras:**
- `data/raw_*/` e `data/.cache/` entram no `.gitignore` — nunca versionar binários.
- `data/mirrors/` entra no `.gitignore` — tarballs são artifacts, não código.
- GitHub Actions cache: `actions/cache@v4` com key estratificada:
  ```yaml
  - uses: actions/cache@v4
    with:
      path: data/.cache/zips
      key: datasets-${{ hashFiles('src/data/checksums.json') }}
      restore-keys: datasets-
  ```

---

## 1.5 Verificação de Integridade (Checksums)

Todo mirror e artefato de download deve acompanhar um manifesto de checksums. A PhysioNet não publica SHA256 universal para todos os datasets, mas os ZIPs consolidados podem ser validados via tamanho esperado e, quando possível, hash local gerado no primeiro download bem-sucedido.

```python
# src/data/checksums.json (gerado automaticamente após primeiro download válido)
{
  "chapman_shaoxing.zip": {"sha256": "<hash>", "size_bytes": 2469600000, "source": "physionet"},
  "mitdb.zip": {"sha256": "<hash>", "size_bytes": 73500000, "source": "physionet"},
  "svdb.zip": {"sha256": "<hash>", "size_bytes": 52000000, "source": "physionet"},
  "afdb.zip": {"sha256": "<hash>", "size_bytes": 439700000, "source": "physionet"},
  "incartdb.zip": {"sha256": "<hash>", "size_bytes": 563500000, "source": "physionet"}
}
```

**Função de validação:**
```python
def verify_checksum(file_path: Path, expected_hash: str) -> bool:
    import hashlib
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_hash
```

---

## 1.6 Retry, Backoff e Dead Letter Queue

Downloads de datasets biomédicos são suscetíveis a instabilidade de rede e rate-limiting da PhysioNet. Implementar:

- **Retry exponencial:** 3 tentativas com backoff `2^attempt * 1s + jitter`.
- **Timeout por registro:** 30s para wfdb, 300s para ZIPs.
- **Dead Letter Queue (DLQ):** registros que falharem após 3 retries são logados em `data/.dlq/failed_downloads.json` com timestamp, erro e stack trace, permitindo reprocessamento seletivo sem rebaixar todo o pipeline.

```python
import time, random, json
from pathlib import Path

def download_with_retry(dl_func, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return dl_func()
        except Exception as e:
            if attempt == max_retries - 1:
                Path("data/.dlq").mkdir(parents=True, exist_ok=True)
                with open("data/.dlq/failed_downloads.json", "a") as f:
                    f.write(json.dumps({"func": dl_func.__name__, "error": str(e), "attempt": attempt}) + "\n")
                raise
            time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 1))
```

---

## 1.7 Proveniência, Governança e Compliance (LGPD)

Embora todos os datasets sejam públicos e de-identificados, a LGPD exige rastreabilidade de fonte e finalidade:

| Dataset | Base Legal | De-identificação | IRB/Waiver | Citação Obrigatória |
| :--- | :--- | :--- | :--- | :--- |
| Chapman-Shaoxing | Uso de pesquisa pública | Remoção de PII, IDs anonimizados | Aprovado pelo IRB de Shaoxing People’s Hospital e Ningbo First Hospital | Zheng et al., Sci Data 7, 48 (2020) |
| MIT-BIH Family | Uso de pesquisa pública (PhysioNet) | De-identificação clínica padrão dos anos 1975–1979 | N/A (dados históricos de-identificados) | Moody & Mark, Computers in Cardiology 1983; Goldberger et al., Circulation 2000 |
| INCART | Uso de pesquisa pública (PhysioNet) | IDs anonimizados, sem nomes | Contribuição do St.-Petersburg Institute of Cardiological Technics | Tihonenko & Khaustov |

**Requisitos de compliance:**
- Documentar `PROVENANCE.md` com DOI/URL de cada dataset.
- Nunca re-identificar pacientes cruzando múltiplas fontes.
- Restringir acesso ao mirror apenas ao time de ML autorizado (ACL no filesystem ou bucket S3/GCS).

---

## 1.8 Extração de Metadados e Catalogação

Antes de qualquer transformação (Camada 2), extrair metadados dos headers WFDB para catalogação e análise exploratória:

```python
def extract_metadata(record_path: Path) -> dict:
    """Extrai Fs, n_leads, gain, age, sex, diagnosis do .hea usando wfdb.rdheader()."""
    import wfdb
    rec = wfdb.rdheader(str(record_path.with_suffix("")))
    return {
        "record_name": rec.record_name,
        "fs": rec.fs,
        "n_sig": rec.n_sig,
        "sig_len": rec.sig_len,
        "duration_sec": rec.sig_len / rec.fs,
        "units": rec.units,
        "gains": rec.adc_gain,
        "comments": rec.comments,  # age, sex, diagnosis frequentemente aqui
    }
```

Os metadados são persistidos em `data/catalog/dataset_catalog.jsonl` (um JSON por linha), permitindo:
- Auditoria de cobertura demográfica (idade/sexo) antes do treinamento.
- Detecção de schema drift (ex: mudança de `fs` ou `n_sig` em nova versão do dataset).
- Filtragem de registros por critérios clínicos no pipeline de fine-tuning.

---

## 1.9 Quality Gate QG0

| Critério | Valor | Como Validar |
| :--- | :--- | :--- |
| Chapman registros | >= 45.000 | `len(list(raw_chapman.rglob("*.hea")))` ou `len(list(raw_chapman.rglob("*.csv")))` |
| MIT-BIH registros | 48 | `len(list(raw_mitbih.glob("*.hea"))) == 48` |
| SVDB registros | 78 | `len(list(raw_svdb.glob("*.hea"))) == 78` |
| AFDB registros | 25 | `len(list(raw_afdb.glob("*.hea"))) == 25` (23 com `.dat`, 2 com `.atr` apenas) |
| INCART registros | 75 | `len(list(raw_incart.glob("*.hea"))) == 75` |
| Integridade de arquivos | 100% | Cada registro: `.hea` obrigatório; `.dat` obrigatório (exceto AFDB 00735/03665); `.atr` obrigatório |
| Checksum de ZIPs | 100% | `verify_checksum()` contra `src/data/checksums.json` |
| Metadados extraídos | 100% | `data/catalog/dataset_catalog.jsonl` contém entrada para cada registro |
| DLQ vazia | 0 falhas | `data/.dlq/failed_downloads.json` não existe ou está vazio |

**Teste:** `pytest tests/test_download.py -v`

---

## 1.10 Referências Verificadas

- PhysioNet MIT-BIH Arrhythmia Database (v1.0.0): https://physionet.org/content/mitdb/ — 48 half-hour excerpts, 360 Hz, 11-bit, MLII + V1/V2/V4/V5. Zip: 73.5 MB; uncompressed: 104.3 MB.
- PhysioNet MIT-BIH Supraventricular Arrhythmia Database (v1.0.0): https://physionet.org/content/svdb/ — 78 half-hour recordings, **250 Hz**, 2 leads (ECG1, ECG2). Zip: ~52.0 MB.
- PhysioNet MIT-BIH Atrial Fibrillation Database (v1.0.0): https://physionet.org/content/afdb/ — 25 long-term recordings (23 with .dat signals; 00735 and 03665 annotations-only), 250 Hz, 10 hours each. Zip: 439.7 MB; uncompressed: 605.9 MB.
- PhysioNet St Petersburg INCART 12-lead Arrhythmia Database (v1.0.0): https://physionet.org/content/incartdb/ — 75 annotated 30-minute recordings extracted from 32 Holters, 12 leads at 257 Hz. Zip: 563.5 MB; uncompressed: 794.5 MB.
- PhysioNet Chapman-Shaoxing (via Challenge 2021): https://physionet.org/content/challenge-2021/ — 45,152 ECGs, 10 seconds, 500 Hz, 12 leads. WFDB uncompressed: 5.1 GB.
- Chapman-Shaoxing Figshare (versão original 10,646): https://figshare.com/collections/ChapmanECG/4560497 — Zheng et al., Sci Data 7, 48 (2020).
- wfdb-python (v4.3.1, Feb 2026): https://wfdb.readthedocs.io/ — `dl_database`, `dl_files`, `rdheader`, `rdrecord`.
- KaggleHub (v0.3.6+): https://github.com/Kaggle/kagglehub — `dataset_download()`, `dataset_load()`, autenticação via `kaggle.json` ou env vars.
