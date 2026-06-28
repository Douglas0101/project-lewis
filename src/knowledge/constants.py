"""Constantes da Camada C11 — Knowledge Layer.

Autor: Douglas Souza
Data: 2026-06-27
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"
SRC_DIR = PROJECT_ROOT / "src"
FIRMWARE_DIR = PROJECT_ROOT / "firmware" / "src"
KNOWLEDGE_DB = PROJECT_ROOT / "data" / "knowledge.db"
LINEAGE_DIR = PROJECT_ROOT / "data" / "lineage" / "knowledge"
DLQ_PATH = PROJECT_ROOT / "data" / ".dlq" / "knowledge_rejected.jsonl"
LOG_QUERIES = PROJECT_ROOT / "logs" / "knowledge_queries.jsonl"
CONFIG_PATH = PROJECT_ROOT / "config" / "knowledge_v2.0.yaml"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
COLLECTION_NAME = "project_lewis_knowledge"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
MAX_CHUNK_CHARS = 1024

TAG_KEYWORDS = {
    "quantizacao": ["quantizacao", "INT8", "PTQ", "QAT", "zero_point", "quantization"],
    "firmware": ["firmware", "STM32", "TFLM", "CMSIS-NN", "bare-metal", "Cortex-M4"],
    "dsp": ["filtro", "Butterworth", "AMPT", "R-peak", "filtfilt", "bandpass", "notch"],
    "ml": ["GroupKFold", "F1-macro", "backbone", "fine-tuning", "AAMI", "inter-patient"],
    "lgpd": ["LGPD", "PII", "anonimizacao", "consentimento", "dados pessoais"],
    "devops": ["uv", "Docker", "CI/CD", "pre-commit", "DVC", "Makefile"],
    "energia": ["energia", "mJ", "mAh", "Renode", "autonomia", "consumo"],
    "dados": ["MIT-BIH", "Chapman", "PhysioNet", "dataset", "download", "resample"],
    "modelagem": ["CNN", "1D-CNN", "softmax", "sigmoid", "backbone", "pre-treino"],
    "seguranca": ["JWT", "OAuth2", "Argon2", "bcrypt", "hash", "CSP", "XSS"],
}

LAYER_MAP = {
    "Camada-01": "C01",
    "Camada-02": "C02",
    "Camada-03": "C03",
    "Camada-04": "C04",
    "Camada-05": "C05",
    "Camada-06": "C06",
    "Camada-07": "C07",
    "Camada-08": "C08",
    "Camada-09": "C09",
    "Camada-10": "C10",
    "SDD_": "SDD",
    "PRD": "PRD",
    "UNIFIED": "UNIFIED",
    "ESPECIFICACAO": "ESPECIFICACAO",
    "SIMULATION": "SIMULATION",
    "DEBITO": "DEBITO_TECNICO",
}

PII_PATTERNS = [
    r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b",  # CPF
    r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b",  # CNPJ
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
    r"\b(?:paciente|patient|nome|name):\s*\w+",  # Nomes próprios em contexto clínico
]

FORBIDDEN_EXTENSIONS = {".dat", ".mat", ".hea", ".atr", ".xyz", ".ecg"}
FORBIDDEN_PATH_PATTERNS = [
    "raw_chapman/",
    "raw_mitbih/",
    "raw_svdb/",
    "raw_afdb/",
    "raw_incart/",
    "raw_ptbxl/",
]
