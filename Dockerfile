# Project-Lewis — Container de produção/desenvolvimento
# Hardware-alvo: STM32F4 (TFLM); container usado para CI/reprodução de ambiente.
FROM python:3.12-slim-bookworm

ARG UV_VERSION=0.11.8
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Instala dependências de sistema para compilação de pacotes científicos (fallback)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Instala uv (gerenciador de pacotes Python rápido/reprodutível) em /usr/local/bin
# para ficar acessível ao usuário não-root.
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv \
    && cp /root/.local/bin/uvx /usr/local/bin/uvx
ENV PATH="/usr/local/bin:${PATH}"

WORKDIR /app

# Copia apenas os metadados de dependências primeiro (cache de layer).
# README.md é necessário porque pyproject.toml o referencia.
COPY pyproject.toml uv.lock README.md ./

# Sincroniza dependências de desenvolvimento (lockfile garante reprodutibilidade)
RUN uv sync --frozen

# Copia o código-fonte do projeto (incluindo testes para CI)
COPY src ./src
COPY tests ./tests
COPY config ./config
COPY Makefile ./
COPY README.md README_EXECUCAO.md ./

# Cria usuário não-root para execução
RUN groupadd -r lewis && useradd -r -g lewis -d /app lewis \
    && chown -R lewis:lewis /app
USER lewis

# Variável de ambiente para expor o pacote src/ no PYTHONPATH via uv
ENV PATH="/app/.venv/bin:${PATH}"

# Comando padrão: shell interativo para desenvolvimento/CI
CMD ["bash"]
