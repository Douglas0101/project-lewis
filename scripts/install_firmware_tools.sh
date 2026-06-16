#!/usr/bin/env bash
# Instalação local do toolchain ARM e Renode para simulação STM32F4.
# Não requer sudo; tudo fica em firmware/tools/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOLS_DIR="${PROJECT_ROOT}/firmware/tools"
mkdir -p "${TOOLS_DIR}"

# ---------------------------------------------------------------------------
# ARM GNU Toolchain
# ---------------------------------------------------------------------------
ARM_VERSION="13.3.rel1"
ARM_TARBALL="arm-gnu-toolchain-${ARM_VERSION}-x86_64-arm-none-eabi.tar.xz"
ARM_URL="https://developer.arm.com/-/media/Files/downloads/gnu/${ARM_VERSION}/binrel/${ARM_TARBALL}"
ARM_DIR="${TOOLS_DIR}/arm-gnu-toolchain-${ARM_VERSION}"

if [[ -d "${ARM_DIR}/bin" ]]; then
    echo "[install] ARM toolchain já instalado em ${ARM_DIR}"
else
    echo "[install] Baixando ARM toolchain ${ARM_VERSION}..."
    curl -fsSL -o "${TOOLS_DIR}/${ARM_TARBALL}" "${ARM_URL}" || {
        echo "[install] Falha no download do ARM toolchain; tentando wget..."
        wget -q -O "${TOOLS_DIR}/${ARM_TARBALL}" "${ARM_URL}"
    }
    echo "[install] Extraindo ARM toolchain..."
    tar -xf "${TOOLS_DIR}/${ARM_TARBALL}" -C "${TOOLS_DIR}"
    mv "${TOOLS_DIR}/arm-gnu-toolchain-${ARM_VERSION}-x86_64-arm-none-eabi" "${ARM_DIR}"
    rm -f "${TOOLS_DIR}/${ARM_TARBALL}"
    echo "[install] ARM toolchain instalado em ${ARM_DIR}"
fi

# ---------------------------------------------------------------------------
# Renode
# ---------------------------------------------------------------------------
RENODE_VERSION="1.15.3"
RENODE_TARBALL="renode-${RENODE_VERSION}.linux-portable.tar.gz"
RENODE_URL="https://github.com/renode/renode/releases/download/v${RENODE_VERSION}/${RENODE_TARBALL}"
RENODE_DIR="${TOOLS_DIR}/renode-${RENODE_VERSION}"

if [[ -x "${RENODE_DIR}/renode" ]]; then
    echo "[install] Renode já instalado em ${RENODE_DIR}"
else
    echo "[install] Baixando Renode ${RENODE_VERSION}..."
    curl -fsSL -L -o "${TOOLS_DIR}/${RENODE_TARBALL}" "${RENODE_URL}" || {
        echo "[install] Falha no download do Renode; tentando wget..."
        wget -q -O "${TOOLS_DIR}/${RENODE_TARBALL}" "${RENODE_URL}"
    }
    echo "[install] Extraindo Renode..."
    tar -xzf "${TOOLS_DIR}/${RENODE_TARBALL}" -C "${TOOLS_DIR}"
    mv "${TOOLS_DIR}/renode_${RENODE_VERSION}_portable" "${RENODE_DIR}" || true
    rm -f "${TOOLS_DIR}/${RENODE_TARBALL}"
    echo "[install] Renode instalado em ${RENODE_DIR}"
fi

# ---------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------
echo ""
echo "[install] Ferramentas instaladas em ${TOOLS_DIR}:"
echo "  ARM GCC:  ${ARM_DIR}/bin/arm-none-eabi-gcc"
echo "  Renode:   ${RENODE_DIR}/renode"
echo ""
echo "[install] Adicione ao PATH temporariamente:"
echo "  export PATH=\"${ARM_DIR}/bin:${RENODE_DIR}:\$PATH\""
