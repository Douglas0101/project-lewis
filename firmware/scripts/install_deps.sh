#!/usr/bin/env bash
# Instala toolchain ARM GCC e Renode localmente em firmware/tools/.
# Uso: ./scripts/install_deps.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOLS_DIR="${PROJECT_ROOT}/tools"
mkdir -p "${TOOLS_DIR}"

cd "${TOOLS_DIR}"

ARM_URL="https://github.com/xpack-dev-tools/arm-none-eabi-gcc-xpack/releases/download/v13.3.1-1.1/xpack-arm-none-eabi-gcc-13.3.1-1.1-linux-x64.tar.gz"
ARM_DIR="${TOOLS_DIR}/xpack-arm-none-eabi-gcc-13.3.1-1.1"

RENODE_URL="https://github.com/renode/renode/releases/download/v1.15.3/renode-1.15.3.linux-portable.tar.gz"
RENODE_DIR="${TOOLS_DIR}/renode_1.15.3_portable"

install_arm() {
    if [[ -d "${ARM_DIR}" ]]; then
        echo "[deps] ARM GCC ja instalado em ${ARM_DIR}"
        return
    fi
    echo "[deps] Baixando ARM GCC..."
    curl -L -o arm-gcc.tar.gz "${ARM_URL}"
    echo "[deps] Extraindo ARM GCC..."
    tar -xzf arm-gcc.tar.gz
    rm arm-gcc.tar.gz
    echo "[deps] ARM GCC instalado."
}

install_renode() {
    if [[ -d "${RENODE_DIR}" ]]; then
        echo "[deps] Renode ja instalado em ${RENODE_DIR}"
        return
    fi
    echo "[deps] Baixando Renode..."
    curl -L -o renode.tar.gz "${RENODE_URL}"
    echo "[deps] Extraindo Renode..."
    tar -xzf renode.tar.gz
    # O pacote portable cria uma pasta adicional com o nome da versao.
    if [[ -d "renode_1.15.3_portable" ]]; then
        mv renode_1.15.3_portable renode_1.15.3_portable_tmp
        mv renode_1.15.3_portable_tmp "${RENODE_DIR}"
    fi
    rm renode.tar.gz
    echo "[deps] Renode instalado."
}

install_arm
install_renode

echo "[deps] Verificando instalacao..."
"${ARM_DIR}/bin/arm-none-eabi-gcc" --version | head -1
"${RENODE_DIR}/renode" --version | head -1
echo "[deps] OK."
