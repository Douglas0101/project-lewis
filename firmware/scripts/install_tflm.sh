#!/usr/bin/env bash
# Instala o TensorFlow Lite Micro (TFLM) em firmware/third_party/tflite-micro/.
# O commit SHA e lido de firmware/third_party/tflite-micro.commit.
# Build nativo (x86_64) e ARM (cortex-m4+fp com CMSIS-NN) sao executados com
# paralelismo reduzido (-j1) para evitar "internal compiler error" por OOM em
# runners com pouca RAM.
#
# Uso:
#   ./firmware/scripts/install_tflm.sh
#   make firmware-tflm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FIRMWARE_DIR="${PROJECT_ROOT}/firmware"
TFLM_DIR="${FIRMWARE_DIR}/third_party/tflite-micro"
COMMIT_FILE="${FIRMWARE_DIR}/third_party/tflite-micro.commit"
MAKEFILE="${FIRMWARE_DIR}/Makefile"

REPO_URL="https://github.com/tensorflow/tflite-micro.git"
BUILD_JOBS="${TFLM_BUILD_JOBS:-1}"
BUILD_TYPE="${TFLM_BUILD_TYPE:-release}"

die() {
    echo "[tflm-install] ERRO: $*" >&2
    exit 1
}

info() {
    echo "[tflm-install] $*"
}

# ---------------------------------------------------------------------------
# Commit SHA
# ---------------------------------------------------------------------------
if [[ ! -f "${COMMIT_FILE}" ]]; then
    die "Arquivo de commit nao encontrado: ${COMMIT_FILE}"
fi
COMMIT_SHA="$(tr -d '[:space:]' < "${COMMIT_FILE}")"
if [[ -z "${COMMIT_SHA}" ]]; then
    die "Commit SHA vazio em ${COMMIT_FILE}"
fi
info "TFLM commit pinned: ${COMMIT_SHA}"

# ---------------------------------------------------------------------------
# Toolchain ARM (mesmo usado pelo Makefile do firmware)
# ---------------------------------------------------------------------------
ARM_DIR="${FIRMWARE_DIR}/tools/xpack-arm-none-eabi-gcc-13.3.1-1.1"
if [[ -f "${MAKEFILE}" ]]; then
    parsed_arm_dir="$(grep -E '^ARM_DIR\s*:=' "${MAKEFILE}" | head -1 | sed -E 's/^ARM_DIR\s*:=\s*//')"
    if [[ -n "${parsed_arm_dir}" ]]; then
        parsed_arm_dir="${parsed_arm_dir//\$(PROJECT_ROOT)/${FIRMWARE_DIR}}"
        parsed_arm_dir="${parsed_arm_dir//\$(TOOLS_DIR)/${FIRMWARE_DIR}/tools}"
        ARM_DIR="${parsed_arm_dir}"
    fi
fi

if [[ ! -d "${ARM_DIR}" ]]; then
    die "Toolchain ARM nao encontrado em ${ARM_DIR}. Execute: make firmware-deps"
fi
info "Toolchain ARM: ${ARM_DIR}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_is_valid_checkout() {
    if [[ ! -d "${TFLM_DIR}/.git" ]]; then
        return 1
    fi
    local current
    current="$(cd "${TFLM_DIR}" && git rev-parse HEAD 2>/dev/null)"
    [[ "${current}" == "${COMMIT_SHA}" ]]
}

_expected_native_lib() {
    echo "${TFLM_DIR}/gen/linux_x86_64_${BUILD_TYPE}_gcc/lib/libtensorflow-microlite.a"
}

_expected_arm_lib() {
    echo "${TFLM_DIR}/gen/cortex_m_generic_cortex-m4+fp_${BUILD_TYPE}_cmsis_nn_gcc/lib/libtensorflow-microlite.a"
}

# ---------------------------------------------------------------------------
# Clone / update
# ---------------------------------------------------------------------------
if _is_valid_checkout; then
    info "TFLM ja esta no commit correto em ${TFLM_DIR}"
else
    info "Clonando TFLM (${COMMIT_SHA}) em ${TFLM_DIR}..."
    rm -rf "${TFLM_DIR}"
    mkdir -p "${TFLM_DIR}"
    cd "${TFLM_DIR}"
    git init -q
    git remote add origin "${REPO_URL}"
    git fetch -q --depth 1 origin "${COMMIT_SHA}"
    git -c advice.detachedHead=false checkout -q "${COMMIT_SHA}"
    cd - >/dev/null
fi

# O TFLM ainda usa http://github.com em alguns scripts de download. Forçamos
# https para evitar timeout/bloqueio em redes que não permitem HTTP.
find "${TFLM_DIR}/tensorflow/lite/micro/tools/make" -type f -name "*.sh" \
    -exec sed -i 's|http://github.com|https://github.com|g' {} +

cd "${TFLM_DIR}"

# ---------------------------------------------------------------------------
# Build nativo (usado pelos testes qg7/qg8 em host)
# ---------------------------------------------------------------------------
NATIVE_LIB="$(_expected_native_lib)"
if [[ -f "${NATIVE_LIB}" ]]; then
    info "Biblioteca nativa ja existe: ${NATIVE_LIB}"
else
    info "Build da biblioteca nativa (jobs=${BUILD_JOBS}, build=${BUILD_TYPE})..."
    make -f tensorflow/lite/micro/tools/make/Makefile \
        BUILD_TYPE="${BUILD_TYPE}" \
        microlite -j"${BUILD_JOBS}"
fi

if [[ ! -f "${NATIVE_LIB}" ]]; then
    die "Biblioteca nativa nao foi gerada em ${NATIVE_LIB}"
fi

# ---------------------------------------------------------------------------
# Build ARM (usado pelo firmware STM32F4)
# ---------------------------------------------------------------------------
ARM_LIB="$(_expected_arm_lib)"
if [[ -f "${ARM_LIB}" ]]; then
    info "Biblioteca ARM ja existe: ${ARM_LIB}"
else
    info "Build da biblioteca ARM (jobs=${BUILD_JOBS}, build=${BUILD_TYPE})..."
    make -f tensorflow/lite/micro/tools/make/Makefile \
        TARGET=cortex_m_generic \
        TARGET_ARCH=cortex-m4+fp \
        OPTIMIZED_KERNEL_DIR=cmsis_nn \
        TARGET_TOOLCHAIN_ROOT="${ARM_DIR}/bin/" \
        BUILD_TYPE="${BUILD_TYPE}" \
        microlite -j"${BUILD_JOBS}"
fi

if [[ ! -f "${ARM_LIB}" ]]; then
    die "Biblioteca ARM nao foi gerada em ${ARM_LIB}"
fi

info "TFLM instalado e compilado com sucesso."
