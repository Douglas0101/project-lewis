# Project-Lewis Firmware Makefile
# Suporta build para STM32F4 (Renode) e host nativo (x86_64).

PROJECT_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_ROOT := $(PROJECT_ROOT:%/=%)

# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------
TOOLS_DIR := $(PROJECT_ROOT)/tools
ARM_DIR   := $(TOOLS_DIR)/xpack-arm-none-eabi-gcc-13.3.1-1.1
RENOD_DIR := $(TOOLS_DIR)/renode-1.15.3

CC_ARM    := $(ARM_DIR)/bin/arm-none-eabi-gcc
CXX_ARM   := $(ARM_DIR)/bin/arm-none-eabi-g++
LD_ARM    := $(ARM_DIR)/bin/arm-none-eabi-ld
OBJCOPY   := $(ARM_DIR)/bin/arm-none-eabi-objcopy
SIZE      := $(ARM_DIR)/bin/arm-none-eabi-size

CC_NATIVE  := gcc
CXX_NATIVE := g++

# ---------------------------------------------------------------------------
# Diretorios
# ---------------------------------------------------------------------------
SRC_DIR    := $(PROJECT_ROOT)/src
BUILD_DIR  := $(PROJECT_ROOT)/build
BUILD_STM  := $(BUILD_DIR)/stm32f4
BUILD_NAT  := $(BUILD_DIR)/native

# ---------------------------------------------------------------------------
# Fontes (comuns)
# ---------------------------------------------------------------------------
C_SOURCES_COMMON := \
	$(SRC_DIR)/app/main.c \
	$(SRC_DIR)/app/command_loop.c \
	$(SRC_DIR)/utils/debug.c \
	$(SRC_DIR)/dsp/adc_stub.c

CXX_SOURCES_COMMON := \
	$(SRC_DIR)/ml/inference.cpp

C_SOURCES_STM := \
	$(SRC_DIR)/platform/startup_stm32f4.c \
	$(SRC_DIR)/platform/syscalls.c \
	$(C_SOURCES_COMMON)

CXX_SOURCES_STM := \
	$(SRC_DIR)/platform/cxx_stubs.cpp \
	$(CXX_SOURCES_COMMON)
C_SOURCES_NAT := $(C_SOURCES_COMMON)

# HAL selecionada por alvo
C_SOURCES_SIM := $(SRC_DIR)/hal/simulator/hal_sim.c
C_SOURCES_UART_ARM := $(SRC_DIR)/hal/target/uart_stm32f4.c
C_SOURCES_UART_NAT := $(SRC_DIR)/hal/native/uart_host.c

INCLUDES = \
	-I$(SRC_DIR) \
	-I$(SRC_DIR)/ml

# ---------------------------------------------------------------------------
# Flags comuns
# ---------------------------------------------------------------------------
WARNINGS := -Wall -Wextra -Werror -Wshadow -Wundef -Wdouble-promotion
OPT      := -O2
LEWIS_USE_TFLM ?= 0
RENODE_SIMULATION ?= 0
ALLOW_STUB ?= 0
LEWIS_ARENA_SIZE ?= 65536

# Linker script padrao (192 KB de RAM); pode ser sobrescrito por alvo.
LDSCRIPT ?= $(PROJECT_ROOT)/stm32f407vg.ld

# ---------------------------------------------------------------------------
# CI / quality gates
# ---------------------------------------------------------------------------
ifeq ($(CI),1)
  ifeq ($(ALLOW_STUB),1)
    $(error CI build cannot use ALLOW_STUB=1 (HG-02))
  endif
endif

# Host-native usa TFLM real por padrao; stub somente com ALLOW_STUB=1.
ifneq ($(filter native,$(MAKECMDGOALS)),)
  ifeq ($(ALLOW_STUB),0)
    LEWIS_USE_TFLM = 1
  else
    LEWIS_USE_TFLM = 0
  endif
endif
ifneq ($(filter native-tflm,$(MAKECMDGOALS)),)
  LEWIS_USE_TFLM = 1
endif

# ---------------------------------------------------------------------------
# Alvo STM32F4 (Renode)
# ---------------------------------------------------------------------------
CFLAGS_STM_COMMON = -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 \
	-DSTM32F407xx \
	-DLEWIS_ARENA_SIZE=$(LEWIS_ARENA_SIZE) \
	$(OPT) $(WARNINGS) -g \
	$(INCLUDES) \
	-include $(BUILD_STM)/config.h \
	--specs=nosys.specs

CFLAGS_STM = $(CFLAGS_STM_COMMON) -std=gnu11

CXXFLAGS_STM = $(CFLAGS_STM_COMMON) -fno-rtti -fno-exceptions -std=c++17 \
	-Wno-undef -Wno-unused-parameter

LDFLAGS_STM = -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 \
	-T$(LDSCRIPT) \
	--specs=nosys.specs

OBJECTS_STM := $(patsubst $(SRC_DIR)/%.c,$(BUILD_STM)/%.o,$(C_SOURCES_STM) $(C_SOURCES_SIM) $(C_SOURCES_UART_ARM))
OBJECTS_STM_CXX := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_STM)/%.o,$(CXX_SOURCES_STM))

CONFIG_STM_STAMP := $(BUILD_STM)/.config_$(LEWIS_USE_TFLM)_$(RENODE_SIMULATION)_$(LEWIS_ARENA_SIZE).stamp

.PHONY: _config_state_force
_config_state_force:

$(BUILD_STM)/.config_state: _config_state_force $(MAKEFILE_LIST)
	@mkdir -p $(dir $@)
	@echo "$(LEWIS_USE_TFLM)_$(RENODE_SIMULATION)_$(LEWIS_ARENA_SIZE)" > $@.tmp
	@if [ ! -f $@ ] || ! cmp -s $@.tmp $@; then mv $@.tmp $@; else rm $@.tmp; fi

$(BUILD_STM)/config.h: $(BUILD_STM)/.config_state
	@mkdir -p $(dir $@)
	@echo "#define LEWIS_USE_TFLM $(LEWIS_USE_TFLM)" > $@
	@echo "#define RENODE_SIMULATION $(RENODE_SIMULATION)" >> $@

$(CONFIG_STM_STAMP): $(MAKEFILE_LIST)
	@mkdir -p $(dir $@)
	@touch $@

$(BUILD_STM)/%.o: $(SRC_DIR)/%.c $(CONFIG_STM_STAMP) $(MAKEFILE_LIST) $(BUILD_STM)/config.h
	@mkdir -p $(dir $@)
	$(CC_ARM) $(CFLAGS_STM) -c $< -o $@

$(BUILD_STM)/%.o: $(SRC_DIR)/%.cpp $(CONFIG_STM_STAMP) $(MAKEFILE_LIST) $(BUILD_STM)/config.h
	@mkdir -p $(dir $@)
	$(CXX_ARM) $(CXXFLAGS_STM) -c $< -o $@

$(BUILD_STM)/lewis.elf: $(OBJECTS_STM) $(OBJECTS_STM_CXX)
	@mkdir -p $(dir $@)
	$(CC_ARM) $(OBJECTS_STM) $(OBJECTS_STM_CXX) $(LDFLAGS_STM) -o $@
	$(SIZE) $@

$(BUILD_STM)/lewis.bin: $(BUILD_STM)/lewis.elf
	$(OBJCOPY) -O binary $< $@

# ---------------------------------------------------------------------------
# TFLM
# ---------------------------------------------------------------------------
LDFLAGS_NAT := -lm
TFLM_DIR        := $(PROJECT_ROOT)/third_party/tflite-micro
TFLM_MAKEFILE   := $(TFLM_DIR)/tensorflow/lite/micro/tools/make/Makefile
TFLM_LIB_ARM    := $(TFLM_DIR)/gen/cortex_m_generic_cortex-m4+fp_release_cmsis_nn_gcc/lib/libtensorflow-microlite.a
TFLM_LIB_NATIVE := $(TFLM_DIR)/gen/linux_x86_64_release_gcc/lib/libtensorflow-microlite.a
TFLM_INCLUDE    := -I$(TFLM_DIR) \
                   -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
                   -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
                   -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/kissfft

# Quando TFLM estiver ativo, inclui headers e linka com a lib.
ifeq ($(LEWIS_USE_TFLM),1)
INCLUDES += $(TFLM_INCLUDE)
CFLAGS_STM_COMMON += -DTF_LITE_STATIC_MEMORY
LDFLAGS_STM += -L$(dir $(TFLM_LIB_ARM)) -ltensorflow-microlite -lm
LDFLAGS_NAT += -L$(dir $(TFLM_LIB_NATIVE)) -ltensorflow-microlite -lm
endif

# ---------------------------------------------------------------------------
# Alvo nativo (host x86_64) — para iteracao rapida sem emulador
# ---------------------------------------------------------------------------
ifeq ($(LEWIS_USE_TFLM),1)
CFLAGS_NAT := -DLEWIS_HOST_NATIVE -DTF_LITE_STATIC_MEMORY \
	$(OPT) $(WARNINGS) -g -std=gnu11 \
	$(INCLUDES) \
	-include $(BUILD_NAT)/config.h

CXXFLAGS_NAT := -DLEWIS_HOST_NATIVE -DTF_LITE_STATIC_MEMORY \
	$(OPT) $(WARNINGS) -g -fno-rtti -fno-exceptions -std=c++17 \
	-Wno-undef -Wno-unused-parameter \
	$(INCLUDES) \
	-include $(BUILD_NAT)/config.h
else
CFLAGS_NAT := -DLEWIS_HOST_NATIVE \
	$(OPT) $(WARNINGS) -g -std=gnu11 \
	$(INCLUDES) \
	-include $(BUILD_NAT)/config.h

CXXFLAGS_NAT := -DLEWIS_HOST_NATIVE \
	$(OPT) $(WARNINGS) -g -fno-rtti -fno-exceptions -std=c++17 \
	$(INCLUDES) \
	-include $(BUILD_NAT)/config.h
endif

CXX_SOURCES_NAT := $(CXX_SOURCES_COMMON)

OBJECTS_NAT := $(patsubst $(SRC_DIR)/%.c,$(BUILD_NAT)/%.o,$(C_SOURCES_NAT) $(C_SOURCES_SIM) $(C_SOURCES_UART_NAT))
OBJECTS_NAT_CXX := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_NAT)/%.o,$(CXX_SOURCES_NAT))

CONFIG_NAT_STAMP := $(BUILD_NAT)/.config_$(ALLOW_STUB)_$(LEWIS_USE_TFLM).stamp

.PHONY: _config_state_force_nat
_config_state_force_nat:

$(BUILD_NAT)/.config_state: _config_state_force_nat $(MAKEFILE_LIST)
	@mkdir -p $(dir $@)
	@echo "$(ALLOW_STUB)_$(LEWIS_USE_TFLM)_$(RENODE_SIMULATION)" > $@.tmp
	@if [ ! -f $@ ] || ! cmp -s $@.tmp $@; then mv $@.tmp $@; else rm $@.tmp; fi

$(BUILD_NAT)/config.h: $(BUILD_NAT)/.config_state
	@mkdir -p $(dir $@)
	@echo "#define LEWIS_USE_TFLM $(LEWIS_USE_TFLM)" > $@
	@echo "#define RENODE_SIMULATION $(RENODE_SIMULATION)" >> $@

$(CONFIG_NAT_STAMP): $(MAKEFILE_LIST)
	@mkdir -p $(dir $@)
	@touch $@

$(BUILD_NAT)/%.o: $(SRC_DIR)/%.c $(CONFIG_NAT_STAMP) $(MAKEFILE_LIST) $(BUILD_NAT)/config.h
	@mkdir -p $(dir $@)
	$(CC_NATIVE) $(CFLAGS_NAT) -c $< -o $@

$(BUILD_NAT)/%.o: $(SRC_DIR)/%.cpp $(CONFIG_NAT_STAMP) $(MAKEFILE_LIST) $(BUILD_NAT)/config.h
	@mkdir -p $(dir $@)
	$(CXX_NATIVE) $(CXXFLAGS_NAT) -c $< -o $@

$(BUILD_NAT)/lewis: $(OBJECTS_NAT) $(OBJECTS_NAT_CXX)
	@mkdir -p $(dir $@)
	$(CXX_NATIVE) $(OBJECTS_NAT) $(OBJECTS_NAT_CXX) $(LDFLAGS_NAT) -o $@

# ---------------------------------------------------------------------------
# Targets publicos
# ---------------------------------------------------------------------------
.PHONY: all stm32f4 stm32f4-48k native native-tflm clean clean-all firmware-run firmware-test size tflm-lib tflm-lib-native firmware-deps

all: stm32f4

stm32f4: $(BUILD_STM)/lewis.elf $(BUILD_STM)/lewis.bin

# QG12: firmware com 48 KB de RAM e arena reduzida (deve falhar INIT).
stm32f4-48k: LDSCRIPT := $(PROJECT_ROOT)/stm32f407vg_48k.ld
stm32f4-48k: LEWIS_ARENA_SIZE := 16384
stm32f4-48k: $(BUILD_STM)/lewis_48k.elf $(BUILD_STM)/lewis_48k.bin

$(BUILD_STM)/lewis_48k.elf: $(OBJECTS_STM) $(OBJECTS_STM_CXX)
	@mkdir -p $(dir $@)
	$(CC_ARM) $(OBJECTS_STM) $(OBJECTS_STM_CXX) $(LDFLAGS_STM) -o $@
	$(SIZE) $@

$(BUILD_STM)/lewis_48k.bin: $(BUILD_STM)/lewis_48k.elf
	$(OBJCOPY) -O binary $< $@

native: $(BUILD_NAT)/lewis

ifeq ($(ALLOW_STUB),0)
native: tflm-lib-native
endif

native-tflm: tflm-lib-native native

tflm-lib-native:
	cd $(TFLM_DIR) && make -f $(TFLM_MAKEFILE) TARGET=linux microlite \
		BUILD_TYPE=release -j$(shell nproc 2>/dev/null || echo 2)

clean:
	rm -rf $(BUILD_DIR)

clean-all: clean
	$(MAKE) -f $(TFLM_MAKEFILE) TARGET=linux clean
	$(MAKE) -f $(TFLM_MAKEFILE) TARGET=cortex_m_generic TARGET_ARCH=cortex-m4 OPTIMIZED_KERNEL_DIR=cmsis_nn clean

firmware-deps:
	$(PROJECT_ROOT)/scripts/install_deps.sh

tflm-lib:
	cd $(TFLM_DIR) && make -f $(TFLM_MAKEFILE) TARGET=cortex_m_generic TARGET_ARCH=cortex-m4 \
		OPTIMIZED_KERNEL_DIR=cmsis_nn TARGET_TOOLCHAIN_ROOT=$(ARM_DIR)/bin/ \
		BUILD_TYPE=release -j$(shell nproc 2>/dev/null || echo 2)

firmware-run: stm32f4
	$(RENOD_DIR)/renode --disable-xwt --console $(PROJECT_ROOT)/renode/stm32f4_discovery.resc

firmware-test:
	$(MAKE) RENODE_SIMULATION=1 LEWIS_USE_TFLM=1 stm32f4
	python3 $(PROJECT_ROOT)/scripts/run_renode_tests.py \
		--renode $(RENOD_DIR)/renode \
		--resc $(PROJECT_ROOT)/renode/stm32f4_discovery.resc \
		--robot $(PROJECT_ROOT)/renode/test_shutdown.robot \
		--bin $(BUILD_STM)/lewis.bin \
		--elf $(BUILD_STM)/lewis.elf \
		--run-time 5

size: stm32f4
	$(SIZE) $(BUILD_STM)/lewis.elf
