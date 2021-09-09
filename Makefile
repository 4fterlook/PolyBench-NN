.PHONY: all clean

CC := clang
TARGET_CPU ?= x86

OUT_DIR := prem_based/
# SRC_FILES := $(wildcard */*/*.c)
# SRC_FILES += $(wildcard */*/*/*.c)
# SRC_FILES := $(filter-out medley/nussinov/Nussinov.orig.c, $(SRC_FILES))
SRC_FILES := $(shell find ./polyNN -name "*.c")
SRC_FILES := $(sort $(SRC_FILES))

OUTS := $(addprefix $(OUT_DIR), $(SRC_FILES:.c=.out))

# SYSROOT := "--sysroot=/usr/aarch64-linux-gnu/"
CROSS_INCLUDE_FLAGS := -I /usr/aarch64-linux-gnu/include/ 
CROSS_LD_FLAGS := -B /usr/lib/gcc-cross/aarch64-linux-gnu/9 -L /usr/lib/gcc-cross/aarch64-linux-gnu/9 -L /usr/aarch64-linux-gnu/lib
ARM_CROSS_FLAGS := -target aarch64-linux-gnu $(CROSS_INCLUDE_FLAGS) $(CROSS_LD_FLAGS)

BENCH_SIZE ?= -DMINI_DATASET
POLLY_FLAGS = -DPOLYBENCH_USE_SCALAR_LB -DPOLYBENCH_TIME 
# POLLY_FLAGS += -DPOLYBENCH_CYCLE_ACCURATE_TIMER
POLLY_FLAGS += -DPOLYBENCH_NO_FLUSH_CACHE
# POLLY_FLAGS += -DPOLYBENCH_DUMP_ARRAYS
POLLY_FLAGS += -O3 
# POLLY_FLAGS += -mllvm -polly 
# POLLY_FLAGS += -mllvm -polly-parallel 
# POLLY_FLAGS += -mllvm -polly-omp-backend=GNU 
# POLLY_FLAGS += -mllvm -polly-num-threads=4 
# POLLY_FLAGS += -mllvm -polly-scheduling=runtime 
LD_FLAGS := -static -lm

all :$(OUTS)
	@echo "Building done."

$(OUTS) :$(OUT_DIR)%.out : %.c utilities/polybench.c
	@mkdir -p $(dir $@)
ifeq ($(TARGET_CPU), aarch64)
	@echo "building $< in arm"
	@$(CC) $(ARM_CROSS_FLAGS) -I utilities/ -I $(dir $<) $(BENCH_SIZE) $(POLLY_FLAGS) $< utilities/polybench.c -o $@ $(LD_FLAGS)
else ifeq ($(TARGET_CPU), x86)
	@echo "building $< in x86"
	@$(CC) -I utilities/ -I $(dir $<) $(BENCH_SIZE) $(POLLY_FLAGS) $< utilities/polybench.c -o $@ $(LD_FLAGS)
else
	@(echo "Unknown cpu target, not supported yet."; exit 1)
endif


clean :
	rm -rf $(OUT_DIR)