.PHONY: all clean

CC := clang
POLY_CC := /home/chao/src/ppcg/pprem
TARGET_CPU ?= x86

OUT_DIR := cache_based/
SCOP_DIR := scops/
# SRC_FILES := $(wildcard */*/*.c)
# SRC_FILES += $(wildcard */*/*/*.c)
# SRC_FILES := $(filter-out medley/nussinov/Nussinov.orig.c, $(SRC_FILES))
SRC_FILES := $(shell find ./polyNN -name "*.c")
SRC_FILES := $(sort $(SRC_FILES))

OUTS := $(addprefix $(OUT_DIR), $(SRC_FILES:.c=.out))
SCOPS := $(addprefix $(SCOP_DIR), $(subst ./, ,$(SRC_FILES:.c=.scop)))


# SYSROOT := "--sysroot=/usr/aarch64-linux-gnu/"
CROSS_INCLUDE_FLAGS := -I /usr/aarch64-linux-gnu/include/ 
CROSS_LD_FLAGS := -B /usr/lib/gcc-cross/aarch64-linux-gnu/9 -L /usr/lib/gcc-cross/aarch64-linux-gnu/9 -L /usr/aarch64-linux-gnu/lib
ARM_CROSS_FLAGS := -target aarch64-linux-gnu $(CROSS_INCLUDE_FLAGS) $(CROSS_LD_FLAGS)

BENCH_SIZE ?= -D LARGE_DATASET
POLLY_FLAGS = -D POLYBENCH_USE_SCALAR_LB 
# POLLY_FLAGS += -DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS
CC_FLAGS := -O3 
CC_FLAGS += -mllvm -polly 
# CC_FLAGS += -mllvm -polly-parallel 
# CC_FLAGS += -mllvm -polly-omp-backend=GNU 
# CC_FLAGS += -mllvm -polly-num-threads=4 
# CC_FLAGS += -mllvm -polly-scheduling=runtime 
LD_FLAGS := -lgomp -lm

all :$(OUTS)
	@echo "Building done."

scop :$(SCOPS)
	@echo "Scop Generation done."

$(OUTS) :$(OUT_DIR)%.out : %.c utilities/polybench.c
	@mkdir -p $(dir $@)
ifeq ($(TARGET_CPU), aarch64)
	@echo "building $< in arm"
	@$(CC) $(ARM_CROSS_FLAGS) -I utilities/ -I $(dir $<) $(BENCH_SIZE) $(POLLY_FLAGS) $(CC_FLAGS) $< utilities/polybench.c -o $@ $(LD_FLAGS)
else ifeq ($(TARGET_CPU), x86)
	@echo "building $< in x86"
	@$(CC) -I utilities/ -I $(dir $<) $(BENCH_SIZE) $(POLLY_FLAGS) $(CC_FLAGS) $< utilities/polybench.c -o $@ $(LD_FLAGS)
else
	@(echo "Unknown cpu target, not supported yet."; exit 1)
endif

$(SCOPS) :$(SCOP_DIR)%.scop : %.c utilities/polybench.c
	@mkdir -p $(dir $@)
	@echo "generating polyhedral info for $<"
	@$(POLY_CC) -I utilities/ -I $(dir $<) $(BENCH_SIZE) $(POLLY_FLAGS) -o $@ $< 

clean :
	-rm -rf $(OUT_DIR)
	-rm -rf $(SCOP_DIR)
