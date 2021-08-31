.PHONY: all clean

CC := clang

OUT_DIR := out/
SRC_FILES := $(wildcard */*/*.c)
SRC_FILES += $(wildcard */*/*/*.c)
SRC_FILES := $(sort $(SRC_FILES))
SRC_FILES := $(filter-out medley/nussinov/Nussinov.orig.c, $(SRC_FILES))
OUTS := $(addprefix $(OUT_DIR), $(SRC_FILES:.c=.out))

# SYSROOT := "--sysroot=/usr/aarch64-linux-gnu/"
CROSS_INCLUDE_FLAGS := -I /usr/aarch64-linux-gnu/include/ 
CROSS_LD_FLAGS := -B /usr/lib/gcc-cross/aarch64-linux-gnu/9 -L /usr/lib/gcc-cross/aarch64-linux-gnu/9 -L /usr/aarch64-linux-gnu/lib
ARM_CROSS_FLAGS := -target aarch64-linux-gnu $(CROSS_INCLUDE_FLAGS) $(CROSS_LD_FLAGS)

POLLY_FLAGS = -DMEDIUM_DATASET -DPOLYBENCH_USE_SCALAR_LB 
POLLY_FLAGS += -O3 
POLLY_FLAGS += -mllvm -polly 
POLLY_FLAGS += -mllvm -polly-parallel 
POLLY_FLAGS += -mllvm -polly-omp-backend=GNU 
POLLY_FLAGS += -mllvm -polly-num-threads=8 
POLLY_FLAGS += -mllvm -polly-scheduling=runtime 
POLLY_FLAGS += -lgomp -lm


all :$(OUTS)
	@echo $(OUTS)

$(OUTS) :$(SRC_FILES) utilities/polybench.c
	mkdir -p $(dir $@)
	$(CC) $(ARM_CROSS_FLAGS) -I utilities/ -I $(<D) $(POLLY_FLAGS) $< utilities/polybench.c -o $@

clean :
	rm -rf out