# ----------------------------------------------------------------
# Gunrock -- Fast and Efficient GPU Graph Library
# ----------------------------------------------------------------
# This source code is distributed under the terms of LICENSE.TXT
# in the root directory of this source distribution.
# ----------------------------------------------------------------

#-------------------------------------------------------------------------------
# (make test) Test driver for ALGO
#-------------------------------------------------------------------------------

include ../BaseMakefile.mk
ALGO = ss
EXEC = bin/$(ALGO)_main_$(NVCC_VERSION)_$(ARCH_SUFFIX)
SRC = $(ALGO)_main.cu

test: $(EXEC)

$(EXEC) : $(SRC) $(DEPS)
	mkdir -p bin
	$(NVCC) -w $(DEFINES) $(SM_TARGETS) -o $(EXEC) $(SRC) $(EXTRA_SOURCE) $(NVCCFLAGS) $(ARCH) $(INC) -O3

a : $(SRC) $(DEPS)
	mkdir -p bin
	rm -f $(EXEC)
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o $(EXEC) $(SRC) $(EXTRA_SOURCE) $(NVCCFLAGS) $(ARCH) $(INC) -O3 > a.txt 2>&1 || true
	grep rror: a.txt; grep arn a.txt

.DEFAULT_GOAL := test
