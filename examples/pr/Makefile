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
ALGO = pr
EXEC = bin/test_$(ALGO)_$(NVCC_VERSION)_$(ARCH_SUFFIX)

test: $(EXEC)

$(EXEC) : test_$(ALGO).cu $(DEPS)
	mkdir -p bin
	$(NVCC) -w $(DEFINES) $(SM_TARGETS) -o $(EXEC) test_$(ALGO).cu $(EXTRA_SOURCE) $(NVCCFLAGS) $(ARCH) $(INC) -O3

a : test_$(ALGO).cu $(DEPS)
	mkdir -p bin
	rm -f $(EXEC)
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o $(EXEC) test_$(ALGO).cu $(EXTRA_SOURCE) $(NVCCFLAGS) $(ARCH) $(INC) -O3 > a.txt 2>&1 || true
	grep rror: a.txt; grep arn a.txt

.DEFAULT_GOAL := test
