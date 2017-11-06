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

ALGO = cc
test: bin/test_$(ALGO)_$(NVCC_VERSION)_$(ARCH_SUFFIX)

bin/test_$(ALGO)_$(NVCC_VERSION)_$(ARCH_SUFFIX) : test_$(ALGO).cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_$(ALGO)_$(NVCC_VERSION)_$(ARCH_SUFFIX) test_$(ALGO).cu $(EXTRA_SOURCE) $(NVCCFLAGS) $(ARCH) $(INC) -O3 

.DEFAULT_GOAL := test
