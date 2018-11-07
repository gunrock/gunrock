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
ALGO = mf
EXEC = bin/test_$(ALGO)_$(NVCC_VERSION)_$(ARCH_SUFFIX)
MAXFLOWVK_PATH = ../../gunrock/app/mf/maxflow-v3.04.src
MAXFLOWVK = $(MAXFLOWVK_PATH)/graph.cpp $(MAXFLOWVK_PATH)/maxflow.cpp

test: $(EXEC)

$(EXEC) : test_$(ALGO).cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o $(EXEC) test_$(ALGO).cu $(EXTRA_SOURCE) $(NVCCFLAGS) $(MAXFLOWVK) -I. -w $(ARCH) $(INC) -O3

a : test_$(ALGO).cu $(DEPS)
	mkdir -p bin
	rm -f $(EXEC)
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o $(EXEC) test_$(ALGO).cu $(EXTRA_SOURCE) $(NVCCFLAGS) $(MAXFLOWVK) -I. $(ARCH) $(INC) -O3 > a.txt 2>&1 || true
	grep rror: a.txt; grep arn a.txt

.DEFAULT_GOAL := test
