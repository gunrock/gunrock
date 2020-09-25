# ----------------------------------------------------------------
# Gunrock -- Fast and Efficient GPU Graph Library
# ----------------------------------------------------------------
# This source code is distributed under the terms of LICENSE.TXT
# in the root directory of this source distribution.
# ----------------------------------------------------------------

include ../Makefile.inc
ALGO = graph
EXEC = bin/test_$(ALGO)

test: $(EXEC)

$(EXEC) : test_$(ALGO).cu $(DEPS)
	mkdir -p bin
	$(NVCC) -w $(DEFINES) $(SM_TARGETS) -o $(EXEC) test_$(ALGO).cu $(EXTRA_SOURCE) $(NVCCFLAGS) $(ARCH) $(INC) -O3

.DEFAULT_GOAL := test
