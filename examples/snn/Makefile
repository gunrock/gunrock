# ----------------------------------------------------------------
# Gunrock -- Fast and Efficient GPU Graph Library
# ----------------------------------------------------------------
# This source code is distributed under the terms of LICENSE.TXT
# in the root directory of this source distribution.
# ----------------------------------------------------------------

#-------------------------------------------------------------------------------
# (make test) Test driver for KNN
#-------------------------------------------------------------------------------

include ../BaseMakefile.mk
ALGO = snn
EXEC = bin/test_$(ALGO)_$(NVCC_VERSION)_$(ARCH_SUFFIX)

FAISS_INC =
FAISS_LINK =

ifdef FAISS_INCLUDE_DIR
    ifdef FAISS_LIBRARY
        FAISS_INC = -I$(FAISS_INCLUDE_DIR)
        FAISS_LINK = -L$(FAISS_LIBRARY) -lfaiss -DFAISS_FOUND=1
    else
        $(info FAISS not found. In order to use FAISS, please define FAISS_LIBRARY)
    endif
else
    $(info FAISS not found. In order to use FAISS, please define FAISS_INCLUDE_DIR)
endif

INC += $(FAISS_INC) $(FAISS_LINK)

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
