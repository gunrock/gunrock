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

objs : $(EXTRA_SOURCE) $(DEPS)
	mkdir -p objs
	$(NVCC) -g $(DEFINES) $(SM_TARGETS) -c $(EXTRA_SOURCE) $(NVCCFLAGS) $(ARCH) $(INC) #-O3 
	mv *.o objs

bin : test_bc.cu $(DEPS)
	mkdir -p bin
	$(NVCC) -g $(DEFINES) $(SM_TARGETS) -o bin/test_bc test_bc.cu objs/*.o $(NVCCFLAGS) $(ARCH) $(INC) #-O3 


# --
# Orig

# include ../BaseMakefile.mk

# ALGO = bc
# test: bin/test_$(ALGO)_$(NVCC_VERSION)_$(ARCH_SUFFIX)

# bin/test_$(ALGO)_$(NVCC_VERSION)_$(ARCH_SUFFIX) : test_$(ALGO).cu $(DEPS)
# 	mkdir -p bin
# 	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_$(ALGO)_$(NVCC_VERSION)_$(ARCH_SUFFIX) test_$(ALGO).cu $(EXTRA_SOURCE) $(NVCCFLAGS) $(ARCH) $(INC) -O3 

# .DEFAULT_GOAL := test
