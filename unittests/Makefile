NVCC = "$(shell which nvcc)"
NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'))

GEN_SM70 = -gencode=arch=compute_70,code=\"sm_70,compute_70\" # Volta V100
SM_TARGETS = $(GEN_SM70)

CUDA_INC 	= -I"$(shell dirname $(NVCC))/../include"
MGPU_INC 	= -I"../externals/moderngpu/include"
CUB_INC 	= -I"../externals/cub"
CUDA_CXX 	= -I"../externals/cuda-api-wrappers/src"
R_JSON		= -I"../../externals/rapidjson/include"

GUNROCK_DEF = -Xcompiler -DGUNROCKVERSION=2.0.0

DEFINES 	= -DGIT_SHA1="\"$(shell git rev-parse HEAD)\""

NVCCFLAGS 	= -lineinfo --std=c++14 --expt-extended-lambda -rdc=true

ifeq ($(verbose), 1)
    NVCCFLAGS += -v
endif

ifeq ($(keep), 1)
    NVCCFLAGS += -keep
endif

EXTRA_SOURCE = ../gunrock/data_structs/array.cuh

DEPS = 	./Makefile \
		$(EXTRA_SOURCE)

#-------------------------------------------------------------------------------
# (make test) Test driver for Gunrock
#-------------------------------------------------------------------------------

ALGO = array
EXEC = bin/test_$(ALGO)_$(NVCC_VERSION)_$(ARCH_SUFFIX)

test: $(EXEC)

$(EXEC) : test_$(ALGO).cu $(DEPS)
	mkdir -p bin
	$(NVCC) -w $(DEFINES) $(SM_TARGETS) -o $(EXEC) test_$(ALGO).cu $(EXTRA_SOURCE) $(NVCCFLAGS) $(ARCH) $(INC) -O3

debug : test_$(ALGO).cu $(DEPS)
	mkdir -p bin
	rm -f $(EXEC)
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o $(EXEC) test_$(ALGO).cu $(EXTRA_SOURCE) $(NVCCFLAGS) $(ARCH) $(INC) -O0 -g -G > a.txt 2>&1 || true
	grep rror: a.txt; grep arn a.txt

.DEFAULT_GOAL := test

#-------------------------------------------------------------------------------
# Clean
#-------------------------------------------------------------------------------

clean :
	rm -f bin/*_$(NVCC_VERSION)_$(ARCH_SUFFIX)*
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o