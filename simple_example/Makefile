# ----------------------------------------------------------------
# Gunrock -- Fast and Efficient GPU Graph Library
# ----------------------------------------------------------------
# This source code is distributed under the terms of LICENSE.TXT
# in the root directory of this source distribution.
# ----------------------------------------------------------------

#-------------------------------------------------------------------------------
# Build script for project
#-------------------------------------------------------------------------------

force64 = 1
NVCC = "$(shell which nvcc)"
NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'))

KERNELS =

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])

#-------------------------------------------------------------------------------
# Gen targets
#-------------------------------------------------------------------------------

GEN_SM35 = -gencode=arch=compute_35,code=\"sm_35,compute_35\"
GEN_SM30 = -gencode=arch=compute_30,code=\"sm_30,compute_30\"
GEN_SM20 = -gencode=arch=compute_20,code=\"sm_20,compute_20\"
GEN_SM13 = -gencode=arch=compute_13,code=\"sm_13,compute_13\"
GEN_SM10 = -gencode=arch=compute_10,code=\"sm_10,compute_10\"
SM_TARGETS = $(GEN_SM35)
#$(GEN_SM30) $(GEN_SM20) $(GEN_SM13) $(GEN_SM10)

#-------------------------------------------------------------------------------
# Tune arch
#-------------------------------------------------------------------------------

ifdef tunearch
    TUNE_ARCH = $(tunearch)
else
	TUNE_ARCH = 200
endif

ifeq ($(TUNE_ARCH), 200)
	TUNE_SM_TARGETS = $(GEN_SM20)
endif
ifeq ($(TUNE_ARCH), 130)
	TUNE_SM_TARGETS = $(GEN_SM13)
endif
ifeq ($(TUNE_ARCH), 100)
	TUNE_SM_TARGETS = $(GEN_SM10)
endif

#-------------------------------------------------------------------------------
# Tune size
#-------------------------------------------------------------------------------

ifdef tunesize
    TUNE_SIZE = $(tunesize)
else
	TUNE_SIZE = 4
endif


#-------------------------------------------------------------------------------
# Libs
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Includes
#-------------------------------------------------------------------------------

CUDA_INC = "$(shell dirname $(NVCC))/../include"
MGPU_INC = "../externals/moderngpu/include"
BOOST_INC = "../../boost_1_53_0"
INC = -I$(CUDA_INC) -I$(BOOST_INC) -I$(MGPU_INC) -I.. -I../..

#-------------------------------------------------------------------------------
# Defines
#-------------------------------------------------------------------------------

DEFINES =

#-------------------------------------------------------------------------------
# Compiler Flags
#-------------------------------------------------------------------------------

ifneq ($(force64), 1)
	# Compile with 32-bit device pointers by default
	ARCH_SUFFIX = i386
	ARCH = -m32
else
	ARCH_SUFFIX = x86_64
	ARCH = -m64
endif

NVCCFLAGS = -Xptxas -v -Xcudafe -\#

ifeq (WIN_NT, $(findstring WIN_NT, $(OSUPPER)))
	NVCCFLAGS += -Xcompiler /bigobj -Xcompiler /Zm500
endif

ifeq (,$(findstring 3.0, $(NVCC_VERSION)))
ifneq ($(abi), 1)
	# Disable the ABI by default for 3.1+
	NVCCFLAGS += -Xptxas -abi=no
endif
endif

ifeq ($(verbose), 1)
    NVCCFLAGS += -v
endif

ifeq ($(keep), 1)
    NVCCFLAGS += -keep
endif

ifdef maxregisters
    NVCCFLAGS += -maxrregcount $(maxregisters)
endif

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS =			./Makefile \
				$(wildcard ../gunrock/util/*.cuh) \
				$(wildcard ../gunrock/util/**/*.cuh) \
				$(wildcard ../gunrock/*.cuh) \
				$(wildcard ../gunrock/graphio/*.cuh) \
				$(wildcard ../gunrock/oprtr/*.cuh) \
				$(wildcard ../gunrock/oprtr/**/*.cuh) \
				$(wildcard ../gunrock/app/*.cuh) \
				$(wildcard ../gunrock/app/**/*.cuh)

#-------------------------------------------------------------------------------
# (make simple) Simple example driver for three primitives: CC, BFS and BC
#-------------------------------------------------------------------------------

simple: bin/simple_example_$(NVCC_VERSION)_$(ARCH_SUFFIX)

bin/simple_example_$(NVCC_VERSION)_$(ARCH_SUFFIX) : simple_example.cu ../externals/moderngpu/src/mgpucontext.cu ../externals/moderngpu/src/mgpuutil.cpp $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/simple_example_$(NVCC_VERSION)_$(ARCH_SUFFIX) simple_example.cu ../externals/moderngpu/src/mgpucontext.cu ../externals/moderngpu/src/mgpuutil.cpp $(NVCCFLAGS) $(ARCH) $(INC) -lcuda -O3

#-------------------------------------------------------------------------------
# Clean
#-------------------------------------------------------------------------------

clean :
	rm -f bin/*_$(NVCC_VERSION)_$(ARCH_SUFFIX)*
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o
