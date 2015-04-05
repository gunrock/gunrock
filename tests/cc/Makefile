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
#NVCC = "/usr/local/cuda-6.0/bin/nvcc"
NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'))

KERNELS = 

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])

#-------------------------------------------------------------------------------
# Gen targets
#-------------------------------------------------------------------------------

GEN_SM37 = -gencode=arch=compute_37,code=\"sm_37,compute_37\" 
GEN_SM35 = -gencode=arch=compute_35,code=\"sm_35,compute_35\" 
GEN_SM30 = -gencode=arch=compute_30,code=\"sm_30,compute_30\" 
SM_TARGETS = $(GEN_SM35) $(GEN_SM37)
#$(GEN_SM35)

#-------------------------------------------------------------------------------
# Libs
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Includes
#-------------------------------------------------------------------------------

CUDA_INC = "$(shell dirname $(NVCC))/../include"
MGPU_INC = "../../../moderngpu/include"
BOOST_DIR = "../../../boost_1_53_0"
BOOST_DEPS = -Xlinker $(BOOST_DIR)/bin.v2/libs/timer/build/gcc-4.6/release/link-static/threading-multi/cpu_timer.o -Xlinker $(BOOST_DIR)/bin.v2/libs/chrono/build/gcc-4.6/release/link-static/threading-multi/chrono.o -Xlinker $(BOOST_DIR)/bin.v2/libs/system/build/gcc-4.6/release/link-static/threading-multi/libboost_system.a -Xlinker -lrt
#INC = -Xlinker "/usr/local/lib/libmetis.so" -I$(CUDA_INC) -I$(MGPU_INC) -I$(BOOST_DIR) $(BOOST_DEPS) -I.. -I../..
OMP_DEPS = -Xcompiler -fopenmp -Xlinker -lgomp
INC = -I$(CUDA_INC) -I$(MGPU_INC) -I$(BOOST_DIR) $(BOOST_DEPS) $(OMP_DEPS) -I.. -I../..

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

NVCCFLAGS = -Xptxas -v -Xcudafe -\# -lineinfo

ifeq (WIN_NT, $(findstring WIN_NT, $(OSUPPER)))
	NVCCFLAGS += -Xcompiler /bigobj -Xcompiler /Zm500
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

DEPS = 			./Makefile \
				$(wildcard ../../gunrock/util/*.cuh) \
				$(wildcard ../../gunrock/util/**/*.cuh) \
				$(wildcard ../../gunrock/*.cuh) \
				$(wildcard ../../gunrock/graphio/*.cuh) \
				$(wildcard ../../gunrock/oprtr/*.cuh) \
				$(wildcard ../../gunrock/oprtr/**/*.cuh) \
				$(wildcard ../../gunrock/app/*.cuh) \
				$(wildcard ../../gunrock/app/**/*.cuh)
				 
#-------------------------------------------------------------------------------
# (make test) Test driver for 
#-------------------------------------------------------------------------------

test: bin/test_cc_$(NVCC_VERSION)_$(ARCH_SUFFIX)

bin/test_cc_$(NVCC_VERSION)_$(ARCH_SUFFIX) : test_cc.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_cc_$(NVCC_VERSION)_$(ARCH_SUFFIX) test_cc.cu ../../../moderngpu/src/mgpucontext.cu ../../../moderngpu/src/mgpuutil.cpp $(NVCCFLAGS) $(ARCH) $(INC) -O3

#-------------------------------------------------------------------------------
# Clean
#-------------------------------------------------------------------------------

clean :
	rm -f bin/*_$(NVCC_VERSION)_$(ARCH_SUFFIX)* 
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o

