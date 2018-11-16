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
use_metis = 1
use_boost = 0
NVCC = "$(shell which nvcc)"
NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'))

KERNELS =

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])

#-------------------------------------------------------------------------------
# Gen targets
#-------------------------------------------------------------------------------

# GEN_SM71 = -gencode=arch=compute_71,code=\"sm_71,compute_71\"
GEN_SM70 = -gencode=arch=compute_70,code=\"sm_70,compute_70\"
# GEN_SM61 = -gencode=arch=compute_61,code=\"sm_61,compute_61\"
GEN_SM60 = -gencode=arch=compute_60,code=\"sm_60,compute_60\"
# GEN_SM52 = -gencode=arch=compute_52,code=\"sm_52,compute_52\"
# GEN_SM50 = -gencode=arch=compute_50,code=\"sm_50,compute_50\"
# GEN_SM37 = -gencode=arch=compute_37,code=\"sm_37,compute_37\"
# GEN_SM35 = -gencode=arch=compute_35,code=\"sm_35,compute_35\"
# GEN_SM30 = -gencode=arch=compute_30,code=\"sm_30,compute_30\"

# SM_TARGETS = #$(GEN_SM30) #$(GEN_SM35) $(GEN_SM30) $(GEN_SM60) $(GEN_SM61)
SM_TARGETS = $(GEN_SM70) $(GEN_SM60)
#-------------------------------------------------------------------------------
# Libs
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Includes
#-------------------------------------------------------------------------------

CUDA_INC = -I"$(shell dirname $(NVCC))/../include"
MGPU_INC = -I"../../externals/moderngpu/include"
CUB_INC = -I"../../externals/cub"

BOOST_INC =
BOOST_LINK =
ifeq ($(use_boost), 1)
    BOOST_INC = -I"/usr/local/include"
    BOOST_LINK = -Xcompiler -DBOOST_FOUND -L"/usr/local/lib" -Xlinker -lboost_system -Xlinker -lboost_chrono -Xlinker -lboost_timer -Xlinker -lboost_filesystem
else
    BOOST_INC = -I"../../externals/rapidjson/include"
endif

ifeq (DARWIN, $(findstring DARWIN, $(OSUPPER)))
    OMP_INC  = -I"/usr/local/include/libiomp"
    OMP_LINK = -Xlinker /usr/local/lib/libiomp5.dylib
else
    OMP_INC =
    OMP_LINK = -Xcompiler -fopenmp -Xlinker -lgomp
endif

ifeq (DARWIN, $(findstring DARWIN, $(OSUPPER)))
    use_metis = 0
endif

ifneq ($(use_metis), 1)
	METIS_LINK =
else
	METIS_LINK = -Xlinker -lmetis -Xcompiler -DMETIS_FOUND
endif
GUNROCK_DEF = -Xcompiler -DGUNROCKVERSION=0.4.0
LINK = $(BOOST_LINK) $(OMP_LINK) $(METIS_LINK) $(GUNROCK_DEF)
INC = $(CUDA_INC) $(OMP_INC) $(MGPU_INC) $(CUB_INC) $(BOOST_INC) -I.. -I../.. $(LINK)

#-------------------------------------------------------------------------------
# Defines
#-------------------------------------------------------------------------------

DEFINES = -DGIT_SHA1="\"$(shell git rev-parse HEAD)\""

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

NVCCFLAGS = -lineinfo --std=c++11 --expt-extended-lambda -rdc=true #-ccbin=g++-4.8

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
EXTRA_SOURCE_ = ../../gunrock/util/str_to_T.cu \
	../../gunrock/util/test_utils.cu \
	../../gunrock/util/error_utils.cu \
	../../externals/moderngpu/src/mgpucontext.cu \
	../../externals/moderngpu/src/mgpuutil.cpp \
	../../gunrock/util/gitsha1make.c

ifeq (DARWIN, $(findstring DARWIN, $(OSUPPER)))
    EXTRA_SOURCE = $(EXTRA_SOURCE_) \
	    ../../gunrock/util/misc_utils.cu
else
    EXTRA_SOURCE = $(EXTRA_SOURCE_)
endif

DEPS = ./Makefile \
    ../BaseMakefile.mk \
    $(EXTRA_SOURCE) \
    $(wildcard ../../gunrock/util/*.cuh) \
    $(wildcard ../../gunrock/util/**/*.cuh) \
    $(wildcard ../../gunrock/util/*.c) \
    $(wildcard ../../gunrock/*.cuh) \
    $(wildcard ../../gunrock/graph/*.cuh) \
    $(wildcard ../../gunrock/graphio/*.cuh) \
    $(wildcard ../../gunrock/oprtr/*.cuh) \
    $(wildcard ../../gunrock/oprtr/**/*.cuh) \
    $(wildcard ../../gunrock/app/*.cuh) \
    $(wildcard ../../gunrock/app/**/*.cuh) \
    $(wildcard ../../gunrock/partitioner/*.cuh)

#-------------------------------------------------------------------------------
# (make test) Test driver for
#-------------------------------------------------------------------------------
# leave to indivual algos

#-------------------------------------------------------------------------------
# Clean
#-------------------------------------------------------------------------------

clean :
	rm -f bin/*_$(NVCC_VERSION)_$(ARCH_SUFFIX)*
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o
