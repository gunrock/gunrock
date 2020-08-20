NVCC = "$(shell which nvcc)"
NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'))

#-------------------------------------------------------------------------------
# Gen targets
#-------------------------------------------------------------------------------

GEN_SM75 = -gencode=arch=compute_75,code=\"sm_75,compute_75\" # Turing RTX20XX
GEN_SM70 = -gencode=arch=compute_70,code=\"sm_70,compute_70\" # Volta V100
GEN_SM61 = -gencode=arch=compute_61,code=\"sm_61,compute_61\" # Pascal GTX10XX
GEN_SM60 = -gencode=arch=compute_60,code=\"sm_60,compute_60\" # Pascal P100
GEN_SM52 = -gencode=arch=compute_52,code=\"sm_52,compute_52\" # Maxwell M40, M60, GTX9XX
GEN_SM50 = -gencode=arch=compute_50,code=\"sm_50,compute_50\" # Maxwell M10
GEN_SM37 = -gencode=arch=compute_37,code=\"sm_37,compute_37\" # Kepler K80
GEN_SM35 = -gencode=arch=compute_35,code=\"sm_35,compute_35\" # Kepler K20, K40
GEN_SM30 = -gencode=arch=compute_30,code=\"sm_30,compute_30\" # Kepler K10

# Note: Some of the architectures don't support Gunrock's
# RepeatFor (Cooperative Groups), e.g: SM35

# Add your own SM target (default: V100, P100):
SM_TARGETS = $(GEN_SM61)

#-------------------------------------------------------------------------------
# Includes
#-------------------------------------------------------------------------------

ifeq (DARWIN, $(findstring DARWIN, $(OSUPPER)))
    OMP_INC  = -I"/usr/local/include/libiomp"
    OMP_LINK = -Xlinker /usr/local/lib/libiomp5.dylib
else
    OMP_INC =
    OMP_LINK = -Xcompiler -fopenmp -Xlinker -lgomp
endif

CUDA_INC 	= -I"$(shell dirname $(NVCC))/../include"
MGPU_INC 	= -I"../../externals/moderngpu/include"
CUB_INC 	= -I"../../externals/cub"
CUDA_CPP 	= -I"../../externals/cuda-api-wrappers/src"
R_JSON		= -I"../../externals/rapidjson/include"

GUNROCK_DEF = -Xcompiler -DGUNROCKVERSION=1.1.0
LINK = $(BOOST_LINK) $(OMP_LINK) $(METIS_LINK) $(GUNROCK_DEF)
INC = $(CUDA_CPP) $(OMP_INC) $(MGPU_INC) $(CUB_INC) $(R_JSON) $(CUDA_INC) -I.. -I../.. $(LINK)

#-------------------------------------------------------------------------------
# Defines
#-------------------------------------------------------------------------------

DEFINES 	= -DGIT_SHA1="\"$(shell git rev-parse HEAD)\""

#-------------------------------------------------------------------------------
# Compiler Flags
#-------------------------------------------------------------------------------

# ifneq ($(force64), 1)
# 	# Compile with 32-bit device pointers by default
# 	ARCH_SUFFIX = i386
# 	ARCH = -m32
# else
# 	ARCH_SUFFIX = x86_64
# 	ARCH = -m64
# endif

NVCCFLAGS = -lineinfo --std=c++11 --expt-extended-lambda -rdc=true --expt-relaxed-constexpr

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
EXTRA_SOURCE_ = ../../gunrock/util/gitsha1make.c

ifeq (DARWIN, $(findstring DARWIN, $(OSUPPER)))
    EXTRA_SOURCE = $(EXTRA_SOURCE_) \
	    ../../gunrock/util/misc_utils.cu
else
    EXTRA_SOURCE = $(EXTRA_SOURCE_)
endif

DEPS = ./Makefile \
    ../Makefile.inc \
    $(EXTRA_SOURCE) \
    $(wildcard ../../gunrock/algorithms/*.cuh) \
    $(wildcard ../../gunrock/algorithms/**/*.cuh)  \
    $(wildcard ../../gunrock/util/*.hxx) \
	$(wildcard ../../gunrock/util/*.cuh) \
    $(wildcard ../../gunrock/container/*.cuh) \
	$(wildcard ../../gunrock/framework/*.cuh)

#-------------------------------------------------------------------------------
# Clean
#-------------------------------------------------------------------------------

clean :
	rm -f bin/*_*
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o