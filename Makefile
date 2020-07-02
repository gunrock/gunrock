all: test
CUDA_PATH := /usr/local/cuda-10.1
NVCC := $(CUDA_PATH)/bin/nvcc
GUNROCK_DIR := .
GUNROCK_LIB := $(GUNROCK_DIR)/build/lib/
INCLUDES := -I $(GUNROCK_DIR) -I $(CUDA_PATH)/include
NVCC_OPTIONS := -g -G -gencode arch=compute_70,code=sm_70 --expt-extended-lambda -std=c++11
OBJNVCC := $(NVCC) $(NVCC_OPTIONS) -c
main.o: main.cu
	$(NVCC) $(NVCC_OPTIONS) $(INCLUDES) -c main.cu -o main.o
test: main.o
	$(NVCC)  $(NVCC_OPTIONS) -o test main.o  $(INCLUDES) -L$(GUNROCK_LIB) -lgunrock
clean:
	rm -f test
