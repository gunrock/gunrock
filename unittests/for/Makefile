include ../Makefile.inc

APP = for

test: bin/$(APP)

bin/$(APP) : test_$(APP).cu $(DEPS)
	mkdir -p bin
	$(NVCC) -ccbin=${CXX} ${NVCCFLAGS} ${NVCCOPT} --compiler-options "${CXXFLAGS} ${CXXOPT}" -o bin/$(APP) test_$(APP).cu $(SOURCE) $(ARCH) $(INC)

debug : test_$(APP).cu $(DEPS)
	mkdir -p bin
	$(NVCC) -ccbin=${CXX} ${NVCCFLAGS} ${NVCCDEBUG} --compiler-options "${CXXFLAGS} ${CXXDEBUG}" -o bin/$(APP) test_$(APP).cu $(SOURCE) $(ARCH) $(INC)

.DEFAULT_GOAL := test