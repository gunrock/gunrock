include ../Makefile.inc

APP = color

test: bin/$(APP)

bin/$(APP) : $(APP).cu $(DEPS)
	mkdir -p bin
	$(NVCC) -ccbin=${CXX} ${NVCCFLAGS} ${NVCCOPT} --compiler-options "${CXXFLAGS} ${CXXOPT}" -o bin/$(APP) $(APP).cu $(SOURCE) $(ARCH) $(INC)

debug : $(APP).cu $(DEPS)
	mkdir -p bin
	$(NVCC) -ccbin=${CXX} ${NVCCFLAGS} ${NVCCDEBUG} --compiler-options "${CXXFLAGS} ${CXXDEBUG}" -o bin/$(APP) $(APP).cu $(SOURCE) $(ARCH) $(INC)

.DEFAULT_GOAL := test