# Simple Makefile to handle release/debug modes
# As well as other CMake command line args
# (which are hard to type and remember)

all: release

release:
	mkdir -p build_release
	cd build_release && cmake -DCMAKE_BUILD_TYPE=RELEASE ..
	$(MAKE) -C ./build_release

debug:
	mkdir -p build_debug
	cd build_debug && cmake -DCMAKE_BUILD_TYPE=DEBUG ..
	$(MAKE) -C ./build_debug

test: release
	./build_release/bin/unittests

test_release: test

test_debug: debug
	./build_debug/bin/unittests

clean:
	rm -rf build_debug
	rm -rf build_release
	rm -rf externals
