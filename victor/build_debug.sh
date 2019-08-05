cd ..
rm -fr build_debug
mkdir build_debug
cd build_debug
cmake -DCMAKE_BUILD_TYPE=Debug -DGUNROCK_BUILD_TESTS=ON -DGUNROCK_GOOGLE_TESTS=ON .. && make -j$(nproc)