cd ..
rm -fr build_gtest
mkdir build_gtest
cd build_gtest
#CC=clang-6.0 CXX=clang++-6.0 cmake -DGUNROCK_BUILD_TESTS=ON -DGUNROCK_GOOGLE_TESTS=ON .. && make -j$(nproc)
cmake .. && make -j$(nproc) 2>&1 |tee build.log
ctest -V  2>&1 |tee ctest.log