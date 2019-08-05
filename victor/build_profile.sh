cd ..
rm -fr build_profile
mkdir build_profile
cd build_profile
#CC=clang-6.0 CXX=clang++-6.0 cmake -DGUNROCK_BUILD_TESTS=ON -DGUNROCK_GOOGLE_TESTS=ON .. && make -j$(nproc)
CFLAG="-pg" CXXFLAG="-pg" cmake -DGUNROCK_BUILD_TESTS=ON -DCMAKE_EXPORT_COMPILE_COMMAND=YES .. && make -j$(nproc)