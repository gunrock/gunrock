FROM nvidia/cuda:9.0-devel-ubuntu16.04

########################
# Install Dependencies #
########################

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake git libboost-all-dev && rm -rf /var/lib/apt/lists/*

#################
# Build Gunrock #
#################

RUN git clone --recursive https://github.com/gunrock/gunrock.git && \
    cd gunrock && mkdir build && cd build && cmake .. && make -j$(nproc)
