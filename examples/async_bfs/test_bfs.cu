#include <iostream>
#include <string>
#include "chrono"

#include "../util/time.cuh"

#include "bfs.cuh"
#include "validation_bfs.cuh"

using namespace std;
using namespace std::chrono;

template<typename VertexId, typename SizeT, typename QueueT>
__global__ void warmup_mallocManaged(BFS<VertexId, SizeT, QueueT> bfs, uint32_t *out) {
    
    uint32_t sum = 0;
    for(int i = TID; i < bfs.nodes + 1; i = i + gridDim.x * blockDim.x)
        sum += bfs.csr_offset[i];
    for(int i = TID; i < bfs.edges; i = i + gridDim.x * blockDim.x)
        sum += bfs.csr_indices[i];
    
    out[TID] = sum;
}

void print_help() {
    cout << "./test -f <file> -s <file vertex ID start from 0?=false> -i <min iteration for queue=2500> -r <source node to start=0> -q <number of queues used=4> -d <device id=0>\n";
}

int main(int argc, char *argv[]) {
    
    // --
    // CLI
        
    char *input_file   = NULL;
    uint32_t min_iter  = 2500;
    int source         = 0;
    int num_queue      = 1;
    int device         = 0;
    int rounds         = 10;
     
    if(argc == 1) {
        print_help();
        exit(0);
    }
     
    if(argc > 1) {
        for(int i=1; i<argc; i++) {
            if(string(argv[i]) == "-f")
                input_file = argv[i+1];
            else if(string(argv[i]) == "-i")
                min_iter = stoi(argv[i+1]);
            else if(string(argv[i]) == "-r")
                source = stoi(argv[i+1]);
            else if(string(argv[i]) == "-q")
                num_queue = stoi(argv[i+1]);
            else if(string(argv[i]) == "-d")
                device = stoi(argv[i+1]);
            else if(string(argv[i]) == "-rounds")
                rounds = stoi(argv[i+1]);
         }
    }
     
     if(input_file == NULL) {
         cout << "input file is needed\n";
         print_help();
         exit(0);
     }

    // --
    // Warmup (??)
    
    CUDA_CHECK(cudaSetDevice(device));

    int numBlock  = 56 * 5;
    int numThread = 256;
    cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)MaxCountQueue::_launch_thread<int, uint32_t, BFSThread<int,int>, BFS<int, int>>);

    // --
    // IO
    std::string str_file(input_file);
    Csr<int, int> csr;
    csr.load(input_file);

    std::cout << "==============================" << std::endl;
    std::cout << "nodes  = " << csr.nodes         << std::endl;
    std::cout << "edges  = " << csr.edges         << std::endl;
    std::cout << "source = " << source            << std::endl;
    std::cout << "==============================" << std::endl;
    
    // --
    // Run
    
    BFS<int, int> bfs(csr, min_iter, num_queue);
    
    uint32_t *warmup_out;
    CUDA_CHECK(cudaMalloc(&warmup_out, sizeof(uint32_t) * numBlock * numThread));
    warmup_mallocManaged<<<numBlock, numThread>>>(bfs, warmup_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float elapsed = 0.0;
    GpuTimer timer;
    for(int iteration = 0; iteration < 1; iteration++) {
        
        bfs.reset(); // Reset counters
        bfs.init(source, numBlock, numThread);

        timer.Start();
        bfs.start_threadPerItem(numBlock, numThread);
        timer.Stop();
        
        elapsed += timer.ElapsedMillis();
        std::cout << "Time: " << timer.ElapsedMillis() << std::endl;
    }

    std::cout << "Avg. Time: " << elapsed/rounds << std::endl;
    
    // --
    // Validate
    
    auto t1 = high_resolution_clock::now();
    host::BFSValid<int, int>(csr, bfs, source);
    long long val_elapsed = duration_cast<microseconds>(high_resolution_clock::now() - t1).count();
    std::cout << "Val. Time: " << (float)val_elapsed / 1e3 << std::endl;
    
    // --
    // Clean up
    
    csr.release();
    bfs.release();
    CUDA_CHECK(cudaFree(warmup_out));

    return 0;
}
