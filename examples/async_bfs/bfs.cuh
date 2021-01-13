#ifndef __BFS_H__
#define __BFS_H__

#include "../util/error_util.cuh"
#include "../util/util.cuh"
#include "../comm/csr.cuh"
#include "../queue/queue.cuh"

template<typename VertexT>
struct BFSEntry {
    VertexT node;
    VertexT dep;

    __host__ __device__ BFSEntry(){}
    __host__ __device__ BFSEntry(VertexT _node, VertexT _dep) {node = _node; dep = _dep;}
};

template<typename VertexT, typename EdgeT, typename QueueT=uint32_t>
struct BFS {
    VertexT nodes;
    EdgeT   edges;

    EdgeT   *csr_offset;
    VertexT *csr_indices;
    VertexT *depth;

    MaxCountQueue::Queues<VertexT, QueueT> worklists;

    BFS(Csr<VertexT, EdgeT> &csr, uint32_t min_iter=800, int num_queue=4) {
        nodes       = csr.nodes;
        edges       = csr.edges;
        csr_offset  = csr.row_offset;
        csr_indices = csr.column_indices;
        
        worklists.init( // Why are queue sizes limited?
            min(
                QueueT(1 << 30), 
                max(
                    QueueT(1024), 
                    QueueT(nodes * 1.5)
                )
            ),
            num_queue,
            min_iter
        );

        CUDA_CHECK(cudaMallocManaged(&depth, sizeof(VertexT)*nodes));
        CUDA_CHECK(cudaMemset(depth, (nodes+1), sizeof(VertexT)*nodes));
    }

    void release() {
        worklists.release();
        CUDA_CHECK(cudaFree(depth));
    }

    void reset() {
        worklists.reset();
    }

    void init(VertexT source, int numBlock, int numThread);
    void start_threadPerItem(int numBlock, int numThread);
};

template<typename VertexT, typename EdgeT>
__global__ void push_source(BFS<VertexT, EdgeT> bfs, VertexT source) {
    if(LANE_ == 0) {
        bfs.worklists.push(source);
        bfs.depth[source] = 0;
    }
}

template<typename VertexT, typename EdgeT>
__global__ void init_depth(BFS<VertexT, EdgeT> bfs) {
    for(int i = TID; i < bfs.nodes; i = i + gridDim.x * blockDim.x)
        bfs.depth[i] = bfs.nodes + 1;
}

template<typename VertexT, typename EdgeT, typename QueueT>
void BFS<VertexT, EdgeT, QueueT>::init(VertexT source, int numBlock, int numThread) {
    int gridSize  = 320;
    int blockSize = 512;
    
    init_depth<<<gridSize, blockSize>>>(*this);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    push_source<<<1, 32>>>(*this, source);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // MaxCountQueue::check_end<<<1, 32>>>(worklists); // What does this do? Isn't this redundent w/ the end check in the queue?
    // CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename VertexT, typename EdgeT, typename QueueT=uint32_t>
class BFSThread{
    public:
        __forceinline__ __device__ void operator()(VertexT node, BFS<VertexT, EdgeT, QueueT> bfs) {
            
            VertexT depth = ((volatile VertexT * )bfs.depth)[node];
            
            auto start = bfs.csr_offset[node];
            auto end   = bfs.csr_offset[node + 1];
            
            for(int idx = start; idx < end; idx++) {
                VertexT neib      = bfs.csr_indices[idx];
                VertexT old_depth = atomicMin(bfs.depth + neib, depth + 1);
                if(old_depth > depth + 1) {
                    bfs.worklists.push(neib);
                }
            }
        }
};

template<typename VertexT, typename EdgeT, typename QueueT>
void BFS<VertexT, EdgeT, QueueT>::start_threadPerItem(int numBlock, int numThread) {
    worklists.launch_thread(numBlock, numThread, BFSThread<VertexT, EdgeT>(), *this);
    worklists.sync();
}

#endif
