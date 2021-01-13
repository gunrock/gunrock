#include <cstdlib>  // EXIT_SUCCESS

#include <gunrock/applications/sssp.hxx>
#include "./queue.cuh"

using namespace gunrock;
using namespace memory;

template<typename _vertex_t, typename _edge_t>
struct bfs_problem {
  using vertex_t = _vertex_t;
  using edge_t   = _edge_t;

  vertex_t nodes;
  edge_t   edges;
  edge_t   *csr_offset;
  vertex_t *csr_indices;
  vertex_t *depth;
  
  bfs_problem(
      vertex_t  _nodes, 
      edge_t    _edges, 
      edge_t*   _csr_offset, 
      vertex_t* _csr_indices, 
      vertex_t* _depth
  ) {
    
    nodes       = _nodes;
    edges       = _edges;
    csr_offset  = _csr_offset;
    csr_indices = _csr_indices;
    depth       = _depth;

    CUDA_CHECK(cudaMemset(depth, (nodes+1), sizeof(vertex_t)*nodes));
  } 
};


template<typename problem_t, typename _queue_t=uint32_t>
struct bfs_enactor {
    using vertex_t = problem_t::vertex_t;
    using edge_t   = problem_t::edge_t;
    using queue_t  = _queue_t;

    MaxCountQueue::Queues<vertex_t, queue_t> worklists;

    bfs_enactor(
      problem_t& prob,
      uint32_t min_iter=800, 
      int num_queue=4
    ) { 
        worklists.init(
            min(
                queue_t(1 << 30), 
                max(
                    queue_t(1024), 
                    queue_t(prob.nodes * 1.5)
                )
            ),
            num_queue,
            min_iter
        );
    }

    void reset() {
        worklists.reset();
    }

    void init(vertex_t source, int numBlock, int numThread);
    void start_threadPerItem(int numBlock, int numThread);
};

// >>
// Init

template<typename vertex_t, typename edge_t>
__global__ void push_source(BFS<vertex_t, edge_t> bfs, vertex_t source) {
    if(LANE_ == 0) {
        bfs.worklists.push(source);
        bfs.depth[source] = 0;
    }
}

template<typename vertex_t, typename edge_t>
__global__ void init_depth(BFS<vertex_t, edge_t> bfs) {
    for(int i = TID; i < bfs.nodes; i = i + gridDim.x * blockDim.x)
        bfs.depth[i] = bfs.nodes + 1;
}

template<typename vertex_t, typename edge_t, typename queue_t>
void BFS<vertex_t, edge_t, queue_t>::init(vertex_t source, int numBlock, int numThread) {
    int gridSize  = 320;
    int blockSize = 512;
    
    init_depth<<<gridSize, blockSize>>>(*this);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    push_source<<<1, 32>>>(*this, source);
    CUDA_CHECK(cudaDeviceSynchronize());
}
// <<

// >>
// Run

template<typename vertex_t, typename edge_t, typename queue_t=uint32_t>
class BFSThread{
    public:
        __forceinline__ __device__ void operator()(vertex_t node, BFS<vertex_t, edge_t, queue_t> bfs) {
            
            vertex_t depth = ((volatile vertex_t * )bfs.depth)[node];
            
            auto start = bfs.csr_offset[node];
            auto end   = bfs.csr_offset[node + 1];
            
            for(int idx = start; idx < end; idx++) {
                vertex_t neib      = bfs.csr_indices[idx];
                vertex_t old_depth = atomicMin(bfs.depth + neib, depth + 1);
                if(old_depth > depth + 1) {
                    bfs.worklists.push(neib);
                }
            }
        }
};

template<typename problem_t>
void run(problem_t& prob, int numBlock, int numThread) {
    using vertex_t = typename problem_t::vertex_t;
    using edge_t   = typename problem_t::edge_t;
    using queue_t  = typename problem_t::queue_t;
    
    auto fn = [prob] __device__ (vertex_t node, MaxCountQueue::Queues<int, uint32_t> q) -> void {
        vertex_t depth = ((volatile vertex_t * )prob.depth)[node];
        
        auto start = prob.csr_offset[node];
        auto end   = prob.csr_offset[node + 1];
        
        for(int idx = start; idx < end; idx++) {
            vertex_t neib      = prob.csr_indices[idx];
            vertex_t old_depth = atomicMin(prob.depth + neib, depth + 1);
            if(old_depth > depth + 1) {
                q.push(neib);
            }
        }
    };
    
    prob.worklists.launch_thread(numBlock, numThread, fn, prob.worklists);
    prob.worklists.sync();
}

// <<

void test_async_bfs(int num_arguments, char** argument_array) {
  if (num_arguments != 2) {
    std::cerr << "usage: ./bin/<program-name> filename.mtx" << std::endl;
    exit(1);
  }

  // --
  // Define types

  using vertex_t = int;
  using edge_t   = int;
  using weight_t = float;

  // --
  // IO

  std::string filename = argument_array[1];

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  format::csr_t<memory::memory_space_t::device, vertex_t, edge_t, weight_t> csr;
  csr.from_coo(mm.load(filename));
  
  // --
  // Build graph

  // auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
  //     n_nodes,               // rows
  //     csr.number_of_columns,            // columns
  //     n_edges,           // nonzeros
  //     csr.row_offsets.data().get(),     // row_offsets
  //     csr.column_indices.data().get(),  // column_indices
  //     csr.nonzero_values.data().get()   // values
  // );  // supports row_indices and column_offsets (default = nullptr)
  
  int numBlock  = 56 * 5;
  int numThread = 256;
  
  vertex_t n_nodes = csr.number_of_rows;
  edge_t   n_edges = csr.number_of_nonzeros;
  
  vertex_t* depth;
  CUDA_CHECK(cudaMalloc(&depth, sizeof(vertex_t) * n_nodes));
  
  bfs_problem<vertex_t, edge_t> problem(
    n_nodes, 
    n_edges, 
    csr.row_offsets.data().get(), 
    csr.column_indices.data().get(),
    depth
  );
  
  bfs_enactor enactor(problem);
  
  // bfs.reset();
  // bfs.init(0, numBlock, numThread);
  // run(bfs, numBlock, numThread);
  
  // thrust::device_vector<edge_t> d_bfs_depth(bfs.depth + 0, bfs.depth + n_nodes);
  // thrust::host_vector<edge_t>   h_bfs_depth = d_bfs_depth;
  // for(vertex_t i = 0 ; i < n_nodes; i++) {
  //   printf("%d ", h_bfs_depth[i]);
  // }
  
  // printf("\n");
}

int main(int argc, char** argv) {
  test_async_bfs(argc, argv);
  return EXIT_SUCCESS;
}
