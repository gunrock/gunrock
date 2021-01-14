#include <cstdlib>  // EXIT_SUCCESS

#include <gunrock/applications/sssp.hxx>
#include "./queue.cuh"
#include "./util/time.cuh"

using namespace gunrock;
using namespace memory;

namespace async {
  namespace bfs {
    template <typename vertex_t>
    struct param_t {
      vertex_t single_source;
      param_t(vertex_t _single_source) : single_source(_single_source) {}
    };

    template <typename edge_t>
    struct result_t {
      edge_t* depth;
      result_t(edge_t* _depth)
          : depth(_depth) {}
    };

    template<typename problem_t>
    __global__ void _problem_init_depth(problem_t problem) {
        auto n_vertices = problem.G.get_number_of_vertices();
        for(int i = TID; i < n_vertices; i = i + gridDim.x * blockDim.x)
            problem.result.depth[i] = n_vertices + 1;
    }

    template<typename graph_type, typename _vertex_t, typename _edge_t, typename param_type, typename result_type>
    struct problem_t {
      using vertex_t = _vertex_t;
      using edge_t   = _edge_t;
      
      graph_type G;
      param_type param;
      result_type result;
      
      problem_t(
          graph_type&  _G,
          param_type&  _param,
          result_type& _result
      ) : G(_G), param(_param), result(_result) {}
      
      void init() {}
      
      void reset() {
        auto n_vertices = G.get_number_of_vertices();
        CUDA_CHECK(cudaMemset(result.depth, (n_vertices + 1), sizeof(vertex_t) * n_vertices));
        _problem_init_depth<<<320, 512>>>(*this);
      }
      
    };

    // --
    // Enactor

    template<typename enactor_t, typename problem_t>
    __global__ void _enactor_push_source(
      enactor_t enactor, 
      problem_t problem
    ) {
        if(LANE_ == 0) {
          auto single_source = problem.param.single_source;
          enactor.worklists.push(single_source);
          problem.result.depth[single_source] = 0;
        }
    }

    template<typename problem_t, typename queue_t=uint32_t>
    struct enactor_t {
        using vertex_t   = typename problem_t::vertex_t;
        using edge_t     = typename problem_t::edge_t;
        using worklist_t = MaxCountQueue::Queues<vertex_t, queue_t>;
        
        problem_t problem;
        worklist_t worklists;
        
        int numBlock  = 56 * 5;
        int numThread = 256;

        enactor_t(
          problem_t _problem,
          uint32_t  min_iter=800, 
          int       num_queue=4
        ) : problem(_problem) { 
          
            worklists.init(
                min(
                    queue_t(1 << 30), 
                    max(
                        queue_t(1024), 
                        queue_t(problem.G.get_number_of_vertices() * 1.5)
                    )
                ),
                num_queue,
                min_iter
            );
            
            worklists.reset();
        }

        void prepare_frontier() {
          _enactor_push_source<<<1, 32>>>(*this, problem);
        }
        
        void enact() {
          
          prepare_frontier();
          
          auto G        = problem.G;
          edge_t* depth = problem.result.depth;
          
          auto bfs_kernel = [G, depth] __device__ (vertex_t node, worklist_t worklists) -> void {
              
              vertex_t d = ((volatile vertex_t * )depth)[node];
              
              const vertex_t start = G.get_starting_edge(node);
              const vertex_t end   = start + G.get_number_of_neighbors(node);
              
              for(int idx = start; idx < end; idx++) {
                  vertex_t neib  = G.get_destination_vertex(idx);
                  vertex_t old_d = atomicMin(depth + neib, d + 1);
                  if(old_d > d + 1) {
                      worklists.push(neib);
                  }
              }
              
          };
          
          worklists.launch_thread(numBlock, numThread, bfs_kernel, worklists);
          worklists.sync();
      }
    }; // struct enactor_t
    
  } // namespace bfs
} // namespace bfs


void test_async_bfs(int num_arguments, char** argument_array) {
  if (num_arguments != 2) {
    std::cerr << "usage: ./bin/<program-name> filename.mtx" << std::endl;
    exit(1);
  }

  // --
  // Define types

  using vertex_t = int;
  using edge_t   = int;
  using weight_t = int;

  // --
  // IO

  std::string filename = argument_array[1];

  // >> Replace w/ binary loading for speed
  // io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  // format::csr_t<memory::memory_space_t::device, vertex_t, edge_t, weight_t> csr;
  // csr.from_coo(mm.load(filename));
  
  // vertex_t  n_nodes        = csr.number_of_rows;
  // edge_t    n_edges        = csr.number_of_nonzeros;
  // edge_t*   row_offsets    = csr.row_offsets.data().get();
  // vertex_t* column_indices = csr.column_indices.data().get();
  // --
  vertex_t  n_nodes;
  edge_t    n_edges;
  edge_t*   row_offsets;
  vertex_t* column_indices;
  edge_t*   h_row_offsets;
  vertex_t* h_column_indices;
  
  std::ifstream fin(filename);
  if(fin.is_open()) {
      fin.read((char *)&n_nodes, sizeof(vertex_t));
      fin.read((char *)&n_edges, sizeof(edge_t));
      
      h_row_offsets      = (edge_t*)malloc(sizeof(edge_t) * (n_nodes + 1));
      h_column_indices = (vertex_t*)malloc(sizeof(vertex_t) * n_edges);
      
      CUDA_CHECK(cudaMalloc(&row_offsets, sizeof(edge_t) * (n_nodes + 1)));
      CUDA_CHECK(cudaMalloc(&column_indices, sizeof(vertex_t) * n_edges));
      
      fin.read((char *)h_row_offsets, sizeof(edge_t)*(n_nodes + 1));
      fin.read((char *)h_column_indices, sizeof(vertex_t) * n_edges);
      
      cudaMemcpy(row_offsets,    h_row_offsets, (n_nodes + 1) * sizeof(edge_t), cudaMemcpyHostToDevice);
      cudaMemcpy(column_indices, h_column_indices, n_edges * sizeof(vertex_t), cudaMemcpyHostToDevice);
      
      fin.close();
      
      CUDA_CHECK(cudaDeviceSynchronize());
  }
  // <<

 auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      n_nodes,         // rows
      n_nodes,         // columns
      n_edges,         // nonzeros
      row_offsets,     // row_offsets
      column_indices,  // column_indices
      column_indices   // dummy
  );  // supports row_indices and column_offsets (default = nullptr)
  
  using graph_t = decltype(G);
  
  vertex_t single_source = 0;
  
  vertex_t* depth;
  CUDA_CHECK(cudaMalloc(&depth, sizeof(vertex_t) * n_nodes));
  
  using param_type   = async::bfs::param_t<vertex_t>;
  using result_type  = async::bfs::result_t<edge_t>;
  
  param_type param(single_source);
  result_type result(depth);
  
  using problem_type = async::bfs::problem_t<graph_t, vertex_t, edge_t, param_type, result_type>;
  using enactor_type = async::bfs::enactor_t<problem_type>;
  
  problem_type problem(G, param, result);
  problem.init();
  problem.reset();
  
  enactor_type enactor(problem);
  
  GpuTimer timer;
  timer.Start();
  enactor.enact();
  timer.Stop();
  auto elapsed = timer.ElapsedMillis();
  
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // --
  // Validate
  
  thrust::device_vector<edge_t> d_bfs_depth(result.depth + 0, result.depth + n_nodes);
  thrust::host_vector<edge_t>   h_bfs_depth = d_bfs_depth;
  
  edge_t acc = 0;
  for(vertex_t i = 0 ; i < n_nodes; i++) {
    // printf("%d ", h_bfs_depth[i]);
    acc += h_bfs_depth[i];
  }
  
  printf("\n");
  printf("elapsed=%f\n", elapsed);
  printf("acc=%d\n", acc);
}

int main(int argc, char** argv) {
  test_async_bfs(argc, argv);
  return EXIT_SUCCESS;
}
