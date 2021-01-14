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
  result_t(edge_t* _depth) : depth(_depth) {}
};

template<typename graph_type, typename param_type, typename result_type>
struct problem_t {
  using vertex_t = typename graph_type::vertex_type;
  using edge_t   = typename graph_type::edge_type;
  using weight_t = typename graph_type::weight_type;
  
  graph_type graph_slice;
  param_type param;
  result_type result;
  
  problem_t(
      graph_type&  G,
      param_type&  _param,
      result_type& _result
  ) : graph_slice(G), param(_param), result(_result) {}
  
  auto get_graph() {return graph_slice;}
  
  void init() {}
  
  void reset() {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();
    
    auto single_source = param.single_source;
    auto d_depth       = thrust::device_pointer_cast(this->result.depth);
    thrust::fill(thrust::device, d_depth + 0, d_depth + n_vertices, n_vertices + 1);
    thrust::fill(thrust::device, d_depth + single_source, d_depth + single_source + 1, 0);
  }
};

// --
// Enactor

template<typename queue_t, typename val_t>
__global__ void _push_one(queue_t q, val_t val) {
    if(LANE_ == 0) q.push(val);
}

template<typename problem_t, typename single_queue_t=uint32_t>
struct enactor_t {
    using vertex_t   = typename problem_t::vertex_t;
    using edge_t     = typename problem_t::edge_t;
    using queue_t = MaxCountQueue::Queues<vertex_t, single_queue_t>;
    
    problem_t* problem;
    queue_t q;
    
    int numBlock  = 56 * 5;
    int numThread = 256;

    enactor_t(
      problem_t* _problem,
      uint32_t  min_iter=800, 
      int       num_queue=4
    ) : problem(_problem) { 
        
        auto n_vertices = problem->get_graph().get_number_of_vertices();
        
        q.init(
            min(
                single_queue_t(1 << 30), 
                max(
                    single_queue_t(1024), 
                    single_queue_t(n_vertices * 1.5)
                )
            ),
            num_queue,
            min_iter
        );
        
        q.reset();
    }

    void prepare_frontier() {
      _push_one<<<1, 32>>>(q, problem->param.single_source);
    }
    
    void enact() {
      
      prepare_frontier();
      
      // <user-defined>
      auto G        = problem->get_graph();
      edge_t* depth = problem->result.depth;
      
      auto kernel = [G, depth] __device__ (vertex_t node, queue_t q) -> void {
          
          vertex_t d = ((volatile vertex_t * )depth)[node];
          
          const vertex_t start  = G.get_starting_edge(node);
          const vertex_t degree = G.get_number_of_neighbors(node);
          
          for(int idx = 0; idx < degree; idx++) {
              vertex_t neib  = G.get_destination_vertex(start + idx);
              vertex_t old_d = atomicMin(depth + neib, d + 1);
              if(old_d > d + 1) {
                  q.push(neib);
              }
          }
      };
      // </user-defined>
      
      q.launch_thread(numBlock, numThread, kernel);
      q.sync();
  }
}; // struct enactor_t

template <typename graph_type>
float run(graph_type& G,
          typename graph_type::vertex_type& single_source,  // Parameter
          typename graph_type::weight_type* depth           // Output
) {
  
  // <user-defined>
  using vertex_t = typename graph_type::vertex_type;
  using edge_t   = typename graph_type::edge_type;

  using param_type   = param_t<vertex_t>;
  using result_type  = result_t<edge_t>;
  
  param_type param(single_source);
  result_type result(depth);
  // </user-defined>
  
  // <boiler-plate>
  using problem_type = problem_t<graph_type, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;
  
  problem_type problem(G, param, result);
  problem.init();
  problem.reset();
  
  enactor_type enactor(&problem);
  
  GpuTimer timer;
  timer.Start();
  enactor.enact();
  timer.Stop();
  return timer.ElapsedMillis();
  // </boiler-plate>
}

} // namespace bfs
} // namespace async


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
  
  printf("io\n");
  
  std::string filename = argument_array[1];

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  format::csr_t<memory::memory_space_t::device, vertex_t, edge_t, weight_t> csr;
  csr.from_coo(mm.load(filename));

  // --
  // Build graph
  
  printf("build graph\n");
  
  auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      csr.number_of_rows,
      csr.number_of_columns,
      csr.number_of_nonzeros,
      csr.row_offsets.data().get(),
      csr.column_indices.data().get(),
      csr.nonzero_values.data().get()
  );
  
  // --
  // Params and memory allocation
  
  vertex_t single_source = 0;
  vertex_t n_vertices    = G.get_number_of_vertices();
  thrust::device_vector<vertex_t> depth(n_vertices);
  
  // --
  // Run problem
  
  printf("run\n");
  
  float elapsed = async::bfs::run(G, single_source, depth.data().get());
  
  // --
  // Log + Validate
  
  thrust::host_vector<edge_t> h_depth = depth;
  
  edge_t acc = 0;
  for(vertex_t i = 0 ; i < n_vertices; i++) acc += h_depth[i];
  
  printf("\n");
  printf("elapsed=%f\n", elapsed);
  printf("acc=%d\n", acc);
}

int main(int argc, char** argv) {
  test_async_bfs(argc, argv);
  return EXIT_SUCCESS;
}
