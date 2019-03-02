// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_astar.cu
 *
 * @brief Simple test driver program for single source shortest path.
 */

// Handle graph reading task (original graph should be the same,
// but add mapping to names and longitude,latitude tuple)
//
// Refactor current BGL code into a function called RefAStar
//
// Add GPU Kernel driver code
//
// test on 1k city list
// test on 42k city list

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <iostream>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// A* includes
#include <gunrock/app/astar/astar_enactor.cuh>
#include <gunrock/app/astar/astar_problem.cuh>
#include <gunrock/app/astar/astar_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/priority_queue/kernel.cuh>

#include <moderngpu.cuh>

// Boost includes for CPU A* Search Algorithm

#include <boost/graph/astar_search.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/random.hpp>
#include <boost/random.hpp>
#include <boost/graph/graphviz.hpp>

using namespace boost;
using namespace gunrock;
using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::astar;
using namespace std;

void Usage() {
  printf(
      "test <graph-type> [graph-type-arguments]\n"
      "Graph type and graph type arguments:\n"
      "    market <matrix-market-file-name>\n"
      "        Reads a Matrix-Market coordinate-formatted graph of\n"
      "        directed/undirected edges from STDIN (or from the\n"
      "        optionally-specified file).\n"
      "    rmat (default: rmat_scale = 10, a = 0.57, b = c = 0.19)\n"
      "        Generate R-MAT graph as input\n"
      "        --rmat_scale=<vertex-scale>\n"
      "        --rmat_nodes=<number-nodes>\n"
      "        --rmat_edgefactor=<edge-factor>\n"
      "        --rmat_edges=<number-edges>\n"
      "        --rmat_a=<factor> --rmat_b=<factor> --rmat_c=<factor>\n"
      "        --rmat_seed=<seed>\n"
      "    rgg (default: rgg_scale = 10, rgg_thfactor = 0.55)\n"
      "        Generate Random Geometry Graph as input\n"
      "        --rgg_scale=<vertex-scale>\n"
      "        --rgg_nodes=<number-nodes>\n"
      "        --rgg_thfactor=<threshold-factor>\n"
      "        --rgg_threshold=<threshold>\n"
      "        --rgg_vmultipiler=<vmultipiler>\n"
      "        --rgg_seed=<seed>\n\n"
      "Optional arguments:\n"
      "[--device=<device_index>] Set GPU(s) for testing (Default: 0).\n"
      "[--undirected]            Treat the graph as undirected (symmetric).\n"
      "[--instrumented]          Keep kernels statics [Default: Disable].\n"
      "                          total_queued, search_depth and barrier duty.\n"
      "                          (a relative indicator of load imbalance.)\n"
      "[--src=<Vertex-ID|randomize|largestdegree>]\n"
      "                          Begins traversal from the source (Default: "
      "0).\n"
      "                          If randomize: from a random source vertex.\n"
      "                          If largestdegree: from largest degree "
      "vertex.\n"
      "[--dst-node=<Vertex-ID>]\n"
      "[--mapfile=filename]\n"
      "[--quick]                 Skip the CPU reference validation process.\n"
      "[--mark-pred]             Keep both label info and predecessor info.\n"
      "[--disable-size-check]    Disable frontier queue size check.\n"
      "[--grid-size=<grid size>] Maximum allowed grid size setting.\n"
      "[--queue-sizing=<factor>] Allocates a frontier queue sized at: \n"
      "                          (graph-edges * <factor>). (Default: 1.0)\n"
      "[--in-sizing=<in/out_queue_scale_factor>]\n"
      "                          Allocates a frontier queue sized at: \n"
      "                          (graph-edges * <factor>). (Default: 1.0)\n"
      "[--v]                     Print verbose per iteration debug info.\n"
      "[--iteration-num=<num>]   Number of runs to perform the test.\n"
      "[--traversal-mode=<0|1>]  Set traversal strategy, 0 for Load-Balanced\n"
      "                          1 for Dynamic-Cooperative (Default: dynamic\n"
      "                          determine based on average degree).\n"
      "[--partition-method=<random|biasrandom|clustered|metis>]\n"
      "                          Choose partitioner (Default use random).\n"
      "[--delta_factor=<factor>] Delta factor for delta-stepping SSSP.\n"
      "[--quiet]                 No output (unless --json is specified).\n"
      "[--json]                  Output JSON-format statistics to STDOUT.\n"
      "[--jsonfile=<name>]       Output JSON-format statistics to file <name>\n"
      "[--jsondir=<dir>]         Output JSON-format statistics to <dir>/name,\n"
      "                          where name is auto-generated.\n");
}

/**
 * @brief Displays the SSSP result (i.e., distance from source)
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] source_path Search depth from the source for each node.
 * @param[in] num_nodes Number of nodes in the graph.
 */
template <typename VertexId, typename SizeT>
void DisplaySolution(VertexId *source_path, std::string *names, VertexId dst,
                     SizeT num_nodes) {
  cout << names[dst] << "->";
  VertexId next = source_path[dst];
  while (next != -1) {
    cout << names[next] << "->";
    next = source_path[next];
  }
  cout << endl;
}

// auxiliary types
struct location {
  float y, x;
};

// euclidean distance heuristic
template <class Graph, class CostType, class LocMap>
class distance_heuristic : public astar_heuristic<Graph, CostType> {
 public:
  typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
  distance_heuristic(LocMap l, Vertex goal) : m_location(l), m_goal(goal) {}
  CostType operator()(Vertex u) {
    CostType dx = m_location[m_goal].x - m_location[u].x;
    CostType dy = m_location[m_goal].y - m_location[u].y;
    return ::sqrt(dx * dx + dy * dy);
  }

 private:
  LocMap m_location;
  Vertex m_goal;
};

struct found_goal {};  // exception for termination

// visitor that terminates when we find the goal
template <class Vertex>
class astar_goal_visitor : public boost::default_astar_visitor {
 public:
  astar_goal_visitor(Vertex goal) : m_goal(goal) {}
  template <class Graph>
  void examine_vertex(Vertex u, Graph &g) {
    if (u == m_goal) throw found_goal();
  }

 private:
  Vertex m_goal;
};

/******************************************************************************
 * A* Testing Routines
 *****************************************************************************/

/**
 * @brief A simple CPU-based reference A* path finding implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam DIRECTED
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] node_values Host-side vector to store CPU computed labels for each
 * node
 * @param[in] node_preds Host-side vector to store CPU computed predecessors for
 * each node
 * @param[in] src Source node where SSSP starts
 * @param[in] quiet Don't print out anything to stdout
 */
template <typename VertexId, typename SizeT, typename Value>
void ReferenceAStarDirected(const Csr<VertexId, SizeT, Value> &graph,
                            std::string *names, Value *lats, Value *lons,
                            VertexId *h_preds, VertexId src, VertexId dst) {
  // specify some types
  typedef adjacency_list<listS, vecS, directedS, no_property,
                         property<edge_weight_t, float> >
      mygraph_t;
  typedef property_map<mygraph_t, edge_weight_t>::type WeightMap;
  typedef mygraph_t::vertex_descriptor vertex;
  typedef mygraph_t::edge_descriptor edge_descriptor;
  typedef std::pair<int, int> edge;

  location *locations = new location[graph.nodes];
  for (int i = 0; i < graph.nodes; ++i) {
    locations[i].x = lats[i];
    locations[i].y = lons[i];
  }

  edge *edge_array = new edge[graph.edges];
  Value *weights = new Value[graph.edges];
  SizeT idx = 0;

  for (SizeT node = 0; node < graph.nodes; node++) {
    for (SizeT e = graph.row_offsets[node]; e < graph.row_offsets[node + 1];
         e++) {
      edge_array[idx] = edge(node, graph.column_indices[e]);
      weights[idx++] = graph.edge_values[e];
    }
  }
  unsigned int num_edges = graph.edges;

  // create graph
  mygraph_t g(graph.nodes);
  WeightMap weightmap = get(edge_weight, g);
  for (std::size_t j = 0; j < num_edges; ++j) {
    edge_descriptor e;
    bool inserted;
    boost::tie(e, inserted) =
        add_edge(edge_array[j].first, edge_array[j].second, g);
    weightmap[e] = weights[j];
  }

  // cout << "Start vertex: " << names[src] << endl;
  // cout << "Goal vertex: " << names[dst] << endl;

  vector<mygraph_t::vertex_descriptor> p(num_vertices(g));
  vector<float> d(num_vertices(g));
  try {
    // call astar named parameter interface
    astar_search_tree(
        g, src,
        distance_heuristic<mygraph_t, float, location *>(locations, dst),
        predecessor_map(
            make_iterator_property_map(p.begin(), get(vertex_index, g)))
            .distance_map(
                make_iterator_property_map(d.begin(), get(vertex_index, g)))
            .visitor(astar_goal_visitor<vertex>(dst)));

  } catch (found_goal fg) {  // found a path to the goal
    list<vertex> shortest_path;
    int idx = 0;
    for (vertex v = dst;; v = p[v]) {
      shortest_path.push_back(v);
      h_preds[idx] = v;
      ++idx;
      if (p[v] == v) break;
    }

    /*cout << "Shortest path from " << names[src] << " to "
         << names[dst] << ": ";
    list<vertex>::iterator spi = shortest_path.begin();
    cout << names[src];
    for(++spi; spi != shortest_path.end(); ++spi)
      cout << " -> " << names[*spi];
    cout << endl << "Total travel time: " << d[dst] << endl;*/
  }

  delete[] weights;
  delete[] edge_array;
  delete[] locations;
}

template <typename VertexId, typename SizeT, typename Value>
void ReferenceAStarUndirected(const Csr<VertexId, SizeT, Value> &graph,
                              std::string *names, Value *lats, Value *lons,
                              VertexId *h_preds, VertexId src, VertexId dst) {
  // specify some types
  typedef adjacency_list<listS, vecS, undirectedS, no_property,
                         property<edge_weight_t, float> >
      mygraph_t;
  typedef property_map<mygraph_t, edge_weight_t>::type WeightMap;
  typedef mygraph_t::vertex_descriptor vertex;
  typedef mygraph_t::edge_descriptor edge_descriptor;
  typedef std::pair<int, int> edge;

  location *locations = new location[graph.nodes];
  for (int i = 0; i < graph.nodes; ++i) {
    locations[i].x = lats[i];
    locations[i].y = lons[i];
  }

  edge *edge_array = new edge[graph.edges];
  Value *weights = new Value[graph.edges];
  SizeT idx = 0;

  for (SizeT node = 0; node < graph.nodes; node++) {
    for (SizeT e = graph.row_offsets[node]; e < graph.row_offsets[node + 1];
         e++) {
      edge_array[idx] = edge(node, graph.column_indices[e]);
      weights[idx++] = graph.edge_values[e];
    }
  }
  unsigned int num_edges = graph.edges;

  // create graph
  mygraph_t g(graph.nodes);
  WeightMap weightmap = get(edge_weight, g);
  for (std::size_t j = 0; j < num_edges; ++j) {
    edge_descriptor e;
    bool inserted;
    boost::tie(e, inserted) =
        add_edge(edge_array[j].first, edge_array[j].second, g);
    weightmap[e] = weights[j];
  }

  // cout << "Start vertex: " << names[src] << endl;
  // cout << "Goal vertex: " << names[dst] << endl;

  vector<mygraph_t::vertex_descriptor> p(num_vertices(g));
  vector<float> d(num_vertices(g));
  try {
    // call astar named parameter interface
    astar_search_tree(
        g, src,
        distance_heuristic<mygraph_t, float, location *>(locations, dst),
        predecessor_map(
            make_iterator_property_map(p.begin(), get(vertex_index, g)))
            .distance_map(
                make_iterator_property_map(d.begin(), get(vertex_index, g)))
            .visitor(astar_goal_visitor<vertex>(dst)));

  } catch (found_goal fg) {  // found a path to the goal
    list<vertex> shortest_path;
    int idx = 0;
    for (vertex v = dst;; v = p[v]) {
      shortest_path.push_back(v);
      h_preds[idx] = v;
      ++idx;
      if (p[v] == v) break;
    }

    /*cout << "Shortest path from " << names[src] << " to "
         << names[dst] << ": ";
    list<vertex>::iterator spi = shortest_path.begin();
    cout << names[src];
    for(++spi; spi != shortest_path.end(); ++spi)
      cout << " -> " << names[*spi];
    cout << endl << "Total travel time: " << d[dst] << endl;*/
  }

  delete[] weights;
  delete[] edge_array;
  delete[] locations;
}

/**
 * @brief Run A* tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 * @tparam DIRECTED
 * @tparam DISTANCE_HEURISTIC
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <typename VertexId, typename SizeT, typename Value,
          // bool INSTRUMENT,
          // bool DEBUG,
          // bool SIZE_CHECK,
          bool DIRECTED, int DISTANCE_HEURISTIC>
cudaError_t RunTests(Info<VertexId, SizeT, Value> *info, Value *latitudes,
                     Value *longitudes, std::string *names) {
  typedef ASTARProblem<VertexId, SizeT, Value,
                       true,  // always mark predecessor
                       DISTANCE_HEURISTIC>
      Problem;

  typedef ASTAREnactor<Problem /*,
         INSTRUMENT,
         DEBUG,
         SIZE_CHECK*/> Enactor;

  // parse configurations from mObject info
  Csr<VertexId, SizeT, Value> *graph = info->csr_ptr;
  VertexId src = info->info["source_vertex"].get_int64();
  VertexId dst = info->info["destination_vertex"].get_int64();
  int max_grid_size = info->info["max_grid_size"].get_int();
  int num_gpus = info->info["num_gpus"].get_int();
  double max_queue_sizing = info->info["max_queue_sizing"].get_real();
  double max_queue_sizing1 = info->info["max_queue_sizing1"].get_real();
  double max_in_sizing = info->info["max_in_sizing"].get_real();
  std::string partition_method = info->info["partition_method"].get_str();
  double partition_factor = info->info["partition_factor"].get_real();
  int partition_seed = info->info["partition_seed"].get_int();
  bool quiet_mode = info->info["quiet_mode"].get_bool();
  bool quick_mode = info->info["quick_mode"].get_bool();
  bool stream_from_host = info->info["stream_from_host"].get_bool();
  std::string traversal_mode = info->info["traversal_mode"].get_str();
  bool instrument = info->info["instrument"].get_bool();
  bool debug = info->info["debug_mode"].get_bool();
  bool size_check = info->info["size_check"].get_bool();
  int iterations = info->info["num_iteration"].get_int();
  int delta_factor = info->info["delta_factor"].get_int();
  std::string src_type = info->info["source_type"].get_str();
  int src_seed = info->info["source_seed"].get_int();
  int communicate_latency = info->info["communicate_latency"].get_int();
  float communicate_multipy = info->info["communicate_multipy"].get_real();
  int expand_latency = info->info["expand_latency"].get_int();
  int subqueue_latency = info->info["subqueue_latency"].get_int();
  int fullqueue_latency = info->info["fullqueue_latency"].get_int();
  int makeout_latency = info->info["makeout_latency"].get_int();
  if (communicate_multipy > 1) max_in_sizing *= communicate_multipy;

  CpuTimer cpu_timer;
  cudaError_t retval = cudaSuccess;
  if (max_queue_sizing < 1.2) max_queue_sizing = 1.2;

  cpu_timer.Start();
  json_spirit::mArray device_list = info->info["device_list"].get_array();
  int *gpu_idx = new int[num_gpus];
  for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();

  // TODO: remove after merge mgpu-cq
  ContextPtr *context = (ContextPtr *)info->context;
  cudaStream_t *streams = (cudaStream_t *)info->streams;

  // Allocate host-side array (for both reference and GPU-computed results)
  VertexId *reference_preds = new VertexId[graph->nodes];
  VertexId *h_preds = new VertexId[graph->nodes];
  // VertexId *reference_check_pred  = (quick_mode) ? NULL : reference_preds;

  size_t *org_size = new size_t[num_gpus];
  for (int gpu = 0; gpu < num_gpus; gpu++) {
    size_t dummy;
    if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
    if (retval = util::GRError(cudaMemGetInfo(&(org_size[gpu]), &dummy),
                               "cudaMemGetInfo failed", __FILE__, __LINE__))
      return retval;
  }

  // Allocate problem on GPU
  Problem *problem = new Problem;
  if (retval = util::GRError(
          problem->Init(stream_from_host, latitudes, longitudes, graph, NULL,
                        num_gpus, gpu_idx, partition_method, streams,
                        delta_factor, max_queue_sizing, max_in_sizing,
                        partition_factor, partition_seed),
          "A* Problem Init failed", __FILE__, __LINE__))
    return retval;

  // Allocate SSSP enactor map
  Enactor *enactor =
      new Enactor(num_gpus, gpu_idx, instrument, debug, size_check);
  if (retval = util::GRError(
          enactor->Init(context, problem, max_grid_size, traversal_mode),
          "SSSP Enactor Init failed", __FILE__, __LINE__))
    return retval;

  enactor->communicate_latency = communicate_latency;
  enactor->communicate_multipy = communicate_multipy;
  enactor->expand_latency = expand_latency;
  enactor->subqueue_latency = subqueue_latency;
  enactor->fullqueue_latency = fullqueue_latency;
  enactor->makeout_latency = makeout_latency;

  if (retval = util::SetDevice(gpu_idx[0])) return retval;
  if (retval = util::latency::Test(
          streams[0], problem->data_slices[0]->latency_data,
          communicate_latency, communicate_multipy, expand_latency,
          subqueue_latency, fullqueue_latency, makeout_latency))
    return retval;

  cpu_timer.Stop();
  info->info["preprocess_time"] = cpu_timer.ElapsedMillis();

  // perform SSSP
  double total_elapsed = 0.0;
  double single_elapsed = 0.0;
  double max_elapsed = 0.0;
  double min_elapsed = 1e10;
  json_spirit::mArray process_times;
  if (src_type == "random2") {
    if (src_seed == -1) src_seed = time(NULL);
    if (!quiet_mode) printf("src_seed = %d\n", src_seed);
    srand(src_seed);
  }
  if (!quiet_mode) printf("Using traversal mode %s\n", traversal_mode.c_str());
  for (int iter = 0; iter < iterations; ++iter) {
    if (src_type == "random2") {
      bool src_valid = false;
      while (!src_valid) {
        src = rand() % graph->nodes;
        if (graph->row_offsets[src] != graph->row_offsets[src + 1])
          src_valid = true;
      }
    }

    // TODO: For Graph Heuristic, need to specify sample_weight.
    Value sample_weight = 1.0f;

    if (retval = util::GRError(
            problem->Reset(src, dst, sample_weight, enactor->GetFrontierType(),
                           max_queue_sizing, max_queue_sizing1),
            "SSSP Problem Data Reset Failed", __FILE__, __LINE__))
      return retval;

    if (retval = util::GRError(enactor->Reset(), "SSSP Enactor Reset failed",
                               __FILE__, __LINE__))
      return retval;

    for (int gpu = 0; gpu < num_gpus; gpu++) {
      if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
      if (retval =
              util::GRError(cudaDeviceSynchronize(),
                            "cudaDeviceSynchronize failed", __FILE__, __LINE__))
        return retval;
    }

    if (!quiet_mode) {
      printf("__________________________\n");
      fflush(stdout);
    }
    cpu_timer.Start();
    if (retval = util::GRError(enactor->Enact(src, traversal_mode),
                               "SSSP Problem Enact Failed", __FILE__, __LINE__))
      return retval;
    cpu_timer.Stop();
    single_elapsed = cpu_timer.ElapsedMillis();
    total_elapsed += single_elapsed;
    process_times.push_back(single_elapsed);
    if (single_elapsed > max_elapsed) max_elapsed = single_elapsed;
    if (single_elapsed < min_elapsed) min_elapsed = single_elapsed;
    if (!quiet_mode) {
      printf(
          "--------------------------\n"
          "iteration %d elapsed: %lf ms, src = %lld, #iteration = %lld\n",
          iter, single_elapsed, (long long)src,
          (long long)enactor->enactor_stats->iteration);
      fflush(stdout);
    }
  }
  total_elapsed /= iterations;
  info->info["process_times"] = process_times;
  info->info["min_process_time"] = min_elapsed;
  info->info["max_process_time"] = max_elapsed;

  // compute reference CPU SSSP solution for source-distance
  if (!quick_mode) {
    if (!quiet_mode) {
      printf("Computing reference value ...\n");
    }
    if (DIRECTED)
      ReferenceAStarDirected<VertexId, SizeT, Value>(
          *graph, names, latitudes, longitudes, reference_preds, src, dst);
    else
      ReferenceAStarUndirected<VertexId, SizeT, Value>(
          *graph, names, latitudes, longitudes, reference_preds, src, dst);
    if (!quiet_mode) {
      printf("\n");
    }
  }

  cpu_timer.Start();
  // Copy out results
  if (retval = util::GRError(problem->Extract(h_preds),
                             "A* Problem Data Extraction Failed", __FILE__,
                             __LINE__))
    return retval;

  // Verify the result
  if (!quick_mode) {
    if (!quiet_mode) {
      printf("Path Validity: ");
    }
    VertexId *h_path = new VertexId[enactor->enactor_stats->iteration + 1];
    int idx = 0;
    h_path[idx++] = dst;
    VertexId next = h_preds[dst];
    h_path[idx] = next;
    while (next != -1) {
      next = h_preds[next];
      h_path[++idx] = next;
    }
    int error_num =
        CompareResults(h_path, reference_preds, idx, true, quiet_mode);
    if (error_num > 0) {
      if (!quiet_mode) {
        printf("%d errors occurred.\n", error_num);
      }
    }
    delete[] h_path;
  }

  info->ComputeTraversalStats(  // compute running statistics
      enactor->enactor_stats.GetPointer(), total_elapsed, h_preds);

  if (!quiet_mode) {
    {
      printf("\nFirst 40 preds of the GPU result.\n");
      DisplaySolution(h_preds, names, dst, graph->nodes);
    }

    printf("\n\tMemory Usage(B)\t");
    for (int gpu = 0; gpu < num_gpus; gpu++)
      if (num_gpus > 1) {
        if (gpu != 0)
          printf(" #keys%d,0\t #keys%d,1\t #ins%d,0\t #ins%d,1", gpu, gpu, gpu,
                 gpu);
        else
          printf(" #keys%d,0\t #keys%d,1", gpu, gpu);
      } else
        printf(" #keys%d,0\t #keys%d,1", gpu, gpu);
    if (num_gpus > 1) printf(" #keys%d", num_gpus);
    printf("\n");
    double max_queue_sizing_[2] = {0, 0}, max_in_sizing_ = 0;
    for (int gpu = 0; gpu < num_gpus; gpu++) {
      size_t gpu_free, dummy;
      cudaSetDevice(gpu_idx[gpu]);
      cudaMemGetInfo(&gpu_free, &dummy);
      printf("GPU_%d\t %ld", gpu_idx[gpu], org_size[gpu] - gpu_free);
      for (int i = 0; i < num_gpus; i++) {
        for (int j = 0; j < 2; j++) {
          SizeT x =
              problem->data_slices[gpu]->frontier_queues[i].keys[j].GetSize();
          printf("\t %lld", (long long)x);
          double factor =
              1.0 * x /
              (num_gpus > 1 ? problem->graph_slices[gpu]->in_counter[i]
                            : problem->graph_slices[gpu]->nodes);
          if (factor > max_queue_sizing_[j]) max_queue_sizing_[j] = factor;
        }
        if (num_gpus > 1 && i != 0)
          for (int t = 0; t < 2; t++) {
            SizeT x = problem->data_slices[gpu][0].keys_in[t][i].GetSize();
            printf("\t %lld", (long long)x);
            double factor = 1.0 * x / problem->graph_slices[gpu]->in_counter[i];
            if (factor > max_in_sizing_) max_in_sizing_ = factor;
          }
      }
      if (num_gpus > 1)
        printf("\t %lld", (long long)(problem->data_slices[gpu]
                                          ->frontier_queues[num_gpus]
                                          .keys[0]
                                          .GetSize()));
      printf("\n");
    }
    printf("\t queue_sizing =\t %lf \t %lf", max_queue_sizing_[0],
           max_queue_sizing_[1]);
    if (num_gpus > 1) printf("\t in_sizing =\t %lf", max_in_sizing_);
    printf("\n");
  }

  // Clean up
  if (org_size) {
    delete[] org_size;
    org_size = NULL;
  }
  if (enactor) {
    if (retval = util::GRError(enactor->Release(), "A* Enactor Release failed",
                               __FILE__, __LINE__))
      return retval;
    delete enactor;
    enactor = NULL;
  }
  if (problem) {
    if (retval = util::GRError(problem->Release(), "A* Problem Release failed",
                               __FILE__, __LINE__))
      return retval;
    delete problem;
    problem = NULL;
  }
  if (reference_preds) {
    delete[] reference_preds;
    reference_preds = NULL;
  }
  if (h_preds) {
    delete[] h_preds;
    h_preds = NULL;
  }
  cpu_timer.Stop();
  info->info["postprocess_time"] = cpu_timer.ElapsedMillis();
  return retval;
}

/******************************************************************************
 * Main
 ******************************************************************************/
template <typename VertexId,  // use int as the vertex identifier
          typename SizeT,     // use int as the graph size type
          typename Value>     // use int as the value type
int main_(CommandLineArgs *args) {
  CpuTimer cpu_timer, cpu_timer2;
  cpu_timer.Start();
  Csr<VertexId, SizeT, Value> csr(false);  // graph we process on
  Info<VertexId, SizeT, Value> *info = new Info<VertexId, SizeT, Value>;

  // graph construction or generation related parameters
  info->info["undirected"] = args->CheckCmdLineFlag("undirected");
  info->info["edge_value"] = true;  // require per edge weight values

  cpu_timer2.Start();
  info->Init("ASTAR", *args, csr);  // initialize Info structure
  info->info["load_time"] = cpu_timer2.ElapsedMillis();

  // Here, read in latitude/longitude
  std::string *names = new std::string[csr.nodes];
  Value *latitudes = new Value[csr.nodes];
  Value *longitudes = new Value[csr.nodes];
  std::string mapfile_name;
  args->GetCmdLineArgument("mapfile", mapfile_name);
  ifstream mapfile;
  mapfile.open(mapfile_name.c_str());
  for (int i = 0; i < csr.nodes; ++i) {
    int idx;
    mapfile >> idx >> names[i] >> latitudes[i] >> longitudes[i];
  }

  if (info->info["undirected"].get_bool())
    RunTests<VertexId, SizeT, Value, false, 1>(
        info, latitudes, longitudes,
        names);  // run test, now only test the distance_heuristic
  else
    RunTests<VertexId, SizeT, Value, true, 1>(
        info, latitudes, longitudes,
        names);  // run test, now only test the distance_heuristic

  cpu_timer.Stop();
  info->info["total_time"] = cpu_timer.ElapsedMillis();

  if (!(info->info["quiet_mode"].get_bool())) {
    info->DisplayStats();  // display collected statistics
  }

  info->CollectInfo();  // collected all the info and put into JSON mObject

  delete[] names;
  delete[] latitudes;
  delete[] longitudes;
  return 0;
}

template <typename VertexId,  // the vertex identifier type, usually int or long
                              // long
          typename SizeT>     // the size tyep, usually int or long long
int main_Value(CommandLineArgs *args) {
  //    if (args -> CheckCmdLineFlag("64bit-Value"))
  //        return main_<VertexId, SizeT, long long>(args);
  //    else
  //        return main_<VertexId, SizeT, int      >(args);
  return main_<VertexId, SizeT, float>(args);  // Value = VertexId for A*
}

template <typename VertexId>
int main_SizeT(CommandLineArgs *args) {
  // disabled to reduce compile time
  if (args->CheckCmdLineFlag("64bit-SizeT"))
    return main_Value<VertexId, long long>(args);
  else
    return main_Value<VertexId, int>(args);
}

int main_VertexId(CommandLineArgs *args) {
  // disabled, because oprtr::filter::KernelPolicy::SmemStorage is too large for
  // 64bit VertexId
  // if (args -> CheckCmdLineFlag("64bit-VertexId"))
  //    return main_SizeT<long long>(args);
  // else
  return main_SizeT<int>(args);
}

int main(int argc, char **argv) {
  CommandLineArgs args(argc, argv);
  int graph_args = argc - args.ParsedArgc() - 1;
  if (argc < 2 || graph_args < 1 || args.CheckCmdLineFlag("help")) {
    Usage();
    return 1;
  }

  return main_VertexId(&args);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
