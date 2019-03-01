#include <stdio.h>
#include <string>
#include <omp.h>

// Utilities and correctness-checking
#include <gunrock/util/multithread_utils.cuh>
#include <gunrock/util/sort_omp.cuh>
#include <gunrock/csr.cuh>
#include <gunrock/graphio/grmat.cuh>
#include <gunrock/coo.cuh>

#include <moderngpu.cuh>

// boost includes
#include <boost/config.hpp>
#include <boost/utility.hpp>

#include <gunrock/util/shared_utils.cuh>

using namespace gunrock;
// using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::graphio;
using namespace gunrock::graphio::grmat;

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
      "        --rmat_self_loops If this option is supplied, then self loops "
      "will be retained\n"
      "        --rmat_undirected If this option is not mentioned, then the "
      "graps will be undirected\n\n"
      "Optional arguments:\n"
      "[--file_name=<file name>] If the graph needs to be saved to a file, "
      "else it will not be saved physically.\n"
      "[--device=<device_index>] Set GPU(s) for testing (Default: 0).\n"
      "[--quiet]                 No output (unless --json is specified).\n"
      "[--normalized]\n");
}

template <typename VertexId, typename Tuple, typename SizeT>
Tuple *RemoveSelfLoops(Tuple *coo, SizeT &final_edges, SizeT coo_edges) {
  Tuple *new_coo = (Tuple *)malloc(sizeof(Tuple) * coo_edges);
  int num_threads = 1;
  SizeT *edge_counts = NULL;
  Tuple **new_coo_arr = NULL;
  final_edges = 0;
#pragma omp parallel
  {
    num_threads = omp_get_num_threads();
    int thread_num = omp_get_thread_num();
    if (thread_num == 0) {
      edge_counts = new SizeT[num_threads + 1];
      new_coo_arr = new Tuple *[num_threads + 1];
    }
#pragma omp barrier
    SizeT edge_start = (long long)(coo_edges)*thread_num / num_threads;
    SizeT edge_end = (long long)(coo_edges) * (thread_num + 1) / num_threads;
    new_coo_arr[thread_num] =
        (Tuple *)malloc(sizeof(Tuple) * (edge_end - edge_start));
    SizeT edge = edge_start;
    SizeT new_edge = 0;
    for (edge = edge_start; edge < edge_end; edge++) {
      VertexId col = coo[edge].col;
      VertexId row = coo[edge].row;
      if ((col != row) &&
          (edge == 0 || col != coo[edge - 1].col || row != coo[edge - 1].row)) {
        new_coo_arr[thread_num][new_edge].col = (VertexId)col;
        new_coo_arr[thread_num][new_edge].row = (VertexId)row;
        new_coo_arr[thread_num][new_edge].val = coo[edge].val;
        new_edge++;
      }
    }
    edge_counts[thread_num] = new_edge;
  }

  SizeT edge = 0;
  for (int i = 0; i < num_threads; i++) {
    for (SizeT tmp_edge = 0; tmp_edge < edge_counts[i]; tmp_edge++) {
      new_coo[edge].row = new_coo_arr[i][tmp_edge].row;
      new_coo[edge].col = new_coo_arr[i][tmp_edge].col;
      new_coo[edge].val = new_coo_arr[i][tmp_edge].val;
      edge++;
    }
    free(new_coo_arr[i]);
  }
  delete[] new_coo_arr;
  new_coo_arr = NULL;
  final_edges = edge;

  delete[] edge_counts;
  edge_counts = NULL;
  free(coo);
  coo = new_coo;

  return coo;
}

/**
 * @brief Modify COO graph as per the expectation of user
 *
 * @param[in] output_file Output file to dump the graph topology info
 * @param[in] coo Pointer to COO-format graph
 * @param[in] coo_nodes Number of nodes in COO-format graph
 * @param[in] coo_edges Number of edges in COO-format graph
 * @param[in] ordered_rows Are the rows sorted? If not, sort them.
 * @param[in] undirected Is the graph directed or not?
 * @param[in] reversed Is the graph reversed or not?
 * @param[in] quiet Don't print out anything.
 * @param[in] self_loops is true if self loops are accepted in the graph.
 *
 * Default: Assume rows are not sorted.
 */
template <typename VertexId, typename Tuple, typename SizeT>
Tuple *FromCoo_MM(FILE *f, Tuple *coo, SizeT coo_nodes, SizeT coo_edges,
                  bool ordered_rows = false, bool undirected = false,
                  bool quiet = false, bool self_loops = false) {
  SizeT rows = 0;
  SizeT cols = 0;
  util::CpuTimer cpu_timer;

  if ((!quiet) && (!self_loops)) {
    printf(
        "Converting %lld vertices, %lld %s edges ( %s tuples) "
        "to remove self loops...\n",
        (long long)coo_nodes, (long long)coo_edges,
        undirected ? "undirected" : "directed",
        ordered_rows ? "ordered" : "unordered");
  }

  cpu_timer.Start();
  int num_threads = 1;
  SizeT *cols_tmp = NULL;

// Find the max col
#pragma omp parallel
  {
    num_threads = omp_get_num_threads();
    int thread_num = omp_get_thread_num();
    if (thread_num == 0) {
      cols_tmp = new SizeT[num_threads + 1];
    }
#pragma omp barrier
    SizeT edge_start = (long long)(coo_edges)*thread_num / num_threads;
    SizeT edge_end = (long long)(coo_edges) * (thread_num + 1) / num_threads;
    SizeT edge = edge_start;
    SizeT max = coo[edge_start].col;
    for (edge = edge_start + 1; edge < edge_end; edge++) {
      if (max < coo[edge].col) {
        max = coo[edge].col;
      }
    }
    cols_tmp[thread_num] = max;
  }
  cols = cols_tmp[0];
  for (int i = 1; i < num_threads; i++) {
    if (cols < cols_tmp[i]) {
      cols = cols_tmp[i];
    }
  }

  if (cols_tmp != NULL) {
    delete[] cols_tmp;
    cols_tmp = NULL;
  }
  // If not ordered, order it as per row
  if (!ordered_rows) {
    util::omp_sort(coo, coo_edges, RowFirstTupleCompare<Tuple>);
    rows = coo[coo_edges - 1].row;
  }

  SizeT final_edges = coo_edges;

  if (self_loops == false) {
    coo = RemoveSelfLoops<VertexId>(coo, final_edges, coo_edges);
  }

  cpu_timer.Stop();

  if (!quiet) {
    printf("Time Elapsed for sorting %s is %f ms\n\n",
           (self_loops == true) ? "" : "and removing self loops",
           cpu_timer.ElapsedMillis());
    printf(
        "Number of edges %lld, Number of rows %lld and Number of columns "
        "%lld\n\n",
        (long long)final_edges, (long long)rows, (long long)cols);
  }

  if (f != NULL) {
    fprintf(f, "%%MatrixMarket matrix coordinate pattern %s\n",
            (undirected == true) ? "symmetric" : "");
    fprintf(f, "%lld %lld %lld\n", (long long)rows, (long long)cols,
            (long long)final_edges);
    for (SizeT i = 0; i < final_edges; i++) {
      fprintf(f, "%lld %lld\n", (long long)(coo[i].row),
              (long long)(coo[i].col));
    }
  }

  return coo;
}

template <typename VertexId, typename SizeT, typename Value>
int main_(CommandLineArgs *args) {
  // cudaError_t retval = cudaSuccess;
  CpuTimer cpu_timer, cpu_timer2;
  SizeT rmat_nodes = 1 << 10;
  SizeT rmat_edges = 1 << 10;
  SizeT rmat_scale = 10;
  SizeT rmat_edgefactor = 48;
  double rmat_a = 0.57;
  double rmat_b = 0.19;
  double rmat_c = 0.19;
  double rmat_d = 1 - (rmat_a + rmat_b + rmat_c);
  double rmat_vmin = 1;
  double rmat_vmultipiler = 64;
  int rmat_seed = -1;
  bool undirected = false;
  bool self_loops = false;
  SizeT rmat_all_edges = rmat_edges;
  std::string file_name;
  bool quiet = false;

  typedef Coo<VertexId, Value> EdgeTupleType;

  cpu_timer.Start();

  if (args->CheckCmdLineFlag("rmat_scale") &&
      args->CheckCmdLineFlag("rmat_nodes")) {
    printf("Please mention scale or nodes, not both \n");
    return cudaErrorInvalidConfiguration;
  } else if (args->CheckCmdLineFlag("rmat_edgefactor") &&
             args->CheckCmdLineFlag("rmat_edges")) {
    printf("Please mention edgefactor or edge, not both \n");
    return cudaErrorInvalidConfiguration;
  }

  self_loops = args->CheckCmdLineFlag("rmat_self_loops");
  // graph construction or generation related parameters
  if (args->CheckCmdLineFlag("normalized"))
    undirected = args->CheckCmdLineFlag("rmat_undirected");
  else
    undirected = true;  // require undirected input graph when unnormalized
  quiet = args->CheckCmdLineFlag("quiet");

  args->GetCmdLineArgument("rmat_scale", rmat_scale);
  rmat_nodes = 1 << rmat_scale;
  args->GetCmdLineArgument("rmat_nodes", rmat_nodes);
  args->GetCmdLineArgument("rmat_edgefactor", rmat_edgefactor);
  rmat_edges = rmat_nodes * rmat_edgefactor;
  args->GetCmdLineArgument("rmat_edges", rmat_edges);
  args->GetCmdLineArgument("rmat_a", rmat_a);
  args->GetCmdLineArgument("rmat_b", rmat_b);
  args->GetCmdLineArgument("rmat_c", rmat_c);
  rmat_d = 1 - (rmat_a + rmat_b + rmat_c);
  args->GetCmdLineArgument("rmat_d", rmat_d);
  args->GetCmdLineArgument("rmat_seed", rmat_seed);
  args->GetCmdLineArgument("rmat_vmin", rmat_vmin);
  args->GetCmdLineArgument("rmat_vmultipiler", rmat_vmultipiler);
  args->GetCmdLineArgument("file_name", file_name);
  Coo<VertexId, Value> *coo = NULL;

  if (undirected == true) {
    rmat_all_edges = 2 * rmat_edges;
  } else {
    rmat_all_edges = rmat_edges;
  }

  std::vector<int> temp_devices;
  if (args->CheckCmdLineFlag("device"))  // parse device list
  {
    args->GetCmdLineArguments<int>("device", temp_devices);
  } else  // use single device with index 0
  {
    int gpu_idx;
    util::GRError(cudaGetDevice(&gpu_idx), "cudaGetDevice failed", __FILE__,
                  __LINE__);
    temp_devices.push_back(gpu_idx);
  }
  int *gpu_idx = new int[temp_devices.size()];
  for (int i = 0; i < temp_devices.size(); i++) gpu_idx[i] = temp_devices[i];

  if (!quiet) {
    printf(
        "---------Graph properties-------\n"
        "      Undirected : %s\n"
        "      Nodes : %lld\n"
        "      Edges : %lld\n"
        "      a = %f, b = %f, c = %f, d = %f\n\n\n",
        ((undirected == true) ? "True" : "False"), (long long)rmat_nodes,
        (long long)(rmat_edges * ((undirected == true) ? 2 : 1)), rmat_a,
        rmat_b, rmat_c, rmat_d);
  }
  cpu_timer2.Start();
  coo =
      (Coo<VertexId, Value> *)BuildRmatGraph_coo<true, VertexId, SizeT, Value>(
          rmat_nodes, rmat_edges, undirected, rmat_a, rmat_b, rmat_c, rmat_d,
          rmat_vmultipiler, rmat_vmin, rmat_seed, quiet, temp_devices.size(),
          gpu_idx);
  cpu_timer2.Stop();
  if (coo != NULL) {
    if (!quiet) printf("Graph has been generated \n");
  } else {
    return cudaErrorMemoryAllocation;
  }

  FILE *f = NULL;
  if (!(file_name.empty())) {
    f = fopen(file_name.c_str(), "w");
    if (f == NULL) {
      if (!quiet) printf("Error: File path doesn't exist \n");
    }
    // Convert the COO format to Matrix Market format
    if (!quiet) printf("Converting the COO format to Matrix Market format \n");
  }
  coo = FromCoo_MM<VertexId>(f, coo, rmat_nodes, rmat_all_edges, false,
                             undirected, quiet, self_loops);
  if (f != NULL) {
    fclose(f);
    if (!quiet) printf("Converted the COO format to Matrix Market format\n");
  }
  cpu_timer.Stop();

  if (coo == NULL) {
    if (!quiet) printf("Error: Failed to create the Graph \n");
    return cudaErrorMemoryAllocation;
  } else {
    if (!quiet)
      printf(
          "Time to generate the graph %f ms\n"
          "Total time %f ms\n",
          cpu_timer2.ElapsedMillis(), cpu_timer.ElapsedMillis());
    free(coo);
  }

  return cudaSuccess;
}

template <typename VertexId,  // the vertex identifier type, usually int or long
                              // long
          typename SizeT>
int main_Value(CommandLineArgs *args) {
  // can be disabled to reduce compile time
  //    if (args -> CheckCmdLineFlag("64bit-Value"))
  //        return main_<VertexId, SizeT, double>(args);
  //    else
  return main_<VertexId, SizeT, float>(args);
}

template <typename VertexId>
int main_SizeT(CommandLineArgs *args) {
  // can be disabled to reduce compile time
  if (args->CheckCmdLineFlag("64bit-SizeT") || sizeof(VertexId) > 4)
    return main_Value<VertexId, long long>(args);
  else
    return main_Value<VertexId, int>(args);
}

int main_VertexId(CommandLineArgs *args) {
  // can be disabled to reduce compile time
  if (args->CheckCmdLineFlag("64bit-VertexId"))
    return main_SizeT<long long>(args);
  else
    return main_SizeT<int>(args);
}

int main(int argc, char **argv) {
  CommandLineArgs args(argc, argv);
  int graph_args = argc - args.ParsedArgc() - 1;
  if (argc < 2 || args.CheckCmdLineFlag("help")) {
    Usage();
    return 1;
  }

  return main_VertexId(&args);
}
