// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.

/**
 * @file
 * grmat.cuh
 *
 * @brief gpu based R-MAT Graph Construction Routines
 */

#pragma once

#include <curand_kernel.h>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace graphio {
namespace grmat {

template <typename VertexId, typename Value>
struct Coo_nv {
  VertexId *row;
  VertexId *col;
  Value *val;

  Coo_nv() {
    row = NULL;
    col = NULL;
    val = NULL;
  }
};

__device__ __forceinline__ double Sprng(curandState &rand_state) {
  return curand_uniform(&rand_state);
}

__device__ __forceinline__ bool Flip(curandState &rand_state) {
  return Sprng(rand_state) >= 0.5;
}

template <typename VertexId>
__device__ __forceinline__ void ChoosePartition(VertexId &u, VertexId &v,
                                                VertexId step, double a,
                                                double b, double c, double d,
                                                curandState &rand_state) {
  double p = Sprng(rand_state);

  if (p < a) {
    // do nothing
  } else if ((a < p) && (p < a + b)) {
    v += step;
  } else if ((a + b < p) && (p < a + b + c)) {
    u += step;
  } else if ((a + b + c < p) && (p < a + b + c + d)) {
    u += step;
    v += step;
  }
}

__device__ __forceinline__ void VaryParams(double &a, double &b, double &c,
                                           double &d, curandState &rand_state) {
  double v, S;

  // Allow a max. of 5% variation
  v = 0.05;

  if (Flip(rand_state)) {
    a += a * v * Sprng(rand_state);
  } else {
    a -= a * v * Sprng(rand_state);
  }

  if (Flip(rand_state)) {
    b += b * v * Sprng(rand_state);
  } else {
    b -= b * v * Sprng(rand_state);
  }

  template <bool WITH_VALUES, typename VertexId, typename SizeT, typename Value>
  __global__ void Rmat_Kernel(
      SizeT num_nodes, SizeT edge_count, Coo<VertexId, Value> * d_edges,
      bool undirected, Value vmultipiler, Value vmin, double a0, double b0,
      double c0, double d0, curandState *d_rand_states) {
    SizeT i = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
    curandState &rand_state = d_rand_states[i];

    while (i < edge_count) {
      double a = a0, b = b0, c = c0, d = d0;
      VertexId u = 1, v = 1, step = num_nodes >> 1;
      Coo<VertexId, Value> *edge = d_edges + i;

      while (step >= 1) {
        ChoosePartition(u, v, step, a, b, c, d, rand_state);
        step >>= 1;
        VaryParams(a, b, c, d, rand_state);
      }
      edge->row = u - 1;
      edge->col = v - 1;
      if (WITH_VALUES)
        edge->val = Sprng(rand_state) * vmultipiler + vmin;
      else
        edge->val = 1;

      if (undirected) {
        edge = d_edges + (i + edge_count);
        edge->row = v - 1;
        edge->col = u - 1;
        if (WITH_VALUES)
          edge->val = Sprng(rand_state) * vmultipiler + vmin;
        else
          edge->val = 1;
      }
      i += STRIDE;
    }
  }

  template <bool WITH_VALUES, typename VertexId, typename SizeT, typename Value>
  __global__ void Rmat_Kernel_nv(
      SizeT num_nodes, SizeT edge_count, VertexId * rows, VertexId * cols,
      Value * vals, bool undirected, Value vmultipiler, Value vmin, double a0,
      double b0, double c0, double d0, curandState *d_rand_states) {
    SizeT i = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
    curandState &rand_state = d_rand_states[i];

    while (i < edge_count) {
      double a = a0, b = b0, c = c0, d = d0;
      VertexId u = 1, v = 1, step = num_nodes >> 1;
      VertexId *row = rows + i;
      VertexId *col = cols + i;
      Value *val = NULL;
      if (WITH_VALUES) val = vals + i;

      while (step >= 1) {
        ChoosePartition(u, v, step, a, b, c, d, rand_state);
        step >>= 1;
        VaryParams(a, b, c, d, rand_state);
      }
      *row = u - 1;
      *col = v - 1;
      if (WITH_VALUES) *val = Sprng(rand_state) * vmultipiler + vmin;

      if (undirected) {
        row = rows + (i + edge_count);
        col = cols + (i + edge_count);
        if (WITH_VALUES) val = vals + (i + edge_count);
        *row = v - 1;
        *col = u - 1;
        if (WITH_VALUES) *val = Sprng(rand_state) * vmultipiler + vmin;
      }
      i += STRIDE;
    }
  }

  template <typename SizeT>
  __global__ void Rand_Init(unsigned int seed, curandState *d_states) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, d_states + id);
  }

  /**
   * @brief Builds a R-MAT CSR graph.
   *
   * @tparam WITH_VALUES Whether or not associate with per edge weight values.
   * @tparam VertexId Vertex identifier.
   * @tparam Value Value type.
   * @tparam SizeT Graph size type.
   *
   * @param[in] nodes
   * @param[in] edges
   * @param[in] graph
   * @param[in] undirected
   * @param[in] a0
   * @param[in] b0
   * @param[in] c0
   * @param[in] d0
   * @param[in] vmultipiler
   * @param[in] vmin
   * @param[in] seed
   */
  template <bool WITH_VALUES, typename VertexId, typename SizeT, typename Value>
  cudaError_t BuildRmatGraph(
      SizeT num_nodes, SizeT num_edges, Csr<VertexId, SizeT, Value> & graph,
      bool undirected, double a0 = 0.55, double b0 = 0.2, double c0 = 0.2,
      double d0 = 0.05, double vmultipiler = 1.00, double vmin = 1.00,
      int seed = -1, bool quiet = false, int num_gpus = 1,
      int *gpu_idx = NULL) {
    typedef Coo<VertexId, Value> EdgeTupleType;
    cudaError_t retval = cudaSuccess;

    if ((num_nodes < 0) || (num_edges < 0)) {
      fprintf(stderr, "Invalid graph size: nodes=%lld, edges=%lld",
              (long long)num_nodes, (long long)num_edges);
      return util::GRError("Invalid graph size");
    }

    SizeT directed_edges = (undirected) ? num_edges * 2 : num_edges;
    EdgeTupleType *coo =
        (EdgeTupleType *)malloc(sizeof(EdgeTupleType) * directed_edges);

    if (seed == -1) seed = time(NULL);
    if (!quiet) {
      printf("rmat_seed = %lld\n", (long long)seed);
    }

    cudaStream_t *streams = new cudaStream_t[num_gpus];
    util::Array1D<SizeT, EdgeTupleType> *edges =
        new util::Array1D<SizeT, EdgeTupleType>[num_gpus];
    util::Array1D<SizeT, curandState> *rand_states =
        new util::Array1D<SizeT, curandState>[num_gpus];

    for (int gpu = 0; gpu < num_gpus; gpu++) {
      if (gpu_idx != NULL) {
        if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
      }

      if (retval = util::GRError(cudaStreamCreate(streams + gpu),
                                 "cudaStreamCreate failed", __FILE__, __LINE__))
        return retval;
      SizeT start_edge = num_edges * 1.0 / num_gpus * gpu;
      SizeT end_edge = num_edges * 1.0 / num_gpus * (gpu + 1);
      SizeT edge_count = end_edge - start_edge;
      if (undirected) edge_count *= 2;
      unsigned int seed_ = seed + 616 * gpu;
      if (retval = edges[gpu].Allocate(edge_count, util::DEVICE)) return retval;
      if (retval = edges[gpu].SetPointer(
              coo + start_edge * (undirected ? 2 : 1), edge_count, util::HOST))
        return retval;

      int block_size = (sizeof(VertexId) == 4) ? 1024 : 512;
      int grid_size = edge_count / block_size + 1;
      if (grid_size > 480) grid_size = 480;
      if (retval =
              rand_states[gpu].Allocate(block_size * grid_size, util::DEVICE))
        return retval;
      Rand_Init<SizeT><<<grid_size, block_size, 0, streams[gpu]>>>(
          seed_, rand_states[gpu].GetPointer(util::DEVICE));

      Rmat_Kernel<WITH_VALUES, VertexId, SizeT, Value>
          <<<grid_size, block_size, 0, streams[gpu]>>>(
              num_nodes, (undirected ? edge_count / 2 : edge_count),
              edges[gpu].GetPointer(util::DEVICE), undirected, vmultipiler,
              vmin, a0, b0, c0, d0, rand_states[gpu].GetPointer(util::DEVICE));

      if (retval = edges[gpu].Move(util::DEVICE, util::HOST, edge_count, 0,
                                   streams[gpu]))
        return retval;
    }

    for (int gpu = 0; gpu < num_gpus; gpu++) {
      if (gpu_idx != NULL) {
        if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
      }
      if (retval =
              util::GRError(cudaStreamSynchronize(streams[gpu]),
                            "cudaStreamSynchronize failed", __FILE__, __LINE__))
        return retval;
      if (retval =
              util::GRError(cudaStreamDestroy(streams[gpu]),
                            "cudaStreamDestroy failed", __FILE__, __LINE__))
        return retval;
      if (retval = edges[gpu].Release()) return retval;
      if (retval = rand_states[gpu].Release()) return retval;
    }

    delete[] rand_states;
    rand_states = NULL;
    delete[] edges;
    edges = NULL;
    delete[] streams;
    streams = NULL;

    // convert COO to CSR
    char *out_file = NULL;  // TODO: currently does not support write CSR file
    graph.template FromCoo<WITH_VALUES, EdgeTupleType>(
        out_file, coo, num_nodes, directed_edges, false, undirected, false,
        quiet);

    free(coo);

    return retval;
  }

  /**
   * @brief Builds a meta R-MAT CSR graph that connects n R-MAT graphs by a
   * single root node.
   *
   * @tparam WITH_VALUES Whether or not associate with per edge weight values.
   * @tparam VertexId Vertex identifier.
   * @tparam Value Value type.
   * @tparam SizeT Graph size type.
   *
   * @param[in] nodes
   * @param[in] edges
   * @param[in] graph
   * @param[in] undirected
   * @param[in] a0
   * @param[in] b0
   * @param[in] c0
   * @param[in] d0
   * @param[in] vmultipiler
   * @param[in] vmin
   * @param[in] seed
   */
  template <bool WITH_VALUES, typename VertexId, typename SizeT, typename Value>
  cudaError_t BuildMetaRmatGraph(
      SizeT num_nodes, SizeT num_edges, Csr<VertexId, SizeT, Value> & graph,
      bool undirected, double a0 = 0.55, double b0 = 0.2, double c0 = 0.2,
      double d0 = 0.05, double vmultipiler = 1.00, double vmin = 1.00,
      int seed = -1, bool quiet = false, int num_gpus = 1,
      int *gpu_idx = NULL) {
    // Do not build any meta root node if num_gpus == 1
    typedef Coo<VertexId, Value> EdgeTupleType;
    cudaError_t retval = cudaSuccess;

    if ((num_nodes < 0) || (num_edges < 0)) {
      fprintf(stderr, "Invalid graph size: nodes=%lld, edges=%lld",
              (long long)num_nodes, (long long)num_edges);
      return util::GRError("Invalid graph size");
    }

    SizeT directed_edges = (undirected) ? num_edges * 2 : num_edges;
    EdgeTupleType *coo = (EdgeTupleType *)malloc(
        sizeof(EdgeTupleType) *
        (directed_edges * num_gpus + ((num_gpus > 1) ? num_gpus : 0)));

    if (num_gpus > 1) {
      for (int i = 0; i < num_gpus; ++i) {
        coo[i].row = 0;
        coo[i].col = i * num_nodes + 1;
        coo[i].val =
            1;  // to simplify the implementation, give 1 as weight for now.
      }
    }

    if (seed == -1) seed = time(NULL);
    if (!quiet) {
      printf("rmat_seed = %lld\n", (long long)seed);
    }

    cudaStream_t *streams = new cudaStream_t[num_gpus];
    util::Array1D<SizeT, EdgeTupleType> *edges =
        new util::Array1D<SizeT, EdgeTupleType>[num_gpus];
    util::Array1D<SizeT, curandState> *rand_states =
        new util::Array1D<SizeT, curandState>[num_gpus];

    for (int gpu = 0; gpu < num_gpus; gpu++) {
      if (gpu_idx != NULL) {
        if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
      }

      if (retval = util::GRError(cudaStreamCreate(streams + gpu),
                                 "cudaStreamCreate failed", __FILE__, __LINE__))
        return retval;
      SizeT start_edge = num_edges * 1.0 / num_gpus * gpu;
      SizeT end_edge = num_edges * 1.0 / num_gpus * (gpu + 1);
      SizeT edge_count = end_edge - start_edge;
      if (undirected) edge_count *= 2;
      unsigned int seed_ = seed + 616 * gpu;
      if (retval = edges[gpu].Allocate(edge_count, util::DEVICE)) return retval;

      int block_size = 1024;
      int grid_size = edge_count / block_size + 1;
      if (grid_size > 480) grid_size = 480;
      if (retval =
              rand_states[gpu].Allocate(block_size * grid_size, util::DEVICE))
        return retval;
      Rand_Init<SizeT><<<grid_size, block_size, 0, streams[gpu]>>>(
          seed_, rand_states[gpu].GetPointer(util::DEVICE));

      Rmat_Kernel<WITH_VALUES, VertexId, SizeT, Value>
          <<<grid_size, block_size, 0, streams[gpu]>>>(
              num_nodes, (undirected ? edge_count / 2 : edge_count),
              edges[gpu].GetPointer(util::DEVICE), undirected, vmultipiler,
              vmin, a0, b0, c0, d0, rand_states[gpu].GetPointer(util::DEVICE));

      // for source node: add num_nodes
      // for dest node: add num_nodes
      VertexId pre_offset = (num_gpus > 1) ? 1 : 0;
      for (int copy_idx = 0; copy_idx < num_gpus; ++copy_idx) {
        VertexId offset = (copy_idx) ? num_nodes : pre_offset;
        EdgeTupleType *edges_pointer = edges[gpu].GetPointer(util::DEVICE);
        util::MemsetAddEdgeValKernel<<<256, 1024>>>(edges_pointer, offset,
                                                    edge_count);
        if (retval =
                edges[gpu].SetPointer(coo + ((num_gpus > 1) ? num_gpus : 0) +
                                          directed_edges * copy_idx +
                                          start_edge * (undirected ? 2 : 1),
                                      edge_count, util::HOST))
          return retval;
        if (retval = edges[gpu].Move(util::DEVICE, util::HOST, edge_count, 0,
                                     streams[gpu]))
          return retval;
      }
    }

    for (int gpu = 0; gpu < num_gpus; gpu++) {
      if (gpu_idx != NULL) {
        if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
      }
      if (retval =
              util::GRError(cudaStreamSynchronize(streams[gpu]),
                            "cudaStreamSynchronize failed", __FILE__, __LINE__))
        return retval;
      if (retval =
              util::GRError(cudaStreamDestroy(streams[gpu]),
                            "cudaStreamDestroy failed", __FILE__, __LINE__))
        return retval;
      if (retval = edges[gpu].Release()) return retval;
      if (retval = rand_states[gpu].Release()) return retval;
    }

    delete[] rand_states;
    rand_states = NULL;
    delete[] edges;
    edges = NULL;
    delete[] streams;
    streams = NULL;

    // convert COO to CSR
    char *out_file = NULL;  // TODO: currently does not support write CSR file
    graph.template FromCoo<WITH_VALUES, EdgeTupleType>(
        out_file, coo, num_nodes * num_gpus,
        directed_edges * num_gpus + ((num_gpus > 1) ? num_gpus : 0), false,
        undirected, false, quiet);

    free(coo);

    return retval;
  }

  /**
   * @brief Builds a R-MAT CSR graph and returns it in coo format.
   *
   * @tparam WITH_VALUES Whether or not associate with per edge weight values.
   * @tparam VertexId Vertex identifier.
   * @tparam Value Value type.
   * @tparam SizeT Graph size type.
   *
   * @param[in] nodes
   * @param[in] edges
   * @param[in] undirected
   * @param[in] a0
   * @param[in] b0
   * @param[in] c0
   * @param[in] d0
   * @param[in] vmultipiler
   * @param[in] vmin
   * @param[in] seed
   */
  template <bool WITH_VALUES, typename VertexId, typename SizeT, typename Value>
  void *BuildRmatGraph_coo(SizeT num_nodes, SizeT num_edges,
                           //    Csr<VertexId, SizeT, Value> &graph,
                           bool undirected, double a0 = 0.55, double b0 = 0.2,
                           double c0 = 0.2, double d0 = 0.05,
                           double vmultipiler = 1.00, double vmin = 1.00,
                           int seed = -1, bool quiet = false, int num_gpus = 1,
                           int *gpu_idx = NULL) {
    typedef Coo<VertexId, Value> EdgeTupleType;
    // cudaError_t retval = cudaSuccess;

    if ((num_nodes < 0) || (num_edges < 0)) {
      fprintf(stderr, "Invalid graph size: nodes=%lld, edges=%lld",
              (long long)num_nodes, (long long)num_edges);
      util::GRError("Invalid graph size");
      return NULL;
    }

    SizeT directed_edges = (undirected) ? num_edges * 2 : num_edges;
    EdgeTupleType *coo =
        (EdgeTupleType *)malloc(sizeof(EdgeTupleType) * directed_edges);

    if (seed == -1) seed = time(NULL);
    if (!quiet) {
      printf("rmat_seed = %lld\n", (long long)seed);
    }

    cudaStream_t *streams = new cudaStream_t[num_gpus];
    util::Array1D<SizeT, EdgeTupleType> *edges =
        new util::Array1D<SizeT, EdgeTupleType>[num_gpus];
    util::Array1D<SizeT, curandState> *rand_states =
        new util::Array1D<SizeT, curandState>[num_gpus];

    for (int gpu = 0; gpu < num_gpus; gpu++) {
      if (gpu_idx != NULL) {
        if (util::SetDevice(gpu_idx[gpu])) return NULL;
      }

      if (util::GRError(cudaStreamCreate(streams + gpu),
                        "cudaStreamCreate failed", __FILE__, __LINE__))
        return NULL;
      SizeT start_edge = num_edges * 1.0 / num_gpus * gpu;
      SizeT end_edge = num_edges * 1.0 / num_gpus * (gpu + 1);
      SizeT edge_count = end_edge - start_edge;
      if (undirected) edge_count *= 2;
      unsigned int seed_ = seed + 616 * gpu;
      if (edges[gpu].Allocate(edge_count, util::DEVICE)) return NULL;
      if (edges[gpu].SetPointer(coo + start_edge * (undirected ? 2 : 1),
                                edge_count, util::HOST))
        return NULL;

      int block_size = (sizeof(VertexId) == 4) ? 1024 : 512;
      int grid_size = edge_count / block_size + 1;
      if (grid_size > 480) grid_size = 480;
      if (rand_states[gpu].Allocate(block_size * grid_size, util::DEVICE))
        return NULL;
      Rand_Init<SizeT><<<grid_size, block_size, 0, streams[gpu]>>>(
          seed_, rand_states[gpu].GetPointer(util::DEVICE));

      Rmat_Kernel<WITH_VALUES, VertexId, SizeT, Value>
          <<<grid_size, block_size, 0, streams[gpu]>>>(
              num_nodes, (undirected ? edge_count / 2 : edge_count),
              edges[gpu].GetPointer(util::DEVICE), undirected, vmultipiler,
              vmin, a0, b0, c0, d0, rand_states[gpu].GetPointer(util::DEVICE));

      if (edges[gpu].Move(util::DEVICE, util::HOST, edge_count, 0,
                          streams[gpu]))
        return NULL;
    }

    for (int gpu = 0; gpu < num_gpus; gpu++) {
      if (gpu_idx != NULL) {
        if (util::SetDevice(gpu_idx[gpu])) return NULL;
      }
      if (util::GRError(cudaStreamSynchronize(streams[gpu]),
                        "cudaStreamSynchronize failed", __FILE__, __LINE__))
        return NULL;
      if (util::GRError(cudaStreamDestroy(streams[gpu]),
                        "cudaStreamDestroy failed", __FILE__, __LINE__))
        return NULL;
      if (edges[gpu].Release()) return NULL;
      if (rand_states[gpu].Release()) return NULL;
    }

    delete[] rand_states;
    rand_states = NULL;
    delete[] edges;
    edges = NULL;
    delete[] streams;
    streams = NULL;

    return coo;
  }

  /**
   * @brief Builds a R-MAT graph in GPU passes the pointers back to calling
   * function.
   *
   * @tparam WITH_VALUES Whether or not associate with per edge weight values.
   * @tparam VertexId Vertex identifier.
   * @tparam Value Value type.
   * @tparam SizeT Graph size type.
   *
   * @param[in] nodes
   * @param[in] edges
   * @param[in] undirected
   * @param[in] a0
   * @param[in] b0
   * @param[in] c0
   * @param[in] d0
   * @param[in] vmultipiler
   * @param[in] vmin
   * @param[in] seed
   * @param[in] coo
   */
  template <bool WITH_VALUES, typename VertexId, typename SizeT, typename Value,
            typename EdgeTupleType>
  cudaError_t BuildRmatGraph_coo_nv(
      SizeT num_nodes, SizeT num_edges, EdgeTupleType & coo, bool undirected,
      double a0 = 0.55, double b0 = 0.2, double c0 = 0.2, double d0 = 0.05,
      double vmultipiler = 1.00, double vmin = 1.00, int seed = -1,
      bool quiet = false, int num_gpus = 1, int *gpu_idx = NULL) {
    cudaError_t retval = cudaSuccess;

    if ((num_nodes < 0) || (num_edges < 0)) {
      fprintf(stderr, "Invalid graph size: nodes=%lld, edges=%lld",
              (long long)num_nodes, (long long)num_edges);
      util::GRError("Invalid graph size");
      return util::GRError("Invalid graph size");
      ;
    }

    if (seed == -1) seed = time(NULL);
    if (!quiet) {
      printf("rmat_seed = %lld\n", (long long)seed);
      fflush(stdout);
    }

    cudaStream_t *streams = new cudaStream_t[num_gpus];
    util::Array1D<SizeT, curandState> *rand_states =
        new util::Array1D<SizeT, curandState>[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++) {
      if (gpu_idx != NULL) {
        if (retval = util::SetDevice(gpu_idx[gpu])) {
          return retval;
        }
      }

      cudaStream_t *streams = new cudaStream_t[num_gpus];
      util::Array1D<SizeT, EdgeTupleType> *edges =
          new util::Array1D<SizeT, EdgeTupleType>[num_gpus];
      util::Array1D<SizeT, curandState> *rand_states =
          new util::Array1D<SizeT, curandState>[num_gpus];

      for (int gpu = 0; gpu < num_gpus; gpu++) {
        if (gpu_idx != NULL) {
          if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
        }

        if (retval =
                util::GRError(cudaStreamCreate(streams + gpu),
                              "cudaStreamCreate failed", __FILE__, __LINE__))
          return retval;
        SizeT start_edge = num_edges * 1.0 / num_gpus * gpu;
        SizeT end_edge = num_edges * 1.0 / num_gpus * (gpu + 1);
        SizeT edge_count = end_edge - start_edge;
        if (undirected) edge_count *= 2;
        unsigned int seed_ = seed + 616 * gpu;
        if (retval = edges[gpu].Allocate(edge_count, util::DEVICE))
          return retval;

        int block_size = 1024;
        int grid_size = edge_count / block_size + 1;
        if (grid_size > 480) grid_size = 480;
        if (retval =
                rand_states[gpu].Allocate(block_size * grid_size, util::DEVICE))
          return retval;
        Rand_Init<SizeT><<<grid_size, block_size, 0, streams[gpu]>>>(
            seed_, rand_states[gpu].GetPointer(util::DEVICE));

        Rmat_Kernel<WITH_VALUES, VertexId, SizeT, Value>
            <<<grid_size, block_size, 0, streams[gpu]>>>(
                num_nodes, (undirected ? edge_count / 2 : edge_count),
                edges[gpu].GetPointer(util::DEVICE), undirected, vmultipiler,
                vmin, a0, b0, c0, d0,
                rand_states[gpu].GetPointer(util::DEVICE));

        // for source node: add num_nodes
        // for dest node: add num_nodes
        VertexId pre_offset = (num_gpus > 1) ? 1 : 0;
        for (int copy_idx = 0; copy_idx < num_gpus; ++copy_idx) {
          VertexId offset = (copy_idx) ? num_nodes : pre_offset;
          EdgeTupleType *edges_pointer = edges[gpu].GetPointer(util::DEVICE);
          util::MemsetAddEdgeValKernel<<<256, 1024>>>(edges_pointer, offset,
                                                      edge_count);
          if (retval =
                  edges[gpu].SetPointer(coo + ((num_gpus > 1) ? num_gpus : 0) +
                                            directed_edges * copy_idx +
                                            start_edge * (undirected ? 2 : 1),
                                        edge_count, util::HOST))
            return retval;
          if (retval = edges[gpu].Move(util::DEVICE, util::HOST, edge_count, 0,
                                       streams[gpu]))
            return retval;
        }
      }
      SizeT start_edge = num_edges * 1.0 / num_gpus * gpu;
      SizeT end_edge = num_edges * 1.0 / num_gpus * (gpu + 1);
      SizeT edge_count = end_edge - start_edge;
      if (undirected) edge_count *= 2;
      unsigned int seed_ = seed + 616 * gpu;

      int block_size = (sizeof(VertexId) == 4) ? 1024 : 512;
      int grid_size = edge_count / block_size + 1;
      if (grid_size > 480) grid_size = 480;
      if (retval =
              rand_states[gpu].Allocate(block_size * grid_size, util::DEVICE)) {
        return retval;
      }
      Rand_Init<SizeT><<<grid_size, block_size, 0, streams[gpu]>>>(
          seed_, rand_states[gpu].GetPointer(util::DEVICE));

      Rmat_Kernel_nv<WITH_VALUES, VertexId, SizeT, Value>
          <<<grid_size, block_size, 0, streams[gpu]>>>(
              num_nodes, (undirected ? edge_count / 2 : edge_count), coo.row,
              coo.col, coo.val, undirected, vmultipiler, vmin, a0, b0, c0, d0,
              rand_states[gpu].GetPointer(util::DEVICE));
    }

    for (int gpu = 0; gpu < num_gpus; gpu++) {
      if (gpu_idx != NULL) {
        if (retval = util::SetDevice(gpu_idx[gpu])) {
          return retval;
        }
      }
      if (retval =
              util::GRError(cudaStreamSynchronize(streams[gpu]),
                            "cudaStreamSynchronize failed", __FILE__, __LINE__))
        return retval;
      if (retval =
              util::GRError(cudaStreamDestroy(streams[gpu]),
                            "cudaStreamDestroy failed", __FILE__, __LINE__))
        return retval;
      if (retval = rand_states[gpu].Release()) {
        return retval;
      }
    }

    delete[] rand_states;
    rand_states = NULL;
    delete[] streams;
    streams = NULL;

    return retval;
  }

}  // namespace grmat
}  // namespace grmat
}  // namespace graphio

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
