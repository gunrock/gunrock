// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * frontier.cuh
 *
 * @brief Defination of frontier
 */

#pragma once

#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace app {

/**
 * @brief Enumeration of global frontier queue configurations
 */
enum FrontierType {
  VERTEX_FRONTIER,  // O(|V|) ping-pong global vertex frontier
  EDGE_FRONTIER,    // O(|E|) ping-pong global edge frontier
};

/**
 * @brief Structure for frontier
 */
template <typename VertexT, typename SizeT = VertexT,
          util::ArrayFlag FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
struct Frontier {
  typedef util::Array1D<SizeT, SizeT, FLAG, cudaHostRegisterFlag> EdgeQT;
  typedef util::Array1D<SizeT, VertexT, FLAG, cudaHostRegisterFlag> VertexQT;

  std::string frontier_name;
  util::CtaWorkProgressLifetime<SizeT> work_progress;  // Queue size counters
  SizeT queue_length;        // the length of the current queue
  unsigned int queue_index;  // the index of the current queue
  bool queue_reset;          // whether to reset the next queue

  unsigned int num_queues;         // how many queues to support
  unsigned int num_vertex_queues;  // num of vertex queues
  unsigned int num_edge_queues;    // num of edge queues
  util::Location target;           // where the queues allocated
  util::Array1D<SizeT, FrontierType, FLAG, cudaHostRegisterFlag>
      queue_types;  // types of each queue
  util::Array1D<SizeT, unsigned int, FLAG, cudaHostRegisterFlag>
      queue_map;  // mapping queue index to vertex_queue / edge_queue indices

  util::Array1D<SizeT, SizeT, FLAG | util::PINNED,
                cudaHostRegisterFlag | cudaHostAllocMapped |
                    cudaHostAllocPortable>
      output_length;

  util::Array1D<SizeT, SizeT, FLAG, cudaHostRegisterFlag>
      num_segments;  // how many segments of each queue
  util::Array1D<SizeT, SizeT, FLAG, cudaHostRegisterFlag>
      *segment_offsets;  // offsets of segments for each queue

  util::Array1D<SizeT, SizeT, FLAG, cudaHostRegisterFlag> queue_offsets;  //

  util::Array1D<SizeT, SizeT, FLAG, cudaHostRegisterFlag>
      output_offsets;  // length / offsets for offsets of the frontier queues
  util::Array1D<SizeT, SizeT, FLAG, cudaHostRegisterFlag> block_input_starts;
  util::Array1D<SizeT, SizeT, FLAG, cudaHostRegisterFlag> block_output_starts;
  util::Array1D<uint64_t, char, FLAG, cudaHostRegisterFlag> cub_temp_space;

  // Frontier queues. Used to track working frontier.
  VertexQT *vertex_queues;  // vertex queues
  EdgeQT *edge_queues;      // edge queues

  Frontier() {
    SetName("");
    segment_offsets = NULL;
    vertex_queues = NULL;
    edge_queues = NULL;
  }

  cudaError_t SetName(std::string name) {
    cudaError_t retval = cudaSuccess;

    frontier_name = name;
    if (name != "") name = name + "::";
    // work_progress .SetName(name + "work_progress");
    GUARD_CU(queue_types.SetName(name + "queue_types"));
    GUARD_CU(queue_map.SetName(name + "queue_map"));
    GUARD_CU(output_length.SetName(name + "output_length"));
    GUARD_CU(num_segments.SetName(name + "num_segments"));
    GUARD_CU(queue_offsets.SetName(name + "queue_offsets"));
    GUARD_CU(output_offsets.SetName(name + "output_offsets"));
    GUARD_CU(cub_temp_space.SetName(name + "cub_temp_space"));
    return retval;
  }

  cudaError_t Init(unsigned int num_queues = 2, FrontierType *types = NULL,
                   std::string frontier_name = "",
                   util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(SetName(frontier_name));
    this->num_queues = num_queues;
    this->target = target;
    GUARD_CU(queue_types.Allocate(num_queues, util::HOST));
    GUARD_CU(queue_map.Allocate(num_queues, util::HOST));

    num_vertex_queues = 0;
    num_edge_queues = 0;
    for (unsigned int q = 0; q < num_queues; q++) {
      FrontierType queue_type = VERTEX_FRONTIER;
      if (types != NULL && types[q] == EDGE_FRONTIER)
        queue_type = EDGE_FRONTIER;

      queue_types[q] = queue_type;
      if (queue_type == VERTEX_FRONTIER) {
        queue_map[q] = num_vertex_queues;
        num_vertex_queues++;
      } else if (queue_type == EDGE_FRONTIER) {
        queue_map[q] = num_edge_queues;
        num_edge_queues++;
      }
    }

    if (frontier_name != "") frontier_name = frontier_name + "::";
    GUARD_CU(num_segments.Allocate(num_queues, util::HOST | target));
    segment_offsets =
        new util::Array1D<SizeT, SizeT, FLAG, cudaHostRegisterFlag>[num_queues];
    vertex_queues = new VertexQT[num_vertex_queues];
    edge_queues = new EdgeQT[num_edge_queues];
    for (unsigned int q = 0; q < num_queues; q++) {
      segment_offsets[q].SetName(frontier_name + "segment_offsets[" +
                                 std::to_string(q) + "]");
      if (queue_types[q] == VERTEX_FRONTIER) {
        auto &queue = vertex_queues[queue_map[q]];
        queue.SetName(frontier_name + "queues[" + std::to_string(q) + "]");
      } else if (queue_types[q] == EDGE_FRONTIER) {
        auto &queue = vertex_queues[queue_map[q]];
        queue.SetName(frontier_name + "queues[" + std::to_string(q) + "]");
      }
    }

    GUARD_CU(output_length.Allocate(1, target | util::HOST));
    GUARD_CU(queue_offsets.Allocate(num_queues, target | util::HOST));

    // TODO: work_progress.Init on HOST
    GUARD_CU(work_progress.Init());
    return retval;
  }

  cudaError_t Allocate(SizeT num_nodes, SizeT num_edges,
                       std::vector<double> &queue_factors) {
    cudaError_t retval = cudaSuccess;

    SizeT max_queue_size = 0;
    for (unsigned int q = 0; q < num_queues; q++) {
      double factor = queue_factors[q % queue_factors.size()];
      if (queue_types[q] == VERTEX_FRONTIER) {
        auto &queue = vertex_queues[queue_map[q]];
        GUARD_CU(queue.Allocate(num_nodes * factor, target));
        if (max_queue_size < num_nodes * factor)
          max_queue_size = num_nodes * factor;
      } else {
        auto &queue = edge_queues[queue_map[q]];
        GUARD_CU(queue.Allocate(num_edges * factor, target));
      }
    }
    GUARD_CU(output_offsets.Allocate(max_queue_size, target));
    GUARD_CU(block_input_starts.Allocate(2048, target));
    GUARD_CU(block_output_starts.Allocate(2048, target));
    return retval;
  }

  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;

    for (unsigned int q = 0; q < num_queues; q++) {
      GUARD_CU(segment_offsets[q].Release(target));

      if (queue_types[q] == VERTEX_FRONTIER) {
        auto &queue = vertex_queues[queue_map[q]];
        GUARD_CU(queue.Release(target));
      } else if (queue_types[q] == EDGE_FRONTIER) {
        auto &queue = edge_queues[queue_map[q]];
        GUARD_CU(queue.Release(target));
      }
    }

    GUARD_CU(queue_types.Release(target));
    GUARD_CU(queue_map.Release(target));
    GUARD_CU(output_length.Release(target));
    GUARD_CU(num_segments.Release(target));
    GUARD_CU(queue_offsets.Release(target));
    GUARD_CU(output_offsets.Release(target));
    GUARD_CU(cub_temp_space.Release(target));
    // TODO: make cta_work_progress::Release run on Host
    GUARD_CU(work_progress.Release());
    delete[] segment_offsets;
    segment_offsets = NULL;
    delete[] vertex_queues;
    vertex_queues = NULL;
    delete[] edge_queues;
    edge_queues = NULL;
    num_queues = 0;
    return retval;
  }

  VertexQT *V_Q(SizeT index = util::PreDefinedValues<SizeT>::InvalidValue) {
    if (index == util::PreDefinedValues<SizeT>::InvalidValue)
      index = queue_index;
    return (num_queues == 0) ? NULL
                             : vertex_queues + queue_map[index % num_queues];
  }

  VertexQT *Next_V_Q(
      SizeT index = util::PreDefinedValues<SizeT>::InvalidValue) {
    if (index == util::PreDefinedValues<SizeT>::InvalidValue)
      index = queue_index;
    index++;
    return V_Q(index);
  }

  EdgeQT *E_Q(SizeT index = util::PreDefinedValues<SizeT>::InvalidValue) {
    if (index == util::PreDefinedValues<SizeT>::InvalidValue)
      index = queue_index;
    return (num_queues == 0) ? NULL
                             : edge_queues + queue_map[index % num_queues];
  }

  EdgeQT *Next_E_Q(SizeT index = util::PreDefinedValues<SizeT>::InvalidValue) {
    if (index == util::PreDefinedValues<SizeT>::InvalidValue)
      index = queue_index;
    index++;
    return E_Q(index);
  }

  cudaError_t Reset(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    queue_reset = true;
    queue_index = 0;
    queue_length = 0;
    GUARD_CU(work_progress.Reset_());
    return retval;
  }

  cudaError_t GetQueueLength(cudaStream_t stream = 0) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(work_progress.GetQueueLength(queue_index, queue_length, false,
                                          stream, true));
    return retval;
  }
};  // struct Frontier

}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
