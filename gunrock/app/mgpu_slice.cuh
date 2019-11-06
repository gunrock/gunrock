// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * data_slice_base.cuh
 *
 * @brief Structure for base data slice. Only for temp dummping of code, will be
 * refactored latter
 */
#pragma once

namespace gunrock {
namespace app {

/**
 * @brief Base data slice structure which contains common data structural needed
 * for primitives.
 *
 * @tparam SizeT               Type of unsigned integer to use for array
 * indexing. (e.g., uint32)
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g.,
 * uint32)
 * @tparam Value               Type to use as vertex / edge associated values
 */
template <typename VertexT, typename SizeT, typename ValueT,
          util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
// int MAX_NUM_VERTEX_ASSOCIATES,
// int MAX_NUM_VALUE__ASSOCIATES>
struct MgpuSlice {
  int num_gpus;      // Number of GPUs
  int gpu_idx;       // GPU index
  int wait_counter;  // Wait counter for iteration loop control
  int max_num_vertex_associates;
  int max_num_value__associates;
  // int    gpu_mallocing       ; // Whether GPU is in malloc
  // int    num_vertex_associate; // Number of associate values in VertexId type
  // for each vertex int    num_value__associate; // Number of associate values
  // in Value type for each vertex
  int num_stages;  // Number of stages
  // SizeT  nodes               ; // Number of vertices
  // SizeT  edges               ; // Number of edges
  // bool   use_double_buffer   ;
  // typedef unsigned char MaskT;

  // Incoming VertexId type associate values
  util::Array1D<SizeT, VertexT, ARRAY_FLAG, cudaHostRegisterFlag>
      *vertex_associate_in[2];

  // Device pointers to incoming VertexId type associate values
  // util::Array1D<SizeT, VertexT*, ARRAY_FLAG, cudaHostRegisterFlag>
  //    *vertex_associate_ins [2];

  // Outgoing VertexId type associate values
  util::Array1D<SizeT, VertexT, ARRAY_FLAG, cudaHostRegisterFlag>
      *vertex_associate_out;

  // Device pointers to outgoing VertexId type associate values
  util::Array1D<SizeT, VertexT *, ARRAY_FLAG,
                cudaHostRegisterFlag | cudaHostAllocMapped |
                    cudaHostAllocPortable>
      vertex_associate_outs;

  // Device pointers to device points to outgoing VertexId type associate values
  // util::Array1D<SizeT, VertexT**, ARRAY_FLAG, cudaHostRegisterFlag>
  //    vertex_associate_outss  ;

  // Device pointers to original VertexId type associate values
  util::Array1D<SizeT, VertexT *, ARRAY_FLAG, cudaHostRegisterFlag>
      vertex_associate_orgs;

  // Incoming Value type associate values
  util::Array1D<SizeT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      *value__associate_in[2];

  // Device pointers to incoming Value type associate values
  // util::Array1D<SizeT, ValueT*      >
  //    *value__associate_ins [2];

  // Outgoing Value type associate values
  util::Array1D<SizeT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      *value__associate_out;

  // Device pointers to outgoing Value type associate values
  util::Array1D<SizeT, ValueT *, ARRAY_FLAG,
                cudaHostRegisterFlag | cudaHostAllocMapped |
                    cudaHostAllocPortable>
      value__associate_outs;

  // Device pointers to device pointers to outgoing Value type associate values
  // util::Array1D<SizeT, ValueT**     >
  //    value__associate_outss  ;

  // Device pointers to original Value type associate values
  util::Array1D<SizeT, ValueT *, ARRAY_FLAG, cudaHostRegisterFlag>
      value__associate_orgs;

  // Number of outgoing vertices to peers
  util::Array1D<SizeT, SizeT, ARRAY_FLAG,
                cudaHostRegisterFlag | cudaHostAllocMapped |
                    cudaHostAllocPortable>
      out_length;

  // Number of incoming vertices from peers
  util::Array1D<SizeT, SizeT, ARRAY_FLAG, cudaHostRegisterFlag> in_length[2];
  util::Array1D<SizeT, SizeT, ARRAY_FLAG,
                cudaHostRegisterFlag | cudaHostAllocMapped |
                    cudaHostAllocPortable>
      in_length_out;

  // Incoming iteration numbers
  util::Array1D<SizeT, VertexT, ARRAY_FLAG, cudaHostRegisterFlag>
      in_iteration[2];

  // Incoming vertices
  util::Array1D<SizeT, VertexT, ARRAY_FLAG, cudaHostRegisterFlag> *keys_in[2];

  // Outgoing vertices
  util::Array1D<SizeT, VertexT *, ARRAY_FLAG,
                cudaHostRegisterFlag | cudaHostAllocMapped |
                    cudaHostAllocPortable>
      keys_outs;

  // Device pointers to outgoing vertices
  util::Array1D<SizeT, VertexT, ARRAY_FLAG, cudaHostRegisterFlag> *keys_out;

  // Markers to separate vertices to peer GPUs
  // util::Array1D<SizeT, SizeT   , ARRAY_FLAG, cudaHostRegisterFlag>
  //    *keys_marker             ;

  // Device pointer to the markers
  // util::Array1D<SizeT, SizeT*  , ARRAY_FLAG, cudaHostRegisterFlag>
  //    keys_markers            ;

  // Vertex lookup array
  // util::Array1D<SizeT, SizeT   , ARRAY_FLAG, cudaHostRegisterFlag>
  //    *visit_lookup            ;

  // Vertex valid in
  // util::Array1D<SizeT, VertexT , ARRAY_FLAG, cudaHostRegisterFlag>
  //    *valid_in                ;

  // Vertex valid out
  // util::Array1D<SizeT, VertexT , ARRAY_FLAG, cudaHostRegisterFlag>
  //    *valid_out               ;

  // GPU stream events arrays
  util::Array1D<SizeT, cudaEvent_t *, ARRAY_FLAG, cudaHostRegisterFlag>
      events[4];

  // Whether the GPU stream events are set
  util::Array1D<SizeT, bool *, ARRAY_FLAG, cudaHostRegisterFlag> events_set[4];

  //
  util::Array1D<SizeT, int, ARRAY_FLAG, cudaHostRegisterFlag> wait_marker;

  // GPU streams
  // util::Array1D<SizeT, cudaStream_t, ARRAY_FLAG, cudaHostRegisterFlag>
  //    streams                 ;

  // current stages of each streams
  util::Array1D<SizeT, int, ARRAY_FLAG, cudaHostRegisterFlag> stages;

  // whether to show debug information for the streams
  util::Array1D<SizeT, bool, ARRAY_FLAG, cudaHostRegisterFlag> to_show;

  // compressed data structure for make_out kernel
  // util::Array1D<SizeT, char    , ARRAY_FLAG, cudaHostRegisterFlag>
  //    make_out_array          ;

  // compressed data structure for expand_incoming kernel
  // util::Array1D<SizeT, char    , ARRAY_FLAG, cudaHostRegisterFlag>
  //    *expand_incoming_array   ;

  // predecessors of vertices
  // util::Array1D<SizeT, VertexT , ARRAY_FLAG, cudaHostRegisterFlag>
  //    preds                   ;

  // temporary storages for predecessors
  // util::Array1D<SizeT, VertexT , ARRAY_FLAG, cudaHostRegisterFlag>
  //    temp_preds              ;

  // Used for source distance
  // util::Array1D<SizeT, VertexT , ARRAY_FLAG, cudaHostRegisterFlag>
  //    labels                  ;

  // util::Array1D<SizeT, MaskT   , ARRAY_FLAG, cudaHostRegisterFlag>
  //    visited_mask;

  util::Array1D<SizeT, int, ARRAY_FLAG, cudaHostRegisterFlag> latency_data;

  // arrays used to track data race, containing info about pervious assigment
  util::Array1D<SizeT, int, ARRAY_FLAG, cudaHostRegisterFlag>
      org_checkpoint;  // checkpoint number
  util::Array1D<SizeT, VertexT *, ARRAY_FLAG, cudaHostRegisterFlag>
      org_d_out;  // d_out address
  util::Array1D<SizeT, SizeT, ARRAY_FLAG, cudaHostRegisterFlag>
      org_offset1;  // offset1
  util::Array1D<SizeT, SizeT, ARRAY_FLAG, cudaHostRegisterFlag>
      org_offset2;  // offset2
  util::Array1D<SizeT, VertexT, ARRAY_FLAG, cudaHostRegisterFlag>
      org_queue_idx;  // queue index
  util::Array1D<SizeT, int, ARRAY_FLAG, cudaHostRegisterFlag>
      org_block_idx;  // blockIdx.x
  util::Array1D<SizeT, int, ARRAY_FLAG, cudaHostRegisterFlag>
      org_thread_idx;  // threadIdx.x

  /**
   * @brief DataSliceBase default constructor
   */
  MgpuSlice() {
    // Assign default values
    num_stages = 4;
    // num_vertex_associate     = 0;
    // num_value__associate     = 0;
    // gpu_idx                  = 0;
    // gpu_mallocing            = 0;
    // use_double_buffer        = false;

    // Assign NULs to pointers
    keys_out = NULL;
    keys_in[0] = NULL;
    keys_in[1] = NULL;
    vertex_associate_in[0] = NULL;
    vertex_associate_in[1] = NULL;
    vertex_associate_out = NULL;
    value__associate_in[0] = NULL;
    value__associate_in[1] = NULL;
    value__associate_out = NULL;

    // Assign names to arrays
    keys_outs.SetName("keys_outs");
    vertex_associate_outs.SetName("vertex_associate_outs");
    value__associate_outs.SetName("value__associate_outs");
    vertex_associate_orgs.SetName("vertex_associate_orgs");
    value__associate_orgs.SetName("value__associate_orgs");
    out_length.SetName("out_length");
    in_length[0].SetName("in_length[0]");
    in_length[1].SetName("in_length[1]");
    in_length_out.SetName("in_length_out");
    in_iteration[0].SetName("in_iteration[0]");
    in_iteration[1].SetName("in_iteration[0]");
    wait_marker.SetName("wait_marker");
    stages.SetName("stages");
    to_show.SetName("to_show");
    org_checkpoint.SetName("org_checkpoint");
    org_d_out.SetName("org_d_out");
    org_offset1.SetName("org_offset1");
    org_offset2.SetName("org_offset2");
    org_queue_idx.SetName("org_queue_idx");
    org_block_idx.SetName("org_block_idx");
    org_thread_idx.SetName("org_thread_idx");
    latency_data.SetName("latency_data");

    for (int i = 0; i < 4; i++) {
      events[i].SetName("events[]");
      events_set[i].SetName("events_set[]");
    }
  }  // end DataSliceBase()

  /**
   * @brief DataSliceBase default destructor to release host / device memory
   */
  /*virtual ~MgpuSlice()
  {
      Release();
  }*/

  cudaError_t Release(util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    // Set device by index
    GUARD_CU(util::SetDevice(gpu_idx));

    // Release VertexId type incoming associate values and related pointers
    if (vertex_associate_in[0] != NULL) {
      for (int gpu = 0; gpu < num_gpus; gpu++) {
        GUARD_CU(vertex_associate_in[0][gpu].Release(target));
        GUARD_CU(vertex_associate_in[1][gpu].Release(target));
      }
      if (target & util::HOST) {
        delete[] vertex_associate_in[0];
        delete[] vertex_associate_in[1];
        vertex_associate_in[0] = NULL;
        vertex_associate_in[1] = NULL;
      }
    }

    // Release Value type incoming associate values and related pointers
    if (value__associate_in[0] != NULL) {
      for (int gpu = 0; gpu < num_gpus; gpu++) {
        GUARD_CU(value__associate_in[0][gpu].Release(target));
        GUARD_CU(value__associate_in[1][gpu].Release(target));
      }
      if (target & util::HOST) {
        delete[] value__associate_in[0];
        delete[] value__associate_in[1];
        value__associate_in[0] = NULL;
        value__associate_in[1] = NULL;
      }
    }

    // Release incoming keys and related pointers
    if (keys_in[0] != NULL) {
      for (int gpu = 0; gpu < num_gpus; gpu++) {
        GUARD_CU(keys_in[0][gpu].Release(target));
        GUARD_CU(keys_in[1][gpu].Release(target));
      }
      if (target & util::HOST) {
        delete[] keys_in[0];
        delete[] keys_in[1];
        keys_in[0] = NULL;
        keys_in[1] = NULL;
      }
    }

    // Release VertexId type outgoing associate values and pointers
    if (vertex_associate_out != NULL) {
      for (int gpu = 0; gpu < num_gpus; gpu++) {
        if (target & util::HOST) vertex_associate_outs[gpu] = NULL;
        GUARD_CU(vertex_associate_out[gpu].Release(target));
      }
      if (target & util::HOST) {
        delete[] vertex_associate_out;
        vertex_associate_out = NULL;
      }
      GUARD_CU(vertex_associate_outs.Release(target));
    }

    // Release Value type outgoing associate values and pointers
    if (value__associate_out != NULL) {
      for (int gpu = 0; gpu < num_gpus; gpu++) {
        if (target & util::HOST) value__associate_outs[gpu] = NULL;
        GUARD_CU(value__associate_out[gpu].Release(target));
      }
      if (target & util::HOST) {
        delete[] value__associate_out;
        value__associate_out = NULL;
      }
      GUARD_CU(value__associate_outs.Release(target));
    }

    // Release events and markers
    if (target & util::HOST)
      for (int i = 0; i < 4; i++) {
        if (events[i].GetPointer() != NULL)
          for (int gpu = 0; gpu < num_gpus * 2; gpu++) {
            for (int stage = 0; stage < num_stages; stage++)
              GUARD_CU2(cudaEventDestroy(events[i][gpu][stage]),
                        "cudaEventDestroy failed");
            delete[] events[i][gpu];
            events[i][gpu] = NULL;
            delete[] events_set[i][gpu];
            events_set[i][gpu] = NULL;
          }
        GUARD_CU(events[i].Release(target));
        GUARD_CU(events_set[i].Release(target));
      }

    // Release frontiers
    /*if (frontier_queues != NULL)
    {
        for (int gpu = 0; gpu <= num_gpus; gpu++)
        {
            for (int i = 0; i < 2; ++i)
            {
                GUARD_CU(frontier_queues[gpu].keys  [i].Release());
                GUARD_CU(frontier_queues[gpu].values[i].Release());
            }
        }
        delete[] frontier_queues; frontier_queues = NULL;
    }

    // Release scanned_edges
    if (scanned_edges != NULL)
    {
        for (int gpu = 0; gpu <= num_gpus; gpu++)
            GUARD_CU(scanned_edges          [gpu].Release());
        delete[] scanned_edges;
        scanned_edges           = NULL;
    }*/

    // Release all other arrays
    GUARD_CU(keys_outs.Release(target));
    GUARD_CU(in_length[0].Release(target));
    GUARD_CU(in_length[1].Release(target));
    GUARD_CU(in_length_out.Release(target));
    GUARD_CU(in_iteration[0].Release(target));
    GUARD_CU(in_iteration[1].Release(target));
    GUARD_CU(wait_marker.Release(target));
    GUARD_CU(out_length.Release(target));
    GUARD_CU(vertex_associate_orgs.Release(target));
    GUARD_CU(value__associate_orgs.Release(target));
    GUARD_CU(stages.Release(target));
    GUARD_CU(to_show.Release(target));
    GUARD_CU(latency_data.Release(target));

    GUARD_CU(org_checkpoint.Release(target));
    GUARD_CU(org_d_out.Release(target));
    GUARD_CU(org_offset1.Release(target));
    GUARD_CU(org_offset2.Release(target));
    GUARD_CU(org_queue_idx.Release(target));
    GUARD_CU(org_block_idx.Release(target));
    GUARD_CU(org_thread_idx.Release(target));
    return retval;
  }  // end Release()

  /**
   * @brief Initiate DataSliceBase
   *
   * @param[in] num_gpus             Number of GPUs
   * @param[in] gpu_idx              GPU index
   * @param[in] use_double_buffer
   * @param[in] graph                Pointer to the CSR formated sub-graph
   * @param[in] num_in_nodes         Number of incoming vertices from peers
   * @param[in] num_out_nodes        Number of outgoing vertices to peers
   * @param[in] in_sizing            Preallocation factor for incoming /
   * outgoing vertices
   * @param[in] skip_makeout_selection
   * \return                         Error occurred if any, otherwise
   * cudaSuccess
   */
  cudaError_t Init(int num_gpus, int gpu_idx,
                   // bool   use_double_buffer   ,
                   // Csr<VertexId, SizeT, Value>
                   //      *graph               ,
                   SizeT num_nodes, SizeT max_queue_length, SizeT *num_in_nodes,
                   SizeT *num_out_nodes, double trans_factor = 1.0,
                   bool skip_makeout_selection = false) {
    cudaError_t retval = cudaSuccess;
    // Copy input values
    this->num_gpus = num_gpus;
    this->gpu_idx = gpu_idx;
    // this->use_double_buffer    = use_double_buffer;
    // this->nodes                = graph->nodes;
    // this->edges                = graph->edges;
    // this->num_vertex_associate = num_vertex_associate;
    // this->num_value__associate = num_value__associate;

    // Set device by index
    GUARD_CU(util::SetDevice(gpu_idx));

    GUARD_CU(in_length[0].Allocate(num_gpus, util::HOST));
    GUARD_CU(in_length[1].Allocate(num_gpus, util::HOST));
    // GUARD_CU(in_length_out.Init(num_gpus, util::HOST | util::DEVICE,
    //    true, cudaHostAllocMapped | cudaHostAllocPortable));
    GUARD_CU(in_length_out.Allocate(num_gpus, util::HOST | util::DEVICE));
    GUARD_CU(in_iteration[0].Allocate(num_gpus, util::HOST));
    GUARD_CU(in_iteration[1].Allocate(num_gpus, util::HOST));
    // GUARD_CU(out_length .Init(num_gpus, util::HOST | util::DEVICE,
    //    true, cudaHostAllocMapped | cudaHostAllocPortable));
    GUARD_CU(out_length.Allocate(num_gpus, util::HOST | util::DEVICE));
    GUARD_CU(vertex_associate_orgs.Allocate(max_num_vertex_associates,
                                            util::HOST | util::DEVICE));
    GUARD_CU(value__associate_orgs.Allocate(max_num_value__associates,
                                            util::HOST | util::DEVICE));
    GUARD_CU(latency_data.Allocate(120 * 1024, util::HOST | util::DEVICE));
    for (SizeT i = 0; i < 120 * 1024; i++) latency_data[i] = rand();
    GUARD_CU(latency_data.Move(util::HOST, util::DEVICE));

    // Allocate / create event related variables
    GUARD_CU(wait_marker.Allocate(num_gpus * 2, util::HOST));
    GUARD_CU(stages.Allocate(num_gpus * 2, util::HOST));
    GUARD_CU(to_show.Allocate(num_gpus * 2, util::HOST));
    for (int gpu = 0; gpu < num_gpus; gpu++) {
      wait_marker[gpu] = 0;
    }
    for (int i = 0; i < 4; i++) {
      GUARD_CU(events[i].Allocate(num_gpus * 2, util::HOST));
      GUARD_CU(events_set[i].Allocate(num_gpus * 2, util::HOST));
      for (int gpu = 0; gpu < num_gpus * 2; gpu++) {
        events[i][gpu] = new cudaEvent_t[num_stages];
        events_set[i][gpu] = new bool[num_stages];
        for (int stage = 0; stage < num_stages; stage++) {
          GUARD_CU2(cudaEventCreateWithFlags(&(events[i][gpu][stage]),
                                             cudaEventDisableTiming),
                    "cudaEventCreate failed.");
          events_set[i][gpu][stage] = false;
        }
      }
    }
    for (int gpu = 0; gpu < num_gpus; gpu++) {
      for (int i = 0; i < 2; i++) {
        in_length[i][gpu] = 0;
        in_iteration[i][gpu] = 0;
      }
    }

    if (num_gpus == 1) return retval;
    // Create incoming buffer on device
    keys_in[0] = new util::Array1D<SizeT, VertexT>[num_gpus];
    keys_in[1] = new util::Array1D<SizeT, VertexT>[num_gpus];
    vertex_associate_in[0] = new util::Array1D<SizeT, VertexT>[num_gpus];
    vertex_associate_in[1] = new util::Array1D<SizeT, VertexT>[num_gpus];
    value__associate_in[0] = new util::Array1D<SizeT, ValueT>[num_gpus];
    value__associate_in[1] = new util::Array1D<SizeT, ValueT>[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++) {
      for (int t = 0; t < 2; t++) {
        SizeT num_in_node = num_in_nodes[gpu] * trans_factor;

        vertex_associate_in[t][gpu].SetName("vertex_associate_in[][]");
        GUARD_CU(vertex_associate_in[t][gpu].Allocate(
            num_in_node * max_num_vertex_associates, util::DEVICE));

        value__associate_in[t][gpu].SetName("vertex_associate_in[][]");
        GUARD_CU(value__associate_in[t][gpu].Allocate(
            num_in_node * max_num_value__associates, util::DEVICE));

        keys_in[t][gpu].SetName("keys_in");
        if (gpu != 0) {
          GUARD_CU(keys_in[t][gpu].Allocate(num_in_node, util::DEVICE));
        }
      }
    }

    // Allocate outgoing buffer on device
    vertex_associate_out = new util::Array1D<SizeT, VertexT>[num_gpus];
    value__associate_out = new util::Array1D<SizeT, ValueT>[num_gpus];
    keys_out = new util::Array1D<SizeT, VertexT>[num_gpus];

    // GUARD_CU(vertex_associate_outs. Init(num_gpus, util::HOST | util::DEVICE,
    //    true, cudaHostAllocMapped | cudaHostAllocPortable));
    GUARD_CU(
        vertex_associate_outs.Allocate(num_gpus, util::HOST | util::DEVICE));

    // GUARD_CU(value__associate_outs. Init(num_gpus, util::HOST | util::DEVICE,
    //    true, cudaHostAllocMapped | cudaHostAllocPortable));
    GUARD_CU(
        value__associate_outs.Allocate(num_gpus, util::HOST | util::DEVICE));

    // GUARD_CU(keys_outs            . Init(num_gpus, util::HOST | util::DEVICE,
    //    true, cudaHostAllocMapped | cudaHostAllocPortable));
    GUARD_CU(keys_outs.Allocate(num_gpus, util::HOST | util::DEVICE));
    for (int gpu = 0; gpu < num_gpus; gpu++) {
      SizeT num_out_node = num_nodes * trans_factor;
      keys_out[gpu].SetName("keys_out[]");
      if (gpu != 0) {
        GUARD_CU(keys_out[gpu].Allocate(num_out_node, util::DEVICE));
        keys_outs[gpu] = keys_out[gpu].GetPointer(util::DEVICE);
      }

      vertex_associate_out[gpu].SetName("vertex_associate_outs[]");
      if (gpu != 0) {
        GUARD_CU(vertex_associate_out[gpu].Allocate(
            num_out_node * max_num_vertex_associates, util::DEVICE));
        vertex_associate_outs[gpu] =
            vertex_associate_out[gpu].GetPointer(util::DEVICE);
      }

      value__associate_out[gpu].SetName("value__associate_outs[]");
      if (gpu != 0) {
        GUARD_CU(value__associate_out[gpu].Allocate(
            num_out_node * max_num_value__associates, util::DEVICE));
        value__associate_outs[gpu] =
            value__associate_out[gpu].GetPointer(util::DEVICE);
      }
      if (skip_makeout_selection && gpu == 1) break;
    }

    if (skip_makeout_selection) {
      for (int gpu = 2; gpu < num_gpus; gpu++) {
        keys_out[gpu].SetPointer(keys_out[1].GetPointer(util::DEVICE),
                                 keys_out[1].GetSize(), util::DEVICE);
        keys_outs[gpu] = keys_out[gpu].GetPointer(util::DEVICE);

        vertex_associate_out[gpu].SetPointer(
            vertex_associate_out[1].GetPointer(util::DEVICE),
            vertex_associate_out[1].GetSize(), util::DEVICE);
        vertex_associate_outs[gpu] =
            vertex_associate_out[gpu].GetPointer(util::DEVICE);

        value__associate_out[gpu].SetPointer(
            value__associate_out[1].GetPointer(util::DEVICE),
            value__associate_out[1].GetSize(), util::DEVICE);
        value__associate_outs[gpu] =
            value__associate_out[gpu].GetPointer(util::DEVICE);
      }
    }
    GUARD_CU(keys_outs.Move(util::HOST, util::DEVICE));
    GUARD_CU(vertex_associate_outs.Move(util::HOST, util::DEVICE));
    GUARD_CU(value__associate_outs.Move(util::HOST, util::DEVICE));

    if (false)  //(TO_TRACK)
    {
      GUARD_CU(org_checkpoint.Allocate(max_queue_length, util::DEVICE));
      GUARD_CU(org_d_out.Allocate(max_queue_length, util::DEVICE));
      GUARD_CU(org_offset1.Allocate(max_queue_length, util::DEVICE));
      GUARD_CU(org_offset2.Allocate(max_queue_length, util::DEVICE));
      GUARD_CU(org_queue_idx.Allocate(max_queue_length, util::DEVICE));
      GUARD_CU(org_block_idx.Allocate(max_queue_length, util::DEVICE));
      GUARD_CU(org_thread_idx.Allocate(max_queue_length, util::DEVICE));
    }
    return retval;
  }  // end Init(..)

  /**
   * @brief Performs reset work needed for mgpu slice. Must be called prior to
   * each search \return cudaError_t object which indicates the success of all
   * CUDA function calls.
   */
  cudaError_t Reset(util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    // if (retval = util::SetDevice(gpu_idx)) return retval;
    for (int gpu = 0; gpu < num_gpus * 2; gpu++) wait_marker[gpu] = 0;
    for (int i = 0; i < 4; i++)
      for (int gpu = 0; gpu < num_gpus * 2; gpu++)
        for (int stage = 0; stage < num_stages; stage++)
          events_set[i][gpu][stage] = false;
    for (int gpu = 0; gpu < num_gpus; gpu++)
      for (int i = 0; i < 2; i++) in_length[i][gpu] = 0;
    for (int peer = 0; peer < num_gpus; peer++) out_length[peer] = 1;
    for (int peer = 0; peer < num_gpus; peer++) out_length[peer] = 1;

    return retval;
  }  // end Reset(...)

};  // end DataSliceBase

}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
