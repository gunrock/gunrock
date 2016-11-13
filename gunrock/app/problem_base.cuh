// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * problem_base.cuh
 *
 * @brief Base structure for all the application types
 */

#pragma once

#include <vector>
#include <string>

// Graph construction utilities
#include <gunrock/graphio/market.cuh>
#include <gunrock/graphio/rmat.cuh>
#include <gunrock/graphio/grmat.cuh>
#include <gunrock/graphio/rgg.cuh>
#include <gunrock/graphio/small_world.cuh>

// Information stats utilities
#include <boost/filesystem.hpp>
#include <gunrock/util/sysinfo.h>
#include <gunrock/util/gitsha1.h>
#include <gunrock/util/json_spirit_writer_template.h>

// Gunrock test error utilities
#include <gunrock/util/basic_utils.h>
#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/multiple_buffering.cuh>
#include <gunrock/util/io/modified_load.cuh>
#include <gunrock/util/io/modified_store.cuh>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/test_utils.h>
#include <gunrock/util/track_utils.cuh>

// Graph partitioner utilities
#include <gunrock/app/rp/rp_partitioner.cuh>
#include <gunrock/app/cp/cp_partitioner.cuh>
#include <gunrock/app/brp/brp_partitioner.cuh>
#include <gunrock/app/metisp/metis_partitioner.cuh>
#include <gunrock/app/sp/sp_partitioner.cuh>
#include <gunrock/app/dup/dup_partitioner.cuh>

#include <moderngpu.cuh>

// this is the "stringize macro macro" hack
#define STR(x) #x
#define XSTR(x) STR(x)

namespace gunrock {
namespace app {

/**
 * @brief Enumeration of global frontier queue configurations
 */
enum FrontierType
{
    VERTEX_FRONTIERS,       // O(n) ping-pong global vertex frontiers
    EDGE_FRONTIERS,         // O(m) ping-pong global edge frontiers
    MIXED_FRONTIERS         // O(n) global vertex frontier, O(m) global edge frontier
};

/**
 * @brief Graph slice structure which contains common graph structural data.
 *
 * @tparam SizeT    Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam VertexId Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam Value    Type to use as vertex / edge associated values
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value >
struct GraphSlice
{
    int             num_gpus; // Number of GPUs
    int             index   ; // Slice index
    VertexId        nodes   ; // Number of nodes in slice
    SizeT           edges   ; // Number of edges in slice
    SizeT           inverse_edges; // Number of inverse_edges in slice

    Csr<VertexId, SizeT, Value   > *graph             ; // Pointer to CSR format subgraph
    util::Array1D<SizeT, SizeT   > row_offsets        ; // CSR format row offset
    util::Array1D<SizeT, VertexId> column_indices     ; // CSR format column indices
    util::Array1D<SizeT, SizeT   > out_degrees        ;
    util::Array1D<SizeT, SizeT   > column_offsets     ; // CSR format column offset
    util::Array1D<SizeT, VertexId> row_indices        ; // CSR format row indices
    util::Array1D<SizeT, SizeT   > in_degrees         ;
    util::Array1D<SizeT, int     > partition_table    ; // Partition number for vertices, local is always 0
    util::Array1D<SizeT, VertexId> convertion_table   ; // IDs of vertices in their hosting partition
    util::Array1D<SizeT, VertexId> original_vertex    ; // Original IDs of vertices
    util::Array1D<SizeT, SizeT   > in_counter         ; // Incoming vertices counter from peers
    util::Array1D<SizeT, SizeT   > out_offset         ; // Outgoing vertices offsets
    util::Array1D<SizeT, SizeT   > out_counter        ; // Outgoing vertices counter
    util::Array1D<SizeT, SizeT   > backward_offset    ; // Backward offsets for partition and conversion tables
    util::Array1D<SizeT, int     > backward_partition ; // Remote peers having the same vertices
    util::Array1D<SizeT, VertexId> backward_convertion; // IDs of vertices in remote peers

    /**
     * @brief GraphSlice Constructor
     *
     * @param[in] index GPU index.
     */
    GraphSlice(int index) :
        index(index),
        graph(NULL),
        num_gpus(0),
        nodes(0),
        edges(0)
    {
        row_offsets        .SetName("row_offsets"        );
        column_indices     .SetName("column_indices"     );
        out_degrees        .SetName("out_degrees"        );
        column_offsets     .SetName("column_offsets"     );
        row_indices        .SetName("row_indices"        );
        in_degrees         .SetName("in_degrees"         );
        partition_table    .SetName("partition_table"    );
        convertion_table   .SetName("convertion_table"   );
        original_vertex    .SetName("original_vertex"    );
        in_counter         .SetName("in_counter"         );
        out_offset         .SetName("out_offset"         );
        out_counter        .SetName("out_counter"        );
        backward_offset    .SetName("backward_offset"    );
        backward_partition .SetName("backward_partition" );
        backward_convertion.SetName("backward_convertion");
    }  // end GraphSlice(int index)

    /**
     * @brief GraphSlice Destructor to free all device memories.
     */
    virtual ~GraphSlice()
    {
        Release();
    }

    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;

        // Set device (use slice index)
        if (retval = util::SetDevice(index)) return retval;

        // Release allocated host / device memory
        if (retval = row_offsets        .Release()) return retval;
        if (retval = column_indices     .Release()) return retval;
        if (retval = out_degrees        .Release()) return retval;
        if (retval = column_offsets     .Release()) return retval;
        if (retval = row_indices        .Release()) return retval;
        if (retval = in_degrees         .Release()) return retval;
        if (retval = partition_table    .Release()) return retval;
        if (retval = convertion_table   .Release()) return retval;
        if (retval = original_vertex    .Release()) return retval;
        if (retval = in_counter         .Release()) return retval;
        if (retval = out_offset         .Release()) return retval;
        if (retval = out_counter        .Release()) return retval;
        if (retval = backward_offset    .Release()) return retval;
        if (retval = backward_partition .Release()) return retval;
        if (retval = backward_convertion.Release()) return retval;

        return retval;
    } // end ~GraphSlice()

    /**
      * @brief Initialize graph slice
      *
      * @param[in] stream_from_host    Whether to stream data from host
      * @param[in] num_gpus            Number of GPUs
      * @param[in] graph               Pointer to the sub graph
      * @param[in] inverstgraph        Pointer to the invert graph
      * @param[in] partition_table     The partition table
      * @param[in] convertion_table    The conversion table
      * @param[in] original_vertex     The original vertex table
      * @param[in] in_counter          In_counters
      * @param[in] out_offset          Out_offsets
      * @param[in] out_counter         Out_counters
      * @param[in] backward_offsets    Backward_offsets
      * @param[in] backward_partition  The backward partition table
      * @param[in] backward_convertion The backward conversion table
      * \return cudaError_t            Object indicating the success of all CUDA function calls
      */
    cudaError_t Init(
        bool                       stream_from_host,
        int                        num_gpus,
        Csr<VertexId, SizeT, Value>* graph,
        Csr<VertexId, SizeT, Value>* inverstgraph,
        int*                       partition_table,
        VertexId*                  convertion_table,
        VertexId*                  original_vertex,
        SizeT*                     in_counter,
        SizeT*                     out_offset,
        SizeT*                     out_counter,
        SizeT*                     backward_offsets   = NULL,
        int*                       backward_partition = NULL,
        VertexId*                  backward_convertion = NULL)
    {
        cudaError_t retval     = cudaSuccess;

        // Set local variables / array pointers
        this->num_gpus         = num_gpus;
        this->graph            = graph;
        this->nodes            = graph->nodes;
        this->edges            = graph->edges;
        if (inverstgraph != NULL)
            this->inverse_edges    = inverstgraph -> edges;
        else this -> inverse_edges = 0;
        if (partition_table  != NULL) this->partition_table    .SetPointer(partition_table      , nodes     );
        if (convertion_table != NULL) this->convertion_table   .SetPointer(convertion_table     , nodes     );
        if (original_vertex  != NULL) this->original_vertex    .SetPointer(original_vertex      , nodes     );
        if (in_counter       != NULL) this->in_counter         .SetPointer(in_counter           , num_gpus + 1);
        if (out_offset       != NULL) this->out_offset         .SetPointer(out_offset           , num_gpus + 1);
        if (out_counter      != NULL) this->out_counter        .SetPointer(out_counter          , num_gpus + 1);
        this->row_offsets        .SetPointer(graph->row_offsets   , nodes + 1   );
        this->column_indices     .SetPointer(graph->column_indices, edges     );
        if (inverstgraph != NULL)
        {
            this->column_offsets .SetPointer(inverstgraph->row_offsets, nodes + 1);
            this->row_indices    .SetPointer(inverstgraph->column_indices   , inverstgraph -> edges  );
        }

        // Set device using slice index
        if (retval = util::SetDevice(index)) return retval;

        // Allocate and initialize row_offsets
        if (retval = this->row_offsets   .Allocate(nodes + 1 , util::DEVICE)) return retval;
        if (retval = this->row_offsets   .Move    (util::HOST, util::DEVICE)) return retval;

        // Allocate and initialize column_indices
        if (retval = this->column_indices.Allocate(edges     , util::DEVICE)) return retval;
        if (retval = this->column_indices.Move    (util::HOST, util::DEVICE)) return retval;

        // Allocate out degrees for each node
        if (retval = this->out_degrees   .Allocate(nodes     , util::DEVICE)) return retval;
        // count number of out-going degrees for each node
        util::MemsetMadVectorKernel <<< 128, 128>>>(
            this->out_degrees.GetPointer(util::DEVICE),
            this->row_offsets.GetPointer(util::DEVICE),
            this->row_offsets.GetPointer(util::DEVICE) + 1,
            (SizeT)-1, nodes);


        if (inverstgraph != NULL)
        {
            // Allocate and initialize column_offsets
            if (retval = this->column_offsets.Allocate(nodes + 1 , util::DEVICE)) return retval;
            if (retval = this->column_offsets.Move    (util::HOST, util::DEVICE)) return retval;

            // Allocate and initialize row_indices
            if (retval = this->row_indices   .Allocate(inverstgraph -> edges, util::DEVICE)) return retval;
            if (retval = this->row_indices   .Move    (util::HOST, util::DEVICE)) return retval;

            if (retval = this->in_degrees    .Allocate(nodes     , util::DEVICE)) return retval;
            // count number of in-going degrees for each node
            util::MemsetMadVectorKernel <<< 128, 128>>>(
                this->in_degrees    .GetPointer(util::DEVICE),
                this->column_offsets.GetPointer(util::DEVICE),
                this->column_offsets.GetPointer(util::DEVICE) + 1,
                (SizeT)-1, nodes);
        }

        // For multi-GPU cases
        if (num_gpus > 1)
        {
            // Allocate and initialize convertion_table
            if (retval = this->partition_table.Allocate (nodes     , util::DEVICE)) return retval;
            if (partition_table  != NULL)
                if (retval = this->partition_table.Move (util::HOST, util::DEVICE)) return retval;

            // Allocate and initialize convertion_table
            if (retval = this->convertion_table.Allocate(nodes     , util::DEVICE)) return retval;
            if (convertion_table != NULL)
                if (retval = this->convertion_table.Move(util::HOST, util::DEVICE)) return retval;

            // Allocate and initialize original_vertex
            if (retval = this->original_vertex .Allocate(nodes     , util::DEVICE)) return retval;
            if (original_vertex  != NULL)
                if (retval = this->original_vertex .Move(util::HOST, util::DEVICE)) return retval;

            // If need backward information proration
            if (backward_offsets != NULL)
            {
                // Allocate and initialize backward_offset
                this->backward_offset    .SetPointer(backward_offsets     , nodes + 1);
                if (retval = this->backward_offset    .Allocate(nodes + 1, util::DEVICE)) return retval;
                if (retval = this->backward_offset    .Move(util::HOST, util::DEVICE)) return retval;

                // Allocate and initialize backward_partition
                this->backward_partition .SetPointer(backward_partition   , backward_offsets[nodes]);
                if (retval = this->backward_partition .Allocate(backward_offsets[nodes], util::DEVICE)) return retval;
                if (retval = this->backward_partition .Move(util::HOST, util::DEVICE)) return retval;

                // Allocate and initialize backward_convertion
                this->backward_convertion.SetPointer(backward_convertion  , backward_offsets[nodes]);
                if (retval = this->backward_convertion.Allocate(backward_offsets[nodes], util::DEVICE)) return retval;
                if (retval = this->backward_convertion.Move(util::HOST, util::DEVICE)) return retval;
            }
        } // end if num_gpu>1

        return retval;
    } // end of Init(...)

    /**
     * @brief overloaded = operator
     *
     * @param[in] other GraphSlice to copy from
     *
     * \return GraphSlice& a copy of local GraphSlice
     */
    GraphSlice& operator=(GraphSlice other)
    {
        num_gpus            = other.num_gpus           ;
        index               = other.index              ;
        nodes               = other.nodes              ;
        edges               = other.edges              ;
        graph               = other.graph              ;
        row_offsets         = other.row_offsets        ;
        column_indices      = other.column_indices     ;
        column_offsets      = other.column_offsets     ;
        row_indices         = other.row_indices        ;
        partition_table     = other.partition_table    ;
        convertion_table    = other.convertion_table   ;
        original_vertex     = other.original_vertex    ;
        in_counter          = other.in_counter         ;
        out_offset          = other.out_offset         ;
        out_counter         = other.out_counter        ;
        backward_offset     = other.backward_offset    ;
        backward_partition  = other.backward_partition ;
        backward_convertion = other.backward_convertion;
        return *this;
    } // end operator=()

}; // end GraphSlice

/**
 * @brief Base data slice structure which contains common data structural needed for primitives.
 *
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam Value               Type to use as vertex / edge associated values
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value,
    int MAX_NUM_VERTEX_ASSOCIATES,
    int MAX_NUM_VALUE__ASSOCIATES>
struct DataSliceBase
{
    int    num_gpus            ; // Number of GPUs
    int    gpu_idx             ; // GPU index
    int    wait_counter        ; // Wait counter for iteration loop control
    int    gpu_mallocing       ; // Whether GPU is in malloc
    //int    num_vertex_associate; // Number of associate values in VertexId type for each vertex
    //int    num_value__associate; // Number of associate values in Value type for each vertex
    int    num_stages          ; // Number of stages
    SizeT  nodes               ; // Number of vertices
    SizeT  edges               ; // Number of edges
    bool   use_double_buffer   ;
    typedef unsigned char MaskT;

    util::Array1D<SizeT, VertexId    >  *vertex_associate_in  [2]; // Incoming VertexId type associate values
    //util::Array1D<SizeT, VertexId*   >  *vertex_associate_ins [2]; // Device pointers to incoming VertexId type associate values
    util::Array1D<SizeT, VertexId    >  *vertex_associate_out    ; // Outgoing VertexId type associate values
    util::Array1D<SizeT, VertexId*   >   vertex_associate_outs   ; // Device pointers to outgoing VertexId type associate values
    //util::Array1D<SizeT, VertexId**  >   vertex_associate_outss  ; // Device pointers to device points to outgoing VertexId type associate values
    util::Array1D<SizeT, VertexId*   >   vertex_associate_orgs   ; // Device pointers to original VertexId type associate values
    util::Array1D<SizeT, Value       >  *value__associate_in  [2]; // Incoming Value type associate values
    //util::Array1D<SizeT, Value*      >  *value__associate_ins [2]; // Device pointers to incoming Value type associate values
    util::Array1D<SizeT, Value       >  *value__associate_out    ; // Outgoing Value type associate values
    util::Array1D<SizeT, Value*      >   value__associate_outs   ; // Device pointers to outgoing Value type associate values
    //util::Array1D<SizeT, Value**     >   value__associate_outss  ; // Device pointers to device pointers to outgoing Value type associate values
    util::Array1D<SizeT, Value*      >   value__associate_orgs   ; // Device pointers to original Value type associate values
    util::Array1D<SizeT, SizeT       >   out_length              ; // Number of outgoing vertices to peers
    util::Array1D<SizeT, SizeT       >   in_length            [2]; // Number of incoming vertices from peers
    util::Array1D<SizeT, SizeT       >   in_length_out           ;
    util::Array1D<SizeT, VertexId    >   in_iteration         [2]; // Incoming iteration numbers
    util::Array1D<SizeT, VertexId    >  *keys_in              [2]; // Incoming vertices
    util::Array1D<SizeT, VertexId*   >   keys_outs               ; // Outgoing vertices
    util::Array1D<SizeT, VertexId    >  *keys_out                ; // Device pointers to outgoing vertices
    //util::Array1D<SizeT, SizeT       >  *keys_marker             ; // Markers to separate vertices to peer GPUs
    //util::Array1D<SizeT, SizeT*      >   keys_markers            ; // Device pointer to the markers

    //util::Array1D<SizeT, SizeT       >  *visit_lookup            ; // Vertex lookup array
    //util::Array1D<SizeT, VertexId    >  *valid_in                ; // Vertex valid in
    //util::Array1D<SizeT, VertexId    >  *valid_out               ; // Vertex valid out

    util::Array1D<SizeT, cudaEvent_t*>   events               [4]; // GPU stream events arrays
    util::Array1D<SizeT, bool*       >   events_set           [4]; // Whether the GPU stream events are set
    util::Array1D<SizeT, int         >   wait_marker             ; //
    util::Array1D<SizeT, cudaStream_t>   streams                 ; // GPU streams
    util::Array1D<SizeT, int         >   stages                  ; // current stages of each streams
    util::Array1D<SizeT, bool        >   to_show                 ; // whether to show debug information for the streams
    //util::Array1D<SizeT, char        >   make_out_array          ; // compressed data structure for make_out kernel
    //util::Array1D<SizeT, char        >  *expand_incoming_array   ; // compressed data structure for expand_incoming kernel
    util::Array1D<SizeT, VertexId    >   preds                   ; // predecessors of vertices
    util::Array1D<SizeT, VertexId    >   temp_preds              ; // temporary storages for predecessors
    util::Array1D<SizeT, VertexId    >   labels                  ; // Used for source distance

    //Frontier queues. Used to track working frontier.
    util::DoubleBuffer<VertexId, SizeT, Value>  *frontier_queues ; // frontier queues
    util::Array1D<SizeT, SizeT       >  *scanned_edges           ; // length / offsets for offsets of the frontier queues
    util::Array1D<SizeT, unsigned char> *cub_scan_space;
    util::Array1D<SizeT, MaskT        > visited_mask;
    util::Array1D<SizeT, int          > latency_data;

    // arrays used to track data race, containing info about pervious assigment
    util::Array1D<SizeT, int         > org_checkpoint            ; // checkpoint number
    util::Array1D<SizeT, VertexId*   > org_d_out                 ; // d_out address
    util::Array1D<SizeT, SizeT       > org_offset1               ; // offset1
    util::Array1D<SizeT, SizeT       > org_offset2               ; // offset2
    util::Array1D<SizeT, VertexId    > org_queue_idx             ; // queue index
    util::Array1D<SizeT, int         > org_block_idx             ; // blockIdx.x
    util::Array1D<SizeT, int         > org_thread_idx            ; // threadIdx.x

    /**
     * @brief DataSliceBase default constructor
     */
    DataSliceBase()
    {
        // Assign default values
        num_stages               = 4;
        //num_vertex_associate     = 0;
        //num_value__associate     = 0;
        gpu_idx                  = 0;
        gpu_mallocing            = 0;
        use_double_buffer        = false;

        // Assign NULs to pointers
        keys_out                 = NULL;
        //keys_marker              = NULL;
        keys_in              [0] = NULL;
        keys_in              [1] = NULL;
        //visit_lookup             = NULL;
        //valid_in                 = NULL;
        //valid_out                = NULL;
        vertex_associate_in  [0] = NULL;
        vertex_associate_in  [1] = NULL;
        //vertex_associate_ins [0] = NULL;
        //vertex_associate_ins [1] = NULL;
        vertex_associate_out     = NULL;
        //vertex_associate_outs    = NULL;
        value__associate_in  [0] = NULL;
        value__associate_in  [1] = NULL;
        //value__associate_ins [0] = NULL;
        //value__associate_ins [1] = NULL;
        value__associate_out     = NULL;
        //value__associate_outs    = NULL;
        frontier_queues          = NULL;
        scanned_edges            = NULL;
        //cub_scan_space           = NULL;
        //expand_incoming_array    = NULL;

        // Assign names to arrays
        keys_outs              .SetName("keys_outs"              );
        vertex_associate_outs  .SetName("vertex_associate_outs"  );
        value__associate_outs  .SetName("value__associate_outs"  );
        //vertex_associate_outss .SetName("vertex_associate_outss" );
        //value__associate_outss .SetName("value__associate_outss" );
        vertex_associate_orgs  .SetName("vertex_associate_orgs"  );
        value__associate_orgs  .SetName("value__associate_orgs"  );
        out_length             .SetName("out_length"             );
        in_length           [0].SetName("in_length[0]"           );
        in_length           [1].SetName("in_length[1]"           );
        in_length_out          .SetName("in_length_out"          );
        in_iteration        [0].SetName("in_iteration[0]"        );
        in_iteration        [1].SetName("in_iteration[0]"        );
        wait_marker            .SetName("wait_marker"            );
        //keys_markers           .SetName("keys_marker"            );
        stages                 .SetName("stages"                 );
        to_show                .SetName("to_show"                );
        //make_out_array         .SetName("make_out_array"         );
        streams                .SetName("streams"                );
        preds                  .SetName("preds"                  );
        temp_preds             .SetName("temp_preds"             );
        labels                 .SetName("labels"                 );
        visited_mask           .SetName("visited_mask"           );
        org_checkpoint         .SetName("org_checkpoint"         );
        org_d_out              .SetName("org_d_out"              );
        org_offset1            .SetName("org_offset1"            );
        org_offset2            .SetName("org_offset2"            );
        org_queue_idx          .SetName("org_queue_idx"          );
        org_block_idx          .SetName("org_block_idx"          );
        org_thread_idx         .SetName("org_thread_idx"         );
        latency_data           .SetName("latency_data"           );

        for (int i = 0; i < 4; i++)
        {
            events[i].SetName("events[]");
            events_set[i].SetName("events_set[]");
        }
    } // end DataSliceBase()

    /**
     * @brief DataSliceBase default destructor to release host / device memory
     */
    virtual ~DataSliceBase()
    {
        Release();
    }

    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;
        // Set device by index
        if (retval = util::SetDevice(gpu_idx)) return retval;

        // Release VertexId type incoming associate values and related pointers
        if (vertex_associate_in[0] != NULL)
        {
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                //for (int i = 0; i < MAX_NUM_VERTEX_ASSOCIATES; i++)
                //{
                //    if (retval = vertex_associate_in[0][gpu][i].Release()) return retval;
                //    if (retval = vertex_associate_in[1][gpu][i].Release()) return retval;
                //}
                //delete[] vertex_associate_in[0][gpu];
                //delete[] vertex_associate_in[1][gpu];
                //vertex_associate_in [0][gpu] = NULL;
                //vertex_associate_in [1][gpu] = NULL;
                if (retval = vertex_associate_in[0][gpu].Release()) return retval;
                if (retval = vertex_associate_in[1][gpu].Release()) return retval;
            }
            delete[] vertex_associate_in [0];
            delete[] vertex_associate_in [1];
            //delete[] vertex_associate_ins[0];
            //delete[] vertex_associate_ins[1];
            vertex_associate_in [0] = NULL;
            vertex_associate_in [1] = NULL;
            //vertex_associate_ins[0] = NULL;
            //vertex_associate_ins[1] = NULL;
        }

        // Release Value type incoming associate values and related pointers
        if (value__associate_in[0] != NULL)
        {
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                //for (int i = 0; i < MAX_NUM_VALUE__ASSOCIATES; i++)
                //{
                //    if (retval = value__associate_in[0][gpu][i].Release()) return retval;
                //    if (retval = value__associate_in[1][gpu][i].Release()) return retval;
                //}
                //delete[] value__associate_in[0][gpu];
                //delete[] value__associate_in[1][gpu];
                //value__associate_in [0][gpu] = NULL;
                //value__associate_in [1][gpu] = NULL;
                if (retval = value__associate_in[0][gpu].Release()) return retval;
                if (retval = value__associate_in[1][gpu].Release()) return retval;
            }
            delete[] value__associate_in [0];
            delete[] value__associate_in [1];
            //delete[] value__associate_ins[0];
            //delete[] value__associate_ins[1];
            value__associate_in [0] = NULL;
            value__associate_in [1] = NULL;
            //value__associate_ins[0] = NULL;
            //value__associate_ins[1] = NULL;
        }

        // Release incoming keys and related pointers
        if (keys_in[0] != NULL)
        {
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                if (retval = keys_in[0][gpu].Release()) return retval;
                if (retval = keys_in[1][gpu].Release()) return retval;
            }
            delete[] keys_in[0];
            delete[] keys_in[1];
            keys_in[0] = NULL;
            keys_in[1] = NULL;
        }

        /*if (visit_lookup != NULL)
        {
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                if (retval = visit_lookup[gpu].Release()) return retval;
            }
            delete[] visit_lookup;
            visit_lookup = NULL;
        }

        if (valid_in != NULL)
        {
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                if (retval = valid_in[gpu].Release()) return retval;
            }
            delete[] valid_in;
            valid_in = NULL;
        }

        if (valid_out != NULL)
        {
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                if (retval = valid_out[gpu].Release()) return retval;
            }
            delete[] valid_out;
            valid_out = NULL;
        }*/

        // Release outgoing keys and markers
        //if (keys_marker != NULL)
        //{
        //    for (int gpu = 0; gpu < num_gpus; gpu++)
        //    {
        //        if (retval = keys_out   [gpu].Release()) return retval;
        //        if (retval = keys_marker[gpu].Release()) return retval;
        //    }
        //    delete[] keys_out   ; keys_out    = NULL;
        //    delete[] keys_marker; keys_marker = NULL;
        //    if (retval = keys_markers.Release()) return retval;
        //}

        // Release VertexId type outgoing associate values and pointers
        if (vertex_associate_out != NULL)
        {
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                //for (int i = 0; i < MAX_NUM_VERTEX_ASSOCIATES; i++)
                //    if (retval = vertex_associate_out[gpu][i].Release()) return retval;
                //delete[] vertex_associate_out[gpu];
                vertex_associate_outs [gpu] = NULL;
                if (retval = vertex_associate_out[gpu].Release()) return retval;
            }
            delete[] vertex_associate_out;
            //delete[] vertex_associate_outs;
            vertex_associate_out = NULL;
            //vertex_associate_outs = NULL;
            if (retval = vertex_associate_outs.Release()) return retval;
        }

        // Release Value type outgoing associate values and pointers
        if (value__associate_out != NULL)
        {
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                //for (int i = 0; i < MAX_NUM_VALUE__ASSOCIATES; i++)
                //    if (retval = value__associate_out[gpu][i].Release()) return retval;
                //delete[] value__associate_out[gpu];
                value__associate_outs [gpu] = NULL;
                if (retval = value__associate_out[gpu].Release()) return retval;
            }
            delete[] value__associate_out ;
            //delete[] value__associate_outs;
            value__associate_out = NULL;
            //value__associate_outs = NULL;
            if (retval = value__associate_outs.Release()) return retval;
        }

        // Release events and markers
        for (int i = 0; i < 4; i++)
        {
            if (events[i].GetPointer() != NULL)
            for (int gpu = 0; gpu < num_gpus * 2; gpu++)
            {
                for (int stage = 0; stage < num_stages; stage++)
                    if (retval = util::GRError(cudaEventDestroy(events[i][gpu][stage]),
                        "cudaEventDestroy failed", __FILE__, __LINE__)) return retval;
                delete[] events    [i][gpu]; events    [i][gpu] = NULL;
                delete[] events_set[i][gpu]; events_set[i][gpu] = NULL;
            }
            if (retval = events    [i].Release()) return retval;
            if (retval = events_set[i].Release()) return retval;
        }

        // Release expand_incoming_arrays
        //if (expand_incoming_array != NULL)
        //{
        //    for (int gpu = 0; gpu < num_gpus; gpu++)
        //        if (retval = expand_incoming_array[gpu].Release()) return retval;
        //    delete[] expand_incoming_array;
        //    expand_incoming_array = NULL;
        //}

        // Release frontiers
        if (frontier_queues != NULL)
        {
            for (int gpu = 0; gpu <= num_gpus; gpu++)
            {
                for (int i = 0; i < 2; ++i)
                {
                    if (retval = frontier_queues[gpu].keys  [i].Release()) return retval;
                    if (retval = frontier_queues[gpu].values[i].Release()) return retval;
                }
            }
            delete[] frontier_queues; frontier_queues = NULL;
        }

        // Release scanned_edges
        if (scanned_edges != NULL)
        {
            for (int gpu = 0; gpu <= num_gpus; gpu++)
                if (retval = scanned_edges          [gpu].Release()) return retval;
            delete[] scanned_edges;
            scanned_edges           = NULL;
        }

        /*if (cub_scan_space != NULL)
        {
            for (int gpu = 0; gpu <= num_gpus; gpu++)
                if (retval = cub_scan_space[gpu].Release()) return retval;
            delete[] cub_scan_space;
            cub_scan_space = NULL;
        }*/

        //Release all other arrays
        if (retval = keys_outs     .Release()) return retval;
        if (retval = in_length  [0].Release()) return retval;
        if (retval = in_length  [1].Release()) return retval;
        if (retval = in_length_out .Release()) return retval;
        if (retval = in_iteration[0].Release()) return retval;
        if (retval = in_iteration[1].Release()) return retval;
        if (retval = wait_marker   .Release()) return retval;
        if (retval = out_length    .Release()) return retval;
        if (retval = vertex_associate_orgs.Release()) return retval;
        if (retval = value__associate_orgs.Release()) return retval;
        if (retval = streams       .Release()) return retval;
        if (retval = stages        .Release()) return retval;
        if (retval = to_show       .Release()) return retval;
        //if (retval = make_out_array.Release()) return retval;
        if (retval = preds         .Release()) return retval;
        if (retval = temp_preds    .Release()) return retval;
        if (retval = labels        .Release()) return retval;
        if (retval = visited_mask  .Release()) return retval;
        if (retval = latency_data  .Release()) return retval;

        if (retval = org_checkpoint.Release()) return retval;
        if (retval = org_d_out     .Release()) return retval;
        if (retval = org_offset1   .Release()) return retval;
        if (retval = org_offset2   .Release()) return retval;
        if (retval = org_queue_idx .Release()) return retval;
        if (retval = org_block_idx .Release()) return retval;
        if (retval = org_thread_idx.Release()) return retval;
        return retval;
    } // end Release()

    /**
     * @brief Initiate DataSliceBase
     *
     * @param[in] num_gpus             Number of GPUs
     * @param[in] gpu_idx              GPU index
     * @param[in] use_double_buffer
     * @param[in] graph                Pointer to the CSR formated sub-graph
     * @param[in] num_in_nodes         Number of incoming vertices from peers
     * @param[in] num_out_nodes        Number of outgoing vertices to peers
     * @param[in] in_sizing            Preallocation factor for incoming / outgoing vertices
     * @param[in] skip_makeout_selection
     * \return                         Error occurred if any, otherwise cudaSuccess
     */
    cudaError_t Init(
        int    num_gpus            ,
        int    gpu_idx             ,
        bool   use_double_buffer   ,
        Csr<VertexId, SizeT, Value>
              *graph               ,
        SizeT *num_in_nodes        ,
        SizeT *num_out_nodes       ,
        float  in_sizing = 1.0     ,
        bool   skip_makeout_selection = false)
    {
        cudaError_t retval         = cudaSuccess;
        // Copy input values
        this->num_gpus             = num_gpus;
        this->gpu_idx              = gpu_idx;
        this->use_double_buffer    = use_double_buffer;
        this->nodes                = graph->nodes;
        this->edges                = graph->edges;
        //this->num_vertex_associate = num_vertex_associate;
        //this->num_value__associate = num_value__associate;

        // Set device by index
        if (retval = util::SetDevice(gpu_idx))  return retval;

        // Allocate frontiers and scanned_edges
        this->frontier_queues      = new util::DoubleBuffer<VertexId, SizeT, Value>[num_gpus + 1];
        this->scanned_edges        = new util::Array1D<SizeT, SizeT>[num_gpus + 1];
        //this->cub_scan_space       = new util::Array1D<SizeT, unsigned char>[num_gpus + 1];
        for (int i = 0; i < num_gpus + 1; i++)
        {
            this->scanned_edges[i].SetName("scanned_edges[]");
        }
        if (retval = in_length[0].Allocate(num_gpus, util::HOST)) return retval;
        if (retval = in_length[1].Allocate(num_gpus, util::HOST)) return retval;
        if (retval = in_length_out.Init(num_gpus, util::HOST | util::DEVICE, true, cudaHostAllocMapped | cudaHostAllocPortable)) return retval;
        if (retval = in_iteration[0].Allocate(num_gpus, util::HOST)) return retval;
        if (retval = in_iteration[1].Allocate(num_gpus, util::HOST)) return retval;
        //if (retval = out_length  .Allocate(num_gpus, util::HOST | util::DEVICE)) return retval;
        if (retval = out_length .Init(num_gpus, util::HOST | util::DEVICE, true, cudaHostAllocMapped | cudaHostAllocPortable));
        if (retval = vertex_associate_orgs.Allocate(
            MAX_NUM_VERTEX_ASSOCIATES, util::HOST | util::DEVICE)) return retval;
        if (retval = value__associate_orgs.Allocate(
            MAX_NUM_VALUE__ASSOCIATES, util::HOST | util::DEVICE)) return retval;
        if (retval = latency_data         .Allocate(
            120 * 1024, util::HOST | util::DEVICE)) return retval;
        for (SizeT i = 0; i< 120 * 1024; i++)
            latency_data[i] = rand();
        if (retval = latency_data.Move(util::HOST, util::DEVICE)) return retval;

        // Allocate / create event related variables
        wait_marker .Allocate(num_gpus * 2);
        stages      .Allocate(num_gpus * 2);
        to_show     .Allocate(num_gpus * 2);
        for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            wait_marker[gpu] = 0;
        }
        for (int i = 0; i < 4; i++)
        {
            events    [i].Allocate(num_gpus * 2);
            events_set[i].Allocate(num_gpus * 2);
            for (int gpu = 0; gpu < num_gpus * 2; gpu++)
            {
                events    [i][gpu] = new cudaEvent_t[num_stages];
                events_set[i][gpu] = new bool       [num_stages];
                for (int stage = 0; stage < num_stages; stage++)
                {
                    if (retval = util::GRError(
                        //cudaEventCreate(&(events[i][gpu][stage])),
                        cudaEventCreateWithFlags(&(events[i][gpu][stage]), cudaEventDisableTiming),
                       "cudaEventCreate failed.", __FILE__, __LINE__))
                        return retval;
                    events_set[i][gpu][stage] = false;
                }
            }
        }
        for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            for (int i = 0; i < 2; i++)
            {
                in_length[i][gpu] = 0;
                in_iteration[i][gpu] = 0;
            }
        }

        //visit_lookup            = new util::Array1D<SizeT, SizeT    > [num_gpus];
        //valid_in                = new util::Array1D<SizeT, VertexId > [num_gpus];
        //valid_out               = new util::Array1D<SizeT, VertexId > [num_gpus];

        if (num_gpus == 1) return retval;
        // Create incoming buffer on device
        keys_in             [0] = new util::Array1D<SizeT, VertexId > [num_gpus];
        keys_in             [1] = new util::Array1D<SizeT, VertexId > [num_gpus];
        vertex_associate_in [0] = new util::Array1D<SizeT, VertexId > [num_gpus];
        vertex_associate_in [1] = new util::Array1D<SizeT, VertexId > [num_gpus];
        //vertex_associate_ins[0] = new util::Array1D<SizeT, VertexId*> [num_gpus];
        //vertex_associate_ins[1] = new util::Array1D<SizeT, VertexId*> [num_gpus];
        value__associate_in [0] = new util::Array1D<SizeT, Value    > [num_gpus];
        value__associate_in [1] = new util::Array1D<SizeT, Value    > [num_gpus];
        //value__associate_ins[0] = new util::Array1D<SizeT, Value   *> [num_gpus];
        //value__associate_ins[1] = new util::Array1D<SizeT, Value   *> [num_gpus];
        for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            for (int t = 0; t < 2; t++)
            {
                SizeT num_in_node = num_in_nodes[gpu] * in_sizing;
                //vertex_associate_in [t][gpu] =
                //    new util::Array1D<SizeT, VertexId>[MAX_NUM_VERTEX_ASSOCIATES];
                //for (int i = 0; i < MAX_NUM_VERTEX_ASSOCIATES; i++)
                //{
                //    vertex_associate_in [t][gpu][i].SetName("vertex_associate_in[]");
                //    if (gpu == 0) continue;
                //    if (retval = vertex_associate_in[t][gpu][i]
                //        .Allocate(num_in_node, util::DEVICE))
                //        return retval;
                //}
                vertex_associate_in[t][gpu].SetName("vertex_associate_in[][]");
                if (retval = vertex_associate_in[t][gpu].Allocate(num_in_node * MAX_NUM_VERTEX_ASSOCIATES, util::DEVICE))
                    return retval;

                //value__associate_in [t][gpu] =
                //    new util::Array1D<SizeT, Value   >[MAX_NUM_VALUE__ASSOCIATES];
                //for (int i = 0; i < MAX_NUM_VALUE__ASSOCIATES; i++)
                //{
                //    value__associate_in[t][gpu][i].SetName("value__associate_ins[]");
                //    if (gpu == 0) continue;
                //    if (retval = value__associate_in[t][gpu][i]
                //        .Allocate(num_in_node, util::DEVICE))
                //        return retval;
                //}
                value__associate_in[t][gpu].SetName("vertex_associate_in[][]");
                if (retval = value__associate_in[t][gpu].Allocate(num_in_node * MAX_NUM_VALUE__ASSOCIATES, util::DEVICE))
                    return retval;

                //vertex_associate_ins[t][gpu].SetName("vertex_associate_ins");
                //if (retval = vertex_associate_ins[t][gpu].Allocate(
                //    MAX_NUM_VERTEX_ASSOCIATES, util::DEVICE | util::HOST)) return retval;
                //for (int i = 0; i < MAX_NUM_VERTEX_ASSOCIATES; i++)
                //    vertex_associate_ins[t][gpu][i] =
                //        vertex_associate_in[t][gpu][i].GetPointer(util::DEVICE);
                //if (retval = vertex_associate_ins[t][gpu].Move(util::HOST, util::DEVICE))
                //    return retval;

                //value__associate_ins[t][gpu].SetName("value__associate_ins");
                //if (retval = value__associate_ins[t][gpu].Allocate(
                //    MAX_NUM_VALUE__ASSOCIATES, util::DEVICE | util::HOST)) return retval;
                //for (int i = 0; i < MAX_NUM_VALUE__ASSOCIATES; i++)
                //    value__associate_ins[t][gpu][i] =
                //        value__associate_in[t][gpu][i].GetPointer(util::DEVICE);
                //if (retval = value__associate_ins[t][gpu].Move(util::HOST, util::DEVICE))
                //    return retval;

                keys_in[t][gpu].SetName("keys_in");
                if (gpu != 0)
                    if (retval = keys_in[t][gpu].Allocate(num_in_node, util::DEVICE))
                        return retval;
            }
        }

        /*for (int gpu = 0; gpu < num_gpus; ++gpu)
        {
            SizeT num_in_node = num_in_nodes[gpu] * in_sizing;
            SizeT num_out_node = num_out_nodes[gpu] * in_sizing;
            visit_lookup[gpu].SetName("visit_lookup");
            if (gpu != 0)
                if (retval = visit_lookup[gpu].Allocate(num_out_node, util::DEVICE))
                    return retval;
            valid_in[gpu].SetName("valid_in");
            if (gpu != 0)
                if (retval = valid_in[gpu].Allocate(num_in_node, util::DEVICE))
                    return retval;
            valid_out[gpu].SetName("valid_out");
            if (gpu != 0)
                if (retval = valid_out[gpu].Allocate(num_in_node, util::DEVICE))
                    return retval;
        }*/

        // Allocate outgoing buffer on device
        vertex_associate_out  = new util::Array1D<SizeT, VertexId > [num_gpus];
        //vertex_associate_outs = new util::Array1D<SizeT, VertexId*> [num_gpus];
        value__associate_out  = new util::Array1D<SizeT, Value    > [num_gpus];
        //value__associate_outs = new util::Array1D<SizeT, Value*   > [num_gpus];
        //keys_marker           = new util::Array1D<SizeT, SizeT    > [num_gpus];
        keys_out              = new util::Array1D<SizeT, VertexId > [num_gpus];
        //if (retval = vertex_associate_outss.Allocate(num_gpus, util::HOST | util::DEVICE)) return retval;
        if (retval = vertex_associate_outs. Init(num_gpus, util::HOST | util::DEVICE, true, cudaHostAllocMapped | cudaHostAllocPortable))
            return retval;
        //if (retval = value__associate_outss.Allocate(num_gpus, util::HOST | util::DEVICE)) return retval;
        if (retval = value__associate_outs. Init(num_gpus, util::HOST | util::DEVICE, true, cudaHostAllocMapped | cudaHostAllocPortable))
            return retval;
        //if (retval = keys_markers          .Allocate(num_gpus, util::HOST | util::DEVICE)) return retval;
        //if (retval = keys_outs             .Allocate(num_gpus, util::HOST | util::DEVICE)) return retval;
        if (retval = keys_outs. Init(num_gpus, util::HOST | util::DEVICE, true, cudaHostAllocMapped | cudaHostAllocPortable))
            return retval;
        for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            //SizeT num_out_node = num_out_nodes[gpu] * in_sizing;
            SizeT num_out_node = nodes * in_sizing;
            //keys_marker[gpu].SetName("keys_marker[]");
            //if (retval = keys_marker[gpu].Allocate(
            //    num_out_nodes[num_gpus] * in_sizing, util::DEVICE)) return retval;
            //keys_markers[gpu] = keys_marker[gpu].GetPointer(util::DEVICE);
            keys_out   [gpu].SetName("keys_out[]");
            if (gpu != 0)
            {
                if (retval = keys_out[gpu].Allocate(num_out_node, util::DEVICE))
                    return retval;
                keys_outs[gpu] = keys_out[gpu].GetPointer(util::DEVICE);
            }

            //vertex_associate_out  [gpu] = new util::Array1D<SizeT, VertexId>
            //    [MAX_NUM_VERTEX_ASSOCIATES];
            vertex_associate_out [gpu].SetName("vertex_associate_outs[]");
            //if (retval = vertex_associate_outs[gpu].Allocate(
            //    MAX_NUM_VERTEX_ASSOCIATES, util::HOST | util::DEVICE)) return retval;
            //vertex_associate_outss[gpu] = vertex_associate_outs[gpu].GetPointer(util::DEVICE);
            //for (int i = 0; i < MAX_NUM_VERTEX_ASSOCIATES; i++)
            //{
            //    vertex_associate_out[gpu][i].SetName("vertex_associate_out[][]");
            //    if (gpu != 0)
            //        if (retval = vertex_associate_out[gpu][i].Allocate(
            //            num_out_node, util::DEVICE))
            //            return retval;
            //    vertex_associate_outs[gpu][i] =
            //        vertex_associate_out[gpu][i].GetPointer(util::DEVICE);
            //}
            //if (retval = vertex_associate_outs[gpu].Move(util::HOST, util::DEVICE)) return retval;
            if (gpu != 0)
            {
                if (retval = vertex_associate_out[gpu].Allocate(num_out_node * MAX_NUM_VERTEX_ASSOCIATES, util::DEVICE))
                    return retval;
                vertex_associate_outs[gpu] = vertex_associate_out[gpu].GetPointer(util::DEVICE);
            }

            //value__associate_out [gpu] = new util::Array1D<SizeT, Value>
            //    [MAX_NUM_VALUE__ASSOCIATES];
            value__associate_out[gpu].SetName("value__associate_outs[]");
            //if (retval = value__associate_outs[gpu].Allocate(
            //    MAX_NUM_VALUE__ASSOCIATES, util::HOST | util::DEVICE)) return retval;
            //value__associate_outss[gpu] = value__associate_outs[gpu].GetPointer(util::DEVICE);
            //for (int i = 0; i < MAX_NUM_VALUE__ASSOCIATES; i++)
            //{
            //    value__associate_out[gpu][i].SetName("value__associate_out[][]");
            //    if (gpu != 0)
            //        if (retval = value__associate_out[gpu][i].Allocate(num_out_node, util::DEVICE))
            //            return retval;
            //    value__associate_outs[gpu][i] =
            //        value__associate_out[gpu][i].GetPointer(util::DEVICE);
            //}
            //if (retval = value__associate_outs[gpu].Move(util::HOST, util::DEVICE)) return retval;
            if (gpu != 0)
            {
                if (retval = value__associate_out[gpu].Allocate(num_out_node * MAX_NUM_VALUE__ASSOCIATES, util::DEVICE))
                    return retval;
                value__associate_outs[gpu] = value__associate_out[gpu].GetPointer(util::DEVICE);
            }
            if (skip_makeout_selection && gpu == 1) break;
        }
        if (skip_makeout_selection)
        {
            for (int gpu = 2; gpu < num_gpus; gpu++)
            {
                keys_out[gpu].SetPointer(
                    keys_out[1].GetPointer(util::DEVICE),
                    keys_out[1].GetSize(), util::DEVICE);
                keys_outs[gpu] = keys_out[gpu].GetPointer(util::DEVICE);

                vertex_associate_out[gpu].SetPointer(
                    vertex_associate_out[1].GetPointer(util::DEVICE),
                    vertex_associate_out[1].GetSize(), util::DEVICE);
                vertex_associate_outs[gpu] = vertex_associate_out[gpu].GetPointer(util::DEVICE);

                value__associate_out[gpu].SetPointer(
                    value__associate_out[1].GetPointer(util::DEVICE),
                    value__associate_out[1].GetSize(), util::DEVICE);
                value__associate_outs[gpu] = value__associate_out[gpu].GetPointer(util::DEVICE);
            }
        }
        //if (retval = keys_markers          .Move(util::HOST, util::DEVICE)) return retval;
        if (retval = keys_outs            .Move(util::HOST, util::DEVICE)) return retval;
        if (retval = vertex_associate_outs.Move(util::HOST, util::DEVICE)) return retval;
        if (retval = value__associate_outs.Move(util::HOST, util::DEVICE)) return retval;

        // Allocate make_out_array and expand_incoming array
        //if (retval = make_out_array.Allocate(
        //     sizeof(SizeT*   ) * num_gpus +
        //     sizeof(VertexId*) * num_gpus +
        //     sizeof(VertexId*) * MAX_NUM_VERTEX_ASSOCIATES +
        //     sizeof(Value*   ) * MAX_NUM_VALUE__ASSOCIATES +
        //     sizeof(VertexId*) * MAX_NUM_VERTEX_ASSOCIATES * num_gpus +
        //     sizeof(Value*   ) * MAX_NUM_VALUE__ASSOCIATES * num_gpus +
        //     sizeof(SizeT    ) * num_gpus,
        //     util::HOST | util::DEVICE)) return retval;
        //expand_incoming_array = new util::Array1D<SizeT, char>[num_gpus];
        //for (int i = 0; i < num_gpus; i++)
        //{
        //    expand_incoming_array[i].SetName("expand_incoming_array[]");
        //    if (retval = expand_incoming_array[i].Allocate(
        //         sizeof(Value*   ) * MAX_NUM_VERTEX_ASSOCIATES * 2 +
        //         sizeof(VertexId*) * MAX_NUM_VALUE__ASSOCIATES * 2,
        //         util::HOST | util::DEVICE)) return retval;
        //}

        return retval;
    } // end Init(..)

    /**
     * @brief Performs reset work needed for DataSliceBase. Must be called prior to each search
     *
     * @param[in] frontier_type      The frontier type (i.e., edge/vertex/mixed)
     * @param[in] graph_slice        Pointer to the corresponding graph slice
     * @param[in] queue_sizing       Sizing scaling factor for work queue allocation. 1.0 by default. Reserved for future use.
     * @param[in] _USE_DOUBLE_BUFFER Whether to use double buffer
     * @param[in] queue_sizing1      Scaling factor for frontier_queue1
     * @param[in] skip_scanned_edges
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Reset(
        FrontierType frontier_type,
        GraphSlice<VertexId, SizeT, Value>
        *graph_slice,
        double  queue_sizing       = 2.0,
        bool    _USE_DOUBLE_BUFFER = false,
        double  queue_sizing1      = -1.0,
        bool    skip_scanned_edges = false)
    {
        //printf("DataSliceBase reset, queue_sizing = %lf, %lf, nodes = %lld, edges = %lld\n",
        //    queue_sizing, queue_sizing1,
        //    (long long)graph_slice -> nodes, (long long)graph_slice -> edges);

        cudaError_t retval = cudaSuccess;
        if (retval = util::SetDevice(gpu_idx)) return retval;
        for (int gpu = 0; gpu < num_gpus * 2; gpu++)
            wait_marker[gpu] = 0;
        for (int i=0; i<4; i++)
        for (int gpu = 0; gpu < num_gpus * 2; gpu++)
        for (int stage=0; stage < num_stages; stage++)
            events_set[i][gpu][stage] = false;
        for (int gpu = 0; gpu < num_gpus; gpu++)
        for (int i=0; i<2; i++)
            in_length[i][gpu] = 0;
        for (int peer = 0; peer < num_gpus; peer++)
            out_length[peer] = 1;

        SizeT max_queue_length = 0;

        for (int peer = 0; peer < num_gpus; peer++)
            out_length[peer] = 1;
        if (queue_sizing1 < 0) queue_sizing1 = queue_sizing;

        //
        // Allocate frontier queues if necessary
        //

        // Determine frontier queue sizes
        SizeT new_frontier_elements[2] = {0, 0};
        // if (num_gpus > 1) util::cpu_mt::PrintCPUArray<int, SizeT>("in_counter", graph_slice->in_counter.GetPointer(util::HOST), num_gpus + 1, gpu_idx);

        for (int peer = 0; peer < (num_gpus > 1 ? num_gpus + 1 : 1); peer++)
        for (int i = 0; i < 2; i++)
        {
            double queue_sizing_ = i == 0 ? queue_sizing : queue_sizing1;
            switch (frontier_type)
            {
            case VERTEX_FRONTIERS :
                // O(n) ping-pong global vertex frontiers
                new_frontier_elements[0] = ((num_gpus > 1) ?
                    graph_slice->in_counter[peer] : graph_slice->nodes) * queue_sizing_ + 2;
                new_frontier_elements[1] = new_frontier_elements[0];
                break;

            case EDGE_FRONTIERS :
                // O(m) ping-pong global edge frontiers
                new_frontier_elements[0] = graph_slice->edges * queue_sizing_ + 2;
                new_frontier_elements[1] = new_frontier_elements[0];
                break;

            case MIXED_FRONTIERS :
                // O(n) global vertex frontier, O(m) global edge frontier
                new_frontier_elements[0] = ((num_gpus > 1) ?
                    graph_slice->in_counter[peer] : graph_slice->nodes) * queue_sizing_ + 2;
                new_frontier_elements[1] = graph_slice->edges * queue_sizing_ + 2;
                break;
            }

            //printf("new_frontier_elements = %d, %d\n", new_frontier_elements[0], new_frontier_elements[1]);
            // if froniter_queue is not big enough
            if (frontier_queues[peer].keys[i].GetSize() < new_frontier_elements[i])
            {

                // If previously allocated
                if (frontier_queues[peer].keys[i].GetPointer(util::DEVICE) != NULL &&
                    frontier_queues[peer].keys[i].GetSize() != 0)
                {
                    if (retval = frontier_queues[peer].keys[i].EnsureSize(
                        new_frontier_elements[i]))
                        return retval;
                }
                else
                {
                    if (retval = frontier_queues[peer].keys[i].Allocate(
                        new_frontier_elements[i], util::DEVICE))
                        return retval;
                }

                // If use double buffer
                if (_USE_DOUBLE_BUFFER)
                {
                    if (frontier_queues[peer].values[i].GetPointer(util::DEVICE) != NULL &&
                        frontier_queues[peer].values[i].GetSize() != 0)
                    {
                        if (retval = frontier_queues[peer].values[i].EnsureSize(
                            new_frontier_elements[i]))
                            return retval;
                    }
                    else
                    {
                        if (retval = frontier_queues[peer].values[i].Allocate(
                            new_frontier_elements[i], util::DEVICE))
                            return retval;
                    }
                }

            } //end if

            if (new_frontier_elements[0] > max_queue_length)
                max_queue_length = new_frontier_elements[0];
            if (new_frontier_elements[1] > max_queue_length)
                max_queue_length = new_frontier_elements[1];

            if (i == 1 || skip_scanned_edges) continue;

            // Allocate scanned_edges
            SizeT max_elements = new_frontier_elements[0];
            if (new_frontier_elements[1] > max_elements)
                max_elements = new_frontier_elements[1];
            if (scanned_edges[peer].GetSize() < max_elements)
            {
                if (scanned_edges[peer].GetPointer(util::DEVICE) != NULL &&
                    scanned_edges[peer].GetSize() != 0)
                {
                    if (retval = scanned_edges[peer].EnsureSize(max_elements))
                        return retval;
                }
                else
                {
                    if (retval = scanned_edges[peer].Allocate(max_elements, util::DEVICE))
                        return retval;
                }
            }

            /*SizeT cub_request_size = 0;
            cub::DeviceScan::ExclusiveSum(NULL, cub_request_size, froniter_queue.keys[0], froniter_queue.keys[0], max_queue_length);
            if (cub_scan_space[peer].GetSize() < cub_request_size)
            {
                if (cub_scan_space[peer].GetPointer(util::DEVICE) != NULL && cub_scan_space[peer].GetSize() != 0)
                {
                    if (retval = cub_scan_space[peer].EnsureSize(cub_request_size))
                        return retval;
                } else {
                    if (retval = cub_scan_space[peer].Allocate(cub_request_size, util::DEVICE))
                        return retval;
                }
            }*/
            //return retval;
        }

        if (TO_TRACK)
        {
            if (retval = org_checkpoint.Allocate(max_queue_length, util::DEVICE))
                return retval;
            if (retval = org_d_out     .Allocate(max_queue_length, util::DEVICE))
                return retval;
            if (retval = org_offset1   .Allocate(max_queue_length, util::DEVICE))
                return retval;
            if (retval = org_offset2   .Allocate(max_queue_length, util::DEVICE))
                return retval;
            if (retval = org_queue_idx .Allocate(max_queue_length, util::DEVICE))
                return retval;
            if (retval = org_block_idx .Allocate(max_queue_length, util::DEVICE))
                return retval;
            if (retval = org_thread_idx.Allocate(max_queue_length, util::DEVICE))
                return retval;
        }

        return retval;
    } // end Reset(...)

}; // end DataSliceBase

/**
 * @brief Base test parameter structure
 */
struct TestParameter_Base
{
public:
    json_spirit::mObject        info; // JSON information storing running stats
    bool                     g_quiet; // Don't print anything unless specifically directed
    bool          g_quick           ; // Whether or not to skip CPU based computation
    bool          g_stream_from_host; // Whether or not to use stream data from host
    bool          g_undirected      ; // Whether or not to use undirected graph
    bool          instrumented      ; // Whether or not to collect instrumentation from kernels
    bool          debug             ; // Whether or not to use debug mode
    bool          size_check        ; // Whether or not to enable size_check
    bool          mark_predecessors ; // Whether or not to mark src-distance vs. parent vertices
    bool          enable_idempotence; // Whether or not to enable idempotent operation
    void         *graph             ; // Pointer to the input CSR graph
    long long    *src               ; // Source vertex IDs
    int           max_grid_size     ; // maximum grid size (0: leave it up to the enactor)
    int           num_gpus          ; // Number of GPUs for multi-GPU enactor to use
    double        max_queue_sizing  ; // Maximum size scaling factor for work queues (e.g., 1.0 creates n and m-element vertex and edge frontiers).
    double        max_in_sizing     ; // Maximum size scaling factor for data communication
    void         *context           ; // GPU context array used by MordernGPU
    std::string   partition_method  ; // Partition method
    int          *gpu_idx           ; // Array of GPU indices
    cudaStream_t *streams           ; // Array of GPU streams
    float         partition_factor  ; // Partition factor
    int           partition_seed    ; // Partition seed
    int           iterations        ; // Number of repeats
    std::string   traversal_mode    ; // Load-balanced or Dynamic cooperative

    /**
     * @brief TestParameter_Base constructor
     */
    TestParameter_Base()
    {
        // Assign default values
        g_quiet            = false;
        g_quick            = false;
        g_stream_from_host = false;
        g_undirected       = false;
        instrumented       = false;
        debug              = false;
        size_check         = true;
        graph              = NULL;
        src                = NULL;
        max_grid_size      = 0;
        num_gpus           = 1;
        max_queue_sizing   = 1.0;
        max_in_sizing      = 1.0;
        context            = NULL;
        partition_method   = "random";
        gpu_idx            = NULL;
        streams            = NULL;
        partition_factor   = -1;
        partition_seed     = -1;
        iterations         = 1;
        traversal_mode     = "LB";
    }  // end TestParameter_Base()

    /**
     * @brief TestParameter_Base destructor
     */
    ~TestParameter_Base()
    {
        // Clear pointers
        graph   = NULL;
        context = NULL;
        gpu_idx = NULL;
        streams = NULL;
    } // end ~TestParameter_Base()

    /**
     * @brief Utility function to print parameters for debugging
     */
    void PrintParameters()
    {
        using std::cout;
        using std::endl;

        cout << endl << "______________________________"       << endl;
        cout << "==> Test Parameters:  "                       << endl;
        cout << "g_quiet:            \t" << g_quiet            << endl;
        cout << "g_quick:            \t" << g_quick            << endl;
        cout << "g_stream_from_host: \t" << g_stream_from_host << endl;
        cout << "g_undirected:       \t" << g_undirected       << endl;
        cout << "instrumented:       \t" << instrumented       << endl;
        cout << "debug:              \t" << debug              << endl;
        cout << "size_check:         \t" << size_check         << endl;
        cout << "mark_predecessors:  \t" << mark_predecessors  << endl;
        cout << "enable_idempotence: \t" << enable_idempotence << endl;
        cout << "src:                \t" << src                << endl;
        cout << "max_grid_size:      \t" << max_grid_size      << endl;
        cout << "num_gpus:           \t" << num_gpus           << endl;
        cout << "max_queue_sizing:   \t" << max_queue_sizing   << endl;
        cout << "max_in_sizing:      \t" << max_in_sizing      << endl;
        cout << "partition_method:   \t" << partition_method   << endl;
        cout << "partition_factor:   \t" << partition_factor   << endl;
        cout << "partition_seed:     \t" << partition_seed     << endl;
        cout << "iterations:         \t" << iterations         << endl;
        cout << "traversal_mode:     \t" << traversal_mode     << endl;
        cout << "------------------------------" << endl       << endl;
    }

    /**
     * @brief Utility function to print collected info
     */
    void PrintInfo()
    {
        json_spirit::write_stream(json_spirit::mValue(info), std::cout,
                                  json_spirit::pretty_print);
    }

    /**
     * @brief Initialization process for TestParameter_Base
     *
     * @param[in] args Command line arguments
     */
    void Init(util::CommandLineArgs &args)
    {
        bool disable_size_check = true;

        // Get settings from command line arguments
        instrumented       = args.CheckCmdLineFlag("instrumented");
        disable_size_check = args.CheckCmdLineFlag("disable-size-check");
        size_check         = !disable_size_check;
        debug              = args.CheckCmdLineFlag("v");
        g_quick            = args.CheckCmdLineFlag("quick");
        g_undirected       = args.CheckCmdLineFlag("undirected");
        args.GetCmdLineArgument("queue-sizing"    , max_queue_sizing);
        args.GetCmdLineArgument("in-sizing"       , max_in_sizing   );
        args.GetCmdLineArgument("grid-size"       , max_grid_size   );
        args.GetCmdLineArgument("partition-factor", partition_factor);
        args.GetCmdLineArgument("partition-seed"  , partition_seed  );
        args.GetCmdLineArgument("iteration-num"   , iterations      );
        if (args.CheckCmdLineFlag  ("partition-method"))
            args.GetCmdLineArgument("partition-method", partition_method);
    }  // end Init(..)
};

/**
 * @brief Base problem structure.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to use double buffer
 * @tparam _MARK_PREDCESSORS    Whether or not to mark predecessors for vertices
 * @tparam _ENABLE_IDEMPOTENCE  Whether or not to use idempotent
 * @tparam _USE_DOUBLE_BUFFER   Whether or not to use double buffer for frontier queues
 * @tparam _ENABLE_BACKWARD     Whether or not to use backward propagation
 * @tparam _KEEP_ORDER          Whether or not to keep vertices order after partitioning
 * @tparam _KEEP_NODE_NUM       Whether or not to keep vertex IDs after partitioning
 */
template <
    typename    _VertexId,
    typename    _SizeT,
    typename    _Value,
    bool        _MARK_PREDECESSORS,
    bool        _ENABLE_IDEMPOTENCE> //,
    //bool        _USE_DOUBLE_BUFFER,
    //bool        _ENABLE_BACKWARD = false,
    //bool        _KEEP_ORDER      = false,
    //bool        _KEEP_NODE_NUM   = false >
struct ProblemBase
{
    typedef _VertexId           VertexId;
    typedef _SizeT              SizeT;
    typedef _Value              Value;
    static const bool           MARK_PREDECESSORS  = _MARK_PREDECESSORS ;
    static const bool           ENABLE_IDEMPOTENCE = _ENABLE_IDEMPOTENCE;
    //static const bool           USE_DOUBLE_BUFFER  = _USE_DOUBLE_BUFFER ;
    //static const bool           ENABLE_BACKWARD    = _ENABLE_BACKWARD   ;
    bool use_double_buffer;
    bool enable_backward;
    bool keep_order;
    bool keep_node_num;
    bool skip_makeout_selection;
    bool unified_receive;
    bool use_inv_graph;
    bool undirected   ;

    /**
     * Load instruction cache-modifier const defines.
     */
    static const util::io::ld::CacheModifier QUEUE_READ_MODIFIER                    = util::io::ld::cg;             // Load instruction cache-modifier for reading incoming frontier vertex-ids. Valid on SM2.0 or newer
    static const util::io::ld::CacheModifier COLUMN_READ_MODIFIER                   = util::io::ld::NONE;           // Load instruction cache-modifier for reading CSR column-indices.
    static const util::io::ld::CacheModifier EDGE_VALUES_READ_MODIFIER              = util::io::ld::NONE;           // Load instruction cache-modifier for reading edge values.
    static const util::io::ld::CacheModifier ROW_OFFSET_ALIGNED_READ_MODIFIER       = util::io::ld::cg;             // Load instruction cache-modifier for reading CSR row-offsets (8-byte aligned)
    static const util::io::ld::CacheModifier ROW_OFFSET_UNALIGNED_READ_MODIFIER     = util::io::ld::NONE;           // Load instruction cache-modifier for reading CSR row-offsets (4-byte aligned)
    static const util::io::st::CacheModifier QUEUE_WRITE_MODIFIER                   = util::io::st::cg;             // Store instruction cache-modifier for writing outgoing frontier vertex-ids. Valid on SM2.0 or newer

    // Members
    int                 num_gpus              ; // Number of GPUs to be sliced over
    int                 *gpu_idx              ; // GPU indices
    SizeT               nodes                 ; // Number of vertices in the graph
    SizeT               edges                 ; // Number of edges in the graph
    GraphSlice<VertexId, SizeT, Value>
    **graph_slices        ; // Set of graph slices (one for each GPU)
    Csr<VertexId, SizeT, Value> *sub_graphs     ; // Subgraphs for multi-GPU implementation
    Csr<VertexId, SizeT, Value> *inv_subgraphs  ; // inverse subgraphs
    Csr<VertexId, SizeT, Value> *org_graph      ; // Original graph
    PartitionerBase<VertexId, SizeT, Value> //, _ENABLE_BACKWARD, _KEEP_ORDER, _KEEP_NODE_NUM>
    *partitioner          ; // Partitioner
    int                 **partition_tables    ; // Partition tables indicating which GPU the vertices are hosted
    VertexId            **convertion_tables   ; // Conversions tables indicating vertex IDs on local / remote GPUs
    VertexId            **original_vertexes   ; // Vertex IDs in the original graph
    SizeT               **in_counter          ; // Number of in vertices
    SizeT               **out_offsets         ; // Out offsets for data communication
    SizeT               **out_counter         ; // Number of out vertices
    SizeT               **backward_offsets    ; // Offsets for backward propagation
    int                 **backward_partitions ; // Partition tables for backward propagation
    VertexId            **backward_convertions; // Conversion tables for backward propagation

    // Methods

    /**
     * @brief ProblemBase default constructor
     */
    ProblemBase(
        bool _use_double_buffer,
        bool _enable_backward,
        bool _keep_order,
        bool _keep_node_num,
        bool _skip_makeout_selection = false,
        bool _unified_receive = false,
        bool _use_inv_graph   = false,
        bool _undirected      = false) :
        num_gpus            (0   ),
        gpu_idx             (NULL),
        nodes               (0   ),
        edges               (0   ),
        graph_slices        (NULL),
        sub_graphs          (NULL),
        org_graph           (NULL),
        partitioner         (NULL),
        partition_tables    (NULL),
        convertion_tables   (NULL),
        original_vertexes   (NULL),
        in_counter          (NULL),
        out_offsets         (NULL),
        out_counter         (NULL),
        backward_offsets    (NULL),
        backward_partitions (NULL),
        backward_convertions(NULL),
        use_double_buffer   (_use_double_buffer),
        enable_backward     (_enable_backward  ),
        keep_order          (_keep_order       ),
        keep_node_num       (_keep_node_num    ),
        skip_makeout_selection(_skip_makeout_selection),
        unified_receive     (_unified_receive),
        use_inv_graph       (_use_inv_graph),
        undirected          (_undirected)
    {
    } // end ProblemBase()

    /**
     * @brief ProblemBase default destructor to free all graph slices allocated.
     */
    virtual ~ProblemBase()
    {
        Release();
    }

    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;
        // Cleanup graph slices on the heap
        if (graph_slices != NULL)
        {
            for (int i = 0; i < num_gpus; ++i)
            {
                if (retval = graph_slices[i]->Release()) return retval;
                delete graph_slices[i]; graph_slices[i] = NULL;
            }
            delete[] graph_slices; graph_slices = NULL;
        }
        if (partitioner != NULL)
        {
            if (retval = partitioner -> Release()) return retval;
            delete partitioner; partitioner = NULL;
        }
        if (gpu_idx != NULL)
        {
            delete[] gpu_idx;      gpu_idx      = NULL;
        }
        return retval;
    }  // end ~ProblemBase()

    /**
     * @brief Get the GPU index for a specified vertex id.
     *
     * @tparam VertexId Type of signed integer to use as vertex id
     * @param[in] vertex Vertex Id to search
     * \return Index of the GPU that owns the neighbor list of the specified vertex
     */
    //template <typename VertexId>
    int GpuIndex(VertexId vertex)
    {
        if (num_gpus <= 1)
        {

            // Special case for only one GPU, which may be set as with
            // an ordinal other than 0.
            return graph_slices[0]->index;
        }
        else
        {
            return partition_tables[0][vertex];
        }
    }

    /**
     * @brief Get the row offset for a specified vertex id.
     *
     * @tparam VertexId Type of signed integer to use as vertex id
     * @param[in] vertex Vertex Id to search
     * \return Row offset of the specified vertex. If a single GPU is used,
     * this will be the same as the vertex id.
     */
    //template <typename VertexId>
    VertexId GraphSliceRow(VertexId vertex)
    {
        if (num_gpus <= 1)
        {
            return vertex;
        }
        else
        {
            return convertion_tables[0][vertex];
        }
    }

    /**
     * @brief Initialize problem from host CSR graph.
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph            Pointer to the input CSR graph.
     * @param[in] inverse_graph    Pointer to the inversed input CSR graph.
     * @param[in] num_gpus         Number of GPUs
     * @param[in] gpu_idx          Array of GPU indices
     * @param[in] partition_method Partition methods
     * @param[in] queue_sizing     Queue sizing
     * @param[in] partition_factor Partition factor
     * @param[in] partition_seed   Partition seed
     *
     * \return cudaError_t object indicates the success of all CUDA calls.
     */
    cudaError_t Init(
        bool        stream_from_host,
        Csr<VertexId, SizeT, Value> *graph,
        Csr<VertexId, SizeT, Value> *inverse_graph = NULL,
        int         num_gpus          = 1,
        int         *gpu_idx          = NULL,
        std::string partition_method  = "random",
        float       queue_sizing      = 2.0,
        float       partition_factor  = -1,
        int         partition_seed    = -1)
    {
        cudaError_t retval      = cudaSuccess;
        this->org_graph         = graph;
        this->nodes             = graph->nodes;
        this->edges             = graph->edges;
        this->num_gpus          = num_gpus;
        this->gpu_idx           = new int [num_gpus];
        bool have_inv_graph     = false;
        if (num_gpus == 1 && gpu_idx == NULL)
        {
            if (retval = util::GRError(cudaGetDevice(&(this->gpu_idx[0])),
                "ProblemBase cudaGetDevice failed", __FILE__, __LINE__)) return retval;
        }
        else
        {
            for (int gpu = 0; gpu < num_gpus; gpu++)
                this->gpu_idx[gpu] = gpu_idx[gpu];
        }

        graph_slices = new GraphSlice<VertexId, SizeT, Value>*[num_gpus];
        //graph->DisplayGraph("org_graph",graph->nodes);

        if (num_gpus > 1)
        {
            util::CpuTimer cpu_timer;

            //printf("partition_method = %s\n", partition_method.c_str());

            if      (partition_method == "random")
            {
                partitioner = new rp::RandomPartitioner     <VertexId, SizeT, Value
                    /*, _ENABLE_BACKWARD, _KEEP_ORDER, _KEEP_NODE_NUM*/>
                    (*graph, num_gpus, NULL, enable_backward, keep_order, keep_node_num);
            }
            else if (partition_method == "metis")
            {
                partitioner = new metisp::MetisPartitioner  <VertexId, SizeT, Value
                    /*, _ENABLE_BACKWARD, _KEEP_ORDER, _KEEP_NODE_NUM*/>
                    (*graph, num_gpus, NULL, enable_backward, keep_order, keep_node_num);
            }
            else if (partition_method == "static")
            {
                partitioner = new sp::StaticPartitioner     <VertexId, SizeT, Value
                    /*, _ENABLE_BACKWARD, _KEEP_ORDER, _KEEP_NODE_NUM*/>
                    (*graph, num_gpus, NULL, enable_backward, keep_order, keep_node_num);
            }
            else if (partition_method == "cluster")
            {
               partitioner = new cp::ClusterPartitioner     <VertexId, SizeT, Value
                    /*, _ENABLE_BACKWARD, _KEEP_ORDER, _KEEP_NODE_NUM*/>
                    (*graph, num_gpus, NULL, enable_backward, keep_order, keep_node_num);
            }
            else if (partition_method == "biasrandom")
            {
                partitioner = new brp::BiasRandomPartitioner<VertexId, SizeT, Value
                    /*, _ENABLE_BACKWARD, _KEEP_ORDER, _KEEP_NODE_NUM*/>
                    (*graph, num_gpus, NULL, enable_backward, keep_order, keep_node_num);
            }
            else if (partition_method == "duplicate")
            {
                partitioner = new dup::DuplicatePartitioner<VertexId, SizeT, Value>
                    (*graph, num_gpus, NULL, enable_backward, keep_order, keep_node_num);
            }
            else
            {
                util::GRError("partition_method invalid", __FILE__, __LINE__);
            }
            cpu_timer.Start();
            retval = partitioner->Partition(
                sub_graphs,
                partition_tables,
                convertion_tables,
                original_vertexes,
                in_counter,
                out_offsets,
                out_counter,
                backward_offsets,
                backward_partitions,
                backward_convertions,
                partition_factor,
                partition_seed);
            cpu_timer.Stop();

            printf("partition end. (%f ms)\n", cpu_timer.ElapsedMillis());
            //graph -> DisplayGraph("org_graph");

            if (inverse_graph != NULL && keep_node_num && num_gpus > 1 && use_inv_graph && !undirected)
            {
                inv_subgraphs = new Csr<VertexId, SizeT, Value>[num_gpus];
                SizeT *inv_edge_counters = new SizeT[num_gpus];
                for (int gpu = 0; gpu < num_gpus; gpu++)
                    inv_edge_counters[gpu] = 0;

                for (VertexId v = 0; v < graph -> nodes; v++)
                    inv_edge_counters[partition_tables[0][v]] += 
                        inverse_graph -> row_offsets[v+1] - inverse_graph -> row_offsets[v];

                for (int gpu = 0; gpu < num_gpus; gpu++)
                {
                    Csr<VertexId, SizeT, Value> *inv_subgraph = inv_subgraphs + gpu;
                    //Csr<VertexId, SizeT, Value> *sub_graph = sub_graphs + gpu;
                    inv_subgraph -> template FromScratch<false, false>(graph->nodes, inv_edge_counters[gpu]);
                    for (VertexId v = 0; v< graph -> nodes +1; v++)
                        inv_subgraph -> row_offsets[v] = 0;
                }

                for (VertexId v = 0; v < graph -> nodes; v++)
                {
                    inv_subgraphs[partition_tables[0][v]].row_offsets[v+1] =
                        inverse_graph -> row_offsets[v+1] - inverse_graph -> row_offsets[v];
                }

                for (int gpu = 0; gpu < num_gpus; gpu ++)
                {
                    Csr<VertexId, SizeT, Value> *inv_subgraph = inv_subgraphs + gpu;
                    for (VertexId v = 0; v < graph -> nodes; v++)
                    {
                        SizeT offset = inv_subgraph -> row_offsets[v];
                        SizeT in_degree = inv_subgraph -> row_offsets[v+1];
                        if (in_degree > 0)
                        {
                            memcpy(inv_subgraph -> column_indices + offset,
                                inverse_graph -> column_indices + inverse_graph -> row_offsets[v],
                                sizeof(VertexId) * in_degree);
                        }
                        inv_subgraph -> row_offsets[v+1] += offset;
                    }

                    //printf("GPU %d\n", gpu);
                    //sub_graphs[gpu].DisplayGraph("sub_graph");
                    //inv_subgraphs[gpu].DisplayGraph("inv_graph");
                }
                have_inv_graph = true;
            }

            //graph->DisplayGraph("org_graph",graph->nodes);
            //util::cpu_mt::PrintCPUArray<SizeT,int>("partition0",partition_tables[0],graph->nodes);
            //util::cpu_mt::PrintCPUArray<SizeT,VertexId>("convertion0",convertion_tables[0],graph->nodes);
            //util::cpu_mt::PrintCPUArray<SizeT,Value>("edge_value",graph->edge_values,graph->edges);
            //for (int gpu=0;gpu<num_gpus;gpu++)
            //{
            //    sub_graphs[gpu].DisplayGraph("sub_graph",sub_graphs[gpu].nodes);
            //    printf("%d\n",gpu);
            //    util::cpu_mt::PrintCPUArray<SizeT,int     >("partition"           , partition_tables    [gpu+1], sub_graphs[gpu].nodes);
            //    util::cpu_mt::PrintCPUArray<SizeT,VertexId>("convertion"          , convertion_tables   [gpu+1], sub_graphs[gpu].nodes);
                //util::cpu_mt::PrintCPUArray<SizeT,SizeT   >("backward_offsets"    , backward_offsets    [gpu], sub_graphs[gpu].nodes);
                //util::cpu_mt::PrintCPUArray<SizeT,int     >("backward_partitions" , backward_partitions [gpu], backward_offsets[gpu][sub_graphs[gpu].nodes]);
                //util::cpu_mt::PrintCPUArray<SizeT,VertexId>("backward_convertions", backward_convertions[gpu], backward_offsets[gpu][sub_graphs[gpu].nodes]);
            //}
            //for (int gpu=0;gpu<num_gpus;gpu++)
            //{
            //    cross_counter[gpu][num_gpus]=0;
            //    for (int peer=0;peer<num_gpus;peer++)
            //    {
            //        cross_counter[gpu][peer]=out_offsets[gpu][peer+1]-out_offsets[gpu][peer];
            //    }
            //    cross_counter[gpu][num_gpus]=in_offsets[gpu][num_gpus];
            //}
            /*for (int gpu=0;gpu<num_gpus;gpu++)
            for (int peer=0;peer<=num_gpus;peer++)
            {
                in_offsets[gpu][peer]*=2;
                out_offsets[gpu][peer]*=2;
            }*/
            if (retval) return retval;
        } else {
            sub_graphs = graph;
        }

        for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            graph_slices[gpu] = new GraphSlice<VertexId, SizeT, Value>(this->gpu_idx[gpu]);
            if (num_gpus > 1)
            {
                if (enable_backward)
                {
                    retval = graph_slices[gpu]->Init(
                        stream_from_host,
                        num_gpus,
                        &(sub_graphs     [gpu]),
                        (have_inv_graph) ? inv_subgraphs + gpu : NULL,
                        partition_tables    [gpu + 1],
                        convertion_tables   [gpu + 1],
                        original_vertexes   [gpu],
                        in_counter          [gpu],
                        out_offsets         [gpu],
                        out_counter         [gpu],
                        backward_offsets    [gpu],
                        backward_partitions [gpu],
                        backward_convertions[gpu]);
                }
                else
                {
                    retval = graph_slices[gpu]->Init(
                        stream_from_host,
                        num_gpus,
                        &(sub_graphs[gpu]),
                        (have_inv_graph) ? inv_subgraphs + gpu : NULL,
                        partition_tables [gpu + 1],
                        convertion_tables[gpu + 1],
                        original_vertexes[gpu],
                        in_counter       [gpu],
                        out_offsets      [gpu],
                        out_counter      [gpu],
                        NULL,
                        NULL,
                        NULL);
                }
            }
            else
            {
                retval = graph_slices[gpu]->Init(
                    stream_from_host,
                    num_gpus,
                    &(sub_graphs[gpu]),
                    inverse_graph,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL);
            }
            if (retval) return retval;
        }  // end for (gpu)

        return retval;
    } // end Init(...)

};

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
