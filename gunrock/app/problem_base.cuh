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
 * @brief Base struct for all the application types
 */

#pragma once

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
#include <gunrock/app/rp/rp_partitioner.cuh>
#include <gunrock/app/cp/cp_partitioner.cuh>
#include <gunrock/app/brp/brp_partitioner.cuh>
#include <gunrock/app/metisp/metis_partitioner.cuh>
#include <gunrock/app/sp/sp_partitioner.cuh>
#include <vector>
#include <string>

#include <moderngpu.cuh>

namespace gunrock {
namespace app {

/**
 * @brief Enumeration of global frontier queue configurations
 */
enum FrontierType {
    VERTEX_FRONTIERS,       // O(n) ping-pong global vertex frontiers
    EDGE_FRONTIERS,         // O(m) ping-pong global edge frontiers
    MIXED_FRONTIERS         // O(n) global vertex frontier, O(m) global edge frontier
};

/**
 * @brief Graph slice structure which contains common graph structural data.
 *
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam Value               Type to use as vertex / edge associated values
 */
template <
    typename SizeT,
    typename VertexId,
    typename Value>
struct GraphSlice
{
    int             num_gpus; // Number of GPUs
    int             index   ; // Slice index
    VertexId        nodes   ; // Number of nodes in slice
    SizeT           edges   ; // Number of edges in slice

    Csr<VertexId, Value, SizeT   > *graph             ; // Pointer to CSR format subgraph
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
    util::Array1D<SizeT, SizeT   > backward_offset    ; // Backward offsets for partition and convertion tables
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
    } // end GraphSlice(int index)

    /**
     * @brief GraphSlice Destructor to free all device memories.
     */
    virtual ~GraphSlice()
    {
        // Set device (use slice index)
        util::SetDevice(index);

        // Release allocated host / device memory
        row_offsets        .Release();
        column_indices     .Release();
        out_degrees        .Release();
        column_offsets     .Release();
        row_indices        .Release();
        in_degrees         .Release();
        partition_table    .Release();
        convertion_table   .Release();
        original_vertex    .Release();
        in_counter         .Release();
        out_offset         .Release();
        out_counter        .Release();
        backward_offset    .Release();
        backward_partition .Release();
        backward_convertion.Release();
    } // end ~GraphSlice()

   /**
     * @brief Initalize graph slice
     *
     * @param[in] stream_from_host    Whether to stream data from host
     * @param[in] num_gpus            Number of gpus
     * @param[in] graph               Pointer to the sub graph
     * @param[in] inverstgraph        Pointer to the inverst graph
     * @param[in] partition_table     The partition table
     * @param[in] convertion_table    The convertion table
     * @param[in] original_vertex     The original vertex table
     * @param[in] in_counter          In_counters
     * @param[in] out_offset          Out_offsets
     * @param[in] out_counter         Out_counters
     * @param[in] backward_offsets    Backward_offsets
     * @param[in] backward_partition  The backward partition table
     * @param[in] backward_convertion The backward convertion table 
     * \return cudaError_t            Object incidating the success of all CUDA function calls
     */
    cudaError_t Init(
        bool                       stream_from_host,
        int                        num_gpus,
        Csr<VertexId,Value,SizeT>* graph,
        Csr<VertexId,Value,SizeT>* inverstgraph,
        int*                       partition_table,
        VertexId*                  convertion_table,
        VertexId*                  original_vertex,
        SizeT*                     in_counter,
        SizeT*                     out_offset,
        SizeT*                     out_counter,
        SizeT*                     backward_offsets   = NULL,
        int*                       backward_partition = NULL,
        VertexId*                  backward_convertion= NULL)
    {
        cudaError_t retval     = cudaSuccess;

        // Set local variables / array pointers
        this->num_gpus         = num_gpus;
        this->graph            = graph;
        this->nodes            = graph->nodes;
        this->edges            = graph->edges;
        if (partition_table  != NULL) this->partition_table    .SetPointer(partition_table      , nodes     );
        if (convertion_table != NULL) this->convertion_table   .SetPointer(convertion_table     , nodes     );
        if (original_vertex  != NULL) this->original_vertex    .SetPointer(original_vertex      , nodes     );
        if (in_counter       != NULL) this->in_counter         .SetPointer(in_counter           , num_gpus+1);
        if (out_offset       != NULL) this->out_offset         .SetPointer(out_offset           , num_gpus+1);
        if (out_counter      != NULL) this->out_counter        .SetPointer(out_counter          , num_gpus+1);
        this->row_offsets        .SetPointer(graph->row_offsets   , nodes+1   );
        this->column_indices     .SetPointer(graph->column_indices, edges     );
        if (inverstgraph != NULL)
        {
            this->column_offsets .SetPointer(inverstgraph->row_offsets, nodes+1);
            this->row_indices    .SetPointer(inverstgraph->column_indices   , edges  );
        }

        do {
            // Set device using slice index
            if (retval = util::GRError(cudaSetDevice(index), 
                             "GpuSlice cudaSetDevice failed", __FILE__, __LINE__)) break;

            // Allocate and initialize row_offsets
            if (retval = this->row_offsets.Allocate(nodes+1      ,util::DEVICE)) break;
            if (retval = this->row_offsets.Move    (util::HOST   ,util::DEVICE)) break;
            
            // Allocate and initialize column_indices
            if (retval = this->column_indices.Allocate(edges     ,util::DEVICE)) break;
            if (retval = this->column_indices.Move    (util::HOST,util::DEVICE)) break;

            // Allocate out degrees for each node
            if (retval = this->out_degrees.Allocate(nodes        ,util::DEVICE)) break;
            // count number of out-going degrees for each node
            util::MemsetMadVectorKernel<<<128, 128>>>(
                this->out_degrees.GetPointer(util::DEVICE),
                this->row_offsets.GetPointer(util::DEVICE),
                this->row_offsets.GetPointer(util::DEVICE) + 1,
                -1, nodes);
           

            if (inverstgraph != NULL)
            {
                // Allocate and initialize column_offsets
                if (retval = this->column_offsets.Allocate(nodes+1      ,util::DEVICE)) break;
                if (retval = this->column_offsets.Move    (util::HOST   ,util::DEVICE)) break;
                
                // Allocate and initialize row_indices
                if (retval = this->row_indices.Allocate(edges     ,util::DEVICE)) break;
                if (retval = this->row_indices.Move    (util::HOST,util::DEVICE)) break;

                // Allocate in degrees for each node
                if (retval = this->in_degrees .Allocate(nodes,  util::DEVICE)) break;
                // count number of in-going degrees for each node
                util::MemsetMadVectorKernel<<<128, 128>>>(
                    this->in_degrees    .GetPointer(util::DEVICE),
                    this->column_offsets.GetPointer(util::DEVICE),
                    this->column_offsets.GetPointer(util::DEVICE) + 1,
                    -1, nodes);
            }

            // For multi-GPU cases
            if (num_gpus > 1)
            {
                // Allocate and initalize convertion_table
                if (retval = this->partition_table.Allocate (nodes     ,util::DEVICE)) break;
                if (partition_table  != NULL)
                    if (retval = this->partition_table.Move (util::HOST,util::DEVICE)) break;
                
                // Allocate and initalize convertion_table
                if (retval = this->convertion_table.Allocate(nodes     ,util::DEVICE)) break;
                if (convertion_table != NULL)
                    if (retval = this->convertion_table.Move(util::HOST,util::DEVICE)) break;

                // Allocate and initalize original_vertex
                if (retval = this->original_vertex .Allocate(nodes     ,util::DEVICE)) break;
                if (original_vertex  != NULL)
                    if (retval = this->original_vertex .Move(util::HOST,util::DEVICE)) break;
                
                // If need backward information progation
                if (backward_offsets!=NULL)
                {
                    // Allocate and initalize backward_offset
                    this->backward_offset    .SetPointer(backward_offsets     , nodes+1);
                    if (retval = this->backward_offset    .Allocate(nodes+1, util::DEVICE)) break;
                    if (retval = this->backward_offset    .Move(util::HOST, util::DEVICE)) break;

                    // Allocate and initalize backward_partition
                    this->backward_partition .SetPointer(backward_partition   , backward_offsets[nodes]);
                    if (retval = this->backward_partition .Allocate(backward_offsets[nodes], util::DEVICE)) break;
                    if (retval = this->backward_partition .Move(util::HOST, util::DEVICE)) break;

                    // Allocate and initalize backward_convertion
                    this->backward_convertion.SetPointer(backward_convertion  , backward_offsets[nodes]);
                    if (retval = this->backward_convertion.Allocate(backward_offsets[nodes], util::DEVICE)) break;
                    if (retval = this->backward_convertion.Move(util::HOST, util::DEVICE)) break;
                }
            } // end if num_gpu>1
        } while (0);

        return retval;
    } // end of Init(...)

    /**
     * @brief overloaded = operator
     *
     * @param[in] GraphSlice to copy from
     * \return a copy of local GraphSlice
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
 * @brief Baase data slice structure which contains common data structural needed for permitives.
 *
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam Value               Type to use as vertex / edge associated values
 */
template <
    typename SizeT,
    typename VertexId,
    typename Value>
struct DataSliceBase
{
    int    num_gpus            ; // Number of GPUs
    int    gpu_idx             ; // GPU index
    int    wait_counter        ; // Wait counter for interation loop control
    int    gpu_mallocing       ; // Whether gpu is in malloc
    int    num_vertex_associate; // Number of associate values in VertexId type for each vertex
    int    num_value__associate; // Number of associate values in Value type for each vertex
    int    num_stages          ; // Number of stages
    SizeT  nodes               ; // Numver of vertices
    util::Array1D<SizeT, VertexId    > **vertex_associate_in  [2]; // Incoming VertexId type associate values
    util::Array1D<SizeT, VertexId*   >  *vertex_associate_ins [2]; // Device pointers to incoming VertexId type associate values
    util::Array1D<SizeT, VertexId    > **vertex_associate_out    ; // Outgoing VertexId type associate values
    util::Array1D<SizeT, VertexId*   >  *vertex_associate_outs   ; // Device pointers to outgoing VertexId type associate values
    util::Array1D<SizeT, VertexId**  >   vertex_associate_outss  ; // Device pointers to device points to outgoing VertexId type associate values
    util::Array1D<SizeT, VertexId*   >   vertex_associate_orgs   ; // Device pointers to original VertexId type associate values
    util::Array1D<SizeT, Value       > **value__associate_in  [2]; // Incoming Value type associate values
    util::Array1D<SizeT, Value*      >  *value__associate_ins [2]; // Device pointers to incomnig Value type associate values
    util::Array1D<SizeT, Value       > **value__associate_out    ; // Outgoing Value type associate values
    util::Array1D<SizeT, Value*      >  *value__associate_outs   ; // Device pointers to outgoing Value type assocaite values
    util::Array1D<SizeT, Value**     >   value__associate_outss  ; // Device pointers to device pointers to outgoing Value type associate values
    util::Array1D<SizeT, Value*      >   value__associate_orgs   ; // Device pointers to original Value type associate values
    util::Array1D<SizeT, SizeT       >   out_length              ; // Number of outgoing vertices to peers  
    util::Array1D<SizeT, SizeT       >   in_length            [2]; // Number of incoming vertices from peers
    util::Array1D<SizeT, VertexId    >  *keys_in              [2]; // Incoming vertices
    util::Array1D<SizeT, VertexId*   >   keys_outs               ; // Outgoing vertices
    util::Array1D<SizeT, VertexId    >  *keys_out                ; // Device pointers to outgoing vertices
    util::Array1D<SizeT, SizeT       >  *keys_marker             ; // Markers to separate vertices to peer GPUs
    util::Array1D<SizeT, SizeT*      >   keys_markers            ; // Device pointer to the markers
    util::Array1D<SizeT, cudaEvent_t*>   events               [4]; // GPU stream events arrays
    util::Array1D<SizeT, bool*       >   events_set           [4]; // Whether the GPU stream events are set
    util::Array1D<SizeT, int         >   wait_marker             ; //
    util::Array1D<SizeT, cudaStream_t>   streams                 ; // GPU streams
    util::Array1D<SizeT, int         >   stages                  ; // current stages of each streams
    util::Array1D<SizeT, bool        >   to_show                 ; // whether to show debug information for the streams
    util::Array1D<SizeT, char        >   make_out_array          ; // compressed data structure for make_out kernel
    util::Array1D<SizeT, char        >  *expand_incoming_array   ; // compressed data structure for expand_incoming kernel
    util::Array1D<SizeT, VertexId    >   preds                   ; // predecessors of vertices
    util::Array1D<SizeT, VertexId    >   temp_preds              ; // tempory storages for predecessors
    
    //Frontier queues. Used to track working frontier.
    util::DoubleBuffer<SizeT, VertexId, Value>  *frontier_queues ; // frontier queues
    util::Array1D<SizeT, SizeT       >  *scanned_edges           ; // length / offsets for offsets of the frontier queues

    /**
     * @brief DataSliceBase default constructor
     */
    DataSliceBase()
    {
        // Assign default values
        num_stages               = 4;
        num_vertex_associate     = 0;
        num_value__associate     = 0;
        gpu_idx                  = 0;
        gpu_mallocing            = 0;

        // Assign NULs to pointers
        keys_out                 = NULL;
        keys_marker              = NULL;
        keys_in              [0] = NULL;
        keys_in              [1] = NULL;
        vertex_associate_in  [0] = NULL;
        vertex_associate_in  [1] = NULL;
        vertex_associate_ins [0] = NULL;
        vertex_associate_ins [1] = NULL;
        vertex_associate_out     = NULL;
        vertex_associate_outs    = NULL;
        value__associate_in  [0] = NULL;
        value__associate_in  [1] = NULL;
        value__associate_ins [0] = NULL;
        value__associate_ins [1] = NULL;
        value__associate_out     = NULL;
        value__associate_outs    = NULL;
        frontier_queues          = NULL;
        scanned_edges            = NULL;
        expand_incoming_array    = NULL;

        // Assign names to arrays
        keys_outs              .SetName("keys_outs"              );
        vertex_associate_outss .SetName("vertex_associate_outss" );  
        value__associate_outss .SetName("value__associate_outss" );
        vertex_associate_orgs  .SetName("vertex_associate_orgs"  );
        value__associate_orgs  .SetName("value__associate_orgs"  ); 
        out_length             .SetName("out_length"             );  
        in_length           [0].SetName("in_length[0]"           );  
        in_length           [1].SetName("in_length[1]"           );  
        wait_marker            .SetName("wait_marker"            );
        keys_markers           .SetName("keys_marker"            );
        stages                 .SetName("stages"                 );
        to_show                .SetName("to_show"                );
        make_out_array         .SetName("make_out_array"         );
        streams                .SetName("streams"                );
        preds                  .SetName("preds"                  );
        temp_preds             .SetName("temp_preds"             );
        for (int i=0;i<4;i++)
        {
            events[i].SetName("events[]");
            events_set[i].SetName("events_set[]");
        }
    } // end DataSliceBase()

    /**
     * @brief DataSliceBase default destructor to release host / device memory
     */
    ~DataSliceBase()
    {
        // Set device by index
        if (util::SetDevice(gpu_idx)) return;

        // Release VertexId type incoming associate values and related pointers
        if (vertex_associate_in[0] != NULL)
        {
            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                for (int i=0;i<num_vertex_associate;i++)
                {
                    vertex_associate_in[0][gpu][i].Release();
                    vertex_associate_in[1][gpu][i].Release();
                }
                delete[] vertex_associate_in[0][gpu];
                delete[] vertex_associate_in[1][gpu];
                vertex_associate_in [0][gpu] = NULL;
                vertex_associate_in [1][gpu] = NULL;
                vertex_associate_ins[0][gpu].Release();
                vertex_associate_ins[1][gpu].Release();
            }
            delete[] vertex_associate_in [0];
            delete[] vertex_associate_in [1];
            delete[] vertex_associate_ins[0];
            delete[] vertex_associate_ins[1];
            vertex_associate_in [0] = NULL;
            vertex_associate_in [1] = NULL;
            vertex_associate_ins[0] = NULL;
            vertex_associate_ins[1] = NULL;
        }

        // Release Value type incoming associate values and related pointers
        if (value__associate_in[0] != NULL)
        {
            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                for (int i=0;i<num_value__associate;i++)
                {
                    value__associate_in[0][gpu][i].Release();
                    value__associate_in[1][gpu][i].Release();
                }
                delete[] value__associate_in[0][gpu];
                delete[] value__associate_in[1][gpu];
                value__associate_in [0][gpu] = NULL;
                value__associate_in [1][gpu] = NULL;
                value__associate_ins[0][gpu].Release();
                value__associate_ins[1][gpu].Release();
            }
            delete[] value__associate_in [0];
            delete[] value__associate_in [1];
            delete[] value__associate_ins[0];
            delete[] value__associate_ins[1];
            value__associate_in [0] = NULL;
            value__associate_in [1] = NULL;
            value__associate_ins[0] = NULL;
            value__associate_ins[1] = NULL;
        }
        
        // Release incoming keys and related pointers
        if (keys_in[0] != NULL)
        {
            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                keys_in[0][gpu].Release();
                keys_in[1][gpu].Release();
            }
            delete[] keys_in[0];
            delete[] keys_in[1];
            keys_in[0] = NULL;
            keys_in[1] = NULL;
        }

        // Release outgoing keys and markers
        if (keys_marker !=NULL)
        {
            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                keys_out   [gpu].Release();
                keys_marker[gpu].Release();
            }
            delete[] keys_out   ; keys_out    = NULL;
            delete[] keys_marker; keys_marker = NULL;
            keys_markers.Release();
        }

        // Release VertexId type outgoing associate values and pointers
        if (vertex_associate_out != NULL)
        {
            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                for (int i=0;i<num_vertex_associate;i++)
                    vertex_associate_out[gpu][i].Release();
                delete[] vertex_associate_out[gpu];
                vertex_associate_out [gpu]=NULL;
                vertex_associate_outs[gpu].Release();
            }
            delete[] vertex_associate_out;
            delete[] vertex_associate_outs;
            vertex_associate_out =NULL;
            vertex_associate_outs=NULL;
            vertex_associate_outss.Release();
        }

        // Release Value type outgoing associate values and pointers
        if (value__associate_out != NULL)
        {
            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                for (int i=0;i<num_value__associate;i++)
                    value__associate_out[gpu][i].Release();
                delete[] value__associate_out[gpu];
                value__associate_out [gpu]=NULL;
                value__associate_outs[gpu].Release();
            }
            delete[] value__associate_out ;
            delete[] value__associate_outs;
            value__associate_out =NULL;
            value__associate_outs=NULL;
            value__associate_outss.Release();
        }

        // Release events and markers
        for (int i=0;i<4;i++)
        {
            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                for (int stage=0;stage<num_stages;stage++)
                    cudaEventDestroy(events[i][gpu][stage]);
                delete[] events    [i][gpu]; events    [i][gpu]=NULL;
                delete[] events_set[i][gpu]; events_set[i][gpu]=NULL;
            }
            events    [i].Release();
            events_set[i].Release();
        }

        // Release expand_incoming_arrays
        if (expand_incoming_array!=NULL)
        {
            for (int gpu=0;gpu<num_gpus;gpu++)
                expand_incoming_array[gpu].Release();
            delete[] expand_incoming_array;
            expand_incoming_array = NULL;
        }

        // Release frontiers
        if (frontier_queues!=NULL)
        {
            for (int gpu = 0; gpu<=num_gpus; gpu++)
            {
                for (int i = 0; i < 2; ++i) {
                    frontier_queues[gpu].keys  [i].Release();
                    frontier_queues[gpu].values[i].Release();
                }
            }
            delete[] frontier_queues; frontier_queues=NULL;
        }

        // Release scanned_edges
        if (scanned_edges != NULL)
        {
            for (int gpu=0;gpu<=num_gpus;gpu++)
                scanned_edges          [gpu].Release();
            delete[] scanned_edges;
            scanned_edges           = NULL;
        }

        //Release all other arrays
        keys_outs     .Release();
        in_length  [0].Release();
        in_length  [1].Release();
        wait_marker   .Release();
        out_length    .Release();
        vertex_associate_orgs.Release();
        value__associate_orgs.Release();
        streams       .Release();
        stages        .Release();
        to_show       .Release();
        make_out_array.Release();
        preds         .Release();
        temp_preds    .Release();
    } // end ~DataSliceBase()

    /**
     * @brief Initiate DataSliceBase
     *
     * @param[in] num_gpus             Number of GPUs
     * @param[in] gpu_idx              GPU index
     * @param[in] num_vertex_associate Number of VertexId type associate values
     * @param[in] num_value__associate Numver of Value type associate values
     * @param[in] graph                Pointer to the CSR formated sub-graph
     * @param[in] num_in_nodes         Number of incoming vertices from peers
     * @param[in] num_out_nodes        Number of outgoing vertices to peers
     * @param[in] in_sizing            Preallocation factor for incoming / outgoing vertices
     * \return                         Error occured if any, otherwise cudaSuccess
     */
    cudaError_t Init(
        int    num_gpus            ,
        int    gpu_idx             ,
        int    num_vertex_associate,
        int    num_value__associate,
        Csr<VertexId, Value, SizeT> 
              *graph               ,
        SizeT *num_in_nodes        ,
        SizeT *num_out_nodes       ,
        float  in_sizing = 1.0     )
    {
        cudaError_t retval         = cudaSuccess;
        // Copy input values
        this->num_gpus             = num_gpus;
        this->gpu_idx              = gpu_idx;
        this->nodes                = graph->nodes;
        this->num_vertex_associate = num_vertex_associate;
        this->num_value__associate = num_value__associate;
        
        // Set device by index
        if (retval = util::SetDevice(gpu_idx))  return retval;

        // Allocate frontiers and scanned_edges
        this->frontier_queues      = new util::DoubleBuffer<SizeT, VertexId, Value>[num_gpus+1]; 
        this->scanned_edges        = new util::Array1D<SizeT, SizeT>[num_gpus+1];
        for (int i=0; i<num_gpus+1; i++)
        {
            this->scanned_edges[i].SetName("scanned_edges[]");
        }
        if (retval = in_length[0].Allocate(num_gpus,util::HOST)) return retval;
        if (retval = in_length[1].Allocate(num_gpus,util::HOST)) return retval;
        if (retval = out_length  .Allocate(num_gpus,util::HOST | util::DEVICE)) return retval;
        if (retval = vertex_associate_orgs.Allocate(num_vertex_associate, util::HOST | util::DEVICE)) return retval;
        if (retval = value__associate_orgs.Allocate(num_value__associate, util::HOST | util::DEVICE)) return retval;

        // Allocate / create event related variables
        wait_marker .Allocate(num_gpus*2);
        stages      .Allocate(num_gpus*2);
        to_show     .Allocate(num_gpus*2);
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            wait_marker[gpu]=0;
        }
        for (int i=0;i<4;i++) 
        {
            events    [i].Allocate(num_gpus);
            events_set[i].Allocate(num_gpus);
            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                events    [i][gpu]=new cudaEvent_t[num_stages];
                events_set[i][gpu]=new bool       [num_stages];
                for (int stage=0;stage<num_stages;stage++)
                {
                    if (retval = util::GRError(cudaEventCreate(&(events[i][gpu][stage])), 
                                              "cudaEventCreate failed.", __FILE__, __LINE__)) return retval;
                    events_set[i][gpu][stage]=false;
                } 
            }
        }
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            for (int i=0;i<2;i++)
                in_length[i][gpu]=0;
        }

        if (num_gpus==1) return retval;
        // Create incoming buffer on device
        keys_in             [0] = new util::Array1D<SizeT,VertexId > [num_gpus];
        keys_in             [1] = new util::Array1D<SizeT,VertexId > [num_gpus];
        vertex_associate_in [0] = new util::Array1D<SizeT,VertexId >*[num_gpus];
        vertex_associate_in [1] = new util::Array1D<SizeT,VertexId >*[num_gpus];
        vertex_associate_ins[0] = new util::Array1D<SizeT,VertexId*> [num_gpus];
        vertex_associate_ins[1] = new util::Array1D<SizeT,VertexId*> [num_gpus];
        value__associate_in [0] = new util::Array1D<SizeT,Value    >*[num_gpus];
        value__associate_in [1] = new util::Array1D<SizeT,Value    >*[num_gpus];
        value__associate_ins[0] = new util::Array1D<SizeT,Value   *> [num_gpus];
        value__associate_ins[1] = new util::Array1D<SizeT,Value   *> [num_gpus];
        for (int gpu=0;gpu<num_gpus;gpu++)
        for (int t=0;t<2;t++)
        {
            SizeT num_in_node = num_in_nodes[gpu] * in_sizing;
            vertex_associate_in [t][gpu] = new util::Array1D<SizeT,VertexId>[num_vertex_associate];
            for (int i=0;i<num_vertex_associate;i++)
            {
                vertex_associate_in [t][gpu][i].SetName("vertex_associate_in[]");
                if (gpu!=0) if (retval = vertex_associate_in[t][gpu][i].Allocate(num_in_node,util::DEVICE)) return retval;
            }
            value__associate_in [t][gpu] = new util::Array1D<SizeT,Value   >[num_value__associate];
            for (int i=0;i<num_value__associate;i++)
            {
                value__associate_in[t][gpu][i].SetName("value__associate_ins[]");
                if (gpu!=0) if (retval = value__associate_in[t][gpu][i].Allocate(num_in_node,util::DEVICE)) return retval;
            }
                
            vertex_associate_ins[t][gpu].SetName("vertex_associate_ins");
            if (retval = vertex_associate_ins[t][gpu].Allocate(num_vertex_associate, util::DEVICE | util::HOST)) return retval;
            for (int i=0;i<num_vertex_associate;i++)
                vertex_associate_ins[t][gpu][i] = vertex_associate_in[t][gpu][i].GetPointer(util::DEVICE);
            if (retval = vertex_associate_ins[t][gpu].Move(util::HOST, util::DEVICE)) return retval;

            value__associate_ins[t][gpu].SetName("value__associate_ins");
            if (retval = value__associate_ins[t][gpu].Allocate(num_value__associate, util::DEVICE | util::HOST)) return retval;
            for (int i=0;i<num_value__associate;i++)
                value__associate_ins[t][gpu][i] = value__associate_in[t][gpu][i].GetPointer(util::DEVICE);
            if (retval = value__associate_ins[t][gpu].Move(util::HOST, util::DEVICE)) return retval;

            keys_in[t][gpu].SetName("keys_in");
            if (gpu!=0) if (retval = keys_in[t][gpu].Allocate(num_in_node,util::DEVICE)) return retval;
        }

        // Allocate outgoing buffer on device
        vertex_associate_out  = new util::Array1D<SizeT,VertexId >*[num_gpus];
        vertex_associate_outs = new util::Array1D<SizeT,VertexId*> [num_gpus];
        value__associate_out  = new util::Array1D<SizeT,Value    >*[num_gpus];
        value__associate_outs = new util::Array1D<SizeT,Value*   > [num_gpus];
        keys_marker           = new util::Array1D<SizeT,SizeT    > [num_gpus];
        keys_out              = new util::Array1D<SizeT,VertexId > [num_gpus];
        if (retval = vertex_associate_outss.Allocate(num_gpus, util::HOST | util::DEVICE)) return retval;
        if (retval = value__associate_outss.Allocate(num_gpus, util::HOST | util::DEVICE)) return retval;
        if (retval = keys_markers          .Allocate(num_gpus, util::HOST | util::DEVICE)) return retval;
        if (retval = keys_outs             .Allocate(num_gpus, util::HOST | util::DEVICE)) return retval;
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            SizeT num_out_node = num_out_nodes[gpu] * in_sizing;
            keys_marker[gpu].SetName("keys_marker[]");
            if (retval = keys_marker[gpu].Allocate(num_out_nodes[num_gpus] * in_sizing, util::DEVICE)) return retval;
            keys_markers[gpu]=keys_marker[gpu].GetPointer(util::DEVICE);
            keys_out   [gpu].SetName("keys_out[]");
            if (gpu!=0)
            {
                if (retval = keys_out[gpu].Allocate(num_out_node, util::DEVICE)) return retval;
                keys_outs[gpu]=keys_out[gpu].GetPointer(util::DEVICE);
            }

            vertex_associate_out  [gpu] = new util::Array1D<SizeT,VertexId>[num_vertex_associate];
            vertex_associate_outs [gpu].SetName("vertex_associate_outs[]");
            if (retval = vertex_associate_outs[gpu].Allocate(num_vertex_associate, util::HOST | util::DEVICE)) return retval;
            vertex_associate_outss[gpu] = vertex_associate_outs[gpu].GetPointer(util::DEVICE);
            for (int i=0;i<num_vertex_associate;i++)
            {
                vertex_associate_out[gpu][i].SetName("vertex_associate_out[][]");
                if (gpu!=0)
                    if (retval = vertex_associate_out[gpu][i].Allocate(num_out_node, util::DEVICE)) return retval;
                vertex_associate_outs[gpu][i]=vertex_associate_out[gpu][i].GetPointer(util::DEVICE);
            }
            if (retval = vertex_associate_outs[gpu].Move(util::HOST, util::DEVICE)) return retval;

            value__associate_out [gpu] = new util::Array1D<SizeT,Value>[num_value__associate];
            value__associate_outs[gpu].SetName("value__associate_outs[]");
            if (retval = value__associate_outs[gpu].Allocate(num_value__associate, util::HOST | util::DEVICE)) return retval;
            value__associate_outss[gpu] = value__associate_outs[gpu].GetPointer(util::DEVICE);
            for (int i=0;i<num_value__associate;i++)
            {
                value__associate_out[gpu][i].SetName("value__associate_out[][]");
                if (gpu!=0)
                    if (retval = value__associate_out[gpu][i].Allocate(num_out_node, util::DEVICE)) return retval;
                value__associate_outs[gpu][i]=value__associate_out[gpu][i].GetPointer(util::DEVICE);
            }
            if (retval = value__associate_outs[gpu].Move(util::HOST, util::DEVICE)) return retval;
        }
        if (retval = keys_markers          .Move(util::HOST, util::DEVICE)) return retval;
        if (retval = vertex_associate_outss.Move(util::HOST, util::DEVICE)) return retval;
        if (retval = value__associate_outss.Move(util::HOST, util::DEVICE)) return retval;
       
        // Allocate make_out_array and expand_incoming array
        if (retval = make_out_array.Allocate(
            sizeof(SizeT*   ) * num_gpus + 
            sizeof(VertexId*) * num_gpus + 
            sizeof(VertexId*) * num_vertex_associate + 
            sizeof(Value*   ) * num_value__associate + 
            sizeof(VertexId*) * num_vertex_associate * num_gpus + 
            sizeof(Value*   ) * num_value__associate * num_gpus +
            sizeof(SizeT    ) * num_gpus, 
            util::HOST | util::DEVICE)) return retval;
        expand_incoming_array = new util::Array1D<SizeT, char>[num_gpus];
        for (int i=0;i<num_gpus;i++)
        {
            expand_incoming_array[i].SetName("expand_incoming_array[]");
            if (retval = expand_incoming_array[i].Allocate(
                sizeof(Value*   ) * num_value__associate * 2 + 
                sizeof(VertexId*) * num_vertex_associate * 2, 
                util::HOST | util::DEVICE)) return retval;
        }

        return retval;
    } // end Init(..)

    /** 
     * @brief Performs reset work needed for DataSliceBase. Must be called prior to each search
     *
     * @param[in] frontier_type      The frontier type (i.e., edge/vertex/mixed)
     * @param[in] graph_slice        Pointer to the correspoding graph slice
     * @param[in] queue_sizing       Sizing scaling factor for work queue allocation. 1.0 by default. Reserved for future use.
     * @param[in] _USE_DOUBLE_BUFFER Whether to use double buffer
     * @param[in] queue_sizing1      Scaling factor for frontier_queue1
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Reset(
        FrontierType frontier_type,
        GraphSlice<SizeT, VertexId, Value>  
               *graph_slice,
        double  queue_sizing       = 2.0,
        bool    _USE_DOUBLE_BUFFER = false, 
        double  queue_sizing1      = -1.0)
    {   
        cudaError_t retval = cudaSuccess;
        for (int peer=0; peer<num_gpus; peer++)
            out_length[peer] = 1;
        if (queue_sizing1<0) queue_sizing1 = queue_sizing;

        //  
        // Allocate frontier queues if necessary
        //  

        // Determine frontier queue sizes
        SizeT new_frontier_elements[2] = {0,0};
        if (num_gpus>1) util::cpu_mt::PrintCPUArray<int, SizeT>("in_counter", graph_slice->in_counter.GetPointer(util::HOST), num_gpus+1, gpu_idx);

        for (int peer=0;peer<(num_gpus>1?num_gpus+1:1);peer++)
        for (int i=0; i< 2; i++)
        {
            double queue_sizing_ = i==0?queue_sizing : queue_sizing1;
            switch (frontier_type) {
                case VERTEX_FRONTIERS :
                    // O(n) ping-pong global vertex frontiers
                    new_frontier_elements[0] = double(num_gpus>1? graph_slice->in_counter[peer]:graph_slice->nodes) * queue_sizing_ +2;
                    new_frontier_elements[1] = new_frontier_elements[0];
                    break;

                case EDGE_FRONTIERS :
                    // O(m) ping-pong global edge frontiers
                    new_frontier_elements[0] = double(graph_slice->edges) * queue_sizing_ +2;
                    new_frontier_elements[1] = new_frontier_elements[0];
                    break;

                case MIXED_FRONTIERS :
                    // O(n) global vertex frontier, O(m) global edge frontier
                    new_frontier_elements[0] = double(num_gpus>1?graph_slice->in_counter[peer]:graph_slice->nodes) * queue_sizing_ +2;
                    new_frontier_elements[1] = double(graph_slice->edges) * queue_sizing_ +2;
                    break;
            }    

            // if froniter_queue is not big enough
            if (frontier_queues[peer].keys[i].GetSize() < new_frontier_elements[i]) {

                // If previously allocated
                if (frontier_queues[peer].keys[i].GetPointer(util::DEVICE) != NULL && frontier_queues[peer].keys[i].GetSize()!=0) {
                    if (retval = frontier_queues[peer].keys[i].EnsureSize(new_frontier_elements[i])) return retval;
                } else {
                    if (retval = frontier_queues[peer].keys[i].Allocate(new_frontier_elements[i], util::DEVICE)) return retval;
                }

                // If use double buffer
                if (_USE_DOUBLE_BUFFER) {
                    if (frontier_queues[peer].values[i].GetPointer(util::DEVICE) != NULL &&frontier_queues[peer].values[i].GetSize()!=0) {
                        if (retval = frontier_queues[peer].values[i].EnsureSize(new_frontier_elements[i])) return retval;
                    } else {
                        if (retval = frontier_queues[peer].values[i].Allocate(new_frontier_elements[i], util::DEVICE)) return retval;
                    }
                }

            } //end if

            if (i==1) continue;

            // Allocate scanned_edges
            SizeT max_elements = new_frontier_elements[0];
            if (new_frontier_elements[1] > max_elements) max_elements=new_frontier_elements[1];
            if (scanned_edges[peer].GetSize() < max_elements)
            {
                if (scanned_edges[peer].GetPointer(util::DEVICE) != NULL && scanned_edges[peer].GetSize() != 0) {
                    if (retval = scanned_edges[peer].EnsureSize(max_elements)) return retval;
                } else {
                    if (retval = scanned_edges[peer].Allocate(max_elements, util::DEVICE)) return retval;
                }
            }
        }

        return retval;
    } // end Reset(...)

}; // end DataSliceBase

/**
 * @brief Base test parameter structure
 */
struct TestParameter_Base {
public:
    bool          g_quick           ; // Whether or not to skip CPU based computation
    bool          g_stream_from_host; // Whether or not to use stream data from host
    bool          g_undirected      ; // Whether or not to use undirected graph
    bool          instrumented      ; // Whether or not to collect instrumentation from kernels
    bool          debug             ; // Whether or not to use debug mode  
    bool          size_check        ; // Whether or not to enable size_check
    bool          mark_predecessors ; // Whether or not to mark src-distance vs. parent vertices
    bool          enable_idempotence; // Whether or not to enable idempotence operation
    void         *graph             ; // Pointer to the input CSR graph  
    long long     src               ; // Source vertex ID
    int           max_grid_size     ; // maximum grid size (0: leave it up to the enactor)
    int           num_gpus          ; // Number of GPUs for multi-gpu enactor to use
    double        max_queue_sizing  ; // Maximum size scaling factor for work queues (e.g., 1.0 creates n and m-element vertex and edge frontiers).
    double        max_in_sizing     ; // Maximum size scaling factor for data communication  
    void         *context           ; // GPU context array used by morden gpu
    std::string   partition_method  ; // Partition method
    int          *gpu_idx           ; // Array of GPU indices 
    cudaStream_t *streams           ; // Array of GPU streams
    float         partition_factor  ; // Partition factor
    int           partition_seed    ; // Partition seed
    int           iterations        ; // Number of repeats
    int           traversal_mode    ; // Load-balacned or Dynamic cooperative

    /**
     * @brief TestParameter_Base constructor
     */
    TestParameter_Base()
    {  
        // Assign default values 
        g_quick            = false;
        g_stream_from_host = false;
        g_undirected       = false;
        instrumented       = false;
        debug              = false;
        size_check         = true;
        graph              = NULL;
        src                = -1; 
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
        traversal_mode     = -1;
    } // end TestParameter_Base() 
  
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
            args.GetCmdLineArgument("partition-method",partition_method);
    } // end Init(..)
};

/**
 * @brief Base problem structure.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to use double buffer
 * @tparam _MARK_PREDCESSORS    Whether or not to mark predecessors for vertices
 * @tparam _ENABLE_IDEMPOTENCE  Whether or not to use idempotence
 * @tparam _USE_DOUBLE_BUFFER   Whether or not to use double buffer for frontier queues
 * @tparam _ENABLE_BACKWARD     Whether or not to use backward propergation
 * @tparam _KEEP_ORDER          Whether or not to keep vertices order after partitioning
 * @tparam _KEEP_NODE_NUM       Whether or not to keep vertex IDs after partitioning
 */
template <
    typename    _VertexId,
    typename    _SizeT,
    typename    _Value,
    bool        _MARK_PREDECESSORS,
    bool        _ENABLE_IDEMPOTENCE,
    bool        _USE_DOUBLE_BUFFER,
    bool        _ENABLE_BACKWARD = false,
    bool        _KEEP_ORDER      = false,
    bool        _KEEP_NODE_NUM   = false>
struct ProblemBase
{
    typedef _VertexId           VertexId;
    typedef _SizeT              SizeT;
    typedef _Value              Value;
    static const bool           MARK_PREDECESSORS  = _MARK_PREDECESSORS ;
    static const bool           ENABLE_IDEMPOTENCE = _ENABLE_IDEMPOTENCE;
    static const bool           USE_DOUBLE_BUFFER  = _USE_DOUBLE_BUFFER ;
    static const bool           ENABLE_BACKWARD    = _ENABLE_BACKWARD   ;

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
    GraphSlice<SizeT, VertexId, Value>  
                        **graph_slices        ; // Set of graph slices (one for each GPU)
    Csr<VertexId,Value,SizeT> *sub_graphs     ; // Subgraphs for multi-gpu implementation
    Csr<VertexId,Value,SizeT> *org_graph      ; // Original graph
    PartitionerBase<VertexId,SizeT,Value,_ENABLE_BACKWARD, _KEEP_ORDER, _KEEP_NODE_NUM>
                        *partitioner          ; // Partitioner
    int                 **partition_tables    ; // Partition tables indicating which GPU the vertices are hosted
    VertexId            **convertion_tables   ; // Convertions tables indicating vertex IDs on local / remote GPUs 
    VertexId            **original_vertexes   ; // Vertex IDs in the original graph
    SizeT               **in_counter          ; // Number of in vertices
    SizeT               **out_offsets         ; // Out offsets for data communication
    SizeT               **out_counter         ; // Number of out vertices
    SizeT               **backward_offsets    ; // Offsets for backward propergation
    int                 **backward_partitions ; // Partition tables for backward propergation
    VertexId            **backward_convertions; // Convertion tables for backward propergation

    // Methods
    
    /**
     * @brief ProblemBase default constructor
     */
    ProblemBase() :
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
        backward_convertions(NULL)
    {
    } // end ProblemBase()
    
    /**
     * @brief ProblemBase default destructor to free all graph slices allocated.
     */
    virtual ~ProblemBase()
    {
        // Cleanup graph slices on the heap
        for (int i = 0; i < num_gpus; ++i)
        {
            delete   graph_slices     [i  ]; graph_slices     [i  ] = NULL;
        }
        if (num_gpus > 1)
        {
            delete   partitioner;           partitioner          = NULL;
        }
        delete[] graph_slices; graph_slices = NULL;
        delete[] gpu_idx;      gpu_idx      = NULL;
    } // end ~ProblemBase()

    /**
     * @brief Get the GPU index for a specified vertex id.
     *
     * @tparam VertexId Type of signed integer to use as vertex id
     * @param[in] vertex Vertex Id to search
     * \return Index of the gpu that owns the neighbor list of the specified vertex
     */
    template <typename VertexId>
    int GpuIndex(VertexId vertex)
    {
        if (num_gpus <= 1) {
            
            // Special case for only one GPU, which may be set as with
            // an ordinal other than 0.
            return graph_slices[0]->index;
        } else {
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
    template <typename VertexId>
    VertexId GraphSliceRow(VertexId vertex)
    {
        if (num_gpus <= 1) {
            return vertex;
        } else {
            return convertion_tables[0][vertex];
        }
    }

    /**
     * @brief Initialize problem from host CSR graph.
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph            Pointer to the input CSR graph.
     * @param[in] inverse_graph    Pointer to the inversed input CSR graph.
     * @param[in] num_gpus         Number of gpus
     * @param[in] gpu_idx          Array of gpu indices
     * @param[in] partition_method Partition methods
     * @param[in] queue_sizing     Queue sizing
     * @param[in] partition_factor Partition factor
     * @param[in] partition_seed   Partition seed
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
        bool        stream_from_host,
        Csr<VertexId, Value, SizeT> *graph,
        Csr<VertexId, Value, SizeT> *inverse_graph = NULL,
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

        do {
            if (num_gpus==1 && gpu_idx==NULL)
            {
                if (retval = util::GRError(cudaGetDevice(&(this->gpu_idx[0])), "ProblemBase cudaGetDevice failed", __FILE__, __LINE__)) break;
            } else {
                for (int gpu=0;gpu<num_gpus;gpu++)
                    this->gpu_idx[gpu]=gpu_idx[gpu];
            }

            graph_slices = new GraphSlice<SizeT, VertexId, Value>*[num_gpus];

            if (num_gpus >1)
            {
                util::CpuTimer cpu_timer;

                printf("partition_method = %s\n", partition_method.c_str());
                if (partition_method=="random")
                    partitioner=new rp::RandomPartitioner   <VertexId, SizeT, Value, _ENABLE_BACKWARD, _KEEP_ORDER, _KEEP_NODE_NUM>
                        (*graph,num_gpus);
                else if (partition_method=="metis")
                    partitioner=new metisp::MetisPartitioner<VertexId, SizeT, Value, _ENABLE_BACKWARD, _KEEP_ORDER, _KEEP_NODE_NUM>
                        (*graph,num_gpus);
                else if (partition_method=="static")
                    partitioner=new sp::StaticPartitioner<VertexId, SizeT, Value, _ENABLE_BACKWARD, _KEEP_ORDER, _KEEP_NODE_NUM>
                        (*graph,num_gpus);
                else if (partition_method=="cluster")
                    partitioner=new cp::ClusterPartitioner  <VertexId, SizeT, Value, _ENABLE_BACKWARD, _KEEP_ORDER, _KEEP_NODE_NUM>
                        (*graph,num_gpus);
                else if (partition_method=="biasrandom")
                    partitioner=new brp::BiasRandomPartitioner <VertexId, SizeT, Value, _ENABLE_BACKWARD, _KEEP_ORDER, _KEEP_NODE_NUM>
                        (*graph,num_gpus);
                else util::GRError("partition_method invalid", __FILE__,__LINE__);
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
                printf("partition end. (%f ms)\n", cpu_timer.ElapsedMillis());fflush(stdout);
                
                /*graph->DisplayGraph("org_graph",graph->nodes);
                util::cpu_mt::PrintCPUArray<SizeT,int>("partition0",partition_tables[0],graph->nodes);
                util::cpu_mt::PrintCPUArray<SizeT,VertexId>("convertion0",convertion_tables[0],graph->nodes);
                //util::cpu_mt::PrintCPUArray<SizeT,Value>("edge_value",graph->edge_values,graph->edges);
                for (int gpu=0;gpu<num_gpus;gpu++)
                {
                    sub_graphs[gpu].DisplayGraph("sub_graph",sub_graphs[gpu].nodes);
                    printf("%d\n",gpu);
                    util::cpu_mt::PrintCPUArray<SizeT,int     >("partition"           , partition_tables    [gpu+1], sub_graphs[gpu].nodes);
                    util::cpu_mt::PrintCPUArray<SizeT,VertexId>("convertion"          , convertion_tables   [gpu+1], sub_graphs[gpu].nodes);
                    //util::cpu_mt::PrintCPUArray<SizeT,SizeT   >("backward_offsets"    , backward_offsets    [gpu], sub_graphs[gpu].nodes);
                    //util::cpu_mt::PrintCPUArray<SizeT,int     >("backward_partitions" , backward_partitions [gpu], backward_offsets[gpu][sub_graphs[gpu].nodes]);
                    //util::cpu_mt::PrintCPUArray<SizeT,VertexId>("backward_convertions", backward_convertions[gpu], backward_offsets[gpu][sub_graphs[gpu].nodes]);
                }*/
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
                if (retval) break;
            } else {
                sub_graphs=graph;
            }

            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                graph_slices[gpu] = new GraphSlice<SizeT, VertexId, Value>(this->gpu_idx[gpu]);
                if (num_gpus > 1)
                {
                    if (_ENABLE_BACKWARD)
                        retval = graph_slices[gpu]->Init(
                            stream_from_host,
                            num_gpus,
                            &(sub_graphs     [gpu]),
                            NULL,
                            partition_tables    [gpu+1],
                            convertion_tables   [gpu+1],
                            original_vertexes   [gpu],
                            in_counter          [gpu],
                            out_offsets         [gpu],
                            out_counter         [gpu],
                            backward_offsets    [gpu],
                            backward_partitions [gpu],
                            backward_convertions[gpu]);
                    else  
                        retval = graph_slices[gpu]->Init(
                            stream_from_host,
                            num_gpus,
                            &(sub_graphs[gpu]),
                            NULL,
                            partition_tables [gpu+1],
                            convertion_tables[gpu+1],
                            original_vertexes[gpu],
                            in_counter       [gpu],
                            out_offsets      [gpu],
                            out_counter      [gpu],
                            NULL,
                            NULL,
                            NULL);
                } else retval = graph_slices[gpu]->Init(
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
               if (retval) break;
            }// end for (gpu)

       } while (0);

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
