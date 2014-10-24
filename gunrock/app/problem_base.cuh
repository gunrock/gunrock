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

#include <gunrock/util/basic_utils.cuh>
#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/multiple_buffering.cuh>
#include <gunrock/util/io/modified_load.cuh>
#include <gunrock/util/io/modified_store.cuh>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/app/rp/rp_partitioner.cuh>
#include <gunrock/app/cp/cp_partitioner.cuh>
#include <gunrock/app/brp/brp_partitioner.cuh>
#include <gunrock/app/metisp/metis_partitioner.cuh>
#include <vector>
#include <string>

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

    template <
        typename SizeT,
        typename VertexId,
        typename Value>
    struct DataSliceBase
    {
        int                               num_gpus,gpu_idx,wait_counter,gpu_mallocing;
        int                               num_vertex_associate,num_value__associate,num_stages;
        SizeT                             nodes;
        util::Array1D<SizeT, VertexId    > **vertex_associate_in  [2];
        util::Array1D<SizeT, VertexId*   >  *vertex_associate_ins [2];
        util::Array1D<SizeT, VertexId    > **vertex_associate_out    ;
        util::Array1D<SizeT, VertexId*   >  *vertex_associate_outs   ;
        util::Array1D<SizeT, VertexId*   >   vertex_associate_orgs   ;
        util::Array1D<SizeT, Value       > **value__associate_in  [2];
        util::Array1D<SizeT, Value*      >  *value__associate_ins [2];
        util::Array1D<SizeT, Value       > **value__associate_out    ;
        util::Array1D<SizeT, Value*      >  *value__associate_outs   ;
        util::Array1D<SizeT, Value*      >   value__associate_orgs   ;
        util::Array1D<SizeT, SizeT       >   out_length    ;   
        util::Array1D<SizeT, SizeT       >   in_length  [2];   
        util::Array1D<SizeT, VertexId    >  *keys_in    [2];
        //util::Array1D<SizeT, VertexId    >  *keys_out      ;
        util::Array1D<SizeT, SizeT       >  *keys_marker   ;
        util::Array1D<SizeT, cudaEvent_t*>   events     [4];
        util::Array1D<SizeT, bool*       >   events_set [4];
        util::Array1D<SizeT, int         >   wait_marker   ;
        util::Array1D<SizeT, cudaStream_t>   streams       ;
        util::Array1D<SizeT, int         >   stages        ;
        util::Array1D<SizeT, cudaEvent_t >   local_events  ;
        util::Array1D<SizeT, bool        >   to_show       ;

        DataSliceBase()
        {
            num_stages               = 8;
            num_vertex_associate     = 0;
            num_value__associate     = 0;
            gpu_idx                  = 0;
            gpu_mallocing            = 0;
            //keys_out                 = NULL;
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
            //vertex_associate_ins[0].SetName("vertex_associate_ins[0]");
            //vertex_associate_ins[1].SetName("vertex_associate_ins[1]");
            //vertex_associate_outs  .SetName("vertex_associate_outs"  );  
            //vertex_associate_orgs  .SetName("vertex_associate_orgs"  );  
            //value__associate_ins[0].SetName("value__associate_ins[0]");
            //value__associate_ins[1].SetName("value__associate_ins[1]");
            //value__associate_outs  .SetName("value__associate_outs"  );  
            //value__associate_orgs  .SetName("value__associate_orgs"  );  
            out_length             .SetName("out_length"             );  
            in_length           [0].SetName("in_length[0]"           );  
            in_length           [1].SetName("in_length[1]"           );  
            //keys_in             [0].SetName("keys_in[0]"             );  
            //keys_in             [1].SetName("keys_in[1]"             );
            wait_marker            .SetName("wait_marker"            );
            stages                 .SetName("stages"                 );
            local_events           .SetName("local_events"           );
            to_show                .SetName("to_show"                );
            for (int i=0;i<4;i++)
            {
                events[i].SetName("events[]");
                events_set[i].SetName("events_set[]");
            }
            streams                .SetName("streams"                );
        } // DataSliceBase()

        ~DataSliceBase()
        {
            util::cpu_mt::PrintMessage("~DataSliceBase() begin.");
            if (util::SetDevice(gpu_idx)) return;

            /*if (num_gpus > 1)
            {
                for (int gpu=0;gpu<num_gpus;gpu++)
                    util::GRError(cudaStreamDestroy(streams[gpu]), "cudaStreamDestroy failed.", __FILE__, __LINE__);
            }*/

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
                //vertex_associate_ins[0].Release();
                //vertex_associate_ins[1].Release();
            }

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
                //value__associate_ins[0].Release();
                //value__associate_ins[1].Release();
            }
            
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

            if (keys_marker !=NULL)
            {
                for (int gpu=0;gpu<num_gpus;gpu++)
                {
                    //keys_out   [gpu].Release();
                    keys_marker[gpu].Release();
                }
                //delete[] keys_out   ; keys_out    = NULL;
                delete[] keys_marker; keys_marker = NULL;
            }

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
            }

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
            }

            for (int gpu=0;gpu<num_gpus;gpu++)
                cudaEventDestroy(local_events[gpu]);
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
            //keys_in    [0].Release();
            //keys_in    [1].Release();
            in_length  [0].Release();
            in_length  [1].Release();
            wait_marker   .Release();
            local_events  .Release();
            out_length    .Release();
            vertex_associate_orgs.Release();
            value__associate_orgs.Release();
            streams       .Release();
            stages        .Release();
            to_show       .Release();

            util::cpu_mt::PrintMessage("~DataSliceBase() end.");
        } // ~DataSliceBase()

        cudaError_t Init(
            int   num_gpus,
            int   gpu_idx,
            int   num_vertex_associate,
            int   num_value__associate,
            Csr<VertexId, Value, SizeT> *graph,
            SizeT *num_in_nodes,
            SizeT *num_out_nodes,
            float in_sizing = 1.0)
        {
            cudaError_t retval         = cudaSuccess;
            this->num_gpus             = num_gpus;
            this->gpu_idx              = gpu_idx;
            this->nodes                = graph->nodes;
            this->num_vertex_associate = num_vertex_associate;
            this->num_value__associate = num_value__associate;
            if (retval = util::SetDevice(gpu_idx))  return retval;
            if (retval = in_length[0].Allocate(num_gpus,util::HOST)) return retval;
            if (retval = in_length[1].Allocate(num_gpus,util::HOST)) return retval;
            if (retval = out_length  .Allocate(num_gpus,util::HOST | util::DEVICE)) return retval;
            //if (retval = streams     .Allocate(num_gpus,util::HOST)) return retval;
            if (retval = vertex_associate_orgs.Allocate(num_vertex_associate, util::HOST | util::DEVICE)) return retval;
            if (retval = value__associate_orgs.Allocate(num_value__associate, util::HOST | util::DEVICE)) return retval;

            wait_marker .Allocate(num_gpus);
            stages      .Allocate(num_gpus);
            local_events.Allocate(num_gpus);
            to_show     .Allocate(num_gpus);
            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                wait_marker[gpu]=0;
                if (retval = util::GRError(cudaEventCreate(&(local_events[gpu])), "cudaEventCreate failed", __FILE__, __LINE__)) return retval;
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
                        if (retval = util::GRError(cudaEventCreate(&(events[i][gpu][stage])), "cudaEventCreate failed.", __FILE__, __LINE__)) return retval;
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
                //if (num_in_nodes[gpu] <= 0)
                //{
                //    vertex_associate_in [t][gpu] = NULL;
                //    value__associate_in [t][gpu] = NULL;
                //} else {
                    vertex_associate_in [t][gpu] = new util::Array1D<SizeT,VertexId>[num_vertex_associate];
                    for (int i=0;i<num_vertex_associate;i++)
                    {
                        vertex_associate_in [t][gpu][i].SetName("vertex_associate_in[]");
                        if (retval = vertex_associate_in[t][gpu][i].Allocate(num_in_node,util::DEVICE)) return retval;
                    }
                    value__associate_in [t][gpu] = new util::Array1D<SizeT,Value   >[num_value__associate];
                    for (int i=0;i<num_value__associate;i++)
                    {
                        value__associate_in[t][gpu][i].SetName("value__associate_ins[]");
                        if (retval = value__associate_in[t][gpu][i].Allocate(num_in_node,util::DEVICE)) return retval;
                    }
                //}
                    
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

            // Create outgoing buffer on device
            //if (num_out_nodes > 0)
            {
                vertex_associate_out  = new util::Array1D<SizeT,VertexId >*[num_gpus];
                vertex_associate_outs = new util::Array1D<SizeT,VertexId*> [num_gpus];
                value__associate_out  = new util::Array1D<SizeT,Value    >*[num_gpus];
                value__associate_outs = new util::Array1D<SizeT,Value*   > [num_gpus];
                //keys_out              = new util::Array1D<SizeT,VertexId > [num_gpus];
                keys_marker           = new util::Array1D<SizeT,SizeT    > [num_gpus]; 
                for (int gpu=0;gpu<num_gpus;gpu++)
                {
                    SizeT num_out_node = num_out_nodes[gpu] * in_sizing;
                    keys_marker[gpu].SetName("keys_marker[]");
                    if (retval = keys_marker[gpu].Allocate(num_out_nodes[num_gpus]*in_sizing, util::DEVICE)) return retval;

                    vertex_associate_out [gpu] = new util::Array1D<SizeT,VertexId>[num_vertex_associate];
                    vertex_associate_outs[gpu].SetName("vertex_associate_outs[]");
                    if (retval = vertex_associate_outs[gpu].Allocate(num_vertex_associate, util::HOST | util::DEVICE)) return retval;
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
                    for (int i=0;i<num_value__associate;i++)
                    {
                        value__associate_out[gpu][i].SetName("value__associate_out[][]");
                        if (gpu!=0)
                            if (retval = value__associate_out[gpu][i].Allocate(num_out_node, util::DEVICE)) return retval;
                        value__associate_outs[gpu][i]=value__associate_out[gpu][i].GetPointer(util::DEVICE);
                    }
                    if (retval = value__associate_outs[gpu].Move(util::HOST, util::DEVICE)) return retval;
                }
            }
            
            return retval;
        } // Init(..)

    }; // end DataSliceBase
 
/**
 * @brief Base problem structure.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to use double buffer
 */
template <
    typename    _VertexId,
    typename    _SizeT,
    typename    _Value,
    bool        _USE_DOUBLE_BUFFER,
    bool        _ENABLE_BACKWARD = false>

struct ProblemBase
{
    typedef _VertexId           VertexId;
    typedef _SizeT              SizeT;
    typedef _Value              Value;

    /**
     * Load instruction cache-modifier const defines.
     */

    static const util::io::ld::CacheModifier QUEUE_READ_MODIFIER                    = util::io::ld::cg;             // Load instruction cache-modifier for reading incoming frontier vertex-ids. Valid on SM2.0 or newer
    static const util::io::ld::CacheModifier COLUMN_READ_MODIFIER                   = util::io::ld::NONE;           // Load instruction cache-modifier for reading CSR column-indices.
    static const util::io::ld::CacheModifier EDGE_VALUES_READ_MODIFIER              = util::io::ld::NONE;           // Load instruction cache-modifier for reading edge values.
    static const util::io::ld::CacheModifier ROW_OFFSET_ALIGNED_READ_MODIFIER       = util::io::ld::cg;             // Load instruction cache-modifier for reading CSR row-offsets (8-byte aligned)
    static const util::io::ld::CacheModifier ROW_OFFSET_UNALIGNED_READ_MODIFIER     = util::io::ld::NONE;           // Load instruction cache-modifier for reading CSR row-offsets (4-byte aligned)
    static const util::io::st::CacheModifier QUEUE_WRITE_MODIFIER                   = util::io::st::cg;             // Store instruction cache-modifier for writing outgoing frontier vertex-ids. Valid on SM2.0 or newer

    /**
     * @brief Graph slice structure which contains common graph structural data and input/output queue.
     */
    struct GraphSlice
    {
        int             num_gpus;
        int             index;                              // Slice index
        VertexId        nodes;                              // Number of nodes in slice
        SizeT           edges;                              // Number of edges in slice
        //cudaStream_t    stream;                             // CUDA stream to use for processing the slice

        Csr<VertexId, Value, SizeT   > *graph             ; // Pointer to CSR format subgraph
        util::Array1D<SizeT, SizeT   > row_offsets        ; // CSR format row offset on device memory
        util::Array1D<SizeT, VertexId> column_indices     ; // CSR format column indices on device memory
        util::Array1D<SizeT, SizeT   > column_offsets     ; // CSR format column offset on device memory
        util::Array1D<SizeT, VertexId> row_indices        ; // CSR format row indices on device memory
        util::Array1D<SizeT, int     > partition_table    ; // Partition number for vertexes, local is always 0
        util::Array1D<SizeT, VertexId> convertion_table   ; // Vertex number of vertexes in their hosting partition
        util::Array1D<SizeT, VertexId> original_vertex    ;
        //util::Array1D<SizeT, SizeT   > in_offset          ;
        util::Array1D<SizeT, SizeT   > in_counter         ;
        util::Array1D<SizeT, SizeT   > out_offset         ;
        util::Array1D<SizeT, SizeT   > out_counter        ;
        util::Array1D<SizeT, SizeT   > backward_offset    ;
        util::Array1D<SizeT, int     > backward_partition ;
        util::Array1D<SizeT, VertexId> backward_convertion;

        //Frontier queues. Used to track working frontier.
        util::DoubleBuffer<SizeT, VertexId, VertexId>  *frontier_queues;
        //SizeT                                          **frontier_elements;

        /**
         * @brief GraphSlice Constructor
         *
         * @param[in] index GPU index, reserved for multi-GPU use in future.
         * @param[in] stream CUDA Stream we use to allocate storage for this graph slice.
         */
        GraphSlice(int index) :
            index(index),
            graph(NULL),
            num_gpus(0),
            //frontier_elements(NULL),
            frontier_queues(NULL),
            //d_row_offsets(NULL),
            //d_column_indices(NULL),
            //d_column_offsets(NULL),
            //d_row_indices(NULL),
            nodes(0),
            edges(0)//,
            //stream(stream)
        {
            util::cpu_mt::PrintMessage("GraphSlice() begin.");
            row_offsets        .SetName("row_offsets"        );
            column_indices     .SetName("column_indices"     );
            column_offsets     .SetName("column_offsets"     );
            row_indices        .SetName("row_indices"        );
            partition_table    .SetName("partition_table"    );
            convertion_table   .SetName("convertion_table"   );
            original_vertex    .SetName("original_vertex"    );
            //in_offset          .SetName("in_offset"          );
            in_counter         .SetName("in_counter"         );  
            out_offset         .SetName("out_offset"         );
            out_counter        .SetName("out_counter"      );
            backward_offset    .SetName("backward_offset"    );
            backward_partition .SetName("backward_partition" );
            backward_convertion.SetName("backward_convertion");

            // Initialize double buffer frontier queue lengths
            /*for (int i = 0; i < 2; ++i)
            {
                frontier_elements[i] = 0;
            }*/
            util::cpu_mt::PrintMessage("GraphSlice() end.");
        }

        /**
         * @brief GraphSlice Destructor to free all device memories.
         */
        virtual ~GraphSlice()
        {
            util::cpu_mt::PrintMessage("~GraphSlice() begin.");
            // Set device (use slice index)
            //util::GRError(cudaSetDevice(index), "GpuSlice cudaSetDevice failed", __FILE__, __LINE__);
            util::SetDevice(index);

            // Free pointers
            row_offsets        .Release();
            column_indices     .Release();
            column_offsets     .Release();
            row_indices        .Release();
            partition_table    .Release();
            convertion_table   .Release();
            original_vertex    .Release();
            //in_offset          .Release();
            in_counter         .Release();
            out_offset         .Release();
            out_counter        .Release();
            backward_offset    .Release();
            backward_partition .Release();
            backward_convertion.Release();

            for (int gpu = 0; gpu<=num_gpus; gpu++)
            {
                //delete[] frontier_elements[gpu];frontier_elements[gpu]=NULL;
                for (int i = 0; i < 2; ++i) {
                    frontier_queues[gpu].keys  [i].Release();
                    frontier_queues[gpu].values[i].Release();
                }
            }
            //delete[] frontier_elements; frontier_elements=NULL;
            delete[] frontier_queues; frontier_queues=NULL;

            // Destroy stream
            /*if (stream) {
                util::GRError(cudaStreamDestroy(stream), "GpuSlice cudaStreamDestroy failed", __FILE__, __LINE__);
            }*/
            util::cpu_mt::PrintMessage("~GraphSlice() end.");
        }

       /**
         * @brief Initalize graph slice
         * @param[in] stream_from_host Whether to stream data from host
         * @param[in] num_gpus Number of gpus
         * @param[in] graph Pointer to the sub_graph
         * @param[in] partition_table 
         * @param[in] convertion_table
         * @param[in] in_offset
         * @param[in] out_offset
         * \return cudaError_t Object incidating the success of all CUDA function calls
         */
        cudaError_t Init(
            bool                       stream_from_host,
            int                        num_gpus,
            Csr<VertexId,Value,SizeT>* graph,
            Csr<VertexId,Value,SizeT>* inverstgraph,
            int*                       partition_table,
            VertexId*                  convertion_table,
            VertexId*                  original_vertex,
            //SizeT*                     in_offset,
            SizeT*                     in_counter,
            SizeT*                     out_offset,
            SizeT*                     out_counter,
            SizeT*                     backward_offsets   = NULL,
            int*                       backward_partition = NULL,
            VertexId*                  backward_convertion= NULL)
        {
            util::cpu_mt::PrintMessage("GraphSlice Init() begin.");
            cudaError_t retval     = cudaSuccess;
            this->num_gpus         = num_gpus;
            this->graph            = graph;
            nodes                  = graph->nodes;
            edges                  = graph->edges;
            this->partition_table    .SetPointer(partition_table      , nodes     );
            this->convertion_table   .SetPointer(convertion_table     , nodes     );
            this->original_vertex    .SetPointer(original_vertex      , nodes     );
            //this->in_offset          .SetPointer(in_offset            , num_gpus+1);
            this->in_counter         .SetPointer(in_counter           , num_gpus+1);
            this->out_offset         .SetPointer(out_offset           , num_gpus+1);
            this->out_counter        .SetPointer(out_counter          , num_gpus+1);
            this->row_offsets        .SetPointer(graph->row_offsets   , nodes+1   );
            this->column_indices     .SetPointer(graph->column_indices, edges     );

            do {
                if (retval = util::GRError(cudaSetDevice(index), "GpuSlice cudaSetDevice failed", __FILE__, __LINE__)) break;
                // Allocate and initialize row_offsets
                if (retval = this->row_offsets.Allocate(nodes+1      ,util::DEVICE)) break;
                if (retval = this->row_offsets.Move    (util::HOST   ,util::DEVICE)) break;
                
                // Allocate and initialize column_indices
                if (retval = this->column_indices.Allocate(edges     ,util::DEVICE)) break;
                if (retval = this->column_indices.Move    (util::HOST,util::DEVICE)) break;
                frontier_queues   = new util::DoubleBuffer<SizeT, VertexId, VertexId>[num_gpus+1]; 
                //frontier_elements = new SizeT*[num_gpus+1];
                //for (int gpu=0;gpu<=num_gpus;gpu++)
                //{ 
                //    frontier_elements[gpu] = new SizeT[2];
                //    frontier_elements[gpu][0]=0;
                //    frontier_elements[gpu][1]=0;
                //}
                /*if (graph->column_offsets !=NULL)
                {
                    // Allocate and initalize column_offsets
                    this->column_offsets.SetPointer(column_offsets, nodes+1);
                    if (retval = this->column_offsets.Allocate(nodes+1   , util::DEVICE)) break;
                    if (retval = this->column_offsets.Move    (util::HOST, util::DEVICE)) break; 
                }

                if (graph->row_indices !=NULL)
                {
                    // Allocate and initalize row_indices
                    this->row_indices.SetPointer(row_indices, edges);
                    if (retval = this->row_indices.Allocate(edges     , util::DEVICE)) break;
                    if (retval = this->row_indices.Move    (util::HOST, util::DEVICE)) break;
                }*/

                // For multi-GPU cases
                if (num_gpus > 1)
                {
                    // Allocate and initalize convertion_table
                    if (retval = this->partition_table.Allocate (nodes     ,util::DEVICE)) break;
                    if (retval = this->partition_table.Move     (util::HOST,util::DEVICE)) break;
                    
                    // Allocate and initalize convertion_table
                    if (retval = this->convertion_table.Allocate(nodes     ,util::DEVICE)) break;
                    if (retval = this->convertion_table.Move    (util::HOST,util::DEVICE)) break;

                    // Allocate and initalize original_vertex
                    if (retval = this->original_vertex .Allocate(nodes     ,util::DEVICE)) break;
                    if (retval = this->original_vertex .Move    (util::HOST,util::DEVICE)) break;
                    
                    // Allocate and initalize in_offset
                    //if (retval = this->in_offset       .Allocate(num_gpus+1,util::DEVICE)) break;
                    //if (retval = this->in_offset       .Move    (util::HOST,util::DEVICE)) break;

                    if (_ENABLE_BACKWARD)
                    {
                        this->backward_offset    .SetPointer(backward_offsets     , in_counter[0]+1);
                        this->backward_partition .SetPointer(backward_partition   , in_counter[num_gpus]);
                        this->backward_convertion.SetPointer(backward_convertion  , in_counter[num_gpus]);

                        if (retval = this->backward_offset    .Allocate(in_counter[0]+1, util::DEVICE)) break;
                        if (retval = this->backward_offset    .Move(util::HOST, util::DEVICE)) break;
                        
                        if (retval = this->backward_partition .Allocate(in_counter[num_gpus], util::DEVICE)) break;
                        if (retval = this->backward_partition .Move(util::HOST, util::DEVICE)) break;
                        
                        if (retval = this->backward_convertion.Allocate(in_counter[num_gpus], util::DEVICE)) break;
                        if (retval = this->backward_convertion.Move(util::HOST, util::DEVICE)) break;
                    }
                } // end if num_gpu>1
            } while (0);

            util::cpu_mt::PrintMessage("GraphSlice Init() end.");
            return retval;
        } // end of Init(...)
 
        /** 
         * @brief Performs any initialization work needed for GraphSlice. Must be called prior to each search
         *
         * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
         * @param[in] queue_sizing Sizing scaling factor for work queue allocation. 1.0 by default. Reserved for future use.
         *
         * \return cudaError_t object which indicates the success of all CUDA function calls.
         */
        cudaError_t Reset(
            FrontierType frontier_type,     // The frontier type (i.e., edge/vertex/mixed)
            double queue_sizing = 2.0)            // Size scaling factor for work queue allocation
        {   
            util::cpu_mt::PrintMessage("GraphSlice Reset() begin.");
            cudaError_t retval = cudaSuccess;

            // Set device
            if (retval = util::SetDevice(index)) return retval;

            //  
            // Allocate frontier queues if necessary
            //  

            // Determine frontier queue sizes
            SizeT new_frontier_elements[2] = {0,0};
            if (num_gpus>1) util::cpu_mt::PrintCPUArray<int, SizeT>("in_counter", in_counter.GetPointer(util::HOST), num_gpus+1, index);

            for (int peer=0;peer<(num_gpus>1?num_gpus+1:1);peer++)
            {
                //printf("gpu = %d, peer = %d, cross_counter = %d\n", index, peer, cross_counter[peer]);fflush(stdout);
                switch (frontier_type) {
                    case VERTEX_FRONTIERS :
                        // O(n) ping-pong global vertex frontiers
                        new_frontier_elements[0] = double(num_gpus>1?in_counter[peer]:nodes) * queue_sizing +1;
                        new_frontier_elements[1] = new_frontier_elements[0] +1;
                        break;

                    case EDGE_FRONTIERS :
                        // O(m) ping-pong global edge frontiers
                        new_frontier_elements[0] = double(edges) * queue_sizing +1;
                        new_frontier_elements[1] = new_frontier_elements[0] +1;
                        break;

                    case MIXED_FRONTIERS :
                        // O(n) global vertex frontier, O(m) global edge frontier
                        new_frontier_elements[0] = double(num_gpus>1?in_counter[peer]:nodes) * queue_sizing +1;
                        new_frontier_elements[1] = double(edges) * queue_sizing+1;
                        break;
                 }    

                // Iterate through global frontier queue setups
                for (int i = 0; i < 2; i++) {
                    //frontier_elements[i] = new_frontier_elements[i];
                    // Allocate frontier queue if not big enough
                    //frontier_queues.keys[i].EnsureSize(frontier_elements[i]);
                    //if (_USE_DOUBLE_BUFFER) frontier_queues.values[i].EnsureSize(frontier_elements[i]);
                    //printf("peer = %d, i = %d, [] = %d\n", peer, i, new_frontier_elements[i]);
                    if (frontier_queues[peer].keys[i].GetSize() < new_frontier_elements[i]) {

                        // Free if previously allocated
                        if (retval = frontier_queues[peer].keys[i].Release()) return retval;

                        // Free if previously allocated
                        if (_USE_DOUBLE_BUFFER) {
                            if (retval = frontier_queues[peer].values[i].Release()) return retval;
                        }

                        //frontier_elements[peer][i] = new_frontier_elements[i];

                        if (retval = frontier_queues[peer].keys[i].Allocate(new_frontier_elements[i],util::DEVICE)) return retval;
                        if (_USE_DOUBLE_BUFFER) {
                            if (retval = frontier_queues[peer].values[i].Allocate(new_frontier_elements[i],util::DEVICE)) return retval;
                        }
                    } //end if
                } // end for i<2
            }
            util::cpu_mt::PrintMessage("GraphSlice Reset() end.");
            return retval;
        } // end Reset(...)

    }; // end GraphSlice

    // Members
    int                 num_gpus              ; // Number of GPUs to be sliced over
    int                 *gpu_idx              ; // GPU indices 
    SizeT               nodes                 ; // Size of the graph
    SizeT               edges                 ;
    GraphSlice          **graph_slices        ; // Set of graph slices (one for each GPU)
    Csr<VertexId,Value,SizeT> *sub_graphs     ; // Subgraphs for multi-gpu implementation
    Csr<VertexId,Value,SizeT> *org_graph      ; // Original graph
    PartitionerBase<VertexId,SizeT,Value,_ENABLE_BACKWARD>
                        *partitioner          ; // Partitioner
    int                 **partition_tables    ; // Multi-gpu partition table and convertion table
    VertexId            **convertion_tables   ;
    VertexId            **original_vertexes   ;
    //SizeT               **in_offsets          ; // Offsets for data movement between GPUs
    SizeT               **in_counter          ;
    SizeT               **out_offsets         ;
    SizeT               **out_counter         ;
    SizeT               **backward_offsets    ;
    int                 **backward_partitions ;
    VertexId            **backward_convertions;

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
        //in_offsets          (NULL),
        in_counter          (NULL),
        out_offsets         (NULL),
        out_counter         (NULL),
        backward_offsets    (NULL),
        backward_partitions (NULL),
        backward_convertions(NULL)
    {
        util::cpu_mt::PrintMessage("ProblemBase() begin.");
        util::cpu_mt::PrintMessage("ProblemBase() end.");
    }
    
    /**
     * @brief ProblemBase default destructor to free all graph slices allocated.
     */
    virtual ~ProblemBase()
    {
        util::cpu_mt::PrintMessage("~ProblemBase() begin.");
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
        util::cpu_mt::PrintMessage("~ProblemBase() end.");
   }

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
     * @param[in] nodes Number of nodes in the CSR graph.
     * @param[in] edges Number of edges in the CSR graph.
     * @param[in] h_row_offsets Host-side row offsets array.
     * @param[in] h_column_indices Host-side column indices array.
     * @param[in] h_column_offsets Host-side column offsets array.
     * @param[in] h_row_indices Host-side row indices array.
     * @param[in] num_gpus Number of the GPUs used.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
        bool        stream_from_host,
        //SizeT       nodes,
        //SizeT       edges,
        //SizeT       *h_row_offsets,
        //VertexId    *h_column_indices,
        Csr<VertexId, Value, SizeT> *graph,
        Csr<VertexId, Value, SizeT> *inverse_graph = NULL,
        //SizeT       *column_offsets = NULL,
        //VertexId    *row_indices    = NULL,
        int         num_gpus          = 1,
        int         *gpu_idx          = NULL,
        std::string partition_method  = "random",
        float       queue_sizing      = 2.0)
    {
        util::cpu_mt::PrintMessage("ProblemBase Init() begin.");
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

            graph_slices = new GraphSlice*[num_gpus];

            if (num_gpus >1)
            {
                util::CpuTimer cpu_timer;

                printf("partition_method=%s\n", partition_method.c_str());
                if (partition_method=="random")
                    partitioner=new rp::RandomPartitioner   <VertexId, SizeT, Value, _ENABLE_BACKWARD>
                        (*graph,num_gpus);
                else if (partition_method=="metis")
                    partitioner=new metisp::MetisPartitioner<VertexId, SizeT, Value, _ENABLE_BACKWARD>
                        (*graph,num_gpus);
                else if (partition_method=="cluster")
                    partitioner=new cp::ClusterPartitioner  <VertexId, SizeT, Value, _ENABLE_BACKWARD>
                        (*graph,num_gpus);
                else if (partition_method=="biasrandom")
                    partitioner=new brp::BiasRandomPartitioner <VertexId, SizeT, Value, _ENABLE_BACKWARD>
                        (*graph,num_gpus);
                else util::GRError("partition_method invalid", __FILE__,__LINE__);
                printf("partition begin.\n");fflush(stdout);
                cpu_timer.Start();
                retval = partitioner->Partition(
                    sub_graphs,
                    partition_tables,
                    convertion_tables,
                    original_vertexes,
                    //in_offsets,
                    in_counter,
                    out_offsets,
                    out_counter,
                    backward_offsets,
                    backward_partitions,
                    backward_convertions);
                cpu_timer.Stop();
                printf("partition end. (%f ms)\n", cpu_timer.ElapsedMillis());fflush(stdout);
                
                /*graph->DisplayGraph("org_graph");
                util::cpu_mt::PrintCPUArray<SizeT,int>("partition0",partition_tables[0],graph->nodes);
                util::cpu_mt::PrintCPUArray<SizeT,VertexId>("convertion0",convertion_tables[0],graph->nodes);
                //util::cpu_mt::PrintCPUArray<SizeT,Value>("edge_value",graph->edge_values,graph->edges);
                for (int gpu=0;gpu<num_gpus;gpu++)
                {
                    sub_graphs[gpu].DisplayGraph("sub_graph");
                //    printf("%d\n",gpu);
                    util::cpu_mt::PrintCPUArray<SizeT,int>("partition",partition_tables[gpu+1],sub_graphs[gpu].nodes);
                    util::cpu_mt::PrintCPUArray<SizeT,VertexId>("convertion",convertion_tables[gpu+1],sub_graphs[gpu].nodes);
                }*/
                /*for (int gpu=0;gpu<num_gpus;gpu++)
                {
                    cross_counter[gpu][num_gpus]=0;
                    for (int peer=0;peer<num_gpus;peer++)
                    {
                        cross_counter[gpu][peer]=out_offsets[gpu][peer+1]-out_offsets[gpu][peer];
                    }
                    cross_counter[gpu][num_gpus]=in_offsets[gpu][num_gpus];
                }*/
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
                graph_slices[gpu] = new GraphSlice(this->gpu_idx[gpu]);
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
                            //in_offsets          [gpu],
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
                            //in_offsets[gpu],
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
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        //NULL,
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        NULL);
               if (retval) break;
            }// end for (gpu)

       } while (0);

        util::cpu_mt::PrintMessage("ProblemBase Init() end.");
        return retval;
    }

    /**
     * @brief Performs any initialization work needed for ProblemBase. Must be called prior to each search
     *
     * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     * @param[in] queue_sizing Sizing scaling factor for work queue allocation. 1.0 by default. Reserved for future use.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Reset(
        FrontierType frontier_type,     // The frontier type (i.e., edge/vertex/mixed)
        double queue_sizing = 2.0)            // Size scaling factor for work queue allocation
        {
            util::cpu_mt::PrintMessage("ProblemBase Reset() begin.");
            cudaError_t retval = cudaSuccess;

            for (int gpu = 0; gpu < num_gpus; ++gpu) {
                if (retval = graph_slices[gpu]->Reset(frontier_type,queue_sizing)) break;
            }
            
            util::cpu_mt::PrintMessage("ProblemBase Reset() end.");
            return retval;
        }
};

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
