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
    bool        _USE_DOUBLE_BUFFER>

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
        int             index;                           // Slice index
        VertexId        nodes;                           // Number of nodes in slice
        SizeT           edges;                           // Number of edges in slice
        cudaStream_t    stream;                          // CUDA stream to use for processing the slice

        Csr<VertexId, Value, SizeT   > *graph          ; // Pointer to CSR format subgraph
        util::Array1D<SizeT, SizeT   > row_offsets     ; // CSR format row offset on device memory
        util::Array1D<SizeT, VertexId> column_indices  ; // CSR format column indices on device memory
        util::Array1D<SizeT, SizeT   > column_offsets  ; // CSR format column offset on device memory
        util::Array1D<SizeT, VertexId> row_indices     ; // CSR format row indices on device memory
        util::Array1D<SizeT, int     > partition_table ; // Partition number for vertexes, local is always 0
        util::Array1D<SizeT, VertexId> convertion_table; // Vertex number of vertexes in their hosting partition
        util::Array1D<SizeT, VertexId> original_vertex ;
        util::Array1D<SizeT, SizeT   > in_offset       ;
        util::Array1D<SizeT, SizeT   > out_offset      ;

        //Frontier queues. Used to track working frontier.
        util::DoubleBuffer<SizeT, VertexId, VertexId>  frontier_queues;
        SizeT                                          frontier_elements[2];

        /**
         * @brief GraphSlice Constructor
         *
         * @param[in] index GPU index, reserved for multi-GPU use in future.
         * @param[in] stream CUDA Stream we use to allocate storage for this graph slice.
         */
        GraphSlice(int index, cudaStream_t stream) :
            index(index),
            graph(NULL),
            //d_row_offsets(NULL),
            //d_column_indices(NULL),
            //d_column_offsets(NULL),
            //d_row_indices(NULL),
            nodes(0),
            edges(0),
            stream(stream)
        {
            util::cpu_mt::PrintMessage("GraphSlice() begin.");
            row_offsets     .SetName("row_offsets"     );
            column_indices  .SetName("column_indices"  );
            column_offsets  .SetName("column_offsets"  );
            row_indices     .SetName("row_indices"     );
            partition_table .SetName("partition_table" );
            convertion_table.SetName("convertion_table");
            original_vertex .SetName("original_vertex" );
            in_offset       .SetName("in_offset"       );  
            out_offset      .SetName("out_offset"      );  

            // Initialize double buffer frontier queue lengths
            for (int i = 0; i < 2; ++i)
            {
                frontier_elements[i] = 0;
            }
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
            row_offsets     .Release();
            column_indices  .Release();
            column_offsets  .Release();
            row_indices     .Release();
            partition_table .Release();
            convertion_table.Release();
            original_vertex .Release();
            in_offset       .Release();
            out_offset      .Release();

            for (int i = 0; i < 2; ++i) {
                frontier_queues.keys  [i].Release();
                frontier_queues.values[i].Release();
            }

            // Destroy stream
            if (stream) {
                util::GRError(cudaStreamDestroy(stream), "GpuSlice cudaStreamDestroy failed", __FILE__, __LINE__);
            }
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
            SizeT*                     in_offset,
            SizeT*                     out_offset)
        {
            util::cpu_mt::PrintMessage("GraphSlice Init() begin.");
            cudaError_t retval     = cudaSuccess;
            this->graph            = graph;
            nodes                  = graph->nodes;
            edges                  = graph->edges;
            this->partition_table .SetPointer(partition_table   , nodes     );
            this->convertion_table.SetPointer(convertion_table  , nodes     );
            this->original_vertex .SetPointer(original_vertex   , nodes     );
            this->in_offset       .SetPointer(in_offset         , num_gpus+1);
            this->out_offset      .SetPointer(out_offset        , num_gpus+1);
            this->row_offsets     .SetPointer(graph->row_offsets, nodes+1   );
            this->column_indices  .SetPointer(graph->column_indices, edges  );

            do {
                if (retval = util::GRError(cudaSetDevice(index), "GpuSlice cudaSetDevice failed", __FILE__, __LINE__)) break;
                // Allocate and initialize row_offsets
                if (retval = this->row_offsets.Allocate(nodes+1      ,util::DEVICE)) break;
                if (retval = this->row_offsets.Move    (util::HOST   ,util::DEVICE)) break;
                
                // Allocate and initialize column_indices
                if (retval = this->column_indices.Allocate(edges     ,util::DEVICE)) break;
                if (retval = this->column_indices.Move    (util::HOST,util::DEVICE)) break;
              
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
                    if (retval = this->partition_table.Allocate(nodes     ,util::DEVICE)) break;
                    if (retval = this->partition_table.Move    (util::HOST,util::DEVICE)) break;
                    
                    // Allocate and initalize convertion_table
                    if (retval = this->convertion_table.Allocate(nodes     ,util::DEVICE)) break;
                    if (retval = this->convertion_table.Move    (util::HOST,util::DEVICE)) break;

                    // Allocate and initalize original_vertex
                    if (retval = this->original_vertex .Allocate(nodes     ,util::DEVICE)) break;
                    if (retval = this->original_vertex .Move    (util::HOST,util::DEVICE)) break;
                    
                    // Allocate and initalize in_offset
                    if (retval = this->in_offset.Allocate(num_gpus+1,util::DEVICE)) break;
                    if (retval = this->in_offset.Move    (util::HOST,util::DEVICE)) break;
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

            switch (frontier_type) {
                case VERTEX_FRONTIERS :
                    // O(n) ping-pong global vertex frontiers
                    new_frontier_elements[0] = double(nodes) * queue_sizing;
                    new_frontier_elements[1] = new_frontier_elements[0];
                    break;

                case EDGE_FRONTIERS :
                    // O(m) ping-pong global edge frontiers
                    new_frontier_elements[0] = double(edges) * queue_sizing;
                    new_frontier_elements[1] = new_frontier_elements[0];
                    break;

                case MIXED_FRONTIERS :
                    // O(n) global vertex frontier, O(m) global edge frontier
                    new_frontier_elements[0] = double(nodes) * queue_sizing;
                    new_frontier_elements[1] = double(edges) * queue_sizing;
                    break;
             }   

            // Iterate through global frontier queue setups
            for (int i = 0; i < 2; i++) {

                // Allocate frontier queue if not big enough
                if (frontier_elements[i] < new_frontier_elements[i]) {

                    // Free if previously allocated
                    if (retval = frontier_queues.keys[i].Release()) return retval;

                    // Free if previously allocated
                    if (_USE_DOUBLE_BUFFER) {
                        if (retval = frontier_queues.values[i].Release()) return retval;
                    }

                    frontier_elements[i] = new_frontier_elements[i];

                    if (retval = frontier_queues.keys[i].Allocate(frontier_elements[i],util::DEVICE)) return retval;
                    if (_USE_DOUBLE_BUFFER) {
                        if (retval = frontier_queues.values[i].Allocate(frontier_elements[i],util::DEVICE)) return retval;
                    }
                } //end if
            } // end for i<2

            util::cpu_mt::PrintMessage("GraphSlice Reset() end.");
            return retval;
        } // end Reset(...)

    }; // end GraphSlice

    // Members
    int                 num_gpus           ; // Number of GPUs to be sliced over
    int                 *gpu_idx           ; // GPU indices 
    SizeT               nodes              ; // Size of the graph
    SizeT               edges              ;
    GraphSlice          **graph_slices     ; // Set of graph slices (one for each GPU)
    Csr<VertexId,Value,SizeT> *sub_graphs  ; // Subgraphs for multi-gpu implementation
    PartitionerBase<VertexId,SizeT,Value>
                        *partitioner       ; // Partitioner
    int                 **partition_tables ; // Multi-gpu partition table and convertion table
    VertexId            **convertion_tables;
    VertexId            **original_vertexes;
    SizeT               **in_offsets       ; // Offsets for data movement between GPUs
    SizeT               **out_offsets      ;

    // Methods
    
    /**
     * @brief ProblemBase default constructor
     */
    ProblemBase() :
        num_gpus         (0   ),
        gpu_idx          (NULL),
        nodes            (0   ),
        edges            (0   ),
        graph_slices     (NULL),
        sub_graphs       (NULL),
        partitioner      (NULL),
        partition_tables (NULL),
        convertion_tables(NULL),
        original_vertexes(NULL),
        in_offsets       (NULL),
        out_offsets      (NULL)
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
            if (num_gpus > 1)
            {
                free (partition_tables    [i+1]); partition_tables [i+1] = NULL;
                free (convertion_tables   [i+1]); convertion_tables[i+1] = NULL;
                free (original_vertexes   [i  ]); original_vertexes[i  ] = NULL;
                delete[] out_offsets      [i  ] ; out_offsets      [i  ] = NULL;
                delete[] in_offsets       [i  ] ; in_offsets       [i  ] = NULL;
            }
        }
        if (num_gpus > 1)
        {
            delete[] partition_tables [0];  partition_tables [0] = NULL;
            delete[] convertion_tables[0];  convertion_tables[0] = NULL;
            delete[] partition_tables;      partition_tables     = NULL;
            delete[] convertion_tables;     convertion_tables    = NULL;
            delete[] original_vertexes;     original_vertexes    = NULL;
            delete[] out_offsets;           out_offsets          = NULL;
            delete[] in_offsets;            in_offsets           = NULL;
            delete   partitioner;           partitioner          = NULL;
            delete[] sub_graphs;            sub_graphs           = NULL;
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
        std::string partition_method  = "random")
    {
        util::cpu_mt::PrintMessage("ProblemBase Init() begin.");
        cudaError_t retval      = cudaSuccess;
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
                    partitioner=new rp::RandomPartitioner<VertexId, SizeT, Value>(*graph,num_gpus);
                else if (partition_method=="metis")
                    partitioner=new metisp::MetisPartitioner<VertexId, SizeT, Value>(*graph,num_gpus);
                else util::GRError("partition_method invalid", __FILE__,__LINE__);
                printf("partition begin.\n");fflush(stdout);
                cpu_timer.Start();
                retval = partitioner->Partition(
                    sub_graphs,
                    partition_tables,
                    convertion_tables,
                    original_vertexes,
                    in_offsets,
                    out_offsets);
                cpu_timer.Stop();
                printf("partition end. (%f ms)\n", cpu_timer.ElapsedMillis());fflush(stdout);
                //util::cpu_mt::PrintCPUArray<SizeT,int>("partition0",partition_tables[0],graph->nodes);
                //util::cpu_mt::PrintCPUArray<SizeT,VertexId>("convertion0",convertion_tables[0],graph->nodes);
                //util::cpu_mt::PrintCPUArray<SizeT,Value>("edge_value",graph->edge_values,graph->edges);
                //for (int gpu=0;gpu<num_gpus;gpu++)
                //{
                //    printf("%d\n",gpu);
                //    util::cpu_mt::PrintCPUArray<SizeT,int>("partition",partition_tables[gpu+1],sub_graphs[gpu].nodes);
                //    util::cpu_mt::PrintCPUArray<SizeT,VertexId>("convertion",convertion_tables[gpu+1],sub_graphs[gpu].nodes);
                //}
                if (retval) break;
            } else {
                sub_graphs=graph;
            }

            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                graph_slices[gpu] = new GraphSlice(this->gpu_idx[gpu], 0);
                if (num_gpus > 1)
                {
                    retval = graph_slices[gpu]->Init(
                        stream_from_host,
                        num_gpus,
                        &(sub_graphs[gpu]),
                        NULL,
                        partition_tables [gpu+1],
                        convertion_tables[gpu+1],
                        original_vertexes[gpu],
                        in_offsets[gpu],
                        out_offsets[gpu]);
                } else retval = graph_slices[gpu]->Init(
                        stream_from_host,
                        num_gpus,
                        &(sub_graphs[gpu]),
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
