// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * andersen_problem.cuh
 *
 * @brief GPU Storage management Structure for Andersen Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace andersen {

/**
 * @brief Connected Component Problem structure stores device-side vectors for doing connected component computing on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of float or double to use for computing BC value.
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to use double buffer
 */
template <
    typename    VertexId,
    typename    SizeT,
    typename    Value,
    bool        _USE_DOUBLE_BUFFER>
struct AndersenProblem : ProblemBase<VertexId, SizeT, Value,
    true, // _MARK_PREDECESSORS
    false, // _ENABLE_IDEMPOTENCE,
    _USE_DOUBLE_BUFFER, 
    false, // _EnABLE_BACKWARD
    false, // _KEEP_ORDER
    true>  // _KEEP_NODE_NUM
{
    //Helper structures

    /** 
     * @brief Data slice structure which contains Andersen problem specific data.
     */
    struct DataSlice : DataSliceBase<SizeT, VertexId, Value>
    {
        // device storage arrays
        GraphSlice<SizeT, VertexId, Value> *    pts_graphslice;
        //GraphSlice<SizeT, VertexId, Value> * ptsInv_graphslice;
        GraphSlice<SizeT, VertexId, Value> *ptsIadd_graphslice;
        GraphSlice<SizeT, VertexId, Value> *pts_add_graphslice;
        GraphSlice<SizeT, VertexId, Value> *pts_inc_graphslice;
        GraphSlice<SizeT, VertexId, Value> *copyInv_graphslice;
        GraphSlice<SizeT, VertexId, Value> *copyIad_graphslice;
        GraphSlice<SizeT, VertexId, Value> *copyIin_graphslice;
        GraphSlice<SizeT, VertexId, Value> *loadInv_graphslice;
        GraphSlice<SizeT, VertexId, Value> *  store_graphslice;
        GraphSlice<SizeT, VertexId, Value> * gepInv_graphslice;
        util::Array1D<SizeT, int     > partition_table;
        util::Array1D<SizeT, SizeT   > gepInv_offset;
        util::Array1D<SizeT, VertexId> pts_hash;
        util::Array1D<SizeT, VertexId> copyInv_hash;
        util::Array1D<SizeT, int     > t_marker;
        util::Array1D<SizeT, SizeT   > t_length;

        util::Array1D<SizeT, SizeT   > r_offsets;
        util::Array1D<SizeT, VertexId> r_indices;
        //util::Array1D<SizeT, SizeT   > r_offsets2;
        SizeT*                         r_offsets2;
        util::Array1D<SizeT, SizeT   > s_offsets;
        util::Array1D<SizeT, VertexId> s_indices;
        util::Array1D<SizeT, VertexId> t_indices;
        util::Array1D<SizeT, SizeT   > t_offsets;
        util::Array1D<SizeT, VertexId> t_hash;
        bool                           t_conflict;

        util::Array1D<SizeT, VertexId> labels;
        util::Array1D<SizeT, VertexId> preds;
        util::Array1D<SizeT, VertexId> temp_preds;

        util::Array1D<SizeT, DataSlice> *data_slice;
        SizeT                          h_stride;
        SizeT                          h_size;
        bool                           to_continue;

        DataSlice() :
                pts_graphslice(NULL),
            // ptsInv_graphslice(NULL),
            ptsIadd_graphslice(NULL),
            pts_add_graphslice(NULL),
            pts_inc_graphslice(NULL),
            copyInv_graphslice(NULL),
            copyIad_graphslice(NULL),
            copyIin_graphslice(NULL),
            loadInv_graphslice(NULL),
            store_graphslice  (NULL),
            gepInv_graphslice (NULL),
            data_slice        (NULL)
        {
            partition_table       .SetName("partition_table"       );
            gepInv_offset         .SetName("gepInv_offset"         );
            pts_hash              .SetName("pts_hash"              );
            copyInv_hash          .SetName("copyInv_hash"          );
            t_marker              .SetName("t_marker"              );
            t_length              .SetName("t_length"              );
            labels                .SetName("labels"                );
            preds                 .SetName("preds"                 );
            temp_preds            .SetName("temp_preds"            );
            to_continue            = true;
            r_offsets2             = NULL;
        }

        ~DataSlice()
        {
            if (util::SetDevice(this->gpu_idx)) return;
            partition_table       .Release();
            gepInv_offset         .Release();
            pts_hash              .Release();
            copyInv_hash          .Release();
            t_marker              .Release();
            t_length              .Release();
            labels                .Release();
            preds                 .Release();
            temp_preds            .Release();
        }

        cudaError_t Init(
            int   num_gpus,
            int   gpu_idx,
            int   num_vertex_associate,
            int   num_value__associate,
            //Csr<VertexId, Value, SizeT> *graph,
            Csr<VertexId, Value, SizeT> *pts_graph,
            Csr<VertexId, Value, SizeT> *copyInv_graph,
            Csr<VertexId, Value, SizeT> *loadInv_graph,
            Csr<VertexId, Value, SizeT> *store_graph,
            Csr<VertexId, Value, SizeT> *gepInv_graph,
            SizeT *_gepInv_offset,
            SizeT *num_in_nodes,
            SizeT *num_out_nodes,
            VertexId *original_vertex,
            float queue_sizing = 2.0,
            float in_sizing    = 1.0)
        {   
            cudaError_t retval = cudaSuccess;
            //SizeT       nodes  = pts_graph->nodes;
            //SizeT       edges  = pts_graph->edges;

            if (retval = DataSliceBase<SizeT, VertexId, Value>::Init(
                num_gpus,
                gpu_idx,
                num_vertex_associate,
                num_value__associate,
                pts_graph,
                num_in_nodes,
                num_out_nodes,
                in_sizing)) return retval;

            if (retval = partition_table.Allocate(pts_graph    ->nodes  , util::DEVICE)) return retval;
            if (retval = gepInv_offset  .Allocate(gepInv_graph ->edges  , util::DEVICE)) return retval;
            if (retval = pts_hash       .Allocate(pts_graph    ->edges*2, util::DEVICE)) return retval;
            if (retval = copyInv_hash   .Allocate(copyInv_graph->edges*2, util::DEVICE)) return retval;
            if (retval = t_length       .Allocate(pts_graph    ->nodes+1, util::HOST | util::DEVICE)) return retval;
            if (retval = t_marker       .Allocate(max(pts_graph->edges, copyInv_graph->edges)*2, util::DEVICE)) return retval;
            pts_graphslice     = new GraphSlice<SizeT, VertexId, Value>(gpu_idx);
            ptsIadd_graphslice = new GraphSlice<SizeT, VertexId, Value>(gpu_idx);
            pts_add_graphslice = new GraphSlice<SizeT, VertexId, Value>(gpu_idx);
            pts_inc_graphslice = new GraphSlice<SizeT, VertexId, Value>(gpu_idx);
            copyInv_graphslice = new GraphSlice<SizeT, VertexId, Value>(gpu_idx);
            copyIad_graphslice = new GraphSlice<SizeT, VertexId, Value>(gpu_idx);
            copyIin_graphslice = new GraphSlice<SizeT, VertexId, Value>(gpu_idx);
            loadInv_graphslice = new GraphSlice<SizeT, VertexId, Value>(gpu_idx);
            store_graphslice   = new GraphSlice<SizeT, VertexId, Value>(gpu_idx);
            gepInv_graphslice  = new GraphSlice<SizeT, VertexId, Value>(gpu_idx);
            if (retval = pts_graphslice    ->Init(false, num_gpus, pts_graph    , NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)) return retval;
            if (retval = copyInv_graphslice->Init(false, num_gpus, copyInv_graph, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)) return retval;
            if (retval = loadInv_graphslice->Init(false, num_gpus, loadInv_graph, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)) return retval;
            if (retval = store_graphslice  ->Init(false, num_gpus, store_graph  , NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)) return retval;
            if (retval = gepInv_graphslice ->Init(false, num_gpus, gepInv_graph , NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)) return retval;
            
            if (retval = ptsIadd_graphslice->Init(false, num_gpus, pts_graph    , NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)) return retval;
            if (retval = pts_add_graphslice->Init(false, num_gpus, pts_graph    , NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)) return retval;
            if (retval = pts_inc_graphslice->Init(false, num_gpus, pts_graph    , NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)) return retval;
            if (retval = copyIad_graphslice->Init(false, num_gpus, copyInv_graph, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)) return retval;
            if (retval = copyIin_graphslice->Init(false, num_gpus, copyInv_graph, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)) return retval;
            ptsIadd_graphslice->edges = 0;
            pts_add_graphslice->edges = 0;
            pts_inc_graphslice->edges = 0;
            copyIad_graphslice->edges = 0;
            copyIin_graphslice->edges = 0;
            
            return retval;
        }
    };

    // Set of data slices (one for each GPU)
    util::Array1D<SizeT, DataSlice> *data_slices;
   
    // Methods

    /**
     * @brief AndersenProblem default constructor
     */
    AndersenProblem()
    {
        data_slices    = NULL;
    }

    /**
     * @brief AndersenProblem default destructor
     */
    ~AndersenProblem()
    {
        if (data_slices == NULL) return;
        for (int i = 0; i < this->num_gpus; ++i)
        {
            util::SetDevice(this->gpu_idx[i]);
            data_slices[i].Release();
        }
        delete[] data_slices;data_slices=NULL;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Copy result component ids computed on the GPU back to a host-side vector.
     *
     * @param[out] h_component_ids host-side vector to store computed component ids.
     *
     *\return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Extract(VertexId *h_component_ids)
    {
        cudaError_t retval = cudaSuccess;

        do {
            if (this->num_gpus == 1) {
                if (retval = util::SetDevice(this->gpu_idx[0])) return retval;
            } else {
                for (int gpu=0; gpu< this->num_gpus; gpu++)
                {
                    if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
                }
                
            } //end if
        } while(0);

        return retval;
    }

    /**
     * @brief AndersenProblem initialization
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on. @see Csr
     * @param[in] _num_gpus Number of the GPUs used.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
            bool          stream_from_host,       // Only meaningful for single-GPU
            Csr<VertexId, Value, SizeT> 
                         *pts_graph,
            Csr<VertexId, Value, SizeT>
                         *copyInv_graph,
            Csr<VertexId, Value, SizeT>
                         *loadInv_graph,
            Csr<VertexId, Value, SizeT>
                         *store_graph,
            Csr<VertexId, Value, SizeT>
                         *gepInv_graph,
            SizeT        *gepInv_offset,
            int           num_gpus         = 1,
            int          *gpu_idx          = NULL,
            std::string   partition_method = "random",
            cudaStream_t *streams          = NULL,
            float         queue_sizing     = 2.0f,
            float         in_sizing        = 1.0f,
            float         partition_factor = -1.0f,
            int           partition_seed   = -1)
    {
        ProblemBase<VertexId, SizeT, Value, true, false, _USE_DOUBLE_BUFFER, false, false, true>::Init(
            stream_from_host,
            pts_graph,
            NULL,
            num_gpus,
            gpu_idx,
            partition_method,
            queue_sizing,
            partition_factor,
            partition_seed);

        // No data in DataSlice needs to be copied from host
         
        /**
         * Allocate output labels/preds
         */
        cudaError_t retval = cudaSuccess;
        data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];
        
        do {
            for (int gpu=0; gpu<this->num_gpus; gpu++)
            {
                data_slices[gpu].SetName("data_slices[]");
                if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
                if (retval = data_slices[gpu].Allocate(1, util::DEVICE | util::HOST)) return retval;
                DataSlice* data_slice_ = data_slices[gpu].GetPointer(util::HOST);
                data_slice_->streams.SetPointer(&streams[gpu*num_gpus*2], num_gpus*2);

                data_slice_->data_slice = &data_slices[gpu];
                if (retval = data_slice_->Init(
                    this->num_gpus,
                    this->gpu_idx[gpu],
                    this->num_gpus>1? 1:0,
                    0,
                    pts_graph,
                    copyInv_graph,
                    loadInv_graph,
                    store_graph,
                    gepInv_graph,
                    gepInv_offset,
                    this->num_gpus>1? this->graph_slices[gpu]->in_counter .GetPointer(util::HOST) : NULL,
                    this->num_gpus>1? this->graph_slices[gpu]->out_counter.GetPointer(util::HOST) : NULL,
                    this->num_gpus>1? this->graph_slices[gpu]->original_vertex.GetPointer(util::HOST) : NULL,
                    queue_sizing,
                    in_sizing)) return retval;
            }
        } while (0);

        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for Andersen problem type. Must be called prior to each Andersen run.
     *
     *  @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     * 
     *  \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Reset(
        FrontierType frontier_type,   // The frontier type (i.e., edge/vertex/mixed)

        double       queue_sizing)
    {
        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            //SizeT nodes = this->sub_graphs[gpu].nodes;
            //SizeT edges = this->sub_graphs[gpu].edges;
            DataSlice *data_slice_ = data_slices[gpu].GetPointer(util::HOST);
            // Set device
            if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
            if (retval = data_slices[gpu]->Reset(frontier_type, this->graph_slices[gpu], queue_sizing, _USE_DOUBLE_BUFFER)) return retval;

        }
       
        return retval;
    }

    /** @} */
};

} //namespace andersen
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
