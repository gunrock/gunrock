// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * rw_problem.cuh
 *
 * @brief GPU Storage management Structure for RW Problem Data
 */

#pragma once

#include <limits>
#include <stdio.h>
#include <stdlib.h>

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/array_utils.cuh>

#define ELEMS_PER_THREAD 4
#define THREAD_BLOCK 128


enum MODE{
    RAW    = 0,
    DEVICE = 1,
    BLOCK  = 2,
};

namespace gunrock {
namespace app {
namespace rw {

/**
 * @brief RW Problem structure stores device-side vectors for doing RW computing on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of value used for computed values.
 */
template <
    typename    VertexId,
    typename    SizeT,
    typename    Value>
struct RWProblem : ProblemBase<VertexId, SizeT, Value,
    true,//_MARK_PREDECESSORS
    false> //ENABLE_IDEMPOTENCE
{
    static const bool MARK_PREDECESSORS     = true;
    static const bool ENABLE_IDEMPOTENCE    = false;

    static const int  MAX_NUM_VERTEX_ASSOCIATES = 0;
    static const int  MAX_NUM_VALUE__ASSOCIATES = 1;
    typedef ProblemBase   <VertexId, SizeT, Value,
        MARK_PREDECESSORS, ENABLE_IDEMPOTENCE> BaseProblem;
    typedef DataSliceBase <VertexId, SizeT, Value,
        MAX_NUM_VERTEX_ASSOCIATES, MAX_NUM_VALUE__ASSOCIATES> BaseDataSlice;
    typedef unsigned char MaskT;

    //Helper structures

    /**
     * @brief Data slice structure which contains RW problem specific data.
     */
    struct DataSlice : BaseDataSlice
    {
        // device storage arrays
        // util::Array1D<SizeT, Value>    user_specific_array;     /**< users can add arbitrary device arrays here. */

        util::Array1D<SizeT, SizeT>       node_id;    /**< Used for the mapping between original vertex IDs and local vertex IDs on multi-GPUs */
        util::Array1D<SizeT, Value>       d_row_offsets;
        util::Array1D<SizeT, Value>       d_col_indices;
        util::Array1D<SizeT, Value>       num_neighbor;  
             /**< Used for randomly choosing next neighbor node */
        util::Array1D<SizeT, Value>       paths;              /**< Used for store output paths */
        util::Array1D<SizeT, float>       d_rand;
        util::Array1D<SizeT, Value>       trailing_slice; //used in sorted random walk to deal with 
        SizeT                             size; //frontier size
        SizeT                             trailing;
        SizeT                             grid_size;
        //bool                              block; // 0, raw, 1, device sort, 2, block sort, 3: pre-process
        //bool                              device;





        //util::Array1D<SizeT, Value>       path_length;

        /*
         * @brief Default constructor
         */
        DataSlice() : BaseDataSlice()
        {
            node_id       .SetName("node_id" );
            num_neighbor  .SetName("num_neighbor" );
            d_row_offsets .SetName("d_row_offsets");
            d_col_indices .SetName("d_col_indices"); 
            d_rand        .SetName("d_rand");
            paths         .SetName("paths" );
            trailing_slice.SetName("padding_slice" );
            size          = 1;
            trailing      = 0;
            grid_size     = 1;
            //path_length.SetName("path_length");
        }

        /*
         * @brief Default destructor
         */
        ~DataSlice()
        {
            if (util::SetDevice(this->gpu_idx)) return ;
            node_id       .Release();
            d_row_offsets .Release();
            d_col_indices .Release();
            d_rand        .Release();
            num_neighbor  .Release();
            paths         .Release();
            trailing_slice .Release();

        }

        /**
         * @brief initialization function of dataslice struct.
         * Define and allocate mem for all the needed data/data array for this problem
         * Define specific data needed for this problem/primitive here
         *
         * @param[in] walk_length Number of the rw walks.
         * @param[in] block Block sort rw or not.
         * @param[in] device Device sort rw or not.
         * @param[in] num_gpus Number of the GPUs used.
         * @param[in] gpu_idx GPU index used for testing.
         * @param[in] use_double_buffer Whether to use double buffer.
         * @param[in] graph Pointer to the graph we process on.
         * @param[in] graph_slice Pointer to the GraphSlice object.
         * @param[in] num_in_nodes
         * @param[in] num_out_nodes
         * @param[in] queue_sizing Maximum queue sizing factor.
         * @param[in] in_sizing
         * @param[in] skip_makeout_selection
         * @param[in] keep_node_num
         *
         * \return cudaError_t object Indicates the success of all CUDA calls.
         */
        cudaError_t Init(
            int   walk_length,
            int   mode,
            int   num_gpus,
            int   gpu_idx,
            bool  use_double_buffer,
            Csr<VertexId, SizeT, Value> *graph,
            SizeT *num_in_nodes,
            SizeT *num_out_nodes,
            float queue_sizing = 2.0,
            float in_sizing    = 1.0)
        {
            cudaError_t retval  = cudaSuccess;

            if (retval = BaseDataSlice::Init(
                num_gpus,
                gpu_idx,
                use_double_buffer,
                graph,
                num_in_nodes,
                num_out_nodes,
                in_sizing)) return retval;


        
        //enum { GRID = DSIZE / ELEMS_PER_THREAD };

        //int frontierSize=grid*ITEM_PER_THERAD*tb;


        //ITEM_PER_THERAD = 4
    //block size-> 128
    //if  data size  = 70
    // 70/4 = 17, -> 17 threads/block
    // 70/128 = 0-> 1 block(gridsize) 
   // only sorting 68 element


            //this-> walk_length = walk_length;
            if(mode == BLOCK){
                this->grid_size = graph->nodes/(THREAD_BLOCK*ELEMS_PER_THREAD);
                this->size = ELEMS_PER_THREAD*THREAD_BLOCK*grid_size;
                this->trailing = graph->nodes - this->size;
                if (retval = this->node_id.Allocate(this->size, util::DEVICE)) return retval;
                if (retval = this->paths.Allocate(this->size*walk_length, util::DEVICE)) return retval;
                if (retval = this->trailing_slice.Allocate(this->trailing*walk_length, util::DEVICE)) return retval;

            }else if(mode == DEVICE){
              this->size = graph->nodes;
                if (retval = this->node_id.Allocate(graph->nodes, util::DEVICE)) return retval;
                if (retval = this->paths.Allocate(graph->nodes*walk_length, util::DEVICE)) return retval;


            }else{
                this->size = graph->nodes;
                if (retval = this->paths.Allocate(graph->nodes*walk_length, util::DEVICE)) return retval;
            }

            // labels is a required array that defined in BaseProblem class.
            //if (retval = this->node_id.Allocate(graph->nodes, util::DEVICE)) return retval;

            //problem.walk_leg

            if (retval = this->d_row_offsets.Allocate(graph->nodes+1, util::DEVICE)) return retval;

            if (retval = this->d_rand.Allocate(graph->nodes, util::DEVICE)) return retval;

            if (retval = this->d_col_indices.Allocate(graph->edges, util::DEVICE)) return retval;

 
            if (retval = this->num_neighbor.Allocate(graph->nodes, util::HOST | util::DEVICE)) return retval;

            // calculate number of neighbor node from row_offsets of csr graph

            for (VertexId node=0; node < graph->nodes; node++)
            {
                num_neighbor[node] = graph->row_offsets[node+1] - graph->row_offsets[node];

            }
            if (retval = num_neighbor.Move(util::HOST, util::DEVICE)) return retval;
            if (retval = num_neighbor.Release(util::HOST)) return retval;


            return retval;
        } // Init

        /**
         * @brief Reset problem function of datas_slice struct. Must be called prior to each run.
         * Will be call within problem->reset()
         *
         * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed).
         * @param[in] graph_slice Pointer to the graph slice we process on.
         * @param[in] queue_sizing Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively).
         * @param[in] queue_sizing1 Size scaling factor for work queue allocation.
         *
         * \return cudaError_t object Indicates the success of all CUDA calls.
         */
        cudaError_t Reset(
            int                                 walk_length,
            int                                 mode,      
            FrontierType                        frontier_type,
            GraphSlice<VertexId, SizeT, Value>  *graph_slice,
            double                              queue_sizing = 2.0,
            bool                                use_double_buffer  = false,
            double                              queue_sizing1 = -1.0,
            bool                                skip_scanned_edges = false)
        {
            cudaError_t retval = cudaSuccess;


            //printf("graph slice edges: %d, queue_sizing:%f\n", graph_slice->edges, queue_sizing);

            if (retval = BaseDataSlice::Reset(
                frontier_type,
                graph_slice,
                queue_sizing,
                use_double_buffer,
                queue_sizing1,
                skip_scanned_edges))
                return retval;            

            SizeT node = graph_slice ->nodes;
            SizeT edge = graph_slice ->edges;


            if (d_row_offsets.GetPointer(util::DEVICE) == NULL){
                printf("d_row_offsets pointer is null.\n");
                if (retval = d_row_offsets.Allocate(node+1, util::DEVICE))
                    return retval;
            }

            if (d_col_indices.GetPointer(util::DEVICE) == NULL){
                printf("d_col_indices pointer is null.\n");
                if (retval = d_row_offsets.Allocate(edge, util::DEVICE))
                    return retval;
            }


            if (d_rand.GetPointer(util::DEVICE) == NULL){
                printf("d_rand pointer is null.\n");
                if (retval = d_rand.Allocate(node, util::DEVICE))
                    return retval;
            }


            if (num_neighbor.GetPointer(util::DEVICE) == NULL)
                if (retval = num_neighbor.Allocate(node, util::DEVICE))
                    return retval;


            d_row_offsets.SetPointer((SizeT*)graph_slice -> row_offsets.GetPointer(util::DEVICE), 
                                    node+1, util::DEVICE);

            d_col_indices.SetPointer((VertexId*)graph_slice -> column_indices.GetPointer(util::DEVICE), 
                                    edge, util::DEVICE);

            /*
            util::MemsetIdxKernel<<<128, 128>>>(
                node_id.GetPointer(util::DEVICE), this->size);

            util::MemsetIdxKernel<<<128, 128>>>(
                paths.GetPointer(util::DEVICE), this->size);*/

            if(mode == BLOCK){

                if (node_id.GetPointer(util::DEVICE) == NULL)
                  if (retval = node_id.Allocate(this->size*walk_length, util::DEVICE))
                      return retval;

                if (paths.GetPointer(util::DEVICE) == NULL)
                  if (retval = paths.Allocate(this->size*walk_length, util::DEVICE))
                      return retval;
                
                if (trailing_slice.GetPointer(util::DEVICE) == NULL)
                  if (retval = trailing_slice.Allocate(this->trailing*walk_length, util::DEVICE))
                      return retval;

                util::MemsetIdxKernel<<<128, 128>>>(
                  node_id.GetPointer(util::DEVICE), this->size);

                util::MemsetIdxKernel<<<128, 128>>>(
                    paths.GetPointer(util::DEVICE), this->size);

                util::MemsetIdxKernel<<<128, 128>>>(
                    trailing_slice.GetPointer(util::DEVICE), this->trailing);

                //offset node_id in the trailing slice by "size" number
                util::MemsetAddKernel<<<128, 128>>>(
                    trailing_slice.GetPointer(util::DEVICE), this->size,this->trailing);

                
            }else if(mode == DEVICE){
              if (node_id.GetPointer(util::DEVICE) == NULL)
                  if (retval = node_id.Allocate(node, util::DEVICE))
                      return retval;

              if (paths.GetPointer(util::DEVICE) == NULL)
                  if (retval = paths.Allocate(node*walk_length, util::DEVICE))
                      return retval;

              util::MemsetIdxKernel<<<128, 128>>>(
                  node_id.GetPointer(util::DEVICE), node);

              util::MemsetIdxKernel<<<128, 128>>>(
                  paths.GetPointer(util::DEVICE), node);

            }else{
              if (paths.GetPointer(util::DEVICE) == NULL)
                  if (retval = paths.Allocate(node*walk_length, util::DEVICE))
                      return retval;
                
                util::MemsetIdxKernel<<<128, 128>>>(
                  paths.GetPointer(util::DEVICE), node);
            }


	    



	    
            /*
            util::MemsetIdxKernel<<<128, 128>>>(
                num_neighbor.GetPointer(util::DEVICE), node);*/

            return retval;
        }
    }; // DataSlice



    // Members
    // Set of data slices (one for each GPU)
    util::Array1D<SizeT, DataSlice>          *data_slices;
    SizeT                                    walk_length;
    SizeT                                    mode;
    // Methods

    /**
     * @brief RWProblem default constructor
     */

    RWProblem(SizeT _walk_length, SizeT _mode) : BaseProblem(
        false, // use_double_buffer
        false, // enable_backward
        false, // keep_order
        false  // keep_node_num
        ),  // unified_receive
        data_slices(NULL),
        walk_length(_walk_length),
        mode       (_mode)
    {
        //this->walk_length = length;
        //this->block = block;
    }

    /**
     * @brief RWProblem default destructor
     */
    ~RWProblem()
    {
        
        if (data_slices == NULL) return;
        
        for (int i = 0; i < this -> num_gpus; ++i)
        {
            //************ not sure why this SetDevice cause seg-fault
            //util::SetDevice(this -> gpu_idx[i]);

            data_slices[i].Release();
        }

        delete[] data_slices; 
        data_slices = NULL;
    }


    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Copy result distancess computed on the GPU back to host-side vectors.
     *
     *\return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Extract(
      SizeT    *h_paths /*，

      bool      block=false,
      SizeT    *h_trailing=NULL*/)
    {
      cudaError_t retval = cudaSuccess;
       if (this->num_gpus == 1)
        {
            // Set device
            if (retval = util::SetDevice(this->gpu_idx[0])) return retval;

            data_slices[0]->paths.SetPointer(h_paths);
            if (retval = data_slices[0]->paths.Move(util::DEVICE,util::HOST)) return retval;
            

            if(mode == BLOCK){
              if (retval = data_slices[0]->trailing_slice.Move(util::DEVICE,util::HOST)) return retval;
            }

        
         
          
        
/*      if (this -> num_gpus == 1)
      {
          // Set device
          int gpu = 0;
          DataSlice *data_slice = data_slices[gpu].GetPointer(util::HOST);
          if (retval = util::SetDevice( this -> gpu_idx[gpu]))
              return retval;

          data_slice -> paths.SetPointer(h_paths);
          if (retval = data_slice -> paths.Move(util::DEVICE, util::HOST, num_nodes*this->walk_length))
              return retval;
  */    
      } else
      {
          // TODO: multi-GPU extract result
      }
      return retval;
    }


    /**


    /**
     * @brief initialization function of Problem Struct.
     *data_slice is initialized here
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Pointer to the CSR graph object we process on. @see Csr
     * @param[in] inversegraph Pointer to the inversed CSR graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     * @param[in] gpu_idx GPU index used for testing.
     * @param[in] partition_method Partition method to partition input graph.
     * @param[in] streams CUDA stream.
     * @param[in] queue_sizing Maximum queue sizing factor.
     * @param[in] in_sizing
     * @param[in] partition_factor Partition factor for partitioner.
     * @param[in] partition_seed Partition seed used for partitioner.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Init(
            bool          stream_from_host,       // Only meaningful for single-GPU
            Csr<VertexId, SizeT, Value> *graph,
            Csr<VertexId, SizeT, Value> *inversegraph = NULL,
            int           num_gpus          = 1,
            int          *gpu_idx           = NULL,
            std::string   partition_method  = "random",
            cudaStream_t *streams           = NULL,
            float         queue_sizing      = 2.0,
            float         in_sizing         = 1.0,
            float         partition_factor  = -1.0,
            int           partition_seed    = -1)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = BaseProblem::Init(
            stream_from_host,
            graph,
            inversegraph,
            num_gpus,
            gpu_idx,
            partition_method,
            queue_sizing,
            partition_factor,
            partition_seed))
            return retval;

        // No data in DataSlice needs to be copied from host

        data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

        for (int gpu = 0; gpu < this -> num_gpus; gpu++)
        {
            data_slices[gpu].SetName("data_slices[]");
            if (retval = util::SetDevice(this -> gpu_idx[gpu]))
                return retval;
            if (retval = data_slices[gpu].Allocate(1, util::DEVICE | util::HOST))
                return retval;
            DataSlice *data_slice
                = data_slices[gpu].GetPointer(util::HOST);
            GraphSlice<VertexId, SizeT, Value> *graph_slice
                = this->graph_slices[gpu];
            data_slice -> streams.SetPointer(streams + gpu * num_gpus * 2, num_gpus * 2);

            if (retval = data_slice->Init(
                walk_length,
                mode,
                this -> num_gpus,
                this -> gpu_idx[gpu],
                this -> use_double_buffer,
              &(this -> sub_graphs[gpu]),
                this -> num_gpus > 1? graph_slice -> in_counter     .GetPointer(util::HOST) : NULL,
                this -> num_gpus > 1? graph_slice -> out_counter    .GetPointer(util::HOST) : NULL,
                in_sizing))
                return retval;
        }


        // no mgpu support
        return retval;
    }

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * Initialize frontier type and frontier queue_size
     * Fill in data_slice inital values of problem here
     *
     * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed).
     * @param[in] queue_sizing Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively).
     * @param[in] queue_sizing1
     *
     *  \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Reset(
            FrontierType frontier_type,
            double queue_sizing,
            double queue_sizing1 = -1)
    {

        cudaError_t retval = cudaSuccess;
        if (queue_sizing1 < 0) queue_sizing1 = queue_sizing;
        

        if (retval = util::SetDevice(this->gpu_idx[0])) return retval;


        //single gpu
        if (retval = data_slices[0] -> Reset(walk_length,
                                             mode,
                                             frontier_type, 
                                             this->graph_slices[0],
                                             queue_sizing, 
                                             queue_sizing1)) return retval;

        if (retval = data_slices[0].Move(util::HOST, util::DEVICE)) return retval;


        /*
        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            // Set device
            if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
            if (retval = data_slices[gpu] -> Reset(
                frontier_type, this->graph_slices[gpu],
                queue_sizing, queue_sizing1)) return retval;
            if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE)) return retval;
        }
        */

        return retval;
    }

    /** @} */

};

} //namespace rw
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
