// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * rw_enactor.cuh
 *
 * @brief RW Problem Enactor
 */

#pragma once



#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/sort_utils.cuh>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <cub/cub.cuh>
#include <moderngpu.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/rw/rw_functor.cuh>
#include <gunrock/app/rw/rw_problem.cuh>

#define ELEMS_PER_THREAD 4
#define THREAD_BLOCK 128


namespace gunrock {
namespace app {
namespace rw {


/**
 * @brief RW Problem enactor class.
 *
 * @tparam _Problem Problem type we process on
 * @tparam _INSTRUMENT Whether or not to collect per-CTA clock-count stats.
 * @tparam _DEBUG Whether or not to enable debug mode.
 * @tparam _SIZE_CHECK Whether or not to enable size check.
 */
template <typename _Problem>
class RWEnactor : public EnactorBase<typename _Problem::SizeT>
{

public:
    typedef _Problem                   Problem;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value   ;
    typedef EnactorBase<SizeT>         BaseEnactor;
    Problem    *problem;
    ContextPtr *context;

    /**
     * @brief RWEnactor constructor
     */
    RWEnactor(
        int   num_gpus   = 1,
        int  *gpu_idx    = NULL,
        bool  instrument = false,
        bool  debug      = false,
        bool  size_check = true) :
        BaseEnactor(
            VERTEX_FRONTIERS, num_gpus, gpu_idx,
            instrument, debug, size_check),
        problem       (NULL),
        context       (NULL)
    {
    }

    /**
     * @brief RWEnactor destructor
     */
    virtual ~RWEnactor()
    {
        Release();
    }

    cudaError_t Release()
    {
	cudaError_t retval = cudaSuccess;
        if (retval = BaseEnactor::Release()) return retval;
        problem = NULL;
        return retval;
    }

    /** @} */

    /**
     * @brief Initialize the problem.
     *
     * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
     * @tparam FilterKernelPolicy Kernel policy for filter operator.
     *
     * @param[in] context CudaContext pointer for ModernGPU API.
     * @param[in] problem Pointer to Problem object.
     * @param[in] max_grid_size Maximum grid size for kernel calls.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t InitRW(
        ContextPtr  *context,
        Problem     *problem,
        int         max_grid_size = 0)
    {
        cudaError_t retval = cudaSuccess;

        // Init parent class
        if (retval = BaseEnactor::Init(
            max_grid_size,
            AdvanceKernelPolicy::CTA_OCCUPANCY,
            FilterKernelPolicy::CTA_OCCUPANCY))
            return retval;

        this->problem = problem;
        this->context = context;

        return retval;
    }

    /**
     * @brief Reset enactor
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Reset()
    {
        return BaseEnactor::Reset();

    }

    /** @} */

    /**
     * @brief Enacts computing on the specified graph.
     *
     * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
     * @tparam FilterKernelPolicy Kernel policy for filter operator.
     *
     * @param[in] src Source node to start primitive.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t EnactRW()
    {
        typedef RWFunctor<VertexId, SizeT, Value, Problem> RWFunctor;
        typedef typename Problem::DataSlice DataSlice;

        // single gpu graph slice
        GraphSlice<VertexId, SizeT, Value>
                  *graph_slice   =  problem -> graph_slices[0];
        DataSlice *data_slice    =  problem -> data_slices [0].GetPointer(util::HOST);
        util::CtaWorkProgressLifetime<SizeT>
                  *work_progress = &this->work_progress    [0];
        SizeT      nodes         = graph_slice -> nodes;
        cudaError_t retval       = cudaSuccess;
        SizeT walk_length        = problem->walk_length;

        //make curandGen as a field of Enactor, destroy generator in enactor destructor
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); /* Set seed */
        curandSetPseudoRandomGeneratorSeed(gen, time(NULL)); /* Generate n floats on device */


 

        for(SizeT i = 0; i < walk_length-1; i++){ //should be walk_length-1
            //curandSetPseudoRandomGeneratorSeed(gen, 1234ULL+i);
	    curandGenerateUniform(gen, data_slice->d_rand.GetPointer(util::DEVICE), nodes);
	    /*if(i == 0){
		for(SizeT i=0; i < nodes; i++){
   			printf("d_rand[%d]: %.6f -> %d\n", i, d_rand[i]);
	    	}
	    } */

           rw::RandomNext<<<128,128>>>(data_slice ->paths.GetPointer(util::DEVICE),
                                        data_slice ->num_neighbor.GetPointer(util::DEVICE),
                                        data_slice ->d_rand.GetPointer(util::DEVICE),
                                        data_slice -> d_row_offsets.GetPointer(util::DEVICE),
                                        data_slice -> d_col_indices.GetPointer(util::DEVICE),
                                        nodes,
                                        i);
        }
        curandDestroyGenerator(gen);


        // check if any of the frontiers overflowed due to redundant expansion
        bool overflowed = false;
        if (retval = work_progress -> CheckOverflow(overflowed)) return retval;
        if (overflowed)
        {
            retval = util::GRError(
                cudaErrorInvalidConfiguration,
                "Frontier queue overflow. Please increase queus size factor.",
                __FILE__, __LINE__);
            return retval;
        }


        if (this -> debug) printf("\nGPU RW Done.\n");
        return retval;
    }

    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t EnactSortedRW()
    {
        typedef RWFunctor<VertexId, SizeT, Value, Problem> RWFunctor;
        typedef typename Problem::DataSlice DataSlice;

        // single gpu graph slice
        GraphSlice<VertexId, SizeT, Value>
                  *graph_slice   =  problem -> graph_slices[0];
        DataSlice *data_slice    =  problem -> data_slices [0].GetPointer(util::HOST);
        util::CtaWorkProgressLifetime<SizeT>
                  *work_progress = &this->work_progress    [0];
        SizeT      nodes         = graph_slice -> nodes;
        cudaError_t retval       = cudaSuccess;
        SizeT walk_length        = problem->walk_length;


        //make curandGen as a field of Enactor, destroy generator in enactor destructor
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); /* Set seed */
        curandSetPseudoRandomGeneratorSeed(gen, time(NULL)); /* Generate n floats on device */


        for(SizeT i = 0; i < walk_length-1; i++){ 
            	//curandSetPseudoRandomGeneratorSeed(gen, 1234ULL+i);
                curandGenerateUniform(gen, data_slice->d_rand.GetPointer(util::DEVICE), nodes);

		if(i!= 0){
            	//sort the node index
            	util::CUBRadixSort<SizeT, VertexId>(
                    	true, nodes,
                    	data_slice -> paths.GetPointer(util::DEVICE)+i*nodes,
                    	data_slice -> node_id.GetPointer(util::DEVICE));
            
	    	
            	util::CUBRadixSort<SizeT, VertexId>(
                   	true, nodes,
                    	data_slice -> paths.GetPointer(util::DEVICE)+i*nodes);
            	}
           	rw::SortedRandomNext<<<128,128>>>(data_slice ->paths.GetPointer(util::DEVICE),
                                        data_slice ->node_id.GetPointer(util::DEVICE),
                                        data_slice ->num_neighbor.GetPointer(util::DEVICE),
                                        data_slice ->d_rand.GetPointer(util::DEVICE),
                                        data_slice -> d_row_offsets.GetPointer(util::DEVICE),
                                        data_slice -> d_col_indices.GetPointer(util::DEVICE),
                                        nodes,
                                        i);
           if(i!=0){
		
           util::CUBRadixSort<SizeT, VertexId>(
                    true, nodes,
                    data_slice -> node_id.GetPointer(util::DEVICE),
                    data_slice -> paths.GetPointer(util::DEVICE)+i*nodes);
            }
            util::MemsetIdxKernel<<<128, 128>>>(
                data_slice->node_id.GetPointer(util::DEVICE), nodes);

        }
        //curandDestroyGenerator(gen);


        // check if any of the frontiers overflowed due to redundant expansion
        bool overflowed = false;
        if (retval = work_progress -> CheckOverflow(overflowed)) return retval;
        if (overflowed)
        {
            retval = util::GRError(
                cudaErrorInvalidConfiguration,
                "Frontier queue overflow. Please increase queus size factor.",
                __FILE__, __LINE__);
            return retval;
        }


        if (this -> debug) printf("\nGPU RW Done.\n");
        return retval;
    }


    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t EnactBlockSortedRW()
    {
        typedef RWFunctor<VertexId, SizeT, Value, Problem> RWFunctor;
        typedef typename Problem::DataSlice DataSlice;

        // single gpu graph slice
        GraphSlice<VertexId, SizeT, Value>
                  *graph_slice   =  problem -> graph_slices[0];
        DataSlice *data_slice    =  problem -> data_slices [0].GetPointer(util::HOST);
        util::CtaWorkProgressLifetime<SizeT>
                  *work_progress = &this->work_progress    [0];
        //SizeT      nodes         = graph_slice -> nodes;
        cudaError_t retval       = cudaSuccess;
        SizeT walk_length        = problem->walk_length;


        //const int tb = (nodes / ELEMS_PER_THREAD) > THREAD_BLOCK ?  THREAD_BLOCK : (nodes / ELEMS_PER_THREAD);
        //int grid = nodes/(THREAD_BLOCK*ELEMS_PER_THREAD);
        // { GRID = DSIZE / ELEMS_PER_THREAD };

        //int frontierSize=grid*THREAD_BLOCK*ELEMS_PER_THREAD;
        //int const& const_tb = tb;

        //need to work out how much padding needed to be done after block-wise sort
        //in problem
        //int block_left = nodes - THREAD_BLOCK*ELEMS_PER_THREAD*;


        //ITEM_PER_THERAD = 4
        //block size-> 128
        //if  data size  = 70
        // 70/4 = 17, -> 17 threads/block
        // 70/128 = 0-> 1 block(gridsize) 
        // only sorting 68 element

        int block_nodes = data_slice->size;
        int trailing_nodes = data_slice->trailing;
        //make curandGen as a field of Enactor, destroy generator in enactor destructor
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); /* Set seed */
        curandSetPseudoRandomGeneratorSeed(gen, time(NULL)); /* Generate n floats on device */

        for(SizeT i = 0; i < walk_length-1; i++){
                //curandSetPseudoRandomGeneratorSeed(gen, 1234ULL+i);
                curandGenerateUniform(gen, data_slice->d_rand.GetPointer(util::DEVICE), block_nodes);

                rw::BlockSortKernel<THREAD_BLOCK, ELEMS_PER_THREAD><<<data_slice->grid_size, THREAD_BLOCK>>>(
                                                            data_slice ->paths.GetPointer(util::DEVICE)+i*block_nodes, 
                                                            data_slice ->node_id.GetPointer(util::DEVICE)); 

                rw::BlockRandomNext<<<128,128>>>(data_slice ->paths.GetPointer(util::DEVICE)+i*block_nodes,
                                        data_slice ->node_id.GetPointer(util::DEVICE),
                                        data_slice ->num_neighbor.GetPointer(util::DEVICE),
                                        data_slice ->d_rand.GetPointer(util::DEVICE),
                                        data_slice -> d_row_offsets.GetPointer(util::DEVICE),
                                        data_slice -> d_col_indices.GetPointer(util::DEVICE),
                                        block_nodes);

                rw::BlockSortKernel<THREAD_BLOCK, ELEMS_PER_THREAD><<<data_slice->grid_size, THREAD_BLOCK>>>(
                                                            data_slice ->node_id.GetPointer(util::DEVICE),
                                                            data_slice ->paths.GetPointer(util::DEVICE)+i*block_nodes);

        }
        


        //need to raw rw walk the trailing slice from the graph

        
        for(SizeT i = 0; i < walk_length-1; i++){ //should be walk_length-1
            //curandSetPseudoRandomGeneratorSeed(gen, 1234ULL+i);
            curandGenerateUniform(gen, data_slice->d_rand.GetPointer(util::DEVICE), trailing_nodes);
        

           rw::RandomNext<<<128,128>>>( data_slice ->trailing_slice.GetPointer(util::DEVICE),
                                        data_slice ->num_neighbor.GetPointer(util::DEVICE),
                                        data_slice ->d_rand.GetPointer(util::DEVICE),
                                        data_slice -> d_row_offsets.GetPointer(util::DEVICE),
                                        data_slice -> d_col_indices.GetPointer(util::DEVICE),
                                        trailing_nodes,
                                        i);
        }
        
        curandDestroyGenerator(gen);



        // check if any of the frontiers overflowed due to redundant expansion
        bool overflowed = false;
        if (retval = work_progress -> CheckOverflow(overflowed)) return retval;
        if (overflowed)
        {
            retval = util::GRError(
                cudaErrorInvalidConfiguration,
                "Frontier queue overflow. Please increase queus size factor.",
                __FILE__, __LINE__);
            return retval;
        }


        if (this -> debug) printf("\nGPU RW Done.\n");
        return retval;
    }


    typedef gunrock::oprtr::filter::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        0,                                  // SATURATION QUIT
        true,                               // DEQUEUE_PROBLEM_SIZE
        8,                                  // MIN_CTA_OCCUPANCY
        8,                                  // LOG_THREADS
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        5,                                  // END_BITMASK_CULL
        8>                                  // LOG_SCHEDULE_GRANULARITY
    FilterKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,            // Problem data type
        300,                // CUDA_ARCH
        //INSTRUMENT,         // INSTRUMENT
        8,                  // MIN_CTA_OCCUPANCY
        10,                 // LOG_THREADS
        9,                  // LOG_BLOCKS
        32 * 128,           // LIGHT_EDGE_THRESHOLD
        1,                  // LOG_LOAD_VEC_SIZE
        0,                  // LOG_LOADS_PER_TILE
        5,                  // LOG_RAKING_THREADS
        32,                 // WARP_GATHER_THRESHOLD
        128 * 4,            // CTA_GATHER_THRESHOLD
        7,                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB_LIGHT>
    AdvanceKernelPolicy;




    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Sample Enact kernel entry.
     *
     * @param[in] src Source node to start primitive.
     * @param[in] traversal_mode Load-balanced or Dynamic cooperative.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    //template <typename SampleProblem>
    cudaError_t Enact(int mode)
    {
        int min_sm_version = -1;
        for (int i = 0; i < this->num_gpus; i++)
        {
            if (min_sm_version == -1 ||
                this->cuda_props[i].device_sm_version < min_sm_version)
            {
                min_sm_version = this->cuda_props[i].device_sm_version;
            }
        }

        if (min_sm_version >= 300)
        {
            if(mode == DEVICE){
                return EnactSortedRW<AdvanceKernelPolicy, FilterKernelPolicy> ();
            }else if(mode == BLOCK){
               	//return EnactSortedRW<AdvanceKernelPolicy, FilterKernelPolicy> (walk_length);
                return EnactBlockSortedRW<AdvanceKernelPolicy, FilterKernelPolicy> ();

            }else{
                return EnactRW<AdvanceKernelPolicy, FilterKernelPolicy> ();

            }
        }

        // to reduce compile time, get rid of other architecture for now
        // TODO: add all the kernel policy setting for all architectures

        printf("Not yet tuned for this architecture.\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /**
     * @brief Sample Enact kernel entry.
     *
     * @param[in] context CudaContext pointer for ModernGPU API.
     * @param[in] problem Pointer to Problem object.
     * @param[in] max_grid_size Maximum grid size for kernel calls.
     * @param[in] traversal_mode Load-balanced or Dynamic cooperative.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Init(
        ContextPtr   *context,
        Problem      *problem,
        int         max_grid_size = 0)
    {
        int min_sm_version = -1;
        for (int i = 0; i < this->num_gpus; i++)
        {
            if (min_sm_version == -1 ||
                this->cuda_props[i].device_sm_version < min_sm_version)
            {
                min_sm_version = this->cuda_props[i].device_sm_version;
            }
        }

        if (min_sm_version >= 300)
        {
            return InitRW<AdvanceKernelPolicy, FilterKernelPolicy> (
                context, problem, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernel policy settings for all archs
        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }


    /** @} */

};



} // namespace rw
} // namespace app
} // namespace gunrock




// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
