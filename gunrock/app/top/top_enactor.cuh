// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * top_enactor.cuh
 *
 * @brief TOP Problem Enactor
 */

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/edge_map_forward/kernel_policy.cuh>
#include <gunrock/oprtr/vertex_map/kernel.cuh>
#include <gunrock/oprtr/vertex_map/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/top/top_problem.cuh>
//#include <gunrock/app/top/top_functor.cuh>

#include <moderngpu.cuh>

namespace gunrock {
namespace app {
namespace top {

using namespace mgpu;

/**
 * @brief TOP problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template<bool INSTRUMENT>
class TOPEnactor : public EnactorBase
{
    // Members
    protected:

    /**
     * CTA duty kernel stats
     */
    util::KernelRuntimeStatsLifetime edge_map_kernel_stats;
    util::KernelRuntimeStatsLifetime vertex_map_kernel_stats;

    unsigned long long total_runtimes;  // Total working time by each CTA
    unsigned long long total_lifetimes; // Total life time of each CTA
    unsigned long long total_queued;

    /**
     * A pinned, mapped word that the traversal kernels will signal when done
     */
    volatile int        *done;
    int                 *d_done;
    cudaEvent_t         throttle_event;

    /**
     * Current iteration, also used to get the final search depth of the TOP search
     */
    long long           iteration;

    // Methods
    protected:

    /**
     * @brief Prepare the enactor for TOP kernel call. Must be called prior to each TOP iteration.
     *
     * @param[in] problem TOP Problem object which holds the graph data and TOP problem data to compute.
     * @param[in] edge_map_grid_size CTA occupancy for edge mapping kernel call.
     * @param[in] vertex_map_grid_size CTA occupancy for vertex mapping kernel call.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename ProblemData>
    cudaError_t Setup(
        ProblemData *problem,
        int edge_map_grid_size,
        int vertex_map_grid_size)
    {
        typedef typename ProblemData::SizeT         SizeT;
        typedef typename ProblemData::VertexId      VertexId;
        
        cudaError_t retval = cudaSuccess;

        do {
            //initialize the host-mapped "done"
            if (!done) 
            {
                int flags = cudaHostAllocMapped;

                // Allocate pinned memory for done
                if (retval = util::GRError(cudaHostAlloc((void**)&done, sizeof(int) * 1, flags),
                    "TOPEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) break;

                // Map done into GPU space
                if (retval = util::GRError(cudaHostGetDevicePointer((void**)&d_done, (void*) done, 0),
                    "TOPEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;

                // Create throttle event
                if (retval = util::GRError(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
                    "TOPEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) break;
            }

            //initialize runtime stats
            if (retval = edge_map_kernel_stats.Setup(edge_map_grid_size)) break;
            if (retval = vertex_map_kernel_stats.Setup(vertex_map_grid_size)) break;

            //Reset statistics
            iteration           =  0;
            total_runtimes      =  0;
            total_lifetimes     =  0;
            total_queued        =  0;
            done[0]             = -1;

            //graph slice
            typename ProblemData::GraphSlice *graph_slice = problem->graph_slices[0];

            // Bind row-offsets and bitmask texture
            cudaChannelFormatDesc   row_offsets_desc = cudaCreateChannelDesc<SizeT>();
            if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                    graph_slice->d_row_offsets,
                    row_offsets_desc,
                    (graph_slice->nodes + 1) * sizeof(SizeT)),
                    "TOPEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

            /*cudaChannelFormatDesc   column_indices_desc = cudaCreateChannelDesc<VertexId>();
            if (retval = util::GRError(cudaBindTexture(
                            0,
                            gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref,
                            graph_slice->d_column_indices,
                            column_indices_desc,
                            graph_slice->edges * sizeof(VertexId)),
                        "TOPEnactor cudaBindTexture column_indices_tex_ref failed", __FILE__, __LINE__)) break;*/
        } while (0);
        
        return retval;
    }

    public:

    /**
     * @brief TOPEnactor constructor
     */
    TOPEnactor(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG),
        iteration(0),
        total_queued(0),
        done(NULL),
        d_done(NULL)
    {}

    /**
     * @brief TOPEnactor destructor
     */
    virtual ~TOPEnactor()
    {
        if (done) 
        {
            util::GRError(cudaFreeHost((void*)done),
                "TOPEnactor cudaFreeHost done failed", __FILE__, __LINE__);

            util::GRError(cudaEventDestroy(throttle_event),
                "TOPEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
        }
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics about the last TOP search enacted.
     *
     * @param[out] total_queued Total queued elements in TOP kernel running.
     * @param[out] search_depth Search depth of TOP algorithm.
     * @param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
     */
    template <typename VertexId>
    void GetStatistics(
        long long   &total_queued,
        VertexId    &search_depth,
        double      &avg_duty)
    {
        cudaThreadSynchronize();

        total_queued = this->total_queued;
        search_depth = this->iteration;

        avg_duty = (total_lifetimes > 0) ?
            double(total_runtimes) / total_lifetimes : 0.0;
    }

    /** @} */

    /**
     * @brief Enacts a breadth-first search computing on the specified graph.
     *
     * @tparam EdgeMapPolicy Kernel policy for forward edge mapping.
     * @tparam VertexMapPolicy Kernel policy for vertex mapping.
     * @tparam TOPProblem TOP Problem type.
     *
     * @param[in] problem TOPProblem object.
     * @param[in] max_grid_size Max grid size for TOP kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename EdgeMapPolicy,
        typename VertexMapPolicy,
        typename TOPProblem>
    cudaError_t EnactTOP(
    CudaContext     &context,
    TOPProblem      *problem,
    int             max_grid_size = 0)
    {
        typedef typename TOPProblem::SizeT      SizeT;
        typedef typename TOPProblem::VertexId   VertexId;
        
        /*
        typedef TOPFunctor<
            VertexId,
            SizeT,
            VertexId,
            TOPProblem> TopFunctor;
        */
        
        cudaError_t retval = cudaSuccess;

        do {
            // Add Enactor Code here
			// determine grid size for edge and vertex mapping
			int edge_map_occupancy = EdgeMapPolicy::CTA_OCCUPANCY;
		 	int edge_map_grid_size = MaxGridSize(edge_map_occupancy, max_grid_size);
		 	int vertex_map_occupancy = VertexMapPolicy::CTA_OCCUPANCY;
			int vertex_map_grid_size = MaxGridSize(vertex_map_occupancy, max_grid_size);

			// initialization
			if (retval = Setup(problem, edge_map_grid_size, vertex_map_grid_size)) break;
			
			// single gpu graph slice
			typename TOPProblem::GraphSlice *graph_slice = problem->graph_slices[0];
			typename TOPProblem::DataSlice	*data_slice	= problem->d_data_slices[0];

			fflush(stdout);

			if (DEBUG)
			{
				printf("data_slices[0] d_node_id");
            	util::DisplayDeviceResults(problem->data_slices[0]->d_node_id, graph_slice->nodes);
            	printf("data_slices[0] d_degrees");
            	util::DisplayDeviceResults(problem->data_slices[0]->d_degrees, graph_slice->nodes);
        	}

            // sort by key using mgpu
       	   	MergesortPairs(problem->data_slices[0]->d_degrees,
                problem->data_slices[0]->d_node_id, graph_slice->nodes, mgpu::less<int>(), context);
			
			if (DEBUG)
			{
            	printf("sorted data_slices[0] d_node_id");
            	util::DisplayDeviceResults(problem->data_slices[0]->d_node_id, graph_slice->nodes);
            	printf("sorted data_slices[0] d_degrees");
            	util::DisplayDeviceResults(problem->data_slices[0]->d_degrees, graph_slice->nodes);
			}

        }while(0);

        printf("\n ----- GPU TOP Done ----- \n");
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief TOP Enact kernel entry.
     *
     * @tparam TOPProblem TOP Problem type. @see TOPProblem
     *
     * @param[in] problem Pointer to TOPProblem object.
     * @param[in] src Source node for TOP.
     * @param[in] max_grid_size Max grid size for TOP kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename TOPProblem>
    cudaError_t Enact(
        CudaContext     &context,
        TOPProblem      *problem,
        int             max_grid_size = 0)
    {
        if (this->cuda_props.device_sm_version >= 300) 
        {
            typedef gunrock::oprtr::vertex_map::KernelPolicy<
                TOPProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                8,                                  // MIN_CTA_OCCUPANCY
                8,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                5,                                  // END_BITMASK_CULL
                8>                                  // LOG_SCHEDULE_GRANULARITY
                VertexMapPolicy;

            typedef gunrock::oprtr::edge_map_forward::KernelPolicy<
                TOPProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY
                8,                                  // LOG_THREADS
                0,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                32,                                 // WARP_GATHER_THRESHOLD
                128 * 4,                            // CTA_GATHER_THRESHOLD
                7>                                  // LOG_SCHEDULE_GRANULARITY
                EdgeMapPolicy;

            return  EnactTOP<EdgeMapPolicy, VertexMapPolicy, TOPProblem>(
                        context, problem, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */

};

} // namespace top
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
