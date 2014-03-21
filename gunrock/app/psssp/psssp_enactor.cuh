// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sssp_enactor.cuh
 *
 * @brief SSSP Problem Enactor
 */

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/edge_map_partitioned/kernel.cuh>
#include <gunrock/oprtr/edge_map_partitioned/kernel_policy.cuh>
#include <gunrock/oprtr/vertex_map/kernel.cuh>
#include <gunrock/oprtr/vertex_map/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/psssp/psssp_problem.cuh>
#include <gunrock/app/psssp/psssp_functor.cuh>

#include <moderngpu.cuh>

using namespace mgpu;

namespace gunrock {
namespace app {
namespace psssp {

/**
 * @brief SSSP problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template<bool INSTRUMENT>
class PSSSPEnactor : public EnactorBase
{
    // Members
    protected:

    /**
     * CTA duty kernel stats
     */
    util::KernelRuntimeStatsLifetime edge_map_kernel_stats;
    util::KernelRuntimeStatsLifetime vertex_map_kernel_stats;

    unsigned long long total_runtimes;              // Total working time by each CTA
    unsigned long long total_lifetimes;             // Total life time of each CTA
    unsigned long long total_queued;

    /**
     * A pinned, mapped word that the traversal kernels will signal when done
     */
    volatile int        *done;
    int                 *d_done;
    cudaEvent_t         throttle_event;

    /**
     * Current iteration, also used to get the final search depth of the SSSP search
     */
    long long                           iteration;

    // Methods
    protected:

    /**
     * @brief Prepare the enactor for SSSP kernel call. Must be called prior to each SSSP search.
     *
     * @param[in] problem SSSP Problem object which holds the graph data and SSSP problem data to compute.
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
            if (!done) {
                int flags = cudaHostAllocMapped;

                // Allocate pinned memory for done
                if (retval = util::GRError(cudaHostAlloc((void**)&done, sizeof(int) * 1, flags),
                    "SSSPEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) break;

                // Map done into GPU space
                if (retval = util::GRError(cudaHostGetDevicePointer((void**)&d_done, (void*) done, 0),
                    "SSSPEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;

                // Create throttle event
                if (retval = util::GRError(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
                    "SSSPEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) break;
            }

            //initialize runtime stats
            if (retval = edge_map_kernel_stats.Setup(edge_map_grid_size)) break;
            if (retval = vertex_map_kernel_stats.Setup(vertex_map_grid_size)) break;

            //Reset statistics
            iteration           = 0;
            total_runtimes      = 0;
            total_lifetimes     = 0;
            total_queued        = 0;
            done[0]             = -1;

            //graph slice
            //typename ProblemData::GraphSlice *graph_slice = problem->graph_slices[0];

            // Bind row-offsets and bitmask texture
            /*cudaChannelFormatDesc   row_offsets_desc = cudaCreateChannelDesc<SizeT>();
            if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                    graph_slice->d_row_offsets,
                    row_offsets_desc,
                    (graph_slice->nodes + 1) * sizeof(SizeT)),
                        "SSSPEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;*/

            /*cudaChannelFormatDesc   column_indices_desc = cudaCreateChannelDesc<VertexId>();
            if (retval = util::GRError(cudaBindTexture(
                            0,
                            gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref,
                            graph_slice->d_column_indices,
                            column_indices_desc,
                            graph_slice->edges * sizeof(VertexId)),
                        "SSSPEnactor cudaBindTexture column_indices_tex_ref failed", __FILE__, __LINE__)) break;*/
        } while (0);
        
        return retval;
    }

    public:

    /**
     * @brief SSSPEnactor constructor
     */
    PSSSPEnactor(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG),
        iteration(0),
        total_queued(0),
        done(NULL),
        d_done(NULL)
    {}

    /**
     * @brief SSSPEnactor destructor
     */
    virtual ~PSSSPEnactor()
    {
        if (done) {
            util::GRError(cudaFreeHost((void*)done),
                "SSSPEnactor cudaFreeHost done failed", __FILE__, __LINE__);

            util::GRError(cudaEventDestroy(throttle_event),
                "SSSPEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
        }
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics about the last SSSP search enacted.
     *
     * @param[out] total_queued Total queued elements in SSSP kernel running.
     * @param[out] search_depth Search depth of SSSP algorithm.
     * @param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
     */
    template <typename VertexId>
    void GetStatistics(
        long long &total_queued,
        VertexId &search_depth,
        double &avg_duty)
    {
        cudaThreadSynchronize();

        total_queued = this->total_queued;
        search_depth = this->iteration;

        avg_duty = (total_lifetimes >0) ?
            double(total_runtimes) / total_lifetimes : 0.0;
    }

    /** @} */

    /**
     * @brief Enacts a breadth-first search computing on the specified graph.
     *
     * @tparam EdgeMapPolicy Kernel policy for forward edge mapping.
     * @tparam VertexMapPolicy Kernel policy for vertex mapping.
     * @tparam SSSPProblem SSSP Problem type.
     *
     * @param[in] problem SSSPProblem object.
     * @param[in] src Source node for SSSP.
     * @param[in] max_grid_size Max grid size for SSSP kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename EdgeMapPolicy,
        typename VertexMapPolicy,
        typename PSSSPProblem>
    cudaError_t EnactPSSSP(
    CudaContext				 &context,
    PSSSPProblem                          *problem,
    typename PSSSPProblem::VertexId       src,
    int                                 max_grid_size = 0)
    {
        typedef typename PSSSPProblem::SizeT      SizeT;
        typedef typename PSSSPProblem::VertexId   VertexId;

        typedef PSSSPFunctor<
            VertexId,
            SizeT,
            PSSSPProblem> SsspFunctor;

        cudaError_t retval = cudaSuccess;

        do {
            // Determine grid size(s)
            int edge_map_occupancy      = EdgeMapPolicy::CTA_OCCUPANCY;
            int edge_map_grid_size      = MaxGridSize(edge_map_occupancy, max_grid_size);

            int vertex_map_occupancy    = VertexMapPolicy::CTA_OCCUPANCY;
            int vertex_map_grid_size    = MaxGridSize(vertex_map_occupancy, max_grid_size);

            if (DEBUG) {
                printf("SSSP edge map occupancy %d, level-grid size %d\n",
                        edge_map_occupancy, edge_map_grid_size);
                printf("SSSP vertex map occupancy %d, level-grid size %d\n",
                        vertex_map_occupancy, vertex_map_grid_size);
                printf("Iteration, Edge map queue, Vertex map queue\n");
                printf("0");
            }


            // Lazy initialization
            if (retval = Setup(problem, edge_map_grid_size, vertex_map_grid_size)) break;

            // Single-gpu graph slice
            typename PSSSPProblem::GraphSlice *graph_slice = problem->graph_slices[0];
            typename PSSSPProblem::DataSlice *data_slice = problem->d_data_slices[0];


            SizeT queue_length          = 1;
            VertexId queue_index        = 0;        // Work queue index
            int selector                = 0;
            SizeT num_elements          = 1;

            bool queue_reset = true; 

	    unsigned int *d_node_locks;
            unsigned int *d_node_locks_out;

            if (retval = util::GRError(cudaMalloc(
                            (void**)&d_node_locks,
                            EdgeMapPolicy::BLOCKS * sizeof(unsigned int)),
                        "PBFSProblem cudaMalloc d_node_locks failed", __FILE__, __LINE__)) return retval;

            if (retval = util::GRError(cudaMalloc(
                            (void**)&d_node_locks_out,
                            EdgeMapPolicy::BLOCKS * sizeof(unsigned int)),
                        "PBFSProblem cudaMalloc d_node_locks_out failed", __FILE__, __LINE__)) return retval;

 	    while (done[0] < 0) {
                if (queue_length == 0) break;
                //Partitioned Edge Map
                //
                // Get Rowoffsets
                // Use scan to compute edge_offsets for each vertex in the frontier
                // Use sorted sort to compute partition bound for each work-chunk
                // load edge-expand-partitioned kernel
                int num_block = (queue_length + EdgeMapPolicy::THREADS - 1)/EdgeMapPolicy::THREADS;
                gunrock::oprtr::edge_map_partitioned::GetEdgeCounts<EdgeMapPolicy, PSSSPProblem, SsspFunctor> <<< num_block, EdgeMapPolicy::THREADS >>>(
                                        graph_slice->d_row_offsets,
                                        graph_slice->frontier_queues.d_keys[selector],
                                       problem->data_slices[0]->d_scanned_edges,
                                        queue_length,
                                        graph_slice->frontier_elements[selector],
                                        graph_slice->frontier_elements[selector^1]);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "edge_map_partitioned kernel failed", __FILE__, __LINE__))) break;
                //Scan<MgpuScanTypeInc>((unsigned int*)data_slice->d_scanned_edges, queue_length, context);
                Scan<MgpuScanTypeInc>((int*)problem->data_slices[0]->d_scanned_edges, queue_length, (int)0, mgpu::plus<int>(),
		(int*)0, (int*)0, (int*)problem->data_slices[0]->d_scanned_edges, context);

		        /*//Test Scan
		        int *t_scan = new int[10];
		        for (int i = 0; i < 10; ++i) {
		            t_scan[i] = 1;
		        }
		        int *d_t_scan;
		        cudaMalloc((void**)&d_t_scan, sizeof(int)*10);
                if (retval = util::GRError(cudaMemcpy(
                                d_t_scan,
                                t_scan,
                                sizeof(int)*10,
                                cudaMemcpyHostToDevice),
                            "test scan failed", __FILE__, __LINE__)) return retval;
                Scan<MgpuScanTypeInc>(d_t_scan, 10, (int)0, mgpu::plus<int>(),
		            (int*)0, (int*)0, d_t_scan, context);

                util::DisplayDeviceResults(d_t_scan, 10);
                cudaFree(d_t_scan);
                delete[] t_scan;*/

                SizeT *temp = new SizeT[1];
                cudaMemcpy(temp, problem->data_slices[0]->d_scanned_edges+queue_length-1, sizeof(SizeT), cudaMemcpyDeviceToHost);
                SizeT output_queue_len = temp[0];
                printf("num block:%d, scanned length:%d\n", num_block, output_queue_len);
                
                // Edge Expand Kernel
		{
                        gunrock::oprtr::edge_map_partitioned::RelaxLightEdges<EdgeMapPolicy, PSSSPProblem, SsspFunctor> <<< num_block, EdgeMapPolicy::THREADS >>>(
                                        queue_reset,
                                        queue_index,
                                        iteration,
                                        graph_slice->d_row_offsets,
                                        graph_slice->d_column_indices,
                                        problem->data_slices[0]->d_scanned_edges,
                                        d_done,
                                        graph_slice->frontier_queues.d_keys[selector],
                                        graph_slice->frontier_queues.d_keys[selector^1],
                                        data_slice,
                                        queue_length,
                                        output_queue_len,
                                        graph_slice->frontier_elements[selector],
                                        graph_slice->frontier_elements[selector^1],
                                        work_progress,
                                        this->edge_map_kernel_stats);
                }
		//Only need to reset queue for once
                if (queue_reset)
                    queue_reset = false;

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "edge_map_partitioned kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event); //give host memory mapped visibility to GPU updates


                queue_index++;
                selector ^= 1;

                if (DEBUG) {
                    if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
                    //util::DisplayDeviceResults(graph_slice->frontier_queues.d_keys[selector], queue_length);
                    printf(", %lld", (long long) queue_length);
                }

                if (INSTRUMENT) {
                    if (retval = edge_map_kernel_stats.Accumulate(
                        edge_map_grid_size,
                        total_runtimes,
                        total_lifetimes)) break;
                }

                // Throttle
                if (iteration & 1) {
                    if (retval = util::GRError(cudaEventRecord(throttle_event), "PBFSEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (retval = util::GRError(cudaEventSynchronize(throttle_event), "PBFSEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }

                // Check if done
                if (done[0] == 0) break;
		
		// Vertex Map
                gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, PSSSPProblem, SsspFunctor>
                    <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                            iteration + 1,
                            queue_reset,
                            queue_index,
                            1,
                            num_elements,
                            d_done,
                            graph_slice->frontier_queues.d_keys[selector],      // d_in_queue
                            graph_slice->frontier_queues.d_values[selector],    // d_pred_in_queue
                            graph_slice->frontier_queues.d_keys[selector^1],    // d_out_queue
                            data_slice,
                            problem->data_slices[0]->d_visited_mask,
                            work_progress,
                            graph_slice->frontier_elements[selector],           // max_in_queue
                            graph_slice->frontier_elements[selector^1],         // max_out_queue
                            this->vertex_map_kernel_stats);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "vertex_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates

                queue_index++;
                selector ^= 1;
                iteration++;

                
                if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
                if (INSTRUMENT || DEBUG) {
                    total_queued += queue_length;
                    if (DEBUG) printf(", %lld", (long long) queue_length);
                    if (INSTRUMENT) {
                        if (retval = vertex_map_kernel_stats.Accumulate(
                                    vertex_map_grid_size,
                                    total_runtimes,
                                    total_lifetimes)) break;
                    }
                }
                // Check if done
                if (done[0] == 0) break;

                if (DEBUG) printf("\n%lld", (long long) iteration);

            }

            util::GRError(cudaFree(d_node_locks), "GpuSlice cudaFree d_node_locks failed", __FILE__, __LINE__);
            util::GRError(cudaFree(d_node_locks_out), "GpuSlice cudaFree d_node_locks_out failed", __FILE__, __LINE__);

            if (retval) break;

	} while(0);
        if (DEBUG) printf("\nGPU SSSP Done.\n");
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief SSSP Enact kernel entry.
     *
     * @tparam SSSPProblem SSSP Problem type. @see SSSPProblem
     *
     * @param[in] problem Pointer to SSSPProblem object.
     * @param[in] src Source node for SSSP.
     * @param[in] max_grid_size Max grid size for SSSP kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename PSSSPProblem>
    cudaError_t Enact(
        CudaContext			 &context,
        PSSSPProblem                      *problem,
        typename PSSSPProblem::VertexId    src,
        int                             max_grid_size = 0)
    {
        
        if (this->cuda_props.device_sm_version >= 300) {
            typedef gunrock::oprtr::vertex_map::KernelPolicy<
                PSSSPProblem,                         // Problem data type
            300,                                // CUDA_ARCH
            INSTRUMENT,                         // INSTRUMENT
            0,                                  // SATURATION QUIT
            true,                               // DEQUEUE_PROBLEM_SIZE
            8,                                  // MIN_CTA_OCCUPANCY
            7,                                  // LOG_THREADS
            1,                                  // LOG_LOAD_VEC_SIZE
            0,                                  // LOG_LOADS_PER_TILE
            5,                                  // LOG_RAKING_THREADS
            5,                                  // END_BITMASK_CULL
            8>                                  // LOG_SCHEDULE_GRANULARITY
                VertexMapPolicy;

            typedef gunrock::oprtr::edge_map_partitioned::KernelPolicy<
                PSSSPProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
		1,
		10,
		15,
		32 * 1024>
                    EdgeMapPolicy;

            return EnactPSSSP<EdgeMapPolicy, VertexMapPolicy, PSSSPProblem>(
                    context, problem, src, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */

};

} // namespace sssp
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
