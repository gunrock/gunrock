// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sage_enactor.cuh
 *
 * @brief SSSP Problem Enactor
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/app/sage/sage_problem.cuh>
#include <gunrock/oprtr/oprtr.cuh>

namespace gunrock {
namespace app {
namespace sage {

/**
 * @brief Speciflying parameters for SSSP Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter info
 * \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;
    GUARD_CU(app::UseParameters_enactor(parameters));
    return retval;
}

/**
 * @brief defination of SAGE iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct SAGEIterationLoop : public IterationLoopBase
    <EnactorT, Use_FullQ | Push 
   // |
   // (((EnactorT::Problem::FLAG & Mark_Predecessors) != 0) ?
   //  Update_Predecessors : 0x0)
    >
{
    typedef typename EnactorT::VertexT VertexT;
    typedef typename EnactorT::SizeT   SizeT;
    typedef typename EnactorT::ValueT  ValueT;
    typedef typename EnactorT::Problem::GraphT::CooT CooT;
    typedef typename EnactorT::Problem::GraphT::GpT  GpT;
    typedef IterationLoopBase
        <EnactorT, Use_FullQ | Push 
       // |
       // (((EnactorT::Problem::FLAG & Mark_Predecessors) != 0) ?
       //  Update_Predecessors : 0x0)
       > BaseIterationLoop;

    SAGEIterationLoop() : BaseIterationLoop() {}

    /**
     * @brief Core computation of sage, one iteration
     * @param[in] peer_ Which GPU peers to work on, 0 means local
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Core(int peer_ = 0)
    {
        // Data sage that works on
        auto         &data_slice         =   this -> enactor ->
            problem -> data_slices[this -> gpu_num][0];
        auto         &enactor_slice      =   this -> enactor ->
            enactor_slices[this -> gpu_num * this -> enactor -> num_gpus + peer_];
        auto         &enactor_stats      =   enactor_slice.enactor_stats;
        auto         &graph              =   data_slice.sub_graph[0];
        auto         &W_f_1              =   data_slice.W_f_1_1D;
        auto          Wf1_dim1           =   data_slice.Wf1_dim1;
        auto         &W_a_1              =   data_slice.W_a_1_1D;
        auto          Wa1_dim1           =   data_slice.Wa1_dim1;
        auto         &W_f_2              =   data_slice.W_f_2_1D;
        auto          Wf2_dim0           =   data_slice.Wf2_dim0;
        auto          Wf2_dim1           =   data_slice.Wf2_dim1;
        auto         &W_a_2              =   data_slice.W_a_2_1D;
        auto          Wa2_dim0           =   data_slice.Wa2_dim0;
        auto          Wa2_dim1           =   data_slice.Wa2_dim1;
        auto         &features           =   data_slice.features_1D;
        auto          feature_column     =   data_slice.feature_column;
        auto         &source_result      =   data_slice.source_result;
        auto          result_column      =   data_slice.result_column;
        auto          num_children_per_source = data_slice.num_children_per_source;
        auto          num_leafs_per_child =  data_slice.num_leafs_per_child;
        auto         &sums               =   data_slice.sums;
        auto         &sums_child_feat    =   data_slice.sums_child_feat;
        //auto         &child_temp         =   data_slice.child_temp;
        auto         &children_temp      =   data_slice.children_temp;
        auto         &rand_states        =   data_slice.rand_states;
        auto         &retval             =   enactor_stats.retval;
        auto         &stream             =   enactor_slice.stream;
        auto         &iteration          =   enactor_stats.iteration;
        VertexT       source_start       =   iteration * data_slice.batch_size;
        VertexT       source_end         =  (iteration + 1) * data_slice.batch_size;
        if (source_end >= graph.nodes)
            source_end  = graph.nodes;
        VertexT       num_sources        =   source_end - source_start;
        SizeT         num_children       =   num_sources * data_slice.num_children_per_source;

        GUARD_CU(children_temp.ForEach(
            [] __host__ __device__ (ValueT &val)
            {
                val = 0;
            }, num_sources * Wf2_dim0, util::DEVICE, stream));
       
        GUARD_CU(sums_child_feat.ForEach(
            [] __host__ __device__ (ValueT &val)
            {
                val = 0;
            }, num_sources * feature_column, util::DEVICE, stream));
 
        GUARD_CU(data_slice.child_temp.ForAll(
            [source_start, num_children_per_source,
            graph, feature_column, features,
            num_leafs_per_child, W_f_1, Wf1_dim1,
            W_a_1, Wa1_dim1, Wf2_dim0, children_temp,
            sums_child_feat, sums, rand_states] 
            __host__ __device__ (ValueT *child_temp_, const SizeT &i)
            {
                ValueT *child_temp = child_temp_ + i * Wf2_dim0;
                VertexT source = i / num_children_per_source + source_start;
                SizeT   offset = curand_uniform(rand_states + i) 
                    * graph.GetNeighborListLength(source);
                SizeT   edge   = graph.GetNeighborListOffset(source) + offset;
                VertexT child  = graph.GetEdgeDest(edge); 
                //float sums [64] = {0.0} ; //local vector

                for (int f = 0; f < feature_column; f++)
                    sums[i * feature_column + f] = 0;
                SizeT child_degree = graph.GetNeighborListLength(child);
                for (int j = 0; j < num_leafs_per_child; j++)
                { 
                    //SizeT   offset2 = 0;//cuRand() * child_degree;
                    SizeT   edge2   = graph.GetNeighborListOffset(child) 
                        + curand_uniform(rand_states + i) * child_degree;
                    VertexT leaf    = graph.GetEdgeDest(edge2);
                            offset  = leaf * feature_column;

                    for (int f = 0; f < feature_column; f++) 
                    {
                        sums[i * feature_column + f] += features[offset + f]; 
                        ///num_neigh2;// merged line 176 171
                    }
                }
                for (int f = 0; f < feature_column; f++)
                    sums[i * feature_column + f] /= num_leafs_per_child;
                //agg feaures for leaf nodes alg2 line 11 k = 1; 
       
                offset = child * feature_column; 
                for (int x = 0; x < Wf1_dim1; x++)
                {
                    ValueT val = 0;
                    for (int f =0; f < feature_column; f ++)
                        val += features[offset + f] 
                            * W_f_1[f * Wf1_dim1 + x];
                    child_temp[x] = val;
                } // got 1st half of h_B1^1

                for (int x = 0; x < Wa1_dim1; x++)
                {   
                    SizeT val = 0;
                    for (int f =0; f < feature_column; f ++)
                        val += sums[i * feature_column + f] * W_a_1[f * Wa1_dim1 + x];
                    child_temp[x + Wf1_dim1] = val;
                } // got 2nd half of h_B1^1 
      
                // activation and L-2 normalize 
                double L2_child_temp = 0.0;
                for (int x =0; x < Wf2_dim0; x++)
                {
                    ValueT val = child_temp[x];
                    if (val < 0) // relu()
                        val = 0;
                    L2_child_temp += val * val;
                    child_temp[x] = val;
                }  //finished relu
                L2_child_temp = 1.0 / sqrt(L2_child_temp);
                for (int x =0; x < Wf2_dim0; x++)
                {
                    //child_temp[idx_0] = child_temp[idx_0] /sqrt (L2_child_temp);
                    child_temp[x] *= L2_child_temp;
                }//finished L-2 norm, got h_B1^1, algo2 line13

                offset = i / num_children_per_source * Wf2_dim0;
                // add the h_B1^1 to children_temp, also agg it
                for (int x =0; x < Wf2_dim0; x ++ ) //205
                {
                    atomicAdd(children_temp + (offset + x), 
                        child_temp[x] / num_children_per_source);
                }//finished agg (h_B1^1)
                
                offset = i / num_children_per_source * feature_column;
                for (int f = 0; f < feature_column; f++)
                {
                    atomicAdd(sums_child_feat + offset + f, 
                        features[child * feature_column + f] / num_children_per_source); 
                    //merge 220 and 226
                }
                // end of for each child
            }, num_children, util::DEVICE, stream));

        GUARD_CU(data_slice.source_temp.ForAll(
            [feature_column, features, source_start, 
            W_f_1, Wf1_dim1, children_temp,
            sums_child_feat, W_a_1, Wa1_dim1,
            W_f_2, Wf2_dim1, Wf2_dim0, W_a_2, Wa2_dim1, Wa2_dim0,
            source_result, result_column] 
            __host__ __device__ (ValueT *source_temp_, const SizeT &i)
            {
                ValueT  *source_temp = source_temp_ + i * Wf2_dim0;
                VertexT source = source_start + i;
                SizeT offset = source * feature_column;
                // get ebedding vector for child node (h_{B2}^{1}) alg2 line 12            
                for (int x = 0; x < Wf1_dim1; x++)
                {
                    ValueT val = 0;
                    for (int f =0; f < feature_column; f++)
                        val += features[offset + f] * W_f_1[f * Wf1_dim1 + x];
                    source_temp[x] = val;
                } // got 1st half of h_B2^1

                offset = i * feature_column;
                for (int x = 0; x < Wa1_dim1; x++)
                {
                    ValueT val = 0;
                    for (int f=0; f < feature_column; f++)
                        val += sums_child_feat[offset + f] * W_a_1[f * Wa1_dim1 + x];
                    source_temp[Wf1_dim1 + x] = val;
                } // got 2nd half of h_B2^1         

                double L2_source_temp = 0.0;
                for (int x =0; x < Wf2_dim0; x++)
                {
                    ValueT val = source_temp[x];
                    if (val < 0)
                        val = 0; // relu()
                    L2_source_temp += val * val;
                    source_temp[x] = val;
                } //finished relu
                L2_source_temp = 1.0 / sqrt(L2_source_temp);
                for (int x =0; x < Wf2_dim0; x++)
                {
                    //source_temp[idx_0] = source_temp[idx_0] /sqrt (L2_source_temp);
                    source_temp[x] *= L2_source_temp;
                }//finished L-2 norm for source temp

                //////////////////////////////////////////////////////////////////////////////////////
                // get h_B2^2 k =2.
                offset = i * result_column;
                for (int x = 0; x < Wf2_dim1; x++)
                {
                    ValueT val = 0; //source_result[offset + x];
                    //printf ("source_r1_0:%f", source_result[idx_0] );
                    for (int y =0; y < Wf2_dim0; y ++)
                        val += source_temp[y] * W_f_2[y * Wf2_dim1 + x];
                    source_result[offset + x] = val;
                    //printf ("source_r1:%f", source_result[idx_0] );
                } // got 1st half of h_B2^2

                for (int x = 0; x < Wa2_dim1; x++)
                {
                    //printf ("source_r2_0:%f", source_result[idx_0] );
                    ValueT val = 0; //source_result[offset + x];
                    for (int y = 0; y < Wa2_dim0; y ++)
                        val += children_temp[i * Wa2_dim0 + y] * W_a_2[y * Wa2_dim1 + x];
                    source_result[offset + Wf2_dim1 + x] = val;
                } // got 2nd half of h_B2^2
                
                double L2_source_result = 0.0;
                for (int x =0; x < result_column; x ++ )
                {
                    ValueT val = source_result[offset + x];
                    if (val < 0) // relu()
                        val = 0;
                    L2_source_result += val * val;
                    source_result[offset + x] = val;
                } //finished relu
                L2_source_result = 1.0 / sqrt(L2_source_result);
                for (int x =0; x < result_column; x ++ )
                {
                    source_result[offset + x] *= L2_source_result;
                    //printf ("source_r:%f", source_result[idx_0] );
                    //printf ("ch_t:%f", children_temp[idx_0]);
                }//finished L-2 norm for source result   
                 
           }, num_sources, util::DEVICE, stream));
       
        GUARD_CU2(cudaMemcpyAsync(
            data_slice.host_source_result + (source_start * result_column),
            source_result.GetPointer(util::DEVICE),
            num_sources * result_column * sizeof(ValueT),
            cudaMemcpyDeviceToHost, stream),
            "source_result D2H copy failed");

        return retval;
    }

    /**
     * @brief Routine to combine received data and local data
     * @tparam NUM_VERTEX_ASSOCIATES Number of data associated with each transmition item, typed VertexT
     * @tparam NUM_VALUE__ASSOCIATES Number of data associated with each transmition item, typed ValueT
     * @param  received_length The numver of transmition items received
     * @param[in] peer_ which peer GPU the data came from
     * \return cudaError_t error message(s), if any
     */
    template <
        int NUM_VERTEX_ASSOCIATES,
        int NUM_VALUE__ASSOCIATES>
    cudaError_t ExpandIncoming(SizeT &received_length, int peer_)
    {
        auto         &data_slice         =   this -> enactor ->
            problem -> data_slices[this -> gpu_num][0];
        auto         &enactor_slice      =   this -> enactor ->
            enactor_slices[this -> gpu_num * this -> enactor -> num_gpus + peer_];
        //auto iteration = enactor_slice.enactor_stats.iteration;
        //auto         &distances          =   data_slice.distances;
        //auto         &labels             =   data_slice.labels;
        //auto         &preds              =   data_slice.preds;
        //auto          label              =   this -> enactor ->
        //    mgpu_slices[this -> gpu_num].in_iteration[iteration % 2][peer_];

        auto expand_op = [ ]
        __host__ __device__(
            VertexT &key, const SizeT &in_pos,
            VertexT *vertex_associate_ins,
            ValueT  *value__associate_ins) -> bool
        {
            /*
            ValueT in_val  = value__associate_ins[in_pos];
            ValueT old_val = atomicMin(distances + key, in_val);
            if (old_val <= in_val)
                return false;
            if (labels[key] == label)
                return false;
            labels[key] = label;
            if (!preds.isEmpty())
                preds[key] = vertex_associate_ins[in_pos];
            */
            return true;
        };

        cudaError_t retval = BaseIterationLoop:: template ExpandIncomingBase
            <NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            (received_length, peer_, expand_op);
        return retval;
    }

    bool Stop_Condition(int gpu_num = 0)
    {
        int num_gpus = this -> enactor -> num_gpus;
        auto &enactor_slices = this -> enactor -> enactor_slices;

        for (int gpu = 0; gpu < num_gpus * num_gpus; gpu++)
        {   
            auto &retval = enactor_slices[gpu].enactor_stats.retval;
            if (retval == cudaSuccess) continue;
            printf("(CUDA error %d @ GPU %d: %s\n",
                retval, gpu % num_gpus, cudaGetErrorString(retval));
            fflush(stdout);
            return true;
        }   
    
        auto         &data_slice         =   this -> enactor ->
            problem -> data_slices[this -> gpu_num][0]; 
        auto         &enactor_slice      =   this -> enactor ->
            enactor_slices[this -> gpu_num * this -> enactor -> num_gpus];
        if (enactor_slice.enactor_stats.iteration * data_slice.batch_size
            < data_slice.sub_graph -> nodes)
            return false;
        return true;
    }

    cudaError_t Compute_OutputLength(int peer_)
    {
        return cudaSuccess;
    }

    cudaError_t Check_Queue_Size(int peer_)
    {
        return cudaSuccess;
    }
}; // end of SSSPIteration

/**
 * @brief SSSP enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <
    typename _Problem,
    util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
    unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor :
    public EnactorBase<
        typename _Problem::GraphT,
        typename _Problem::VertexT,
        typename _Problem::ValueT,
        ARRAY_FLAG, cudaHostRegisterFlag>
{
public:
    // Definations
    typedef _Problem                   Problem ;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexT  VertexT ;
    typedef typename Problem::ValueT   ValueT  ;
    typedef typename Problem::GraphT   GraphT  ;
    typedef typename Problem::LabelT   LabelT  ;
    typedef EnactorBase<GraphT , LabelT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
        BaseEnactor;
    typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag>
        EnactorT;
    typedef SAGEIterationLoop<EnactorT> IterationT;

    // Members
    Problem     *problem   ;
    IterationT  *iterations;

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief SSSPEnactor constructor
     */
    Enactor() :
        BaseEnactor("sage"),
        problem    (NULL  )
    {
        this -> max_num_vertex_associates = 0;
        this -> max_num_value__associates = 1;
    }

    /**
     * @brief SSSPEnactor destructor
     */
    virtual ~Enactor()
    {
        //Release();
    }

    /*
     * @brief Releasing allocated memory space
     * @param target The location to release memory from
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Release(util::Location target = util::LOCATION_ALL)
    {
        cudaError_t retval = cudaSuccess;
        GUARD_CU(BaseEnactor::Release(target));
        delete []iterations; iterations = NULL;
        problem = NULL;
        return retval;
    }

    /**
     * @brief Initialize the enactor.
     * @param[in] problem The problem object.
     * @param[in] target Target location of data
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Init(
        Problem          &problem,
        util::Location    target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;
        this->problem = &problem;

        GUARD_CU(BaseEnactor::Init(
            problem, Enactor_None, 0, NULL, target, false));
        for (int gpu = 0; gpu < this -> num_gpus; gpu ++)
        {
            GUARD_CU(util::SetDevice(this -> gpu_idx[gpu]));
            auto &enactor_slice
                = this -> enactor_slices[gpu * this -> num_gpus + 0];
            auto &graph = problem.sub_graphs[gpu];
            GUARD_CU(enactor_slice.frontier.Allocate(
                graph.nodes, graph.edges, this -> queue_factors));
        }

        iterations = new IterationT[this -> num_gpus];
        for (int gpu = 0; gpu < this -> num_gpus; gpu ++)
        {
            GUARD_CU(iterations[gpu].Init(this, gpu));
        }

        GUARD_CU(this -> Init_Threads(this,
            (CUT_THREADROUTINE)&(GunrockThread<EnactorT>)));
        return retval;
    }

    /**
     * @brief Reset enactor
     * @param[in] src Source node to start primitive.
     * @param[in] target Target location of data
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Reset(VertexT src, util::Location target = util::DEVICE)
    {
        typedef typename GraphT::GpT GpT;
        cudaError_t retval = cudaSuccess;
        GUARD_CU(BaseEnactor::Reset(target));
        for (int gpu = 0; gpu < this->num_gpus; gpu++)
        { 
            /*
            if ((this->num_gpus == 1) ||
                (gpu == this->problem->org_graph->GpT::partition_table[src]))
            {
                this -> thread_slices[gpu].init_size = 1;
                for (int peer_ = 0; peer_ < this -> num_gpus; peer_++)
                {
                    auto &frontier = this ->
                        enactor_slices[gpu * this -> num_gpus + peer_].frontier;
                    frontier.queue_length = (peer_ == 0) ? 1 : 0;
                    if (peer_ == 0)
                    {
                        GUARD_CU(frontier.V_Q() -> ForEach(
                            [src]__host__ __device__ (VertexT &v)
                        
                            v = src;
                        }
                    }
                }
            }

            else { */
                this -> thread_slices[gpu].init_size = 0;
                for (int peer_ = 0; peer_ < this -> num_gpus; peer_++)
                {
                    this -> enactor_slices[gpu * this -> num_gpus + peer_]
                        .frontier.queue_length = 0;
                }
        //    }
        }
        GUARD_CU(BaseEnactor::Sync());
        return retval;
    }

    /**
      * @brief one run of sage, to be called within GunrockThread
      * @param thread_data Data for the CPU thread
      * \return cudaError_t error message(s), if any
      */
    cudaError_t Run(ThreadSlice &thread_data)
    {
        gunrock::app::Iteration_Loop<
            0, 1, IterationT>(
            thread_data, iterations[thread_data.thread_num]);
        return cudaSuccess;
    }

    /**
     * @brief Enacts a SSSP computing on the specified graph.
     * @param[in] src Source node to start primitive.
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Enact( )
    {
        cudaError_t  retval     = cudaSuccess;
        GUARD_CU(this -> Run_Threads(this));
        util::PrintMsg("GPU SAGE Done.", this -> flag & Debug);
        return retval;
    }

    /** @} */
};

} // namespace sage
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
