/// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_sage.cu
 *
 * @brief Simple test driver program for single source shortest path.
 */

#include <gunrock/app/sage/sage_app.cu>
#include <gunrock/app/test_base.cuh>

using namespace gunrock;

/******************************************************************************
* Main
******************************************************************************/

/**
 * @brief Enclosure to the main function
 */
struct main_struct
{
    /**
     * @brief the actual main function, after type switching
     * @tparam VertexT    Type of vertex identifier
     * @tparam SizeT      Type of graph size, i.e. type of edge identifier
     * @tparam ValueT     Type of edge values
     * @param  parameters Command line parameters
     * @param  v,s,val    Place holders for type deduction
     * \return cudaError_t error message(s), if any
     */
    template <
        typename VertexT, // Use int as the vertex identifier
        typename SizeT,   // Use int as the graph size type
        typename ValueT>  // Use int as the value type
    cudaError_t operator()(util::Parameters &parameters,
        VertexT v, SizeT s, ValueT val)
    {
        typedef typename app::TestGraph<VertexT, SizeT, ValueT,
            graph::HAS_EDGE_VALUES | graph::HAS_CSR>
            GraphT;
        typedef typename GraphT::CsrT CsrT;

        cudaError_t retval = cudaSuccess;
        util::CpuTimer cpu_timer;
        GraphT graph; // graph we process on

        cpu_timer.Start();
        GUARD_CU(graphio::LoadGraph(parameters, graph));
        // force edge values to be 1, don't enable this unless you really want to
        //for (SizeT e=0; e < graph.edges; e++)
        //    graph.CsrT::edge_values[e] = 1;
        cpu_timer.Stop();
        parameters.Set("load-time", cpu_timer.ElapsedMillis());
        //GUARD_CU(graph.CsrT::edge_values.Print("", 100)); 
        //util::PrintMsg("sizeof(VertexT) = " + std::to_string(sizeof(VertexT))
        //    + ", sizeof(SizeT) = " + std::to_string(sizeof(SizeT))
        //    + ", sizeof(ValueT) = " + std::to_string(sizeof(ValueT)));

        //GUARD_CU(app::Set_Srcs    (parameters, graph));
        //ValueT  **ref_distances = NULL;
        //int num_srcs = 0;
        bool quick = parameters.Get<bool>("quick");
        // compute reference CPU Sage solution for source-distance
        if (!quick)
        {
            bool quiet = parameters.Get<bool>("quiet");
            std::string wf1_file = parameters.Get<std::string>("Wf1"); 
            std::string wa1_file = parameters.Get<std::string>("Wa1");
            std::string wf2_file = parameters.Get<std::string>("Wf2");
            std::string wa2_file = parameters.Get<std::string>("Wf2");
            std::string feature_file = parameters.Get<std::string>("features");
            int Wf1_dim_0 = parameters.Get<int> ("feature-column");//("Wf1-dim0");
            int Wa1_dim_0 = parameters.Get<int> ("feature-column");//("Wa1-dim0");
            int Wf1_dim_1 = parameters.Get<int> ("Wf1-dim1");
            int Wa1_dim_1 = parameters.Get<int> ("Wa1-dim1");
            int Wf2_dim_0 = Wf1_dim_1 + Wa1_dim_1; //parameters.Get<int> ("Wf2-dim0");
            int Wa2_dim_0 = Wf1_dim_1 + Wa1_dim_1; //parameters.Get<int> ("Wa2-dim0");
            int Wf2_dim_1 = parameters.Get<int> ("Wf2-dim1");
            int Wa2_dim_1 = parameters.Get<int> ("Wa2-dim1");
            int num_neigh1 = parameters.Get<int> ("num-children-per-source");
            int num_neigh2 = parameters.Get<int> ("num-leafs-per-child");
            int batch_size = parameters.Get<int> ("batch-size");

            ValueT ** W_f_1 = app::sage::template ReadMatrix <ValueT,SizeT> (wf1_file, Wf1_dim_0,Wf1_dim_1); 
            ValueT ** W_a_1 = app::sage::template ReadMatrix <ValueT,SizeT> (wa1_file, Wa1_dim_0, Wa1_dim_1);
            ValueT ** W_f_2 = app::sage::template ReadMatrix <ValueT,SizeT> (wf2_file, Wf2_dim_0, Wf2_dim_1);
            ValueT ** W_a_2 = app::sage::template ReadMatrix <ValueT,SizeT> (wa2_file, Wa2_dim_0, Wa2_dim_1); 
            ValueT ** features = app::sage::template ReadMatrix<ValueT,SizeT> (feature_file, graph.nodes, Wf1_dim_0);
            //num_srcs = srcs.size();
            //SizeT nodes = graph.nodes;
            //ref_distances = new ValueT*[num_srcs];
           // 
          //      ref_distances[i] = (ValueT*)malloc(sizeof(ValueT) * nodes);
          //      VertexT src = srcs[i];
            util::PrintMsg("__________________________", !quiet);
            float elapsed = app::sage::CPU_Reference(
                graph,batch_size, num_neigh1, num_neigh2, 
                features, W_f_1,W_a_1,W_f_2,W_a_2,  quiet, false);
            util::PrintMsg("--------------------------\n"
                "CPU Reference elapsed: "
                + std::to_string(elapsed) + " ms.", !quiet);  
        }

        std::vector<std::string> switches{"advance-mode", "batch-size"};
        GUARD_CU(app::Switch_Parameters(parameters, graph, switches,
            [](util::Parameters &parameters, GraphT &graph)
            {
                return app::sage::RunTests(parameters, graph);
            }));

       
        return retval;
    }
};

int main(int argc, char** argv)
{
    cudaError_t retval = cudaSuccess;
    util::Parameters parameters("test sage");
    GUARD_CU(graphio::UseParameters(parameters));
    GUARD_CU(app::sage::UseParameters(parameters));
    GUARD_CU(app::UseParameters_test(parameters));
    GUARD_CU(parameters.Parse_CommandLine(argc, argv));
    if (parameters.Get<bool>("help"))
    {
        parameters.Print_Help();
        return cudaSuccess;
    }
    GUARD_CU(parameters.Check_Required());

    return app::Switch_Types<
        app::VERTEXT_U32B | app::VERTEXT_U64B |
        app::SIZET_U32B | app::SIZET_U64B |
        app::VALUET_F32B | app::DIRECTED | app::UNDIRECTED>
        (parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
