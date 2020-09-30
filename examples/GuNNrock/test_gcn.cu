// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_pr.cugraphsum
 *
 * @brief Simple test driver program for PageRank.
 */

 #include <gunrock/app/GuNNrock/gcn_app.cu>
 #include <gunrock/app/test_base.cuh>
 #include <gunrock/app/app_base.cuh>
 #include <cstdio>
 #include <iostream>
 
 using namespace gunrock;
 
 /******************************************************************************
  * Main
  ******************************************************************************/
 
 template <typename T>
 cudaError_t load_graph(util::Parameters &p, T &g) {
   typedef typename T::CsrT CsrT;
   auto retval = cudaSuccess;
   static std::vector<typename T::SizeT> r_offsets, col_offsets;
   r_offsets.push_back(0);
   typename T::SizeT node = 0;
   std::ifstream graph_file(p.Get<std::string>("graph_file"));
   while(true) {
     std::string line;
     getline(graph_file, line);
     if (graph_file.eof()) break;
 
     // Implicit self connection
     col_offsets.push_back(node);
     r_offsets.push_back(r_offsets.back() + 1);
     node++;
 
     std::istringstream ss(line);
     while (true) {
       typename T::SizeT neighbor;
       ss >> neighbor;
       if (ss.fail()) break;
       col_offsets.push_back(neighbor);
         r_offsets.back() += 1;
     }
   }
 //  std::cout << r_offsets.back() << ", ";
   g.CsrT::Allocate(node, col_offsets.size(), gunrock::util::HOST);
   g.CsrT::row_offsets.SetPointer(r_offsets.data(), r_offsets.size(), gunrock::util::HOST);
   g.CsrT::column_indices.SetPointer(col_offsets.data(), col_offsets.size(), gunrock::util::HOST);
 //  g.row_offsets.Print();
   g.nodes = node;
   g.edges = col_offsets.size();
   GUARD_CU(graphio::LoadGraph(p, g))
 
   return retval;
 }
 
 /**
  * @brief Enclosure to the main function
  */
 struct main_struct {
   /**
    * @brief the actual main function, after type switching
    * @tparam VertexT    Type of vertex identifier
    * @tparam SizeT      Type of graph size, i.e. type of edge identifier
    * @tparam ValueT     Type of edge values
    * @param  parameters Command line parameters
    * @param  v,s,val    Place holders for type deduction
    * \return cudaError_t error message(s), if any
    */
   template <typename VertexT=int,  // Use int as the vertex identifier
             typename SizeT=int,    // Use int as the graph size type
             typename ValueT=double>   // Use int as the value type
   cudaError_t
   operator()(util::Parameters &parameters, VertexT v, SizeT s, ValueT val) {
     typedef typename app::TestGraph<VertexT, SizeT, ValueT, graph::HAS_CSR>
         GraphT;
     // typedef typename GraphT::CooT CooT;
 
     cudaError_t retval = cudaSuccess;
     bool quick = parameters.Get<bool>("quick");
     bool quiet = parameters.Get<bool>("quiet");
 
     util::CpuTimer cpu_timer;
     GraphT graph;
 
     cpu_timer.Start();
     load_graph(parameters, graph);
 //    graph.row_offsets.Print();
 //    graph.column_indices.Print();
     cpu_timer.Stop();
     parameters.Set("load-time", cpu_timer.ElapsedMillis());
 
     gcn(parameters, graph);
     return retval;
   }
 };
 
 int main(int argc, char **argv) {
   cudaError_t retval = cudaSuccess;
   util::Parameters parameters("test graphsum");
   GUARD_CU(graphio::UseParameters(parameters));
   GUARD_CU(app::gcn::UseParameters(parameters));
   GUARD_CU(app::UseParameters_test(parameters));
   GUARD_CU(parameters.Parse_CommandLine(argc, argv));
   if (parameters.Get<bool>("help")) {
     parameters.Print_Help();
     return cudaSuccess;
   }
   GUARD_CU(parameters.Set("graph-type", "by-pass"))
   GUARD_CU(parameters.Set("undirected", {0}))
   GUARD_CU(parameters.Check_Required());
 
   return app::Switch_Types<app::VERTEXT_U32B |  // app::VERTEXT_U64B |
                            app::SIZET_U32B |    // app::SIZET_U64B |
                            app::VALUET_F64B |   // app::VALUET_F64B |
                            app::DIRECTED | app::UNDIRECTED>(parameters,
                                                             main_struct());
 }
 
 // Leave this at the end of the file
 // Local Variables:
 // mode:c++
 // c-file-style: "NVIDIA"
 // End: