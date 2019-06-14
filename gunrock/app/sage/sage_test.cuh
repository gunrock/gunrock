// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sage_test.cu
 *
 * @brief Test related functions for SSSP
 */

#pragma once

namespace gunrock {
namespace app {
namespace sage {

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/

/**
 * @brief Displays the SAGE result.
 * @tparam T Type of values to display
 * @tparam SizeT Type of size counters
 * @param[in] preds Search depth from the source for each node.
 * @param[in] num_nodes Number of nodes in the graph.
 */
template <typename T, typename SizeT>
void DisplaySolution(T *array, SizeT length) {
  if (length > 40) length = 40;

  util::PrintMsg("[", true, false);
  for (SizeT i = 0; i < length; ++i) {
    util::PrintMsg(std::to_string(i) + ":" + std::to_string(array[i]) + " ",
                   true, false);
  }
  util::PrintMsg("]");
}

/******************************************************************************
 * SAGE Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference SSSP implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the distances
 * @param[in]   graph         Input graph
 * @param[out]  distances     Computed distances from the source to each vertex
 * @param[out]  preds         Computed predecessors for each vertex
 * @param[in]   src           The source vertex
 * @param[in]   quiet         Whether to print out anything to stdout
 * @param[in]   mark_preds    Whether to compute predecessor info
 * \return      double        Time taken for the SSSP
 */

template <typename ValueT, typename SizeT>
ValueT **ReadMatrix(std::string filename, SizeT dim0, SizeT dim1) {
  if (filename == "") {
    // util::PrintMsg("random generated file");
    ValueT **matrix = new ValueT *[dim0];
    for (SizeT i = 0; i < dim0; i++) {
      matrix[i] = new ValueT[dim1];
      for (SizeT j = 0; j < dim1; j++) matrix[i][j] = 1.0 * rand() / RAND_MAX;
    }
    return matrix;
  }

  std::FILE *fin = fopen(filename.c_str(), "r");
  if (fin == NULL) {
    util::PrintMsg("Error in reading " + filename);
    return NULL;
  }

  ValueT **matrix = new ValueT *[dim0];
  for (SizeT i = 0; i < dim0; i++) {
    matrix[i] = new ValueT[dim1];
    for (SizeT j = 0; j < dim1; j++) fscanf(fin, "%f", matrix[i] + j);
  }
  fclose(fin);

  return matrix;
}

template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double CPU_Reference(util::Parameters &para, const GraphT &graph,
                     // int       batch_size,
                     // int       num_neigh1,
                     // int       num_neigh2,
                     ValueT **features, ValueT **W_f_1, ValueT **W_a_1,
                     ValueT **W_f_2, ValueT **W_a_2, ValueT *source_embedding,
                     bool quiet) {
  typedef std::mt19937 Engine;
  typedef std::uniform_real_distribution<float> Distribution;

  util::CpuTimer cpu_timer;
  cpu_timer.Start();
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  // typedef std::pair<VertexT, ValueT> PairT;
  // struct GreaterT
  //{
  //    bool operator()(const PairT& lhs, const PairT& rhs)
  //    {
  //        return lhs.second > rhs.second;
  //    }
  //};
  // typedef std::priority_queue<PairT, std::vector<PairT>, GreaterT> PqT;
  // auto &para = this -> parameters;

  int batch_size = para.template Get<int>("batch-size");
  int feature_column = para.template Get<int>("feature-column");
  int num_children_per_source =
      para.template Get<int>("num-children-per-source");
  int Wf1_dim0 = feature_column;
  int Wf1_dim1 = para.template Get<int>("Wf1-dim1");
  int Wa1_dim0 = feature_column;
  int Wa1_dim1 = para.template Get<int>("Wa1-dim1");
  int Wf2_dim0 = Wf1_dim1 + Wa1_dim1;
  int Wf2_dim1 = para.template Get<int>("Wf2-dim1");
  int Wa2_dim0 = Wf1_dim1 + Wa1_dim1;
  int Wa2_dim1 = para.template Get<int>("Wa2-dim1");
  int result_column = Wa2_dim1 + Wf2_dim1;
  int num_leafs_per_child = para.template Get<int>("num-leafs-per-child");
  if (!util::isValid(num_leafs_per_child))
    num_leafs_per_child = num_children_per_source;
  bool debug = para.template Get<bool>("v");
  int num_threads = para.template Get<int>("omp-threads");
  int rand_seed = para.template Get<int>("rand-seed");

  if (!util::isValid(rand_seed)) rand_seed = time(NULL);
  util::PrintMsg("rand-seed = " + std::to_string(rand_seed), !quiet);

  // util::PrintMsg("CPU_Reference entered", !quiet);
  // int num_batch = graph.nodes / batch_size ;
  // int off_site = graph.nodes - num_batch * batch_size ;
  // batch of nodes
  SizeT num_dangling_vertices = 0;
  for (VertexT source_start = 0; source_start < graph.nodes;
       source_start += batch_size) {
    int num_source =
        (source_start + batch_size <= graph.nodes ? batch_size
                                                  : graph.nodes - source_start);

    util::PrintMsg("Processing sources [" + std::to_string(source_start) +
                       ", " + std::to_string(source_start + num_source) + ")",
                   debug);
#pragma omp parallel for num_threads(num_threads)
    for (VertexT source = source_start; source < source_start + num_source;
         source++) {
      Engine engine(rand_seed + source);
      Distribution distribution(0.0, 1.0);

      // store edges between sources and children
      // std::vector <SizeT> edges_source_child;
      // SizeT edges_source_child[num_children_per_source];
      VertexT children[num_children_per_source];
      float children_temp[Wa2_dim0] = {0.0};  // agg(h_B1^1)
      float source_temp[Wf2_dim0] = {0.0};    // h_B2^1
      // float source_result [256] = {0.0}; // h_B2_2, result
      auto source_result =
          source_embedding + ((uint64_t)source) * result_column;
      for (uint64_t i = 0; i < result_column; i++) source_result[i] = 0.0;

      for (int i = 0; i < num_children_per_source; i++) {
        SizeT num_source_neigh = graph.GetNeighborListLength(source);
        if (num_source_neigh == 0) {
          SizeT old_counter = 0;
#pragma omp atomic capture
          {
            old_counter = num_dangling_vertices;
            num_dangling_vertices++;
          }
          util::PrintMsg(
              "Warning: "
              "Vertex " +
                  std::to_string(source) +
                  " has no neighbors. "
                  "GraphSAGE is not designed to run with dangling vertices.",
              !quiet && old_counter == 0);
          children[i] = source;
          continue;
        }

        SizeT offset =
            distribution(engine) * graph.GetNeighborListLength(source);
        // rand() % graph.GetNeighborListLength(source); //
        SizeT pos = graph.GetNeighborListOffset(source) + offset;
        // edges_source_child.push_back (pos);
        children[i] = graph.GetEdgeDest(pos);
      }  // sample child (B1 nodes), save edge list.

      // get each child's h_v^1
      for (int i = 0; i < num_children_per_source; i++) {
        // SizeT pos = edges_source_child[i];
        // VertexT child = graph.GetEdgeDest(pos);
        VertexT child = children[i];
        float sums[feature_column] = {0.0};

        // sample leaf node for each child
        for (int j = 0; j < num_leafs_per_child; j++) {
          SizeT num_child_neigh = graph.GetNeighborListLength(child);
          VertexT leaf = 0;
          if (num_child_neigh == 0) {
            // util::PrintMsg("Warning: "
            //    "Vertex " + std::to_string(child) + " has no neighbors. "
            //    "GraphSAGE is not designed to run with dangling vertices.");
            leaf = child;
          } else {
            SizeT offset2 =
                distribution(engine) * graph.GetNeighborListLength(child);
            // rand() % graph.GetNeighborListLength(child);
            SizeT pos2 = graph.GetNeighborListOffset(child) + offset2;
            leaf = graph.GetEdgeDest(pos2);
          }
          for (int m = 0; m < feature_column; m++) {
            sums[m] += features[leaf][m];
          }
        }  // agg feaures for leaf nodes alg2 line 11 k = 1
        for (int m = 0; m < feature_column; m++) {
          sums[m] = sums[m] / num_leafs_per_child;
        }  // get mean  agg of leaf features.
        // get ebedding vector for child node (h_{B1}^{1}) alg2 line 12
        float child_temp[Wa2_dim0] = {0.0};
        for (int idx_0 = 0; idx_0 < Wf1_dim1; idx_0++) {
          for (int idx_1 = 0; idx_1 < feature_column; idx_1++)
            child_temp[idx_0] += features[child][idx_1] * W_f_1[idx_1][idx_0];
        }  // got 1st half of h_B1^1

        for (int idx_0 = 0; idx_0 < Wa1_dim1; idx_0++) {
          for (int idx_1 = 0; idx_1 < feature_column; idx_1++)
            child_temp[idx_0 + Wf1_dim1] += sums[idx_1] * W_a_1[idx_1][idx_0];
        }  // got 2nd half of h_B!^1

        // activation and L-2 normalize
        auto L2_child_temp = 0.0;
        for (int idx_0 = 0; idx_0 < Wa2_dim0; idx_0++) {
          child_temp[idx_0] =
              child_temp[idx_0] > 0.0 ? child_temp[idx_0] : 0.0;  // relu()
          L2_child_temp += child_temp[idx_0] * child_temp[idx_0];
        }  // finished relu
        for (int idx_0 = 0; idx_0 < Wa2_dim0; idx_0++) {
          child_temp[idx_0] = child_temp[idx_0] / sqrt(L2_child_temp);
        }  // finished L-2 norm, got h_B1^1, algo2 line13

        // add the h_B1^1 to children_temp, also agg it
        for (int idx_0 = 0; idx_0 < Wa2_dim0; idx_0++) {
          children_temp[idx_0] += child_temp[idx_0] / num_children_per_source;
        }  // finished agg (h_B1^1)
      }    // for each child in B1, got h_B1^1

      //////////////////////////////////////////////////////////////////////////////////////
      // get h_B2^1, k =1 ; this time, child is like leaf, and source is like
      // child
      float sums_child_feat[feature_column] = {0.0};  // agg(h_B1^0)
      for (int i = 0; i < num_children_per_source; i++) {
        // SizeT pos = edges_source_child[i];
        // VertexT child = graph.GetEdgeDest(pos);
        VertexT child = children[i];
        for (int m = 0; m < feature_column; m++) {
          sums_child_feat[m] += features[child][m];
        }

      }  // for each child
      for (int m = 0; m < feature_column; m++) {
        sums_child_feat[m] = sums_child_feat[m] / num_children_per_source;
      }  // got agg(h_B1^0)

      // get ebedding vector for child node (h_{B2}^{1}) alg2 line 12
      for (int idx_0 = 0; idx_0 < Wf1_dim1; idx_0++) {
        for (int idx_1 = 0; idx_1 < feature_column; idx_1++)
          source_temp[idx_0] += features[source][idx_1] * W_f_1[idx_1][idx_0];
      }  // got 1st half of h_B2^1

      for (int idx_0 = 0; idx_0 < Wa1_dim1; idx_0++) {
        for (int idx_1 = 0; idx_1 < feature_column; idx_1++)
          source_temp[Wf1_dim1 + idx_0] +=
              sums_child_feat[idx_1] * W_a_1[idx_1][idx_0];
      }  // got 2nd half of h_B2^1

      auto L2_source_temp = 0.0;
      for (int idx_0 = 0; idx_0 < Wf2_dim0; idx_0++) {
        source_temp[idx_0] =
            source_temp[idx_0] > 0.0 ? source_temp[idx_0] : 0.0;  // relu()
        L2_source_temp += source_temp[idx_0] * source_temp[idx_0];
      }  // finished relu
      for (int idx_0 = 0; idx_0 < Wf2_dim0; idx_0++) {
        source_temp[idx_0] = source_temp[idx_0] / sqrt(L2_source_temp);
        // printf("source_temp,%f", source_temp[idx_0]);
      }  // finished L-2 norm for source temp
      //////////////////////////////////////////////////////////////////////////////////////
      // get h_B2^2 k =2.
      for (uint64_t idx_0 = 0; idx_0 < Wf2_dim1; idx_0++) {
        // printf ("source_r1_0:%f", source_result[idx_0] );
        for (int idx_1 = 0; idx_1 < Wf2_dim0; idx_1++)
          source_result[idx_0] += source_temp[idx_1] * W_f_2[idx_1][idx_0];
        // printf ("source_r1:%f", source_result[idx_0] );
      }  // got 1st half of h_B2^2

      for (uint64_t idx_0 = 0; idx_0 < Wa2_dim1; idx_0++) {
        // printf ("source_r2_0:%f", source_result[idx_0] );
        for (int idx_1 = 0; idx_1 < Wa2_dim0; idx_1++)
          source_result[Wf2_dim1 + idx_0] +=
              children_temp[idx_1] * W_a_2[idx_1][idx_0];
        // printf ("source_r2_1:%f", source_result[idx_0] );

      }  // got 2nd half of h_B2^2
      auto L2_source_result = 0.0;
      for (uint64_t idx_0 = 0; idx_0 < result_column; idx_0++) {
        source_result[idx_0] =
            source_result[idx_0] > 0.0 ? source_result[idx_0] : 0.0;  // relu()
        L2_source_result += source_result[idx_0] * source_result[idx_0];
      }  // finished relu
      for (uint64_t idx_0 = 0; idx_0 < result_column; idx_0++) {
        source_result[idx_0] = source_result[idx_0] / sqrt(L2_source_result);

      }  // finished L-2 norm for source result

    }  // for each source
    // printf ("node %d \n", source);
  }  // for each batch
  // util::PrintMsg("CPU_Reference exited", !quiet);
  cpu_timer.Stop();
  return cpu_timer.ElapsedMillis();
}  // cpu reference

/**
 * @brief Validation of SAGE results
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the distances
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
 * @param[in]  src           The source vertex
 * @param[in]  h_distances   Computed distances from the source to each vertex
 * @param[in]  h_preds       Computed predecessors for each vertex
 * @param[in]  ref_distances Reference distances from the source to each vertex
 * @param[in]  ref_preds     Reference predecessors for each vertex
 * @param[in]  verbose       Whether to output detail comparsions
 * \return     GraphT::SizeT Number of errors
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
typename GraphT::SizeT Validate_Results(util::Parameters &parameters,
                                        GraphT &graph, ValueT *embed_result,
                                        uint64_t result_column,
                                        bool verbose = true) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;

  SizeT num_errors = 0;
  bool quiet = parameters.Get<bool>("quiet");

  util::PrintMsg("Embedding validation: ", !quiet, false);
  // Verify the result
  for (SizeT v = 0; v < graph.nodes; v++) {
    double L2_vec = 0.0;
    uint64_t offset = v * result_column;
    for (uint64_t j = 0; j < result_column; j++) {
      L2_vec += embed_result[offset + j] * embed_result[offset + j];
    }
    if (abs(L2_vec - 1.0) > 0.000001) {
      if (num_errors == 0) {
        util::PrintMsg("FAIL. L2(embedding[" + std::to_string(v) +
                           "]) = " + std::to_string(L2_vec) + ", should be 1",
                       !quiet);
      }
      num_errors += 1;
    }
  }

  if (num_errors == 0)
    util::PrintMsg("PASS", !quiet);
  else
    util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);

  return num_errors;
}

}  // namespace sage
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
