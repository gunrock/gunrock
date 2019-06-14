// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * market.cuh
 *
 * @brief MARKET Graph Construction Routines
 */

#pragma once

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <libgen.h>
#include <iostream>

#include <gunrock/util/parameters.h>
#include <gunrock/graph/coo.cuh>
//#include <gunrock/graphio/utils.cuh>

namespace gunrock {
namespace graphio {
namespace market {

typedef std::map<std::string, std::string> MetaData;

/**
 * @brief Reads a MARKET graph from an input-stream into a CSR sparse format
 *
 * Here is an example of the matrix market format
 * +----------------------------------------------+
 * |%%MatrixMarket matrix coordinate real general | <--- header line
 * |%                                             | <--+
 * |% comments                                    |    |-- 0 or more comment
 * lines
 * |%                                             | <--+
 * |  M N L                                       | <--- rows, columns, entries
 * |  I1 J1 A(I1, J1)                             | <--+
 * |  I2 J2 A(I2, J2)                             |    |
 * |  I3 J3 A(I3, J3)                             |    |-- L lines
 * |     . . .                                    |    |
 * |  IL JL A(IL, JL)                             | <--+
 * +----------------------------------------------+
 *
 * Indices are 1-based i.2. A(1,1) is the first element.
 *
 * @param[in] f_in          Input MARKET graph file.
 * @param[in] output_file   Output file name for binary i/o.
 * @param[in] csr_graph     Csr graph object to store the graph data.
 * @param[in] undirected    Is the graph undirected or not?
 * @param[in] reversed      Whether or not the graph is inversed.
 *
 * \return If there is any File I/O error along the way.
 */
template <typename GraphT>
cudaError_t ReadMarketStream(util::Parameters &parameters, GraphT &graph,
                             MetaData &meta_data,
                             std::string graph_prefix = "") {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::EdgePairT EdgePairT;
  typedef typename GraphT::CooT CooT;

  cudaError_t retval = cudaSuccess;
  bool quiet = parameters.Get<bool>("quiet");
  FILE *f_in = NULL;
  std::string filename =
      parameters.Get<std::string>(graph_prefix + "graph-file");
  if (filename == "") {  // Read from stdin
    util::PrintMsg("  Reading from stdin:", !quiet);
    f_in = stdin;
  }

  else {  // Read from file
    f_in = fopen(filename.c_str(), "r");
    if (f_in == NULL) {
      return util::GRError(cudaErrorUnknown, "Unable to open file " + filename,
                           __FILE__, __LINE__);
    }
    util::PrintMsg("  Reading from " + filename + ":", !quiet);
  }

  // bool undirected             = parameters.Get<bool>(
  //    graph_prefix + "undirected");
  // bool random_edge_values = parameters.Get<bool>(
  //    graph_prefix + "random-edge-values");
  // ValueT edge_value_min       = parameters.Get<ValueT>(
  //    graph_prefix + "edge-value-min");
  // ValueT edge_value_range     = parameters.Get<ValueT>(
  //    graph_prefix + "edge-value-range");
  // bool vertex_start_from_zero = parameters.Get<bool>(
  //    graph_prefix + "vertex-start-from-zero");
  // long edge_value_seed        = parameters.Get<long>(
  //    graph_prefix + "edge-value-seed");
  // if (parameters.UseDefault(graph_prefix + "edge-value-seed"))
  //    edge_value_seed = time(NULL);
  // bool read_from_binary       = parameters.Get<bool>(
  //    graph_prefix + "read-from-binary");
  // bool store_to_binary        = parameters.Get<bool>(
  //    graph_prefix + "store-to-binary");
  // std::string binary_prefix   =
  //    parameters.UseDefault(graph_prefix + "binary-prefix") ?

  auto &edge_pairs = graph.CooT::edge_pairs;
  SizeT edges_read = util::PreDefinedValues<SizeT>::InvalidValue;  //-1;
  SizeT nodes = 0;
  SizeT edges = 0;
  // util::Array1D<SizeT, EdgePairT> temp_edge_pairs;
  // temp_edge_pairs.SetName("graphio::market::ReadMarketStream::temp_edge_pairs");
  // EdgeTupleType *coo = NULL; // read in COO format
  bool got_edge_values = false;
  bool symmetric = false;  // whether the graph is undirected
  bool skew =
      false;  // whether edge values are the inverse for symmetric matrices
  bool array = false;  // whether the mtx file is in dense array format

  time_t mark0 = time(NULL);
  util::PrintMsg(
      "  Parsing MARKET COO format"  //+ (
                                     //(GraphT::FLAG & graph::HAS_EDGE_VALUES) ?
                                     //" edge-value-seed = " +
                                     //std::to_string(edge_value_seed) : "")
      ,
      !quiet, false);
  // if (GraphT::FLAG & graph::HAS_EDGE_VALUES)
  //    srand(edge_value_seed);

  char line[1024];

  // bool ordered_rows = true;
  GUARD_CU(graph.CooT::Release());

  while (true) {
    if (fscanf(f_in, "%[^\n]\n", line) <= 0) {
      break;
    }

    if (line[0] == '%') {  // Comment
      if (strlen(line) >= 2 && line[1] == '%') {
        symmetric = (strstr(line, "symmetric") != NULL);
        skew = (strstr(line, "skew") != NULL);
        array = (strstr(line, "array") != NULL);
      }
    }

    else if (!util::isValid(edges_read))  //(edges_read == -1)
    {                                     // Problem description
      long long ll_nodes_x, ll_nodes_y, ll_edges;
      int items_scanned =
          sscanf(line, "%lld %lld %lld", &ll_nodes_x, &ll_nodes_y, &ll_edges);

      if (array && items_scanned == 2) {
        ll_edges = ll_nodes_x * ll_nodes_y;
      }

      else if (!array && items_scanned == 3) {
        if (ll_nodes_x != ll_nodes_y) {
          return util::GRError(cudaErrorUnknown,
                               "Error parsing MARKET graph: not square (" +
                                   std::to_string(ll_nodes_x) + ", " +
                                   std::to_string(ll_nodes_y) + ")",
                               __FILE__, __LINE__);
        }
        // if (undirected) ll_edges *=2;
      }

      else {
        return util::GRError(cudaErrorUnknown,
                             "Error parsing MARKET graph:"
                             " invalid problem description.",
                             __FILE__, __LINE__);
      }

      nodes = ll_nodes_x;
      edges = ll_edges;

      util::PrintMsg(" (" + std::to_string(ll_nodes_x) + " nodes, " +
                         std::to_string(ll_edges) + " directed edges)... ",
                     !quiet, false);

      // Allocate coo graph
      GUARD_CU(graph.CooT::Allocate(nodes
                                    //+ ((vertex_start_from_zero) ? 0 : 1)
                                    ,
                                    edges, util::HOST));

      edges_read = 0;
    }

    else {  // Edge description (v -> w)
      if (edge_pairs.GetPointer(util::HOST) == NULL) {
        return util::GRError(cudaErrorUnknown,
                             "Error parsing MARKET graph: "
                             "invalid format",
                             __FILE__, __LINE__);
      }
      if (edges_read >= edges) {
        GUARD_CU(graph.CooT::Release());
        return util::GRError(cudaErrorUnknown,
                             "Error parsing MARKET graph: "
                             "encountered more than " +
                                 std::to_string(edges) + " edges",
                             __FILE__, __LINE__);
      }

      long long ll_row, ll_col;
      ValueT ll_value;  // used for parse float / double
      double lf_value;  // used to sscanf value variable types
      int num_input;
      if (GraphT::FLAG & graph::HAS_EDGE_VALUES) {
        num_input = sscanf(line, "%lld %lld %lf", &ll_row, &ll_col, &lf_value);

        if (array && (num_input == 1)) {
          ll_value = ll_row;
          ll_col = edges_read / nodes;
          ll_row = edges_read - nodes * ll_col;
          // printf("%f\n", ll_value);
        }

        else if (array || num_input < 2) {
          GUARD_CU(graph.CooT::Release());
          return util::GRError(cudaErrorUnknown,
                               "Error parsing MARKET graph: "
                               "badly formed edge",
                               __FILE__, __LINE__);
        }

        else if (num_input == 2) {
          // if (random_edge_values)
          //{
          //    auto x = rand();
          //    double int_x = 0;
          //    std::modf(x * 1.0 / edge_value_range, &int_x);
          //    ll_value = x - int_x * edge_value_range;
          //    ll_value += edge_value_min;
          //}
          // else
          //{
          ll_value = 1;
          //}
        } else if (num_input > 2) {
          if (typeid(ValueT) == typeid(float) ||
              typeid(ValueT) == typeid(double) ||
              typeid(ValueT) == typeid(long double))
            ll_value = (ValueT)lf_value;
          else
            ll_value = (ValueT)(lf_value + 1e-10);
          got_edge_values = true;
        }
      } else {
        num_input = sscanf(line, "%lld %lld", &ll_row, &ll_col);

        if (array && (num_input == 1)) {
          ll_value = ll_row;
          ll_col = edges_read / nodes;
          ll_row = edges_read - nodes * ll_col;
        }

        else if (array || (num_input != 2)) {
          GUARD_CU(graph.CooT::Release());
          return util::GRError(cudaErrorUnknown,
                               "Error parsing MARKET graph: "
                               "badly formed edge",
                               __FILE__, __LINE__);
        }
      }

      edge_pairs[edges_read].x = ll_row;  // zero-based array
      edge_pairs[edges_read].y = ll_col;  // zero-based array
      // ordered_rows = false;

      if (GraphT::FLAG & graph::HAS_EDGE_VALUES) {
        graph.CooT::edge_values[edges_read] = ll_value;  // * (skew ? -1 : 1);
      }

      edges_read++;

      /*if (undirected)
      {
          // Go ahead and insert reverse edge
          edge_pairs[edges_read].x = ll_col;       // zero-based array
          edge_pairs[edges_read].y = ll_row;       // zero-based array

          if (GraphT::FLAG & graph::HAS_EDGE_VALUES)
          {
              graph.CooT::edge_values[edges_read] = ll_value * (skew ? -1 : 1);
          }

          //ordered_rows = false;
          edges_read++;
      }*/
    }
  }

  if (edge_pairs.GetPointer(util::HOST) == NULL) {
    return util::GRError(cudaErrorUnknown, "No graph found", __FILE__,
                         __LINE__);
  }

  if (edges_read != edges) {
    GUARD_CU(graph.CooT::Release());
    return util::GRError(cudaErrorUnknown,
                         "Error parsing MARKET graph: "
                         "only " +
                             std::to_string(edges_read) + "/" +
                             std::to_string(edges) + " edges read",
                         __FILE__, __LINE__);
  }

  // if (vertex_start_from_zero)
  //{
  //    GUARD_CU(edge_pairs.ForEach(
  //        []__host__ __device__ (EdgePairT &edge_pair){
  //            edge_pair.x -= 1;
  //            edge_pair.y -= 1;
  //        }, edges, util::HOST));
  //}

  time_t mark1 = time(NULL);
  util::PrintMsg("  Done (" + std::to_string(mark1 - mark0) + " s).", !quiet);

  if (filename == "") {
    fclose(f_in);
  }

  meta_data["symmetric"] = symmetric ? "true" : "false";
  meta_data["array"] = array ? "true" : "false";
  meta_data["skew"] = skew ? "true" : "false";
  meta_data["got_edge_values"] = got_edge_values ? "true" : "false";
  meta_data["num_edges"] = std::to_string(edges);
  meta_data["num_vertices"] = std::to_string(nodes);
  return retval;
}

template <typename GraphT>
cudaError_t ReadBinary(util::Parameters &parameters, GraphT &graph,
                       MetaData &meta_data, std::string graph_prefix = "",
                       bool quiet = false) {
  typedef typename GraphT::CooT CooT;
  cudaError_t retval = cudaSuccess;

  std::string filename = "";
  if (parameters.UseDefault(graph_prefix + "binary-prefix"))
    filename = parameters.Get<std::string>(graph_prefix + "graph-file");
  else
    filename = parameters.Get<std::string>(graph_prefix + "binary-prefix");

  util::PrintMsg("  Reading edge lists from " + filename + ".coo_edge_pairs",
                 !quiet);
  retval =
      graph.CooT::edge_pairs.ReadBinary(filename + ".coo_edge_pairs", true);
  if (retval == cudaErrorInvalidValue)
    return retval;
  else
    GUARD_CU(retval);

  if ((GraphT::FLAG & graph::HAS_EDGE_VALUES) != 0 &&
      meta_data["got_edge_values"] == "true") {
    util::PrintMsg(
        "  Reading edge values from " + filename + ".coo_edge_values", !quiet);
    retval =
        graph.CooT::edge_values.ReadBinary(filename + ".coo_edge_values", true);
    if (retval == cudaErrorInvalidValue)
      return retval;
    else
      GUARD_CU(retval);
  }

  if (GraphT::FLAG & graph::HAS_NODE_VALUES) {
    util::PrintMsg(
        "  Reading node values from " + filename + ".coo_node_values", !quiet);
    retval =
        graph.CooT::node_values.ReadBinary(filename + ".coo_node_values", true);
    if (retval == cudaErrorInvalidValue)
      return retval;
    else
      GUARD_CU(retval);
  }
  return retval;
}

template <typename GraphT>
cudaError_t WriteBinary(util::Parameters &parameters, GraphT &graph,
                        MetaData &meta_data, std::string graph_prefix = "") {
  typedef typename GraphT::CooT CooT;
  cudaError_t retval = cudaSuccess;
  bool quiet = parameters.Get<bool>("quiet");

  std::string filename = "";
  if (parameters.UseDefault(graph_prefix + "binary-prefix"))
    filename = parameters.Get<std::string>(graph_prefix + "graph-file");
  else
    filename = parameters.Get<std::string>(graph_prefix + "binary-prefix");

  util::PrintMsg(
      "  Writting edge pairs in binary into " + filename + ".coo_edge_pairs",
      !quiet);
  retval =
      graph.CooT::edge_pairs.WriteBinary(filename + ".coo_edge_pairs", true);
  if (retval == cudaErrorInvalidValue)
    return retval;
  else
    GUARD_CU(retval);

  if ((GraphT::FLAG & graph::HAS_EDGE_VALUES) != 0 &&
      meta_data["got_edge_values"] == "true") {
    util::PrintMsg("  Writting edge values in binary into " + filename +
                       ".coo_edge_values",
                   !quiet);
    retval = graph.CooT::edge_values.WriteBinary(filename + ".coo_edge_values",
                                                 true);
    if (retval == cudaErrorInvalidValue)
      return retval;
    else
      GUARD_CU(retval);
  }

  if (GraphT::FLAG & graph::HAS_NODE_VALUES) {
    util::PrintMsg("  Writting node values in binary into " + filename +
                       ".coo_node_values",
                   !quiet);
    retval = graph.CooT::node_values.WriteBinary(filename + ".coo_node_values",
                                                 true);
    if (retval == cudaErrorInvalidValue)
      return retval;
    else
      GUARD_CU(retval);
  }
  return retval;
}

/**
 * \defgroup Public Interface
 * @{
 */

cudaError_t UseParameters(util::Parameters &parameters,
                          std::string graph_prefix = "") {
  cudaError_t retval = cudaSuccess;

  return retval;
}

cudaError_t WriteMeta(util::Parameters &parameters, std::string filename,
                      MetaData &meta_data) {
  cudaError_t retval = cudaSuccess;
  std::ofstream fout;
  bool quiet = parameters.Get<bool>("quiet");
  fout.open((filename + ".meta"));
  if (!fout.is_open()) return cudaErrorUnknown;
  util::PrintMsg("  Writing meta data into " + filename + ".meta", !quiet);

  for (auto it : meta_data) {
    fout << it.first << " " << it.second << std::endl;
  }
  fout.close();
  return retval;
}

cudaError_t ReadMeta(util::Parameters &parameters, std::string filename,
                     MetaData &meta_data) {
  cudaError_t retval = cudaSuccess;
  std::ifstream fin;
  bool quiet = parameters.Get<bool>("quiet");
  fin.open((filename + ".meta"));
  if (!fin.is_open()) return cudaErrorUnknown;
  util::PrintMsg("  Reading meta data from " + filename + ".meta", !quiet);

  while (!fin.eof()) {
    std::string line;
    std::getline(fin, line);
    if (fin.eof()) break;
    auto pos = line.find(" ");
    auto name = line.substr(0, pos);
    auto value = line.substr(pos + 1, line.length());
    meta_data[name] = value;
    // util::PrintMsg(" " + name + " <- " + value);
  }
  fin.close();
  return retval;
}

template <typename GraphT>
cudaError_t Read(util::Parameters &parameters, GraphT &graph,
                 std::string graph_prefix = "") {
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::ValueT ValueT;
  cudaError_t retval = cudaSuccess;
  bool quiet = parameters.Get<bool>("quiet");
  MetaData meta_data;
  util::CpuTimer timer;
  timer.Start();

  util::PrintMsg("Loading Matrix-market coordinate-formatted " + graph_prefix +
                     "graph ...",
                 !quiet);

  std::string filename =
      parameters.Get<std::string>(graph_prefix + "graph-file");

  bool read_done = false;
  retval = ReadMeta(parameters, filename, meta_data);

  if (parameters.UseDefault("dataset")) {
    std::string dir, file, extension;
    util::SeperateFileName(filename, dir, file, extension);
    parameters.Set("dataset", file);
  }

  if (retval == cudaSuccess) {
    auto num_vertices = util::strtoT<SizeT>(meta_data["num_vertices"]);
    auto num_edges = util::strtoT<SizeT>(meta_data["num_edges"]);
    GUARD_CU(graph.Allocate(num_vertices, num_edges, util::HOST));
    retval = ReadBinary(parameters, graph, meta_data, graph_prefix, quiet);
    if (retval == cudaSuccess)
      read_done = true;
    else {
      GUARD_CU(graph.Release());
    }
  }

  if (!read_done) {
    std::ifstream fp(filename.c_str());
    if (filename == "" || !fp.is_open()) {
      return util::GRError(cudaErrorUnknown,
                           "Input graph file " + filename + " does not exist.",
                           __FILE__, __LINE__);
    }
    fp.close();
#if 0
        if (parameters.UseDefault("dataset"))
        {
            std::string dir, file, extension;
            util::SeperateFileName(filename, dir, file, extension);
            parameters.Set("dataset", file);
        }
#endif
    GUARD_CU(ReadMarketStream(parameters, graph, meta_data, graph_prefix));

    if (parameters.Get<bool>(graph_prefix + "store-to-binary")) {
      retval = WriteMeta(parameters, filename, meta_data);
      if (retval == cudaErrorUnknown)
        // return cudaSuccess;
        retval = cudaSuccess;
      else if (retval)
        GUARD_CU2(retval, "Writting meta failed.");

      retval = WriteBinary(parameters, graph, meta_data, graph_prefix);
      if (retval == cudaErrorInvalidValue)
        // return cudaSuccess;
        retval = cudaSuccess;
      else if (retval)
        GUARD_CU2(retval, "Writting binary failed");
    }
  }

  if (meta_data["got_edge_values"] != "true" &&
      (GraphT::FLAG & graph::HAS_EDGE_VALUES) != 0) {
    if (parameters.Get<bool>(graph_prefix + "random-edge-values")) {
      GUARD_CU(graph.GenerateEdgeValues(
          parameters.Get<ValueT>(graph_prefix + "edge-value-min"),
          parameters.Get<ValueT>(graph_prefix + "edge-value-range"),
          parameters.Get<long>(graph_prefix + "edge-value-seed"), quiet));
    } else {
      util::PrintMsg(
          "  Assigning 1 to all " + std::to_string(graph.edges) + " edges",
          !quiet);
      GUARD_CU(graph.edge_values.ForEach(
          [] __host__ __device__(ValueT & value) { value = 1; }, graph.edges,
          util::HOST));
    }
  }
  // GUARD_CU(graph.Display());

  if (parameters.Get<bool>(graph_prefix + "vertex-start-from-zero")) {
    util::PrintMsg("  Substracting 1 from node Ids...", !quiet);
    GUARD_CU(graph.edge_pairs.ForEach(
        [] __host__ __device__(typename GraphT::EdgePairT & edge_pair) {
          edge_pair.x -= 1;
          edge_pair.y -= 1;
        },
        graph.edges, util::HOST));
  }

  if (parameters.Get<bool>(graph_prefix + "undirected") ||
      meta_data["symmetric"] == "true") {
    GUARD_CU(graph.EdgeDouble(meta_data["skew"] == "true" ? true : false));
  }

  bool remove_self_loops =
      parameters.Get<bool>(graph_prefix + "remove-self-loops");
  bool remove_duplicate_edges =
      parameters.Get<bool>(graph_prefix + "remove-duplicate-edges");
  if (remove_self_loops && remove_duplicate_edges) {
    GUARD_CU(graph.RemoveSelfLoops_DuplicateEdges(
        gunrock::graph::BY_ROW_ASCENDING, util::HOST, 0, quiet));
  } else if (remove_self_loops) {
    GUARD_CU(graph.RemoveSelfLoops(gunrock::graph::BY_ROW_ASCENDING, util::HOST,
                                   0, quiet));
  } else if (remove_duplicate_edges) {
    GUARD_CU(graph.RemoveDuplicateEdges(gunrock::graph::BY_ROW_ASCENDING,
                                        util::HOST, 0, quiet));
  }

  // GUARD_CU(graph.Display());
  timer.Stop();
  util::PrintMsg("  " + graph_prefix + "graph loaded as COO in " +
                     std::to_string(timer.ElapsedMillis() / 1000.0) + "s.",
                 !quiet);
  return retval;
}

template <typename GraphT, bool COO_SWITCH>
struct CooSwitch {
  static cudaError_t Load(util::Parameters &parameters, GraphT &graph,
                          std::string graph_prefix = "") {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(Read(parameters, graph.coo(), graph_prefix));
    bool quiet = parameters.Get<bool>("quiet");
    GUARD_CU(graph.FromCoo(graph, util::HOST, 0, quiet, true));
    return retval;
  }
};

template <typename GraphT>
struct CooSwitch<GraphT, false> {
  static cudaError_t Load(util::Parameters &parameters, GraphT &graph,
                          std::string graph_prefix = "") {
    typedef graph::Coo<typename GraphT::VertexT, typename GraphT::SizeT,
                       typename GraphT::ValueT, GraphT::FLAG | graph::HAS_COO,
                       GraphT::cudaHostRegisterFlag>
        CooT;
    cudaError_t retval = cudaSuccess;
    CooT coo;
    GUARD_CU(Read(parameters, coo, graph_prefix));
    bool quiet = parameters.Get<bool>("quiet");
    GUARD_CU(graph.FromCoo(coo, util::HOST, 0, quiet, false));
    GUARD_CU(coo.Release());
    return retval;
  }
};

template <typename GraphT>
cudaError_t Load(util::Parameters &parameters, GraphT &graph,
                 std::string graph_prefix = "") {
  return CooSwitch<GraphT, (GraphT::FLAG & graph::HAS_COO) != 0>::Load(
      parameters, graph, graph_prefix);
}

template <typename GraphT>
cudaError_t Write(util::Parameters &parameters, GraphT &graph,
                  std::string graph_prefix = "") {
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CooT CooT;
  typedef typename GraphT::CooT::EdgePairT EdgePairT;
  cudaError_t retval = cudaSuccess;

  bool quiet = parameters.Get<bool>("quiet");
  util::PrintMsg(
      "Saving Matrix-market coordinate-formatted " + graph_prefix + "graph ...",
      !quiet);

  std::string filename =
      parameters.Get<std::string>(graph_prefix + "output-file");

  std::ofstream fout;
  fout.open(filename.c_str());
  if (fout.is_open()) {
    fout << "%%MatrixMarket matrix coordinate pattern";
    if (graph.undirected) fout << " symmetric";
    fout << std::endl;
    fout << graph.nodes << " " << graph.nodes << " " << graph.edges
         << std::endl;
    for (SizeT e = 0; e < graph.edges; e++) {
      EdgePairT &edge_pair = graph.CooT::edge_pairs[e];
      if (graph.undirected && edge_pair.x > edge_pair.y) continue;
      fout << edge_pair.x << " " << edge_pair.y;
      if (GraphT::FLAG & graph::HAS_EDGE_VALUES)
        fout << " " << graph.CooT::edge_values[e];
      fout << std::endl;
    }
    fout.close();
  } else {
    return util::GRError(cudaErrorUnknown, "Unable to write file " + filename,
                         __FILE__, __LINE__);
  }
  return retval;
}
/**@}*/

}  // namespace market
}  // namespace graphio
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
