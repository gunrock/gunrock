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
#include <fstream>
#include <string>

#include <gunrock/util/parameters.h>
#include <gunrock/graph/coo.cuh>

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
 * @param[in] parameters    Running parameters
 * @param[in] graph         Graph object to store the graph data.
 * @param[in] meta_data     Meta data that stores graph attributes.
 * @param[in] graph_prefix  Prefix to show graph attributes.
 *
 * \return If there is any File I/O error along the way.
 */
template <typename GraphT>
cudaError_t ReadMarketStream(util::Parameters &parameters,
                             GraphT &graph,
                             MetaData &meta_data,
                             std::string graph_prefix = "") {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::EdgePairT EdgePairT;
  typedef typename GraphT::CooT CooT;

  cudaError_t retval = cudaSuccess;
  bool quiet = parameters.Get<bool>("quiet");

  // Conditional read from file or stdin
  bool from_stdin = false;
  std::ifstream file;
  std::string filename =
      parameters.Get<std::string>(graph_prefix + "graph-file");
  if (filename == "") {  // Read from stdin
    util::PrintMsg("  Reading from stdin:", !quiet);
    from_stdin = true;
  } else {  // Read from file
    file.open(filename.c_str(), std::ios::in);
    if (!file.is_open()) {
      return util::GRError(cudaErrorUnknown, "Unable to open file " + filename,
                           __FILE__, __LINE__);
    }
    util::PrintMsg("  Reading from " + filename + ":", !quiet);
  }
  std::istream &input_stream = (from_stdin) ? std::cin : file;

  auto &edge_pairs = graph.CooT::edge_pairs;
  SizeT nodes = 0;
  SizeT edges = 0;
  bool got_edge_values = false;
  bool symmetric = false;  // whether the graph is undirected
  bool skew = false;  // whether edge values are inverse for symmetric matrices
  bool array = false;  // whether the mtx file is in dense array format

  time_t mark0 = time(NULL);
  util::PrintMsg("  Parsing MARKET COO format", !quiet, false);

  GUARD_CU(graph.CooT::Release());

  // Read file head
  std::string line;
  while (true) {
    std::getline(input_stream, line);
    if (line[0] != '%') {
      break;
    } else {  // Comment
      if (strlen(line.c_str()) >= 2 && line[1] == '%') {
        symmetric = (strstr(line.c_str(), "symmetric") != NULL);
        skew = (strstr(line.c_str(), "skew") != NULL);
        array = (strstr(line.c_str(), "array") != NULL);
      }
    }
  }

  // Get file meta-data: matrix row/col and number of entries(edges)
  long long ll_nodes_x, ll_nodes_y, ll_edges;
  int items_scanned =
          sscanf(line.c_str(), "%lld %lld %lld", &ll_nodes_x, &ll_nodes_y, &ll_edges);

  if (array && items_scanned == 2) {
    ll_edges = ll_nodes_x * ll_nodes_y;
  } else if (!array && items_scanned == 3) {
    if (ll_nodes_x != ll_nodes_y) {
      return util::GRError(cudaErrorUnknown,
                            "Error parsing MARKET graph: not square (" +
                                std::to_string(ll_nodes_x) + ", " +
                                std::to_string(ll_nodes_y) + ")",
                            __FILE__, __LINE__);
    }
  } else {
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
  GUARD_CU(graph.CooT::Allocate(nodes, edges, util::HOST));

  if (edge_pairs.GetPointer(util::HOST) == NULL) {
        return util::GRError(cudaErrorUnknown,
                             "Error parsing MARKET graph: "
                             "invalid format",
                             __FILE__, __LINE__);
  }

  for (int i = 0; i < edges; ++i) {
    std::string line;
    std::getline(input_stream, line);

    long long ll_row, ll_col;
    ValueT ll_value;  // used for parse float / double
    double lf_value;  // used to sscanf value variable types
    int num_input;
    if (GraphT::FLAG & graph::HAS_EDGE_VALUES) {
      num_input = sscanf(line.c_str(), "%lld %lld %lf", &ll_row, &ll_col,
      &lf_value);

      if (array && (num_input == 1)) {
        ll_value = ll_row;
        ll_col = i / nodes;
        ll_row = i - nodes * ll_col;
      } else if (array || num_input < 2) {
          GUARD_CU(graph.CooT::Release());
          return util::GRError(cudaErrorUnknown,
                               "Error parsing MARKET graph: "
                               "badly formed edge",
                               __FILE__, __LINE__);
        } else if (num_input == 2) {
        ll_value = 1;
      } else if (num_input > 2) {
        if (typeid(ValueT) == typeid(float) ||
            typeid(ValueT) == typeid(double) ||
            typeid(ValueT) == typeid(long double))
          ll_value = (ValueT)lf_value;
        else
          ll_value = (ValueT)(lf_value + 1e-10);
        got_edge_values = true;
      }

    } else { // if (GraphT::FLAG & graph::HAS_EDGE_VALUES)
      num_input = sscanf(line.c_str(), "%lld %lld", &ll_row, &ll_col);

      if (array && (num_input == 1)) {
        ll_value = ll_row;
        ll_col = i / nodes;
        ll_row = i - nodes * ll_col;
      } else if (array || (num_input != 2)) {
          GUARD_CU(graph.CooT::Release());
          return util::GRError(cudaErrorUnknown,
                               "Error parsing MARKET graph: "
                               "badly formed edge",
                               __FILE__, __LINE__);
      }
    }

    edge_pairs[i].x = ll_row;  // zero-based array
    edge_pairs[i].y = ll_col;  // zero-based array

    if (GraphT::FLAG & graph::HAS_EDGE_VALUES) {
      graph.CooT::edge_values[i] = ll_value;
    }
  } // endfor

  time_t mark1 = time(NULL);
  util::PrintMsg("  Done (" + std::to_string(mark1 - mark0) + " s).", !quiet);

  if (filename != "") {
    file.close();
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

template<typename ParametersT>
cudaError_t UseParameters(ParametersT &parameters,
                          std::string graph_prefix = "") {
  cudaError_t retval = cudaSuccess;

  return retval;
}

template<typename ParametersT>
cudaError_t WriteMeta(ParametersT &parameters, std::string filename,
                      MetaData &meta_data) {
  cudaError_t retval = cudaSuccess;
  std::ofstream fout;
  bool quiet = parameters.template Get<bool>("quiet");
  fout.open((filename + ".meta"));
  if (!fout.is_open()) return cudaErrorUnknown;
  util::PrintMsg("  Writing meta data into " + filename + ".meta", !quiet);

  for (auto it : meta_data) {
    fout << it.first << " " << it.second << std::endl;
  }
  fout.close();
  return retval;
}

template<typename ParametersT>
cudaError_t ReadMeta(ParametersT &parameters, std::string filename,
                     MetaData &meta_data) {
  cudaError_t retval = cudaSuccess;
  std::ifstream fin;
  bool quiet = parameters.template Get<bool>("quiet");
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

  bool read_from_binary = parameters.Get<bool>(graph_prefix + "read-from-binary");

  if (read_from_binary) {
    retval = ReadMeta(parameters, filename, meta_data);
  }

  if (parameters.UseDefault("dataset")) {
    std::string dir, file, extension;
    util::SeperateFileName(filename, dir, file, extension);
    parameters.Set("dataset", file);
  }

  if (retval == cudaSuccess && read_from_binary) {
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
