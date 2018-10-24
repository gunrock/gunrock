// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * labels.cuh
 *
 * @brief LABELS Construction Routines
 */

#pragma once

#include <libgen.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <iostream>

#include <gunrock/util/parameters.h>
#include <gunrock/graph/coo.cuh>

namespace gunrock {
namespace graphio {
namespace labels {

/**
 * @brief Reads a user->labels file from an input-stream into a CSR sparse
 *
 * Here is an example of the labels format
 * +----------------------------------------------+
 * |%%Labels Formatted File 			  | <--- header line
 * |%                                             | <--+
 * |% comments                                    |    |-- 0 or more comment
 * lines
 * |%                                             | <--+
 * |  N L L                                       | <--- nodes, labels, labels
 * |  I1 L1A L1B                                  | <--+
 * |  I2 L2A L2B                                  |    |
 * |     . . .                                    |    |
 * |  IN LNA LNB                                  | <--+
 * +----------------------------------------------+
 *
 *
 * @param[in] f_in          Input labels graph file.
 * @param[in] labels_a	    Array for first labels column
 * @param[in] labels_b	    Array for second labels column (optional)
 *
 * \return If there is any File I/O error along the way.
 */
template <typename ArrayT>
cudaError_t ReadLabelsStream(FILE *f_in, util::Parameters &parameters,
                             ArrayT &labels_a, ArrayT &labels_b) {
  typedef typename ArrayT::ValueT ValueT;

  cudaError_t retval = cudaSuccess;
  bool quiet = parameters.Get<bool>("quiet");

  int labels_read = -1;
  long long nodes = 0;

  // bool label_b_exists = false; // change this to a parameter
  // util::Array1D<SizeT, EdgePairT> temp_edge_pairs;
  // temp_edge_pairs.SetName("graphio::market::ReadMarketStream::temp_edge_pairs");
  // EdgeTupleType *coo = NULL; // read in COO format

  time_t mark0 = time(NULL);
  util::PrintMsg("  Parsing LABELS", !quiet);

  char line[1024];

  while (true) {
    if (fscanf(f_in, "%[^\n]\n", line) <= 0) {
      break;
    }

    if (line[0] == '%') {  // Comment
      if (strlen(line) >= 2 && line[1] == '%') {
        // Header -> Can be used to extract info for labels
      }
    }  // -> if

    else if (!util::isValid(labels_read)) {  // Problem description-> First line
                                             // with nodes and labels info
      long long ll_nodes, ll_label_x, ll_label_y;
      int items_scanned =
          sscanf(line, "%lld %lld %lld", &ll_nodes, &ll_label_x, &ll_label_y);
      if (ll_label_x != ll_label_y) {
        return util::GRError(
            "Error parsing LABELS, problem description invalid (" +
                std::to_string(ll_label_x) +
                " =/= " + std::to_string(ll_label_y) + ")",
            __FILE__, __LINE__);
      }

      nodes = ll_nodes;

      util::PrintMsg(" (" + std::to_string(ll_nodes) + " nodes) ", !quiet);

      for (int k = 0; k < nodes; k++) {
        labels_a[k] = util::PreDefinedValues<ValueT>::InvalidValue;
        labels_b[k] = util::PreDefinedValues<ValueT>::InvalidValue;
      }

      labels_read = 0;
    }  // -> else if

    else {  // Now we can start storing labels
      if (labels_read >= nodes) {
        return util::GRError(
            "Error parsing LABELS: "
            "encountered more than " +
                std::to_string(nodes) + " nodes",
            __FILE__, __LINE__);
      }

      long long ll_node;  // Active node

      // Used for sscanf
      double lf_label_a = util::PreDefinedValues<ValueT>::InvalidValue;
      double lf_label_b = util::PreDefinedValues<ValueT>::InvalidValue;

      ValueT ll_label_a, ll_label_b;  // Used to parse float/double

      int num_input =
          sscanf(line, "%lld %lf %lf", &ll_node, &lf_label_a, &lf_label_b);

      if (typeid(ValueT) == typeid(float) || typeid(ValueT) == typeid(double) ||
          typeid(ValueT) == typeid(long double)) {
        ll_label_a = (ValueT)lf_label_a;
        ll_label_b = (ValueT)lf_label_b;
      } else {
        ll_label_a = lf_label_a;
        ll_label_b = lf_label_b;
      }

      if (!util::isValid(
              ll_label_a)) {  // Populate the missing labels as invalid (-1)
        ll_label_a = util::PreDefinedValues<ValueT>::InvalidValue;
      }

      if (!util::isValid(
              ll_label_b)) {  // Populate the missing label b as invalid (-1)
        ll_label_b = util::PreDefinedValues<ValueT>::InvalidValue;
      }

      labels_a[ll_node - 1] = ll_label_a;
      labels_b[ll_node - 1] = ll_label_b;
      labels_read++;

    }  // -> else
  }    // -> while

  if (labels_read != nodes) {
    return util::GRError(
        "Error parsing LABELS: "
        "only " +
            std::to_string(labels_read) + "/" + std::to_string(nodes) +
            " nodes read",
        __FILE__, __LINE__);
  }

  time_t mark1 = time(NULL);
  util::PrintMsg("Done parsing (" + std::to_string(mark1 - mark0) + " s).",
                 !quiet);

  return retval;
}

/**
 * \defgroup Public Interface
 * @{
 */

/**
 * @brief Loads a LABELS-formatted array(s) from the specified file.
 *
 * @param[in] filename Labels file name, if empty, it is loaded from STDIN.
 * @param[in] parameters Idk if we need any parameters (placeholder).
 * @param[in] labels_a Array that we can populate with the 2nd column values.
 * @param[in] labels_b (optional) Array that we can populate with 3rd column
 * values.
 *
 * \return If there is any File I/O error along the way. 0 for no error.
 */
template <typename ArrayT>
cudaError_t BuildLabelsArray(std::string filename, util::Parameters &parameters,
                             ArrayT &labels_a, ArrayT &labels_b) {
  typedef typename ArrayT::ValueT ValueT;

  cudaError_t retval = cudaSuccess;
  bool quiet = parameters.Get<bool>("quiet");

  FILE *f_in = fopen(filename.c_str(), "r");
  if (f_in) {
    util::PrintMsg("Reading from " + filename + ":", !quiet);
    if (retval = ReadLabelsStream(f_in, parameters, labels_a, labels_b)) {
      fclose(f_in);
      return retval;
    }
  }

  else {
    return util::GRError("Unable to open file " + filename, __FILE__, __LINE__);
  }
  return retval;
}

cudaError_t UseParameters(util::Parameters &parameters,
                          std::string graph_prefix = "") {
  cudaError_t retval = cudaSuccess;

  return retval;
}

template <typename ArrayT>
cudaError_t Read(util::Parameters &parameters, ArrayT &labels_a,
                 ArrayT &labels_b) {
  cudaError_t retval = cudaSuccess;
  bool quiet = parameters.Get<bool>("quiet");
  util::PrintMsg("Loading Labels into an array ...", !quiet);

  std::string filename = parameters.Get<std::string>("labels-file");

  std::ifstream fp(filename.c_str());
  if (filename == "" || !fp.is_open()) {
    return util::GRError("Input labels file " + filename + " does not exist.",
                         __FILE__, __LINE__);
  }

  if (parameters.UseDefault("dataset")) {
    std::string dir, file, extension;
    util::SeperateFileName(filename, dir, file, extension);
    // util::PrintMsg("filename = " + filename
    //    + ", dir = " + dir
    //    + ", file = " + file
    //    + ", extension = " + extension);
    parameters.Set("dataset", file);
  }

  GUARD_CU(BuildLabelsArray(filename, parameters, labels_a, labels_b));
  return retval;
}

/**@}*/

}  // namespace labels
}  // namespace graphio
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
