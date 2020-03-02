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
#include <vector>
#include <time.h>
#include <iostream>
#include <sstream>
#include <string>

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

template <typename SizeT, typename ValueT>
cudaError_t ReadLabelsStream(FILE *f_in, util::Parameters &parameters,
                              util::Array1D<SizeT, ValueT>& labels) {

  cudaError_t retval = cudaSuccess;
  bool quiet = parameters.Get<bool>("quiet");
  bool transpose = parameters.Get<bool>("transpose");
  if (transpose)
      printf("table is gonna be tranposed\n");
  else
      printf("table is not tranposed\n");
  long long dim; 
  long long num_labels;
  long long labels_read = -1;
  time_t mark0 = time(NULL);
  long long ll_node = 0;

  char line[10000];

  while (true) {
      if (fscanf(f_in, "%[^\n]\n", line) <= 0){
        break;
      }

#if DEBUG_LABEL
      std::cerr << line << std::endl;
#endif

      if (line[0] == '%' || line[0] == '#') {  // Comment
          if (strlen(line) >= 2 && line[1] == '%'){
          }
      }  // -> if comment

      else if (!util::isValid(labels_read)) {  // Problem description-> First line
          // with nodes and labels info
          long long ll_num_labels, ll_dim;
          int items_scanned = 
              sscanf(line, "%lld %lld", &ll_num_labels, &ll_dim);
          if ((!util::isValid(ll_num_labels)) or (!util::isValid(ll_dim))){
              return util::GRError(
                      "Error parsing LABELS, problem description invalid (" +
                      std::to_string(ll_num_labels) + " < 0" + 
                      std::to_string(ll_dim) + " < 0",
                      __FILE__, __LINE__);
          }
          num_labels = ll_num_labels;
          dim = ll_dim;
  
          util::PrintMsg("Number of labels " + std::to_string(num_labels) + 
                  ", dimension " + std::to_string(dim), !quiet);

          parameters.Set("n", num_labels);
          parameters.Set("dim", dim);

          // Allocation memory for points
          GUARD_CU(labels.Allocate(num_labels*dim, util::HOST));

          for (int k = 0; k < num_labels; k++) {
              for (int d = 0; d < dim; d++){
                  labels[k * dim + d] = 
                      util::PreDefinedValues<ValueT>::InvalidValue;
              }
          }

          labels_read = 0;
    }  // -> else if problem description

    else {  // Now we can start storing labels
      if (labels_read >= num_labels) {
          // Reading labels is done
          break;
      }

      int d = 0;
      while (d < dim){
          
          double lf_label = util::PreDefinedValues<ValueT>::InvalidValue;
          int num_input = sscanf(line, "%lf", &lf_label);
          if (d < dim-1){
              int i=0; 
              while (line[i] != ' ' && line[i] != '\n') ++i;
              memmove(line, line+i+1, 10000-(i+2));
          }

          ValueT ll_label;
          if (typeid(ValueT) == typeid(float) || typeid(ValueT) == typeid(double) ||
          typeid(ValueT) == typeid(long double)) {
              ll_label = (ValueT)lf_label;
          }else{
              ll_label = lf_label;
          }
          if (!util::isValid(ll_label)){
              return util::GRError(
                      "Error parsing LABELS: "  
                      "Invalid " + std::to_string(d) + 
                      "th element of label: " + 
                      std::to_string(ll_label),
                      __FILE__, __LINE__);
          }
          if (!transpose){
              //N M
              //   DA  DB  .. DM 
              //I1 L1A L1B .. L1M
              //I2 L2A L2B .. L2M
              //.. ..  ..  .. ..
              //IN LNA LNB .. LNM
              labels[ll_node * dim + d] = ll_label;
          }else{
              //N M
              //   I1  I2  .. IN 
              //DA L1A L2A .. LNA
              //DB L1B L2B .. LNB
              //.. ..  ..  .. ..
              //DM L1M L2M .. LNM
              labels[d * num_labels + ll_node] = ll_label;
          }
          ++d;
      } // -> while reading line
      ++ll_node;
      if (d < dim){
              return util::GRError(
                      "Error parsing LABELS: "
                      "Invalid length of label: " + std::to_string(d),
                      __FILE__, __LINE__);
      }
      labels_read++;
    }  // -> else storing labels
  }    // -> while

  if (labels_read != num_labels) {
      return util::GRError(
              "Error parsing LABELS: " 
              "only " + std::to_string(labels_read) + 
              "/" + std::to_string(num_labels) +
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

template <typename SizeT, typename ValueT>
cudaError_t BuildLabelsArray(std::string filename, 
        util::Parameters &parameters, 
        util::Array1D<SizeT, ValueT> &labels) {
          
  cudaError_t retval = cudaSuccess;
  bool quiet = parameters.Get<bool>("quiet");

  FILE *f_in = fopen(filename.c_str(), "r");
  if (f_in) {
    util::PrintMsg("Reading from " + filename + ":", !quiet);
    if (retval = ReadLabelsStream(f_in, parameters, labels)) {
      fclose(f_in);
      return retval;
    }
  } else {
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

template <typename SizeT, typename ValueT>
cudaError_t Read(util::Parameters &parameters, 
        util::Array1D<SizeT, ValueT> &labels) {

    // TO DO initialized graph
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

  GUARD_CU(BuildLabelsArray(filename, parameters, labels));
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
