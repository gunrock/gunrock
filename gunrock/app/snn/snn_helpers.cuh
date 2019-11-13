// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * snn_helpers.cuh
 *
 * @brief Helper functions for KNN algorithm.
 */

#pragma once

#include <map>
#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace app {
namespace snn {

template<typename SizeT>
class DisjointSet{
    std::vector<SizeT> rank;
    std::vector<SizeT> parent;
    SizeT number;

public:

    DisjointSet(SizeT Number): number(Number){
      rank.resize(Number);
      parent.resize(Number);
      make_sets();
    }

    void make_sets(){
      for (SizeT i = 0; i < number; ++i){
        parent[i] = i;
        rank[i] = 1;
      }
    }

    SizeT Find(SizeT x){
      if (parent[x] != x){
        parent[x] = Find(parent[x]);
      }
      return parent[x];
    }

    void Union(SizeT x, SizeT y){
      SizeT x_set = Find(x);
      SizeT y_set = Find(y);
      if (x_set == y_set) return;
      if (rank[x_set] < rank[y_set]){
        parent[x_set] = y_set;
      }else if (rank[x_set] > rank[y_set]){
        parent[y_set] = x_set;
      }else{
        parent[x_set] = y_set;
        rank[y_set] = rank[y_set] + 1;
      }
    }
};

}  // namespace snn
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
