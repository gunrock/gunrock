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

//#define SNN_HELPERS_DEBUG 1
#ifdef SNN_HELPERS_DEBUG
    #define debug(a...) printf(a)
#else
    #define debug(a...)
#endif

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

template <typename SizeT>
__device__ __host__
SizeT SNNsimilarity(SizeT x, SizeT y, util::Array1D<SizeT, SizeT> knns_sorted, 
        SizeT eps, SizeT k, bool fast=false){
    SizeT y_it = 0, counter = 0;
    debug("x = %d, y = %d\n", x, y);
    for (SizeT x_it = 0; x_it < k; ++x_it){
        SizeT x_neighbor = knns_sorted[x * k + x_it];
        SizeT y_neighbor = knns_sorted[y * k + y_it];
        debug("try x_neighbor %d, y_neighbor %d\t", x_neighbor, y_neighbor);
        if (x_neighbor == y_neighbor){
            debug("the same\n");
            ++counter;
            ++y_it; // x_it is increasing in 'for' loop 
            if (y_it == k) break;
            if (fast && counter >= eps) return counter;
        }else if (x_neighbor > y_neighbor){
            while (x_neighbor > y_neighbor){
                ++y_it; 
                if (y_it == k) break;
                y_neighbor = knns_sorted[y * k + y_it];
                debug("try y_neighbor %d\t", y_neighbor);
            }
            if (x_neighbor == y_neighbor){
                debug("the same");
                ++counter;
                ++y_it; // x_it is increasing in 'for' loop 
                if (y_it == k) break;
                if (fast && counter >= eps) return counter;
            }
            debug("\n");
        }
        if (y_it == k) break;
    }
    return counter;
}


}  // namespace snn
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
