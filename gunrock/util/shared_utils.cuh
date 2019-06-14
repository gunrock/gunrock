// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * shared_utils.cuh
 *
 * @brief shared utilities for the test programs
 */

#pragma once

namespace gunrock {

template <typename Problem>
void Display_Memory_Usage(int num_gpus, int *gpu_idx, size_t *org_size,
                          Problem *problem) {
  typedef typename Problem::SizeT SizeT;

  printf("\n\tMemory Usage(B)\t");
  for (int gpu = 0; gpu < num_gpus; gpu++)
    if (num_gpus > 1) {
      if (gpu != 0) {
        printf(" #keys%d,0\t #keys%d,1\t #ins%d,0\t #ins%d,1", gpu, gpu, gpu,
               gpu);
      } else {
        printf(" #keys%d,0\t #keys%d,1", gpu, gpu);
      }
    } else {
      printf(" #keys%d,0\t #keys%d,1", gpu, gpu);
    }
  if (num_gpus > 1) {
    printf(" #keys%d", num_gpus);
  }
  printf("\n");
  double max_queue_sizing_[2] = {0, 0}, max_in_sizing_ = 0;
  for (int gpu = 0; gpu < num_gpus; gpu++) {
    size_t gpu_free, dummy;
    cudaSetDevice(gpu_idx[gpu]);
    cudaMemGetInfo(&gpu_free, &dummy);
    printf("GPU_%d\t %ld", gpu_idx[gpu], org_size[gpu] - gpu_free);
    for (int i = 0; i < num_gpus; i++) {
      for (int j = 0; j < 2; j++) {
        SizeT x =
            problem->data_slices[gpu]->frontier_queues[i].keys[j].GetSize();
        printf("\t %lld", (long long)x);
        double factor =
            1.0 * x /
            (num_gpus > 1 ? problem->graph_slices[gpu]->in_counter[i]
                          : problem->graph_slices[gpu]->nodes);
        if (factor > max_queue_sizing_[j]) {
          max_queue_sizing_[j] = factor;
        }
      }
      if (num_gpus > 1 && i != 0) {
        for (int t = 0; t < 2; t++) {
          SizeT x = problem->data_slices[gpu][0].keys_in[t][i].GetSize();
          printf("\t %lld", (long long)x);
          double factor = 1.0 * x / problem->graph_slices[gpu]->in_counter[i];
          if (factor > max_in_sizing_) {
            max_in_sizing_ = factor;
          }
        }
      }
    }
    if (num_gpus > 1) {
      printf("\t %lld", (long long)(problem->data_slices[gpu]
                                        ->frontier_queues[num_gpus]
                                        .keys[0]
                                        .GetSize()));
    }
    printf("\n");
  }
  printf("\t queue_sizing =\t %lf \t %lf", max_queue_sizing_[0],
         max_queue_sizing_[1]);
  if (num_gpus > 1) {
    printf("\t in_sizing =\t %lf", max_in_sizing_);
  }
  printf("\n");
}

template <typename T>
void V2Str(std::vector<T> &v, std::string &str) {
  str = "";
  for (auto it = v.begin(); it != v.end(); it++)
    str = str + (it == v.begin() ? "" : " ") + std::to_string(*it);
}

template <typename Enactor>
void Display_Performance_Profiling(Enactor *enactor) {
#ifdef ENABLE_PERFORMANCE_PROFILING
  typedef typename Enactor::SizeT SizeT;

  int num_iterations = enactor->iter_total_time[0].size();
  int num_gpus = enactor->num_gpus;
  std::string str;

  printf("\nPerformance profiling log begins\n");
  printf("Iter\t GPU\t Item\t Per-iter readings\n");
  for (int i = 0; i < num_iterations; i++) {
    // printf("Iteration %d\n", i);
    for (int gpu = 0; gpu < num_gpus; gpu++) {
      // printf("\tGPU %d\n", gpu);

      for (int peer = 0; peer < num_gpus; peer++) {
        std::vector<SizeT> &iter_in_length =
            enactor->enactor_stats[gpu * num_gpus + peer].iter_in_length[i];
        if (iter_in_length.size() != 0) {
          V2Str(iter_in_length, str);
          printf("%d\t %d\t In length %d\t %s\n", i, gpu, peer, str.c_str());
        }

        std::vector<SizeT> &iter_nodes_queued =
            enactor->enactor_stats[gpu * num_gpus + peer].iter_nodes_queued[i];
        if (iter_nodes_queued.size() != 0) {
          V2Str(iter_nodes_queued, str);
          printf("%d\t %d\t Nodes queued %d\t %s\n", i, gpu, peer, str.c_str());
        }

        std::vector<SizeT> &iter_edges_queued =
            enactor->enactor_stats[gpu * num_gpus + peer].iter_edges_queued[i];
        if (iter_edges_queued.size() != 0) {
          V2Str(iter_edges_queued, str);
          printf("%d\t %d\t Edges queued %d\t %s\n", i, gpu, peer, str.c_str());
        }
      }

      std::vector<double> &iter_sub_queue_time =
          enactor->iter_sub_queue_time[gpu][i];
      if (iter_sub_queue_time.size() != 0) {
        V2Str(iter_sub_queue_time, str);
        printf("%d\t %d\t Sub queue time\t %s\n", i, gpu, str.c_str());
      }

      std::vector<SizeT> &iter_full_queue_nodes_queued =
          enactor->iter_full_queue_nodes_queued[gpu][i];
      if (iter_full_queue_nodes_queued.size() != 0) {
        V2Str(iter_full_queue_nodes_queued, str);
        printf("%d\t %d\t Fu-queue nodes\t %s\n", i, gpu, str.c_str());
      }

      std::vector<SizeT> &iter_full_queue_edges_queued =
          enactor->iter_full_queue_edges_queued[gpu][i];
      if (iter_full_queue_edges_queued.size() != 0) {
        V2Str(iter_full_queue_edges_queued, str);
        printf("%d\t %d\t Fu-queue edges\t %s\n", i, gpu, str.c_str());
      }

      std::vector<double> &iter_full_queue_time =
          enactor->iter_full_queue_time[gpu][i];
      if (iter_full_queue_time.size() != 0) {
        V2Str(iter_full_queue_time, str);
        printf("%d\t %d\t Fu-queue time\t %s\n", i, gpu, str.c_str());
      }

      std::vector<double> &iter_total_time = enactor->iter_total_time[gpu][i];
      if (iter_total_time.size() != 0) {
        std::vector<double> other_time;
        other_time.clear();
        for (unsigned int j = 0; j < iter_total_time.size(); j++) {
          double elapsed_time = iter_total_time[j];
          if (j < iter_sub_queue_time.size())
            elapsed_time -= iter_sub_queue_time[j];
          if (j < iter_full_queue_time.size())
            elapsed_time -= iter_full_queue_time[j];
          other_time.push_back(elapsed_time);
        }
        V2Str(other_time, str);
        printf("%d\t %d\t Other time\t %s\n", i, gpu, str.c_str());
        V2Str(iter_total_time, str);
        printf("%d\t %d\t Iteration time\t %s\n", i, gpu, str.c_str());
      }

      for (int peer = 0; peer < num_gpus; peer++) {
        std::vector<SizeT> &iter_out_length =
            enactor->enactor_stats[gpu * num_gpus + peer].iter_out_length[i];
        if (iter_out_length.size() != 0) {
          V2Str(iter_out_length, str);
          printf("%d\t %d\t Out length %d\t %s\n", i, gpu, peer, str.c_str());
        }
      }
    }
    printf("\n");
  }
  printf("Performance profiling log ended\n");
#endif
}

}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
