/*
 * @brief Subgraph matching test for shared library advanced interface
 * @file test_smapi.cu
 */

 #include <stdio.h>

 #include <thrust/copy.h>
 #include <thrust/host_vector.h>
 #include <thrust/device_vector.h>
 
 #include <gunrock/gunrock.h>
 #include <gunrock/app/sssp/sssp_app.cu>
 //#include <gunrock/app/test_base.cuh>
 
 // __global__ 
 // void kernel_print(unsigned long* list_subgraphs, unsigned long* size) {
 //   int tid = threadIdx.x;
 //   if(tid > size[0]) return;
 //   printf("%lu\n", list_subgraphs[tid]);
 // }
 
 int main(int argc, char *argv[]) {
  //  const int num_data_nodes = 5, num_data_edges = 10;
  // //  int data_row_offsets[6] = {0, 2, 6, 7, 9, 10};
  // //  int data_col_indices[10] = {1, 3, 0, 2, 3, 4, 1, 0, 1, 1};
  //  int data_col_indices[10] = {1, 3, 4, 2, 3, 0, 3, 1, 2, 3};
  //  int data_row_offsets[6] = {0, 3, 5, 7, 7, 10};
  //  float data_edge_values[10] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  int num_nodes = 7, num_edges = 15;  // number of nodes and edges
  int row_offsets[8]  = {0, 3, 6, 9, 11, 14, 15, 15};
  int col_indices[15] = {1, 2, 3, 0, 2, 4, 3, 4, 5, 5, 6, 2, 5, 6, 6};
  const float edge_values[15] = {39, 6, 41, 51, 63, 17, 10, 44, 41, 13, 58, 43, 50, 59, 35}; 
  int source = 1;
  const bool mark_pred = false;

 
   //printing contents of a thrust array that's currently allocated on device
   //thrust::copy(device_data_row_offsets.begin(), device_data_row_offsets.end(), std::ostream_iterator<int>(std::cout, " "));
 
  //  int num_query_nodes = 3, num_query_edges = 6;
   
 
  
 
   unsigned int memspace;
 
   if(*argv[1] == '1'){
     //allocating arrays on CPU. If memspace ==2, need to allocate on gpu
     memspace = 0x01;
     
    //  unsigned long *sm_counts = (unsigned long *)malloc(sizeof(unsigned long));
    //  unsigned long *list_sm = (unsigned long *)malloc(sizeof(unsigned long)); 
     float *distances = (float *)malloc(num_nodes * sizeof(float));
     int *preds = (int*)NULL;//(int *)malloc(sizeof(int));

     double elapsed =
       sssp(num_nodes, num_edges, row_offsets, col_indices,
            edge_values, source, mark_pred, distances, preds, memspace);
     
     printf("Distances: ");
     for(int i = 0; i < num_nodes; i++){
       printf("%f ", distances[i]);
     }
     printf("\n");
     //printf("List of matched subgraphs: [%i]\n", preds[0]);
   }
 
    if(*argv[1] == '2'){
 
     memspace = 0x02;
 
     thrust::device_vector<int> device_row_offsets(row_offsets, row_offsets+8);
     thrust::device_vector<int> device_col_indices(col_indices, col_indices+15);
     thrust::device_vector<float> device_edge_values(edge_values, edge_values+15);
     
     thrust::device_vector<float> device_distances(num_nodes);
     thrust::device_vector<int> device_preds(1);

     double elapsed =
        sssp(num_nodes, num_edges, device_row_offsets.data().get(), device_col_indices.data().get(),
          device_edge_values.data().get(), source, mark_pred, device_distances.data().get(), device_preds.data().get(), memspace);
     
     printf("Distances: ");
     thrust::copy(device_distances.begin(), device_distances.end(), std::ostream_iterator<int>(std::cout, " "));     
     //thrust::copy(device_list_sm.begin(), device_list_sm.end(), std::ostream_iterator<int>(std::cout, " "));
 
    }
 
 
 
 
 
 
 
 //  double elapsed =
 //       sm(num_data_nodes, num_data_edges, device_data_row_offsets.data().get(), data_col_indices,
 //          num_query_nodes, num_query_edges, query_row_offsets, query_col_indices,
 //          1, sm_counts, list_sm, location);
 
 
 
 
   // cudaFree(device_data_row_offsets);
   //if (sm_counts) free(sm_counts);
   //if (list_sm) free(list_sm);
 
   return 0;
 }
 