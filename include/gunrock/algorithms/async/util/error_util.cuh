#ifndef ERROR_UTIL
#define ERROR_UTIL
 #define CUDA_CHECK(call) {                                    \
          cudaError_t err =                                                         call;                                                    \
          if( cudaSuccess != err) {                                                 \
                       fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",         \
                                                __FILE__, __LINE__, cudaGetErrorString( err) );               \
                       exit(EXIT_FAILURE);                                                   \
                   } }

 #define MALLOC_CHECK(call) {     \
          if(call==NULL) \
          {              \
                       std::cout << "malloc fail in file: "<<__FILE__ << " in line "<<       __LINE__<<".\n";     \
                       exit(1);   \
                   }              \
      }

#endif
