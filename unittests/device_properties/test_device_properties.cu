#include <iostream>
#include <string>
#include <gunrock/cuda/device_properties.hxx>

#ifdef __CUDA_ARCH__ // Device-side compilation
#define GR_CUDA_ARCH __CUDA_ARCH__
#else // Host-side compilation
#define GR_CUDA_ARCH 300
#endif

int main(int argc, char** argv) {
    using namespace std;
    using namespace gunrock::cuda::properties;
    int arch_val = (argc > 1) ? stoi(argv[1]) : 300;
    gunrock::cuda::architecture_t arch{arch_val};
    cout << "Architecture Version Major: " << arch.major << endl;
    cout << "Architecture Version Minor: " << arch.minor << endl;
    cout << endl;
    cout << "cta_max_threads:           " << cta_max_threads() << endl;
    cout << "warp_max_threads:          " << warp_max_threads() << endl;
    cout << "sm_max_ctas:               " << sm_max_ctas(arch) << endl;
    cout << "sm_max_threads:            " << sm_max_threads(arch) << endl;
    cout << "sm_registers:              " << sm_registers(arch) << endl;
    cout << "sm_max_smem_bytes:         " << sm_max_smem_bytes(arch) << endl;
    cout << "shared_memory_banks:       " << shared_memory_banks() << endl;
    cout << "shared_memory_bank_stride: " << shared_memory_bank_stride() << endl;
}
