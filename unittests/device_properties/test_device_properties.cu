#include <iostream>
#include <string>
#include <gunrock/cuda/device_properties.hxx>

int main(int argc, char** argv) {
    using namespace std;
    using namespace gunrock::cuda;
    using namespace gunrock::cuda::properties;
    int cc_ver = (argc > 1) ? stoi(argv[1]) : 30;
    compute_capability_t cc = make_compute_capability(cc_ver);
    cout << "Compute Capability Version Major: " << cc.major << endl;
    cout << "Compute Capability Version Minor: " << cc.minor << endl;
    cout << endl;
    cout << "cta_max_threads:           " << cta_max_threads() << endl;
    cout << "warp_max_threads:          " << warp_max_threads() << endl;
    cout << "sm_max_ctas:               " << sm_max_ctas(cc) << endl;
    cout << "sm_max_threads:            " << sm_max_threads(cc) << endl;
    cout << "sm_registers:              " << sm_registers(cc) << endl;
    cout << "sm_max_smem_bytes:         " << sm_max_smem_bytes(cc) << endl;
    cout << "shared_memory_banks:       " << shared_memory_banks() << endl;
    cout << "shared_memory_bank_stride: " << shared_memory_bank_stride() << endl;
}
