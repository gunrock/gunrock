
/******************************************************************************
 * Timing
 ******************************************************************************/

struct GpuTimer {
    cudaEvent_t _start;
    cudaEvent_t _stop;

    GpuTimer() {
        cudaEventCreate(&_start);
        cudaEventCreate(&_stop);
    }

    ~GpuTimer() {
        cudaEventDestroy(_start);
        cudaEventDestroy(_stop);
    }

    void start() { cudaEventRecord(_start, 0); }
    void stop()  { cudaEventRecord(_stop,  0); }

    float elapsed() {
        float _elapsed;
        cudaEventSynchronize(_stop);
        cudaEventElapsedTime(&_elapsed, _start, _stop);
        return _elapsed;
    }
};
