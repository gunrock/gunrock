// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_utils.cuh
 *
 * @brief Utility Routines for Tests
 */

#pragma once

//#include <boost/timer/timer.hpp>
//#include <boost/chrono/chrono.hpp>
//#include <boost/detail/lightweight_main.hpp>
#include <stdarg.h>
#include <gunrock/util/test_utils.h>
#include <gunrock/util/error_utils.cuh>

namespace gunrock
{
namespace util
{

/******************************************************************************
 * Templated routines for printing keys/values to the console
 ******************************************************************************/

template<typename T>
inline void PrintValue(T val)
{
    val.Print();
}

template<>
inline void PrintValue<char>(char val)
{
    printf("%d", val);
}

template<>
inline void PrintValue<short>(short val)
{
    printf("%d", val);
}

template<>
inline void PrintValue<int>(int val)
{
    printf("%d", val);
}

template<>
inline void PrintValue<long>(long val)
{
    printf("%ld", val);
}

template<>
inline void PrintValue<long long>(long long val)
{
    printf("%lld", val);
}

template<>
inline void PrintValue<float>(float val)
{
    printf("%f", val);
}

template<>
inline void PrintValue<double>(double val)
{
    printf("%f", val);
}

template<>
inline void PrintValue<unsigned char>(unsigned char val)
{
    printf("%u", val);
}

template<>
inline void PrintValue<unsigned short>(unsigned short val)
{
    printf("%u", val);
}

template<>
inline void PrintValue<unsigned int>(unsigned int val)
{
    printf("%u", val);
}

template<>
inline void PrintValue<unsigned long>(unsigned long val)
{
    printf("%lu", val);
}

template<>
inline void PrintValue<unsigned long long>(unsigned long long val)
{
    printf("%llu", val);
}

template<>
inline void PrintValue<bool>(bool val)
{
    if (val)
    {
        printf("true");
    }
    else
    {
        printf("false");
    }
}

/**
 * Verify the contents of a device array match those
 * of a host array
 */
template <typename T>
int CompareDeviceResults(
    T *h_reference,
    T *d_data,
    size_t num_elements,
    bool verbose = true,
    bool display_data = false)
{
    // Allocate array on host
    T *h_data = (T*) malloc(num_elements * sizeof(T));

    // Reduction data back
    cudaMemcpy(h_data, d_data, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);

    // Display data
    if (display_data)
    {
        printf("Reference:\n");
        for (int i = 0; i < num_elements; i++)
        {
            PrintValue(h_reference[i]);
            printf(", ");
        }
        printf("\n\nData:\n");
        for (int i = 0; i < num_elements; i++)
        {
            PrintValue(h_data[i]);
            printf(", ");
        }
        printf("\n\n");
    }

    // Check
    int retval = CompareResults(h_data, h_reference, num_elements, verbose);

    // Cleanup
    if (h_data) free(h_data);

    return retval;
}

inline int CompareDeviceResults(
    util::NullType *h_reference,
    util::NullType *d_data,
    size_t num_elements,
    bool verbose = true,
    bool display_data = false)
{
    return 0;
}

/**
 * Verify the contents of a device array match those
 * of a host array
 */
template <typename T>
int CompareDeviceDeviceResults(
    T *d_reference,
    T *d_data,
    size_t num_elements,
    bool verbose = true,
    bool display_data = false)
{
    // Allocate array on host
    T *h_reference = (T*) malloc(num_elements * sizeof(T));
    T *h_data = (T*) malloc(num_elements * sizeof(T));

    // Reduction data back
    cudaMemcpy(h_reference, d_reference, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data, d_data, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);

    // Display data
    if (display_data)
    {
        printf("Reference:\n");
        for (int i = 0; i < num_elements; i++)
        {
            PrintValue(h_reference[i]);
            printf(", ");
        }
        printf("\n\nData:\n");
        for (int i = 0; i < num_elements; i++)
        {
            PrintValue(h_data[i]);
            printf(", ");
        }
        printf("\n\n");
    }

    // Check
    int retval = CompareResults(h_data, h_reference, num_elements, verbose);

    // Cleanup
    if (h_reference) free(h_reference);
    if (h_data) free(h_data);

    return retval;
}


/**
 * Verify the contents of a device array match those
 * of a host array
 */
template <typename T>
void DisplayDeviceResults(
    T *d_data,
    size_t num_elements)
{
    // Allocate array on host
    T *h_data = (T*) malloc(num_elements * sizeof(T));

    // Reduction data back
    cudaMemcpy(h_data, d_data, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);

    // Display data
    printf("\n\nData:\n");
    for (int i = 0; i < num_elements; i++)
    {
        PrintValue(h_data[i]);
        printf(", ");
    }
    printf("\n\n");

    // Cleanup
    if (h_data) free(h_data);
}

/**
 * Verify the contents of a device array match those
 * of a host array
 */
template <typename DATATYPE, typename INDEXTYPE>
void DisplayDeviceResults(
    DATATYPE *d_data,
    INDEXTYPE *d_indices,
    size_t num_elements,
    size_t num_indices)
{
    printf("num_elements: %zu\n", num_elements);
    printf("num_indices: %zu\n", num_indices);
    // Allocate array on host
    DATATYPE *h_data = (DATATYPE*) malloc(num_elements * sizeof(DATATYPE));
    INDEXTYPE *h_indices = (INDEXTYPE*) malloc(num_indices * sizeof(INDEXTYPE));

    // Reduction data back
    cudaMemcpy(h_data, d_data, sizeof(DATATYPE) * num_elements, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indices, d_indices, sizeof(INDEXTYPE) * num_indices, cudaMemcpyDeviceToHost);

    // Display data
    printf("\n\nData:\n");
    for (int i = 0; i < num_indices; i++)
    {
        PrintValue(h_indices[i]);
        printf(":");
        assert(h_indices[i] < num_elements);
        PrintValue(h_data[h_indices[i]]);
        printf(", ");
    }
    printf("\n\n");

    // Cleanup
    if (h_data) free(h_data);
    if (h_indices) free(h_indices);
}

/******************************************************************************
 * Timing
 ******************************************************************************/

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float ElapsedMillis()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

// Check available device memory
inline bool EnoughDeviceMemory(unsigned int mem_needed)
{
    size_t free_mem, total_mem;
    if (util::GRError(cudaMemGetInfo(&free_mem, &total_mem),
                      "cudaMemGetInfo failed", __FILE__, __LINE__))
        return false;
    return (mem_needed <= free_mem);
}

/******************************************************************************
 * Helper routines for list construction and validation
 ******************************************************************************/

/**
 * \addtogroup PublicInterface
 * @{
 */

/**
 * @brief Compares the equivalence of two arrays. If incorrect, print the
 * location of the first incorrect value appears, the incorrect value, and
 * the reference value.
 *
 * @tparam T datatype of the values being compared with.
 * @tparam SizeT datatype of the array length.
 *
 * @param[in] computed Vector of values to be compared.
 * @param[in] reference Vector of reference values.
 * @param[in] len Vector length.
 * @param[in] verbose Whether to print values around the incorrect one.
 * @param[in] quiet Don't print out anything unless specified.
 *
 * \return Zero if two vectors are exactly the same, non-zero if there is any
 * difference.
 *
 */
template <typename T, typename SizeT>
int CompareResults(
    T* computed,
    T* reference,
    SizeT len,
    bool verbose = true,
    bool quiet = false)
{
    int flag = 0;
    for (SizeT i = 0; i < len; i++)
    {
        if (computed[i] != reference[i] && flag == 0)
        {
            if (!quiet)
            {
                printf("\nINCORRECT: [%lu]: ", (unsigned long) i);
                PrintValue<T>(computed[i]);
                printf(" != ");
                PrintValue<T>(reference[i]);

                if (verbose)
                {
                    printf("\nresult[...");
                    for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++)
                    {
                        PrintValue<T>(computed[j]);
                        printf(", ");
                    }
                    printf("...]");
                    printf("\nreference[...");
                    for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++)
                    {
                        PrintValue<T>(reference[j]);
                        printf(", ");
                    }
                    printf("...]");
                }
            }
            flag += 1;
            //return flag;
        }
        if (computed[i] != reference[i] && flag > 0) flag += 1;
    }
    if (!quiet)
    {
        printf("\n");
    }
    if (flag == 0)
    {
        if (!quiet)
        {
            printf("CORRECT");
        }
    }
    return flag;
}


/**
 * @brief Compares the equivalence of two arrays. Partial specialization for
 * float type. If incorrect, print the location of the first incorrect value
 * appears, the incorrect value, and the reference value.
 *
 * @tparam SizeT datatype of the array length.
 *
 * @param[in] computed Vector of values to be compared.
 * @param[in] reference Vector of reference values
 * @param[in] len Vector length
 * @param[in] verbose Whether to print values around the incorrect one.
 * @param[in] quiet Don't print out anything unless specified.
 *
 * \return Zero if difference between each element of the two vectors are less
 * than a certain threshold, non-zero if any difference is equal to or larger
 * than the threshold.
 *
 */
template <typename SizeT>
int CompareResults(
    float* computed,
    float* reference,
    SizeT len,
    bool verbose = true,
    bool quiet = false)
{
    float THRESHOLD = 0.05f;
    int flag = 0;
    for (SizeT i = 0; i < len; i++)
    {
        // Use relative error rate here.
        bool is_right = true;
        if (fabs(computed[i] - 0.0) < 0.01f)
        {
            if (fabs(computed[i] - reference[i]) > THRESHOLD)
            {
                is_right = false;
            }
        }
        else
        {
            if (fabs((computed[i] - reference[i]) / reference[i]) > THRESHOLD)
            {
                is_right = false;
            }
        }

        if (!is_right)
        {
            if (!quiet && flag < 10)
            {
                printf("\nINCORRECT: [%lu]: ", (unsigned long) i);
                PrintValue<float>(computed[i]);
                printf(" != ");
                PrintValue<float>(reference[i]);

                if (verbose)
                {
                    printf("\nresult[...");
                    for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++)
                    {
                        PrintValue<float>(computed[j]);
                        printf(", ");
                    }
                    printf("...]");
                    printf("\nreference[...");
                    for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++)
                    {
                        PrintValue<float>(reference[j]);
                        printf(", ");
                    }
                    printf("...]");
                }
            }
            flag += 1;
        }
        if (!is_right && flag > 0) flag += 1;
    }
    if (!quiet)
    {
        printf("\n");
    }
    if (!flag)
    {
        if (!quiet)
        {
            printf("CORRECT");
        }
    }
    return flag;
}

/** @} */

}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
