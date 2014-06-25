// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_utils.h
 *
 * @brief Utility Routines for Tests
 */

#pragma once

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    #undef small            // Windows is terrible for polluting macro namespace
#else
    #include <sys/resource.h>
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>

#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <utility>
#include <gunrock/util/random_bits.h>
#include <gunrock/util/basic_utils.h>

namespace gunrock {
namespace util {

/******************************************************************************
 * Command-line parsing functionality
 ******************************************************************************/

/**
 * CommandLineArgs interface
 */
class CommandLineArgs
{
protected:

    std::map<std::string, std::string> pairs;

public:

    // Constructor
    CommandLineArgs(int argc, char **argv)
    {
        using namespace std;

        for (int i = 1; i < argc; i++)
        {
            string arg = argv[i];

            if ((arg[0] != '-') || (arg[1] != '-')) {
                continue;
            }

            string::size_type pos;
            string key, val;
            if ((pos = arg.find( '=')) == string::npos) {
                key = string(arg, 2, arg.length() - 2);
                val = "";
            } else {
                key = string(arg, 2, pos - 2);
                val = string(arg, pos + 1, arg.length() - 1);
            }
            pairs[key] = val;
        }
    }

    // Checks whether a flag "--<flag>" is present in the commandline
    bool CheckCmdLineFlag(const char* arg_name)
    {
        using namespace std;
        map<string, string>::iterator itr;
        if ((itr = pairs.find(arg_name)) != pairs.end()) {
            return true;
        }
        return false;
    }

    // Returns the value specified for a given commandline parameter --<flag>=<value>
    template <typename T>
    void GetCmdLineArgument(const char *arg_name, T &val);

    // Returns the values specified for a given commandline parameter --<flag>=<value>,<value>*
    template <typename T>
    void GetCmdLineArguments(const char *arg_name, std::vector<T> &vals);

    // The number of pairs parsed
    int ParsedArgc()
    {
        return pairs.size();
    }
};

template <typename T>
void CommandLineArgs::GetCmdLineArgument(
    const char *arg_name,
    T &val)
{
    using namespace std;
    map<string, string>::iterator itr;
    if ((itr = pairs.find(arg_name)) != pairs.end()) {
        istringstream str_stream(itr->second);
        str_stream >> val;
    }
}

template <typename T>
void CommandLineArgs::GetCmdLineArguments(
    const char *arg_name,
    std::vector<T> &vals)
{
    using namespace std;

    // Recover multi-value string
    map<string, string>::iterator itr;
    if ((itr = pairs.find(arg_name)) != pairs.end()) {

        // Clear any default values
        vals.clear();

        string val_string = itr->second;
        istringstream str_stream(val_string);
        string::size_type old_pos = 0;
        string::size_type new_pos = 0;

        // Iterate comma-separated values
        T val;
        while ((new_pos = val_string.find(',', old_pos)) != string::npos) {

            if (new_pos != old_pos) {
                str_stream.width(new_pos - old_pos);
                str_stream >> val;
                vals.push_back(val);
            }

            // skip over comma
            str_stream.ignore(1);
            old_pos = new_pos + 1;
        }

        // Read last value
        str_stream >> val;
        vals.push_back(val);
    }
}

class Statistic
{
    double mean;
    double m2;
    int count;

public:
    Statistic() : mean(0.0), m2(0.0), count(0) {}

    /**
     * @brief Updates running statistic, returning bias-corrected
     * sample variance.
     *
     * Online method as per Knuth.
     *
     * @param[in] sample
     * @returns Something
     */
    double Update(double sample)
    {
        count++;
        double delta = sample - mean;
        mean = mean + (delta / count);
        m2 = m2 + (delta * (sample - mean));
        return m2 / (count - 1);                //bias-corrected
    }
};

/******************************************************************************
 * Templated routines for printing keys/values to the console 
 ******************************************************************************/

template<typename T> 
void PrintValue(T val) {
    val.Print();
}

template<>
void PrintValue<char>(char val) {
    printf("%d", val);
}

template<>
void PrintValue<short>(short val) {
    printf("%d", val);
}

template<>
void PrintValue<int>(int val) {
    printf("%d", val);
}

template<>
void PrintValue<long>(long val) {
    printf("%ld", val);
}

template<>
void PrintValue<long long>(long long val) {
    printf("%lld", val);
}

template<>
void PrintValue<float>(float val) {
    printf("%f", val);
}

template<>
void PrintValue<double>(double val) {
    printf("%f", val);
}

template<>
void PrintValue<unsigned char>(unsigned char val) {
    printf("%u", val);
}

template<>
void PrintValue<unsigned short>(unsigned short val) {
    printf("%u", val);
}

template<>
void PrintValue<unsigned int>(unsigned int val) {
    printf("%u", val);
}

template<>
void PrintValue<unsigned long>(unsigned long val) {
    printf("%lu", val);
}

template<>
void PrintValue<unsigned long long>(unsigned long long val) {
    printf("%llu", val);
}

template<>
void PrintValue<bool>(bool val) {
    if (val)
        printf("true");
    else
        printf("false");
}


/******************************************************************************
 * Helper routines for list construction and validation 
 ******************************************************************************/

/**
 * \addtogroup PublicInterface
 * @{
 */

/**
 * @brief Compares the equivalence of two arrays. If incorrect, print the location
 * of the first incorrect value appears, the incorrect value, and the reference
 * value.
 *
 * @tparam T datatype of the values being compared with.
 * @tparam SizeT datatype of the array length.
 *
 * @param[in] computed Vector of values to be compared.
 * @param[in] reference Vector of reference values
 * @param[in] len Vector length
 * @param[in] verbose Whether to print values around the incorrect one.
 *
 * \return Zero if two vectors are exactly the same, non-zero if there is any difference.
 *
 */
template <typename T, typename SizeT>
int CompareResults(T* computed, T* reference, SizeT len, bool verbose = true)
{
    int flag = 0;
    for (SizeT i = 0; i < len; i++) {

        if (computed[i] != reference[i] && flag == 0) {
            printf("\nINCORRECT: [%lu]: ", (unsigned long) i);
            PrintValue<T>(computed[i]);
            printf(" != ");
            PrintValue<T>(reference[i]);

            if (verbose) {
                printf("\nresult[...");
                for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++) {
                    PrintValue<T>(computed[j]);
                    printf(", ");
                }
                printf("...]");
                printf("\nreference[...");
                for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++) {
                    PrintValue<T>(reference[j]);
                    printf(", ");
                }
                printf("...]");
            }
            flag += 1;
            //return flag;
        }
        if (computed[i] != reference[i] && flag > 0) flag+=1;
    }
    printf("\n");
    if (flag == 0)
        printf("CORRECT");
    return flag;
}


/**
 * @brief Compares the equivalence of two arrays. Partial specialization for
 * float type. If incorrect, print the location of the first incorrect value
 * appears, the incorrect value, and the reference value.
 *
 * @tparam T datatype of the values being compared with.
 * @tparam SizeT datatype of the array length.
 *
 * @param[in] computed Vector of values to be compared.
 * @param[in] reference Vector of reference values
 * @param[in] len Vector length
 * @param[in] verbose Whether to print values around the incorrect one.
 *
 * \return Zero if difference between each element of the two vectors are less
 * than a certain threshold, non-zero if any difference is equal to or larger
 * than the threshold.
 *
 */
template <typename SizeT>
int CompareResults(float* computed, float* reference, SizeT len, bool verbose = true)
{
    float THRESHOLD = 0.05f;
    int flag = 0;
    for (SizeT i = 0; i < len; i++) {

        // Use relative error rate here.
        bool is_right = true;
        if (fabs(computed[i] - 0.0) < 0.01f) {
            if ((computed[i] - reference[i]) > THRESHOLD)
                is_right = false;
        } else {
            if (fabs((computed[i] - reference[i])/reference[i]) > THRESHOLD)
                is_right = false;
        }
        if (!is_right && flag == 0) {
            printf("\nINCORRECT: [%lu]: ", (unsigned long) i);
            PrintValue<float>(computed[i]);
            printf(" != ");
            PrintValue<float>(reference[i]);

            if (verbose) {
                printf("\nresult[...");
                for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++) {
                    PrintValue<float>(computed[j]);
                    printf(", ");
                }
                printf("...]");
                printf("\nreference[...");
                for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++) {
                    PrintValue<float>(reference[j]);
                    printf(", ");
                }
                printf("...]");
            }
            flag += 1;
            //return flag;
        }
        if (!is_right && flag > 0) flag += 1;
    }
    printf("\n");
    if (!flag)
        printf("CORRECT");
    return flag;
}

/** @} */

struct CpuTimer
{
#if defined(_WIN32) || defined(_WIN64)

    LARGE_INTEGER ll_freq;
    LARGE_INTEGER ll_start;
    LARGE_INTEGER ll_stop;

    CpuTimer()
    {
        QueryPerformanceFrequency(&ll_freq);
    }

    void Start()
    {
        QueryPerformanceCounter(&ll_start);
    }

    void Stop()
    {
        QueryPerformanceCounter(&ll_stop);
    }

    float ElapsedMillis()
    {
        double start = double(ll_start.QuadPart) / double(ll_freq.QuadPart);
        double stop  = double(ll_stop.QuadPart) / double(ll_freq.QuadPart);

        return (stop - start) * 1000;
    }

#else

    rusage start;
    rusage stop;

    void Start()
    {
        getrusage(RUSAGE_SELF, &start);
    }

    void Stop()
    {
        getrusage(RUSAGE_SELF, &stop);
    }

    float ElapsedMillis()
    {
        float sec = stop.ru_utime.tv_sec - start.ru_utime.tv_sec;
        float usec = stop.ru_utime.tv_usec - start.ru_utime.tv_usec;

        return (sec * 1000) + (usec / 1000);
    }

#endif
};

// Quite simple KeyValuePair struct for doing
// Key-value sorting according to keys
template<typename A, typename B>
struct KeyValuePair
{
    A Key;
    B Value;
    bool operator<(const KeyValuePair<A, B>& rhs)
    {
        return this->Key < rhs.Key;
    }
};

} //util
} //gunrock
