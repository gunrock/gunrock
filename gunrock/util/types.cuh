// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * types.cuh
 *
 * @brief data types and limits defination
 */

#pragma once

#include <limits.h>
#include <string>
#include <cstdlib>
#include <cstring>
#include <typeinfo>
#include <typeindex>
#include <sstream>
#include <ostream>
#include <istream>
#include <vector>

namespace gunrock {
namespace util {

// Ensure no un-specialized types will be compiled
extern __device__ __host__ void Error_UnsupportedType();

template <typename T>
__device__ __host__ __forceinline__ T MaxValue()
{
    Error_UnsupportedType();
    return 0;
}

template <>
__device__ __host__ __forceinline__ int MaxValue<int>()
{
    return INT_MAX;
}

template <>
__device__ __host__ __forceinline__ long long MaxValue<long long>()
{
    return LLONG_MAX;
}

template <typename T>
__device__ __host__ __forceinline__ T MinValue()
{
    Error_UnsupportedType();
    return 0;
}

template <>
__device__ __host__ __forceinline__ int MinValue<int>()
{
    return INT_MIN;
}

template <>
__device__ __host__ __forceinline__ long long MinValue<long long>()
{
    return LLONG_MIN;
}

/*template <typename T, size_t SIZE>
__device__ __host__ __forceinline__ T AllZeros_N()
{
    Error_UnsupportedSize();
    return 0;
}

template <typename T>
__device__ __host__ __forceinline__ T AllZeros_N<T, 4>()
{
    return (T)0x00000000;
}

template <typename T>
__device__ __host__ __forceinline__ T AllZeros_N<T, 8>()
{
    return (T)0x0000000000000000;
}*/

template <typename T>
__device__ __host__ __forceinline__ T AllZeros()
{
    //return AllZeros_N<T, sizeof(T)>();
    Error_UnsupportedType();
    return 0;
}

template <>
__device__ __host__ __forceinline__ int AllZeros<int>()
{
    return (int)0x00000000;
}

template <>
__device__ __host__ __forceinline__ long long AllZeros<long long>()
{
    return (long long)0x0000000000000000LL;
}


/*template <typename T, size_t SIZE>
__device__ __host__ __forceinline__ T AllOnes_N()
{
    Error_UnsupportedSize();
    return 0;
}

template <typename T>
__device__ __host__ __forceinline__ T AllOnes_N<T, 4>()
{
    return (T)0xFFFFFFFF;
}

template <typename T>
__device__ __host__ __forceinline__ T AllOnes_N<T, 8>()
{
    return (T)0xFFFFFFFFFFFFFFFF;
}*/

template <typename T>
__device__ __host__ __forceinline__ T AllOnes()
{
    //return AllOnes_N<T, sizeof(T)>();
    Error_UnsupportedType();
    return 0;
}

template <>
__device__ __host__ __forceinline__ int AllOnes<int>()
{
    return (int)0xFFFFFFFF;
}

template <>
__device__ __host__ __forceinline__ long long AllOnes<long long>()
{
    return (long long)0xFFFFFFFFFFFFFFFFLL;
}

template <typename T>
__device__ __host__ __forceinline__ T InvalidValue()
{
    //return AllOnes_N<T, sizeof(T)>();
    Error_UnsupportedType();
    return 0;
}

template <>
__device__ __host__ __forceinline__ int InvalidValue<int>()
{
    return (int)-1;
}

template <>
__device__ __host__ __forceinline__ long long InvalidValue<long long>()
{
    return (long long)-1;
}

template <typename T>
__device__ __host__ __forceinline__ bool isValid(T val)
{
    return val >= 0;//(val != InvalidValue<T>());
}

template <typename T, int SIZE>
struct VectorType{ /*typedef UnknownType Type;*/};
template <> struct VectorType<int      , 1> {typedef int       Type;};
template <> struct VectorType<int      , 2> {typedef int2      Type;};
template <> struct VectorType<int      , 3> {typedef int3      Type;};
template <> struct VectorType<int      , 4> {typedef int4      Type;};
template <> struct VectorType<long long, 1> {typedef long long Type;};
template <> struct VectorType<long long, 2> {typedef longlong2 Type;};
template <> struct VectorType<long long, 3> {typedef longlong3 Type;};
template <> struct VectorType<long long, 4> {typedef longlong4 Type;};

std::string TypeName(const std::type_info* t_info)
{
    if (std::type_index(*t_info) == std::type_index(typeid(char)))
        return "char";
    if (std::type_index(*t_info) == std::type_index(typeid(signed char)))
        return "signed char";
    if (std::type_index(*t_info) == std::type_index(typeid(unsigned char)))
        return "unsigned char";
    if (std::type_index(*t_info) == std::type_index(typeid(short)))
        return "short";
    if (std::type_index(*t_info) == std::type_index(typeid(unsigned short)))
        return "unsigned short";
    if (std::type_index(*t_info) == std::type_index(typeid(int)))
        return "int";
    if (std::type_index(*t_info) == std::type_index(typeid(unsigned int)))
        return "unsigned int";
    if (std::type_index(*t_info) == std::type_index(typeid(long)))
        return "long";
    if (std::type_index(*t_info) == std::type_index(typeid(unsigned long)))
        return "unsigned long";
    if (std::type_index(*t_info) == std::type_index(typeid(long long)))
        return "long long";
    if (std::type_index(*t_info) == std::type_index(typeid(unsigned long)))
        return "unsigned long long";
    if (std::type_index(*t_info) == std::type_index(typeid(bool)))
        return "bool";
    if (std::type_index(*t_info) == std::type_index(typeid(float)))
        return "float";
    if (std::type_index(*t_info) == std::type_index(typeid(double)))
        return "double";
    if (std::type_index(*t_info) == std::type_index(typeid(long double)))
        return "long double";
    if (std::type_index(*t_info) == std::type_index(typeid(std::string)))
        return "std::string";
    if (std::type_index(*t_info) == std::type_index(typeid(char*)))
        return "char*";
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<char>)))
        return "std::vector<char>";
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<signed char>)))
        return "std::vector<signed char>";
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<unsigned char>)))
        return "std::vector<unsigned char>";
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<short>)))
        return "std::vector<short>";
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<unsigned short>)))
        return "std::vector<unsigned short>";
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<int>)))
        return "std::vector<int>";
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<unsigned int>)))
        return "std::vector<unsigned int>";
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<long>)))
        return "std::vector<long>";
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<unsigned long>)))
        return "std::vector<unsigned long>";
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<long long>)))
        return "std::vector<long long>";
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<unsigned long>)))
        return "std::vector<unsigned long long>";
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<bool>)))
        return "std::vector<bool>";
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<float>)))
        return "std::vector<float>";
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<double>)))
        return "std::vector<double>";
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<long double>)))
        return "std::vector<long double>";
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<std::string>)))
        return "std::vector<std::string>";
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<char*>)))
        return "std::vector<char*>";
    return std::string(t_info -> name());
}

template <typename T>
T strtoT_simple(
    const char *str, char **str_end, int base = 0)
{
    Error_UnsupportedType();
}

template <> long strtoT_simple <long>(
    const char *str, char **str_end, int base)
{
    return strtol(str, str_end, base);
}

template <> unsigned long strtoT_simple <unsigned long>(
    const char *str, char **str_end, int base)
{
    return strtoul(str, str_end, base);
}

template <> long long strtoT_simple <long long>(
    const char *str, char **str_end, int base)
{
    return strtoll(str, str_end, base);
}

template <> unsigned long long strtoT_simple <unsigned long long>(
    const char *str, char **str_end, int base)
{
    return strtoull(str, str_end, base);
}

template <> char strtoT_simple <char>(
    const char *str, char **str_end, int base)
{
    long val = strtoT_simple<long>(str, str_end, base);
    if (val < CHAR_MIN) val = CHAR_MIN;
    if (val > CHAR_MAX) val = CHAR_MAX;
    return (char) val;
}

template <> signed char strtoT_simple <signed char>(
    const char *str, char **str_end, int base)
{
    signed long val = strtoT_simple<signed long>(str, str_end, base);
    if (val < SCHAR_MIN) val = SCHAR_MIN;
    if (val > SCHAR_MAX) val = SCHAR_MAX;
    return (signed char) val;
}

template <> unsigned char strtoT_simple <unsigned char>(
    const char *str, char **str_end, int base)
{
    unsigned long val = strtoT_simple<unsigned long>(str, str_end, base);
    if (val > UCHAR_MAX) val = UCHAR_MAX;
    return (unsigned char) val;
}

template <> short strtoT_simple <short>(
    const char *str, char **str_end, int base)
{
    long val = strtoT_simple<long>(str, str_end, base);
    if (val < SHRT_MIN) val = SHRT_MIN;
    if (val > SHRT_MAX) val = SHRT_MAX;
    return (short)val;
}

template <> unsigned short strtoT_simple <unsigned short>(
    const char *str, char **str_end, int base)
{
    unsigned long val = strtoT_simple<unsigned long>(str, str_end, base);
    if (val > USHRT_MAX) val = USHRT_MAX;
    return (unsigned short)val;
}

template <> int strtoT_simple <int>(
    const char *str, char **str_end, int base)
{
    long val = strtoT_simple<long>(str, str_end, base);
    if (val < INT_MIN) val = INT_MIN;
    if (val > INT_MAX) val = INT_MAX;
    return (int)val;
}

template <> unsigned int strtoT_simple <unsigned int>(
    const char *str, char **str_end, int base)
{
    unsigned long val = strtoT_simple<unsigned long>(str, str_end, base);
    if (val > UINT_MAX) val = UINT_MAX;
    return (unsigned int)val;
}

template <> float strtoT_simple <float>(
    const char *str, char **str_end, int base)
{
    return strtof(str, str_end);
}

template <> double strtoT_simple <double>(
    const char *str, char **str_end, int base)
{
    return strtod(str, str_end);
}

template <> long double strtoT_simple <long double>(
    const char *str, char **str_end, int base)
{
    return strtold(str, str_end);
}

template <> bool strtoT_simple <bool>(
    const char *str, char **str_end, int base)
{
    unsigned int i = 0;
    unsigned int length = strlen(str);
    while (i < length)
    {
        if (isspace(str[i])) i++;
        else break;
    }

    if (i + 5 >= length)
    {
        if (tolower(str[i]) == 'f' && tolower(str[i+1]) == 'a'
            && tolower(str[i+2]) == 'l' && tolower(str[i+3]) == 's'
            && tolower(str[i+4]) == 'e')
        {
            *str_end = const_cast<char*>(str) + i + 5;
            return false;
        }
    }

    if (i + 4 >= length)
    {
        if (tolower(str[i]) == 't' && tolower(str[i+1]) == 'r'
            && tolower(str[i+2]) == 'u' && tolower(str[i+3]) == 'e')
        {
            *str_end = const_cast<char*>(str) + i + 4;
            return true;
        }
    }

    if (i + 1 >= length)
    {
        if (str[i] == '0' || tolower(str[i]) == 'f')
        {
            *str_end = const_cast<char*>(str) + i + 1;
            return false;
        }
        if (str[i] == '1' || tolower(str[i]) == 't')
        {
            *str_end = const_cast<char*>(str) + i + 1;
            return true;
        }
    }

    *str_end = const_cast<char*>(str) + i;
    return true;
}

template <> char* strtoT_simple <char*>(
    const char *str, char **str_end, int base)
{
    *str_end = const_cast<char*>(str) + strlen(str);
    return const_cast<char*>(str);
}

template <> std::string strtoT_simple <std::string>(
    const char *str, char **str_end, int base)
{
    *str_end = const_cast<char*>(str) + strlen(str);
    return std::string(str);
}

template <typename T, bool is_vector>
class VT_Selector {};

template <typename T>
class VT_Selector<T, true>
{
public:
    static T strtoT(const char *str, char **str_end, int base = 0)
    {
        typedef typename T::value_type Value_Type;
        T vec;
        unsigned int length = strlen(str);
        *str_end = const_cast<char*>(str) + length;
        char *temp_str = new char[length + 1];
        std::strncpy(temp_str, str, length + 1);
        for (unsigned int i=0; i<length; i++)
        {
            if (str[i] == ',') temp_str[i] = '\0';
        }

        char *item_end;
        Value_Type val;
        //std::cout << "Testing " << temp_str << std::endl;
        val = strtoT_simple<Value_Type>(temp_str, &item_end, base);
        vec.push_back(val);
        if (*item_end != '\0')
        {
            *str_end = const_cast<char*>(str) + (item_end - temp_str);
        } else
        {
            for (unsigned int i=0; i<length; i++)
            {
                if (str[i] != ',') continue;
                //std::cout << "Testing " << temp_str + i + 1<< std::endl;
                val = strtoT_simple<Value_Type>(temp_str + i + 1, &item_end, base);
                vec.push_back(val);
                if (*item_end != '\0')
                {
                    //std::cout << item_end - temp_str << "|" << int(*item_end) << std::endl;
                    *str_end = const_cast<char*>(str) + (item_end - temp_str);
                    break;
                }
            }
        }
        delete[] temp_str;temp_str = NULL;
        return vec;
    }
};

template <typename T>
class VT_Selector<T, false>
{
public:
    static T strtoT(const char *str, char **str_end, int base = 0)
    {
        return strtoT_simple<T>(str, str_end, base);
    }
};

template <typename>
struct IS_VECTOR : std::false_type {};

template <typename T, typename A>
struct IS_VECTOR <std::vector<T,A>> : std::true_type {};

template <typename T>
T strtoT(const char *str, char **str_end, int base = 0)
{
    return VT_Selector<T, IS_VECTOR<T>::value>
        ::strtoT(str, str_end, base);
}

template <typename T>
T strtoT(const std::string str, std::string &str_end, int base = 0)
{
    char *char_str_end;
    T val = strtoT<T>(str.c_str(), &char_str_end, base);
    str_end = std::string((char_str_end == NULL) ? "" : char_str_end);
    return val;
}

template <typename T>
bool isValidString(const char *str, int base = 0)
{
    char *str_end;
    strtoT<T>(str, &str_end, base);

    if (*str_end == '\0') return true;
    //std::cout << str_end - str << "|" << int(*str_end) << std::endl;
    return false;
}

template <typename T>
bool isValidString(const std::string str, int base = 0)
{
    return isValidString<T>(str.c_str(), base);
}

bool isValidString(const char *str, const std::type_info* t_info, int base = 0)
{
    if (std::type_index(*t_info) == std::type_index(typeid(char)))
        return isValidString<char>(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(signed char)))
        return isValidString<signed char>(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(unsigned char)))
        return isValidString<unsigned char>(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(short)))
        return isValidString<short>(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(unsigned short)))
        return isValidString<unsigned short>(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(int)))
        return isValidString<int>(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(unsigned int)))
        return isValidString<unsigned int>(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(long)))
        return isValidString<long>(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(unsigned long)))
        return isValidString<unsigned long>(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(long long)))
        return isValidString<long long>(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(unsigned long long)))
        return isValidString<unsigned long long>(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(float)))
        return isValidString<float>(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(double)))
        return isValidString<double>(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(long double)))
        return isValidString<long double>(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(bool)))
        return isValidString<bool>(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(std::string)))
        return isValidString<std::string>(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(char*)))
        return isValidString<char*>(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<char>)))
        return isValidString<std::vector<char> >(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<signed char>)))
        return isValidString<std::vector<signed char> >(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<unsigned char>)))
        return isValidString<std::vector<unsigned char> >(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<short>)))
        return isValidString<std::vector<short> >(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<unsigned short>)))
        return isValidString<std::vector<unsigned short> >(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<int>)))
        return isValidString<std::vector<int> >(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<unsigned int>)))
        return isValidString<std::vector<unsigned int> >(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<long>)))
        return isValidString<std::vector<long> >(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<unsigned long>)))
        return isValidString<std::vector<unsigned long> >(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<long long>)))
        return isValidString<std::vector<long long> >(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<unsigned long long>)))
        return isValidString<std::vector<unsigned long long> >(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<float>)))
        return isValidString<std::vector<float> >(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<double>)))
        return isValidString<std::vector<double> >(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<long double>)))
        return isValidString<std::vector<long double> >(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<bool>)))
        return isValidString<std::vector<bool> >(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<std::string>)))
        return isValidString<std::vector<std::string> >(str, base);
    if (std::type_index(*t_info) == std::type_index(typeid(std::vector<char*>)))
        return isValidString<std::vector<char*> >(str, base);
    return true;
}

bool isValidString(const std::string str, const std::type_info* t_info, int base = 0)
{
    return isValidString(str.c_str(), t_info, base);
}

} // namespace util
} // namespace gunrock

// enable stream to process vectors
template <typename T>
std::ostream& operator<< (std::ostream& sout, const std::vector<T>& vec)
{
    bool first_element = true;
    for (auto item : vec)
    {
        sout<< (first_element ? "" : ",") << item;
        first_element = false;
    }
    return sout;
}

template <>
std::ostream& operator<< (std::ostream& sout, const std::vector<bool>& vec)
{
    bool first_element = true;
    for (auto item : vec)
    {
        sout<< (first_element ? "" : ",") << (item ? "true" : "false");
        first_element = false;
    }
    return sout;
}

template <typename T>
std::istream& operator>> (std::istream& s_in, std::vector<T>& vec)
{
    vec.clear();
    std::string str = "", item = "";
    s_in >> str;
    for (unsigned int i = 0; i < str.length(); i++)
    {
        if (str[i] == ',')
        {
            std::istringstream istr(item);
            T val;
            istr >> val;
            vec.push_back(val);
            item = "";
        } else item = item + str[i];
    }

    std::istringstream istr(item);
    T val;
    istr >> val;
    vec.push_back(val);
    item = "";

    return s_in;
}

template <>
std::istream& operator>> (std::istream& s_in, std::vector<bool>& vec)
{
    vec.clear();
    std::string str = "", item = "";
    char *str_end;
    s_in >> str;
    for (unsigned int i = 0; i < str.length(); i++)
    {
        if (str[i] == ',')
        {
            vec.push_back( gunrock::util::strtoT<bool>(item.c_str(), &str_end));
            item = "";
        } else item = item + str[i];
    }

    vec.push_back( gunrock::util::strtoT<bool>(item.c_str(), &str_end));
    item = "";

    return s_in;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
