// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * str_to_T.cu
 *
 * @brief string to T implementations
 */

//#pragma once

#include <gunrock/util/str_to_T.cuh>

namespace gunrock {
namespace util {

template <>
long strtoT_simple<long>(const char *str, char **str_end, int base) {
  return strtol(str, str_end, base);
}

template <>
unsigned long strtoT_simple<unsigned long>(const char *str, char **str_end,
                                           int base) {
  return strtoul(str, str_end, base);
}

template <>
long long strtoT_simple<long long>(const char *str, char **str_end, int base) {
  return strtoll(str, str_end, base);
}

template <>
unsigned long long strtoT_simple<unsigned long long>(const char *str,
                                                     char **str_end, int base) {
  return strtoull(str, str_end, base);
}

template <>
char strtoT_simple<char>(const char *str, char **str_end, int base) {
  long val = strtoT_simple<long>(str, str_end, base);
  if (val < CHAR_MIN) val = CHAR_MIN;
  if (val > CHAR_MAX) val = CHAR_MAX;
  return (char)val;
}

template <>
signed char strtoT_simple<signed char>(const char *str, char **str_end,
                                       int base) {
  signed long val = strtoT_simple<signed long>(str, str_end, base);
  if (val < SCHAR_MIN) val = SCHAR_MIN;
  if (val > SCHAR_MAX) val = SCHAR_MAX;
  return (signed char)val;
}

template <>
unsigned char strtoT_simple<unsigned char>(const char *str, char **str_end,
                                           int base) {
  unsigned long val = strtoT_simple<unsigned long>(str, str_end, base);
  if (val > UCHAR_MAX) val = UCHAR_MAX;
  return (unsigned char)val;
}

template <>
short strtoT_simple<short>(const char *str, char **str_end, int base) {
  long val = strtoT_simple<long>(str, str_end, base);
  if (val < SHRT_MIN) val = SHRT_MIN;
  if (val > SHRT_MAX) val = SHRT_MAX;
  return (short)val;
}

template <>
unsigned short strtoT_simple<unsigned short>(const char *str, char **str_end,
                                             int base) {
  unsigned long val = strtoT_simple<unsigned long>(str, str_end, base);
  if (val > USHRT_MAX) val = USHRT_MAX;
  return (unsigned short)val;
}

template <>
int strtoT_simple<int>(const char *str, char **str_end, int base) {
  long val = strtoT_simple<long>(str, str_end, base);
  if (val < INT_MIN) val = INT_MIN;
  if (val > INT_MAX) val = INT_MAX;
  return (int)val;
}

template <>
unsigned int strtoT_simple<unsigned int>(const char *str, char **str_end,
                                         int base) {
  unsigned long val = strtoT_simple<unsigned long>(str, str_end, base);
  if (val > UINT_MAX) val = UINT_MAX;
  return (unsigned int)val;
}

template <>
float strtoT_simple<float>(const char *str, char **str_end, int base) {
  return strtof(str, str_end);
}

template <>
double strtoT_simple<double>(const char *str, char **str_end, int base) {
  return strtod(str, str_end);
}

template <>
long double strtoT_simple<long double>(const char *str, char **str_end,
                                       int base) {
  return strtold(str, str_end);
}

template <>
bool strtoT_simple<bool>(const char *str, char **str_end, int base) {
  unsigned int i = 0;
  unsigned int length = strlen(str);
  while (i < length) {
    if (isspace(str[i]))
      i++;
    else
      break;
  }

  if (i + 5 <= length) {
    // std::cout << "Cond 1" << std::endl;
    if (tolower(str[i]) == 'f' && tolower(str[i + 1]) == 'a' &&
        tolower(str[i + 2]) == 'l' && tolower(str[i + 3]) == 's' &&
        tolower(str[i + 4]) == 'e') {
      *str_end = const_cast<char *>(str) + i + 5;
      return false;
    }
  }

  if (i + 4 <= length) {
    // std::cout << "Cond 2" << std::endl;
    if (tolower(str[i]) == 't' && tolower(str[i + 1]) == 'r' &&
        tolower(str[i + 2]) == 'u' && tolower(str[i + 3]) == 'e') {
      *str_end = const_cast<char *>(str) + i + 4;
      return true;
    }
  }

  if (i + 1 <= length) {
    // std::cout << "Cond 3" << std::endl;
    if (str[i] == '0' || tolower(str[i]) == 'f') {
      *str_end = const_cast<char *>(str) + i + 1;
      return false;
    }
    if (str[i] == '1' || tolower(str[i]) == 't') {
      *str_end = const_cast<char *>(str) + i + 1;
      return true;
    }
  }

  *str_end = const_cast<char *>(str) + i;
  return true;
}

template <>
char *strtoT_simple<char *>(const char *str, char **str_end, int base) {
  *str_end = const_cast<char *>(str) + strlen(str);
  return const_cast<char *>(str);
}

template <>
std::string strtoT_simple<std::string>(const char *str, char **str_end,
                                       int base) {
  *str_end = const_cast<char *>(str) + strlen(str);
  return std::string(str);
}

std::string TypeName(const std::type_info *t_info) {
  if (std::type_index(*t_info) == std::type_index(typeid(char))) return "char";
  if (std::type_index(*t_info) == std::type_index(typeid(signed char)))
    return "signed char";
  if (std::type_index(*t_info) == std::type_index(typeid(unsigned char)))
    return "unsigned char";
  if (std::type_index(*t_info) == std::type_index(typeid(short)))
    return "short";
  if (std::type_index(*t_info) == std::type_index(typeid(unsigned short)))
    return "unsigned short";
  if (std::type_index(*t_info) == std::type_index(typeid(int))) return "int";
  if (std::type_index(*t_info) == std::type_index(typeid(unsigned int)))
    return "unsigned int";
  if (std::type_index(*t_info) == std::type_index(typeid(long))) return "long";
  if (std::type_index(*t_info) == std::type_index(typeid(unsigned long)))
    return "unsigned long";
  if (std::type_index(*t_info) == std::type_index(typeid(long long)))
    return "long long";
  if (std::type_index(*t_info) == std::type_index(typeid(unsigned long)))
    return "unsigned long long";
  if (std::type_index(*t_info) == std::type_index(typeid(bool))) return "bool";
  if (std::type_index(*t_info) == std::type_index(typeid(float)))
    return "float";
  if (std::type_index(*t_info) == std::type_index(typeid(double)))
    return "double";
  if (std::type_index(*t_info) == std::type_index(typeid(long double)))
    return "long double";
  if (std::type_index(*t_info) == std::type_index(typeid(std::string)))
    return "std::string";
  if (std::type_index(*t_info) == std::type_index(typeid(char *)))
    return "char*";
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<char>)))
    return "std::vector<char>";
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<signed char>)))
    return "std::vector<signed char>";
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<unsigned char>)))
    return "std::vector<unsigned char>";
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<short>)))
    return "std::vector<short>";
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<unsigned short>)))
    return "std::vector<unsigned short>";
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<int>)))
    return "std::vector<int>";
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<unsigned int>)))
    return "std::vector<unsigned int>";
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<long>)))
    return "std::vector<long>";
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<unsigned long>)))
    return "std::vector<unsigned long>";
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<long long>)))
    return "std::vector<long long>";
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<unsigned long>)))
    return "std::vector<unsigned long long>";
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<bool>)))
    return "std::vector<bool>";
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<float>)))
    return "std::vector<float>";
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<double>)))
    return "std::vector<double>";
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<long double>)))
    return "std::vector<long double>";
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<std::string>)))
    return "std::vector<std::string>";
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<char *>)))
    return "std::vector<char*>";
  return std::string(t_info->name());
}

bool isVector(const std::type_info *t_info) {
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<char>)))
    return true;
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<signed char>)))
    return true;
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<unsigned char>)))
    return true;
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<short>)))
    return true;
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<unsigned short>)))
    return true;
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<int>)))
    return true;
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<unsigned int>)))
    return true;
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<long>)))
    return true;
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<unsigned long>)))
    return true;
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<long long>)))
    return true;
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<unsigned long long>)))
    return true;
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<float>)))
    return true;
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<double>)))
    return true;
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<long double>)))
    return true;
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<bool>)))
    return true;
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<std::string>)))
    return true;
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<char *>)))
    return true;
  return false;
}

const std::type_info *toVector(const std::type_info *t_info) {
  if (std::type_index(*t_info) == std::type_index(typeid(char)))
    return &typeid(std::vector<char>);
  if (std::type_index(*t_info) == std::type_index(typeid(signed char)))
    return &typeid(std::vector<signed char>);
  if (std::type_index(*t_info) == std::type_index(typeid(unsigned char)))
    return &typeid(std::vector<unsigned char>);
  if (std::type_index(*t_info) == std::type_index(typeid(short)))
    return &typeid(std::vector<short>);
  if (std::type_index(*t_info) == std::type_index(typeid(unsigned short)))
    return &typeid(std::vector<unsigned short>);
  if (std::type_index(*t_info) == std::type_index(typeid(int)))
    return &typeid(std::vector<int>);
  if (std::type_index(*t_info) == std::type_index(typeid(unsigned int)))
    return &typeid(std::vector<unsigned int>);
  if (std::type_index(*t_info) == std::type_index(typeid(long)))
    return &typeid(std::vector<long>);
  if (std::type_index(*t_info) == std::type_index(typeid(unsigned long)))
    return &typeid(std::vector<unsigned long>);
  if (std::type_index(*t_info) == std::type_index(typeid(long long)))
    return &typeid(std::vector<long long>);
  if (std::type_index(*t_info) == std::type_index(typeid(unsigned long long)))
    return &typeid(std::vector<unsigned long long>);
  if (std::type_index(*t_info) == std::type_index(typeid(float)))
    return &typeid(std::vector<float>);
  if (std::type_index(*t_info) == std::type_index(typeid(double)))
    return &typeid(std::vector<double>);
  if (std::type_index(*t_info) == std::type_index(typeid(long double)))
    return &typeid(std::vector<long double>);
  if (std::type_index(*t_info) == std::type_index(typeid(bool)))
    return &typeid(std::vector<bool>);
  if (std::type_index(*t_info) == std::type_index(typeid(std::string)))
    return &typeid(std::vector<std::string>);
  if (std::type_index(*t_info) == std::type_index(typeid(char *)))
    return &typeid(std::vector<char *>);
  return NULL;
}

bool isValidString(const char *str, const std::type_info *t_info, int base) {
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
  if (std::type_index(*t_info) == std::type_index(typeid(char *)))
    return isValidString<char *>(str, base);
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<char>)))
    return isValidString<std::vector<char> >(str, base);
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<signed char>)))
    return isValidString<std::vector<signed char> >(str, base);
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<unsigned char>)))
    return isValidString<std::vector<unsigned char> >(str, base);
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<short>)))
    return isValidString<std::vector<short> >(str, base);
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<unsigned short>)))
    return isValidString<std::vector<unsigned short> >(str, base);
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<int>)))
    return isValidString<std::vector<int> >(str, base);
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<unsigned int>)))
    return isValidString<std::vector<unsigned int> >(str, base);
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<long>)))
    return isValidString<std::vector<long> >(str, base);
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<unsigned long>)))
    return isValidString<std::vector<unsigned long> >(str, base);
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<long long>)))
    return isValidString<std::vector<long long> >(str, base);
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<unsigned long long>)))
    return isValidString<std::vector<unsigned long long> >(str, base);
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<float>)))
    return isValidString<std::vector<float> >(str, base);
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<double>)))
    return isValidString<std::vector<double> >(str, base);
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<long double>)))
    return isValidString<std::vector<long double> >(str, base);
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<bool>)))
    return isValidString<std::vector<bool> >(str, base);
  if (std::type_index(*t_info) ==
      std::type_index(typeid(std::vector<std::string>)))
    return isValidString<std::vector<std::string> >(str, base);
  if (std::type_index(*t_info) == std::type_index(typeid(std::vector<char *>)))
    return isValidString<std::vector<char *> >(str, base);
  return true;
}

bool isValidString(const std::string str, const std::type_info *t_info,
                   int base) {
  return isValidString(str.c_str(), t_info, base);
}

}  // namespace util
}  // namespace gunrock

template <>
std::ostream &operator<<(std::ostream &sout, const std::vector<bool> &vec) {
  bool first_element = true;
  for (auto item : vec) {
    sout << (first_element ? "" : ",") << (item ? "true" : "false");
    first_element = false;
  }
  return sout;
}

template <>
std::istream &operator>>(std::istream &s_in, std::vector<bool> &vec) {
  vec.clear();
  std::string str = "", item = "";
  char *str_end;
  s_in >> str;
  for (unsigned int i = 0; i < str.length(); i++) {
    if (str[i] == ',') {
      vec.push_back(gunrock::util::strtoT<bool>(item.c_str(), &str_end));
      item = "";
    } else
      item = item + str[i];
  }

  vec.push_back(gunrock::util::strtoT<bool>(item.c_str(), &str_end));
  item = "";

  return s_in;
}
// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
