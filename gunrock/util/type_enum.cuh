// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * type_enum.cuh
 *
 * @brief type enumulation
 */

#pragma once

namespace gunrock {
namespace util {

using TypeId = unsigned int;

template <typename T>
struct Type2Enum
{
    const TypeId Id = PreDefinedValues<TypeId>::InvalidValue;
};

template <> struct Type2Enum<char>{static const TypeId Id = 0x01;};
template <> struct Type2Enum<unsigned char>{static const TypeId Id = 0x02;};
template <> struct Type2Enum<short>{static const TypeId Id = 0x03;};
template <> struct Type2Enum<unsigned short>{static const TypeId Id = 0x04;};
template <> struct Type2Enum<int>{static const TypeId Id = 0x05;};
template <> struct Type2Enum<unsigned int>{static const TypeId Id = 0x06;};
template <> struct Type2Enum<long>{static const TypeId Id = 0x07;};
template <> struct Type2Enum<unsigned long>{static const TypeId Id = 0x08;};
template <> struct Type2Enum<long long>{static const TypeId Id = 0x09;};
template <> struct Type2Enum<unsigned long long>{static const TypeId Id = 0x0A;};

template <> struct Type2Enum<float>{static const TypeId Id = 0x12;};
template <> struct Type2Enum<double>{static const TypeId Id = 0x14;};

template <> struct Type2Enum<std::string>{static const TypeId Id = 0x22;};
template <> struct Type2Enum<char*>{static const TypeId Id = 0x24;};

template <TypeId id>
struct Enum2Type
{
    typedef NullType Type;
};

template <> struct Enum2Type<0x01>{typedef char Type;};
template <> struct Enum2Type<0x02>{typedef unsigned char Type;};
template <> struct Enum2Type<0x03>{typedef short Type;};
template <> struct Enum2Type<0x04>{typedef unsigned short Type;};
template <> struct Enum2Type<0x05>{typedef int Type;};
template <> struct Enum2Type<0x06>{typedef unsigned int Type;};
template <> struct Enum2Type<0x07>{typedef long Type;};
template <> struct Enum2Type<0x08>{typedef unsigned long Type;};
template <> struct Enum2Type<0x09>{typedef long long Type;};
template <> struct Enum2Type<0x0A>{typedef unsigned long long Type;};

template <> struct Enum2Type<0x12>{typedef float Type;};
template <> struct Enum2Type<0x14>{typedef double Type;};

template <> struct Enum2Type<0x22>{typedef std::string Type;};
template <> struct Enum2Type<0x24>{typedef char* Type;};
} // namespace util
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
