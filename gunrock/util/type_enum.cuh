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

#include <gunrock/util/vector_types.cuh>
#include <gunrock/util/basic_utils.h>

namespace gunrock {
namespace util {

using TypeId = unsigned int;

template <typename T>
struct Type2Enum {
  // static const TypeId Id = PreDefinedValues<TypeId>::InvalidValue;
};

template <>
struct Type2Enum<char> {
  static const TypeId Id = 0x01;
};
template <>
struct Type2Enum<unsigned char> {
  static const TypeId Id = 0x02;
};
template <>
struct Type2Enum<short> {
  static const TypeId Id = 0x03;
};
template <>
struct Type2Enum<unsigned short> {
  static const TypeId Id = 0x04;
};
template <>
struct Type2Enum<int> {
  static const TypeId Id = 0x05;
};
template <>
struct Type2Enum<unsigned int> {
  static const TypeId Id = 0x06;
};
template <>
struct Type2Enum<long> {
  static const TypeId Id = 0x07;
};
template <>
struct Type2Enum<unsigned long> {
  static const TypeId Id = 0x08;
};
template <>
struct Type2Enum<long long> {
  static const TypeId Id = 0x09;
};
template <>
struct Type2Enum<unsigned long long> {
  static const TypeId Id = 0x0A;
};
template <>
struct Type2Enum<float> {
  static const TypeId Id = 0x12;
};
template <>
struct Type2Enum<double> {
  static const TypeId Id = 0x14;
};
template <>
struct Type2Enum<std::string> {
  static const TypeId Id = 0x22;
};
template <>
struct Type2Enum<char*> {
  static const TypeId Id = 0x24;
};

template <>
struct Type2Enum<char1> {
  static const TypeId Id = Type2Enum<char>::Id + 0x000;
};
template <>
struct Type2Enum<uchar1> {
  static const TypeId Id = Type2Enum<unsigned char>::Id + 0x000;
};
template <>
struct Type2Enum<short1> {
  static const TypeId Id = Type2Enum<short>::Id + 0x000;
};
template <>
struct Type2Enum<ushort1> {
  static const TypeId Id = Type2Enum<unsigned short>::Id + 0x000;
};
template <>
struct Type2Enum<int1> {
  static const TypeId Id = Type2Enum<int>::Id + 0x000;
};
template <>
struct Type2Enum<uint1> {
  static const TypeId Id = Type2Enum<unsigned int>::Id + 0x000;
};
template <>
struct Type2Enum<long1> {
  static const TypeId Id = Type2Enum<long>::Id + 0x000;
};
template <>
struct Type2Enum<ulong1> {
  static const TypeId Id = Type2Enum<unsigned long>::Id + 0x000;
};
template <>
struct Type2Enum<longlong1> {
  static const TypeId Id = Type2Enum<long long>::Id + 0x000;
};
template <>
struct Type2Enum<ulonglong1> {
  static const TypeId Id = Type2Enum<unsigned long long>::Id + 0x000;
};
template <>
struct Type2Enum<float1> {
  static const TypeId Id = Type2Enum<float>::Id + 0x000;
};
template <>
struct Type2Enum<double1> {
  static const TypeId Id = Type2Enum<double>::Id + 0x000;
};

template <>
struct Type2Enum<char2> {
  static const TypeId Id = Type2Enum<char>::Id + 0x100;
};
template <>
struct Type2Enum<uchar2> {
  static const TypeId Id = Type2Enum<unsigned char>::Id + 0x100;
};
template <>
struct Type2Enum<short2> {
  static const TypeId Id = Type2Enum<short>::Id + 0x100;
};
template <>
struct Type2Enum<ushort2> {
  static const TypeId Id = Type2Enum<unsigned short>::Id + 0x100;
};
template <>
struct Type2Enum<int2> {
  static const TypeId Id = Type2Enum<int>::Id + 0x100;
};
template <>
struct Type2Enum<uint2> {
  static const TypeId Id = Type2Enum<unsigned int>::Id + 0x100;
};
template <>
struct Type2Enum<long2> {
  static const TypeId Id = Type2Enum<long>::Id + 0x100;
};
template <>
struct Type2Enum<ulong2> {
  static const TypeId Id = Type2Enum<unsigned long>::Id + 0x100;
};
template <>
struct Type2Enum<longlong2> {
  static const TypeId Id = Type2Enum<long long>::Id + 0x100;
};
template <>
struct Type2Enum<ulonglong2> {
  static const TypeId Id = Type2Enum<unsigned long long>::Id + 0x100;
};
template <>
struct Type2Enum<float2> {
  static const TypeId Id = Type2Enum<float>::Id + 0x100;
};
template <>
struct Type2Enum<double2> {
  static const TypeId Id = Type2Enum<double>::Id + 0x100;
};
template <>
struct Type2Enum<util::VectorType<std::string, 2> > {
  static const TypeId Id = Type2Enum<std::string>::Id + 0x100;
};
template <>
struct Type2Enum<util::VectorType<char*, 2> > {
  static const TypeId Id = Type2Enum<char*>::Id + 0x100;
};

template <>
struct Type2Enum<char3> {
  static const TypeId Id = Type2Enum<char>::Id + 0x200;
};
template <>
struct Type2Enum<uchar3> {
  static const TypeId Id = Type2Enum<unsigned char>::Id + 0x200;
};
template <>
struct Type2Enum<short3> {
  static const TypeId Id = Type2Enum<short>::Id + 0x200;
};
template <>
struct Type2Enum<ushort3> {
  static const TypeId Id = Type2Enum<unsigned short>::Id + 0x200;
};
template <>
struct Type2Enum<int3> {
  static const TypeId Id = Type2Enum<int>::Id + 0x200;
};
template <>
struct Type2Enum<uint3> {
  static const TypeId Id = Type2Enum<unsigned int>::Id + 0x200;
};
template <>
struct Type2Enum<long3> {
  static const TypeId Id = Type2Enum<long>::Id + 0x200;
};
template <>
struct Type2Enum<ulong3> {
  static const TypeId Id = Type2Enum<unsigned long>::Id + 0x200;
};
template <>
struct Type2Enum<longlong3> {
  static const TypeId Id = Type2Enum<long long>::Id + 0x200;
};
template <>
struct Type2Enum<ulonglong3> {
  static const TypeId Id = Type2Enum<unsigned long long>::Id + 0x200;
};
template <>
struct Type2Enum<float3> {
  static const TypeId Id = Type2Enum<float>::Id + 0x200;
};
template <>
struct Type2Enum<double3> {
  static const TypeId Id = Type2Enum<double>::Id + 0x200;
};
template <>
struct Type2Enum<util::VectorType<std::string, 3> > {
  static const TypeId Id = Type2Enum<std::string>::Id + 0x200;
};
template <>
struct Type2Enum<util::VectorType<char*, 3> > {
  static const TypeId Id = Type2Enum<char*>::Id + 0x200;
};

template <>
struct Type2Enum<char4> {
  static const TypeId Id = Type2Enum<char>::Id + 0x300;
};
template <>
struct Type2Enum<uchar4> {
  static const TypeId Id = Type2Enum<unsigned char>::Id + 0x300;
};
template <>
struct Type2Enum<short4> {
  static const TypeId Id = Type2Enum<short>::Id + 0x300;
};
template <>
struct Type2Enum<ushort4> {
  static const TypeId Id = Type2Enum<unsigned short>::Id + 0x300;
};
template <>
struct Type2Enum<int4> {
  static const TypeId Id = Type2Enum<int>::Id + 0x300;
};
template <>
struct Type2Enum<uint4> {
  static const TypeId Id = Type2Enum<unsigned int>::Id + 0x300;
};
template <>
struct Type2Enum<long4> {
  static const TypeId Id = Type2Enum<long>::Id + 0x300;
};
template <>
struct Type2Enum<ulong4> {
  static const TypeId Id = Type2Enum<unsigned long>::Id + 0x300;
};
template <>
struct Type2Enum<longlong4> {
  static const TypeId Id = Type2Enum<long long>::Id + 0x300;
};
template <>
struct Type2Enum<ulonglong4> {
  static const TypeId Id = Type2Enum<unsigned long long>::Id + 0x300;
};
template <>
struct Type2Enum<float4> {
  static const TypeId Id = Type2Enum<float>::Id + 0x300;
};
template <>
struct Type2Enum<double4> {
  static const TypeId Id = Type2Enum<double>::Id + 0x300;
};
template <>
struct Type2Enum<util::VectorType<std::string, 4> > {
  static const TypeId Id = Type2Enum<std::string>::Id + 0x300;
};
template <>
struct Type2Enum<util::VectorType<char*, 4> > {
  static const TypeId Id = Type2Enum<char*>::Id + 0x300;
};

// template <typename T2, int SIZE>
// struct Type2Enum<typename util::VectorType<T2, SIZE>::Type >
//{static const TypeId Id = Type2Enum<T2>::Id + 0x100 * (SIZE - 1);};

template <TypeId ID>
struct Enum2Type_Base {
  typedef NullType Type;
};
template <>
struct Enum2Type_Base<0x01> {
  typedef char Type;
};
template <>
struct Enum2Type_Base<0x02> {
  typedef unsigned char Type;
};
template <>
struct Enum2Type_Base<0x03> {
  typedef short Type;
};
template <>
struct Enum2Type_Base<0x04> {
  typedef unsigned short Type;
};
template <>
struct Enum2Type_Base<0x05> {
  typedef int Type;
};
template <>
struct Enum2Type_Base<0x06> {
  typedef unsigned int Type;
};
template <>
struct Enum2Type_Base<0x07> {
  typedef long Type;
};
template <>
struct Enum2Type_Base<0x08> {
  typedef unsigned long Type;
};
template <>
struct Enum2Type_Base<0x09> {
  typedef long long Type;
};
template <>
struct Enum2Type_Base<0x0A> {
  typedef unsigned long long Type;
};

template <>
struct Enum2Type_Base<0x12> {
  typedef float Type;
};
template <>
struct Enum2Type_Base<0x14> {
  typedef double Type;
};

template <>
struct Enum2Type_Base<0x22> {
  typedef std::string Type;
};
template <>
struct Enum2Type_Base<0x24> {
  typedef char* Type;
};

template <TypeId ID>
struct Enum2Type {
  typedef typename Enum2Type_Base<ID & 0xFF>::Type BaseType;
  typedef If<(ID <= 0xFF), BaseType,
             typename VectorType<BaseType, (ID >> 8) + 1>::Type>
      Type;
};

}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
