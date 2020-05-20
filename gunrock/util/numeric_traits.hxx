/**
 * @file numeric_traits.cuh
 *
 * @brief type traits for numeric types
 */

#pragma once

namespace gunrock {
namespace util {

enum representation_t
{
  not_a_number,
  signed_interger,
  unsigned_interger,
  floating_point
};

template<representation_t R>
struct base_traits
{
  static const representation_t REPRESENTATION = R;
};

// Default, non-numeric types
template<typename type_t>
struct numeric_traits : base_traits<not_a_number>
{};

template<>
struct numeric_traits<char> : base_traits<signed_interger>
{};
template<>
struct numeric_traits<signed char> : base_traits<signed_interger>
{};
template<>
struct numeric_traits<short> : base_traits<signed_interger>
{};
template<>
struct numeric_traits<int> : base_traits<signed_interger>
{};
template<>
struct numeric_traits<long> : base_traits<signed_interger>
{};
template<>
struct numeric_traits<long long> : base_traits<signed_interger>
{};

template<>
struct numeric_traits<unsigned char> : base_traits<unsigned_interger>
{};
template<>
struct numeric_traits<unsigned short> : base_traits<unsigned_interger>
{};
template<>
struct numeric_traits<unsigned int> : base_traits<unsigned_interger>
{};
template<>
struct numeric_traits<unsigned long> : base_traits<unsigned_interger>
{};
template<>
struct numeric_traits<unsigned long long> : base_traits<unsigned_interger>
{};

template<>
struct numeric_traits<float> : base_traits<floating_point>
{};
template<>
struct numeric_traits<double> : base_traits<floating_point>
{};

} // namespace util
} // namespace gunrock