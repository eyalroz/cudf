/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Eyal Rozenberg <eyalroz@blazingdb.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef UTIL_MISCELLANY_HPP_
#define UTIL_MISCELLANY_HPP_

#include <cstdlib> // for std::div

extern "C" {
#include <cudf.h>
}

#include <utilities/type_dispatcher.hpp>

namespace cudf {

constexpr inline bool is_an_integer(gdf_dtype element_type)
{
    return
        element_type == GDF_INT8  or
        element_type == GDF_INT16 or
        element_type == GDF_INT32 or
        element_type == GDF_INT64;
}

constexpr inline bool is_integral(const gdf_column& col)
{
    return is_an_integer(col.dtype);
}

constexpr inline bool is_boolean(const gdf_column& col)
{
    return col.dtype == GDF_INT8; // For now!
}


constexpr bool is_nullable(const gdf_column& column)
{
    return column.valid != nullptr;
}



namespace detail {

struct size_of_helper {
    template <typename T>
    constexpr int operator()() const { return sizeof(T); }
};

}

constexpr std::size_t inline size_of(gdf_dtype element_type) {
    return type_dispatcher(element_type, detail::size_of_helper{});
}

inline std::size_t width(const gdf_column& col)
{
    return size_of(col.dtype);
}

inline std::size_t data_size_in_bytes(const gdf_column& col)
{
    return col.size * width(col);
}


namespace util {

/**
* Divides the left-hand-side by the right-hand-side, rounding up
* to an integral multiple of the right-hand-side, e.g. (9,5) -> 2 , (10,5) -> 2, (11,5) -> 3.
*
* @param dividend the number to divide
* @param divisor the number of by which to divide
* @return The least integer multiple of {@link divisor} which is greater-or-equal to
* the non-integral division dividend/divisor.
*
* @note sensitive to overflow, i.e. if dividend > std::numeric_limits<S>::max() - divisor,
* the result will be incorrect
*/
template <typename S, typename T>
constexpr inline S div_rounding_up_unsafe(const S& dividend, const T& divisor) {
    return (dividend + divisor - 1) / divisor;
}


/**
* Divides the left-hand-side by the right-hand-side, rounding up
* to an integral multiple of the right-hand-side, e.g. (9,5) -> 2 , (10,5) -> 2, (11,5) -> 3.
*
* @param dividend the number to divide
* @param divisor the number of by which to divide
* @return The least integer multiple of {@link divisor} which is greater-or-equal to
* the non-integral division dividend/divisor.
*
* @note will not overflow, and may _or may not_ be slower than the intuitive
* approach of using (dividend + divisor - 1) / divisor
*/
template <typename I>
constexpr inline I div_rounding_up_safe(I dividend, I divisor);

template <typename I>
constexpr inline typename std::enable_if<std::is_signed<I>::value, I>::type
div_rounding_up_safe(I dividend, I divisor)
{
#if cplusplus >= 201402L
    auto div_result = std::div(dividend, divisor);
    return div_result.quot + !(!div_result.rem);
#else
    // Hopefully the compiler will optimize the two calls away.
    return std::div(dividend, divisor).quot + !(!std::div(dividend, divisor).rem);
#endif
}

// This variant will be used for unsigned types
template <typename I>
constexpr inline I div_rounding_up_safe(I dividend, I divisor)
{
    // TODO: This could probably be implemented faster
    return (dividend > divisor) ?
        1 + div_rounding_up_unsafe(dividend - divisor, divisor) :
        (dividend > 0);
}

template <typename T, template <typename S> class Trait>
using having_trait_t = typename std::enable_if_t<Trait<T>::value>;

// TODO: Use enable_if_T or having_trait_t to only allow the following
// to be instantiated for integral types I
template <typename I>
constexpr inline bool
is_a_power_of_two(I val)
    // util::having_trait_t<I, std::is_integral> val)
{
    return ((val - 1) & val) == 0;
}

template <typename T>
constexpr inline std::size_t size_in_bits() { return sizeof(T) * CHAR_BIT; }

template <typename T>
constexpr inline std::size_t size_in_bits(const T&) { return size_in_bits<T>(); }

// TODO: Use enable_if_T or having_trait_t to only allow the following
// to be instantiated for integral types I
template <typename I>
constexpr inline I
clear_lower_bits_unsafe(
//    util::having_trait_t<I, std::is_integral>  val,
    I                                          val,
    unsigned                                   num_bits_to_clear)
{
    auto lower_bits_mask = I{1} << (num_bits_to_clear - 1);
    return val & ~lower_bits_mask;
}


// TODO: Use enable_if_T or having_trait_t to only allow the following
// to be instantiated for integral types I
template <typename I>
constexpr inline I
clear_lower_bits_safe(
//    util::having_trait_t<I, std::is_integral> val,
    I                                           val,
    unsigned                                    num_bits_to_clear)
{
    return (num_bits_to_clear > 0) ?
        clear_lower_bits_unsafe(val, num_bits_to_clear) : val;
}

} // namespace util

// TODO: Use the cuda-api-wrappers library instead
inline auto form_naive_1d_grid(
    unsigned int grid_size,
    unsigned int threads_per_block)
{
    struct one_dimensional_grid_params_t {
        unsigned int num_blocks;
        unsigned int threads_per_block;
    };
    auto num_blocks = util::div_rounding_up_safe(grid_size, threads_per_block);
    return one_dimensional_grid_params_t { num_blocks, threads_per_block };
}


} // namespace cudf


#endif // UTIL_MISCELLANY_HPP_
