#ifndef GDF_BASIC_TYPES_HPP_
#define GDF_BASIC_TYPES_HPP_

#include "stdx.hpp"

#include <cstdint>
#include <array>
#include <typeinfo>
#include <type_traits>
#include <nvfunctional>


namespace gdf {

namespace detail {

namespace util {

template <std::size_t Offset, std::size_t... Idx, typename F, typename... Ts>
void visit_numeric_range(F f, stdx::index_sequence<Idx...>, std::size_t n, Ts... args) {
    static std::array<nvstd::function<void()>, sizeof...(Idx)> funcs {{
        [&f](){f(std::integral_constant<std::size_t,Idx>{});}...
    }};

    funcs[n](std::forward<Ts>(args)...);
};

} // namespace detail

/**
 * A mechanism for run-time switching over compile-time integral
 * values - typically the possible values of an enum / enum class.
 *
 * @note Unfortunately, it assumes the full range of values
 * between @tparam Start and @tparam End are potentially valud,
 * so this is not very usable for non-contiguous enums
 *
 * @note I don't like the use of std::function's - that's kind
 * of heavy-handed.
 *
 * @note The way to call this is to specify _just the first two_
 * template parameters, and let the third one be inferred. See:
 * http://coliru.stacked-crooked.com/a/d60df0e07818976e
 *
 * @tparam F type of the functor to apply
 * @tparam Start first enumerated value
 * @tparam End   last enumerated value
 *
 * @param f a functor (something with `operator()`), templated on
 * a numeric paramter (and _nothing_ else) - to call with
 * all possible enum values.
 * @param n The enum value for which we want to apply the
 * instantiated templated functor
 */
template <std::size_t Start, std::size_t End, typename F, typename... Ts>
void visit_numeric_range(F f, std::size_t n, Ts... args) {
    visit_enum<Start>(f, n, stdx::make_index_sequence<End - Start>{}, n, std::forward<Ts>(args)...);
};


} // namespace util

} // namespace gdf


#endif // GDF_BASIC_TYPES_HPP_
