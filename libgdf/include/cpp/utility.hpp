#ifndef GDF_BASIC_TYPES_HPP_
#define GDF_BASIC_TYPES_HPP_

#include "stdx.hpp"

#include <cstdint>
#include <array>
#include <typeinfo>
#include <type_traits>


namespace gdf {
namespace util {
// The contents of this namespaced block allow for
// determining whether a parameter pack (of types)
// has some other type as a member

namespace detail {

struct base_one { enum { value = 1 }; };
struct derived_zero : base_one { enum { value = 0 }; };

template< typename A, typename B >
struct type_equal {
 typedef derived_zero type;
};

template< typename A >
struct type_equal< A, A > {
 typedef base_one type;
};

} // namespace detail

template< typename Key, typename ... Types >
struct any_in_pack {
 enum { value =
     std::common_type< typename detail::type_equal< Key, Types >::type ... >::type::value };
};

namespace detail {


template <std::size_t Offset, std::size_t... Idx, typename F>
void visit_numeric_range(F f, std::size_t n, stdx::index_sequence<Idx...>) {
    static std::array<std::function<void()>, sizeof...(Idx)> funcs {{
        [&f](){f(std::integral_constant<std::size_t,Idx>{});}...
    }};

    funcs[n]();
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
template <std::size_t Start, std::size_t End, typename F>
void visit_numeric_range(F f, std::size_t n) {
    visit_enum<Start>(f, n, stdx::make_index_sequence<End - Start>{});
};


} // namespace util

} // namespace gdf


#endif // GDF_BASIC_TYPES_HPP_
