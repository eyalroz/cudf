/**
 * @file Chooses between alternative definitions of types which
 * have eventually gone into the standard library.
 */

#ifndef STDX_HPP_
#define STDX_HPP_

// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------
//
// Span
//

#if __cplusplus >= 202001L
#include <span>
namespace stdx {

template <class ElementType, std::ptrdiff_t Extent = std::dynamic_extent>
using span = std::span<ElementType, Extent>;

} // namespace stdx
#else // __cplusplus >= 202001L

#include <gsl/span>

namespace stdx {

template <class ElementType, std::ptrdiff_t Extent = gsl::dynamic_extent>
using span = gsl::span<ElementType, Extent>;

} // namespace stdx
#endif // __cplusplus >= 202001L

// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------
//
// Optional
//


// TODO: Optional (Boost / experimental / standard)
#if __cplusplus >= 201701L

#include <optional>

namespace stdx {

template <class T>
using optional = std::optional<T>;

using nullopt = std::nullopt;

} // namespace stdx

#else // __cplusplus >= 201701L

#include <tl/optional.hpp>

namespace stdx {

template <class T>
using optional = tl::optional<T>;

using nullopt = tl::nullopt;

} // namespace stdx

#endif // __cplusplus >= 201701L

// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------
//
// Variant
//

#if __cplusplus >= 201701L

namespace stdx {

template <class... Types>
using variant = std::variant;
// variant helper classes
template <class T>
using variant_size = std::variant_size; // not defined
template <class T>
using variant_size = std::variant_size<const T>;
template <class T>
using variant_size = std::variant_size<volatile T>;
template <class T>
using variant_size = std::variant_size<const volatile T>;


template <class... Types>
using variant_size = std::variant_size<variant<Types...>>;

template <size_t I, class T>
using variant_alternative = std::variant_alternative; // not defined
template <size_t I, class T>
using variant_alternative = std::variant_alternative<I, const T>;
template <size_t I, class T>
using variant_alternative = std::variant_alternative<I, volatile T>;
template <size_t I, class T>
using variant_alternative = std::variant_alternative<I, const volatile T>;
template <size_t I, class T>
using variant_alternative_t = typename std::variant_alternative<I, T>::type;
template <size_t I, class... Types>
using variant_alternative = std::variant_alternative<I, variant<Types...>>;

using variant_npos = std::variant_mpos;

// value access
template <class T, class... Types>
constexpr bool holds_alternative(const variant<Types...>& v) noexcept { return std::holds_alternative(v); }

template <size_t I, class... Types>
constexpr variant_alternative_t<I, variant<Types...>>&
get(variant<Types...>& v) { return std::holds_alternative(v); }
template <size_t I, class... Types>
constexpr variant_alternative_t<I, variant<Types...>>&&
get(variant<Types...>&& v) { return std::holds_alternative(v); }
template <size_t I, class... Types>
constexpr const variant_alternative_t<I, variant<Types...>>&
get(const variant<Types...>& v) { return std::holds_alternative(v); }
template <size_t I, class... Types>
constexpr const variant_alternative_t<I, variant<Types...>>&&
get(const variant<Types...>&& v) { return std::holds_alternative(v); }
template <class T, class... Types>
constexpr T& get(variant<Types...>& v) { return std::holds_alternative(v); }
template <class T, class... Types>
constexpr T&& get(variant<Types...>&& v) { return std::holds_alternative(v); }
template <class T, class... Types>
constexpr const T& get(const variant<Types...>& v) { return std::holds_alternative(v); }
template <class T, class... Types>
constexpr const T&& get(const variant<Types...>&& v) { return std::holds_alternative(v); }
template <size_t I, class... Types>

constexpr add_pointer_t<variant_alternative_t<I, variant<Types...>>>
get_if(variant<Types...>* pv) noexcept
{
    return std::get_if(pv);
}
template <size_t I, class... Types>
constexpr add_pointer_t<const variant_alternative_t<I, variant<Types...>>>
get_if(const variant<Types...>* pv) noexcept
{
    return std::get_if(pv);
}
template <class T, class... Types>
inline constexpr add_pointer_t<T> get_if(variant<Types...>* pv) noexcept
{
    return std::get_if(pv);
}
template <class T, class... Types>
inline constexpr add_pointer_t<const T> get_if(const variant<Types...>* pv) noexcept
{
    return std::get_if(pv);
}

using monostate = std::monostate;
using bad_variant_access = class bad_variant_access;
template <class T>
using hash = std::hash<T>;

} // namespace stdx


#else // __cplusplus >= 201701L
#include <mpark/variant.hpp>

namespace stdx {

template <class... Types>
using variant = mpark::variant;
// variant helper classes
template <class T>
using variant_size = mpark::variant_size; // not defined
template <class T>
using variant_size = mpark::variant_size<const T>;
template <class T>
using variant_size = mpark::variant_size<volatile T>;
template <class T>
using variant_size = mpark::variant_size<const volatile T>;


template <class... Types>
using variant_size = mpark::variant_size<variant<Types...>>;

template <size_t I, class T>
using variant_alternative = mpark::variant_alternative; // not defined
template <size_t I, class T>
using variant_alternative = mpark::variant_alternative<I, const T>;
template <size_t I, class T>
using variant_alternative = mpark::variant_alternative<I, volatile T>;
template <size_t I, class T>
using variant_alternative = mpark::variant_alternative<I, const volatile T>;
template <size_t I, class T>
using variant_alternative_t = typename mpark::variant_alternative<I, T>::type;
template <size_t I, class... Types>
using variant_alternative = mpark::variant_alternative<I, variant<Types...>>;

using variant_npos = mpark::variant_mpos;

// value access
template <class T, class... Types>
constexpr bool holds_alternative(const variant<Types...>& v) noexcept { return mpark::holds_alternative(v); }

template <size_t I, class... Types>
constexpr variant_alternative_t<I, variant<Types...>>&
get(variant<Types...>& v) { return mpark::holds_alternative(v); }
template <size_t I, class... Types>
constexpr variant_alternative_t<I, variant<Types...>>&&
get(variant<Types...>&& v) { return mpark::holds_alternative(v); }
template <size_t I, class... Types>
constexpr const variant_alternative_t<I, variant<Types...>>&
get(const variant<Types...>& v) { return mpark::holds_alternative(v); }
template <size_t I, class... Types>
constexpr const variant_alternative_t<I, variant<Types...>>&&
get(const variant<Types...>&& v) { return mpark::holds_alternative(v); }
template <class T, class... Types>
constexpr T& get(variant<Types...>& v) { return mpark::holds_alternative(v); }
template <class T, class... Types>
constexpr T&& get(variant<Types...>&& v) { return mpark::holds_alternative(v); }
template <class T, class... Types>
constexpr const T& get(const variant<Types...>& v) { return mpark::holds_alternative(v); }
template <class T, class... Types>
constexpr const T&& get(const variant<Types...>&& v) { return mpark::holds_alternative(v); }
template <size_t I, class... Types>

constexpr add_pointer_t<variant_alternative_t<I, variant<Types...>>>
get_if(variant<Types...>* pv) noexcept
{
    return mpark::get_if(pv);
}
template <size_t I, class... Types>
constexpr add_pointer_t<const variant_alternative_t<I, variant<Types...>>>
get_if(const variant<Types...>* pv) noexcept
{
    return mpark::get_if(pv);
}
template <class T, class... Types>
inline constexpr add_pointer_t<T> get_if(variant<Types...>* pv) noexcept
{
    return mpark::get_if(pv);
}
template <class T, class... Types>
inline constexpr add_pointer_t<const T> get_if(const variant<Types...>* pv) noexcept
{
    return mpark::get_if(pv);
}

using monostate = mpark::monostate;
using bad_variant_access = class bad_variant_access;
template <class T>
using hash = mpark::hash<T>;

} // namespace stdx
#endif // __cplusplus >= 201701L

// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------
//
// TODO: Expected
//

// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------
//
// Index Sequence

#include <cstddef> // for size_t
#include <utility>

#if __cplusplus >= 201402L
namespace stdx {

template <size_t... Ints>
using index_sequence = std::index_sequence<Ints...>;

template <size_t N>
using make_index_sequence = std::make_index_sequence<N>;

} // namespace stdx

#else // __cplusplus >= 201402L

namespace stdx {
    template <size_t... Ints>
    struct index_sequence
    {
        using type = index_sequence;
        using value_type = size_t;
        static constexpr std::size_t size() noexcept { return sizeof...(Ints); }
    };

    // --------------------------------------------------------------

    template <class Sequence1, class Sequence2>
    struct _merge_and_renumber;

    template <size_t... I1, size_t... I2>
    struct _merge_and_renumber<index_sequence<I1...>, index_sequence<I2...>>
      : index_sequence<I1..., (sizeof...(I1)+I2)...>
    { };

    // --------------------------------------------------------------

    template <size_t N>
    struct make_index_sequence
      : _merge_and_renumber<typename make_index_sequence<N/2>::type,
                            typename make_index_sequence<N - N/2>::type>
    { };

    template<> struct make_index_sequence<0> : index_sequence<> { };
    template<> struct make_index_sequence<1> : index_sequence<0> { };

    template<typename... Ts>
    using index_sequence_for = make_index_sequence<sizeof...(Ts)>;
} // namespace stdx

#endif // __cplusplus >= 201402L



#endif // STDX_HPP_
