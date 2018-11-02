#ifndef GDF_COLUMN_HPP_
#define GDF_COLUMN_HPP_

extern "C" {
#include <gdf/cffi/types.h>
}

#include "element_data_types.hpp"
#include <cpp/utility.hpp>

#include <gsl/span>
#include <gsl/string_span>
#include <tl/optional.hpp>
#include <mpark/variant.hpp>

#include <type_traits>

namespace gdf {

using size  = gdf_size_type;
using index = gdf_index_type;

enum nullability : bool {
    is_not_nullable = false,
    is_nullable = true
};

namespace column {

class type_erased;

namespace detail {

template<typename... Ts>
struct type_erasure_helper {
    using span = mpark::variant<gsl::span< Ts > ... >;
    using column_wide_data = mpark::variant<column_wide_data< Ts > ... >;
};

using type_erasure_helper_for_gdf_column_types =
    type_erasure_helper<
        int8_t,
        int16_t,
        int32_t,
        int64_t,
        float32_t,
        float64_t,
        chrono::date<32>,
        chrono::date<64>,
        category,
        chrono::bare_timestamp
    >::span;

using var_elements_span = typename type_erasure_helper_for_gdf::span;
using var_column_wide_data = typename type_erasure_helper_for_gdf::column_wide_data;


inline void validate_gdf_column(const gdf_column& gc) {
    assert((gc.valid == nullptr or gc.null_count == 0) and "Invalid gdf_column: Non-zero null count, but no null indicators");
    assert((gc.data == nullptr or gc.size == 0) and "Invalid gdf_column: Non-zero size, but no element data");
    assert(gc.null_count <= gc.size and "Invalid gdf_column: Claims to have more NULLs than elements");
}

/**
 * In this namespace we define columns with no "frills" - no names,
 * no statistics, no pseudo-columns/auxiliary columns, no nullability
 * etc.
 */
namespace basic {

class type_erased;

/**
 * @brief A no-frills basic column structure where the element
 * data type is known at compile-time. Cf. @see `basic::type_erased`
 */
template <typename T>
class typed {
public: // non-mutator methods
    constexpr column_element_type element_type() const noexcept {
        return detail::column_element_type_to_enum<T>::value;
    }
    constexpr gdf::size size() const noexcept { return elements_.size_(); }
    const gsl::span<T>& elements() const noexcept { return elements_; }

public: // mutator methods
    gsl::span<T> elements() noexcept { return elements_; }

public: // constructors and destructor
    constexpr ~typed() = default;

    /**
     * @note For columns whose type does not require additional information to
     * the per-element data, the @p column_wide_data argument will necessarily
     * be an empty struct
     */
    constexpr typed(
        gsl::span<T>                elements,
        column_wide_data_t<T>       column_wide_data,
    ) :
        elements_(element), column_wide_data_(column_wide_data) { }

    /**
     * @note This constructor will only exist for columns whose type requires
     * no additional information to the per-element data.
     */
    constexpr typed(
        std::enable_if<std::is_empty<column_wide_data_t<T>>::value, gsl::span<T>>::type elements,
    ) :
        elements_(element), column_wide_data_({ }) { }

    constexpr typed(gdf_column_ col) :
        typed(gsl::span<T>{col.data, col.size}, column_wide_data::extract_from(col)) { }

    constexpr typed(const typed& other) = default;
    constexpr typed(typed&& other) = default;

protected: // constructors
    explicit typed(const type_erased& other) noexcept;
    explicit typed(type_erased&& other) noexcept; // TODO: Write me!

protected: // operators
    constexpr typed& operator=(typed&& other) = default;
    constexpr typed& operator=(const typed& other) = default;

protected: // data members
     const gsl::span<T>             elements_;
     const gdf::column_wide_data<T>  column_wide_data_;
}; // class gdf::basic::typed


/**
 * @brief A no-frills basic column structure where the element
 * data type is _not_ known at compile time. Cf @see `basic::typed`
 */
class type_erased {

public: // non-mutator methods
    constexpr column_element_type element_type() const noexcept { return gdf_column::dtype; }
    constexpr gdf::size size() const noexcept { return gdf_column::size; }
    gsl::span<T> elements() const noexcept { return mpark::get<T>(elements()); }
    gsl::span<validity_indicator_type> validity_indicators() const
    {
        if (not nullable()) {
            throw std::logic_error("Attempt to use the vailidity indicators (= NULL indicators) of a non-nullable column");
        }
        return { gdf_column::valid, size() };
    }

protected:
    // We are sadly forced to define this template since C++11 doesn't
    // have polymorphic lambdas.
    template<std::underlying_type<gdf_dtype>::type GDFType>
    void typed_initiatization(type_erased& te, const gdf_column& gc)
    {
        using data_type = typename detail::element_type_for<GDFType>::type;
        te.elements_ = gsl::span<data_type>{col.data, col.size};
        column_wide_data_ = column_wide_data<data_type>::extract_from(col.dtype_info);
        break;
    }


public: // constructors and destructor
    constexpr ~type_erased() = default;
    using gdf_column::gdf_column; // inherit the constructor, hopefully
    /**
     * @note For columns whose type does not require additional information to
     * the per-element data, the @p column_wide_data argument will necessarily
     * be an empty struct
     */
    constexpr type_erased(
        var_elements_span     elements,
        var_column_wide_data  column_wide_data)
    :
        elements_(element), column_wide_data_(column_wide_data) { }

    /**
     * @note This construction ignores the validity indicators of the `gdf_column`
     *
     * @todo Perhaps drop this ctor, and take only a void*, size and gdf_dtype?
     */
    explicit type_erased(const gdf_column& col) {
        util::visit_numeric_range<0,N_GDF_TYPES>(typed_initiatization, col.dtype);
/*
        // Discarded in favor of visitation!
        switch (col.dtype) {
        GDF_INT8: {
            using data_type = typename detail::element_type_for<GDF_INT8>::type;
            elements = gsl::span<data_type>{col.data, col.size};
            column_wide_data = column_wide_data<data_type>::extract_from(col.dtype_info);
            break;
        }
        GDF_INT16: {
            using data_type = typename detail::element_type_for<GDF_INT16>::type;
            elements = gsl::span<data_type>{col.data, col.size};
            column_wide_data = column_wide_data<data_type>::extract_from(col.dtype_info);
            break;
        }
        GDF_INT32: {
            using data_type = typename detail::element_type_for<GDF_INT32>::type;
            elements = gsl::span<data_type>{col.data, col.size};
            column_wide_data = column_wide_data<data_type>::extract_from(col.dtype_info);
            break;
        }
        GDF_INT64: {
            using data_type = typename detail::element_type_for<GDF_INT64>::type;
            elements = gsl::span<data_type>{col.data, col.size};
            column_wide_data = column_wide_data<data_type>::extract_from(col.dtype_info);
            break;
        }
        GDF_FLOAT32: {
            using data_type = typename detail::element_type_for<GDF_FLOAT32>::type;
            elements = gsl::span<data_type>{col.data, col.size};
            column_wide_data = column_wide_data<data_type>::extract_from(col.dtype_info);
            break;
        }
        GDF_FLOAT64: {
            using data_type = typename detail::element_type_for<GDF_FLOAT64>::type;
            elements = gsl::span<data_type>{col.data, col.size};
            column_wide_data = column_wide_data<data_type>::extract_from(col.dtype_info);
            break;
        }
        GDF_DATE32: {
            using data_type = typename detail::element_type_for<GDF_DATE32>::type;
            elements = gsl::span<data_type>{col.data, col.size};
            column_wide_data = column_wide_data<data_type>::extract_from(col.dtype_info);
            break;
        }
        GDF_DATE64: {
            using data_type = typename detail::element_type_for<GDF_DATE64>::type;
            elements = gsl::span<data_type>{col.data, col.size};
            column_wide_data = column_wide_data<data_type>::extract_from(col.dtype_info);
            break;
        }
        GDF_TIMESTAMP: {
            using data_type = typename detail::element_type_for<GDF_TIMESTAMP>::type;
            elements = gsl::span<data_type>{col.data, col.size};
            column_wide_data = column_wide_data<data_type>::extract_from(col.dtype_info);
            break;
        }
        GDF_CATEGORY: {
            using data_type = typename detail::element_type_for<GDF_CATEGORY>::type;
            elements = gsl::span<data_type>{col.data, col.size};
            column_wide_data = column_wide_data<data_type>::extract_from(col.dtype_info);
            break;
        }

        default:
            throw std::invalid_argument("Invalid gdf_column data type enum value");
        }
*/
    }
    constexpr type_erased(const type_erased& other) = default;
    constexpr type_erased(type_erased&& other) = default;

public: // operators
    // constexpr type_erased& operator=(gdf_column col);
    constexpr type_erased& operator=(type_erased&& other) = default;
    constexpr type_erased& operator=(const type_erased& other) = default;
    template <typename T>
    operator typed<T>() const{
        // TODO: Perhaps limit the types with an enable_if ?
        return { mpark::get<T>(elements()), column_wide_data };
    }

protected: // data members
    detail::var_elements_span    elements_;
    detail::var_column_wide_data column_wide_data_;
}; // class type_erased


} // namespace basic

} // namespace detail

/**
 * @brief A typed version of the type-erased @ref gdf::column::type_erased class.
 *
 * @note `span`s should work in device-side code, despite not having explicit CUDA support,
 * since all of their methods are constexpr'ed. However, that does require compiling with
 * the "--expt-relaxed-constexpr" flag. However, typed
 *
 * @note nullable columns cannot be constructed from non-nullable ones, for now - as
 * we can't magically get space for the null values.
 *
 * @todo
 * 1. Implement constructors from type_erased (untyped) columns
 * 2. Implement conversion ops to type_erased (untyped) columns
 * 3. Consider whether we even need move ctors; after all, this is just a reference type.
 *
 */
template <typename T>
class typed<T> : detail::basic::typed<T> {
public:
    using parent = detail::basic::typed<T>;
    constexpr bool nullable() const noexcept { return validity_pseuocolumn.has_value(); }
    constexpr gdf::size null_count() const noexcept { return null_count_; }
    constexpr gsl::span<validity_indicator_type>  validity_indicators() const {
        if (not nullable()) {
            throw std::logic_error("Attempt to access validity (NULL) indicator pseudo-column of a non-nullable column");
        }
        return validity_indicators_.value();
    }
    // TODO: Consider a method which returns the null indicators as a non_nullable_typed

public: // constructors and destructor
    constexpr ~typed() = default;

    /**
     * @brief Nullable constructor
     *
     * @note For columns whose type does not require additional information to
     * the per-element data, the @p column_wide_data argument will necessarily
     * be an empty struct
     */
    constexpr typed(
        gsl::span<T>                        elements,
        column_wide_data_t<T>               column_wide_data,
        gsl::span<validity_indicator_type>  validity_indicators,
        gdf::size                           null_count,
        const gsl::string_span              name = { }
    ) :
            elements_(element),
            column_wide_data_(column_wide_data),
            validity(validity_indicators),
            null_count_(null_count),
            name_(name) { };

    /**
     * @brief Nullable constructor
     *
     * @note This constructor will only exist for columns whose type requires
     * no additional information to the per-element data.
     */
    constexpr typed(
        gsl::span<T>                        elements,
        gsl::span<validity_indicator_type>  validity_indicators,
        gdf::size                           null_count,
        std::enable_if<not std::is_empty<column_wide_data_t<T>>::value, const gsl::string_span >
                                            name = { }
    ) :
        elements_(element),
        column_wide_data_({ }),
        validity_indicators_(validity_indicators),
        null_count_(null_count),
        name_(name) { };

    /**
     * @brief Non-nullable constructor
     *
     * @note For columns whose type does not require additional information to
     * the per-element data, the @p column_wide_data argument will necessarily
     * be an empty struct
     */
    constexpr typed(
        gsl::span<T>                        elements,
        column_wide_data_t<T>               column_wide_data,
        const gsl::string_span              name = { }
    ) :
            elements_(element),
            column_wide_data_(column_wide_data),
            name_(name) { };

    /**
     * @brief Non-nullable constructor
     *
     * @note This constructor will only exist for columns whose type requires
     * no additional information to the per-element data.
     */
    constexpr typed(
        gsl::span<T>                        elements,
        std::enable_if<not std::is_empty<column_wide_data_t<T>>::value, const gsl::string_span >
                                            name = { }
    ) :
        elements_(element),
        column_wide_data_({ }),
        name_(name) { };

    constexpr typed(const typed& other) = default;
    constexpr typed(typed&& other) = default;
    // TODO: Constructor from a type_erased column? Maybe a named constructor idiom?
    explicit typed(type_erased&& other) noexcept; // TODO: Write me!

    typed(const gdf_column& gc) :
        parent(gc),
        null_count_(gc.null_count)
    {
        detail::validate_gdf_column();
        if (gc.valid != nullptr) {
            validity = { gc.valid, gc.size };
        }
        assert(gc.valid != nullptr and
            "Attempt to construct a non-nullable typed column from a nullable type_erased (untyped) column");
    };


public: // operators
    constexpr typed& operator=(typed&& other) noexcept = default;
    constexpr typed& operator=(const basic::typed& other) noexcept = default;

protected: // data members
    // TODO: Consider wrapping the basic column in a counted-boolean-column struct,
    // so that the optional covers them both

    /**
     * @brief A pseudo-column of indicators of validity or NULLness for each element
     */
    tl::optional<detail::basic<validity_indicator_type>> validity { };

    /**
     * @brief A materialized result of counting the null indicators in @ref validity
     * (that is, counting the false-valued validity indicators.)
     */
    gdf::size null_count { 0 };
};

} // namespace column

} // namespace gdf


#endif // GDF_COLUMN_HPP_
