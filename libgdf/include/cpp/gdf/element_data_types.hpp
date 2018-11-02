#ifndef GDF_ELEMENT_DATA_TYPES_HPP_
#define GDF_ELEMENT_DATA_TYPES_HPP_

extern "C" {
#include "gdf/cffi/types.h"
}

#include <cstdint>
#include <typeinfo>
#include <chrono>
#include <ratio>


namespace gdf {

using size  = gdf_size_type;
using index = gdf_index_type;

using element_type = gdf_dtype;

static_assert(sizeof(float)  == 4, "float does not have 32 bits");
using float32_t = float;
static_assert(sizeof(double) == 8, "double does not have 64 bits");
using float64_t = double;

namespace chrono {

namespace detail { using duration_base = int64_t; }

template<typename Period>
using duration = std::chrono::duration<duration_base, Period>;

// We use this enum class rather than gdf_time_unit,
// to ensure only valid values are used. Well, sort-of ensure, anyway.
enum class period : std::underlying_type<gdf_time_unit>::type {
    none         = TIME_UNIT_NONE,//!< none
    second       = TIME_UNIT_s,   //!< second
    millisecond  = TIME_UNIT_ms,  //!< millisecond
    microsecond  = TIME_UNIT_us,  //!< microsecond
    nanosecond   = TIME_UNIT_ns   //!< nanosecond
};

// TODO: I'm not sure these next couple of constructs should actually be
// within detail
namespace detail {

template <typename Rep, typename Period>
constexpr inline period period_of(duration<Period>)
{
    switch(Period) {
    case 1:          return period::second;
    case std::milli: return period::millisecond;
    case std::micro: return period::microsecond;
    case std::nano:  return period::nanosecond;
    };
    return period::none;
}

template <period Period>
struct std_chrono_period_of { };

template <> struct std_chrono_period_of< period::second      > { using type = std::ratio<1>; };
template <> struct std_chrono_period_of< period::millisecond > { using type = std::milli;    };
template <> struct std_chrono_period_of< period::microsecond > { using type = std::micro;    };
template <> struct std_chrono_period_of< period::nanosecond  > { using type = std::nano;     };

template <period Period>
using std_chrono_period_of_t =  std_chrono_period_of<Period>::type;

}

template <typename Period>
struct timestamp {
    duration<Period> since_epoch;
};

/**
 * @brief a type-erased version of @ref dimensioned_timestamp,
 * which is used in our type-erased columns
 *
 * @todo
 * 1. What is the epoch?
 * 2. In typed columns, should we really go to the fully-typed timestamp?
 *
 */
struct bare_timestamp {
    duration_base since_epoch;
};

using type_erased_timestamp = bare_timestamp;

namespace detail {

// TODO: We should be using something like Boost's integer.hpp's int<...> types

template <unsigned Bits> struct date_helper     { };
template <>              struct date_helper<32> { using type = gdf_date32; };
template <>              struct date_helper<64> { using type = gdf_date64; };

} // namespace detail

// TODO: Date should probably be some sort of bit field
template <unsigned SizeInBits>
using date = typename detail::date_helper<SizeInBits>::type;

// TODO: Implement some unary binary operators involving durations, timestamps and dates

} // namespace chrono

using category = gdf_category; // TODO: What's this, anyway?

// using extra_element_type_info = gdf_dtype_extra_info;

template <element_type ElementType>
struct column_wide_data {
    static column_wide_data extract_from(const gdf_column& col) { return { }; }
};

template <>
struct column_wide_data<GDF_TIMESTAMP> {
    static column_wide_data extract_from(const gdf_column& col)
    {
        // TODO:
        return (chrono::period) col.dtype_info.time_unit;
    }
    gdf::chrono::period period;
};

//template <>
//struct column_wide_data<GDF_DECIMAL> {
//    unsigned char scale;
//};
//
//template <>
//struct column_wide_data<GDF_CHAR> {
//    size_type length;
//};
//
//template <>
//struct column_wide_data<GDF_VARCHAR> {
//    size_type length;
//};



namespace detail {

// Poor man's reflection facilities for element data types...
// and with lots of code duplication too :-(

template<typename ElementType>
struct element_type_to_enum;
template<element_type ElementType>
struct element_type_for;

template<>
struct element_type_to_enum< int8_t                 > { constexpr const element_type value { GDF_INT8      }; };
struct element_type_to_enum< int16_t                > { constexpr const element_type value { GDF_INT16     }; };
struct element_type_to_enum< int32_t                > { constexpr const element_type value { GDF_INT32     }; };
struct element_type_to_enum< int64_t                > { constexpr const element_type value { GDF_INT64     }; };
struct element_type_to_enum< float32_t              > { constexpr const element_type value { GDF_FLOAT32   }; };
struct element_type_to_enum< float64_t              > { constexpr const element_type value { GDF_FLOAT64   }; };
struct element_type_to_enum< chrono::date<32>       > { constexpr const element_type value { GDF_DATE32    }; };
struct element_type_to_enum< chrono::date<64>       > { constexpr const element_type value { GDF_DATE64    }; };
struct element_type_to_enum< category               > { constexpr const element_type value { GDF_CATEGORY  }; };
struct element_type_to_enum< chrono::bare_timestamp > { constexpr const element_type value { GDF_TIMESTAMP }; };
// struct element_type_to_enum<???       > { constexpr const element_type value { GDF_STRING };    };

template<typename ElementType>
constexpr element_type enum_for() noexcept { return element_type_to_enum<ElementType>::value; }

template<>
struct element_type_for< GDF_INT8      > { using type = int8_t;                 };
struct element_type_for< GDF_INT16     > { using type = int16_t;                };
struct element_type_for< GDF_INT32     > { using type = int32_t;                };
struct element_type_for< GDF_INT64     > { using type = int64_t;                };
struct element_type_for< GDF_FLOAT32   > { using type = float32_t;              };
struct element_type_for< GDF_FLOAT64   > { using type = float64_t;              };
struct element_type_for< GDF_DATE32    > { using type = chrono::date<32>;       };
struct element_type_for< GDF_DATE64    > { using type = chrono::date<32>;       };
struct element_type_for< GDF_CATEGORY  > { using type = category;               };
struct element_type_for< GDF_TIMESTAMP > { using type = chrono::bare_timestamp; };
// struct element_type_for< GDF_STRING    > { using type = ???};

template<element_type ElementType>
using element_type_for_t = typename element_type_for<ElementType>::type;

constexpr std::typeinfo& typeinfo_for(gdf::element_type element_type) noexcept
{
    static constexpr const std::array<const std::type_info, 9> element_type_to_enum_typeinfo =
    {
        typeid( int8_t                 ),
        typeid( int16_t                ),
        typeid( int32_t                ),
        typeid( int64_t                ),
        typeid( float                  ),
        typeid( double                 ),
        typeid( chrono::date<32>       ),
        typeid( chrono::date<64>       ),
        typeid( category               ),
        typeid( chrono::bare_timestamp ),
        // what to put here for GDF_STRING?
        // no decimal, char or varchar
    };
    return element_type_to_enum_typeinfo[static_cast<int>(element_type)];
}

} // namespace detail


/**
 * @brief A bit-holder type, used for indicating whether some column elements
 * are null or not. If the corresponding element is null, its value will be 0;
 * otherwise the value is 1 (a "valid" element)
 */
using validity_indicator_type = gdf_valid_type;

} // namespace gdf


#endif // GDF_ELEMENT_DATA_TYPES_HPP_
