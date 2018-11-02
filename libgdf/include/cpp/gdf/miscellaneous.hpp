#ifndef GDF_BASIC_TYPES_HPP_
#define GDF_BASIC_TYPES_HPP_

#include <boost/variant.hpp>

extern "C" {
#include "gdf/cffi/types.h"
}

#include <cstdint>
#include <typeinfo>


namespace gdf {

using status = gdf_error; // TODO: Perhaps drop this?

enum class hash_function_type : std::underlying_type<gdf_hash_func>::type {
    murmur_3, identity
};

// TODO: Confusing and not general, define these elsewhere
// using quantile_method = gdf_quantile_method;

// TODO: Avoid using this. I really doubt anybody using the library should see this.
using nvtx_color = gdf_color;

// TODO: Try to drop this
struct operator_context {
    using algorithm_type = gdf_method;

    bool input_is_sorted; // TODO: But what if there are multiple input columns?
    algorithm_type algorithm;
    bool input_values_are_distinct;
        // TODO: But what if there are multiple input columns?
        // TODO: What about null values? Is data with multiple nulls considered distinct?
    bool producing_sorted_result;
    bool sorting_in_place_allowed;

    operator gdf_context() const noexcept {
        return {
            input_is_sorted, algorithm,
            input_values_are_distinct,
            producing_sorted_result,
            sorting_in_place_allowed
        };
    }

    // TODO: Consider implementing ctors and a ctor from gdf_context
};

//struct _OpaqueIpcParser;
//typedef struct _OpaqueIpcParser gdf_ipc_parser_type;
//
//
//struct _OpaqueRadixsortPlan;
//typedef struct _OpaqueRadixsortPlan gdf_radixsort_plan_type;
//
//
//struct _OpaqueSegmentedRadixsortPlan;
//typedef struct _OpaqueSegmentedRadixsortPlan gdf_segmented_radixsort_plan_type;

namespace sql {

using ordering_type         = order_by_type;
using comparison_operator   = gdf_comparison_operator;
using window_function_type  = ::window_function_type;
using window_reduction_type = window_reduction_type;
using aggregation_type      = gdf_agg_op;

} // namespace sql

namespace detail {

namespace mixins {

// TODO: Do we really need this?
template <typename Name>
class named {
public:
    const Name name;
};

}  // namespace mixins

} // namespace detail

} // namespace gdf


#endif // GDF_BASIC_TYPES_HPP_
