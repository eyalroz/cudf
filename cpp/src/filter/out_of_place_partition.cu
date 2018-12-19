/*
 * Copyright 2018 BlazingDB, Inc.
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


#include <cudf.h>
#include <utilities/cudf_utils.h>
#include <utilities/error_utils.h>
#include <utilities/miscellany.hpp>
#include <utilities/type_dispatcher.hpp>
#include <rmm/thrust_rmm_allocator.h>

#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

#include <thrust/execution_policy.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda_runtime.h>

#include <functional>
#include <exception>

namespace cudf {

enum class pivoting_mode {
    bipartition,    // The pivoting sets are 1. less than the pivot 2. greater-or-equal than the pivot
    tripartition,   // The pivoting sets are 1. less than the pivot 2. greater than the pivot
                    // (and we can figure out how many were equal to the pivot
};

// TODO: We should not need this. Can't thrust just take these things as pointers?
template <typename T>
inline auto make_thrust_iterator(void* ptr_to_raw_data_on_device)
{
    return thrust::detail::make_normal_iterator(
        thrust::device_pointer_cast(static_cast<T*>(ptr_to_raw_data_on_device))
    );
}

// TODO: We sholdn't need this. gdf_column itself, or some thrust-oriented
// wrapper class for it, should provide a begin() and end() thrust iterators
// (as well as cbegin and cend)
template <typename T>
inline auto thrust_data_begin(const gdf_column& column)
{
    return make_thrust_iterator<T>(column.data);
}

// TODO: We sholdn't need this. gdf_column itself, or some thrust-oriented
// wrapper class for it, should provide a begin() and end() thrust iterators
// (as well as cbegin and cend)
// TODO: A variant of this function which returns a pair of
// zip iterators involving the validity pseudo-column as well
template <typename T>
inline auto make_thrust_data_range(const gdf_column& column)
{
    struct iterator_pair {
        using iterator_type = decltype(make_thrust_iterator<T>(std::declval<void*>()));
        iterator_type begin_, end_;

        const iterator_type& begin() { return begin_; }
        const iterator_type& end() { return end_; }
        std::size_t size() { return std::distance(begin_, end_); }
    };

    decltype(make_thrust_iterator<T>(column.data)) start = thrust_data_begin<T>(column);
    decltype(make_thrust_iterator<T>(column.data)) end = thrust_data_begin<T>(column) + column.size;
    return iterator_pair { start, end };
}

// TODO: We should be using this at all.. our operator implementations should be _receiving_
// streams, not creating their own streams. Separation of concerns + allows for actual
// control of execution flows. For now - this saves repetitions in each and every
// operator implementation, and also ensures stream destruction if an exception
// is thrown somehow
struct scoped_cuda_stream {
    cudaStream_t stream_ { nullptr };

    scoped_cuda_stream() {
        cudaStreamCreate(&stream_);
        assert(stream_ != nullptr and "Failed creating a CUDA stream");
    }
    operator cudaStream_t() { return stream_; }
    ~scoped_cuda_stream() {
        if (not std::uncaught_exception()) {
            cudaStreamSynchronize(stream_);
        }
        cudaStreamDestroy(stream_);
    }
};

/**
 * Partitions the elements of a column into two columns based on their
 * respective (order) relation to a pivot value
 *
 * @tparam PivotingMode determines whether we're interested in below/not below,
 * or in below/equal/above
 * @param unpartitioned A column whose data we wish to partition
 * @param below_pivot An output column, in which we'll store the elements
 * lesser-than the pivot; it will constitute one part of the partition
 * @param above_pivot An output column, in which we'll store the elements
 * greater-than, or greater-or-equal than, the pivot; it will constitute one
 * part of the partition
 * @param pivot An arbitrary value to partition by
 *
 * @todo should we artificially generate a third column of copies of pivot,
 * of the appropriate size? probably not.
 *
 * @note
 * 1. For now, the implementation only supports non-nullable columns
 * 2. The implementation is sub-optimal at least in the case of a
 * tri-partition-evenrything could be done using a single kernel (and
 * a single pass over the data)
 * 3. We should also implement a variant which produces index sets
 * (in which case we'll want 3 of those - one for the equal-to-pivot) *
 * 4. The output columns must have enough space allocated
 */
template <typename ColumnElement, pivoting_mode PivotingMode>
gdf_error partition_by_pivot(
    gdf_column&                  below_pivot,
    gdf_column&                  above_pivot,
    const gdf_column&            unpartitioned,
    const ColumnElement&         pivot,
    cudaStream_t                 stream)
{
    auto unpartitioned_range = make_thrust_data_range<ColumnElement>(unpartitioned);
    auto below_pivot_range = make_thrust_data_range<ColumnElement>(below_pivot);

    auto is_below_pivot =
        [pivot]
         __host__ __device__ (const ColumnElement& x) {
            return x < pivot;
        };
    below_pivot_range.end_ = thrust::copy_if(
        rmm::exec_policy(stream),
        unpartitioned_range.begin(),
        unpartitioned_range.end(),
        below_pivot_range.begin(),
        is_below_pivot);
    below_pivot.size = below_pivot_range.size();

    auto is_above_pivot =
        [pivot]
        __host__ __device__ (const ColumnElement& x) {
            return PivotingMode == pivoting_mode::tripartition ? x > pivot : x >= pivot;
        };
    auto above_pivot_range = make_thrust_data_range<ColumnElement>(above_pivot);
    above_pivot_range.end_ = thrust::copy_if(
        rmm::exec_policy(stream),
        unpartitioned_range.begin(),
        unpartitioned_range.end(),
        above_pivot_range.begin(),
        is_above_pivot);
    above_pivot.size = above_pivot_range.size();

    return GDF_SUCCESS;
}

namespace detail {

// This is necessary since C++ does not support templated lambdas before
// the C++17 version of the standard, while we're working with C++14.
template <pivoting_mode PivotingMode>
struct partition_by_pivot_helper {
    template <typename T>
    gdf_error operator()(
        gdf_column&         below_pivot,
        gdf_column&         above_pivot,
        const gdf_column&   unpartitioned,
        const void*         pivot,
        cudaStream_t        stream)
    {
        return partition_by_pivot<T, PivotingMode>(
            below_pivot,
            above_pivot,
            unpartitioned,
            *static_cast<const T*>(pivot),
            stream);
    }
};

} // namespace detail

template <pivoting_mode PivotingMode>
gdf_error partition_by_pivot(
    gdf_column&        below_pivot,
    gdf_column&        above_pivot,
    const gdf_column&  unpartitioned,
    const void*        pivot)
{
    // TODO: Extend support to nullable columns and remove this check
    GDF_REQUIRE(not is_nullable(unpartitioned), GDF_VALIDITY_UNSUPPORTED);

    for(const auto& output_column : { below_pivot, above_pivot }) {
        GDF_REQUIRE(output_column.dtype == unpartitioned.dtype, GDF_DTYPE_MISMATCH);
        GDF_REQUIRE(is_nullable(output_column) == is_nullable(unpartitioned), GDF_VALIDITY_MISMATCH);
        // TODO: More validity checks? e.g. non-null data pointer?

        // TODO: Consider checking the below_pivot and above_pivot are empty
    }
    // Note: Not checking the allocated output sizes - assuming there's enough space

    scoped_cuda_stream stream;

    return type_dispatcher(
        unpartitioned.dtype,
        detail::partition_by_pivot_helper<PivotingMode>{},
        below_pivot,
        above_pivot,
        unpartitioned,
        pivot,
        static_cast<cudaStream_t>(stream));
}

/**
 * @brief Partitions the elements of a column into those before a pivot and
 * all the rest
 *
 * @param[in] unpartitioned
 *     A column whose data we wish to partition
 * @param[out] strictly_below_pivot
 *     After execution, holds copies of the elements of @p unpartitioned which
 *     which fall before the pivot
 * @param[out] above_or_equal_to_pivot
 *     After execution, holds copies of the elements of @p unpartitioned which
 *     are either equal to the pivot or are after it in order
 * @param[in] pivot
 *    An arbitrary value to partition by
 *
 * @note For now, the implementation only supports non-nullable columns
 *
 * @note  Each column passed must have enough space allocated to accommodate the
 * relevant elements of @p unpartitioned - so unless the caller has information
 * regarding the distribution of @p data_partition elements, they need to each have
 * `unpartitioned.size * sizeof(ColumnElement)` bytes allocated.
 */
template gdf_error partition_by_pivot<pivoting_mode::bipartition>(
    gdf_column&        strictly_below_pivot,
    gdf_column&        above_or_equal_to_pivot,
    const gdf_column&  unpartitioned,
    const void*        pivot);

/**
 * @brief Partitions the elements of a column into two columns - before,
 * at or above a pivot value - without actually materializing
 * a third column of elements equal to the pivot
 *
 * @param[in] unpartitioned
 *     A column whose data we wish to partition
 * @param[out] strictly_below_pivot
 *     After execution, holds copies of the elements of @p unpartitioned
 *     which fall before the pivot
 * @param[out] strictly_above_pivot
 *     After execution, holds copies of the elements of @p unpartitioned
 *     which fall after the pivot
 * @param[in] pivot
 *     An arbitrary value to partition by
 *
 * @note For now, the implementation only supports non-nullable columns
 *
 * @note there's no need to produce an `equal_to_pivot` column - nor, in fact,
 * to count how many elements are equal to the pivot (as long as we can
 * ignore nulls, anyway), since `equal_to_pivot` would simply be a sequence
 * of copies of the pivot, of length determinable using the original column
 * and the two other parts' lengths
 *
 * @note  Each column passed must have enough space allocated to accommodate the
 * relevant elements of @p unpartitioned - so unless the caller has information
 * regarding the distribution of @p data_partition elements, they need to each have
 * `unpartitioned.size * sizeof(ColumnElement)` bytes allocated.
 */
template gdf_error partition_by_pivot<pivoting_mode::tripartition>(
    gdf_column&        strictly_below_pivot,
    gdf_column&        strictly_above_pivot,
    const gdf_column&  unpartitioned,
    const void*        pivot);


/**
 * Partitions the elements of a column into two columns based on their
 * respective (order) relation to a pivot value
 *
 * @tparam PivotingMode determines whether we're interested in below/not below,
 * or in below/equal/above
 * @param unpartitioned A column whose data we wish to partition
 * @param below_pivot An output column, in which we'll store the elements
 * lesser-than the pivot; it will constitute one part of the partition
 * @param above_pivot An output column, in which we'll store the elements
 * greater-than, or greater-or-equal than, the pivot; it will constitute one
 * part of the partition
 * @param pivot An arbitrary value to partition by
 *
 * @note For now, the implementation only supports non-nullable columns
 *
 * @note The output columns must have enough space allocated
 *
 */
template <typename ColumnElement>
gdf_error partition_by_filter(
    gdf_column&         passing_filter,
    gdf_column&         not_passing_filter,
    const gdf_column&   unpartitioned,
    const gdf_column&   filter,
    cudaStream_t        stream)
{
    auto unpartitioned_range      = make_thrust_data_range<ColumnElement>(unpartitioned);
    auto passing_filter_range     = make_thrust_data_range<ColumnElement>(passing_filter);
    auto not_passing_filter_range = make_thrust_data_range<ColumnElement>(not_passing_filter);

    auto filter_data = static_cast<const gdf_bool*>(filter.data);
    auto unpartitioned_data = static_cast<const ColumnElement*>(unpartitioned.data);

    auto get_corresponding_filter_value =
        [filter_data, unpartitioned_data]
        __host__ __device__ (const ColumnElement& x) {
            auto index_of_x_in_unpartitioned_data = &x - unpartitioned_data;
            return filter_data[index_of_x_in_unpartitioned_data];
                // Relying on the implicit compatibility of semantics of GDF booleans here.
        };
    auto ends_pair = thrust::partition_copy(
            rmm::exec_policy(stream),
            unpartitioned_range.begin(),
            unpartitioned_range.end(),
            passing_filter_range.begin(),
            not_passing_filter_range.begin(),
            get_corresponding_filter_value);
    passing_filter_range.end_ = ends_pair.first;
    not_passing_filter_range.end_ = ends_pair.second;
    passing_filter.size = passing_filter_range.size();
    not_passing_filter.size = not_passing_filter_range.size();

    return GDF_SUCCESS;
}



namespace detail {

// This helper structure is necessary since C++ does not support templated
// lambdas before the C++17 version of the standard, while we're working with
// C++14.
struct partition_by_filter_helper {
    template <typename ColumnElement>
    gdf_error operator()(
        gdf_column&         passing_filter,
        gdf_column&         not_passing_filter,
        const gdf_column&   unpartitioned,
        const gdf_column&   filter,
        cudaStream_t        stream)
    {
        // TODO: This is supposed to be very similar, or perhaps
        // even sharing the same codebase, as gdf_apply_stencil -
        // except that the latter is bit-resolution while we're still byte-resolution,
        // plus
        return partition_by_filter<ColumnElement>(
            passing_filter,
            not_passing_filter,
            unpartitioned,
            filter,
            stream);
    }
};

} // namespace detail




/**
 *
 * @note This function is actually rather similar to @ref gpu_apply_stencil , with the
 * following differences: 1. It also outputs the _complement_ of the elements marked
 * in the stencil, i.e. an actual partition. 2. The 'stencil' is a column of bits.
 * Actually, we should be using a column of bits here as well, but that's not
 * well-supported enough right now.
 *
 * @todo Perhaps we'd like to call this `partition_by_stencil` or `partition_by_bitmask`
 *
 * @todo Perhaps this should be templated over a functor which determines the part
 * an element is supposed to be in.
 *
 * @note This function differs from @ref gpu_apply_stencil not only in producing two
 * outputs instead of just the passing elements, but also in that @ref gpu_apply_stencil
 * preserves the relative order of elements (and should probably have been called
 * something like `stably_apply_stencil()`), while this function does _not_ do so.
 * Also, @ref gpu_apply_stencil supports nullable stencil columns, while here the filter
 * is non-nullable booleans only.
 *
 *
 * @param unpartitioned[in] An arbitrary column. For now, we assume this column
 * @param filter[in] A column of boolean values determining which elements of
 * @p unpartitioned go into which output column
 * @param passing_filter[out] The elements of @p unpartitioned which pass the filter,
 * i.e. whose corresponding value in @p filter is true.
 * @param not_passing_filter[out] The elements of @p unpartitioned which do not pass
 * the filter, i.e. whose corresponding value in @p filter is false.
 */
gdf_error partition_by_filter(
    gdf_column&        passing_filter,
    gdf_column&        not_passing_filter,
    const gdf_column&  unpartitioned,
    const gdf_column&  filter)
{
    // TODO: Extend support to nullable columns and remove this check
    GDF_REQUIRE(not is_nullable(unpartitioned), GDF_VALIDITY_UNSUPPORTED);
    for(const auto& output_column : { not_passing_filter, passing_filter}) {
        GDF_REQUIRE(output_column.dtype == unpartitioned.dtype, GDF_DTYPE_MISMATCH);
        GDF_REQUIRE(is_nullable(output_column) == is_nullable(unpartitioned), GDF_VALIDITY_MISMATCH);
        // TODO: More validity checks? e.g. non-null data pointer?
    }
    GDF_REQUIRE(is_boolean(filter), GDF_UNSUPPORTED_DTYPE);

    // Note: Not checking the allocated output sizes - assuming there's enough space

    scoped_cuda_stream stream;

    return type_dispatcher(
        unpartitioned.dtype,
        detail::partition_by_filter_helper{},
        passing_filter,
        not_passing_filter,
        unpartitioned,
        filter,
        static_cast<cudaStream_t>(stream)); // TODO: Do we really need this static cast? probably not
}


} // namespace cudf
