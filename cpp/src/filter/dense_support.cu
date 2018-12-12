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
#include <utilities/type_dispatcher.hpp>
#include <utilities/miscellany.hpp>

#include <cuda_runtime.h>
#include <device_atomic_functions.h>

#include <utility>

/*
 * Note: Many of the functions in this file are not specific to the implementation of
 * the dense_support operator; they can and should be generalized. Some are _already_
 * generalized, but not within libcudf.
 */

namespace cudf {

using cudf_bool_t = int8_t;
// constexpr const gdf_dtype gdf_bool_dtype = GDF_INT8;

namespace detail {



/**
 * @brief returns the aligned larger-size start position in memory
 * in which the specified (aligned) smaller-size element is located
 *
 * Suppose you're on a platform where memory accesses must be aligned,
 * e.g. you can only use `uint64_t`'s at addresses which are a multiple
 * of 8 bytes. Now, if you want to apply some operation to some small
 * value in memory, but the operation only takes a larger value, you
 * need to obtain the aligned address of the larger element; this
 * function is what you'd use.
 *
 * @note This should have been implemented with log-of-size template
 * parameters, but we don't have an unsigned_t<Size> template
 *
 * @tparam Contained Type of the smaller value in memory
 * @tparam Containing Type of the larger value containing the smaller one
 * @param ptr Position for which to find a larger-alignment container
 * @return Position of a larger-alignment in memory containing the value
 * at @p ptr
 */
template <typename Contained, typename Containing>
__host__ __device__ __forceinline__ Containing* containing_element(Contained* ptr)
{
    static_assert(sizeof(Contained*) == sizeof(size_t), "Unexpected pointer size");
    static_assert(util::is_a_power_of_two(sizeof(Contained)), "The Contained type's size is not a power of two");
    static_assert(util::is_a_power_of_two(sizeof(Containing)), "The Containing type's size is not a power of two");

    auto contained_element_address_as_number = reinterpret_cast<size_t>(ptr);
    auto containing_element_address =
        util::clear_lower_bits_unsafe(contained_element_address_as_number, util::size_in_bits(ptr));
    return reinterpret_cast<Containing*>( containing_element_address );
}

} // namespace detail

__device__ void atomicOr(int8_t* address, int8_t value)
{
    static_assert(sizeof(int64_t*) == sizeof(size_t), "Unexpected pointer size");
    constexpr const unsigned int8s_in_int = sizeof(int) / sizeof(int8_t);
    auto native_int_address = detail::containing_element<int8_t, int>(address);
    auto mask = value << (reinterpret_cast<size_t>(address) % int8s_in_int);
    ::atomicOr(native_int_address, mask);
}

__device__ void atomicSet(int8_t* address)
{
    atomicOr(address, 1);
}

namespace kernels {

/*
 * Note:
 * This kernel will (likely) trigger an error if the input sparse array
 * has a value outside of the range 0... domain_size - 1. Specifically,
 * if the type is signed, negative values may trigger an error (and we
 * do not check for their presence)
 *
 * TODO: Experiment with different values for the serialization factor;
 * It's quite possible it should just be 1 or 2.
 *
 */

template <typename T, int serialization_factor = 2>
__global__ void sparse_to_dense(
    const T*          __restrict__  sparse_multiset,
    cudf_bool_t*       __restrict__  dense_support,
    gdf_size_type                   sparse_subset_size,
    gdf_size_type                   domain_size)
{
    static_assert(std::is_integral<T>::value, "Only integral types are supported");

    gdf_size_type input_pos = threadIdx.x + blockIdx.x * blockDim.x * serialization_factor;
        // Note: Since grid dimensions are 32-bits, and gdf_size_type should be
        // 64-bits, this may overflow

    #pragma unroll
    for(int i = 0; i < serialization_factor; i++)
    {
        if (input_pos >= sparse_subset_size) { break; }
        auto sparse_multiset_element = sparse_multiset[input_pos];
        auto output_pos = sparse_multiset_element;
        atomicSet(&dense_support[output_pos]);
        input_pos += blockDim.x;
    }
}

} // namespace kernels

struct visitor {

    template <typename T>
    void form_grid_and_launch_kernel(
        const gdf_column& sparse_multiset,
        const gdf_column& dense_support_set,
        cudaStream_t stream)
    {
        // TODO: Get rid of the magic number and form the grid more intelligently
        // (and generically?)
        auto threads_per_block = 256;
        auto grid { form_naive_1d_grid(sparse_multiset.size, threads_per_block) };
        constexpr const unsigned no_shared_memory { 0 };

        kernels::sparse_to_dense<T>
            <<< grid.num_blocks, grid.threads_per_block, no_shared_memory, stream >>>(
            static_cast<const T*>(sparse_multiset.data),
            static_cast<cudf_bool_t *>(dense_support_set.data),
            sparse_multiset.size,
            dense_support_set.size
        );
    }

    template <typename T> void operator()(
        const gdf_column& sparse_multiset,
        const gdf_column& dense_support_set,
        cudaStream_t stream)
    {
        assert(false and "Attempt to pass a non-integer-type column as a container of element indices.");
    }


};

// TODO: DRY.
template <> void visitor::operator()<int8_t>(const gdf_column& sm, const gdf_column& dss, cudaStream_t stream)
{
    form_grid_and_launch_kernel<int8_t >(sm, dss, stream);
}
template <> void visitor::operator()<int16_t>(const gdf_column& sm, const gdf_column& dss, cudaStream_t stream)
{
    form_grid_and_launch_kernel<int16_t>(sm, dss, stream);
}
template <> void visitor::operator()<int32_t>(const gdf_column& sm, const gdf_column& dss, cudaStream_t stream)
{
    form_grid_and_launch_kernel<int32_t>(sm, dss, stream);
}
template <> void visitor::operator()<int64_t>(const gdf_column& sm, const gdf_column& dss, cudaStream_t stream)
{
    form_grid_and_launch_kernel<int64_t>(sm, dss, stream);
}

void set_column_to_zero(const gdf_column& col, cudaStream_t stream)
{
    if (col.size == 0) { return; }
    cudaMemsetAsync(col.data, 0, data_size_in_bytes(col), stream);
}

/**
 * Produce the dense representation of the set of all elements appearing in
 * the input column
 *
 * @todo Convert this to use bit-resolution booleans.
 * @todo CUDA error checking
 *
 * @note The column types are all signed, while gdf_size_type is unsigned,
 * i.e. in theory it is not possible to create a column for any
 * representable element index. In practice this shouldn't happen (until
 * we work with columns of length 2^63 or higher).
 *
 * @param input[in] A non-nullable column of non-nullable non-negative
 * integral values; this is the sparse representation of the subset
 * @param output[out] A column of boolean values, each corresponding
 * to one element of the integral domain, starting from 0. `output[i]`
 * will be set to `1` if `input` contains at least one occurrence of `i`,
 * and `0` otherwise.
 */
gdf_error gdf_dense_support(
    const gdf_column&    __restrict__  sparse_multiset,
    const gdf_column&    __restrict__  dense_support_set)
{
    GDF_REQUIRE(not is_nullable(dense_support_set), GDF_COLUMN_SIZE_MISMATCH);
    GDF_REQUIRE(is_an_integer(sparse_multiset.dtype), GDF_UNSUPPORTED_DTYPE);
    GDF_REQUIRE(dense_support_set.dtype == GDF_INT8, GDF_UNSUPPORTED_DTYPE);

    // TODO: This is useless. Only this function uses this stream, and it gets synchronized
    // right after we load it with work.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    set_column_to_zero(dense_support_set, stream);

    if (sparse_multiset.size > 0) {
        GDF_REQUIRE(sparse_multiset.data != nullptr, GDF_DATASET_EMPTY);

        cudf::type_dispatcher(
            sparse_multiset.dtype,
            visitor{},
            sparse_multiset, dense_support_set, stream);
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return GDF_SUCCESS;
}

} // namespace cudf
