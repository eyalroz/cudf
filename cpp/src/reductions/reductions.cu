#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.h"
#include "utilities/type_dispatcher.hpp"
#include "utilities/miscellany.hpp"

#include <cub/block/block_reduce.cuh>

#include <limits>
#include <type_traits>

#define REDUCTION_BLOCK_SIZE 128

struct IdentityLoader{
    template<typename T, typename R = T>
    __device__
    T operator() (const T *ptr, int pos) const {
        return ptr[pos];
    }
};

/*
Generic reduction implementation with support for validity mask
*/

template<typename T, typename F, typename Ld, typename R = T>
__global__
void gpu_reduction_op(const T *data, const gdf_valid_type *mask,
                      gdf_size_type size, R *results, F functor, R neutral_value,
                      Ld loader)
{
    typedef cub::BlockReduce<R, REDUCTION_BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = threadIdx.x;
    int blkid = blockIdx.x;
    int blksz = blockDim.x;
    int gridsz = gridDim.x;

    int step = blksz * gridsz;

    R agg = neutral_value;

    for (int base=blkid * blksz; base<size; base+=step) {
        // Threadblock synchronous loop
        int i = base + tid;
        // load
        R loaded = neutral_value;
        if (i < size && gdf_is_valid(mask, i))
            loaded = loader(data, i);
            
        // Block reduce
        R temp = BlockReduce(temp_storage).Reduce(loaded, functor);
        // Add current block
        agg = functor(agg, temp);
    }
    // First thread of each block stores the result.
    if (tid == 0)
        results[blkid] = agg;
}

// TODO:
// 1. Why do the member functions here not take a stream?
// 2. Who guarantees the output fits in a T? Or that the output is
//    even _ever_ a T?
// 3. What happens when there's overflow?
template<typename T, typename F, typename R = T>
struct ReduceOp {
    static
    gdf_error launch(
        gdf_column * __restrict__ input,
        R                         neutral_value,
        R *          __restrict__ output,
        gdf_size_type             intermediate_output_size)
    {

        // 1st round
        //    Partially reduce the input into *intermediate_output_size* length.
        //    Each block computes one output in the *output*,
        //    so intermediate_output_size is the grid size (in block)
        typedef typename F::Loader Ld1;
        F functor1;
        Ld1 loader1;
        launch_once<T, F, typename F::Loader>(reinterpret_cast<const T*>(input->data), input->valid, input->size,
                    output, intermediate_output_size, neutral_value, functor1, loader1);
        CUDA_CHECK_LAST();

        // 2nd round
        //    Finish the partial reduction (if needed).
        //    A single block reduction that computes one output stored to the
        //    first index in *output*.
        if ( intermediate_output_size > 1 ) {
            typedef typename F::second F2;
            F2 functor2;

            launch_once<R, F2, IdentityLoader>(output, nullptr, intermediate_output_size,
                        output, 1, neutral_value, functor2, IdentityLoader{});
            CUDA_CHECK_LAST();
        }

        return GDF_SUCCESS;
    }

    template <typename U, typename Functor, typename Loader>
    static
    void launch_once(const U *data, gdf_valid_type *valid, gdf_size_type size,
                     R *output, gdf_size_type intermediate_output_size, T neutral_value,
                     Functor functor, Loader loader) {
        // find needed gridsize
        // use atmost REDUCTION_BLOCK_SIZE blocks
        int blocksize = REDUCTION_BLOCK_SIZE;
        int gridsize = (intermediate_output_size < REDUCTION_BLOCK_SIZE?
                        intermediate_output_size : REDUCTION_BLOCK_SIZE);

        // launch kernel
        gpu_reduction_op<U, Functor, Loader, R>
            <<<gridsize, blocksize>>>(
            // inputs
            data, valid, size,
            // output
            output,
            // action
            functor,
            // neutral value (applying the functor to it and another T value
            // produces the other value)
            neutral_value,
            // loader
            loader
        );
    }

};

struct DeviceCountNonZeros {
    struct Loader{
        template<typename T>
        __device__
        gdf_size_type operator() (const T * ptr, int pos) const {
            return (ptr[pos] == 0) ? 0 : 1; // same as return ptr[pos] != 0;
        }
    };
    typedef DeviceCountNonZeros second;

    template<typename R>
    __device__
    R operator() (const R &lhs, const R &rhs) {
        return lhs + rhs;
    }
};


struct DeviceSum {
    typedef IdentityLoader Loader;
    typedef DeviceSum second;

    template<typename T, typename R = T>
    __device__
    R operator() (const T &lhs, const T &rhs) {
        return lhs + rhs;
    }

    template<typename R>
    static constexpr R neutral_value { 0 };
};

struct DeviceProduct {
    typedef IdentityLoader Loader;
    typedef DeviceProduct second;

    template<typename T, typename R = T>
    __device__
    R operator() (const T &lhs, const T &rhs) {
        return lhs * rhs;
    }

    template<typename R>
    static constexpr R neutral_value { 1 };
};

struct DeviceSumOfSquares {
    struct Loader {
        template <typename T, typename R = T>
        __device__
        R operator() (const T* ptr, int pos) const {
            auto val = ptr[pos];   // load
            return R{val} * R{val};   // squared
        }
    };
    // round 2 just uses the basic sum reduction
    typedef DeviceSum second;

    template<typename R>
    __device__
    R operator() (const R &lhs, const R &rhs) const {
        return lhs + rhs;
    }

    template<typename R>
    static constexpr R neutral_value { 0 };
};

struct DeviceMin {
    typedef IdentityLoader Loader;
    typedef DeviceMin second;

    template<typename R>
    __device__
    R operator() (const R &lhs, const R &rhs) {
        return lhs <= rhs? lhs: rhs;
    }

    template<typename R>
    static constexpr R neutral_value { std::numeric_limits<R>::max() };
};

struct DeviceMax {
    typedef IdentityLoader Loader;
    typedef DeviceMax second;

    template<typename R>
    __device__
    R operator() (const R &lhs, const R &rhs) {
        return lhs >= rhs? lhs: rhs;
    }

    template<typename R>
    static constexpr R neutral_value { std::numeric_limits<R>::max() };
};

template <typename Op>
struct ReduceDispatcher {
    template <typename T, typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
    gdf_error operator()(
        gdf_column *   col,
        void *         dev_result,
        gdf_size_type  dev_result_size)
    {
        return ReduceOp<T, Op>::launch(
            col,
            Op::template neutral_value<T>,
            reinterpret_cast<T*>(dev_result),
            dev_result_size);
    }

    template <typename T, typename std::enable_if_t<!std::is_arithmetic<T>::value, T>* = nullptr>
    gdf_error operator()(
        gdf_column *   col,
        void *         dev_result,
        gdf_size_type  dev_result_size)
    {
        return GDF_UNSUPPORTED_DTYPE;
    }
};

template <>
struct ReduceDispatcher<DeviceCountNonZeros> {
    using result_type = gdf_size_type;

    template <typename T, typename std::enable_if_t<std::is_integral<T>::value, T>* = nullptr>
    gdf_error operator()(
        gdf_column *   col,
        void *         dev_result,
        gdf_size_type)
    {
        constexpr const gdf_size_type neutral_value = 0;
        return ReduceOp<T, DeviceCountNonZeros, result_type>::launch(
            col, neutral_value,
            reinterpret_cast<result_type*>(dev_result),
            sizeof(result_type));
    }

    template <typename T, typename std::enable_if_t<!std::is_integral<T>::value, T>* = nullptr>
    gdf_error operator()(
        gdf_column *    col,
        void *          dev_result,
        gdf_size_type)
    {
        return GDF_UNSUPPORTED_DTYPE;
    }
};

gdf_error gdf_sum(gdf_column *col,
                  void *dev_result,
                  gdf_size_type dev_result_size)
{   
    return cudf::type_dispatcher(col->dtype, ReduceDispatcher<DeviceSum>(),
                                 col, dev_result, dev_result_size);
}

unsigned int gdf_reduction_get_intermediate_output_size() {
    return REDUCTION_BLOCK_SIZE;
}

gdf_error gdf_product(gdf_column *col,
                      void *dev_result,
                      gdf_size_type dev_result_size)
{
    return cudf::type_dispatcher(col->dtype, ReduceDispatcher<DeviceProduct>(),
                                 col, dev_result, dev_result_size);
}

gdf_error gdf_sum_of_squares(gdf_column *col,
                             void *dev_result,
                             gdf_size_type dev_result_size)
{
    return cudf::type_dispatcher(col->dtype, ReduceDispatcher<DeviceSumOfSquares>(),
                                 col, dev_result, dev_result_size);
}

gdf_error gdf_count_nonzeros(
    gdf_column *    __restrict__  col,
    gdf_size_type * __restrict__  dev_result)
{
    return cudf::type_dispatcher(
        col->dtype, ReduceDispatcher<DeviceSumOfSquares>{},
        col, dev_result, sizeof(*dev_result));
}

gdf_error gdf_min(gdf_column *col,
                  void *dev_result,
                  gdf_size_type dev_result_size)
{
    return cudf::type_dispatcher(col->dtype, ReduceDispatcher<DeviceMin>(),
                                 col, dev_result, dev_result_size);
}

gdf_error gdf_max(gdf_column *col,
                  void *dev_result,
                  gdf_size_type dev_result_size)
{
    return cudf::type_dispatcher(col->dtype, ReduceDispatcher<DeviceMax>(),
                                 col, dev_result, dev_result_size);
}


unsigned int gdf_reduce_optimal_output_size() {
    return REDUCTION_BLOCK_SIZE;
}
