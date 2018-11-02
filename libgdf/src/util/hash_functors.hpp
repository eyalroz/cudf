#ifndef HASH_FUNCTORS_HPP_
#define HASH_FUNCTORS_HPP_

#include <gdf/cpp/types.hpp>

namespace  gdf {

template <gdf::hash_function_type HashType, typename... Key, unsigned HashSize>
struct hash_functor;

/*
template <typename Key>
struct hash_functor<gdf::hash_function_type::murmur_3, Key...;

template <typename Key>
struct MurmurHash3_32

template <typename Key>
struct MurmurHash3_32
*/

} // namespace gdf

} // namespace hash_functions



#endif // HASH_FUNCTORS_HPP_
