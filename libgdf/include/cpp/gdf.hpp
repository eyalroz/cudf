/*
 * Copyright (c) 2018, BlazingDB Inc.
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

/**
 * @file A C++'ified version of the libGDF API, intended (for now)
 * to be used by deeply-C++ parts of the implementation of libGDF.
 *
 * @note Typically, a C API function header in @ref functions.h will be
 * implemented by a C++ wrapper, or bridge, function under `src/c_wrappers`.
 * This wrapper will take care of stream choice/creation/synchronization,
 * most or all memory management etc. - and will call a strictly-C++
 * implementation from @ref `operators.hpp`. While the wrapper or bridging
 * function will need to be aware are respectful of the C'ish headers
 * under `include/gdf` - the strictly-C++ implementations will only need
 * to include this file (and nothing from `include/`)
 */

#ifndef GDF_GDF_HPP_
#define GDF_GDF_HPP_

#include "gdf/element_data_types.hpp"
#include "gdf/column.hpp"
#include "gdf/operators.hpp"
#include "gdf/exceptions.hpp"
#include "gdf/miscellaneous.hpp"

#endif /* GDF_GDF_HPP_ */
