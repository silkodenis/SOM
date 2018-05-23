/* Copyright 2018 Denis Silko. All rights reserved.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 http:www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#ifndef som_types_hpp
#define som_types_hpp

#include <iostream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/OpenCL.h>
#else
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#endif

namespace som {
    
#pragma mark - Constants
    
    static const int HEXAGON_CORNERS_COUNT = 6;
    
#pragma mark - Types
    
    enum Device { ALL_DEVICES, CPU, GPU };
    enum Normalization { NO_NORM, MINMAX_BY_COLUMNS, MINMAX_BY_ROWS };
    enum Weights { RANDOM_01, RANDOM_FROM_DATA };

    enum Metric {
        SAD,       // Sum of Absolute Difference, also known as Manhattan or Taxicab norm
        SSD,       // Sum of Squared Difference, also known as Euclidean norm
        MAE,       // Mean-Absolute Error, is a normalized version SAD
        MSE,       // Mean-Squared Error, is a normalized version SSD
        EUCLIDEAN, // Euclidean Distance
        MANHATTAN, // Manhattan Distance, a special case of the Minkowski distance with p=1 and equivalent to the SAD
        CHEBYSHEV, // Chebyshev Distance, a special case of the Minkowski distance where p goes to infinity
        MINKOWSKI, // Minkowski Distance, with p=3
        CANBERRA,  // Canberra Distance, is a weighted version of the Manhattan distance
        COSINE     // Cosine Distance, contains the dot product scaled by the product of the Euclidean distances from the origin.
    };
    
    struct Cell {
        Cell(cl_float &center_, cl_float &corners_, cl_float &weights_, cl_float &distance_, cl_int &label_, cl_int &state_) :
        center(&center_), corners(&corners_), weights(&weights_), distance(&distance_), label(&label_), state(&state_) {}

        const cl_float *center;
        const cl_float *corners;
        const cl_float *weights;
        const cl_float *distance;
        const cl_int   *label;
        const cl_int   *state;
    };
    
}

#endif /* som_types_hpp */
