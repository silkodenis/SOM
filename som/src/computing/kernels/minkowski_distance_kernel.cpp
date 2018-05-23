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

#include "minkowski_distance_kernel.hpp"

using namespace som;

MinkowskiDistanceKernel::MinkowskiDistanceKernel(cl_context &context, cl_command_queue &commandQueue, cl_device_id &deviceId, Device device) :
WeightDistanceKernel(device == GPU ?
                     "__kernel void minkowskiDistance(__global float *inputVector, __global float *weights, unsigned int vecSize, __global float *result)"
                     "{"
                     "    int id = get_global_id(0);"
                     ""
                     "    float distance = 0.0;"
                     "    float p = 3.0;"
                     ""
                     "    for (int i = 0; i < vecSize; i++) {"
                     "        int index = id * vecSize + i;"
                     ""
                     "        distance += pow(fabs(inputVector[i] - weights[index]), p);"
                     "    }"
                     ""
                     "    result[id] = pow(distance, 1.0/p);"
                     "}"
                     :
                     "__kernel void minkowskiDistance(__global float *inputVector, __global float *weights, unsigned int vecSize, __global float *result)"
                     "{"
                     "    int id = get_global_id(0);"
                     ""
                     "    float distance = 0.0;"
                     "    float p = 3.0;"
                     ""
                     "    for (int i = 0; i < vecSize; i++) {"
                     "        int index = id * vecSize + i;"
                     ""
                     "        distance += pow(fabs(inputVector[i] - weights[index]), p);"
                     "    }"
                     ""
                     "    result[id] = exp((1.0/p) * log(distance));"
                     "}", "minkowskiDistance", context, commandQueue, deviceId) {}
