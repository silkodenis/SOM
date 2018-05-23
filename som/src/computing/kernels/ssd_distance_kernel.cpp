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

#include "ssd_distance_kernel.hpp"

using namespace som;

SSDDistanceKernel::SSDDistanceKernel(cl_context &context, cl_command_queue &commandQueue, cl_device_id &deviceId) :
WeightDistanceKernel("__kernel void ssdDistance(__global float *inputVector, __global float *weights, unsigned int vecSize, __global float *result)"
                     "{"
                     "    int id = get_global_id(0);"
                     ""
                     "    float distance = 0.0;"
                     ""
                     "    for (int i = 0; i < vecSize; i++) {"
                     "        int index = id * vecSize + i;"
                     ""
                     "        distance += pow(inputVector[i] - weights[index], 2);"
                     "    }"
                     ""
                     "    result[id] = distance;"
                     "}", "ssdDistance", context, commandQueue, deviceId) {}
