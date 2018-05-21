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

#include "topological_distance_kernel.hpp"
#include "model.hpp"

using namespace som;

TopologicalDistanceKernel::TopologicalDistanceKernel(cl_context &context, cl_command_queue &commandQueue, cl_device_id &deviceId) :
Kernel("__kernel void pointDistance(__global float *points, __global float *distances, unsigned int bmu_index)"
       "{"
       "    int id = get_global_id(0);"
       "    "
       "    int x = id * 2;"
       "    int y = x + 1;"
       "    int bmu_x = bmu_index * 2;"
       "    int bmu_y = bmu_x + 1;"
       "    "
       "    distances[id] = (points[bmu_x] - points[x]) * (points[bmu_x] - points[x]) + (points[bmu_y] - points[y]) * (points[bmu_y] - points[y]);"
       "}", "pointDistance", context, commandQueue, deviceId) {
    
}

TopologicalDistanceKernel::~TopologicalDistanceKernel() {
    if (distances_) {
        free(distances_);
    }
    
    clReleaseMemObject(pointsBuffer_);
    clReleaseMemObject(distancesBuffer_);
}

void TopologicalDistanceKernel::connect(Model &model) {
    cl_float *points = &model.getPoints();
    auto nodesCount = model.getNodesCount();
    
    distances_ = (cl_float *)malloc(sizeof(cl_float) * nodesCount);
    pointsBuffer_ = clCreateBuffer(context_, CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * nodesCount * 2, points, nullptr);
    distancesBuffer_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, nodesCount * sizeof(cl_float), nullptr, nullptr);
    
    clSetKernelArg(kernel_, 0, sizeof(cl_mem), &pointsBuffer_);
    clSetKernelArg(kernel_, 1, sizeof(cl_mem), &distancesBuffer_);
    
    globalWorkSize_[0] = nodesCount;
}

cl_float & TopologicalDistanceKernel::compute(const size_t index) const {
    clSetKernelArg(kernel_, 2, sizeof(size_t), &index);
    
    clEnqueueNDRangeKernel(commandQueue_, kernel_, 1, nullptr, globalWorkSize_, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(commandQueue_, distancesBuffer_, CL_TRUE, 0, globalWorkSize_[0] * sizeof(cl_float), distances_, 0, nullptr, nullptr);
    
    return *distances_;
}
