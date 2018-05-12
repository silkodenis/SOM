/* Copyright © 2018 Denis Silko. All rights reserved.
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

#include "weight_distance_kernel.hpp"
#include "model.hpp"

using namespace std;
using namespace som;

WeightDistanceKernel::WeightDistanceKernel(const string code, const string name, cl_context &context, cl_command_queue &commandQueue, cl_device_id &deviceId) :
Kernel(code, name, context, commandQueue, deviceId) {}

void WeightDistanceKernel::connect(const Model &model, const cl_mem &inputBuffer, const cl_mem &weightsBuffer, const cl_mem &distancesBuffer) {
    inputBuffer_ = inputBuffer;
    weightsBuffer_ = weightsBuffer;
    distancesBuffer_ = distancesBuffer;
    
    weights_ = &model.getWeights();
    distances_ = &model.getDistances();
    channels_ = model.getChannelsCount();
    nodesCount_ = model.getNodesCount();
    
    clSetKernelArg(kernel_, 0, sizeof(cl_mem), &inputBuffer_);
    clSetKernelArg(kernel_, 1, sizeof(cl_mem), &weightsBuffer_);
    clSetKernelArg(kernel_, 2, sizeof(cl_uint), &channels_);
    clSetKernelArg(kernel_, 3, sizeof(cl_mem), &distancesBuffer_);
    
    globalWorkSize_[0] = nodesCount_;
}

void WeightDistanceKernel::compute(const cl_float &vector) {
    clEnqueueWriteBuffer(commandQueue_, weightsBuffer_, CL_TRUE, 0, nodesCount_ * channels_ * sizeof(cl_float), &weights_[0], 0, nullptr, nullptr);
    clEnqueueWriteBuffer(commandQueue_, inputBuffer_, CL_TRUE, 0, channels_ * sizeof(cl_float), &vector, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue_, kernel_, 1, nullptr, globalWorkSize_, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(commandQueue_, distancesBuffer_, CL_TRUE, 0, nodesCount_ * sizeof(cl_float), &distances_[0], 0, nullptr, nullptr);
}
