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

#include <float.h>
#include "computing.hpp"
#include "model.hpp"
#include "topological_distance_kernel.hpp"
#include "angular_distance_kernel.hpp"
#include "taxicab_distance_kernel.hpp"
#include "squared_distance_kernel.hpp"
#include "euclidean_distance_kernel.hpp"

using namespace std;
using namespace som;

Computing::Computing(Model &model, const Device deviceType) :
model_(model),
context_(nullptr),
deviceId_(nullptr),
commandQueue_(nullptr),
inputVectorBuffer_(nullptr),
weightsBuffer_(nullptr),
weightDistancesBuffer_(nullptr),
angularDistanceKernel_(nullptr),
taxicabDistanceKernel_(nullptr),
euclideanDistanceKernel_(nullptr),
squaredDistanceKernel_(nullptr),
pointDistanceKernel_(nullptr) {
    cl_platform_id platforms;
    cl_uint num_platforms;
    clGetPlatformIDs(1, &platforms, &num_platforms);
    
    cl_uint num_devices;
    
    switch (deviceType) {
        case CPU:         clGetDeviceIDs(platforms, CL_DEVICE_TYPE_CPU, 1, &deviceId_, &num_devices); break;
        case GPU:         clGetDeviceIDs(platforms, CL_DEVICE_TYPE_GPU, 1, &deviceId_, &num_devices); break;
        case ALL_DEVICES: clGetDeviceIDs(platforms, CL_DEVICE_TYPE_ALL, 1, &deviceId_, &num_devices); break;
    }
    
    cl_int error = 0;
    context_ = clCreateContext(nullptr, 1, &deviceId_, nullptr, nullptr, &error);
    commandQueue_ = clCreateCommandQueue(context_, deviceId_, 0, &error);
    
    pointDistanceKernel_ = new TopologicalDistanceKernel(context_, commandQueue_, deviceId_);
    angularDistanceKernel_ = new AngularDistanceKernel(context_, commandQueue_, deviceId_);
    taxicabDistanceKernel_ = new TaxicabDistanceKernel(context_, commandQueue_, deviceId_);
    euclideanDistanceKernel_ = new EuclideanDistanceKernel(context_, commandQueue_, deviceId_);
    squaredDistanceKernel_ = new SquaredDistanceKernel(context_, commandQueue_, deviceId_);
    
    auto channels = model_.getChannelsCount();
    auto nodesCount = model_.getNodesCount();
    
    inputVectorBuffer_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeof(cl_float) * channels, nullptr, &error);
    
    cl_float *weights = &model_.getWeights();
    weightsBuffer_ = clCreateBuffer(context_, CL_MEM_COPY_HOST_PTR, nodesCount * channels * sizeof(cl_float), weights, &error);
    weightDistancesBuffer_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, nodesCount * sizeof(cl_float), nullptr, &error);
    
    pointDistanceKernel_->connect(model_);
    angularDistanceKernel_->connect(model_, inputVectorBuffer_, weightsBuffer_, weightDistancesBuffer_);
    taxicabDistanceKernel_->connect(model_, inputVectorBuffer_, weightsBuffer_, weightDistancesBuffer_);
    euclideanDistanceKernel_->connect(model_, inputVectorBuffer_, weightsBuffer_, weightDistancesBuffer_);
    squaredDistanceKernel_->connect(model_, inputVectorBuffer_, weightsBuffer_, weightDistancesBuffer_);
}

Computing::~Computing() {
    delete pointDistanceKernel_;
    delete angularDistanceKernel_;
    delete taxicabDistanceKernel_;
    delete euclideanDistanceKernel_;
    delete squaredDistanceKernel_;
    
    clReleaseMemObject(inputVectorBuffer_);
    clReleaseMemObject(weightsBuffer_);
    clReleaseMemObject(weightDistancesBuffer_);

    clReleaseCommandQueue(commandQueue_);
    clReleaseDevice(deviceId_);
    clReleaseContext(context_);
}

#pragma mark - BMU

size_t Computing::bmuIndex(const cl_float &inputVector, bool accumulateDistances) {
    cl_uint index = 0;
    
    auto metric = model_.getMetric();
    
    switch (metric) {
        case EUCLIDEAN: euclideanDistanceKernel_->compute(inputVector); break;
        case SQUARED: squaredDistanceKernel_->compute(inputVector); break;
        case TAXICAB: taxicabDistanceKernel_->compute(inputVector); break;
        case ANGULAR: angularDistanceKernel_->compute(inputVector); break;
    }
    
    cl_float *distancesAcumulator = &model_.getDistancesAccumulator();
    cl_float *distances = &model_.getDistances();
    auto nodesCount = model_.getNodesCount();

    auto lowestDistance = FLT_MAX;
    for (auto i = 0; i < nodesCount; i++) {
        auto distance = distances[i];
        
        if (accumulateDistances && distance > 0.0) {
            distancesAcumulator[i] += distance;
        }

        if (distance < lowestDistance) {
            lowestDistance = distance;

            index = i;
        }
    }
    
    return index;
}

#pragma mark - Topological distances

cl_float & Computing::pointDistances(const size_t index) {
    return pointDistanceKernel_->compute(index);
}

#pragma mark - Error

double Computing::error() {
    long double error = 0.0;
    
    auto accumulateDistances = false;
    auto channels = model_.getChannelsCount();
    auto elementsCount = model_.getDataCount();
    
    cl_float *data = &model_.getData();
    cl_float *distances = &model_.getDistances();
    
    for (auto i = 0; i < elementsCount; i++) {
        error += distances[bmuIndex(data[i * channels], accumulateDistances)];
    }
    
    return 1. / elementsCount * error;
}
