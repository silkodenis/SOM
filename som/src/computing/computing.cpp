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
#include "sad_distance_kernel.hpp"
#include "ssd_distance_kernel.hpp"
#include "mae_distance_kernel.hpp"
#include "mse_distance_kernel.hpp"
#include "euclidean_distance_kernel.hpp"
#include "manhattan_distance_kernel.hpp"
#include "chebyshev_distance_kernel.hpp"
#include "minkowski_distance_kernel.hpp"
#include "canberra_distance_kernel.hpp"
#include "cosine_distance_kernel.hpp"

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
sadDistanceKernel_(nullptr),
ssdDistanceKernel_(nullptr),
maeDistanceKernel_(nullptr),
mseDistanceKernel_(nullptr),
euclideanDistanceKernel_(nullptr),
manhattanDistanceKernel_(nullptr),
chebyshevDistanceKernel_(nullptr),
minkowskiDistanceKernel_(nullptr),
canberraDistanceKernel_(nullptr),
cosineDistanceKernel_(nullptr),
pointDistanceKernel_(nullptr) {
    cl_platform_id platforms = nullptr;
    cl_uint num_platforms, num_devices;
    clGetPlatformIDs(1, &platforms, &num_platforms);
    
    cl_int clDeviceIDsStatus;
    switch (deviceType) {
        case CPU:         clDeviceIDsStatus = clGetDeviceIDs(platforms, CL_DEVICE_TYPE_CPU, 1, &deviceId_, &num_devices); break;
        case GPU:         clDeviceIDsStatus = clGetDeviceIDs(platforms, CL_DEVICE_TYPE_GPU, 1, &deviceId_, &num_devices); break;
        case ALL_DEVICES: clDeviceIDsStatus = clGetDeviceIDs(platforms, CL_DEVICE_TYPE_ALL, 1, &deviceId_, &num_devices); break;
    }
    
    assert(clDeviceIDsStatus == CL_SUCCESS);
    
    context_ = clCreateContext(nullptr, 1, &deviceId_, nullptr, nullptr, nullptr);
    commandQueue_ = clCreateCommandQueue(context_, deviceId_, 0, nullptr);
    
    pointDistanceKernel_ = new TopologicalDistanceKernel(context_, commandQueue_, deviceId_);
    sadDistanceKernel_ = new SADDistanceKernel(context_, commandQueue_, deviceId_);
    ssdDistanceKernel_ = new SSDDistanceKernel(context_, commandQueue_, deviceId_);
    maeDistanceKernel_ = new MAEDistanceKernel(context_, commandQueue_, deviceId_);
    mseDistanceKernel_ = new MSEDistanceKernel(context_, commandQueue_, deviceId_);
    euclideanDistanceKernel_ = new EuclideanDistanceKernel(context_, commandQueue_, deviceId_);
    manhattanDistanceKernel_ = new ManhattanDistanceKernel(context_, commandQueue_, deviceId_);
    chebyshevDistanceKernel_ = new ChebyshevDistanceKernel(context_, commandQueue_, deviceId_);
    canberraDistanceKernel_ = new CanberraDistanceKernel(context_, commandQueue_, deviceId_);
    cosineDistanceKernel_ = new CosineDistanceKernel(context_, commandQueue_, deviceId_);
    minkowskiDistanceKernel_ = new MinkowskiDistanceKernel(context_, commandQueue_, deviceId_, deviceType);
    
    auto channels = model_.getChannelsCount();
    auto nodesCount = model_.getNodesCount();
    
    inputVectorBuffer_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeof(cl_float) * channels, nullptr, nullptr);
    
    cl_float *weights = &model_.getWeights();
    weightsBuffer_ = clCreateBuffer(context_, CL_MEM_COPY_HOST_PTR, nodesCount * channels * sizeof(cl_float), weights, nullptr);
    weightDistancesBuffer_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, nodesCount * sizeof(cl_float), nullptr, nullptr);
    
    pointDistanceKernel_->connect(model_);
    sadDistanceKernel_->connect(model_, inputVectorBuffer_, weightsBuffer_, weightDistancesBuffer_);
    ssdDistanceKernel_->connect(model_, inputVectorBuffer_, weightsBuffer_, weightDistancesBuffer_);
    maeDistanceKernel_->connect(model_, inputVectorBuffer_, weightsBuffer_, weightDistancesBuffer_);
    mseDistanceKernel_->connect(model_, inputVectorBuffer_, weightsBuffer_, weightDistancesBuffer_);
    euclideanDistanceKernel_->connect(model_, inputVectorBuffer_, weightsBuffer_, weightDistancesBuffer_);
    manhattanDistanceKernel_->connect(model_, inputVectorBuffer_, weightsBuffer_, weightDistancesBuffer_);
    chebyshevDistanceKernel_->connect(model_, inputVectorBuffer_, weightsBuffer_, weightDistancesBuffer_);
    minkowskiDistanceKernel_->connect(model_, inputVectorBuffer_, weightsBuffer_, weightDistancesBuffer_);
    canberraDistanceKernel_->connect(model_, inputVectorBuffer_, weightsBuffer_, weightDistancesBuffer_);
    cosineDistanceKernel_->connect(model_, inputVectorBuffer_, weightsBuffer_, weightDistancesBuffer_);
}

Computing::~Computing() {
    delete pointDistanceKernel_;
    delete sadDistanceKernel_;
    delete ssdDistanceKernel_;
    delete maeDistanceKernel_;
    delete mseDistanceKernel_;
    delete euclideanDistanceKernel_;
    delete manhattanDistanceKernel_;
    delete chebyshevDistanceKernel_;
    delete minkowskiDistanceKernel_;
    delete canberraDistanceKernel_;
    delete cosineDistanceKernel_;
    
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
        case SAD: sadDistanceKernel_->compute(inputVector); break;
        case SSD: ssdDistanceKernel_->compute(inputVector); break;
        case MAE: maeDistanceKernel_->compute(inputVector); break;
        case MSE: mseDistanceKernel_->compute(inputVector); break;
        case EUCLIDEAN: euclideanDistanceKernel_->compute(inputVector); break;
        case MANHATTAN: manhattanDistanceKernel_->compute(inputVector); break;
        case CHEBYSHEV: chebyshevDistanceKernel_->compute(inputVector); break;
        case MINKOWSKI: minkowskiDistanceKernel_->compute(inputVector); break;
        case CANBERRA: canberraDistanceKernel_->compute(inputVector); break;
        case COSINE: cosineDistanceKernel_->compute(inputVector); break;
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
