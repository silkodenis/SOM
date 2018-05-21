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

#include <assert.h>
#include "model.hpp"
#include "angular_distance_kernel.hpp"
#include "taxicab_distance_kernel.hpp"
#include "squared_distance_kernel.hpp"
#include "euclidean_distance_kernel.hpp"

using namespace som;
using namespace std;

bool cmpf(cl_float a, cl_float b, cl_float epsilon = 0.00005f) {
    return (fabs(a - b) < epsilon);
}

void testKernel(WeightDistanceKernel &kernel, const Model &model, const vector<cl_float> &inputVector, const vector<cl_float> &expectedDistances) {
    kernel.compute(*inputVector.data());
    cl_float *distances = &model.getDistances();
    
    const auto vectorSize = expectedDistances.size();

    for (auto i = 0; i < vectorSize; i++) {
        assert(cmpf(distances[i], expectedDistances[i]));
    }
}

int main(int argc, const char * argv[]) {
    
    // Create OpenCL Host
    cl_platform_id platforms;
    cl_uint num_platforms;
    clGetPlatformIDs(1, &platforms, &num_platforms);
    
    cl_uint num_devices;
    cl_device_id deviceId;
    clGetDeviceIDs(platforms, CL_DEVICE_TYPE_ALL, 1, &deviceId, &num_devices);
    
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue commandQueue = clCreateCommandQueue(context, deviceId, 0, nullptr);
    
    // Create model
    const auto cols = 3;
    const auto rows = 3;
    const auto channels = 3;
    const auto hexSize = 5;
    const auto nodesCount = cols * rows;
    
    Model model(cols, rows, channels, hexSize);
    
    // Create buffers with initial weights
    cl_float *weights = &model.getWeights();
    
    vector<vector<cl_float>> initialWeights {
        {0.262, 0.358, 0.454},
        {0.159, 0.378, 0.375},
        {0.844, 0.588, 0.144},
        {0.520, 0.828, 0.727},
        {0.925, 0.148, 0.575},
        {0.271, 0.778, 0.391},
        {0.353, 0.418, 0.844},
        {0.525, 0.228, 0.716},
        {0.766, 0.878, 0.334},
    };
    
    for (auto i = 0; i < nodesCount; i++) {
        memcpy(&weights[i * channels], initialWeights[i].data(), sizeof(cl_float) * channels);
    }
    
    cl_mem inputVectorBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * channels, nullptr, nullptr);
    cl_mem weightsBuffer = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, nodesCount * channels * sizeof(cl_float), weights, nullptr);
    cl_mem weightDistancesBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, nodesCount * sizeof(cl_float), nullptr, nullptr);
    
    // Test kernel
    AngularDistanceKernel angularKernel(context, commandQueue, deviceId);
    TaxicabDistanceKernel taxicabKernel(context, commandQueue, deviceId);
    SquaredDistanceKernel squaredKernel(context, commandQueue, deviceId);
    EuclideanDistanceKernel euclideanKernel(context, commandQueue, deviceId);
    
    angularKernel.connect(model, inputVectorBuffer, weightsBuffer, weightDistancesBuffer);
    taxicabKernel.connect(model, inputVectorBuffer, weightsBuffer, weightDistancesBuffer);
    squaredKernel.connect(model, inputVectorBuffer, weightsBuffer, weightDistancesBuffer);
    euclideanKernel.connect(model, inputVectorBuffer, weightsBuffer, weightDistancesBuffer);

    vector<cl_float> inputVector {0.233, 0.924, 0.455};
    
    testKernel(angularKernel, model, inputVector, { // expected angular distances
        0.467379,
        0.32034,
        0.746025,
        0.330409,
        1.01348,
        0.0796393,
        0.639088,
        0.822251,
        0.469046
    });
    
    testKernel(taxicabKernel, model, inputVector, { // expected taxicab distances
        0.596,
        0.7,
        1.258,
        0.655,
        1.588,
        0.248,
        1.015,
        1.249,
        0.7
    });
    
    testKernel(squaredKernel, model, inputVector, { // expected squared distances
        0.321198,
        0.309992,
        0.582938,
        0.165569,
        1.09544,
        0.026856,
        0.421757,
        0.637801,
        0.300846
    });
    
    testKernel(euclideanKernel, model, inputVector, { // expected euclidean distances
        0.566743,
        0.556769,
        0.763504,
        0.406902,
        1.04663,
        0.163878,
        0.649428,
        0.798625,
        0.548494
    });

    // Release
    clReleaseMemObject(inputVectorBuffer);
    clReleaseMemObject(weightsBuffer);
    clReleaseMemObject(weightDistancesBuffer);
    clReleaseCommandQueue(commandQueue);
    clReleaseDevice(deviceId);
    clReleaseContext(context);
    
    return 0;
}
