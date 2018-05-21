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
#include "topological_distance_kernel.hpp"

using namespace som;
using namespace std;

bool cmpf(cl_float a, cl_float b, cl_float epsilon = 0.00005f) {
    return (fabs(a - b) < epsilon);
}

void test(const cl_float *distances, const vector<cl_float> &expectedDistances) {
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
    const auto bmuIndex = 3;
    
    Model model(cols, rows, channels, hexSize);
    
    // Test kernel
    TopologicalDistanceKernel kernel(context, commandQueue, deviceId);
    kernel.connect(model);
    
    cl_float *distances = &kernel.compute(bmuIndex);
    
    test(distances, { // expected distances
        18.75,
        18.75,
        75,
        0,
        18.75,
        56.25,
        18.75,
        56.25,
        75
    });
    
    // Release
    clReleaseCommandQueue(commandQueue);
    clReleaseDevice(deviceId);
    clReleaseContext(context);
    
    return 0;
}
