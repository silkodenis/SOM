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

#include <iostream>
#include <assert.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/OpenCL.h>
#else
#include <CL/cl.h>
#endif

bool test(cl_device_type) {
    bool result = false;
    
    cl_platform_id platforms = nullptr;
    cl_uint num_platforms;
    clGetPlatformIDs(1, &platforms, &num_platforms);
    
    cl_uint num_devices;
    cl_device_id deviceId = nullptr;
    
    result = clGetDeviceIDs(platforms, CL_DEVICE_TYPE_ALL, 1, &deviceId, &num_devices) == CL_SUCCESS;
    
    cl_int error = 0;
    cl_context context = nullptr;
    context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, &error);
    
    if (result) {
        assert(error == 0);
    }
    
    cl_command_queue commandQueue = nullptr;
    commandQueue = clCreateCommandQueue(context, deviceId, 0, &error);

    if (result) {
        assert(error == 0);
    }
    
    if (commandQueue) {
        clReleaseCommandQueue(commandQueue);
    }
    
    if (deviceId) {
        clReleaseDevice(deviceId);
    }
    
    if (context) {
        clReleaseContext(context);
    }
    
    return result;
}

int main(int argc, const char * argv[]) {
    
    bool testResult;
    
    testResult = test(CL_DEVICE_TYPE_CPU);
    testResult = test(CL_DEVICE_TYPE_GPU);
    testResult = test(CL_DEVICE_TYPE_ALL);
    
    assert(testResult);
    
    return 0;
}
