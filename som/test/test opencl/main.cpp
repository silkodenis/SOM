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

#ifdef __APPLE__
#include <OpenCL/OpenCL.h>
#else
#include <CL/cl.h>
#endif

int main(int argc, const char * argv[]) {
    
    cl_platform_id platforms;
    cl_uint num_platforms;
    clGetPlatformIDs(1, &platforms, &num_platforms);
    
    cl_uint num_devices;
    cl_device_id deviceId;
    clGetDeviceIDs(platforms, CL_DEVICE_TYPE_ALL, 1, &deviceId, &num_devices);
    
    cl_int error = 0;
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, &error);
    
    assert(error == 0);
    
    __unused cl_command_queue commandQueue = clCreateCommandQueue(context, deviceId, 0, &error);
    
    assert(error == 0);
    
    clReleaseCommandQueue(commandQueue);
    clReleaseDevice(deviceId);
    clReleaseContext(context);
    
    return 0;
}
