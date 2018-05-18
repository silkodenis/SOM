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

#include "kernel.hpp"

using namespace std;
using namespace som;

Kernel::Kernel(const string code, const string name, cl_context &context, cl_command_queue &commandQueue, cl_device_id &deviceId) :
commandQueue_(commandQueue), context_(context)
{
    cl_int error = 0;
    
    const char *source_str = code.c_str();
    size_t source_size = code.size();
    
    program_ = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &error);
    
    clBuildProgram(program_, 1, &deviceId, nullptr, nullptr, nullptr);
    
    kernel_ = clCreateKernel(program_, name.c_str(), &error);
}

Kernel::~Kernel() {
    clReleaseProgram(program_);
    clReleaseKernel(kernel_);
}
