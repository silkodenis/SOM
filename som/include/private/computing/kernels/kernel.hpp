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

#ifndef kernel_hpp
#define kernel_hpp

#include <iostream>

#ifdef __APPLE__
#include <OpenCL/OpenCL.h>
#else
#include <CL/cl.h>
#endif

namespace som {
    
    class Kernel {
        
    public:
        Kernel(const std::string code, const std::string name, cl_context &context, cl_command_queue &commandQueue, cl_device_id &deviceId);
        
        ~Kernel();
        
    protected:
        cl_command_queue commandQueue_;
        cl_program program_;
        cl_kernel kernel_;
        
        cl_context context_;
        
        size_t globalWorkSize_[1];
        
    };
    
}

#endif /* kernel_hpp */
