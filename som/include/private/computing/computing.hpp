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

#ifndef computing_hpp
#define computing_hpp

#include "types.hpp"

namespace som {
    
    using namespace std;
    
    class Model;
    
    class TopologicalDistanceKernel;
    class SADDistanceKernel;
    class SSDDistanceKernel;
    class MAEDistanceKernel;
    class MSEDistanceKernel;
    class EuclideanDistanceKernel;
    class ManhattanDistanceKernel;
    class ChebyshevDistanceKernel;
    class MinkowskiDistanceKernel;
    class CanberraDistanceKernel;
    class CosineDistanceKernel;
    
    class Computing {
        
    public:
        Computing(Model&, const Device);
        ~Computing();
        
        cl_float & pointDistances(const size_t index);
        
        size_t bmuIndex(const cl_float &vector, bool accumulateDistances);

        double error();
        
    private:
        Model &model_;
        
        cl_context context_;
        cl_device_id deviceId_;
        cl_command_queue commandQueue_;
        
        cl_mem inputVectorBuffer_;
        cl_mem weightsBuffer_;
        cl_mem weightDistancesBuffer_;
        
        SADDistanceKernel *sadDistanceKernel_;
        SSDDistanceKernel *ssdDistanceKernel_;
        MAEDistanceKernel *maeDistanceKernel_;
        MSEDistanceKernel *mseDistanceKernel_;
        EuclideanDistanceKernel *euclideanDistanceKernel_;
        ManhattanDistanceKernel *manhattanDistanceKernel_;
        ChebyshevDistanceKernel *chebyshevDistanceKernel_;
        MinkowskiDistanceKernel *minkowskiDistanceKernel_;
        CanberraDistanceKernel *canberraDistanceKernel_;
        CosineDistanceKernel *cosineDistanceKernel_;
        TopologicalDistanceKernel *pointDistanceKernel_;
    };
    
}

#endif /* Computing_hpp */
