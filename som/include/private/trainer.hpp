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

#ifndef trainer_hpp
#define trainer_hpp

#include <iostream>

namespace som {
    
    using namespace std;
    
    class Model;
    class Computing;
    
    class Trainer {
        
    public:
        Trainer(Model&, Computing&);
        
        void learn(const size_t iterationsCount, const double learningRate, bool epochMode);
        bool epoch();
    
    private:
        void adjustWeights(const cl_float *inputVector, const size_t nodeIndex);
        
        Model &model_;
        Computing &computing_;
        
        size_t iterationCount_;
        size_t remainingIterationsCount_;
        
        double timeConstant_;
        double neighbourhoodRadius_;
        double topologicalRadius_;
        double influence_;
        double learningRate_;
        double startLearningRate_;
    };
    
}

#endif /* trainer_hpp */
