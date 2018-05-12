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

#include "model.hpp"
#include "trainer.hpp"
#include "computing.hpp"

using namespace std;
using namespace som;

Trainer::Trainer(Model &model, Computing &computing) :
model_(model),
computing_(computing),
remainingIterationsCount_(0) {}

#pragma mark - Train

void Trainer::learn(const size_t iterationsCount, const double learningRate, bool epochMode) {
    learningRate_ = learningRate;
    startLearningRate_ = learningRate;
    topologicalRadius_ = model_.getTopologicalRadius();
    timeConstant_ = iterationsCount / log(topologicalRadius_);
    remainingIterationsCount_ = iterationsCount;
    iterationCount_ = 0;
    
    if (!epochMode) {
        while (!epoch()) {}
    }
}

bool Trainer::epoch() {
    if (remainingIterationsCount_ > 0) {
        cl_int *activationStates = &model_.getActivationStates();
        auto nodesCount = model_.getNodesCount();
        
        cl_float &vector = model_.getRandomDataVector();
        size_t bmuIndex = computing_.bmuIndex(vector, true);
        
        activationStates[bmuIndex]++;
        
        neighbourhoodRadius_ = topologicalRadius_ * exp(-(double)iterationCount_ / timeConstant_);
        double squareNeighborhood  = neighbourhoodRadius_ * neighbourhoodRadius_;
        
        cl_float *topologicalDistances = &computing_.pointDistances(bmuIndex);

        for (auto i = 0; i < nodesCount; i++) {
            double distance = topologicalDistances[i];

            if (distance <= squareNeighborhood ) {
                influence_ = exp(-(distance) / (2 * squareNeighborhood));

                adjustWeights(&vector, i);
            }
        }
        
        learningRate_ = startLearningRate_ * exp(-(double)iterationCount_ / remainingIterationsCount_);
        
        iterationCount_++;
        remainingIterationsCount_--;
    } else {
        return true;
    }
    
    return false;
}

void Trainer::adjustWeights(const cl_float *inputVector, const size_t nodeIndex) {
    cl_float *weights = &model_.getWeights();
    auto vectorSize = model_.getChannelsCount();
    
    auto index0 = nodeIndex * vectorSize;
    for (auto i = 0; i < vectorSize; i++) {
        auto index_ = index0 + i;
        weights[index_] += learningRate_ * influence_ * (inputVector[i] - weights[index_]);
    }
}
