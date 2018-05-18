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

#ifndef model_hpp
#define model_hpp

#include <stdio.h>
#include <random>
#include <chrono>
#include "grid.hpp"

namespace som {
    
    class Normalizer;
    
    class Model {
        
    public:
        Model();
        Model(const size_t cols, const size_t rows, const size_t channels, const double hexSize);
        Model(const size_t radius, const size_t channels, const double hexSize);
        ~Model();
        
        bool load(const string &filePath);
        bool save(const string &filePath);
        
        void prepare(const vector<vector<cl_float>> &data, const Normalization, const Weights);
        void prepare(const uint8_t *pixelBuffer, const size_t lenght, const Normalization, const Weights);
        
        cl_float & normalizeVector(const vector<cl_float> &inputVector);
        cl_float & normalizeVector(const uint8_t *inputVector);
        
        void setMetric(Metric);
        void setRandomWeights(const double min, const double max);
        void setLabel(cl_int label, size_t index);
        void setLabels(vector<cl_int> labels, vector<size_t> indices);
        
        double getWidth() const;
        double getHeight() const;
        double getTopologicalRadius() const;
        
        cl_float & getRandomDataVector();
        cl_float & getData() const;
        cl_float & getPoints() const;
        cl_int   & getLabels() const;
        cl_float & getDistances() const;
        cl_float & getDistancesAccumulator() const;
        cl_float & getWeights() const;
        cl_int  & getActivationStates() const;
        
        size_t getDataCount() const;
        size_t getNodesCount() const;
        size_t getChannelsCount() const;
        size_t getTopologicalDimensionality() const;
        
        vector<Cell> getCells() const;
        
        Metric getMetric() const;

    private:
        bool create();
        bool create(const size_t cols, const size_t rows, const size_t channels, const double hexSize);
        bool create(const size_t radius, const size_t channels, const double hexSize);
        
        void setWeights(const Weights);
        
        Grid *grid_;
        Normalizer *normalizer_;
        
        Metric metric_;
        
        vector<Cell> cells_;
        
        size_t dataCount_;
        size_t nodesCount_;
        size_t channelsCount_;
        
        cl_float *input_;
        cl_float *data_;
        cl_int   *labels_;
        cl_float *weights_;
        cl_float *distances_;
        cl_float *distancesAccumulator_;
        cl_int  *activationStates_;
        
        uniform_real_distribution<double> uniform_;
        mt19937_64 rng_;
    };
    
}

#endif /* model_hpp */
