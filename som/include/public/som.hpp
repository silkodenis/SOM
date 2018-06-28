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

#ifndef som_hpp
#define som_hpp

#include "som_types.hpp"

namespace som {
    
    using namespace std;
    
    class Model;
    class Trainer;
    class Computing;
    
    class SOM {
        
    public:
        SOM(const Device);
        ~SOM();

        // Create
        bool create(const size_t cols, const size_t rows, const size_t hexSize, const size_t channels);
        bool create(const size_t radius, const size_t hexSize, const size_t channels);
        
        bool load(const string &filePath);
        bool save(const string &filePath);
        
        // Prepare
        void prepare(const vector<vector<float>> &data, const Normalization = NO_NORM, const InitialWeights = RANDOM_FROM_DATA);
        void prepare(const uint8_t *pixelBuffer, const size_t lenght, const Normalization = NO_NORM, const InitialWeights = RANDOM_FROM_DATA);
        
        // Training
        void train(const size_t iterationsCount, const double learningRate, const DistanceMetric = EUCLIDEAN, bool manual = false);
        bool train(size_t epochs);
        
        // Usage
        void setLabel(int label, size_t index);
        void setLabels(vector<int> labels, vector<size_t> indices);
        
        void setRandomWeights(const float min, const float max);
        
        int predict(const vector<float> &vector) const;
        int predict(const uint8_t &pixel) const;
        
        size_t computeBmuIndex(const vector<float> &vector) const;
        size_t computeBmuIndex(const uint8_t &pixel) const;
        
        double computeError();
        
        // Release memory
        void release();
        
        // ModelView
        vector<Cell> getCells() const;
        
        double getWidth() const;
        double getHeight() const;
        
        size_t getNodeDimensionality() const;
        size_t getTopologicalDimensionality() const;

    private:
        Device deviceType_;
        
        Model *model_;
        Trainer *trainer_;
        Computing *computing_;
        
    };
    
}

#endif /* som_hpp */
