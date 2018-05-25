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
#include "computing.hpp"

using namespace som;
using namespace std;

bool cmpf(cl_float a, cl_float b, cl_float epsilon = 0.00005f) {
    return (fabs(a - b) < epsilon);
}

void test(const Model &model, const vector<cl_float> &expectedDistances) {
    cl_float *distances = &model.getDistances();
    
    const auto vectorSize = expectedDistances.size();
    
    for (auto i = 0; i < vectorSize; i++) {
        assert(cmpf(distances[i], expectedDistances[i]));
    }
}

int main(int argc, const char * argv[]) {
    
    // Create model
    const auto cols = 3;
    const auto rows = 3;
    const auto channels = 3;
    const auto hexSize = 5;
    const auto nodesCount = cols * rows;
    
    Model model(cols, rows, channels, hexSize);

    // Create model with initial weights
    cl_float *weights = &model.getWeights();
    
    vector<cl_float> inputVector {0.233, 0.924, 0.455};
    
    vector<vector<cl_float>> initialWeights {
        {0.262, 0.358, 0.454},
        {0.159, 0.378, 0.375},
        {0.844, 0.588, 0.144},
        {0.520, 0.828, 0.727},
        {0.925, 0.148, 0.575},
        {0.271, 0.778, 0.391},
        {0.353, 0.418, 0.844},
        {0.525, 0.228, 0.716},
        {0.766, 0.878, 0.334},
    };
    
    for (auto i = 0; i < nodesCount; i++) {
        memcpy(&weights[i * channels], initialWeights[i].data(), sizeof(cl_float) * channels);
    }

    Computing computing(model, ALL_DEVICES);
    
    model.setMetric(SAD);
    computing.bmuIndex(*inputVector.data(), false);
    
    test(model, { // expected SAD distances
        0.596,
        0.7,
        1.258,
        0.655,
        1.588,
        0.248,
        1.015,
        1.249,
        0.7
    });
   
    model.setMetric(SSD);
    computing.bmuIndex(*inputVector.data(), false);
    
    test(model, { // expected SSD distances
        0.321198,
        0.309992,
        0.582938,
        0.165569,
        1.09544,
        0.026856,
        0.421757,
        0.637801,
        0.300846
    });
    
    model.setMetric(MAE);
    computing.bmuIndex(*inputVector.data(), false);
    
    test(model, { // expected MAE distances
        0.198667,
        0.233333,
        0.419333,
        0.218333,
        0.529333,
        0.0826667,
        0.338333,
        0.416333,
        0.233333
    });
    
    model.setMetric(MSE);
    computing.bmuIndex(*inputVector.data(), false);
    
    test(model, { // expected MSE distances
        0.107066,
        0.103331,
        0.194313,
        0.0551897,
        0.365147,
        0.008952,
        0.140586,
        0.2126,
        0.100282
    });
    
    model.setMetric(EUCLIDEAN);
    computing.bmuIndex(*inputVector.data(), false);
    
    test(model, { // expected EUCLIDEAN distances
        0.566743,
        0.556769,
        0.763504,
        0.406902,
        1.04663,
        0.163878,
        0.649428,
        0.798625,
        0.548494
    });
    
    model.setMetric(MANHATTAN);
    computing.bmuIndex(*inputVector.data(), false);
    
    test(model, { // expected MANHATTAN distances
        0.596,
        0.7,
        1.258,
        0.655,
        1.588,
        0.248,
        1.015,
        1.249,
        0.7
    });
    
    model.setMetric(CHEBYSHEV);
    computing.bmuIndex(*inputVector.data(), false);
    
    test(model, { // expected CHEBYSHEV distances
        0.566,
        0.546,
        0.611,
        0.287,
        0.776,
        0.146,
        0.506,
        0.696,
        0.533
    });
    
    model.setMetric(MINKOWSKI);
    computing.bmuIndex(*inputVector.data(), false);
    
    test(model, { // expected MINKOWSKI distances
        0.566025,
        0.547024,
        0.666529,
        0.35476,
        0.928469,
        0.150798,
        0.575037,
        0.724208,
        0.535184
    });
    
    model.setMetric(CANBERRA);
    computing.bmuIndex(*inputVector.data(), false);
    
    test(model, { // expected CANBERRA distances
        0.501184,
        0.704516,
        1.30874,
        0.666055,
        1.43797,
        0.236828,
        0.881288,
        1.21228,
        0.712419
    });
    
    model.setMetric(COSINE);
    computing.bmuIndex(*inputVector.data(), false);
    
    test(model, { // expected COSINE distances
        0.107248,
        0.0508716,
        0.265607,
        0.0540903,
        0.471089,
        0.00316954,
        0.19736,
        0.319426,
        0.108
    });
    
    return 0;
}
