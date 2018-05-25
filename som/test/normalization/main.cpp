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

#include "normalizer.hpp"
#include <assert.h>

using namespace som;
using namespace std;

bool cmpf(cl_float a, cl_float b, cl_float epsilon = 0.00005f) {
    return (fabs(a - b) < epsilon);
}

void test(const cl_float *outputVector, const vector<cl_float> &expectedOutputVector) {
    const auto vectorSize = expectedOutputVector.size();
    
    for (auto i = 0; i < vectorSize; i++) {
        assert(cmpf(outputVector[i], expectedOutputVector[i]));
    }
}

void test(const cl_float *outputData, const vector<vector<cl_float>> &expectedOutputData) {
    const auto vectorSize = expectedOutputData[0].size();
    const auto dataSize = expectedOutputData.size();
    
    for (auto i = 0; i < dataSize; i++) {
        for (auto j = 0; j < vectorSize; j++) {
            assert(cmpf(outputData[i * vectorSize + j], expectedOutputData[i][j]));
        }
    }
}

int main(int argc, const char * argv[]) {

    // Input
    vector<cl_float> inputVector {13.3, 109.2, 6.4, 210.5, 727.2};
    
    vector<vector<cl_float>> inputData {
        {32.2, 144.2, 2.5,  203.6, 942.2},
        {29.5, 131.9, 5.3,  233.5, 746.4},
        {23.1, 98.4,  9.7,  198.9, 812.3},
        {25.6, 102.3, 12.5, 257.3, 900.6},
        {12.3, 109.7, 4.8,  229.3, 630.5},
    };

    // Test
    Normalizer normalizer(inputVector.size());
    
    cl_float *outputVector = (cl_float *)malloc(sizeof(cl_float) * inputVector.size());
    cl_float *outputData = (cl_float *)malloc(sizeof(cl_float) * inputVector.size() * inputData.size());
    
    // Test normalize by columns
    normalizer.setNormalizationType(MINMAX_BY_COLUMNS);
    normalizer.normalize(inputData, outputData);
    normalizer.normalize(inputVector, *outputVector);
    
    test(outputData, { // expected output data
        {1.000000, 1.000000,  0.00, 0.0804796, 1.000000},
        {0.864322, 0.731441,  0.28, 0.592466,  0.371832},
        {0.542714, 0.000000,  0.72, 0.000000,  0.583253},
        {0.668342, 0.0851529, 1.00, 1.000000,  0.866538},
        {0.000000, 0.246725,  0.23, 0.520548,  0.000000},
    });
    
    test(outputVector, { // expected vector
        0.0502513,
        0.235808,
        0.39,
        0.19863,
        0.310234
    });
    
    // Test normalize by rows
    normalizer.setNormalizationType(MINMAX_BY_ROWS);
    normalizer.normalize(inputData, outputData);
    normalizer.normalize(inputVector, *outputVector);
    
    test(outputData, { // expected output data
        {0.0330186, 0.147866, 0.00256356, 0.208776, 0.966153},
        {0.0371685, 0.166187, 0.00667774, 0.294198, 0.940427},
        {0.0274204, 0.116804, 0.0115142,  0.2361,   0.964225},
        {0.0271579, 0.108526, 0.0132607,  0.272958, 0.955407},
        {0.0180899, 0.161338, 0.00705946, 0.337236, 0.927289}
    });
    
    test(outputVector, { // expected vector
        0.0173849,
        0.142739,
        0.00836567,
        0.275152,
        0.95055
    });
    
    free(outputVector);
    free(outputData);
    
    return 0;
}
