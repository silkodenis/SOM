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
#include <vector>
#include <cstring>

using namespace std;
using namespace som;

Normalizer::Normalizer(const size_t channels) :
type_(NO_NORM),
channels_(channels),
derComponents_(nullptr),
sumComponents_(nullptr) {
    derComponents_ = (cl_float *)malloc(sizeof(cl_float) * channels_);
    sumComponents_ = (cl_float *)malloc(sizeof(cl_float) * channels_);
}

Normalizer::~Normalizer() {
    if (derComponents_) {
        free(derComponents_);
    }
    
    if (sumComponents_) {
        free(sumComponents_);
    }
}

#pragma mark - Save/Load

void Normalizer::load(ifstream &is) {
    is.read((char *)&type_, sizeof(Normalization));
    is.read((char *)derComponents_, streamsize(channels_ * sizeof(cl_float)));
    is.read((char *)sumComponents_, streamsize(channels_ * sizeof(cl_float)));
}

void Normalizer::save(ofstream &os) {
    os.write((char *)&type_, sizeof(Normalization));
    os.write((char *)derComponents_, streamsize(channels_ * sizeof(cl_float)));
    os.write((char *)sumComponents_, streamsize(channels_ * sizeof(cl_float)));
}

#pragma mark - Set

void Normalizer::setNormalizationType(const Normalization normalizationType) {
    type_ = normalizationType;
}

#pragma mark - Normalize data

cl_float & Normalizer::normalize(const vector<vector<cl_float>> &data, cl_float *dst) {
    size_t elementsCount = data.size();
    size_t lenght = elementsCount * channels_;
    
    cl_float *src = (cl_float *)malloc(sizeof(cl_float) * lenght);
    for (auto i = 0; i < elementsCount; i++) {
        memcpy(&src[i * channels_], &data[i].data()[0], sizeof(cl_float) * channels_);
    }
    
    cl_float &dst_ = normalizeData(src, lenght, dst);
    
    free(src);
    
    return dst_;
}

cl_float & Normalizer::normalize(const uint8_t *pixelBuffer, const size_t lenght, cl_float *dst) {
    return normalizeData(pixelBuffer, lenght, dst);
}

#pragma mark - Normalize input vector

cl_float & Normalizer::normalize(const vector<cl_float> &vector, cl_float &dst) {
    return normalizeVector(vector.data(), dst);
}

cl_float & Normalizer::normalize(const uint8_t *vector, cl_float &dst) {
    return normalizeVector(vector, dst);
}

