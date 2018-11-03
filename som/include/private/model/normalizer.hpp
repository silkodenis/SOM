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

#ifndef normalizer_hpp
#define normalizer_hpp

#include "types.hpp"
#include <vector>
#include <fstream>
#include <math.h>
#include <float.h>

namespace som {
    
    using namespace std;
    
    class Normalizer {
        
    public:
        Normalizer(const size_t channels);
        ~Normalizer();
        
        void load(ifstream &is);
        void save(ofstream &os);
        
        void setNormalizationType(const Normalization);
        
        cl_float & normalize(const vector<vector<cl_float>> &src, cl_float *dst);
        cl_float & normalize(const uint8_t *src, const size_t lenght, cl_float *dst);
        cl_float & normalize(const vector<cl_float> &src, cl_float &dst);
        cl_float & normalize(const uint8_t *src, cl_float &dst);
        
    private:
        template <typename T> cl_float & normalizeData(const T *src, const size_t lenght, cl_float *dst);
        template <typename T> cl_float & normalizeVector(const T *src, cl_float &dst);
        template <typename T> void minmaxByRows(const T *src, cl_float *dst);
        template <typename T> void minmaxByCols(const T *src, cl_float *dst);
        template <typename T> void copy(const T *src, cl_float *dst);
        
        Normalization type_;
        
        size_t channels_;
        cl_float *derComponents_;
        cl_float *sumComponents_;
        
    };
    
#pragma mark - Templates
    
    template <typename T> cl_float & Normalizer::normalizeData(const T *src, const size_t lenght, cl_float *dst) {
        auto elementsCount = lenght / channels_;
        
        switch (type_) {
            case NO_NORM:
                for (auto i = 0; i < elementsCount; i++) {
                    auto index = i * channels_;

                    copy(&src[index], &dst[index]);
                }
                
                break;
                
            case MINMAX_BY_COLUMNS:
                for (auto i = 0; i < channels_; i++) {
                    float max_ = FLT_MIN;
                    float min_ = FLT_MAX;
                    
                    for (auto j = 0; j < elementsCount; j++) {
                        cl_float value = (cl_float)src[j * channels_ + i];
                        
                        max_ = max(value, max_);
                        min_ = min(value, min_);
                    }
                    
                    derComponents_[i] = 1.0 / (max_ - min_);
                    sumComponents_[i] = -min_ / (max_ - min_);
                }
                
                for (auto i = 0; i < elementsCount; i++) {
                    auto index = i * channels_;
                    
                    for (auto j = 0; j < channels_; j++) {
                        dst[index + j] = derComponents_[j] * (cl_float)src[index + j] + sumComponents_[j];
                    }
                }
                break;
                
            case MINMAX_BY_ROWS:
                for (auto i = 0; i < elementsCount; i++) {
                    auto index = i * channels_;
                    auto sum = 0.0;
                    
                    for (int j = 0; j < channels_; j++) {
                        auto value = (cl_float)src[index + j];
                        sum += value * value;
                    }
                    
                    float invLenght = 1.0 / sqrt(sum);
                    
                    for (int j = 0; j < channels_; j++) {
                        dst[index + j] = (cl_float)src[index + j] * invLenght;
                    }
                }
                
                break;
        }
        
        return *dst;
    }
    
    template <typename T> cl_float & Normalizer::normalizeVector(const T *src, cl_float &dst) {
        switch (type_) {
            case NO_NORM:
                copy(src, &dst);
                break;
                
            case MINMAX_BY_COLUMNS:
                minmaxByCols(src, &dst);
                break;
                
            case MINMAX_BY_ROWS:
                minmaxByRows(src, &dst);
                break;
        }
        
        return dst;
    }
    
    template <typename T> void Normalizer::copy(const T *src, cl_float *dst) {
        for (auto i = 0; i < channels_; i++) {
            dst[i] = (cl_float)src[i];
        }
    }
    
    template <typename T> void Normalizer::minmaxByRows(const T *src, cl_float *dst) {
        auto sum = 0.0;
        
        for (auto i = 0; i < channels_; i++) {
            sum += src[i] * src[i];
        }
        
        auto invLenght = 1.0 / sqrt(sum);
        
        for (auto i = 0; i < channels_; i++) {
            dst[i] = src[i] * invLenght;
        }
    }
    
    template <typename T> void Normalizer::minmaxByCols(const T *src, cl_float *dst) {
        for (auto i = 0; i < channels_; i++) {
            cl_float value = derComponents_[i] * (cl_float)src[i] + sumComponents_[i];
            
            value = fmax(0.0, value);
            value = fmin(1.0, value);
            
            dst[i] = value;
        }
    }
    
}

#endif /* normalizer_hpp */

