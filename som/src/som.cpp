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

#include "som.hpp"
#include <iomanip>
#include <assert.h>
#include "model.hpp"
#include "trainer.hpp"
#include "computing.hpp"

using namespace std;
using namespace som;

SOM::SOM(const Device deviceType) :
deviceType_(deviceType),
model_(nullptr),
trainer_(nullptr),
computing_(nullptr) {}

SOM::~SOM() {
    release();
}

#pragma mark - Release memory

void SOM::release() {
    delete trainer_;
    trainer_ = nullptr;
    
    delete computing_;
    computing_ = nullptr;
    
    delete model_;
    model_ = nullptr;
}

#pragma mark - load/save

bool SOM::load(const string &filePath) {
    bool canLoad = !trainer_ && !model_ && !computing_;
    
    if (!canLoad) {
        return false;
    }
    
    model_ = new Model();
    
    if (!model_->load(filePath)) {
        return false;
    }
    
    computing_ = new Computing(*model_, deviceType_);
    trainer_ = new Trainer(*model_, *computing_);
    
    return true;
}

bool SOM::save(const string &filePath) {
    bool canSave = model_;
    
    if (!canSave) {
        return false;
    }
    
    return model_->save(filePath);
}

#pragma mark - Create

bool SOM::create(const size_t cols, const size_t rows, const size_t hexSize, const size_t channels) {
    assert(cols > 0 && rows > 0 && hexSize > 0 && channels > 0);
    
    bool canCreate = !trainer_ && !model_ && !computing_;
    
    if (!canCreate) {
        return false;
    }
    
    model_ = new Model(cols, rows, channels, hexSize);
    computing_ = new Computing(*model_, deviceType_);
    trainer_ = new Trainer(*model_, *computing_);
    
    return true;
}

bool SOM::create(const size_t radius, const size_t hexSize, const size_t channels) {
    assert(radius > 0 && hexSize > 0 && channels > 0);
    
    bool canCreate = !trainer_ && !model_ && !computing_;
    
    if (!canCreate) {
        return false;
    }
    
    model_ = new Model(radius, channels, hexSize);
    computing_ = new Computing(*model_, deviceType_);
    trainer_ = new Trainer(*model_, *computing_);
    
    return true;
}

#pragma mark - Prepare

void SOM::prepare(const vector<vector<float>> &data, const Normalization normalization, const Weights initialWeights) {
    assert(model_ && data.size() > 0 && data[0].size() == model_->getChannelsCount());
    
    model_->prepare(data, normalization, initialWeights);
}

void SOM::prepare(const uint8_t *pixelBuffer, const size_t lenght, const Normalization normalization, const Weights initialWeights) {
    assert(model_ && pixelBuffer && lenght >= model_->getChannelsCount());

    model_->prepare(pixelBuffer, lenght, normalization, initialWeights);
}

void SOM::setRandomWeights(const float min, const float max) {
    if (model_) {
        model_->setRandomWeights(min, max);
    }
}

#pragma mark - Training

void SOM::train(const size_t iterationsCount, const double learningRate, const Metric metric, bool epochMode) {
    assert(model_ && trainer_);
    
    clock_t start = clock();
    
    model_->setMetric(metric);
    trainer_->learn(iterationsCount, learningRate, epochMode);
    
    cout << "SOM: Train duration: " << setprecision(4) << (clock() - start) / (double)CLOCKS_PER_SEC << endl;
}

bool SOM::epochs(size_t count) {
    if (trainer_) {
        for (size_t i = 0; i < count; i++) {
            if (trainer_->epoch()) {
                return true;
            }
        }
    }
    
    return false;
}

#pragma mark - Use

void SOM::setLabel(int label, size_t index) {
    if (model_) {
        model_->setLabel(label, index);
    }
}

void SOM::setLabels(vector<int> labels, vector<size_t> indices) {
    if (model_) {
        model_->setLabels(labels, indices);
    }
}

int SOM::predict(const vector<float> &vector) const {
    assert(computing_ && model_ && (vector.size() == model_->getChannelsCount()));
    
    cl_float &input = model_->normalizeVector(vector);
    size_t bmuIndex = computing_->bmuIndex(input, false);
    cl_int *labels = &model_->getLabels();
    
    return labels[bmuIndex];
}

int SOM::predict(const uint8_t &pixel) const {
    assert(computing_ && model_);
    
    cl_float &input = model_->normalizeVector(&pixel);
    size_t bmuIndex = computing_->bmuIndex(input, false);
    cl_int *labels = &model_->getLabels();
    
    return labels[bmuIndex];
}

#pragma mark - BMU

size_t SOM::computeBmuIndex(const vector<float> &vector) const {
    assert(computing_ && model_ && vector.size() == model_->getChannelsCount());
    
    cl_float &input = model_->normalizeVector(vector);
    
    return computing_->bmuIndex(input, false);
}

size_t SOM::computeBmuIndex(const uint8_t &pixel) const {
    assert(computing_ && model_);
    
    cl_float &input = model_->normalizeVector(&pixel);
    
    return computing_->bmuIndex(input, false);
}

#pragma mark - Error

double SOM::computeError() {
    assert(computing_);
    
    return computing_->error();
}

#pragma mark - ModelView

vector<Cell> SOM::getCells() const { return model_->getCells(); }

double SOM::getWidth() const { return model_->getWidth(); }
double SOM::getHeight() const { return model_->getHeight(); }

size_t SOM::getNodeDimensionality() const { return model_->getChannelsCount(); }
size_t SOM::getTopologicalDimensionality() const { return model_->getTopologicalDimensionality(); }

