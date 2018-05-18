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

#include "model.hpp"
#include <assert.h>
#include <cstring>
#include "normalizer.hpp"
#include "hexagon_grid.hpp"
#include "rectangle_grid.hpp"

using namespace som;

namespace som {
    static const double DEFAULT_MIN_WEIGHT_VALUE = 0.0;
    static const double DEFAULT_MAX_WEIGHT_VALUE = 1.0;
}

Model::Model() :
grid_(nullptr),
normalizer_(nullptr),
input_(nullptr),
data_(nullptr),
labels_(nullptr),
weights_(nullptr),
distances_(nullptr),
distancesAccumulator_(nullptr),
activationStates_(nullptr) {
    uint64_t timeSeed = chrono::high_resolution_clock::now().time_since_epoch().count();
    seed_seq seq{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    rng_.seed(seq);
}

Model::Model(const size_t cols, const size_t rows, const size_t channels, const double hexSize) : Model() {
    create(cols, rows, channels, hexSize);
}

Model::Model(const size_t radius, const size_t channels, const double hexSize) : Model() {
    create(radius, channels, hexSize);
}

Model::~Model() {
    delete grid_;
    delete normalizer_;
    
    if (input_) { free(input_); }
    
    if (data_) { free(data_); }
    if (labels_) { free(labels_); }
    if (weights_) { free(weights_); }
    if (distances_) { free(distances_); }
    if (distancesAccumulator_) { free(distancesAccumulator_); }
    if (activationStates_) { free(activationStates_); }
}

bool Model::create(const size_t cols, const size_t rows, const size_t channels, const double hexHeight) {
    auto canCreate = !grid_ && !weights_ && !activationStates_;
    
    if (!canCreate) {
        return false;
    }
    
    grid_ = new RectangleGrid(cols, rows, hexHeight);
    
    nodesCount_ = grid_->getNodesCount();
    channelsCount_ = channels;
    
    return create();
}

bool Model::create(const size_t radius, const size_t channels, const double hexSize) {
    auto canCreate = !grid_ && !weights_ && !activationStates_;
    
    if (!canCreate) {
        return false;
    }
    
    grid_ = new HexagonGrid(radius, hexSize);
    
    nodesCount_ = grid_->getNodesCount();
    channelsCount_ = channels;
    
    return create();
}

bool Model::create() {
    auto canCreate = !weights_;
    
    if (!canCreate) {
        return false;
    }
    
    input_ = (cl_float *)malloc(sizeof(cl_float) * channelsCount_);
    normalizer_ = new Normalizer(channelsCount_);
    
    labels_ = (cl_int *)malloc(sizeof(cl_int) * nodesCount_);
    weights_ = (cl_float *)malloc(sizeof(cl_float) * nodesCount_ * channelsCount_);
    distances_ = (cl_float *)malloc(sizeof(cl_float) * nodesCount_);
    distancesAccumulator_ = (cl_float *)malloc(sizeof(cl_float) * nodesCount_);
    activationStates_ = (cl_int *)malloc(sizeof(cl_int) * nodesCount_);
    
    memset(input_, 0, sizeof(cl_float) * channelsCount_);
    memset(labels_, 0, sizeof(cl_int) * nodesCount_);
    memset(distances_, 0, sizeof(cl_float) * nodesCount_);
    memset(distancesAccumulator_, 0, sizeof(cl_float) * nodesCount_);
    memset(activationStates_, 0, sizeof(cl_int) * nodesCount_);
    
    setRandomWeights(DEFAULT_MIN_WEIGHT_VALUE, DEFAULT_MAX_WEIGHT_VALUE);
    
    cl_float *points = &grid_->getPoints();
    cl_float *corners = &grid_->getCorners();
    size_t topologicalDimensionality = grid_->getTopologicalDimensionality();
    
    for (auto i = 0; i < nodesCount_; i++) {
        Cell cell(points[i * topologicalDimensionality],
                  corners[i * HEXAGON_CORNERS_COUNT * topologicalDimensionality],
                  weights_[i * channelsCount_],
                  distancesAccumulator_[i],
                  labels_[i],
                  activationStates_[i]);
        
        cells_.push_back(cell);
    }
    
    return true;
}

bool Model::load(const string &filePath) {
    auto canLoad = !grid_ && !weights_ && !activationStates_;
    
    if (!canLoad) {
        return false;
    }
    
    ifstream is(filePath.c_str(), ios::binary | ios::in);
    
    if (!is.is_open()) {
        return false;
    }
    
    double hexSize;
    size_t cols, rows;
    int radius;
    
    is.read((char *)&cols, sizeof(size_t));
    is.read((char *)&rows, sizeof(size_t));
    is.read((char *)&radius, sizeof(int));
    is.read((char *)&hexSize, sizeof(double));
    is.read((char *)&channelsCount_, sizeof(size_t));
    is.read((char *)&nodesCount_, sizeof(size_t));
    
    hexSize *= 2;
    
    if (radius > 0) {
        create(radius, channelsCount_, hexSize);
    } else {
        create(cols, rows, channelsCount_, hexSize);
    }
    
    normalizer_->load(is);
    
    is.read((char *)&metric_, sizeof(Metric));
    is.read((char *)weights_, streamsize(nodesCount_ * channelsCount_ * sizeof(cl_float)));
    is.read((char *)distancesAccumulator_, streamsize(nodesCount_ * sizeof(cl_float)));
    is.read((char *)activationStates_, streamsize(nodesCount_ * sizeof(cl_int)));
    is.read((char *)labels_, streamsize(nodesCount_ * sizeof(cl_int)));
    
    is.close();
    
    return true;
}

bool Model::save(const string &filePath) {
    auto canSave = grid_ && weights_ && activationStates_;
    
    if (!canSave) {
        return false;
    }
    
    ofstream os(filePath, fstream::binary | ios::out);
    
    if (!os.is_open()) {
        return false;
    }
    
    auto hexSize = grid_->getHexSize();
    auto cols = grid_->getCols();
    auto rows = grid_->getRows();
    auto radius = grid_->getRaduis();

    os.write((char *)&cols, sizeof(size_t));
    os.write((char *)&rows, sizeof(size_t));
    os.write((char *)&radius, sizeof(int));
    os.write((char *)&hexSize, sizeof(double));
    os.write((char *)&channelsCount_, sizeof(size_t));
    os.write((char *)&nodesCount_, sizeof(size_t));
    
    normalizer_->save(os);
    
    os.write((char *)&metric_, sizeof(Metric));
    os.write((char *)weights_, streamsize(nodesCount_ * channelsCount_ * sizeof(cl_float)));
    os.write((char *)distancesAccumulator_, streamsize(nodesCount_ * sizeof(cl_float)));
    os.write((char *)activationStates_, streamsize(nodesCount_ * sizeof(cl_int)));
    os.write((char *)labels_, streamsize(nodesCount_ * sizeof(cl_int)));
    
    os.close();
    
    return true;
}

#pragma mark - Prepare

void Model::prepare(const vector<vector<cl_float>> &data, const Normalization normalizationType, const Weights initialWeights) {
    if (data_) {
        free(data_);
        data_ = nullptr;
    }
    
    dataCount_ = data.size();
    auto lenght = dataCount_ * channelsCount_;
    
    data_ = (cl_float *)malloc(sizeof(cl_float) * lenght);
    data_ = &normalizer_->normalize(data, data_, normalizationType);
    
    uniform_.reset();
    uniform_ = uniform_real_distribution<double>(0, data.size() - 1);
    
    setWeights(initialWeights);
}

void Model::prepare(const uint8_t *pixelBuffer, const size_t lenght, const Normalization normalizationType, const Weights initialWeights) {
    if (data_) {
        free(data_);
        data_ = nullptr;
    }
    
    dataCount_ = lenght / channelsCount_;
    
    data_ = (cl_float *)malloc(sizeof(cl_float) * lenght);
    data_ = &normalizer_->normalize(pixelBuffer, lenght, data_, normalizationType);
    
    uniform_.reset();
    uniform_ = uniform_real_distribution<double>(0, dataCount_ - 1);
    
    setWeights(initialWeights);
}

void Model::setWeights(const Weights initialWeights) {
    switch (initialWeights) {
        case RANDOM_01:
            setRandomWeights(0, 1);
            break;
            
        case RANDOM_FROM_DATA:
            for (auto i = 0; i < nodesCount_; i++) {
                cl_float &randomVector = getRandomDataVector();
                
                auto index = i * channelsCount_;
                
                memcpy(&weights_[index], &randomVector, sizeof(cl_float) * channelsCount_);
            }
            break;
    }
}

#pragma mark - Normalize input vector

cl_float & Model::normalizeVector(const vector<cl_float> &vector) {
    return normalizer_->normalize(vector, *input_);
}

cl_float & Model::normalizeVector(const uint8_t *vector) {
    return normalizer_->normalize(vector, *input_);
}

#pragma mark - setters

void Model::setRandomWeights(const double min, const double max) {
    if (weights_) {
        uniform_real_distribution<cl_float> unif(min, max);
        
        auto weightsLenght = nodesCount_ * channelsCount_;
        for (auto i = 0; i < weightsLenght; i++) {
            weights_[i] = unif(rng_);
        }
    }
}

void Model::setLabel(cl_int label, size_t index) {
    assert(index < nodesCount_);
    
    labels_[index] = label;
}

void Model::setLabels(vector<cl_int> labels, vector<size_t> indices) {
    assert(labels.size() == indices.size() && indices.size() <= nodesCount_);
    
    for (auto i = 0; i < indices.size(); i++) {
        labels_[i] = labels[i];
    }
}

void Model::setMetric(Metric metric) {
    metric_ = metric;
}

#pragma mark - getters

cl_float & Model::getRandomDataVector() {
    size_t randomIndex = uniform_(rng_);
    
    return data_[randomIndex * channelsCount_];
}

vector<Cell> Model::getCells() const { return cells_; }

double Model::getWidth() const { return grid_->getSize().width; }
double Model::getHeight() const { return grid_->getSize().height; }

double Model::getTopologicalRadius() const { return grid_->getTopologicalRadius(); }
Metric Model::getMetric() const { return metric_; }
size_t Model::getDataCount() const { return dataCount_; }
size_t Model::getNodesCount() const { return nodesCount_; }
size_t Model::getChannelsCount() const { return channelsCount_; }
size_t Model::getTopologicalDimensionality() const { return grid_->getTopologicalDimensionality(); }

cl_float & Model::getPoints() const { return grid_->getPoints(); }
cl_float & Model::getData() const { return *data_; }
cl_int & Model::getLabels() const { return *labels_; };
cl_float & Model::getWeights() const { return *weights_; }
cl_float & Model::getDistances() const { return *distances_; }
cl_float & Model::getDistancesAccumulator() const { return *distancesAccumulator_; }
cl_int & Model::getActivationStates() const { return *activationStates_; }
