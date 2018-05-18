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

#include "grid.hpp"

using namespace std;
using namespace som;

namespace som {
    static const size_t DEFAULT_VERTICAL_OFFSET = 5;
    static const size_t DEFAULT_HORIZONTAL_OFFSET = 5;
}

Grid::Grid(const Orientation orientation, const double hexSize) :
nodesCount_(0),
points_(nullptr),
corners_(nullptr),
size_(0, 0),
offset_(DEFAULT_VERTICAL_OFFSET, DEFAULT_HORIZONTAL_OFFSET),
hexSize_(hexSize),
orientation_(orientation)
{}

Grid::~Grid() {
    if (points_) {
        free(points_);
    }
    
    if (corners_) {
        free(corners_);
    }
}

#pragma mark - getters

size_t Grid::getCols() { return 0; }
size_t Grid::getRows() { return 0; }
int Grid::getRaduis() { return 0; }

cl_float & Grid::getCorners() const { return *corners_; }
cl_float & Grid::getPoints() const { return *points_; }

size_t Grid::getTopologicalDimensionality() const { return topologicalDimensionality_; }
size_t Grid::getNodesCount() const { return nodesCount_; }
double Grid::getHexSize() const { return hexSize_; }
double Grid::getTopologicalRadius() const { return topologicalRadius_; };

Size Grid::getSize() const { return size_; }
Offset Grid::getOffset() const { return offset_; }

