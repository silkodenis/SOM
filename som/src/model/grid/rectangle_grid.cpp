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

#include "rectangle_grid.hpp"

using namespace som;

namespace som {
    static const Orientation POINTY_ORIENTATION(sqrt(3.0), sqrt(3.0) / 2.0, 0.0, 3.0 / 2.0,
                                                sqrt(3.0) / 3.0, -1.0 / 3.0, 0.0, 2.0 / 3.0);
}

RectangleGrid::RectangleGrid(const size_t cols, const size_t rows, const double hexSize) : Grid(POINTY_ORIENTATION, hexSize) {
    cols_ = cols;
    rows_ = rows;
    
    topologicalDimensionality_ = 2;
    nodesCount_ = cols_ * rows_;
    
    points_ = (cl_float *)malloc(sizeof(cl_float) * nodesCount_ * topologicalDimensionality_);
    corners_ = (cl_float *)malloc(sizeof(cl_float) * nodesCount_ * HEXAGON_CORNERS_COUNT * topologicalDimensionality_);
    
    hexSize_ /= 2;
    double hexHeight = hexSize_ * 2;
    double hexWidth = sqrt(3) / 2 * hexHeight;
    double verticalDistance = hexHeight * 3./4.;
    
    size_.width = cols * hexWidth + offset_.horizontal * 2;
    
    if (rows > 1) {
        size_.width += hexWidth / 2;
    }
    
    size_.height = (rows - 1) * verticalDistance + hexHeight + offset_.vertical * 2;
    
    Point origin(hexWidth/2 + offset_.horizontal, hexSize_ + offset_.vertical);
    Layout layout(orientation_, origin, hexSize_);
    
    size_t pointIndex = 0;
    size_t cornerIndex = 0;
    for (auto i = 0; i < cols; i++) {
        for (auto j = 0; j < rows; j++, pointIndex += 2, cornerIndex += HEXAGON_CORNERS_COUNT * topologicalDimensionality_) {
            Hex hex = offsetToHex(ODD, OffsetCoord(i, j));
            
            Point point = hexToPoint(layout, hex);
            points_[pointIndex]     = point.x;
            points_[pointIndex + 1] = point.y;
            
            for (auto c = 0; c < HEXAGON_CORNERS_COUNT; c++) {
                size_t cornerPointIndex0 = cornerIndex + (c * topologicalDimensionality_);
                
                auto angleDeg = 60 * c + 30;
                auto angleRad = M_PI / 180 * angleDeg;
                
                corners_[cornerPointIndex0] = point.x + hexSize_ * cos(angleRad);
                corners_[cornerPointIndex0 + 1] = point.y + hexSize_ * sin(angleRad);
            }
        }
    }
    
    topologicalRadius_ = fmax(size_.width - offset_.horizontal * 2, size_.height - offset_.vertical * 2) / 2;
}

size_t RectangleGrid::getCols() {
    return cols_;
}

size_t RectangleGrid::getRows() {
    return rows_;
}
