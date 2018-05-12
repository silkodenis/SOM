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

#include "hexagon_grid.hpp"

using namespace som;

namespace som {
    static const Orientation FLAT_ORIENTATION(3.0 / 2.0, 0.0, sqrt(3.0) / 2.0, sqrt(3.0),
                                              2.0 / 3.0, 0.0, -1.0 / 3.0, sqrt(3.0) / 3.0);
}

HexagonGrid::HexagonGrid(const size_t radius, const double hexSize) : Grid(FLAT_ORIENTATION, hexSize) {
    radius_ = (int)radius;
    
    hexSize_ /= 2;
    topologicalDimensionality_ = 2;

    double hexWidth = hexSize_ * 2;
    double hexHeight = sqrt(3)/2 * hexWidth;
    double horizontalDistance = hexWidth * 3./4.;
    double verticalDistance = hexHeight;

    size_.width = radius * 2 * horizontalDistance + hexWidth + offset_.horizontal * 2;
    size_.height = radius * 2 * hexHeight + verticalDistance + offset_.vertical * 2;
    
    som::Point origin(size_.width/2, size_.height/2);
    Layout layout(orientation_, origin, hexSize_);
    
    vector<Hex> hexagons;
    for (int q = -radius_; q <= radius_; q++) {
        int r1 = max(-radius_, -q - radius_);
        int r2 = min(radius_, -q + radius_);
        
        for (int r = r1; r <= r2; r++) {
            hexagons.push_back(Hex(q, r));
        }
    }
    
    nodesCount_ = hexagons.size();
    
    points_ = (cl_float *)malloc(sizeof(cl_float) * nodesCount_ * topologicalDimensionality_);
    corners_ = (cl_float *)malloc(sizeof(cl_float) * nodesCount_ * HEXAGON_CORNERS_COUNT * topologicalDimensionality_);
    
    size_t pointIndex = 0;
    size_t cornerIndex = 0;
    for (auto i = 0; i < nodesCount_; i++, pointIndex += 2, cornerIndex += HEXAGON_CORNERS_COUNT * topologicalDimensionality_) {
        Point point = hexToPoint(layout, hexagons[i]);
        points_[pointIndex]     = point.x;
        points_[pointIndex + 1] = point.y;
        
        for (auto c = 0; c < HEXAGON_CORNERS_COUNT; c++) {
            size_t cornerPointIndex0 = cornerIndex + (c * topologicalDimensionality_);

            auto angleDeg = 60 * c;
            auto angleRad = M_PI / 180 * angleDeg;
            
            corners_[cornerPointIndex0] = point.x + hexSize_ * cos(angleRad);
            corners_[cornerPointIndex0 + 1] = point.y + hexSize_ * sin(angleRad);
        }
    }
    
    topologicalRadius_ = fmax(size_.width - offset_.horizontal * 2, size_.height - offset_.vertical * 2) / 2;
}

int HexagonGrid::getRaduis() {
    return radius_;
}
