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

#ifndef hex_hpp
#define hex_hpp

#include "som_types.hpp"

using namespace som;

namespace som {

#pragma mark - Coordinates
    
    static const int ODD = -1;
    
    struct Point {
        const double x, y;
        Point(double x_, double y_): x(x_), y(y_) {}
    };
    
    struct Size {
        double width, height;
        Size(double width_, double height_): width(width_), height(height_) {}
    };
    
    struct Offset {
        const double vertical, horizontal;
        Offset(double vertical_, double horizontal_): vertical(vertical_), horizontal(horizontal_) {}
    };
    
    struct Hex {
        const int q, r, s;
        Hex(int q_, int r_): q(q_), r(r_), s(-q_ - r_) {}
    };
    
    struct OffsetCoord {
        const int col, row;
        OffsetCoord(int col_, int row_): col(col_), row(row_) {}
    };

#pragma mark - Layout
    
    struct Orientation {
        const double f0, f1, f2, f3;
        const double b0, b1, b2, b3;
        
        Orientation(double f0_, double f1_, double f2_, double f3_,
                    double b0_, double b1_, double b2_, double b3_) :
        f0(f0_), f1(f1_), f2(f2_), f3(f3_),
        b0(b0_), b1(b1_), b2(b2_), b3(b3_) {}
    };
    
    struct Layout {
        const Orientation orientation;
        const Point origin;
        const double hexSize;
        
        Layout(Orientation orientation_, Point origin_, double hexSize_)
        : orientation(orientation_), origin(origin_), hexSize(hexSize_) {}
    };
    
#pragma mark - Convertation
    
    extern Point hexToPoint(Layout layout, Hex hex);
    extern Hex offsetToHex(int offset, OffsetCoord h);
}

#endif /* hex_hpp */
