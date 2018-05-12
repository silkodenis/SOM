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

#include "hex.hpp"

using namespace som;

Point som::hexToPoint(Layout layout, Hex hex) {
    const Orientation& M = layout.orientation;
    double x = (M.f0 * hex.q + M.f1 * hex.r) * layout.hexSize;
    double y = (M.f2 * hex.q + M.f3 * hex.r) * layout.hexSize;
    
    return Point(x + layout.origin.x, y + layout.origin.y);
}

Hex som::offsetToHex(int offset, OffsetCoord h) {
    int q = h.col - ((h.row + offset * (h.row & 1)) / 2);
    int r = h.row;
    
    return Hex(q, r);
}
