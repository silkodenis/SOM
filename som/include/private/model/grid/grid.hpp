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

#ifndef grid_hpp
#define grid_hpp

#include <iostream>
#include <vector>
#include <math.h>
#include "hex.hpp"
#include "som_types.hpp"

namespace som {

    using namespace std;
    
    class Grid {
        
    public:
        Grid(const Orientation orientation, const double hexSize);
        virtual ~Grid() = 0;
        
        cl_float & getPoints() const;
        cl_float & getCorners() const;
        
        Size getSize() const;
        Offset getOffset() const;
        
        double getTopologicalRadius() const;
        double getHexSize() const;
        
        size_t getTopologicalDimensionality() const;
        size_t getNodesCount() const;
        
        virtual size_t getCols();
        virtual size_t getRows();
        virtual int getRaduis();
    
    protected:
        size_t nodesCount_;
        size_t topologicalDimensionality_;
        
        cl_float *points_;
        cl_float *corners_;
        
        Size size_;
        Offset offset_;
        
        double hexSize_;
        double topologicalRadius_;
        
        const Orientation orientation_;
    };
    
}

#endif /* grid_hpp */
