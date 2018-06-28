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

#ifndef pseudocolor_schemes_hpp
#define pseudocolor_schemes_hpp

#include <opencv2/imgproc.hpp>

namespace som {
    
    // Color scales equivalent Matlab and Matplotlib
    enum ColorScale {
        COLORSCALE_GRAY,
        COLORSCALE_AUTUMN,
        COLORSCALE_BONE,
        COLORSCALE_JET,
        COLORSCALE_WINTER,
        COLORSCALE_RAINBOW,
        COLORSCALE_OCEAN,
        COLORSCALE_SUMMER,
        COLORSCALE_SPRING,
        COLORSCALE_COOL,
        COLORSCALE_HSV,
        COLORSCALE_PINK,
        COLORSCALE_HOT,
        COLORSCALE_PARULA,
        COLORSCALE_VIRIDIS,
        COLORSCALE_PLASMA,
        COLORSCALE_INFERNO,
        COLORSCALE_MAGMA,
        COLORSCALE_CIVIDIS,
        COLORSCALE_COOLWARM
    };
    
    struct ColormapConfiguration {
        ColormapConfiguration(const ColorScale colorscale_,
                              const size_t colors_ = 256,
                              const float cmin_ = 0.0,
                              const float cmax_ = 1.0,
                              const bool invert_ = false) :
        colorscale(colorscale_),
        colors(colors_),
        cmin(cmin_), cmax(cmax_),
        invert(invert_) {}
        
        const ColorScale colorscale;
        const size_t colors;
        const float cmin;
        const float cmax;
        bool invert;
    };
    
    cv::Mat getColormap(const ColormapConfiguration colormapConfiguration);
}

#endif /* pseudocolor_schemes_hpp */
