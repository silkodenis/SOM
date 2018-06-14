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

#ifndef som_cv_draw_hpp
#define som_cv_draw_hpp

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "pseudocolor_schemes.hpp"
#include "som.hpp"

namespace som {
    
#pragma mark - Colors
    
    static const cv::Scalar COLOR_BLACK(0, 0, 0);
    static const cv::Scalar COLOR_WHITE(255, 255, 255);
    static const cv::Scalar COLOR_GREEN(0, 255, 0);
    static const cv::Scalar COLOR_RED(0, 0, 255);
    static const cv::Scalar COLOR_CORAL(80, 127, 255);
    static const cv::Scalar COLOR_BLUE(255, 0, 0);
    static const cv::Scalar COLOR_YELLOW(0, 255, 255);
    
#pragma mark - Configurations
    
    struct DisplayConfiguration {
        DisplayConfiguration(const bool grid_ = false,
                             const bool onlyActive_ = false,
                             const cv::Mat colorbar_ = cv::Mat(),
                             const cv::Scalar backgroundColor_ = COLOR_WHITE) :
        grid(grid_),
        onlyActive(onlyActive_),
        colorbar(colorbar_),
        backgroundColor(backgroundColor_) {}
        
        const bool grid;
        const bool onlyActive;
        const cv::Mat colorbar;
        const cv::Scalar backgroundColor;
    };
    
    struct ColorbarConfiguration {
        ColorbarConfiguration(const size_t lenght_, const size_t thickness_, const bool horizontal_, const bool coordinates_) :
        lenght(lenght_), thickness(thickness_), horizontal(horizontal_) , coordinates(coordinates_) {}
        
        const size_t lenght;
        const size_t thickness;
        const bool horizontal;
        const bool coordinates;
    };
    
#pragma mark - Draw functions
    
    extern void drawCell(cv::Mat &dst, const Cell &cell, const cv::Scalar color, const cv::LineTypes lineType = cv::LINE_8);
    extern void drawGrid(cv::Mat &dst, const Cell &cell, const cv::Scalar color);
    
    extern cv::Mat draw3DColorBar(const ColorbarConfiguration colorBarConfiguration);
    extern cv::Mat draw1DColorBar(const ColorbarConfiguration colorBarConfiguration, const ColormapConfiguration colormapConfiguration);
    
    extern cv::Mat draw3DMap(const SOM &som, const DisplayConfiguration = DisplayConfiguration());
    
    extern cv::Mat draw1DMap(const SOM &som,
                             const ColormapConfiguration = ColormapConfiguration(COLORSCALE_JET),
                             const DisplayConfiguration = DisplayConfiguration());
    
    extern cv::Mat drawSingleChannelMap(const SOM &som,
                                        const size_t channel,
                                        const ColormapConfiguration = ColormapConfiguration(COLORSCALE_JET),
                                        const DisplayConfiguration = DisplayConfiguration());
    
    extern cv::Mat drawDistancesMap(const SOM &som,
                                    const ColormapConfiguration = ColormapConfiguration(COLORSCALE_JET),
                                    const DisplayConfiguration = DisplayConfiguration());
    
    extern cv::Mat drawApproximationMap(const SOM &som,
                                        const ColormapConfiguration = ColormapConfiguration(COLORSCALE_JET),
                                        const DisplayConfiguration = DisplayConfiguration());
}

#endif /* som_cv_draw_hpp */
