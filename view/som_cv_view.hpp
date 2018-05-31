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
#include "som.hpp"

namespace som {
    
    static const cv::Scalar BLACK_COLOR(0, 0, 0);
    static const cv::Scalar WHITE_COLOR(255, 255, 255);
    static const cv::Scalar GREEN_COLOR(0, 255, 0);
    static const cv::Scalar RED_COLOR(0, 0, 255);
    static const cv::Scalar CORAL_COLOR(80, 127, 255);
    static const cv::Scalar BLUE_COLOR(255, 0, 0);
    static const cv::Scalar YELLOW_COLOR(0, 255, 255);
    
    enum GradientColors {
        DARK_ORANGE_TO_BLUE = 0,
        SKY_BLUE_TO_PINK,
        SKY_BLUE_TO_YELLOW,
        SKY_BLUE_TO_RED,
        BLUE_TO_YELLOW,
        GREEN_TO_PINK,
        BLACK_TO_WHITE,
        HUE
    };
    
    void drawCell(cv::Mat &dst, const Cell &cell, const cv::Scalar color, const cv::LineTypes lineType = cv::LINE_8);
    void drawGrid(cv::Mat &dst, const Cell &cell, const cv::Scalar color);
    
    extern cv::Mat draw3DMap(const SOM &som,
                             const bool grid = false,
                             const bool onlyActive = false,
                             const cv::Scalar backgroundColor = WHITE_COLOR);
    
    extern cv::Mat draw1DMap(const SOM &som,
                             const GradientColors colors = SKY_BLUE_TO_PINK,
                             const bool grid = false,
                             const bool onlyActive = false,
                             const cv::Scalar backgroundColor = WHITE_COLOR);
    
    extern cv::Mat drawSingleChannelMap(const SOM &som,
                                        const size_t channel,
                                        const GradientColors colors = SKY_BLUE_TO_PINK,
                                        const bool grid = false,
                                        const bool onlyActive = false,
                                        const cv::Scalar backgroundColor = WHITE_COLOR);
    
    extern cv::Mat drawDistancesMap(const SOM &som,
                                    const GradientColors colors = SKY_BLUE_TO_PINK,
                                    const bool grid = false,
                                    const cv::Scalar backgroundColor = WHITE_COLOR);
    
    extern cv::Mat drawApproximationMap(const SOM &som,
                                        const GradientColors colors = HUE,
                                        const bool grid = false,
                                        const cv::Scalar backgroundColor = WHITE_COLOR);
    
}

#endif /* som_cv_draw_hpp */
