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

#include "som_cv_view.hpp"
#include "som_types.hpp"
#include <assert.h>
#include <float.h>
#include <iomanip>
#include <sstream>

using namespace std;
using namespace cv;
using namespace som;

namespace som {
    static const Scalar BACKGROUND_COLOR(WHITE_COLOR);
    static const Scalar GRID_COLOR(BLACK_COLOR);
    static const Scalar CIRCLE_COLOR(RED_COLOR);
    
    extern Scalar computeCellColor(const float value, const GradientMap gradientMap);
    
    extern void minmaxByRows(vector<vector<float>> &dst);
    extern void minmaxByCols(vector<vector<float>> &dst);
    extern vector<vector<float>> reduceDimensionality(const SOM &som, const size_t dimensionality, const bool gradient);
    extern vector<float> reduceVector(const Cell &cell, const size_t vectorDimensionality, const size_t targetDimensionality, const bool gradient);
}

void som::drawCell(cv::Mat &dst, const Cell &cell, const Scalar color, const LineTypes lineType) {
    cv::Point points[HEXAGON_CORNERS_COUNT] = {
        cv::Point2f(cell.corners[0], cell.corners[1]),
        cv::Point2f(cell.corners[2], cell.corners[3]),
        cv::Point2f(cell.corners[4], cell.corners[5]),
        cv::Point2f(cell.corners[6], cell.corners[7]),
        cv::Point2f(cell.corners[8], cell.corners[9]),
        cv::Point2f(cell.corners[10], cell.corners[11])
    };
    
    const cv::Point *elementPoints[1] = {points};
    
    cv::fillPoly(dst, elementPoints, &HEXAGON_CORNERS_COUNT, 1, color, lineType);
}

void som::drawGrid(cv::Mat &dst, const Cell &cell, const cv::Scalar color) {
    line(dst, cv::Point2f(cell.corners[0], cell.corners[1]), cv::Point2f(cell.corners[2], cell.corners[3]), color);
    line(dst, cv::Point2f(cell.corners[2], cell.corners[3]), cv::Point2f(cell.corners[4], cell.corners[5]), color);
    line(dst, cv::Point2f(cell.corners[4], cell.corners[5]), cv::Point2f(cell.corners[6], cell.corners[7]), color);
    line(dst, cv::Point2f(cell.corners[6], cell.corners[7]), cv::Point2f(cell.corners[8], cell.corners[9]), color);
    line(dst, cv::Point2f(cell.corners[8], cell.corners[9]), cv::Point2f(cell.corners[10], cell.corners[11]), color);
    line(dst, cv::Point2f(cell.corners[10], cell.corners[11]), cv::Point2f(cell.corners[0], cell.corners[1]), color);
}

cv::Mat som::drawSingleChannelMap(const SOM& som,
                                  const size_t channel,
                                  const GradientMap colors,
                                  const bool grid,
                                  const bool onlyActive,
                                  const Scalar backgroundColor) {
    Mat result(som.getHeight(), som.getWidth(), CV_8UC3, backgroundColor);

    auto cells = som.getCells();

    vector<vector<float>> values;
    for (size_t i = 0; i < cells.size(); i++) {
        values.push_back({cells[i].weights[channel]});
    }

    minmaxByCols(values);

    for (auto i = 0; i < cells.size(); i++) {
        bool canDrawNode = (onlyActive && cells[i].state[0]) || !onlyActive;

        if (canDrawNode) {
            float value = values[i][0] * 255;
            
            Scalar nodeColor = computeCellColor(value, colors);

            drawCell(result, cells[i], nodeColor, grid ? LINE_8 : LINE_AA);
        }

        if (grid) {
            drawGrid(result, cells[i], GRID_COLOR);
        }
    }

    return result;
}

cv::Mat som::draw3DMap(const SOM& som, const bool grid, const bool onlyActive,
                       const Scalar backgroundColor) {
    Mat result(som.getHeight(), som.getWidth(), CV_8UC3, backgroundColor);
    
    auto cells = som.getCells();
    auto channels = som.getNodeDimensionality();
    
    vector<vector<float>> weights;
    
    if (channels > 2) {
        weights = reduceDimensionality(som, 3, false);
    } else {
        weights = reduceDimensionality(som, channels, false);
    }
    
    // draw map
    for (auto i = 0; i < cells.size(); i++) {
        bool canDrawNode = (onlyActive && cells[i].state[0] > 0) || !onlyActive;
        
        if (canDrawNode) {
            Scalar nodeColor;
            
            if (channels > 2) {
                nodeColor = Scalar(weights[i][0] * 255, weights[i][1] * 255, weights[i][2] * 255);
            } else if (channels > 1) {
                nodeColor = Scalar(weights[i][0] * 255, weights[i][1] * 255, 0);
            } else {
                nodeColor = Scalar(weights[i][0] * 255, 0, 0);
            }
            
            drawCell(result, cells[i], nodeColor, grid ? LINE_8 : LINE_AA);
        }
        
        if (grid) {
            drawGrid(result, cells[i], GRID_COLOR);
        }
    }
    
    return result;
}

cv::Mat som::drawApproximationMap(const SOM &som, const GradientMap colors, const bool grid, const Scalar backgroundColor) {
    Mat result(som.getHeight(), som.getWidth(), CV_8UC3, backgroundColor);

    auto cells = som.getCells();

    vector<vector<float>> values;
    for (size_t i = 0; i < cells.size(); i++) {
        values.push_back({(float)cells[i].state[0]});
    }

    minmaxByCols(values);

    for (size_t i = 0; i < cells.size(); i++) {
        
        float value = values[i][0] * 255;
        
        Scalar nodeColor = computeCellColor(value, colors);

        drawCell(result, cells[i], nodeColor, grid ? LINE_8 : LINE_AA);

        if (grid) {
            drawGrid(result, cells[i], GRID_COLOR);
        }
    }

    return result;
}

extern cv::Mat som::drawDistancesMap(const SOM &som,
                                     const GradientMap colors,
                                     const bool grid,
                                     const Scalar backgroundColor) {
    Mat result(som.getHeight(), som.getWidth(), CV_8UC3, backgroundColor);

    auto cells = som.getCells();

    vector<vector<float>> values;
    for (size_t i = 0; i < cells.size(); i++) {
        values.push_back({(float)cells[i].distance[0]});
    }

    minmaxByCols(values);

    for (size_t i = 0; i < cells.size(); i++) {
    
        float value = 255 - values[i][0] * 255;
        
        Scalar nodeColor = computeCellColor(value, colors);

        drawCell(result, cells[i], nodeColor, grid ? LINE_8 : LINE_AA);

        if (grid) {
            drawGrid(result, cells[i], GRID_COLOR);
        }
    }

    return result;
}

extern cv::Mat som::draw1DMap(const SOM &som,
                              const GradientMap colors,
                              const bool grid,
                              const bool onlyActive,
                              const Scalar backgroundColor) {
    Mat result(som.getHeight(), som.getWidth(), CV_8UC3, backgroundColor);

    auto cells = som.getCells();

    vector<vector<float>> weights = reduceDimensionality(som, 1, false);

    for (size_t i = 0; i < cells.size(); i++) {
        bool canDrawNode = (onlyActive && cells[i].state[0] > 0) || !onlyActive;

        if (canDrawNode) {
            
            float value = weights[i][0] * 255;
            
            Scalar nodeColor = computeCellColor(value, colors);

            drawCell(result, cells[i], nodeColor, grid ? LINE_8 : LINE_AA);
        }

        if (grid) {
            drawGrid(result, cells[i], GRID_COLOR);
        }
    }

    return result;
}

#pragma mark - Colors

Scalar computeColor(const float value, const ColormapTypes colormap) {
    Mat3b valuePixel(cv::Vec3b(value, value, value));
    Mat3b colorPixel;
    applyColorMap(valuePixel, colorPixel, colormap);
    
    return Scalar(colorPixel(0));
}

Scalar som::computeCellColor(const float value, const GradientMap gradientMap) {
    Scalar color;
    
    switch (gradientMap) {
        case GRADIENT_SKY_BLUE_TO_PINK: color = Scalar(255, 255 - value, value); break;
        case GRADIENT_SKY_BLUE_TO_YELLOW: color = Scalar(value, 255, 255 - value); break;
        case GRADIENT_SKY_BLUE_TO_RED: color = Scalar(value, value, 255 - value); break;
        case GRADIENT_DARK_ORANGE_TO_BLUE: color = Scalar(value, 0, 255 - value); break;
        case GRADIENT_GREEN_TO_PINK: color = Scalar(255 - value, value, 255 - value); break;
        case GRADIENT_BLUE_TO_YELLOW: color = Scalar(value, 255 - value, 255 - value); break;
        case GRADIENT_BLACK_TO_WHITE: color = Scalar(value, value, value); break;
        case GRADIENT_HUE: {
            Mat3b bgr;
            Mat3b hsv(cv::Vec3b(value, 255, 255));
            cvtColor(hsv, bgr, CV_HSV2RGB);
            
            color = Scalar(bgr(0));
        }
            break;
            
        case GRADIENT_AUTUMN: color = computeColor(value, COLORMAP_AUTUMN); break;
        case GRADIENT_BONE: color = computeColor(value, COLORMAP_BONE); break;
        case GRADIENT_JET: color = computeColor(value, COLORMAP_JET); break;
        case GRADIENT_WINTER: color = computeColor(value, COLORMAP_WINTER); break;
        case GRADIENT_RAINBOW: color = computeColor(value, COLORMAP_RAINBOW); break;
        case GRADIENT_OCEAN: color = computeColor(value, COLORMAP_OCEAN); break;
        case GRADIENT_SUMMER: color = computeColor(value, COLORMAP_SUMMER); break;
        case GRADIENT_SPRING: color = computeColor(value, COLORMAP_SPRING); break;
        case GRADIENT_COOL: color = computeColor(value, COLORMAP_COOL); break;
        case GRADIENT_HSV: color = computeColor(value, COLORMAP_HSV); break;
        case GRADIENT_PINK: color = computeColor(value, COLORMAP_PINK); break;
        case GRADIENT_HOT: color = computeColor(value, COLORMAP_HOT); break;
        case GRADIENT_PARULA: color = computeColor(value, COLORMAP_PARULA); break;
    }
    
    return color;
}

#pragma mark - Normalization

void som::minmaxByRows(vector<vector<float>> &dst) {
    auto channels = dst[0].size();
    
    for (auto i = 0; i < dst.size(); i++) {
        auto sum = 0.0;
        
        for (int j = 0; j < channels; j++) {
            auto value = dst[i][j];
            sum += value * value;
        }
        
        float invLenght = 1.0 / sqrt(sum);
        
        for (int j = 0; j < channels; j++) {
            dst[i][j] *= invLenght;
        }
    }
}

void som::minmaxByCols(vector<vector<float>> &dst) {
    auto channels = dst[0].size();
    
    vector<float> aComponents, bComponents;
    
    for (auto i = 0; i < channels; i++) {
        float max_ = FLT_MIN;
        float min_ = FLT_MAX;
        
        for (auto j = 0; j < dst.size(); j++) {
            float value = dst[j][i];
            
            max_ = max(value, max_);
            min_ = min(value, min_);
        }
        
        aComponents.push_back(1.0 / (max_ - min_));
        bComponents.push_back(-min_ / (max_ - min_));
    }
    
    for (auto i = 0; i < dst.size(); i++) {
        for (auto j = 0; j < channels; j++) {
            dst[i][j] = aComponents[j] * dst[i][j] + bComponents[j];
        }
    }
}

vector<vector<float>> som::reduceDimensionality(const SOM &som, const size_t dimensionality, const bool gradient) {
    auto channels = som.getNodeDimensionality();
    
    auto cells = som.getCells();
    
    vector<vector<float>> weights;
    
    for (size_t i = 0; i < cells.size(); i++) {
        weights.push_back(reduceVector(cells[i], channels, dimensionality, gradient));
    }
    
    minmaxByCols(weights);
    
    return weights;
}

vector<float> som::reduceVector(const Cell &cell, const size_t vectorDimensionality, const size_t targetDimensionality, const bool gradient) {
    vector<float> result;
    for (size_t i = 0; i < targetDimensionality; i++) {
        result.push_back(0.0);
    }
    
    size_t iterations = vectorDimensionality - targetDimensionality + 1;
    
    for (size_t i = 0; i < iterations; i++) {
        for (size_t j = 0; j < targetDimensionality; j++) {
            result[j] += gradient ? cell.weights[i + j] * cell.weights[i + j] : cell.weights[i + j];
        }
    }
    
    return result;
}

