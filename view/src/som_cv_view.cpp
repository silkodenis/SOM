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
    static const Scalar BACKGROUND_COLOR(COLOR_WHITE);
    static const Scalar GRID_COLOR(COLOR_BLACK);
    static const Scalar CIRCLE_COLOR(COLOR_RED);
    
    extern Mat drawColorBar(const Mat &colormap, const size_t width, const size_t height, const bool coordinates);
    extern void addColorBar(Mat &src, const Mat &colorbar, const Scalar backgroundColor);
    
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

#pragma mark - Text

void addText(Mat &dst, const string &text, const double fontScale, Point2f origin, const Scalar color) {
    cv::putText(dst, text, origin, FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale, color, 1, LINE_AA);
}

#pragma mark - Coordinates

void addColorBarCoordinates(Mat &dst) {
    const auto fontScale = 0.32;
    
    if (dst.cols > dst.rows) {
        Mat coordinatesMat(18, dst.cols, CV_8UC3, COLOR_WHITE);
        
        addText(coordinatesMat, "0.0", fontScale, Point2f(2, 12), COLOR_BLACK);
        addText(coordinatesMat, "0.5", fontScale, Point2f(dst.cols / 2.0 - 6, 12), COLOR_BLACK);
        addText(coordinatesMat, "1.0", fontScale, Point2f(dst.cols - 22, 12), COLOR_BLACK);
        
        vconcat(vector<Mat> {dst, coordinatesMat}, dst);
        
    } else {
        Mat coordinatesMat(dst.rows, 21, CV_8UC3, COLOR_WHITE);
        
        addText(coordinatesMat, "1.0", fontScale, Point2f(2, 12), COLOR_BLACK);
        addText(coordinatesMat, "0.5", fontScale, Point2f(2, dst.rows/2.0 + 3), COLOR_BLACK);
        addText(coordinatesMat, "0.0", fontScale, Point2f(2, dst.rows - 5), COLOR_BLACK);
        
        hconcat(vector<Mat> {dst, coordinatesMat}, dst);
    }
}

#pragma mark - Colorbars

Mat som::drawColorBar(const Mat &colormap, const size_t width, const size_t height, const bool coordinates) {
    Mat result(height, width, CV_8UC3);
    
    auto coeff = 256.0 / MAX(width, height);
    
    if (width > height) {
        for (auto i = 0; i < height; i++) {
            for (auto j = 0; j < width; j++) {
                result.at<Vec3b>(i, j) = colormap.at<Vec3b>(coeff * j, 0);
            }
        }
    } else {
        for (auto i = 0; i < width; i++) {
            for (auto j = 0; j < height; j++) {
                result.at<Vec3b>(j, i) = colormap.at<Vec3b>(coeff * (height - j - 1), 0);
            }
        }
    }
    
    if (coordinates) {
        addColorBarCoordinates(result);
    }
    
    return result;
}

Mat som::draw1DColorBar(const ColorbarConfiguration colorBarConfiguration, const ColormapConfiguration colormapConfiguration) {
    Mat colormap = getColormap(colormapConfiguration);
    
    return colorBarConfiguration.horizontal ?
    drawColorBar(colormap, colorBarConfiguration.lenght, colorBarConfiguration.thickness, colorBarConfiguration.coordinates) :
    drawColorBar(colormap, colorBarConfiguration.thickness, colorBarConfiguration.lenght, colorBarConfiguration.coordinates);
}

Mat som::draw3DColorBar(const ColorbarConfiguration colorBarConfiguration) {
    Mat colormap = getColormap(ColormapConfiguration(COLORSCALE_HSV));
    
    Mat colorbar = colorBarConfiguration.horizontal ?
    drawColorBar(colormap, colorBarConfiguration.lenght, colorBarConfiguration.thickness, false) :
    drawColorBar(colormap, colorBarConfiguration.thickness, colorBarConfiguration.lenght, false);
    
    auto isHorizontal = colorbar.cols > colorbar.rows;
    auto koeff = isHorizontal ? 1.0 / colorbar.rows : 1.0 / colorbar.cols;
    
    for (auto i = 0; i < colorbar.rows; i++) {
        for (auto j = 0; j < colorbar.cols; j++) {
            colorbar.at<Vec3b>(i, j) = isHorizontal ? colorbar.at<Vec3b>(i, j) * (koeff * i) : colorbar.at<Vec3b>(i, j) * (koeff * j);
        }
    }
    
    if (colorBarConfiguration.coordinates) {
        addColorBarCoordinates(colorbar);
    }
    
    return colorbar;
}

void som::addColorBar(Mat &src, const Mat &colorbar, const Scalar backgroundColor) {
    if (colorbar.cols > colorbar.rows) { // Horizontal
        if (colorbar.cols > src.cols) {
            int diff = colorbar.cols - src.cols;
            int borderLenght = diff / 2;
            
            hconcat(vector<Mat> {
                Mat(src.rows, borderLenght, CV_8UC3, backgroundColor),
                src,
                Mat(src.rows, colorbar.cols - (src.cols + borderLenght) , CV_8UC3, backgroundColor),
            }, src);
            
            vconcat(vector<Mat> {src, colorbar}, src);
        } else if (colorbar.cols == src.cols) {
            vconcat(vector<Mat> {src, colorbar}, src);
        } else {
            int diff = src.cols - colorbar.cols;
            int borderLenght = diff / 2;
            
            Mat colorbar_ = colorbar.clone();
            
            hconcat(vector<Mat> {
                Mat(colorbar.rows, borderLenght, CV_8UC3, backgroundColor),
                colorbar,
                Mat(colorbar.rows, src.cols - (colorbar.cols + borderLenght) , CV_8UC3, backgroundColor),
            }, colorbar_);
            
            vconcat(vector<Mat> {src, colorbar_}, src);
        }
    } else {                              // Vertical
        if (colorbar.rows > src.rows) {
            int diff = colorbar.rows - src.rows;
            int borderLenght = diff / 2;
            
            vconcat(vector<Mat> {
                Mat(borderLenght, src.cols, CV_8UC3, backgroundColor),
                src,
                Mat(colorbar.rows - (borderLenght + src.rows), src.cols, CV_8UC3, backgroundColor),
            }, src);
            
            hconcat(vector<Mat> {src, colorbar}, src);
        } else if(colorbar.rows == src.rows) {
            hconcat(vector<Mat> {src, colorbar}, src);
        } else {
            int diff = src.rows - colorbar.rows;
            int borderLenght = diff / 2;
            
            Mat colorbar_ = colorbar.clone();
            
            vconcat(vector<Mat> {
                Mat(borderLenght, colorbar.cols, CV_8UC3, backgroundColor),
                colorbar,
                Mat(src.rows - (borderLenght + colorbar.rows), colorbar.cols, CV_8UC3, backgroundColor),
            }, colorbar_);
            
            hconcat(vector<Mat> {src, colorbar_}, src);
        }
    }
}

#pragma mark - Maps

cv::Mat som::draw3DMap(const SOM& som, const DisplayConfiguration displayConfiguration) {
    Mat result(som.getHeight(), som.getWidth(), CV_8UC3, displayConfiguration.backgroundColor);
    
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
        bool canDrawNode = (displayConfiguration.onlyActive && cells[i].state[0] > 0) || !displayConfiguration.onlyActive;
        
        if (canDrawNode) {
            Scalar nodeColor;
            
            float b = weights[i][0] * 255;
            float g = weights[i][1] * 255;
            float r = weights[i][2] * 255;
            
            if (channels > 2) {
                nodeColor = Scalar(b, g, r);
            } else if (channels > 1) {
                nodeColor = Scalar(b, g, 0);
            } else {
                nodeColor = Scalar(b, 0, 0);
            }
            
            drawCell(result, cells[i], nodeColor, displayConfiguration.grid ? LINE_8 : LINE_AA);
        }
        
        if (displayConfiguration.grid) {
            drawGrid(result, cells[i], GRID_COLOR);
        }
    }
    
    if (!displayConfiguration.colorbar.empty()) {
        addColorBar(result, displayConfiguration.colorbar, displayConfiguration.backgroundColor);
    }
    
    return result;
}

cv::Mat som::draw1DMap(const SOM &som, const ColormapConfiguration colormapConfiguration, const DisplayConfiguration displayConfiguration) {
    Mat result(som.getHeight(), som.getWidth(), CV_8UC3, displayConfiguration.backgroundColor);

    auto cells = som.getCells();

    vector<vector<float>> weights = reduceDimensionality(som, 1, false);
    
    Mat colormap = getColormap(colormapConfiguration);

    for (size_t i = 0; i < cells.size(); i++) {
        bool canDrawNode = (displayConfiguration.onlyActive && cells[i].state[0] > 0) || !displayConfiguration.onlyActive;

        if (canDrawNode) {
            float value = weights[i][0] * 255;
            
            Scalar nodeColor = colormap.at<Vec3b>(value, 0);

            drawCell(result, cells[i], nodeColor, displayConfiguration.grid ? LINE_8 : LINE_AA);
        }

        if (displayConfiguration.grid) {
            drawGrid(result, cells[i], GRID_COLOR);
        }
    }
    
    if (!displayConfiguration.colorbar.empty()) {
        addColorBar(result, displayConfiguration.colorbar, displayConfiguration.backgroundColor);
    }

    return result;
}

cv::Mat som::drawSingleChannelMap(const SOM& som,
                                  const size_t channel,
                                  const ColormapConfiguration colormapConfiguration,
                                  const DisplayConfiguration displayConfiguration) {
    Mat result(som.getHeight(), som.getWidth(), CV_8UC3, displayConfiguration.backgroundColor);
    
    auto cells = som.getCells();
    
    vector<vector<float>> values;
    for (size_t i = 0; i < cells.size(); i++) {
        values.push_back({cells[i].weights[channel]});
    }
    
    minmaxByCols(values);
    
    Mat colormap = getColormap(colormapConfiguration);
    
    for (auto i = 0; i < cells.size(); i++) {
        bool canDrawNode = (displayConfiguration.onlyActive && cells[i].state[0]) || !displayConfiguration.onlyActive;
        
        if (canDrawNode) {
            float value = values[i][0] * 255;
            
            Scalar nodeColor = colormap.at<Vec3b>(value, 0);
            
            drawCell(result, cells[i], nodeColor, displayConfiguration.grid ? LINE_8 : LINE_AA);
        }
        
        if (displayConfiguration.grid) {
            drawGrid(result, cells[i], GRID_COLOR);
        }
    }
    
    if (!displayConfiguration.colorbar.empty()) {
        Mat colorbar = displayConfiguration.colorbar;
        
        addColorBar(result, colorbar, displayConfiguration.backgroundColor);
    }
    
    return result;
}

cv::Mat som::drawDistancesMap(const SOM &som, const ColormapConfiguration colormapConfiguration, const DisplayConfiguration displayConfiguration) {
    Mat result(som.getHeight(), som.getWidth(), CV_8UC3, displayConfiguration.backgroundColor);
    
    auto cells = som.getCells();
    
    vector<vector<float>> values;
    for (size_t i = 0; i < cells.size(); i++) {
        values.push_back({(float)cells[i].distance[0]});
    }
    
    minmaxByCols(values);
    
    Mat colormap = getColormap(colormapConfiguration);
    
    for (size_t i = 0; i < cells.size(); i++) {
        bool canDrawNode = (displayConfiguration.onlyActive && cells[i].state[0] > 0) || !displayConfiguration.onlyActive;
        
        if (canDrawNode) {
            float value = values[i][0] * 255;
            
            Scalar nodeColor = colormap.at<Vec3b>(value, 0);
            
            drawCell(result, cells[i], nodeColor, displayConfiguration.grid ? LINE_8 : LINE_AA);
        }
        
        if (displayConfiguration.grid) {
            drawGrid(result, cells[i], GRID_COLOR);
        }
    }
    
    if (!displayConfiguration.colorbar.empty()) {
        Mat colorbar = displayConfiguration.colorbar;
        
        addColorBar(result, colorbar, displayConfiguration.backgroundColor);
    }
    
    return result;
}

cv::Mat som::drawApproximationMap(const SOM &som, const ColormapConfiguration colormapConfiguration, const DisplayConfiguration displayConfiguration) {
    Mat result(som.getHeight(), som.getWidth(), CV_8UC3, displayConfiguration.backgroundColor);
    
    auto cells = som.getCells();
    
    vector<vector<float>> values;
    for (size_t i = 0; i < cells.size(); i++) {
        values.push_back({(float)cells[i].state[0]});
    }
    
    minmaxByCols(values);
    
    Mat colormap = getColormap(colormapConfiguration);
    
    for (size_t i = 0; i < cells.size(); i++) {
        bool canDrawNode = (displayConfiguration.onlyActive && cells[i].state[0] > 0) || !displayConfiguration.onlyActive;
        
        if (canDrawNode) {
            float value = values[i][0] * 255;
            
            Scalar nodeColor = colormap.at<Vec3b>(value, 0);
            
            drawCell(result, cells[i], nodeColor, displayConfiguration.grid ? LINE_8 : LINE_AA);
        }
        
        if (displayConfiguration.grid) {
            drawGrid(result, cells[i], GRID_COLOR);
        }
    }
    
    if (!displayConfiguration.colorbar.empty()) {
        Mat colorbar = displayConfiguration.colorbar;
        
        addColorBar(result, colorbar, displayConfiguration.backgroundColor);
    }
    
    return result;
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

