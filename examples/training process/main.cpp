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

#include <random>
#include <chrono>
#include <iomanip>
#include "som.hpp"
#include "som_cv_view.hpp"

using namespace cv;
using namespace som;
using namespace std;

static const string TRAINING_PROCESS_WINDOW_NAME = "Training process";

extern vector<vector<cl_float>> createRandomDataSet(const size_t channels, const size_t labels, uniform_real_distribution<cl_float> noiseRange);
extern void showTrainingProcess(SOM &som, const size_t iterationsCount, const double learningRate, const size_t step, const Metric metric);

int main(int argc, const char * argv[]) {
    
    // Create random data set
    const size_t channels = 5;
    const size_t labels = 3;
    const uniform_real_distribution<cl_float> noiseRange(-0.8, 0.8);

    vector<vector<cl_float>> data = createRandomDataSet(channels, labels, noiseRange);
    
    // Create SOM
    const auto radius = 10;
    const auto hexSize = 10;
    const auto learningRate = 0.2;
    const auto iterationsCount = 3000;
    const auto step = 10;
    const auto metric = SQUARED;

    SOM som(CPU);
    som.create(radius, hexSize, channels);
    som.prepare(data, NO_NORM, RANDOM_01);
    
    showTrainingProcess(som, iterationsCount, learningRate, step, metric);
    
    waitKey();

    cv::destroyAllWindows();
    
    return 0;
}

void showTrainingProcess(SOM &som, const size_t iterationsCount, const double learningRate, const size_t step, const Metric metric) {
    const auto width = 800;
    const auto height = 600;
    const auto verticalOffset = 50;
    const auto horizontalOffset = 50;
    const auto widthGraph = width - horizontalOffset * 2;
    const auto heightGraph = height - verticalOffset * 2;
    const auto graphThickness = 2;
    const auto axisThickness = 1;
    
    const Scalar bgColor(WHITE_COLOR);
    const Scalar axisColor(BLACK_COLOR);
    const Scalar errorColor(RED_COLOR);
    const Scalar diffColor(CORAL_COLOR);
    const Scalar gridColor3(WHITE_COLOR * 0.60);
    const Scalar fontColor(BLACK_COLOR);
    
    const Point origin(horizontalOffset, verticalOffset + heightGraph);
    
    vector<double> offsets, errors, diffs;
    
    double maxError = FLT_MIN;
    double maxDiff = FLT_MIN;
    double lastError = 0;
    
    som.train(iterationsCount, learningRate, metric, true);
    
    for (size_t i = 0; i <= iterationsCount; i += step) {
        som.epochs(step);
        
        offsets.push_back(origin.x + (float)i / iterationsCount * widthGraph);
        
        double error = som.computeError();
        double diff = abs(error - lastError) * ((float)step / iterationsCount);
        
        errors.push_back(error);
        diffs.push_back(diff);
        maxError = max(maxError, error);
        maxDiff = max(maxDiff, diff);
        lastError = error;
        
        Mat result(height, width, CV_8UC3, bgColor);
        
        // Axes
        // y
        line(result, Point(horizontalOffset, verticalOffset), Point(horizontalOffset, verticalOffset + heightGraph), axisColor, axisThickness);
        // x
        line(result, Point(horizontalOffset, verticalOffset + heightGraph), Point(horizontalOffset + widthGraph, verticalOffset + heightGraph), axisColor, axisThickness);
        
        // Text
        const int fontFace = FONT_HERSHEY_TRIPLEX;
        const double fontScale = 0.6;
        const int thickness = 1;
        
        putText(result, to_string(iterationsCount), Point(horizontalOffset + widthGraph - 10, verticalOffset + heightGraph + 15),
                fontFace, fontScale, fontColor, thickness, 8);
        putText(result, "0", Point(horizontalOffset - 15, verticalOffset + heightGraph + 15), fontFace, fontScale, fontColor, thickness, 8);
        putText(result, "epochs: " + to_string(i), Point(horizontalOffset, verticalOffset * 0.8), fontFace, fontScale, fontColor, thickness, 8);
        
        stringstream errorStringStream; errorStringStream << fixed << setprecision(4) << error;
        string errorText = "error: " + errorStringStream.str();
        putText(result, errorText, cv::Point(width * 0.4, verticalOffset * 0.8), fontFace, fontScale, errorColor, thickness, 8);
        
        stringstream diffStringStream; diffStringStream << fixed << setprecision(4) << diff / step * 100000;
        string diffText = "diff: " + diffStringStream.str();
        cv::putText(result, diffText, cv::Point(width * 0.75, verticalOffset * 0.8), fontFace, fontScale, diffColor, thickness, 8);
        
        // Graphs
        for (size_t j = 1; j < offsets.size(); j++) {
            line(result, Point2f(offsets[j - 1], origin.y - errors[j - 1] / maxError * heightGraph),
                 Point2f(offsets[j], origin.y - errors[j] / maxError * heightGraph), errorColor, graphThickness);
            
            line(result, Point2f(offsets[j - 1], origin.y - diffs[j - 1] / maxDiff * heightGraph),
                 Point2f(offsets[j], origin.y - diffs[j] / maxDiff * heightGraph), diffColor, graphThickness);
        }
        
        // Sliding axle
        line(result, Point2f(offsets[offsets.size() - 1], verticalOffset),
             Point2f(offsets[offsets.size() - 1], verticalOffset + heightGraph), gridColor3, 1);
        
        // Show graph
        imshow(TRAINING_PROCESS_WINDOW_NAME, result);

        waitKey(1);
    }
}

vector<vector<cl_float>> createRandomDataSet(const size_t channels, const size_t labels, uniform_real_distribution<cl_float> noiseRange) {
    mt19937_64 rng;
    uint64_t timeSeed = chrono::high_resolution_clock::now().time_since_epoch().count();
    seed_seq seq{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    rng.seed(seq);
    
    uniform_real_distribution<cl_float> vectorRange(0.0, 1.0);
    
    vector<vector<cl_float>> data;
    for (size_t i = 0; i < labels; i++) {
        vector<cl_float> vector;
        
        for (size_t j = 0; j < channels; j++) {
            vector.push_back(vectorRange(rng));
        }
        
        for (size_t l = 0; l < 300; l++) {
            std::vector<cl_float> noiseVector;
            
            for (size_t j = 0; j < channels; j++) {
                float value = vector[j] + noiseRange(rng);
                
                if (value < 0) {
                    value = 0.0;
                } else if (value > 1) {
                    value = 1.0;
                }
                
                noiseVector.push_back(value);
            }
            
            data.push_back(noiseVector);
        }
        
        data.push_back(vector);
    }
    
    return data;
}
