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
#include "som.hpp"
#include "som_cv_view.hpp"

using namespace cv;
using namespace som;
using namespace std;

static const auto TRAINING_PROCESS_WINDOW_NAME = "Training process";

static const auto WINDOW_HEIGHT = 650;
static const auto WINDOW_WIDTH = 650;
static const auto RETRAIN_COUNT = 100;

int main(int argc, const char * argv[]) {
    for (auto i = 0; i < RETRAIN_COUNT; i++) {
        // Create random data set
        vector<vector<float>> data;
        
        mt19937_64 rng_;
        uint64_t timeSeed = chrono::high_resolution_clock::now().time_since_epoch().count();
        seed_seq seq{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
        rng_.seed(seq);
        
        uniform_real_distribution<float> scatterNoiseRange(200, 300);
        
        for (auto j = 0; j < 360 * 5; j++) {
            data.push_back({
                WINDOW_WIDTH/2 + scatterNoiseRange(rng_) * cos((float)j),
                WINDOW_HEIGHT/2 + scatterNoiseRange(rng_) * sin((float)j)
            });
        }
        
        // Create SOM
        const auto cols = 100;
        const auto rows = 1;
        const auto hexSize = 20;
        const auto channels = 2;
        
        SOM som(CPU);
        som.create(cols, rows, hexSize, channels);
        som.prepare(data, NO_NORM, RANDOM_FROM_DATA);
        
        // Real-time training and vizualization
        const auto learningRate = 0.1;
        const auto iterationsCount = 2000;
        const auto epochs = 0.35 * iterationsCount;
        const auto manual = true;
        const auto step = 5;
        
        som.train(iterationsCount, learningRate, EUCLIDEAN, manual);
        
        namedWindow(TRAINING_PROCESS_WINDOW_NAME);
        moveWindow(TRAINING_PROCESS_WINDOW_NAME, 150, 50);
        
        auto cells = som.getCells();
        
        for (auto j = 0; j < epochs; j += step) {
            som.epochs(step);

            Mat dst(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3, WHITE_COLOR);
            
            for (auto m = 0; m < data.size(); m++) {
                circle(dst, cv::Point2f(data[m][0], data[m][1]), 1, RED_COLOR, 2);
            }
            
            for (auto m = 0; m < cells.size() - 1; m++) {
                cv::Point2f p1(cells[m].weights[0], cells[m].weights[1]);
                cv::Point2f p2(cells[m + 1].weights[0], cells[m + 1].weights[1]);
                
                line(dst, p1, p2, BLUE_COLOR, 2);
            }
            
            imshow(TRAINING_PROCESS_WINDOW_NAME, dst);
            
            waitKey(1);
        }
        
        som.release();
    }
    
    waitKey();
    cv::destroyAllWindows();
    
    return 0;
}
