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

static const auto DATA_COLOR = Scalar(0, 0, 255);
static const auto NODES_THREAD_COLOR = Scalar(255, 0, 0);

int main(int argc, const char * argv[]) {
    const auto retrainCount = 100;
    
    for (auto iter = 0; iter < retrainCount; iter++) {
        // Create random data set
        vector<vector<float>> data;
        
        mt19937_64 rng_;
        uint64_t timeSeed = chrono::high_resolution_clock::now().time_since_epoch().count();
        seed_seq seq{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
        rng_.seed(seq);
        
        uniform_real_distribution<float> uniform_(200, 300);
        
        for (auto i = 0; i < 360 * 5; i++) {
            data.push_back({
                WINDOW_WIDTH/2 + uniform_(rng_) * cos((float)i),
                WINDOW_HEIGHT/2 + uniform_(rng_) * sin((float)i)
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
        
        for (auto i = 0; i < epochs; i += step) {
            som.epochs(step);

            Mat dst(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3, WHITE_COLOR);
            
            for (auto j = 0; j < data.size(); j++) {
                circle(dst, cv::Point2f(data[j][0], data[j][1]), 1, DATA_COLOR, 2);
            }
            
            for (auto j = 0; j < cells.size() - 1; j++) {
                cv::Point2f p1(cells[j].weights[0], cells[j].weights[1]);
                cv::Point2f p2(cells[j + 1].weights[0], cells[j + 1].weights[1]);
                
                line(dst, p1, p2, NODES_THREAD_COLOR, 2);
            }
            
            imshow(TRAINING_PROCESS_WINDOW_NAME, dst);
            imwrite(to_string(i) + ".png", dst);
            
            waitKey(1);
        }
        
        som.release();
    }
    
    waitKey();
    cv::destroyAllWindows();
    
    return 0;
}
