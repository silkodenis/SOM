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

#include "som.hpp"
#include "cv_view.hpp"

using namespace cv;
using namespace som;
using namespace std;

static const string SAVED_MAP_WINDOW_NAME = "Saved SOM";
static const string LOADED_MAP_WINDOW_NAME = "Loaded SOM";

int main(int argc, const char * argv[]) {
    
    // BGR colors
    vector<float> red      {0.00, 0.0, 1.00},
                  green    {0.0,  1.0, 0.00},
                  blue     {1.00, 0.0, 0.00},
                  yellow   {0.20, 1.0, 1.00},
                  orange   {0.25, 0.4, 1.00},
                  purple   {1.0,  0.0, 1.00},
                  dk_green {0.25, 0.5, 0.00},
                  dk_blue  {0.50, 0.0, 0.00};
    
    vector<vector<float>> data{red, green, blue, yellow, orange, purple, dk_green, dk_blue};
    
    const auto cols = 70;
    const auto rows = 70;
    const auto hexSize = 12;
    const auto channels = 3;
    const auto learningRate = 0.003;
    const auto epochs = 20000;
    
    SOM som(CPU);
    
    // Create and training new SOM
    som.create(cols, rows, hexSize, channels);
    som.prepare(data, NO_NORM, RANDOM_FROM_DATA);
    som.setRandomWeights(0, 0.001);
    som.train(epochs, learningRate, SAD);

    // Show trained map
    namedWindow(SAVED_MAP_WINDOW_NAME); moveWindow(SAVED_MAP_WINDOW_NAME, 260, 30);
    imshow(SAVED_MAP_WINDOW_NAME, draw3DMap(som, DisplayConfiguration(false, false, cv::Mat(), COLOR_BLACK)));

    // Save to file and release som object
    som.save("model.som");
    som.release();

    // Load and show saved som
    som.load("model.som");

    namedWindow(LOADED_MAP_WINDOW_NAME); moveWindow(LOADED_MAP_WINDOW_NAME, 440, 80);
    imshow(LOADED_MAP_WINDOW_NAME, draw3DMap(som, DisplayConfiguration(false, false, cv::Mat(), COLOR_BLACK)));

    waitKey();

    cv::destroyAllWindows();
    
    return 0;
}
