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

static const string TRAINED_MAP_WINDOW_NAME = "Trained map";
static const string UNTRAINED_MAP_WINDOW_NAME = "Untrained map";

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

    const auto cols = 15;
    const auto rows = 15;
    const auto hexSize = 30;
    const auto channels = 3;
    const auto learningRate = 0.2;
    const auto epochs = 2000;
    
    SOM som(CPU);
    som.create(cols, rows, hexSize, channels);
    
    namedWindow(UNTRAINED_MAP_WINDOW_NAME); moveWindow(UNTRAINED_MAP_WINDOW_NAME, 40, 100);
    imshow(UNTRAINED_MAP_WINDOW_NAME, draw3DMap(som, true));
    
    som.prepare(data);
    som.train(epochs, learningRate);

    namedWindow(TRAINED_MAP_WINDOW_NAME); moveWindow(TRAINED_MAP_WINDOW_NAME, 700, 100);
    imshow(TRAINED_MAP_WINDOW_NAME, draw3DMap(som, true));
    
    waitKey();
    
    cv::destroyAllWindows();
    
    return 0;
}
