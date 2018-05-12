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

#include "som.hpp"
#include "som_cv_view.hpp"

using namespace cv;
using namespace som;
using namespace std;

static const string CURRENT_MAP_WINDOW_NAME = "current map";
static const string UNTRAINED_MAP_WINDOW_NAME = "untrained map";

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
    
    const auto radius = 20;
    const auto hexSize = 18;
    const auto channels = 3;
    const auto learningRate = 0.1;
    const auto iterationsCount = 7000;
    const auto metric = EUCLIDEAN;
    const auto manual = true;
    
    SOM som(GPU);
    som.create(radius, hexSize, channels);
    som.prepare(data, MINMAX_BY_COLUMNS, RANDOM_FROM_DATA);
    
    namedWindow(UNTRAINED_MAP_WINDOW_NAME); moveWindow(UNTRAINED_MAP_WINDOW_NAME, 40, 100);
    imshow(UNTRAINED_MAP_WINDOW_NAME, draw3DMap(som, true));

    som.train(iterationsCount, learningRate, metric, manual);
    
    namedWindow(CURRENT_MAP_WINDOW_NAME); moveWindow(CURRENT_MAP_WINDOW_NAME, 700, 100);
    
    while (!som.epochs(5)) {
        imshow(CURRENT_MAP_WINDOW_NAME, draw3DMap(som, true));
        waitKey(1);
    }
    
    cout << "SOM: Training completed." << endl;
    
    waitKey();
    
    destroyAllWindows();
    
    return 0;
}
