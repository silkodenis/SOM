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

static const string TRAINING_MAT_WINDOW_NAME = "Training mat";
static const string TRAINED_MAP_WINDOW_NAME = "Trained SOM";

static const string ORIGINAL_TARGET_MAT_WINDOW_NAME = "Target mat";
static const string PROCESSED_TARGET_MAT_WINDOW_NAME = "Processed mat";

int main(int argc, const char * argv[]) {
    // load BGR color images
    Mat trainingMat = imread(TRAIN_IMAGE_PATH);
    Mat targetMat = imread(TARGET_IMAGE_PATH);
    
    // Show training mat
    namedWindow(TRAINING_MAT_WINDOW_NAME); moveWindow(TRAINING_MAT_WINDOW_NAME, 80, 20);
    imshow(TRAINING_MAT_WINDOW_NAME, trainingMat);

    const auto cols = 50;
    const auto rows = 50;
    const auto channels = trainingMat.channels();
    const auto hexSize = 11;
    const auto learningRate = 0.1;
    const auto epochs = 5000;

    // Create and training SOM
    SOM som(CPU);
    som.create(cols, rows, hexSize, channels);

    /* cout << "Map error before training: " << som.computeError() << endl; */

    som.prepare(trainingMat.data, trainingMat.total() * channels);
    som.train(epochs, learningRate);

    /* cout << "Map error after training: " << som.computeError() << endl */

    // Show trained SOM
    namedWindow(TRAINED_MAP_WINDOW_NAME); moveWindow(TRAINED_MAP_WINDOW_NAME, 800, 20);
    imshow(TRAINED_MAP_WINDOW_NAME, draw3DMap(som));

    // Show target mat before processing
    namedWindow(ORIGINAL_TARGET_MAT_WINDOW_NAME); moveWindow(ORIGINAL_TARGET_MAT_WINDOW_NAME, -150, 330);
    imshow(ORIGINAL_TARGET_MAT_WINDOW_NAME, targetMat);

    cout << "Applying trained SOM to target mat" << endl;

    auto cells = som.getCells();

    for (auto i = 0; i < targetMat.total(); i++) {
        auto pixelIndex0 = i * channels;
        auto bmuIndex = som.computeBmuIndex(targetMat.data[pixelIndex0]);

        for (auto j = 0; j < channels; j++) {
            targetMat.data[pixelIndex0 + j] = cells[bmuIndex].weights[j];
        }
    }

    // Show result
    namedWindow(PROCESSED_TARGET_MAT_WINDOW_NAME); moveWindow(PROCESSED_TARGET_MAT_WINDOW_NAME, 600, 330);
    imshow(PROCESSED_TARGET_MAT_WINDOW_NAME, targetMat);

    waitKey();

    cv::destroyAllWindows();
    
    return 0;
}
