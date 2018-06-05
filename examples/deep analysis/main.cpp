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

static const string APPROXIMATION_MAP_WINDOW_NAME = "Approximation";
static const string DISTANCES_MAP_WINDOW_NAME = "Distances";
static const string ACTIVES_ONLY_MAP_WINDOW_NAME = "Actives only 3D + 1D";
static const string CONVOLUTION_MAP_WINDOW_NAME = "Convolution: 3D + 1D";
static const string CHANNELS_MAP_WINDOW_NAME = "Channels";

extern vector<vector<cl_float>> createRandomDataSet(const size_t channels,
                                                    const size_t labels,
                                                    uniform_real_distribution<cl_float> noiseRange);
extern Mat drawSingleChannelMaps(const SOM &som, const GradientMap gradientMap);

int main(int argc, const char * argv[]) {
    
    // Create random data set
    const size_t channels = 30;
    const size_t labels = 30;
    const uniform_real_distribution<cl_float> noiseRange(-0.1, 0.1);

    vector<vector<cl_float>> data = createRandomDataSet(channels, labels, noiseRange);

    // Create SOM
    const auto radius = 40;
    const auto hexSize = 10;
    const auto learningRate = 0.2;
    const auto iterationsCount = 10000;

    SOM som(CPU);
    som.create(radius, hexSize, channels);
    som.prepare(data, MINMAX_BY_ROWS, RANDOM_FROM_DATA);
    som.train(iterationsCount, learningRate, EUCLIDEAN);

    // Single channel maps
    Mat allChannels = drawSingleChannelMaps(som, GRADIENT_JET);
    
    namedWindow(CHANNELS_MAP_WINDOW_NAME);
    moveWindow(CHANNELS_MAP_WINDOW_NAME, 20, 20);
    imshow(CHANNELS_MAP_WINDOW_NAME, allChannels);
    
    // Aproximation maps
    Mat allMapsMat;
    vector<Mat> allMaps;

    allMaps.push_back(drawApproximationMap(som, GRADIENT_PARULA));
    allMaps.push_back(drawApproximationMap(som, GRADIENT_HUE));
    hconcat(allMaps, allMapsMat);

    namedWindow(APPROXIMATION_MAP_WINDOW_NAME);
    moveWindow(APPROXIMATION_MAP_WINDOW_NAME, 200, 180);
    imshow(APPROXIMATION_MAP_WINDOW_NAME, allMapsMat);
    
    // Distances maps
    allMaps.clear();
    allMapsMat.release();
    allMaps.push_back(drawDistancesMap(som, GRADIENT_SKY_BLUE_TO_PINK));
    allMaps.push_back(drawDistancesMap(som, GRADIENT_JET));
    hconcat(allMaps, allMapsMat);
    
    namedWindow(DISTANCES_MAP_WINDOW_NAME);
    moveWindow(DISTANCES_MAP_WINDOW_NAME, 180, 160);
    imshow(DISTANCES_MAP_WINDOW_NAME, allMapsMat);

    // Only activated nodes
    allMaps.clear();
    allMapsMat.release();
    allMaps.push_back(draw3DMap(som, false, true));
    allMaps.push_back(draw1DMap(som, GRADIENT_HUE, false, true));
    hconcat(allMaps, allMapsMat);

    namedWindow(ACTIVES_ONLY_MAP_WINDOW_NAME);
    moveWindow(ACTIVES_ONLY_MAP_WINDOW_NAME, 140, 120);
    imshow(ACTIVES_ONLY_MAP_WINDOW_NAME, allMapsMat);

    // Convolution 3D + 1D
    allMaps.clear();
    allMapsMat.release();
    allMaps.push_back(draw3DMap(som));
    allMaps.push_back(draw1DMap(som, GRADIENT_PARULA));
    hconcat(allMaps, allMapsMat);

    namedWindow(CONVOLUTION_MAP_WINDOW_NAME);
    moveWindow(CONVOLUTION_MAP_WINDOW_NAME, 60, 40);
    imshow(CONVOLUTION_MAP_WINDOW_NAME, allMapsMat);

    waitKey();

    cv::destroyAllWindows();
    
    return 0;
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

Mat drawNumberedChannel(const SOM &som, const size_t channelIndex, const GradientMap gradientMap, const float scale) {
    Mat channel = drawSingleChannelMap(som, channelIndex, gradientMap);
    
    auto text = to_string(channelIndex);
    auto fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
    auto fontScale = 1;
    auto thickness = 3;
    cv::Point textOrg(channel.cols * 0.1, channel.rows * 0.1);
    cv::putText(channel, text, textOrg, fontFace, fontScale, Scalar::all(0), thickness, 8);
    
    resize(channel, channel, Size(channel.cols * scale, channel.rows * scale));
    
    return channel;
}

Mat drawSingleChannelMaps(const SOM &som, const GradientMap gradientMap) {
    const auto channels = som.getNodeDimensionality();
    
    vector<vector<Mat>> allMaps_v;
    
    auto channelMapScale = 0.35;
    auto h_size = 6;
    auto v_size = (int)channels / h_size;
    auto v_surplus = channels % h_size;
    
    for (size_t i = 0; i < v_size; i++) {
        vector<Mat> maps;
        
        for (size_t j = 0; j < 6; j++) {
            auto channelIndex = i * h_size + j;
            
            Mat channel = drawNumberedChannel(som, channelIndex, gradientMap, channelMapScale);
            maps.push_back(channel);
        }
        
        allMaps_v.push_back(maps);
    }
    
    if (v_surplus > 0) {
        vector<Mat> surplusMaps;
        for (size_t i = 0; i < h_size; i++) {
            if (h_size * v_size + i < channels) {
                auto channelIndex = h_size * v_size + i;
                
                Mat channel = drawNumberedChannel(som, channelIndex, gradientMap, channelMapScale);
                surplusMaps.push_back(channel);
            } else {
                surplusMaps.push_back(Mat(surplusMaps[i-1].size(), CV_8UC3, Scalar::all(255)));
            }
        }
        
        allMaps_v.push_back(surplusMaps);
    }
    
    Mat allChannels;
    vector<Mat> v_concat;
    
    for (size_t i = 0; i < allMaps_v.size(); i++) {
        Mat mapsMat;
        hconcat(allMaps_v[i], mapsMat);
        v_concat.push_back(mapsMat);
    }
    
    vconcat(v_concat, allChannels);
    
    return allChannels;
}
