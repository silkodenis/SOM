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
#include <assert.h>

using namespace som;
using namespace std;

int main(int argc, const char * argv[]) {
    
    // Create data set
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
    
    // Create model
    const auto channels = 3;
    const auto radius = 3;
    const auto hexSize = 5;
    const auto learningRate = 0.2;
    const auto iterationsCount = 1000;
    
    SOM som(CPU);
    som.create(radius, hexSize, channels);
    som.prepare(data);
    som.train(iterationsCount, learningRate);
    
    auto cells = som.getCells();
    
    const auto cellsCount = cells.size();
    const auto width = som.getWidth();
    const auto height = som.getHeight();
    const auto topologicalDimensionality = som.getTopologicalDimensionality();
    const auto nodeDimensionality = som.getNodeDimensionality();
    
    vector<vector<cl_float>> weights, points, corners;
    vector<cl_float> distances;
    vector<cl_int> labels, states;
    
    for (auto i = 0; i < cells.size(); i++) {
        vector<cl_float> w, p, c;
        
        for (auto j = 0; j < topologicalDimensionality; j++) {
            p.push_back(cells[i].center[j]);
        }
        
        for (auto j = 0; j < nodeDimensionality; j++) {
            w.push_back(cells[i].weights[j]);
        }
        
        for (auto j = 0; j < HEXAGON_CORNERS_COUNT; j++) {
            c.push_back(cells[i].corners[j]);
        }
        
        som.setLabel(rand(), i);
        
        distances.push_back(cells[i].distance[0]);
        labels.push_back(cells[i].label[0]);
        states.push_back(cells[i].state[0]);
        
        weights.push_back(w);
        points.push_back(p);
        corners.push_back(c);
    }
    
    // Save and release model
    som.save("model.som");
    som.release();
    cells.clear();

    // Load and test saved model
    som.load("model.som");
    cells = som.getCells();
    
    assert(cells.size() == cellsCount);
    assert(som.getWidth() == width);
    assert(som.getHeight() == height);
    assert(som.getTopologicalDimensionality() == topologicalDimensionality);
    assert(som.getNodeDimensionality() == nodeDimensionality);
    
    for (auto i = 0; i < cells.size(); i++) {
        for (auto j = 0; j < topologicalDimensionality; j++) {
            assert(cells[i].center[j] == points[i][j]);
        }
        
        for (auto j = 0; j < nodeDimensionality; j++) {
            assert(cells[i].weights[j] == weights[i][j]);
        }
        
        for (auto j = 0; j < nodeDimensionality; j++) {
            assert(cells[i].corners[j] == corners[i][j]);
        }
        
        assert(cells[i].distance[0] == distances[i]);
        assert(cells[i].label[0] == labels[i]);
        assert(cells[i].state[0] == states[i]);
    }
    
    return 0;
}
