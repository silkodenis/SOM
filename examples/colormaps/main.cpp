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
#include <vector>

using namespace cv;
using namespace som;
using namespace std;

Mat drawColorBar(const string &name, ColorScale colorScale) {
    const auto lenght = 800;
    const auto thickness = 30;
    const auto horizontal = true;
    const auto coordinates = false;
    
    Mat colorbar = draw1DColorBar(ColorbarConfiguration(lenght, thickness, horizontal, coordinates), ColormapConfiguration(colorScale));
    Mat title(colorbar.rows, 80, CV_8UC3, COLOR_WHITE);
    
    cv::putText(title, name, Point(5, colorbar.rows/1.5), FONT_HERSHEY_PLAIN, 0.75, COLOR_BLACK, 1, LINE_AA);
    
    Mat result;
    hconcat(vector<Mat> {title, colorbar}, result);
    
    return result;
}

int main(int argc, const char * argv[]) {
    
    vector<Mat> allColorbars {
        drawColorBar("Gray", COLORSCALE_GRAY),
        drawColorBar("Autumn", COLORSCALE_AUTUMN),
        drawColorBar("Bone", COLORSCALE_BONE),
        drawColorBar("Jet", COLORSCALE_JET),
        drawColorBar("Winter", COLORSCALE_WINTER),
        drawColorBar("Rainbow", COLORSCALE_RAINBOW),
        drawColorBar("Ocean", COLORSCALE_OCEAN),
        drawColorBar("Summer", COLORSCALE_SUMMER),
        drawColorBar("Spring", COLORSCALE_SPRING),
        drawColorBar("Cool", COLORSCALE_COOL),
        drawColorBar("HSV", COLORSCALE_HSV),
        drawColorBar("Pink", COLORSCALE_PINK),
        drawColorBar("Hot", COLORSCALE_HOT),
        drawColorBar("Parula", COLORSCALE_PARULA),
        drawColorBar("Viridis", COLORSCALE_VIRIDIS),
        drawColorBar("Plasma", COLORSCALE_PLASMA),
        drawColorBar("Inferno", COLORSCALE_INFERNO),
        drawColorBar("Magma", COLORSCALE_MAGMA),
        drawColorBar("Cividis", COLORSCALE_CIVIDIS),
        drawColorBar("Cool-Warm", COLORSCALE_COOLWARM)
    };
    
    Mat colormaps;
    vconcat(allColorbars, colormaps);
    
    namedWindow("Available Colormaps");
    moveWindow("Available Colormaps", 150, 150);
    imshow("Available Colormaps", colormaps);
    
    waitKey();

    cv::destroyAllWindows();
    
    return 0;
}
