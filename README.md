# Self-Organizing Map (SOM)

Fast and most complete C++ library using parallel computing based on OpenCL. The map has a hexagon cells, which allows you to get the correct model view. All this makes it possible to use the library for high-level scientific works.

## Installation on a Unix-based OS
Required Packages:
* `CMake 2.8` or higher
* `Git`

Dependencies:
* `OpenCL 1.2 `  
* `OpenCV 3` [optional] (for build view and examples)

These steps have been tested for macOS Mojave 10.14 but should work with other unix-based systems as well.

### Install

```sh
1. $ cd ~/<my_working_directory>
2. $ git clone --branch v1.0 https://github.com/silkodenis/SOM.git
3. $ cd SOM
4. $ mkdir build
5. $ cd build
```

```sh
## to build som, view and examples
6. $ cmake -D CMAKE_INSTALL_PREFIX=/usr/local ..

## to build only lib som
6. $ cmake -D CMAKE_INSTALL_PREFIX=/usr/local ../som
```

```sh
7. $ make
8. $ make test
9. $ sudo make install
```

### Uninstall

```sh
9. $ sudo make uninstall
```

### Notes
```sh
## to generate XCode projects use:
6. $ cmake -G Xcode ..
```

```sh
## to build dynamic libs, use option: [-D BUILD_SHARED_LIBS=true], for example:
6. $ cmake -D BUILD_SHARED_LIBS=true -D CMAKE_INSTALL_PREFIX=/usr/local ..
```

## Examples
Below is a brief overview of the examples, the [source code](https://github.com/silkodenis/SOM/tree/master/examples) of which gives a quick start to work with SOM.

**Simple training**

Hello world!

<p align="center">
<img width="412" height="355" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/simple%20training/untrained_map.png?raw=true">

<img width="412" height="355" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/simple%20training/trained_map.png?raw=true">
</p>

**Real-time training**

A simple example of how to training SOM and get a model view in real time.

<p align="center">
<img width="568" height="649" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/real-time%20training/training_proccess.gif?raw=true">
</p>

**Regression**

A simple example of using SOM for regression analysis.

<p align="center">
<img width="650" height="650" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/regression/training_process.gif?raw=true">
</p>

**Image as dataset**

This example demonstrate using image as a data set. After receiving the clustered map from one image, we apply it to clustering another image.

<p align="center">
<img width="800" height="531" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/image%20as%20dataset/image_as_dataset_target%20mat.png?raw=true">
</p>

<p align="center">
<img width="363" height="347" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/image%20as%20dataset/image_as_dataset_training%20mat.png?raw=true">

<img width="405" height="350" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/image%20as%20dataset/image_as_dataset_trained%20SOM.png?raw=true">
</p>

<p align="center">
<img width="800" height="531" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/image%20as%20dataset/image_as_dataset_processed%20mat.png?raw=true">
</p>

**Single channel analysis**

A simple example of analyzing the channels of a trained map. 

<p align="center">
<img width="283" height="324" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/single%20channel%20analysis/rgb_map.png?raw=true">
</p>

<p align="center">
<img width="849" height="324" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/single%20channel%20analysis/single_channel_maps.png?raw=true">
</p>


**Deep analysis**

This example demonstrates various interpretations model view of a trained map using the additional library som_view.

* Convolution maps, 3D(rgb) + 1D(v).

<p align="center">
<img width="854" height="489" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/deep%20analysis/convolution%203d%20+%201d.png?raw=true">
</p>

* Maps of accumulated distances during training.

<p align="center">
<img width="854" height="489" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/deep%20analysis/distances.png?raw=true">
</p>

* Maps from nodes that have been activated during training.

<p align="center">
<img width="854" height="489" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/deep%20analysis/actives%20only%203d%20+%201d.png?raw=true">
</p>

* Approximation maps. The temperature of the node indicates the frequency of activation during training.

<p align="center">
<img width="854" height="489" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/deep%20analysis/approximation%20gradient%20+%20hue.png?raw=true">
</p>

* Single channel maps. These maps show how the resulting clusters depend on the components of the vectors used in training.
  
<p align="center">  
<img width="600" height="571" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/deep%20analysis/channels.png?raw=true">
</p>

**Debugging training process**

This example demonstrate dynamics of map error on the expiration of training epochs. It's important to timely stop training to avoid problem of overfitting. Observation of the convergence dynamics will help you to justify some learning parameters.

<p align="center">
<img width="800" height="600" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/training%20process/training_process.gif?raw=true">
</p>

**Save and load**

A simple demonstration of saving and loading your model from a binary file.

<p align="center">
<img width="742" height="643" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/save%20and%20load/som.png?raw=true">
</p>

## Distance Metrics

Distance metrics can be very importance in the data analyzing using SOM. At the core of learning algorithm is activation(by computing distances from nodes weights to input vector) of the Best Matching Unit. BMU in turn will affect change the weights of its neighbors. The library provides 10 most popular distance metrics:

Definition:
<p align="center">
<img width="351" height="17" src="https://github.com/silkodenis/SOM/blob/readme_assets/distance%20metrics/definition.png?raw=true">
</p>

**Euclidean:**

It is the natural distance in a geometric interpretation and is classic for many solution.
<p align="center">
<img width="374" height="58" src="https://github.com/silkodenis/SOM/blob/readme_assets/distance%20metrics/euclidean.png?raw=true">
</p>


**Minkowski:**

Is the generalized Lp-norm of the difference. Can be considered as a generalization of both the Euclidean distance the case of p=2 and the Manhattan distance the case of p=1. 
<p align="center">
<img width="272" height="54" src="https://github.com/silkodenis/SOM/blob/readme_assets/distance%20metrics/minkowski.png?raw=true">  
</p>

**Chebyshev:**

Minkowski distance with limiting case of p reaching infinity.
<p align="center">
<img width="414" height="54" src="https://github.com/silkodenis/SOM/blob/readme_assets/distance%20metrics/chebyshev.png?raw=true">
</p>

**Manhattan(Taxicab):**

Special case of the Minkowski distance with p=1 and equivalent to the sum of absolute difference. Also known as Taxicab norm, rectilinear distance or L1-norm. Used in regression analysis since the 18th century.
<p align="center">
<img width="230" height="44" src="https://github.com/silkodenis/SOM/blob/readme_assets/distance%20metrics/manhattan.png?raw=true">
</p>

**Canberra:**

It is a weighted version of Manhattan distance. Is often used for data scattered around an origin, as it is biased for measures around the origin and very sensitive for values close to zero.
<p align="center">
<img width="232" height="44" src="https://github.com/silkodenis/SOM/blob/readme_assets/distance%20metrics/canberra.png?raw=true">
</p>

**Cosine:**

Represents the angular distance while ignoring space scale. Is most commonly used in high-dimensional positive spaces and also to measure cohesion within clusters in the field of data mining.
<p align="center">
<img width="265" height="90" src="https://github.com/silkodenis/SOM/blob/readme_assets/distance%20metrics/cosine.png?raw=true">
</p>

**Sum of Absolute Difference(SAD):**

Is equivalent to the L1-norm of the difference, also known as Manhattan or Taxicab-norm. The abs function makes this metric a bit complicated, but it is more robust than SSD.
<p align="center">
<img width="193" height="44" src="https://github.com/silkodenis/SOM/blob/readme_assets/distance%20metrics/SAD.png?raw=true">
</p>

**Sum of Squared Difference(SSD):**

Is equivalent to the squared L2-norm, also known as Euclidean norm. It is therefore also known as Squared Euclidean distance. Squares cause it to be very sensitive to large outliers. Is a standard approach in regression analysis.
<p align="center">
<img width="199" height="44" src="https://github.com/silkodenis/SOM/blob/readme_assets/distance%20metrics/SSD.png?raw=true">
</p>

**Mean-Absolute Error(MAE):**

Is a normalized version SAD.
<p align="center">
<img width="212" height="44" src="https://github.com/silkodenis/SOM/blob/readme_assets/distance%20metrics/MAE.png?raw=true">
</p>

**Mean-Squared Error(MSE):**

Is a normalized version SSD.
<p align="center">
<img width="220" height="44" src="https://github.com/silkodenis/SOM/blob/readme_assets/distance%20metrics/MSE.png?raw=true">
</p>

## Pseudocolor Schemes

View has 20 most popular Matlab and Matplotlib equivalent colormaps.

<p align="center">
<img width="620" height="711" src="https://github.com/silkodenis/SOM/blob/readme_assets/pseudocolor%20schemes/colormaps.gif?raw=true?raw=true">
</p>

The colormap have three parameters for adjustment (inversion, colors quantization and limits).

<p align="center">
<img width="701" height="711" src="https://github.com/silkodenis/SOM/blob/readme_assets/pseudocolor%20schemes/colormap%20configure.gif?raw=true?raw=true?raw=true">
</p>


## Authors

* **Denis Silko** - *Initial work* - [silkodenis](https://github.com/silkodenis)

## Credits

Thanks to [Amit Patel](http://www-cs-students.stanford.edu/~amitp/) for help in implementing the efficient hexagon grid.

## License

[Apache License 2.0](LICENSE)
