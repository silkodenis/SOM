# Self-Organizing Map (SOM)

Fast and most complete C++ library using parallel computing based on OpenCL. The map has a hexagon cells, which allows you to get the correct model view. All this makes it possible to use the library for high-level scientific works.

## Installation on a Unix-based OS
Required Packages:
* `CMake 2.8` or higher
* `Git`

Required Dependencies:
* `OpenCL 1.1 ` or higher 
* `OpenCV 3` [optional] (for build view and examples)

These steps have been tested for macOS High Sierra 10.13.14 but should work with other unix-based systems as well.

```sh
1. $ cd ~/<my_working_directory>
2. $ git clone https://github.com/silkodenis/SOM.git
3. $ cd SOM
4. $ mkdir build
5. $ cd build
```

```sh
## to build som, view and examples
6. $ cmake -D CMAKE_INSTALL_PREFIX=/usr/local ../

## to build only lib som
6. $ cmake -D CMAKE_INSTALL_PREFIX=/usr/local ../som
```

```sh
7. $ make
8. $ make test
9. $ sudo make install
```

### Notes
```sh
## to generate XCode projects use:
6. $ cmake -G Xcode ../
```

```sh
## to build dynamic libs, use option: [-D BUILD_SHARED_LIBS=true], for example:
6. $ cmake -D BUILD_SHARED_LIBS=true -D CMAKE_INSTALL_PREFIX=/usr/local ../
```

## Examples
Below is a brief overview of the examples, the source code of which gives a quick start to work with SOM.

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

An example of using an image as a data set. After receiving the clustered map from one image, we apply it to clustering another image.

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

<p align="center">
  <img width="854" height="489" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/deep%20analysis/convolution%203d%20+%201d.png?raw=true">
</p>

<p align="center">
  <img width="854" height="489" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/deep%20analysis/distances.png?raw=true">
</p>

<p align="center">
  <img width="854" height="489" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/deep%20analysis/actives%20only%203d%20+%201d.png?raw=true">
</p>

<p align="center">
  <img width="854" height="489" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/deep%20analysis/approximation%20gradient%20+%20hue.png?raw=true">
</p>

<p align="center">
  <img width="600" height="571" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/deep%20analysis/channels.png?raw=true">
</p>

**Save and load**

A simple demonstration of saving and loading your model from a binary file.

<p align="center">
  <img width="742" height="643" src="https://github.com/silkodenis/SOM/blob/readme_assets/examples/save%20and%20load/som.png?raw=true">
</p>

## Authors

* **Denis Silko** - *Initial work* - [silkodenis](https://github.com/silkodenis)

## Credits

Thanks to [Amit Patel](http://www-cs-students.stanford.edu/~amitp/) for help in implementing the efficient hexagon grid.

## License

[Apache License 2.0](LICENSE)
