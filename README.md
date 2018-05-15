# Self-Organizing Map (SOM)

Fast, convenient and complete C++ library using parallel computing based on OpenCL. The map has a hexagonal cells, which allows you to get the correct model view. All this makes it possible to use the library for high-level scientific works.

## Installation in Unix
Required Packages:
* `CMake 2.8` or higher
* `Git`

Required Dependencies:
* `OpenCL 1.1 ` or higher 
* `OpenCV 3` [optional] (for build lib som_view and examples)

These steps have been tested for macOS High Sierra 10.13.14 but should work with other unix systems as well.

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
8. $ sudo make install
```

### Notes
```sh
## to generate XCode project use:
6. $ cmake -G Xcode ../
```

```sh
## to build dynamic libs, use option: [-D BUILD_SHARED_LIBS=true], for example:
6. $ cmake -D BUILD_SHARED_LIBS=true -D CMAKE_INSTALL_PREFIX=/usr/local ../
```

## Examples

**Simple training**
Hello world!

**Real-time training**
A simple example of how to train SOM and get a model view in real time.

**Regression**
A simple example of using SOM for regression analysis.

**Image as dataset**
An example of using an image as a data set. After receiving the clustered map from one image, we apply it to clustering another image.

**Single channel analysis**
A simple example of analyzing the channels of a trained map. 

**Deep analysis**
This example demonstrates various interpretations model view of a trained map using the additional library som_view.

**Save and load**
A simple demonstration of saving and loading your models from a binary file.


## Authors

* **Denis Silko** - *Initial work* - [silkodenis](https://github.com/silkodenis)

## License

[Apache License 2.0](LICENSE)
