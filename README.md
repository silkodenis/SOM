# Self-Organizing Map (SOM)

Fast, convenient and complete SOM C++ library using parallel computing based on OpenCL. The map has a hexagonal grid, which allows you to get the correct model view. All this makes it possible to use the library for scientific works.

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

#### Notes
```sh
## to generate XCode project use:
6. $ cmake -G Xcode ../
```

```sh
## to build dynamic libraries, use option: [-D BUILD_SHARED_LIBS=true], for example:
6. $ cmake -D BUILD_SHARED_LIBS=true -D CMAKE_INSTALL_PREFIX=/usr/local ../
```


## License

[Apache License 2.0](LICENSE)
