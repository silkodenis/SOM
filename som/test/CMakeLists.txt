cmake_minimum_required(VERSION 2.8)

project(tests)

add_subdirectory(normalization)
add_subdirectory(opencl\ host)
add_subdirectory(topological\ distance\ kernel)
add_subdirectory(weight\ distance\ kernels)
add_subdirectory(saved\ model)

