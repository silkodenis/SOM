cmake_minimum_required(VERSION 2.8)

project(tests)

add_subdirectory(test\ normalization)
add_subdirectory(test\ opencl\ host)
add_subdirectory(test\ topological\ kernel)
add_subdirectory(test\ weight\ kernels)
add_subdirectory(test\ saved\ model)

