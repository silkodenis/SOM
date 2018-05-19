cmake_minimum_required(VERSION 2.8)

project(som)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_MACOSX_RPATH 1)

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})

include_directories(include/public)
include_directories(include/private)
include_directories(include/private/computing)
include_directories(include/private/computing/kernels)
include_directories(include/private/model)
include_directories(include/private/model/grid)

set(SOURCE_LIB
src/som.cpp
src/trainer.cpp
src/model/model.cpp
src/model/normalizer.cpp
src/model/grid/grid.cpp
src/model/grid/hex.cpp
src/model/grid/hexagon_grid.cpp
src/model/grid/rectangle_grid.cpp
src/computing/computing.cpp
src/computing/kernels/angular_distance_kernel.cpp
src/computing/kernels/euclidean_distance_kernel.cpp
src/computing/kernels/kernel.cpp
src/computing/kernels/squared_distance_kernel.cpp
src/computing/kernels/taxicab_distance_kernel.cpp
src/computing/kernels/topological_distance_kernel.cpp
src/computing/kernels/weight_distance_kernel.cpp)

set(PUBLIC_HEADERS_LIB
include/public/som_types.hpp
include/public/som.hpp)

set(PRIVATE_HEADERS_LIB
include/private/trainer.hpp
include/private/model/model.hpp
include/private/model/normalizer.hpp
include/private/model/grid/grid.hpp
include/private/model/grid/hex.hpp
include/private/model/grid/hexagon_grid.hpp
include/private/model/grid/rectangle_grid.hpp
include/private/computing/computing.hpp
include/private/computing/kernels/angular_distance_kernel.hpp
include/private/computing/kernels/euclidean_distance_kernel.hpp
include/private/computing/kernels/kernel.hpp
include/private/computing/kernels/squared_distance_kernel.hpp
include/private/computing/kernels/taxicab_distance_kernel.hpp
include/private/computing/kernels/topological_distance_kernel.hpp
include/private/computing/kernels/weight_distance_kernel.hpp)

if(NOT CMAKE_GENERATOR STREQUAL Xcode)
	set(PUBLIC_HEADERS_LIB
	include/public/som.hpp
	include/public/som_types.hpp)
	file(COPY ${PUBLIC_HEADERS_LIB} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/include)
endif()


if(BUILD_SHARED_LIBS)
add_library(som SHARED ${SOURCE_LIB} ${PUBLIC_HEADERS_LIB} ${PRIVATE_HEADERS_LIB})
target_link_libraries(som ${OpenCL_LIBRARY})

set_target_properties(som PROPERTIES PUBLIC_HEADER "${PUBLIC_HEADERS_LIB}")
install(TARGETS som 
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/som
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/som
)

else()
add_library(som STATIC ${SOURCE_LIB} ${PUBLIC_HEADERS_LIB} ${PRIVATE_HEADERS_LIB})
target_link_libraries(som ${OpenCL_LIBRARY})

set_target_properties(som PROPERTIES PUBLIC_HEADER "${PUBLIC_HEADERS_LIB}")
install(TARGETS som 
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/som
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/som
)

endif()

enable_testing()
add_subdirectory(test)