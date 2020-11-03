cmake_minimum_required(VERSION 3.10)

project(tutorial-libtorch VERSION 1.0.0)

# This sets the C++ version to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Import Torch C++ (this looks for the cmake-files is
# a number of paths including `/usr/local/share/cmake`
# where we installed it)
find_package(Torch REQUIRED)

# Make Target
add_executable(lstm-nh main.cpp)

# Link Torch C++ libs
target_link_libraries(lstm-nh "${TORCH_LIBRARIES}")
