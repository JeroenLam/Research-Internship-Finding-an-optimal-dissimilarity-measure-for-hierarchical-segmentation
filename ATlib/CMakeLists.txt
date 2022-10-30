cmake_minimum_required(VERSION 3.10)
find_package( OpenCV REQUIRED )

# Add OpenCV
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(AlphaTree alpha-tree.cpp
                      EdgeQueue.cpp
                      alphaGen.cpp
                      at-fitness.cpp
                      support.cpp
                      simulation.cpp)