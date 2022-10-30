# Check OpenCV 
cmake_minimum_required(VERSION 3.10)
set( PROJECT_NAME atree.out )
project( ${PROJECT_NAME} )
find_package( OpenCV REQUIRED )

# Add OpenCV
# include_directories( ${OpenCV_INCLUDE_DIRS} )
list( APPEND EXTRA_LIBS ${OpenCV_LIBS} )
list( APPEND EXTRA_INCLUDES ${OpenCV_INCLUDE_DIRS} )

# Add example library
add_subdirectory( ATlib )
list( APPEND EXTRA_LIBS AlphaTree )
list( APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/ATlib" )

# Add the executable
add_executable( ${PROJECT_NAME} main.cpp )     

# Link libraries
target_link_libraries( ${PROJECT_NAME} PUBLIC ${EXTRA_LIBS} )

# Link directories
target_include_directories( ${PROJECT_NAME} PUBLIC "${PROJECT_BINARY_DIR}" ${EXTRA_INCLUDES} )



