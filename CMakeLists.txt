cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 11)

project(grayscott)

find_package(MPI REQUIRED)

add_executable(grayscott main.cpp gray_scott.cpp settings.cpp)
target_include_directories(grayscott PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_compile_options(grayscott PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(grayscott ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} adios2)
