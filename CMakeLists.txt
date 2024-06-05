cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


project(labo_1_221 LANGUAGES CXX)

file(GLOB_RECURSE src source/*.cpp)

add_executable(${PROJECT_NAME} ${src})

