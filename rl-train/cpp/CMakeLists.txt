cmake_minimum_required(VERSION 3.12)
project(rl_game)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)

add_library(rl_game MODULE
    environment.cpp
    simulator.cpp
    observation.cpp
    reward.cpp
    bindings.cpp
)

target_link_libraries(rl_game PRIVATE pybind11::module)