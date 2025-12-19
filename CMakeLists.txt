cmake_minimum_required(VERSION 3.15)
project(games)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)
add_subdirectory(src)

add_executable(tic_tac_toe src/main.cpp)
# target_include_directories(tic_tac_toe PUBLIC src)
target_link_libraries(tic_tac_toe PUBLIC games)
