cmake_minimum_required(VERSION 3.6)
project(ED_Lib)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(EDLib STATIC
        ED.cpp
        ED.h
        EDCircles.cpp
        EDCircles.h
        EDColor.cpp
        EDColor.h
        EDLib.h
        EDLines.cpp
        EDLines.h
        EDPF.cpp
        EDPF.h
        NFA.cpp
        NFA.h)

add_compile_options(
        -O3
        -Wshadow
        -g
)

#add_executable(test_it test_it.cpp)
#target_link_libraries(test_it ${OpenCV_LIBS} EDLib)

#add_executable(test_ED test_ED.cpp)
#target_link_libraries(test_ED ${OpenCV_LIBS} EDLib)

add_executable(create_video create_video.cpp)
target_link_libraries(create_video ${OpenCV_LIBS} EDLib)

add_executable(detect_circles detect_circles.cpp)
target_link_libraries(detect_circles ${OpenCV_LIBS} EDLib)
