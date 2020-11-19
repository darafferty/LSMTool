cmake_minimum_required(VERSION 3.16)
project(grouper)      

set(CMAKE_CXX_STANDARD 14)

find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
find_package(OpenMP)

if(Python_FOUND)
    message(STATUS "Python Found: ${Python_EXECUTABLE}")
    message(STATUS "Python Found: ${Python_INCLUDE_DIRS}")
    message(STATUS "Python Found: ${Python_LIBRARIES}")
    message(STATUS "Python Found: ${Python_LIBRARY_DIRS}")
    include_directories(${Python_INCLUDE_DIRS})
endif()

add_library(grouper   SHARED meanshift.cpp)  
SET_TARGET_PROPERTIES(grouper   PROPERTIES PREFIX "")
if(Python_FOUND)
    target_link_libraries(grouper ${Python_LIBRARIES} OpenMP::OpenMP_CXX)                     
endif()

