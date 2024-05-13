cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

option(BUILD_EXTENSIONS "Build C++ extensions" OFF)

if(BUILD_EXTENSIONS)
  set(PYBIND11_NEWPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(_grouper lsmtool/operations/_grouper.cpp)
  install(TARGETS _grouper LIBRARY DESTINATION lsmtool/operations)
endif()
