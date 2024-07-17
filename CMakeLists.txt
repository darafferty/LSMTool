cmake_minimum_required(VERSION 3.15...3.26)

# Do not set language yet, but only when actually building extensions, to
# avoid the need for having a working compiler when it's not used.
project(${SKBUILD_PROJECT_NAME} LANGUAGES)

option(BUILD_EXTENSIONS "Build C++ extensions" OFF)

if(BUILD_EXTENSIONS)
  # Only now set the language, and trigger the search for a working compiler.
  enable_language(CXX)
  set(PYBIND11_NEWPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(_grouper lsmtool/operations/_grouper.cpp)
  install(TARGETS _grouper LIBRARY DESTINATION lsmtool/operations)
endif()
