cmake_minimum_required(VERSION 2.8.12)
project(pyRpp)

# Project version #########################################################
set(VERSION_MAJOR 0)
set(VERSION_MINOR 1)
set(VERSION_PATCH 0)

# Verbose mode ############################################################
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Compile Commands ########################################################
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Sub directories #########################################################
add_subdirectory(rpp)

# Target ##################################################################
set(THIRD_PARTY 3rd)

find_package(OpenCV REQUIRED COMPONENTS core)

add_subdirectory(${THIRD_PARTY}/pybind11)
pybind11_add_module(${CMAKE_PROJECT_NAME} 
    pybind11_opencv.h
    pyRpp.cpp)

target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC
    rpp_lib)

# Build-Dir Install ########################################################
install(TARGETS ${CMAKE_PROJECT_NAME} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/python)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/python)

# Install ##################################################################
set(INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/INSTALL)
install(TARGETS ${CMAKE_PROJECT_NAME}
  RUNTIME DESTINATION ${INSTALL_DIR}/bin
  LIBRARY DESTINATION ${INSTALL_DIR}/lib
  ARCHIVE DESTINATION ${INSTALL_DIR}/lib)
install(TARGETS ${CMAKE_PROJECT_NAME} DESTINATION ${INSTALL_DIR}/lib)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test.py DESTINATION ${INSTALL_DIR}/bin)

# Package Generator  #######################################################
set(CPACK_PACKAGE_DESCRIPTION "pyRpp for Robust Planar Pose")
set(CPACK_PACKAGE_VERSION_MAJOR "${VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${VERSION_PATCH}")
set(CPACK_PACKAGE_VENDOR "Rishabh")
set(CPACK_PACKAGE_CONTACT "rishabhbattulwar@gmail.com")
set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
include (CPack)