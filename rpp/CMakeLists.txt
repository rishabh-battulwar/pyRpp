project(rpp_lib)

find_package(OpenCV REQUIRED COMPONENTS core)

add_library(rpp_lib)
target_sources(rpp_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/RPP.h
    ${CMAKE_CURRENT_SOURCE_DIR}/RPP.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Rpoly.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Rpoly.cpp)

target_include_directories(rpp_lib PRIVATE 
    ${OpenCV_INCLUDE_DIRS})

target_link_libraries(rpp_lib PUBLIC ${OpenCV_LIBS})
