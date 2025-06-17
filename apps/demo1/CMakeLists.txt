add_executable(demo1 App.cpp main.cpp)
target_link_libraries(demo1 renderer)
# how to get include dirs for renderer?

# target_include_directories(demo1 PRIVATE
# ${CMAKE_SOURCE_DIR}/src
# ${CMAKE_SOURCE_DIR}/resources
# ${CMAKE_SOURCE_DIR}/third_party
# )

