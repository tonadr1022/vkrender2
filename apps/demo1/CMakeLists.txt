add_executable(demo1 App.cpp main.cpp)
target_link_libraries(demo1 renderer)

target_compile_options(demo1
    PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:
    -Wall
    -Wextra
    -pedantic
    -Wno-missing-field-initializers
    -Wno-unused-result
    >
    $<$<CXX_COMPILER_ID:MSVC>:
    /W4
    /WX
    /permissive-
    >
)

# target_include_directories(demo1 PRIVATE
# ${CMAKE_SOURCE_DIR}/src
# ${CMAKE_SOURCE_DIR}/resources
# ${CMAKE_SOURCE_DIR}/third_party
# )

