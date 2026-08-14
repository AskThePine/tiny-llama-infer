cmake_minimum_required(VERSION 3.15)

project(llama-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(llama_cpp
    src/config.cpp
    src/generator.cpp
    src/loader.cpp
    src/main.cpp
    src/mapped_file.cpp
    src/model.cpp
    src/sampler.cpp
    src/tensor.cpp
    src/tokenizer.cpp
)

target_include_directories(llama_cpp PRIVATE ${CMAKE_SOURCE_DIR}/include)

find_package(OpenMP REQUIRED)
target_link_libraries(llama_cpp PRIVATE OpenMP::OpenMP_CXX)

find_package(ICU REQUIRED COMPONENTS uc)
target_link_libraries(llama_cpp PRIVATE ICU::uc)
