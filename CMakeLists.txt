cmake_minimum_required(VERSION 3.21)
project(msplat LANGUAGES CXX)

# Apple builds use the Metal backend; elsewhere the CPU backend (core/cpu) implements
# the same interface (core/metal/bindings.h), so training runs on Linux too.
if(APPLE)
    enable_language(OBJCXX)
    set(MSPLAT_BACKEND "metal")
else()
    set(MSPLAT_BACKEND "cpu")
endif()
message(STATUS "msplat backend: ${MSPLAT_BACKEND}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Version
file(READ "VERSION" APP_VERSION)
string(STRIP "${APP_VERSION}" APP_VERSION)
set(GIT_REV "")
execute_process(COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE GIT_REV ERROR_QUIET)
string(REGEX REPLACE "\n$" "" GIT_REV "${GIT_REV}")
if(NOT "${GIT_REV}" STREQUAL "")
    set(DAPP_VERSION "${APP_VERSION} (git ${GIT_REV})")
    set(DAPP_REVISION "${GIT_REV}")
else()
    set(DAPP_VERSION "${APP_VERSION}")
    set(DAPP_REVISION "dev")
endif()
message(STATUS "msplat ${DAPP_VERSION}")

add_compile_options("-DAPP_VERSION=\"${DAPP_VERSION}\"")
add_compile_options("-DAPP_REVISION=\"${DAPP_REVISION}\"")

# --- Dependencies ---

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip)
FetchContent_Declare(cli11
    URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.4.2.zip)
FetchContent_MakeAvailable(nlohmann_json cli11)

if(MSPLAT_BACKEND STREQUAL "metal")

# Metal frameworks
find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)

find_library(IMAGEIO_FRAMEWORK ImageIO REQUIRED)
find_library(COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)

# --- Compile Metal shaders ---

set(METAL_SOURCE ${CMAKE_SOURCE_DIR}/core/metal/msplat_metal.metal)
# -frecord-sources embeds the shader source in the metallib and -gline-tables-only
# emits line tables. Xcode needs both to show shader source and per-line cost in a
# .gputrace; without them a capture has kernel names and nothing else. Neither
# affects codegen, so -O3 still applies.
add_custom_command(
    OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
    COMMAND xcrun -sdk macosx metal -O3 -frecord-sources -gline-tables-only -c
        ${METAL_SOURCE}
        -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/msplat_metal.air
    COMMAND xcrun -sdk macosx metallib
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/msplat_metal.air
        -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
    COMMAND rm -f ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/msplat_metal.air
    DEPENDS ${METAL_SOURCE}
    COMMENT "Compiling Metal kernels → default.metallib (sources embedded for GPU capture)"
)
add_custom_target(metallib ALL
    DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib)

set(MSPLAT_BACKEND_SOURCES
    core/metal/msplat_metal.mm
    core/src/msplat_api.mm
    core/src/loaders/image_io.cpp
)

else()

# CPU backend: multithreaded C++ ports of the Metal kernels, stb for image I/O.
# MSPLAT_CPU_NATIVE tunes the build for this machine (not for redistributable wheels).
option(MSPLAT_CPU_NATIVE "Optimize the CPU backend for the build machine (-march=native)" ON)
find_package(Threads REQUIRED)
set(MSPLAT_BACKEND_SOURCES
    core/cpu/msplat_cpu.cpp
    core/src/loaders/image_io_stb.cpp
)

endif()

# --- Core static library ---

add_library(msplat_core STATIC
    ${MSPLAT_BACKEND_SOURCES}
    core/src/model.cpp
    core/src/input_data.cpp
    core/src/image_sizing.cpp
    core/src/preflight.cpp
    core/src/priors.cpp
    core/src/loaders/load_colmap.cpp
    core/src/loaders/load_nerfstudio.cpp
    core/src/loaders/load_polycam.cpp
    core/src/loaders/load_ply.cpp
    core/src/loaders/save_gaussians.cpp
    core/src/ssim.cpp
)
target_include_directories(msplat_core PUBLIC
    ${CMAKE_SOURCE_DIR}/core/include
    ${CMAKE_SOURCE_DIR}/core/metal
)
target_link_libraries(msplat_core PUBLIC nlohmann_json::nlohmann_json)
if(MSPLAT_BACKEND STREQUAL "metal")
    target_link_libraries(msplat_core PUBLIC
        ${FOUNDATION_LIBRARY}
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
        ${MPS_FRAMEWORK}
        ${IMAGEIO_FRAMEWORK}
        ${COREGRAPHICS_FRAMEWORK}
    )
    target_compile_definitions(msplat_core PRIVATE USE_MPS)
    add_dependencies(msplat_core metallib)
else()
    target_include_directories(msplat_core PRIVATE ${CMAKE_SOURCE_DIR}/core/cpu)
    target_include_directories(msplat_core SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/third_party/stb)
    target_link_libraries(msplat_core PUBLIC Threads::Threads)
    if(MSPLAT_CPU_NATIVE)
        target_compile_options(msplat_core PRIVATE -march=native)
    endif()
    set_target_properties(msplat_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# --- CLI binary ---

add_executable(msplat cli/msplat.cpp)
target_link_libraries(msplat PRIVATE
    msplat_core
    CLI11::CLI11
    pthread
)
target_compile_definitions(msplat PRIVATE USE_MPS)
install(TARGETS msplat DESTINATION bin)

# --- Tests (optional): gradient check of the training step against finite differences ---

option(MSPLAT_BUILD_TESTS "Build the C++ gradient check (ctest)" OFF)

if(MSPLAT_BUILD_TESTS)
    enable_testing()
    add_executable(msplat_gradcheck tests/gradcheck.cpp)
    target_link_libraries(msplat_gradcheck PRIVATE msplat_core)
    add_test(NAME gradcheck COMMAND msplat_gradcheck WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

# --- Python bindings (optional, enabled when building via scikit-build-core) ---

option(MSPLAT_BUILD_PYTHON "Build nanobind Python bindings" OFF)

if(MSPLAT_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(_core python/bindings.cpp)
    target_link_libraries(_core PRIVATE
        msplat_core
        pthread
    )
    target_compile_definitions(_core PRIVATE USE_MPS)

    install(TARGETS _core LIBRARY DESTINATION msplat)
    if(MSPLAT_BACKEND STREQUAL "metal")
        install(FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib DESTINATION msplat)
    endif()
endif()
