cmake_minimum_required(VERSION 3.22.1)

project(haibingtech_kws)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Run export/download_ort_mobile.sh once before building with Gradle.
set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/onnxruntime"
    CACHE PATH "Root of downloaded ONNX Runtime Mobile headers and libraries")

if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ONNX Runtime Mobile headers not found at ${ORT_ROOT}/include. "
        "Run: bash export/download_ort_mobile.sh")
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_ROOT}/lib/${ANDROID_ABI}/libonnxruntime.so")

add_library(kws SHARED
    jni_bridge.cc
    engine/json.cc
    engine/runtime_config.cc
    engine/ctc_beam_search.cc
    engine/frontend/fft.cc
    engine/frontend/fbank.cc
    engine/frontend/feature_pipeline.cc
    engine/kws_engine.cc)

target_include_directories(kws PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${ORT_ROOT}/include")

target_link_libraries(kws
    onnxruntime
    android
    log)
