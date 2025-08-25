cmake_minimum_required(VERSION 3.10)
project(OCR)

include(CTest)
enable_testing()

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(ocr ocr.cpp)

# Link OpenCV
target_link_libraries(ocr ${OpenCV_LIBS})

# Add Tesseract manually for macOS
target_include_directories(ocr PRIVATE "/opt/homebrew/include")
target_link_directories(ocr PRIVATE "/opt/homebrew/lib")
target_link_libraries(ocr tesseract leptonica)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)