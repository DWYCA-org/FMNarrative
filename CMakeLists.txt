cmake_minimum_required(VERSION 3.10)
project(FMNarrative VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig)

# Try to find Tesseract using pkg-config (works on Linux/macOS with proper install)
if(PkgConfig_FOUND)
    pkg_check_modules(TESSERACT tesseract)
    pkg_check_modules(LEPTONICA lept)
endif()

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Platform-specific Tesseract configuration
if(APPLE)
    # macOS with Homebrew
    if(NOT TESSERACT_FOUND)
        message(STATUS "Using macOS Homebrew paths for Tesseract")
        include_directories("/opt/homebrew/include")
        link_directories("/opt/homebrew/lib")
    else
        include_directories(${TESSERACT_INCLUDE_DIRS})
        link_directories(${TESSERACT_LIBRARY_DIRS})
    endif()
elseif(UNIX AND NOT APPLE)
    # Linux
    if(TESSERACT_FOUND)
        include_directories(${TESSERACT_INCLUDE_DIRS})
        link_directories(${TESSERACT_LIBRARY_DIRS})
    else
        # Common Linux paths
        include_directories(/usr/include /usr/local/include)
        link_directories(/usr/lib /usr/local/lib)
    endif()
elseif(WIN32)
    # Windows - users need to set TESSERACT_DIR
    if(DEFINED ENV{TESSERACT_DIR})
        message(STATUS "Using TESSERACT_DIR: $ENV{TESSERACT_DIR}")
        include_directories("$ENV{TESSERACT_DIR}/include")
        link_directories("$ENV{TESSERACT_DIR}/lib")
    else()
        message(WARNING "TESSERACT_DIR environment variable not set. Please set it to your Tesseract installation directory.")
    endif()
endif()

# Add executable
add_executable(ocr ocr.cpp)

# Link OpenCV
target_link_libraries(ocr ${OpenCV_LIBS})

# Link Tesseract and Leptonica
if(WIN32)
    # Windows typically uses different library names
    target_link_libraries(ocr tesseract50 leptonica-1.82.0)
else()
    # macOS and Linux
    target_link_libraries(ocr tesseract leptonica)
endif()

# Optional: Enable testing
include(CTest)
enable_testing()

# Optional: CPack for package generation
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)