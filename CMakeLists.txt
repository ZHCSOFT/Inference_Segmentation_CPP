cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)
MESSAGE(STATUS "Building Measurement Algorithm Unit Test")

if(UNIX)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -ldl")    
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage -fPIC -O0")
endif()

find_package(PkgConfig REQUIRED)
find_package(Qt6 COMPONENTS Core Quick REQUIRED)

set(CMAKE_PREFIX_PATH "/usr/local/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
if(NOT Torch_FOUND)
  message(FATAL_ERROR "Failed to find Torch")
endif()
include_directories(${TORCH_INCLUDE_DIRS})
link_directories(${TORCH_LIBRARY_DIRS})

set(CMAKE_MODULE_PATH "/usr/local/runtime/cmake")
find_package(InferenceEngine REQUIRED)
if(NOT InferenceEngine_FOUND)
  message(FATAL_ERROR "Failed to find InferenceEngine")
endif()

find_package(ngraph REQUIRED)
if(NOT ngraph_FOUND)
  message(FATAL_ERROR "Failed to find ngraph")
endif()

SET(target_name MeasAlgoOBAITest)
project(${target_name})

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE) 

SET(SOURCE    
  src/SegNet_openvino.cpp
  src/SegNet_libtorch.cpp
)

SET(HEADER
  include/SegNet_openvino.h
  include/SegNet_libtorch.h
)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(OpenCV REQUIRED)
# pkg_search_module(OPENCV REQUIRED opencv)
# include_directories(../include/)

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})

add_executable(${target_name} ${SOURCE} ${HEADER})

target_include_directories(${target_name}
  PUBLIC ../include/
  PUBLIC ../src/
)

set (system_libs
)

set (internal_libs
)

set (external_libs
  opencv_core
  opencv_imgproc
  Qt6::Core
  Qt6::Quick
)

target_link_libraries(${target_name} 
	# PUBLIC PackageComm_headers OBAIAlgo_headers
  PUBLIC ${InferenceEngine_LIBRARIES} ${NGRAPH_LIBRARIES} ${TORCH_LIBRARIES} ${OpenCV_LIBS}
	PUBLIC ${system_libs} ${internal_libs} ${external_libs})

set_target_properties(${target_name}
  PROPERTIES
  FOLDER "Test"
)

install(TARGETS ${target_name}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/UnitTest
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
