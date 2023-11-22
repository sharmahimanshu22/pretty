cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(myapp DESCRIPTION "BVHTree" LANGUAGES CXX CUDA)

enable_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE main.cpp src/CudaDeviceManagerInternal.cu src/mortoncoder.cu)

set_target_properties(myapp PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${PROJECT_NAME}
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
)



add_subdirectory(deps/cap)
target_link_libraries(${PROJECT_NAME} PUBLIC cap)

add_subdirectory(deps/tinyobjloader)
target_link_libraries(${PROJECT_NAME} PUBLIC tinyobjloader)


#find_package(cap CONFIG REQUIRED)
#target_link_libraries(myapp cap::cap)


#find_package(CUDAToolkit REQUIRED)
#target_link_libraries(myapp CUDA::cudart)
















#add_subdirectory(deps/fmt)
#target_link_libraries(${PROJECT_NAME} PUBLIC fmt)

#set(OpenCV_DIR "/hpc/users/sharmh15/bin/opencv17/lib64/cmake/opencv4")
#find_package( OpenCV REQUIRED )
#target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} )
#target_link_libraries( ${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} )


#include(cmake/CPM.cmake)

# This will automatically clone CCCL from GitHub and make the exported cmake targets available
#CPMAddPackage(
#    NAME CCCL
#    GITHUB_REPOSITORY nvidia/cccl
#    GIT_TAG main # Fetches the latest commit on the main branch
#)
#target_link_libraries(${PROJECT_NAME} PRIVATE CCCL::CCCL)
