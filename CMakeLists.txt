cmake_minimum_required(VERSION 3.16)
project(Models
  LANGUAGES Swift)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

set(CMAKE_Swift_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/swift)

include(SwiftSupport)

add_subdirectory(Support)
add_subdirectory(Datasets)
add_subdirectory(Models)
add_subdirectory(MiniGo)
add_subdirectory(GAN)
add_subdirectory(DCGAN)
add_subdirectory(FastStyleTransfer)
add_subdirectory(Transformer)
add_subdirectory(Examples)
