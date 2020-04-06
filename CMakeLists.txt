cmake_minimum_required(VERSION 3.16)
project(Models
  LANGUAGES Swift)

if(CMAKE_VERSION VERSION_LESS 3.17)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL Windows)
    set(CMAKE_EXECUTABLE_RUNTIME_Swift_FLAG "-Xlinker -rpath -Xlinker ")
    set(CMAKE_EXECUTABLE_RUNTIME_Swift_FLAG_SEP "")
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

set(CMAKE_Swift_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/swift)

include(CTest)
include(ExternalProject)
include(SwiftSupport)

# Dependencies
if(CMAKE_SYSTEM_NAME STREQUAL Windows)
  ExternalProject_Add(swift-win32
    GIT_REPOSITORY git://github.com/compnerd/swift-win32.git
    GIT_TAG 91ad76252c5ea7be992d23c8ca7710c81fc9dca8
    CMAKE_ARGS
      -DBUILD_SHARED_LIBS=YES
      -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
      -DCMAKE_Swift_COMPILER=${CMAKE_Swift_COMPILER}
      -DCMAKE_Swift_FLAGS=${CMAKE_Swift_FLAGS}
    INSTALL_COMMAND
      ""
    BUILD_BYPRODUCTS
      <BINARY_DIR>/${CMAKE_SHARED_LIBRARY_PREFIX}SwiftWin32${CMAKE_SHARED_LIBRARY_SUFFIX}
      <BINARY_DIR>/${CMAKE_IMPORT_LIBRARY_PREFIX}SwiftWin32${CMAKE_IMPORT_LIBRARY_SUFFIX}
    STEP_TARGETS build)
  ExternalProject_Get_Property(swift-win32 BINARY_DIR)

  add_library(SwiftWin32 SHARED IMPORTED)
  file(MAKE_DIRECTORY ${BINARY_DIR}/swift)
  set_target_properties(SwiftWin32 PROPERTIES
    IMPORTED_LOCATION ${BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}SwiftWin32${CMAKE_SHARED_LIBRARY_SUFFIX}
    IMPORTED_IMPLIB ${BINARY_DIR}/${CMAKE_IMPORT_LIBRARY_PREFIX}SwiftWin32${CMAKE_IMPORT_LIBRARY_SUFFIX}
    INTERFACE_INCLUDE_DIRECTORIES ${BINARY_DIR}/swift)
  add_dependencies(SwiftWin32 swift-win32-build)
endif()

ExternalProject_Add(swift-protobuf
  GIT_REPOSITORY git://github.com/apple/swift-protobuf.git
  GIT_TAG master
  CMAKE_ARGS
    -DBUILD_SHARED_LIBS=YES
    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    -DCMAKE_Swift_COMPILER=${CMAKE_Swift_COMPILER}
    -DCMAKE_Swift_FLAGS=${CMAKE_Swift_FLAGS}
  INSTALL_COMMAND
    ""
  BUILD_BYPRODUCTS
    <BINARY_DIR>/Sources/SwiftProtobuf/${CMAKE_SHARED_LIBRARY_PREFIX}SwiftProtobuf${CMAKE_SHARED_LIBRARY_SUFFIX}
    <BINARY_DIR>/Sources/SwiftProtobuf/${CMAKE_IMPORT_LIBRARY_PREFIX}SwiftProtobuf${CMAKE_IMPORT_LIBRARY_SUFFIX}
  STEP_TARGETS build)
ExternalProject_Get_Property(swift-protobuf BINARY_DIR)

add_library(SwiftProtobuf SHARED IMPORTED)
file(MAKE_DIRECTORY ${BINARY_DIR}/swift)
set_target_properties(SwiftProtobuf PROPERTIES
  IMPORTED_LOCATION ${BINARY_DIR}/Sources/SwiftProtobuf/${CMAKE_SHARED_LIBRARY_PREFIX}SwiftProtobuf${CMAKE_SHARED_LIBRARY_SUFFIX}
  IMPORTED_IMPLIB ${BINARY_DIR}/Sources/SwiftProtobuf/${CMAKE_IMPORT_LIBRARY_PREFIX}SwiftProtobuf${CMAKE_IMPORT_LIBRARY_SUFFIX}
  INTERFACE_INCLUDE_DIRECTORIES ${BINARY_DIR}/swift)
add_dependencies(SwiftProtobuf swift-protobuf-build)

add_subdirectory(Support)
add_subdirectory(Batcher)
add_subdirectory(Datasets)
add_subdirectory(Models)
add_subdirectory(MiniGo)
add_subdirectory(GAN)
add_subdirectory(DCGAN)
add_subdirectory(FastStyleTransfer)
add_subdirectory(Examples)

if(BUILD_TESTING)
  add_subdirectory(Tests)
endif()
