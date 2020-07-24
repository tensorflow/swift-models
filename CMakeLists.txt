cmake_minimum_required(VERSION 3.16)
project(Models
  LANGUAGES C Swift)

if(CMAKE_VERSION VERSION_LESS 3.17)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL Windows)
    set(CMAKE_EXECUTABLE_RUNTIME_Swift_FLAG "-Xlinker -rpath -Xlinker ")
    set(CMAKE_EXECUTABLE_RUNTIME_Swift_FLAG_SEP "")
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_Swift_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/swift)
file(MAKE_DIRECTORY ${CMAKE_Swift_MODULE_DIRECTORY})

include(CTest)
include(ExternalProject)
include(SwiftSupport)

# Dependencies
if(CMAKE_SYSTEM_NAME STREQUAL Windows)
  ExternalProject_Add(swift-win32
    GIT_REPOSITORY git://github.com/compnerd/swift-win32.git
    GIT_TAG d4613fb66fcd7814549b5f13cc30ab31bfd95b97
    CMAKE_ARGS
      -DBUILD_SHARED_LIBS=YES
      -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
      -DCMAKE_Swift_COMPILER=${CMAKE_Swift_COMPILER}
      -DCMAKE_Swift_FLAGS=${CMAKE_Swift_FLAGS}
    INSTALL_COMMAND
      COMMAND
        ${CMAKE_COMMAND} -E copy_if_different <BINARY_DIR>/bin/${CMAKE_SHARED_LIBRARY_PREFIX}SwiftWin32${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/
      COMMAND
        ${CMAKE_COMMAND} -E copy_if_different <BINARY_DIR>/lib/${CMAKE_IMPORT_LIBRARY_PREFIX}SwiftWin32${CMAKE_IMPORT_LIBRARY_SUFFIX} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/
      COMMAND
        ${CMAKE_COMMAND} -E copy_if_different <BINARY_DIR>/swift/SwiftWin32.swiftmodule ${CMAKE_Swift_MODULE_DIRECTORY}/
    BUILD_BYPRODUCTS
      <BINARY_DIR>/bin/${CMAKE_SHARED_LIBRARY_PREFIX}SwiftWin32${CMAKE_SHARED_LIBRARY_SUFFIX}
      <BINARY_DIR>/lib/${CMAKE_IMPORT_LIBRARY_PREFIX}SwiftWin32${CMAKE_IMPORT_LIBRARY_SUFFIX}
      ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}SwiftWin32${CMAKE_SHARED_LIBRARY_SUFFIX}
      ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${CMAKE_IMPORT_LIBRARY_PREFIX}SwiftWin32${CMAKE_IMPORT_LIBRARY_SUFFIX}
    STEP_TARGETS install)

  add_library(SwiftWin32 SHARED IMPORTED)
  set_target_properties(SwiftWin32 PROPERTIES
    IMPORTED_LOCATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}SwiftWin32${CMAKE_SHARED_LIBRARY_SUFFIX}
    IMPORTED_IMPLIB ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${CMAKE_IMPORT_LIBRARY_PREFIX}SwiftWin32${CMAKE_IMPORT_LIBRARY_SUFFIX}
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
  add_dependencies(SwiftWin32 swift-win32-install)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL Windows)
  set(_copy_swift_protobuf_import_library
    ${CMAKE_COMMAND} -E copy_if_different <BINARY_DIR>/Sources/SwiftProtobuf/${CMAKE_IMPORT_LIBRARY_PREFIX}SwiftProtobuf${CMAKE_IMPORT_LIBRARY_SUFFIX} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/)
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
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different <BINARY_DIR>/Sources/SwiftProtobuf/${CMAKE_SHARED_LIBRARY_PREFIX}SwiftProtobuf${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different <BINARY_DIR>/swift/SwiftProtobuf.swiftmodule ${CMAKE_Swift_MODULE_DIRECTORY}/
    COMMAND
      ${_copy_swift_protobuf_import_library}
  BUILD_BYPRODUCTS
    <BINARY_DIR>/Sources/SwiftProtobuf/${CMAKE_SHARED_LIBRARY_PREFIX}SwiftProtobuf${CMAKE_SHARED_LIBRARY_SUFFIX}
    <BINARY_DIR>/Sources/SwiftProtobuf/${CMAKE_IMPORT_LIBRARY_PREFIX}SwiftProtobuf${CMAKE_IMPORT_LIBRARY_SUFFIX}
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}SwiftProtobuf${CMAKE_SHARED_LIBRARY_SUFFIX}
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_IMPORT_LIBRARY_PREFIX}SwiftProtobuf${CMAKE_IMPORT_LIBRARY_SUFFIX}
  STEP_TARGETS install)

ExternalProject_Add(libjpeg-turbo
  GIT_REPOSITORY
    git://github.com/libjpeg-turbo/libjpeg-turbo
  GIT_TAG
    master
  UPDATE_DISCONNECTED
    TRUE
  CMAKE_ARGS
    -D CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    -D ENABLE_SHARED=NO
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/ImageLoader/libjpeg-turbo/module.modulemap <SOURCE_DIR>
  BUILD_BYPRODUCTS
    <BINARY_DIR>/${CMAKE_STATIC_LIBRARY_PREFIX}turbojpeg${CMAKE_STATIC_LIBRARY_SUFFIX}
  STEP_TARGETS
    build)

ExternalProject_Get_Property(libjpeg-turbo BINARY_DIR)
ExternalProject_Get_Property(libjpeg-turbo SOURCE_DIR)

add_library(turbojpeg STATIC IMPORTED)
add_library(SwiftProtobuf SHARED IMPORTED)

set_target_properties(SwiftProtobuf PROPERTIES
  IMPORTED_LOCATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}SwiftProtobuf${CMAKE_SHARED_LIBRARY_SUFFIX}
  IMPORTED_IMPLIB ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_IMPORT_LIBRARY_PREFIX}SwiftProtobuf${CMAKE_IMPORT_LIBRARY_SUFFIX}
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
add_dependencies(SwiftProtobuf swift-protobuf-install)

set_target_properties(turbojpeg PROPERTIES
  IMPORTED_LOCATION ${BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}turbojpeg${CMAKE_STATIC_LIBRARY_SUFFIX}
  IMPORTED_INCLUDE_DIRECTORIES "${BINARY_DIR};${SOURCE_DIR}")

if(CMAKE_SYSTEM_NAME STREQUAL Windows)
  set(_copy_swift_argument_parser_import_library
    ${CMAKE_COMMAND} -E copy_if_different <BINARY_DIR>/Sources/ArgumentParser/${CMAKE_IMPORT_LIBRARY_PREFIX}ArgumentParser${CMAKE_IMPORT_LIBRARY_SUFFIX} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/)
endif()
ExternalProject_Add(swift-argument-parser
  GIT_REPOSITORY git://github.com/apple/swift-argument-parser.git
  GIT_TAG master
  CMAKE_ARGS
    -DBUILD_SHARED_LIBS=YES
    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    -DCMAKE_Swift_COMPILER=${CMAKE_Swift_COMPILER}
    -DCMAKE_Swift_FLAGS=${CMAKE_Swift_FLAGS}
    -DBUILD_EXAMPLES=NO
    -DBUILD_TESTING=NO
  INSTALL_COMMAND
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different <BINARY_DIR>/lib/${CMAKE_SHARED_LIBRARY_PREFIX}ArgumentParser${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different <BINARY_DIR>/swift/ArgumentParser.swiftmodule ${CMAKE_Swift_MODULE_DIRECTORY}/
    COMMAND
      ${_copy_swift_argument_parser_import_library}
  BUILD_BYPRODUCTS
    <BINARY_DIR>/Sources/ArgumentParser/${CMAKE_SHARED_LIBRARY_PREFIX}ArgumentParser${CMAKE_SHARED_LIBRARY_SUFFIX}
    <BINARY_DIR>/Sources/ArgumentParser/${CMAKE_IMPORT_LIBRARY_PREFIX}ArgumentParser${CMAKE_IMPORT_LIBRARY_SUFFIX}
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}ArgumentParser${CMAKE_SHARED_LIBRARY_SUFFIX}
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_IMPORT_LIBRARY_PREFIX}ArgumentParser${CMAKE_IMPORT_LIBRARY_SUFFIX}
  STEP_TARGETS install)

add_library(ArgumentParser SHARED IMPORTED)
set_target_properties(ArgumentParser PROPERTIES
  IMPORTED_LOCATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}ArgumentParser${CMAKE_SHARED_LIBRARY_SUFFIX}
  IMPORTED_IMPLIB ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_IMPORT_LIBRARY_PREFIX}ArgumentParser${CMAKE_IMPORT_LIBRARY_SUFFIX}
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
add_dependencies(ArgumentParser swift-argument-parser-install)
add_dependencies(turbojpeg libjpeg-turbo-build)

add_subdirectory(Autoencoder)
add_subdirectory(Support)
add_subdirectory(Checkpoints)
add_subdirectory(Datasets)
add_subdirectory(Models)
add_subdirectory(MiniGo)
add_subdirectory(GAN)
add_subdirectory(DCGAN)
add_subdirectory(FastStyleTransfer)
add_subdirectory(Examples)
add_subdirectory(TrainingLoop)
add_subdirectory(ImageLoader)

if(BUILD_TESTING)
  add_subdirectory(Tests)
endif()
