add_library(JPEGImageOperations
  JPEGImageOperations.swift)

set_target_properties(JPEGImageOperations PROPERTIES
  Swift_LANGUAGE_VERSION 4.2)

target_include_directories(JPEGImageOperations PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${SOURCE_DIR}
  ${BINARY_DIR})

target_link_libraries(JPEGImageOperations PRIVATE
 turbojpeg)

install(TARGETS JPEGImageOperations
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  COMPONENT lib)
  