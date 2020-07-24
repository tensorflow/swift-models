add_library(JPEG
  JPEG.swift)

target_link_libraries(JPEG PRIVATE
 ImageOps)

set_target_properties(JPEG PROPERTIES
  Swift_LANGUAGE_VERSION 4.2)

install(TARGETS JPEG
  DESTINATION bin
  COMPONENT exe)

