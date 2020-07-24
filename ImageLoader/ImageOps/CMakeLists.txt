
add_library(ImageOps 
  SHARED
  ImageOps.swift)

set_target_properties(ImageOps PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})

target_include_directories(ImageOps PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${SOURCE_DIR}
   ${BINARY_DIR})
   
target_link_libraries(ImageOps 
turbojpeg)

install(TARGETS ImageOps
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  )