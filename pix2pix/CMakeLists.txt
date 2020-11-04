add_library(Pix2Pix
  pix2pix/CheckpointWriter.swift
  pix2pix/CLI.swift
  pix2pix/Dataset.swift
  pix2pix/Discriminator.swift
  pix2pix/Generator.swift
  pix2pix/Layers.swift
  pix2pix/Utils.swift
  pix2pix/PythonCheckpointReader.swift)
  
set_target_properties(Pix2Pix PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
  
target_compile_options(Pix2Pix PRIVATE
  $<$<BOOL:${BUILD_TESTING}>:-enable-testing>)
  
add_executable(Pix2Pix
  main.swift)
  
target_link_libraries(Pix2Pix)

install(TARGETS Pix2Pix
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)

get_swift_host_arch(swift_arch)

install(FILES
  $<TARGET_PROPERTY:Pix2Pix,Swift_MODULE_DIRECTORY>/Pix2Pix.swiftdoc
  $<TARGET_PROPERTY:Pix2Pix,Swift_MODULE_DIRECTORY>/Pix2Pix.swiftmodule
  DESTINATION lib/swift$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:_static>/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>/${swift_arch})

