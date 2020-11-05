add_library(pix2pix
  pix2pix/CheckpointWriter.swift
  pix2pix/CLI.swift
  pix2pix/Dataset.swift
  pix2pix/Discriminator.swift
  pix2pix/Generator.swift
  pix2pix/Layers.swift
  pix2pix/Utils.swift
  pix2pix/PythonCheckpointReader.swift)
  
set_target_properties(pix2pix PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
  
target_compile_options(pix2pix PRIVATE
  $<$<BOOL:${BUILD_TESTING}>:-enable-testing>)
  
target_link_libraries(pix2pix PRIVATE
  pix2pix)

install(TARGETS pix2pix
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)

get_swift_host_arch(swift_arch)

install(FILES
  $<TARGET_PROPERTY:pix2ix,Swift_MODULE_DIRECTORY>/pix2pix.swiftdoc
  $<TARGET_PROPERTY:pix2pix,Swift_MODULE_DIRECTORY>/pix2pix.swiftmodule
  DESTINATION lib/swift$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:_static>/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>/${swift_arch})

