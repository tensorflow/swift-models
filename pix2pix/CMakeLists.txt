add_library(pix2pix
  CheckpointReader.swift
  CheckpointWriter.swift
  CLI.swift
  Dataset.swift
  Discriminator.swift
  Generator.swift
  Layers.swift
  Utils.swift)
  
set_target_properties(pix2pix PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
  
target_compile_options(pix2pix PRIVATE
  $<$<BOOL:${BUILD_TESTING}>:-enable-testing>)
  
target_link_libraries(pix2pix PRIVATE
  ArgumentParser
  Checkpoints
  Datasets
  ModelSupport)

add_executable(pix2pixDemo
  main.swift)
target_link_libraries(pix2pixDemo PRIVATE
  pix2pix)
  
install(TARGETS pix2pix
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)
install(TARGETS pix2pixDemo
  DESTINATION bin)
get_swift_host_arch(swift_arch)
install(FILES
  $<TARGET_PROPERTY:pix2pix,Swift_MODULE_DIRECTORY>/pix2pix.swiftdoc
  $<TARGET_PROPERTY:pix2pix,Swift_MODULE_DIRECTORY>/pix2pix.swiftmodule
  DESTINATION lib/swift$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:_static>/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>/${swift_arch})

