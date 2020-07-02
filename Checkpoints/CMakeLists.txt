add_library(Checkpoints
  CheckpointIndexReader.swift
  CheckpointIndexWriter.swift
  CheckpointReader.swift
  CheckpointWriter.swift
  SnappyDecompression.swift)
set_target_properties(Checkpoints PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})


install(TARGETS Checkpoints
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)
get_swift_host_arch(swift_arch)
install(FILES
  $<TARGET_PROPERTY:Checkpoints,Swift_MODULE_DIRECTORY>/Checkpoints.swiftdoc
  $<TARGET_PROPERTY:Checkpoints,Swift_MODULE_DIRECTORY>/Checkpoints.swiftmodule
  DESTINATION lib/swift$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:_static>/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>/${swift_arch})
