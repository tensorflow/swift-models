add_library(ModelSupport
  Checkpoints/CheckpointReader.swift
  FileManagement.swift
  Image.swift
  Stderr.swift)
set_target_properties(ModelSupport PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})


install(TARGETS ModelSupport
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)
get_swift_host_arch(swift_arch)
install(FILES
  $<TARGET_PROPERTY:ModelSupport,Swift_MODULE_DIRECTORY>/ModelSupport.swiftdoc
  $<TARGET_PROPERTY:ModelSupport,Swift_MODULE_DIRECTORY>/ModelSupport.swiftmodule
  DESTINATION lib/swift$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:_static>/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>/${swift_arch})
