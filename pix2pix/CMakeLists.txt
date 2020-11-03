add_library(Pix2PixLib
  Pix2Pix/CheckpointWriter.swift
  Pix2Pix/CLI.swift
  Pix2Pix/Dataset.swift
  Pix2Pix/Discriminator.swift
  Pix2Pix/Generator.swift
  Pix2Pix/Layers.swift
  Pix2Pix/Utils.swift
  Pix2Pix/PythonCheckpointReader.swift)
  
set_target_properties(Pix2PixLib PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
  
target_compile_options(Pix2PixLib PRIVATE
  $<$<BOOL:${BUILD_TESTING}>:-enable-testing>)
  
add_executable(Pix2Pix
  main.swift)
  
target_link_libraries(Pix2PixLib)

install(TARGETS Pix2PixLib
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)

get_swift_host_arch(swift_arch)

install(FILES
  $<TARGET_PROPERTY:Pix2PixLib,Swift_MODULE_DIRECTORY>/Pix2PixLib.swiftdoc
  $<TARGET_PROPERTY:Pix2PixLib,Swift_MODULE_DIRECTORY>/Pix2PixLib.swiftmodule
  DESTINATION lib/swift$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:_static>/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>/${swift_arch})

