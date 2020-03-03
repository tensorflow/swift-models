add_library(Transformer
  Model.swift
  Operators.swift
  PythonCheckpointReader.swift)
target_link_libraries(Transformer PUBLIC
  ModelSupport)
set_target_properties(Transformer PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})

add_executable(TransformerDemo
  main.swift
  CheckpointWriter.swift
  TransformerLoop.swift)
target_link_libraries(TransformerDemo PRIVATE
  TextModels
  Transformer)

if(CMAKE_SYSTEM_NAME STREQUAL Windows)
  add_executable(TransformerUI
    UI/Windows/main.swift
    CheckpointWriter.swift
    TransformerLoop.swift)
  target_link_libraries(TransformerUI PRIVATE
    Foundation
    TextModels
    Transformer
    User32)
  add_custom_command(TARGET TransformerUI POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/UI/Windows/TransformerUI.exe.manifest $<TARGET_FILE_DIR:TransformerUI>)
endif()


install(TARGETS Transformer
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)
install(TARGETS TransformerDemo
  DESTINATION bin)
get_swift_host_arch(swift_arch)
install(FILES
  $<TARGET_PROPERTY:Transformer,Swift_MODULE_DIRECTORY>/Transformer.swiftdoc
  $<TARGET_PROPERTY:Transformer,Swift_MODULE_DIRECTORY>/Transformer.swiftmodule
  DESTINATION lib/swift$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:_static>/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>/${swift_arch})
