add_library(FastStyleTransfer
  Layers/Helpers.swift
  Layers/Normalization.swift

  Models/TransformerNet.swift

  Utility/ImportableLayer.swift
  Utility/Resize.swift)
target_link_libraries(FastStyleTransfer PUBLIC
  ModelSupport)
set_target_properties(FastStyleTransfer PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})

add_executable(FastStyleTransferDemo
  Demo/ColabDemo.ipynb
  Demo/Helpers.swift
  Demo/main.swift)
target_link_libraries(FastStyleTransferDemo PRIVATE
  FastStyleTransfer)


install(TARGETS FastStyleTransfer
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)
install(TARGETS FastStyleTransferDemo
  DESTINATION bin)
get_swift_host_arch(swift_arch)
install(FILES
  $<TARGET_PROPERTY:FastStyleTransfer,Swift_MODULE_DIRECTORY>/FastStyleTransfer.swiftdoc
  $<TARGET_PROPERTY:FastStyleTransfer,Swift_MODULE_DIRECTORY>/FastStyleTransfer.swiftmodule
  DESTINATION lib/swift$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:_static>/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>/${swift_arch})
