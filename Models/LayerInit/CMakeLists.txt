add_library(LayerInit
  AutoBatchNorm.swift
  AutoConv.swift
  AutoDense.swift
  AutoFlatten.swift
  AutoFunction.swift
  AutoLayer.swift
  AutoLayerKey.swift
  AutoPool.swift
  AutoReuseLayer.swift
  AutoSequenced.swift
  AutoSplitMerge.swift)
set_target_properties(LayerInit PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_compile_options(LayerInit PRIVATE
  $<$<BOOL:${BUILD_TESTING}>:-enable-testing>)


install(TARGETS LayerInit
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)
get_swift_host_arch(swift_arch)
install(FILES
  $<TARGET_PROPERTY:LayerInit,Swift_MODULE_DIRECTORY>/LayerInit.swiftdoc
  $<TARGET_PROPERTY:LayerInit,Swift_MODULE_DIRECTORY>/LayerInit.swiftmodule
  DESTINATION lib/swift$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:_static>/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>/${swift_arch})
