add_library(ImageClassificationModels
  DenseNet121.swift
  LeNet-5.swift
  MobileNet.swift
  MobileNetV2.swift
  ResNet.swift
  ResNetV2.swift
  SqueezeNet.swift
  VGG.swift
  WideResNet.swift)
set_target_properties(ImageClassificationModels PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_compile_options(ImageClassificationModels PRIVATE
  $<$<BOOL:${BUILD_TESTING}>:-enable-testing>)


install(TARGETS ImageClassificationModels
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)
get_swift_host_arch(swift_arch)
install(FILES
  $<TARGET_PROPERTY:ImageClassificationModels,Swift_MODULE_DIRECTORY>/ImageClassificationModels.swiftdoc
  $<TARGET_PROPERTY:ImageClassificationModels,Swift_MODULE_DIRECTORY>/ImageClassificationModels.swiftmodule
  DESTINATION lib/swift$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:_static>/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>/${swift_arch})
