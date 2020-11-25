add_library(TrainingLoop
  LossFunctions.swift
  Metrics.swift
  TrainingLoop.swift
  Callbacks/StatisticsRecorder.swift
  Callbacks/ProgressPrinter.swift
  Callbacks/CSVLogger.swift
  Callbacks/LearningRateScheduler.swift)
target_link_libraries(TrainingLoop PUBLIC
  ModelSupport)
set_target_properties(TrainingLoop PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})


install(TARGETS TrainingLoop
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)
get_swift_host_arch(swift_arch)
install(FILES
  $<TARGET_PROPERTY:TrainingLoop,Swift_MODULE_DIRECTORY>/TrainingLoop.swiftdoc
  $<TARGET_PROPERTY:TrainingLoop,Swift_MODULE_DIRECTORY>/TrainingLoop.swiftmodule
  DESTINATION lib/swift$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:_static>/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>/${swift_arch})
