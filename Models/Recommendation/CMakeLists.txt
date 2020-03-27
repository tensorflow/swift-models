add_library(RecommendationModels
  DLRM.swift
  MLP.swift)
set_target_properties(RecommendationModels PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_compile_options(RecommendationModels PRIVATE
  $<$<BOOL:${BUILD_TESTING}>:-enable-testing>)


install(TARGETS RecommendationModels
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)
get_swift_host_arch(swift_arch)
install(FILES
  $<TARGET_PROPERTY:RecommendationModels,Swift_MODULE_DIRECTORY>/RecommendationModels.swiftdoc
  $<TARGET_PROPERTY:RecommendationModels,Swift_MODULE_DIRECTORY>/RecommendationModels.swiftmodule
  DESTINATION lib/swift$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:_static>/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>/${swift_arch})
