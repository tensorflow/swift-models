add_library(TextModels
  Attention.swift
  BERT.swift
  BERTClassifier.swift
  Embeddings.swift
  Evaluation.swift
  ScheduledParameters.swift
  Transformer.swift
  Utilities.swift
  WeightDecayedAdam.swift)
set_target_properties(TextModels PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_link_libraries(TextModels PUBLIC
  Datasets)

install(TARGETS TextModels
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)
get_swift_host_arch(swift_arch)
install(FILES
  $<TARGET_PROPERTY:TextModels,Swift_MODULE_DIRECTORY>/TextModels.swiftdoc
  $<TARGET_PROPERTY:TextModels,Swift_MODULE_DIRECTORY>/TextModels.swiftmodule
  DESTINATION lib/swift$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:_static>/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>/${swift_arch})
