add_library(RecommendationModelTests
  DLRMTests.swift
  XCTestManifests.swift)
set_target_properties(RecommendationModelTests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:ModelTests>
  LIBRARY_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:ModelTests>)
target_link_libraries(RecommendationModelTests PUBLIC
  RecommendationModels)
