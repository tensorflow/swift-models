add_library(TextTests
  Inference.swift
  WordSegmentationTests/ExampleData.swift
  WordSegmentationTests/ProbeLayers.swift
  WordSegmentationTests/Parameters.swift
  WordSegmentationTests/SemiRing.swift
  XCTestManifests.swift)
set_target_properties(TextTests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:ModelTests>
  LIBRARY_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:ModelTests>)
target_link_libraries(TextTests PUBLIC
  TextModels)
