add_library(ImageClassificationTests
  Inference.swift
  XCTestManifests.swift)
set_target_properties(ImageClassificationTests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:ModelTests>
  LIBRARY_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:ModelTests>)
target_link_libraries(ImageClassificationTests PUBLIC
  ImageClassificationModels)
