add_library(TextTests
  Inference.swift
  XCTestManifests.swift)
set_target_properties(TextTests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:ModelTests>
  LIBRARY_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:ModelTests>)
target_link_libraries(TextTests PUBLIC
  TextModels)
