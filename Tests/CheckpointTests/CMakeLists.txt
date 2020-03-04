add_library(CheckpointTests
  CheckpointIndexReaderTests.swift
  CheckpointWriterTests.swift
  SnappyDecompressionTests.swift
  XCTestManifests.swift)
set_target_properties(CheckpointTests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:ModelTests>
  LIBRARY_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:ModelTests>)
target_link_libraries(CheckpointTests PUBLIC
  ModelSupport)
