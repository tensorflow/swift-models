add_library(FastStyleTransferTests
  FastStyleTransferTests.swift
  XCTestManifests.swift)
set_target_properties(FastStyleTransferTests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:ModelTests>
  LIBRARY_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:ModelTests>)
target_link_libraries(FastStyleTransferTests PUBLIC
  FastStyleTransfer)
