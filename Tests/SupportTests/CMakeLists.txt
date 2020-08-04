add_library(SupportTests
  ImageTests.swift
  SupportTests.swift
  WordSegSupportTests.swift
  AnyLayerTests.swift
  XCTestManifests.swift)
set_target_properties(SupportTests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:ModelTests>
  LIBRARY_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:ModelTests>)
target_link_libraries(SupportTests PUBLIC
  ModelSupport)
