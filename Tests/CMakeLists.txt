add_subdirectory(CheckpointTests)
add_subdirectory(DatasetsTests)
add_subdirectory(FastStyleTransferTests)
add_subdirectory(ImageClassificationTests)
add_subdirectory(MiniGoTests)
add_subdirectory(SupportTests)
add_subdirectory(TextTests)

add_executable(ModelTests
  LinuxMain.swift)
target_link_libraries(ModelTests PRIVATE
  CheckpointTests
  DatasetsTests
  FastStyleTransferTests
  ImageClassificationTests
  MiniGoTests
  SupportTests
  TextTests
  XCTest)

add_test(NAME ModelTests
  COMMAND ModelTests)
