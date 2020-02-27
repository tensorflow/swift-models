add_library(DatasetsTests
  BostonHousing/BostonHousingTests.swift
  CIFAR10/CIFAR10Tests.swift
  Imagenette/ImagenetteTests.swift
  MNIST/MNISTTests.swift
  XCTestManifests.swift)
set_target_properties(DatasetsTests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:ModelTests>
  LIBRARY_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:ModelTests>)
target_link_libraries(DatasetsTests PUBLIC
  Datasets)
