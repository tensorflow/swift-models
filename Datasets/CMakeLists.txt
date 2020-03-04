add_library(Datasets
  CIFAR10/CIFAR10.swift
  DatasetUtilities.swift
  CoLA/Batchable.swift
  CoLA/CoLA.swift
  CoLA/DataUtilities.swift
  ImageClassificationDataset.swift
  Imagenette/Imagenette.swift
  Imagenette/Imagewoof.swift
  LabeledExample.swift
  LanguageModelDataset.swift
  MNIST/MNIST.swift
  MNIST/FashionMNIST.swift
  BostonHousing/BostonHousing.swift)
target_link_libraries(Datasets PUBLIC
  Batcher
  ModelSupport)
set_target_properties(Datasets PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})


install(TARGETS Datasets
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)
get_swift_host_arch(swift_arch)
install(FILES
  $<TARGET_PROPERTY:Datasets,Swift_MODULE_DIRECTORY>/Datasets.swiftdoc
  $<TARGET_PROPERTY:Datasets,Swift_MODULE_DIRECTORY>/Datasets.swiftmodule
  DESTINATION lib/swift$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:_static>/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>/${swift_arch})
