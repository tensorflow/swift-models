add_executable(ResNet-CIFAR10
  main.swift)
target_link_libraries(ResNet-CIFAR10 PRIVATE
  ImageClassificationModels
  Datasets
  TrainingLoop)


install(TARGETS ResNet-CIFAR10
  DESTINATION bin)
