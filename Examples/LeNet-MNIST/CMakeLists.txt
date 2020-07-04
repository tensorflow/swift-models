add_executable(LeNet-MNIST
  main.swift)
target_link_libraries(LeNet-MNIST PRIVATE
  Datasets
  ImageClassificationModels
  TrainingLoop)


install(TARGETS LeNet-MNIST
  DESTINATION bin)
