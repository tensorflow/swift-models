add_executable(LeNet-MNIST
  main.swift)
target_link_libraries(LeNet-MNIST PRIVATE
  ImageClassificationModels
  Datasets)


install(TARGETS LeNet-MNIST
  DESTINATION bin)
