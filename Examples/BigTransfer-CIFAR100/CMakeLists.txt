add_executable(BigTransfer-CIFAR100
  main.swift)
target_link_libraries(BigTransfer-CIFAR100 PRIVATE
  Datasets
  ImageClassificationModels)


install(TARGETS BigTransfer-CIFAR100
  DESTINATION bin)
