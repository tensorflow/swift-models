add_executable(Custom-CIFAR10
  Models.swift
  main.swift)
target_link_libraries(Custom-CIFAR10 PRIVATE
  ImageClassificationModels
  Datasets)


install(TARGETS Custom-CIFAR10
  DESTINATION bin)
