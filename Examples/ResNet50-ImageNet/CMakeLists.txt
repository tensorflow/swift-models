add_executable(ResNet50-ImageNet
  main.swift)
target_link_libraries(ResNet50-ImageNet PRIVATE
  Datasets
  ImageClassificationModels
  TrainingLoop)


install(TARGETS ResNet50-ImageNet
  DESTINATION bin)
