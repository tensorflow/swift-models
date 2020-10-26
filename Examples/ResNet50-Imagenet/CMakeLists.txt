add_executable(ResNet50-Imagenet
  main.swift)
target_link_libraries(ResNet50-Imagenet PRIVATE
  Datasets
  ImageClassificationModels
  TrainingLoop)


install(TARGETS ResNet50-Imagenet
  DESTINATION bin)
