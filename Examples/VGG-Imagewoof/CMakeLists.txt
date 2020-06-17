add_executable(VGG-Imagewoof
  main.swift)
target_link_libraries(VGG-Imagewoof PRIVATE
  Datasets
  ImageClassificationModels
  TrainingLoop)


install(TARGETS VGG-Imagewoof
  DESTINATION bin)
