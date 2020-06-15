add_executable(MobileNetV1-Imagenette
  main.swift)
target_link_libraries(MobileNetV1-Imagenette PRIVATE
  Datasets
  ImageClassificationModels
  TrainingLoop)


install(TARGETS MobileNetV1-Imagenette
  DESTINATION bin)
