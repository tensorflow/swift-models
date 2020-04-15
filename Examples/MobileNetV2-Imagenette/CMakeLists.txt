add_executable(MobileNetV2-Imagenette
  main.swift)
target_link_libraries(MobileNetV2-Imagenette PRIVATE
  ImageClassificationModels
  Datasets)


install(TARGETS MobileNetV2-Imagenette
  DESTINATION bin)
