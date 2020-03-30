add_executable(MobileNet-Imagenette
  main.swift)
target_link_libraries(MobileNet-Imagenette PRIVATE
  ImageClassificationModels
  Datasets)


install(TARGETS MobileNet-Imagenette
  DESTINATION bin)
