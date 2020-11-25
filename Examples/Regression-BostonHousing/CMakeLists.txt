add_executable(Regression-BostonHousing
  main.swift)
target_link_libraries(Regression-BostonHousing PRIVATE
  ImageClassificationModels
  Datasets)


install(TARGETS Regression-BostonHousing
  DESTINATION bin)
