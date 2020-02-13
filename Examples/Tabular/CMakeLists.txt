add_executable(Tabular
  main.swift)
target_link_libraries(Tabular PRIVATE
  ImageClassificationModels
  Datasets)


install(TARGETS Tabular
  DESTINATION bin)
