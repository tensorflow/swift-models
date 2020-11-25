add_executable(NeuMF-MovieLens
  main.swift)
target_link_libraries(NeuMF-MovieLens PRIVATE
  RecommendationModels
  Datasets)

install(TARGETS NeuMF-MovieLens
  DESTINATION bin)
