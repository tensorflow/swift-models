add_executable(GPT2-WikiText2
  main.swift)
target_link_libraries(GPT2-WikiText2 PRIVATE
  Datasets
  TextModels
  TrainingLoop)


install(TARGETS GPT2-WikiText2
  DESTINATION bin)
