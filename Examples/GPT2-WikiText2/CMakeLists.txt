add_executable(GPT2-WikiText2
  main.swift)
target_link_libraries(GPT2-WikiText PRIVATE
  Datasets
  TextModels)


install(TARGETS GPT2-WikiText2
  DESTINATION bin)
