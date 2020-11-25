add_executable(BERT-CoLA
  main.swift)
target_link_libraries(BERT-CoLA PRIVATE
  TextModels
  Datasets
  TrainingLoop)


install(TARGETS BERT-CoLA
  DESTINATION bin)
