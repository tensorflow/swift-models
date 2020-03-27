add_executable(BERT-CoLA
  CoLATraining.swift
  main.swift)
target_link_libraries(BERT-CoLA PRIVATE
  TextModels
  Datasets)


install(TARGETS BERT-CoLA
  DESTINATION bin)
