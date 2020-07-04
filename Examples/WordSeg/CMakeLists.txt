add_executable(WordSeg
  main.swift
  WordSegCommand.swift
  WordSegSettings.swift)
target_link_libraries(WordSeg PRIVATE
  TextModels
  Datasets)


install(TARGETS WordSeg
  DESTINATION bin)
