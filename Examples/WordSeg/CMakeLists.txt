add_executable(WordSeg
  main.swift)
target_link_libraries(WordSeg PRIVATE
  TextModels
  Datasets)


install(TARGETS WordSeg
  DESTINATION bin)
