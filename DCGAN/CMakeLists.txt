add_executable(DCGAN
  main.swift)
target_link_libraries(DCGAN PRIVATE
  Datasets
  ModelSupport)


install(TARGETS DCGAN
  DESTINATION bin)
