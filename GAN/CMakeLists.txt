add_executable(GAN
  main.swift)
target_link_libraries(GAN PRIVATE
  Datasets
  ModelSupport)


install(TARGETS GAN
  DESTINATION bin)
