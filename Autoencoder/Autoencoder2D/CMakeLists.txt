add_executable(Autoencoder2D
  main.swift)
target_link_libraries(Autoencoder2D PRIVATE
  Datasets
  ModelSupport)


install(TARGETS Autoencoder2D
  DESTINATION bin)
