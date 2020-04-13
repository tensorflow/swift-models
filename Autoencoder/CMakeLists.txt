add_executable(Autoencoder
  Autoencoder1D/main.swift
  Autoencoder2D/main.swift
  VAE1D/main.swift)
target_link_libraries(Autoencoder PRIVATE
  Datasets
  ModelSupport)


install(TARGETS Autoencoder
  DESTINATION bin)
