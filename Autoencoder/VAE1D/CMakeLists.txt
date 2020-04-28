add_executable(VariationalAutoencoder1D
  main.swift)
target_link_libraries(VariationalAutoencoder1D PRIVATE
  Datasets
  ModelSupport)


install(TARGETS VariationalAutoencoder1D
  DESTINATION bin)
