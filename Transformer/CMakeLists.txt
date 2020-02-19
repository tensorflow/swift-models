add_executable(Transformer
  main.swift
  Model.swift
  Operators.swift
  PythonCheckpointReader.swift)
target_link_libraries(Transformer PRIVATE
  ModelSupport)


install(TARGETS Transformer
  DESTINATION bin)
