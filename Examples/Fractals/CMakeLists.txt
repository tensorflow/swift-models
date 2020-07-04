add_executable(Fractals
  ComplexTensor.swift
  ImageUtilities.swift
  JuliaSet.swift
  MandelbrotSet.swift
  main.swift)
target_link_libraries(Fractals PRIVATE
  ArgumentParser
  ModelSupport)


install(TARGETS Fractals
  DESTINATION bin)
