add_library(Pix2PixLib
  Pix2Pix/CheckpointWriter.swift
  Pix2Pix/CLI.swift
  Pix2Pix/Dataset.swift
  Pix2Pix/Discriminator.swift
  Pix2Pix/Generator.swift
  Pix2Pix/Layers.swift
  Pix2Pix/Utils.swift
  Pix2Pix/PythonCheckpointReader.swift)
  
  add_executable(Pix2Pix
  main.swift)
  
  target_link_libraries(Pix2PixLib)
