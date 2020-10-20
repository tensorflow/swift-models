add_library(AutoencoderCallback
  SaveImage.swift)
target_link_libraries(AutoencoderCallback PRIVATE
  ModelSupport
  TrainingLoop)


install(TARGETS AutoencoderCallback
  DESTINATION bin)