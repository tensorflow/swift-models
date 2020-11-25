add_library(AutoencoderCallback
  ImageSaver.swift)
target_link_libraries(AutoencoderCallback PRIVATE
  ModelSupport
  TrainingLoop)


install(TARGETS AutoencoderCallback
  DESTINATION bin)