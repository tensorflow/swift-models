add_library(StbImage
  include/stb_image.h
  include/stb_image_write.h
  stb_image_write.c
  stb_image.c)
set_target_properties(StbImage PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})

add_library(ModelSupport
  BijectiveDictionary.swift
  Checkpoints/CheckpointIndexReader.swift
  Checkpoints/CheckpointIndexWriter.swift
  Checkpoints/CheckpointReader.swift
  Checkpoints/CheckpointWriter.swift
  Checkpoints/Protobufs/tensor_bundle.pb.swift
  Checkpoints/Protobufs/tensor_shape.pb.swift
  Checkpoints/Protobufs/tensor_slice.pb.swift
  Checkpoints/Protobufs/types.pb.swift
  Checkpoints/Protobufs/versions.pb.swift
  Checkpoints/SnappyDecompression.swift
  FileManagement.swift
  Image.swift
  Stderr.swift
  Text/BytePairEncoder.swift
  Text/SentencePiece/SentencePieceModel.pb.swift
  Text/TextBatch.swift
  Text/Tokenization.swift)
set_target_properties(ModelSupport PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_compile_options(ModelSupport PRIVATE
  $<$<BOOL:${BUILD_TESTING}>:-enable-testing>)
target_link_libraries(ModelSupport PUBLIC
  StbImage
  SwiftProtobuf)


install(TARGETS StbImage
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)
get_swift_host_arch(swift_arch)
install(FILES
  $<TARGET_PROPERTY:StbImage,Swift_MODULE_DIRECTORY>/StbImage.swiftdoc
  $<TARGET_PROPERTY:StbImage,Swift_MODULE_DIRECTORY>/StbImage.swiftmodule
  DESTINATION lib/swift$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:_static>/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>/${swift_arch})

install(TARGETS ModelSupport
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)
get_swift_host_arch(swift_arch)
install(FILES
  $<TARGET_PROPERTY:ModelSupport,Swift_MODULE_DIRECTORY>/ModelSupport.swiftdoc
  $<TARGET_PROPERTY:ModelSupport,Swift_MODULE_DIRECTORY>/ModelSupport.swiftmodule
  DESTINATION lib/swift$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:_static>/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>/${swift_arch})
