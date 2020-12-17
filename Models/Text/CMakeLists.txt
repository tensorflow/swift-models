add_library(TextModels
  BERT/Attention.swift
  BERT/BERT.swift
  BERT/BERTClassifier.swift
  BERT/BERTCheckpointReader.swift
  BERT/TransformerBERT.swift
  BERT/Utilities.swift
  GPT2/CheckpointWriter.swift
  GPT2/GPT2.swift
  GPT2/TransformerLM.swift
  GPT2/Operators.swift
  GPT2/PythonCheckpointReader.swift
  SentencePiece/SentencePieceModel.pb.swift
  WordSeg/Lattice.swift
  WordSeg/Model.swift
  WordSeg/SemiRing.swift)
set_target_properties(TextModels PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_compile_definitions(TextModels PRIVATE
  $<$<BOOL:${TENSORFLOW_USE_STANDARD_TOOLCHAIN}>:TENSORFLOW_USE_STANDARD_TOOLCHAIN>)
target_compile_options(TextModels PRIVATE
  $<$<BOOL:${BUILD_TESTING}>:-enable-testing>)
target_link_libraries(TextModels PUBLIC
  Checkpoints
  Datasets
#  $<$<AND:$<BOOL:${ENABLE_SWIFT_NUMERICS}>,$<BOOL:${TENSORFLOW_USE_STANDARD_TOOLCHAIN}>>:Numerics>
  SwiftProtobuf)

install(TARGETS TextModels
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)
get_swift_host_arch(swift_arch)
install(FILES
  $<TARGET_PROPERTY:TextModels,Swift_MODULE_DIRECTORY>/TextModels.swiftdoc
  $<TARGET_PROPERTY:TextModels,Swift_MODULE_DIRECTORY>/TextModels.swiftmodule
  DESTINATION lib/swift$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:_static>/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>/${swift_arch})
