add_library(TransformerSupport
  TextBatch.swift
  Tokenization.swift
  BytePairEncoder.swift
  SentencePiece/SentencePieceModel.pb.swift)
target_link_libraries(TransformerSupport PRIVATE
   SwiftProtobuf)

install(TARGETS TransformerSupport
   DESTINATION bin)
