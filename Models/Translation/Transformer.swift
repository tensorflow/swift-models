//
//  Transformer.swift
//  TranslationTransformer
//
//  Created by Andre Carrera on 2/13/20.
//  Copyright © 2020 Lambdo. All rights reserved.
//

import TensorFlow

public struct TransformerModel: Module {
    // layers
    var encoder: Encoder
    var decoder: Decoder
    var sourceEmbed: Sequential<Embedding<Float>, PositionalEncoding>
    var targetEmbed: Sequential<Embedding<Float>, PositionalEncoding>
    public init(sourceVocabSize: Int, targetVocabSize: Int, layerCount: Int = 6, modelSize: Int = 256, feedForwardSize: Int = 1024, h:Int = 8, dropoutProbability: Double = 0.1) {
        
        let attentions = [MultiHeadAttention](repeating: .init(sourceSize: h, targetSize: modelSize), count: 3)
        let feedForwards = [PositionwiseFeedForward](repeating: .init(dimensionalityModel: modelSize, innerLayerDimensionality: feedForwardSize), count: 2)
        let positionalEncodings = [PositionalEncoding](repeating: .init(size: modelSize, dropoutProbability: dropoutProbability), count: 2)
        
        self.encoder = Encoder(layer: .init(size: modelSize, selfAttention: attentions[0], feedForward: feedForwards[0], dropoutProb: dropoutProbability), layerCount: layerCount)
        self.decoder = Decoder(layer: .init(size: modelSize, selfAttention: attentions[1], sourceAttention: attentions[2], feedForward: feedForwards[1], dropoutProb: dropoutProbability), layerCount: layerCount)
        self.sourceEmbed = Sequential(Embedding(vocabularySize: sourceVocabSize, embeddingSize: modelSize), positionalEncodings[0])
        self.targetEmbed = Sequential(Embedding(vocabularySize: targetVocabSize, embeddingSize: modelSize), positionalEncodings[1])
        // todo xavier init.
    }
    
    @differentiable
    public func callAsFunction(_ input: TextBatch) -> Tensor<Float> {
        let sourceAttentionMask = Tensor<Float>(input.mask)
        return self.decode(input: input, memory: self.encode(input: input, attentionMask: sourceAttentionMask), sourceMask: sourceAttentionMask)
    }
    
    @differentiable
    func encode(input: TextBatch, attentionMask: Tensor<Float>) -> Tensor<Float> {
        let embedded = self.sourceEmbed(input.tokenIds)
        return self.encoder(.init(sequence: embedded, attentionMask: attentionMask))
    }
    
    @differentiable
    func decode(input: TextBatch, memory: Tensor<Float>, sourceMask: Tensor<Float>) -> Tensor<Float> {
        let embedded = self.targetEmbed(input.targetTokenIds)
        let targetAttentionMask = createTargetAttentionMask(forTextBatch: input)
        return self.decoder.callAsFunction(.init(sequence: embedded, sourceMask: sourceMask, targetMask: targetAttentionMask, memory: memory))
    }
}

public struct TextProcessor {
    public let tokenizer: Tokenizer
    public var sourceVocabulary: Vocabulary
    public var targetVocabulary: Vocabulary
    public init(tokenizer: Tokenizer, sourceVocabulary: Vocabulary, targetVocabulary: Vocabulary) {
        self.tokenizer = tokenizer
        self.sourceVocabulary = sourceVocabulary
        self.targetVocabulary = targetVocabulary
    }
    // This will take all source and target sequenes
    // return batches where each batch is based on the size targets in the sequence.
    public mutating func preprocess(source: [String], target:[String], maxSequenceLength: Int, batchSize: Int) -> [TextBatch] {
        let padId = Int32(targetVocabulary.add(token: BLANK_WORD))
        
        let tokenizedSource = source.map{ src -> [Int32] in
            let src = src.prefix(maxSequenceLength)
            let tokenizedSequence = tokenizer.tokenize(String(src))
            return tokenizedSequence.map { Int32(self.sourceVocabulary.add(token: $0))}
        }
        let tokenizedTarget = target.map{ tar -> [Int32] in
            let tar = tar.prefix(maxSequenceLength)
            let tokenizedSequence = tokenizer.tokenize(BOS_WORD + tar + EOS_WORD)
            return tokenizedSequence.map { Int32(self.targetVocabulary.add(token: $0))}
        }
        // (sequenceCount, tokenCount)
        
        let groupedSources = Dictionary(grouping: tokenizedSource, by: {$0.count}).values.map { $0}
        let groupedTargets = Dictionary(grouping: tokenizedTarget, by: {$0.count}).values.map { $0}
        // (groupCount, sequencePerGroupCount,tokenPerSequenceCount)
        
        let batches = zip(groupedSources, groupedTargets).flatMap { (sourceGroup: [[Int32]], targetGroup: [[Int32]]) -> [TextBatch] in
            let sourceBatches = sourceGroup.chunked(into: batchSize)
            let targetBatches  = targetGroup.chunked(into: batchSize)
            let textBatches = zip(sourceBatches, targetBatches).map { (sourceBatch: [[Int32]], targetBatch: [[Int32]]) -> TextBatch in
                let sourceTensor = Tensor(sourceBatch.map{ ids in
                    return Tensor<Int32>.init(ids)
                })
                let targetTensor = Tensor(targetBatch.map{ ids in
                    return Tensor<Int32>.init(ids)
                })
                return TextBatch(source: sourceTensor, target: targetTensor, pad: padId)
            }
            return textBatches
        }
        return batches
    }
    
}



let BOS_WORD = "<s>"
let EOS_WORD = "</s>"
let BLANK_WORD = "<blank>"
