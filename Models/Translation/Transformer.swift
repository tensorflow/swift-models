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
    public init(sourceVocabSize: Int, targetVocabSize: Int, layerCount: Int = 6, modelSize: Int = 256, feedForwardSize: Int = 1024, headCount: Int = 8, dropoutProbability: Double = 0.1) {
        
        let attentions = [MultiHeadAttention](repeating: .init(sourceSize: modelSize, targetSize: modelSize, headCount: headCount, headSize: modelSize/headCount,  matrixResult: true), count: 3) // matrix true goes further
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
        let sourcePadId = Int32(sourceVocabulary.add(token: BLANK_WORD))
        let targetPadId = Int32(targetVocabulary.add(token: BLANK_WORD))
        let bosId = Int32(targetVocabulary.add(token: BOS_WORD))
        let eosId = Int32(targetVocabulary.add(token: EOS_WORD))
        
        let tokenizedSource = source.map{ src -> [Int32] in
            let tokenizedSequence = tokenizer
                .tokenize(String(src))
                .prefix(maxSequenceLength)
            return tokenizedSequence.map { Int32(self.sourceVocabulary.add(token: $0))}
        }
        let tokenizedTarget = target.map{ tar -> [Int32] in
            let tokenizedSequence = tokenizer
                .tokenize(String(tar))
                .prefix(maxSequenceLength)
            return [bosId] + tokenizedSequence.map { Int32(self.targetVocabulary.add(token: $0))} + [eosId]
        }
        // (sequenceCount, tokenCount)
        
//        print("tokenizedSource \(tokenizedSource)")
//        print("tokenizedTarget \(tokenizedTarget)")
        
        // will have to group based on the source token count and then pad the target
        let sourceWithTarget = zip(tokenizedSource, tokenizedTarget).map{ $0 }
//        print("sourceWithTarget \(sourceWithTarget)")
        
        let groupedBySourceSize = Dictionary(grouping: sourceWithTarget, by: { $0.0.count}).values.flatMap { (group: [([Int32], [Int32])]) -> [TextBatch] in
            let batchesFromGroup = group.chunked(into: batchSize)
//            print("batchesFromGroup \(batchesFromGroup)")
            return batchesFromGroup.map { (batch: [([Int32], [Int32])]) -> TextBatch in
                // batch has multiple pairs of sources and targets
                let sourceTensor = Tensor(batch.map{ Tensor<Int32>.init($0.0) })
                let maxTargetLength = batch.map{ $0.1.count}.max() ?? 0
                // pad target length up to largest max.
                let targetTensor = Tensor(batch.map{ Tensor<Int32>.init($0.1 + [Int32](repeating: targetPadId, count: (maxTargetLength - $0.1.count))) }) // taraget tensor needs to be padded
                let textBatch = TextBatch(source: sourceTensor, target: targetTensor, sourcePadId: sourcePadId, targetPadId: targetPadId)
//                print(textBatch)
//                 tokenIds: (batchSize, tokens)
                // targetTokenIds: (batchSize, targetTokens + startToken)
                // targetTruth: (batchSize, targetTokens + endToken)
                // mask: (batchSize, tokens) everything except the start is masked?
                // targetMask: ( batchSize, targetTokens + startToken, targetTokens + startToken) upper right triangle for each sequence in batch.
//                print("source: \(textBatch.tokenIds) ")
//                print("source mask: \(textBatch.mask)")
//                print("target: \(textBatch.targetTokenIds.shape)")
//                print("target_y: \(textBatch.targetTruth.shape)")
//                print("target_mask \(textBatch.targetMask.shape)")
                return textBatch
            }
        }
        
//        print(groupedBySourceSize)
        return groupedBySourceSize
////        let groupedBySourceSize = Dictionary(grouping: zip(tokenizedSource, tokenizedTarget), by: { $0.0.count }).values.map { ([Zip2Sequence<[[Int32]], [[Int32]]>.Element]) -> T in
////            <#code#>
////        }
//        let groupedSources = Dictionary(grouping: tokenizedSource, by: {$0.count}).values.map { $0}
//        let groupedTargets = Dictionary(grouping: tokenizedTarget, by: {$0.count}).values.map { $0}
//        // (groupCount, sequencePerGroupCount,tokenPerSequenceCount)
//
//        print("source \(source)")
//        print("grouped sources: \(groupedSources)")
//
//        let batches = zip(groupedSources, groupedTargets).flatMap { (sourceGroup: [[Int32]], targetGroup: [[Int32]]) -> [TextBatch] in
//            let sourceBatches = sourceGroup.chunked(into: batchSize)
//            let targetBatches  = targetGroup.chunked(into: batchSize)
//            let textBatches = zip(sourceBatches, targetBatches).map { (sourceBatch: [[Int32]], targetBatch: [[Int32]]) -> TextBatch in
//                let sourceTensor = Tensor(sourceBatch.map{ ids in
//                    return Tensor<Int32>.init(ids)
//                })
//                let targetTensor = Tensor(targetBatch.map{ ids in
//                    return Tensor<Int32>.init(ids)
//                })
//                return TextBatch(source: sourceTensor, target: targetTensor, pad: padId)
//            }
//            return textBatches
//        }
//        return batches
    }
    
}



let BOS_WORD = "<s>"
let EOS_WORD = "</s>"
let BLANK_WORD = "<blank>"
