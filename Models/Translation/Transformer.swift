//
//  Transformer.swift
//  TranslationTransformer
//
//  Created by Andre Carrera on 2/13/20.
//  Copyright © 2020 Lambdo. All rights reserved.
//

import TensorFlow

struct TransformerModel: Module {
    // helpers
    @noDerivative public let tokenizer: Tokenizer
    @noDerivative public let vocabulary: Vocabulary
    // layers
    var encoder: Encoder
    var decoder: Decoder
    var sourceEmbed: Sequential<Embedding<Float>, PositionalEncoding>
    var targetEmbed: Sequential<Embedding<Float>, PositionalEncoding>
    init(vocabulary: Vocabulary, tokenizer: Tokenizer, sourceVocabSize: Int, targetVocabSize: Int, layerCount: Int = 6, modelSize: Int = 256, feedForwardSize: Int = 1024, h:Int = 8, dropoutProbability: Double = 0.1) {
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        
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
    func callAsFunction(_ input: TextBatch) -> Tensor<Float> {
        let sourceAttentionMask = createAttentionMask(forTextBatch: input) // might not need this?? 
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
    
    func preprocess(sequences: [String], targetSequences:[String], maxSequenceLength: Int) -> TextBatch {
        let sequences = sequences.map(tokenizer.tokenize(_:))
        let targetSequences = targetSequences.map(tokenizer.tokenize(_:))
        // I think for sequences I will only pass in nope nope nope.
        // TODO: confirm batches in original project, there's some fancy rebatching going on that gets confusing.
        
        
        // Truncate the sequences based on the maximum allowed sequence length, while accounting
        // for the '[CLS]' token and for `sequences.count` '[SEP]' tokens. The following is a
        // simple heuristic which will truncate the longer sequence one token at a time. This makes
        // more sense than truncating an equal percent of tokens from each sequence, since if one
        // sequence is very short then each token that is truncated likely contains more
        // information than respective tokens in longer sequences.
//        var totalLength = sequences.map { $0.count }.reduce(0, +)
//        let totalLengthLimit = { () -> Int in
//            return maxSequenceLength - 1 - sequences.count
//        }()
//        while totalLength >= totalLengthLimit {
//            let maxIndex = sequences.enumerated().max(by: { $0.1.count < $1.1.count })!.0
//            sequences[maxIndex] = [String](sequences[maxIndex].dropLast())
//            totalLength = sequences.map { $0.count }.reduce(0, +)
//        }
        
        
        let BOS_WORD = "<s>"
        let EOS_WORD = "</s>"
        let BLANK_WORD = "<blank>"
        
        var sourceTokens: [String] = []// what is the first one supposed to be?
        var targetTokens: [String] = []
        
        for (sequence, target ) in zip(sequences, targetSequences) { // we don't need the enum
            for token in sequence {
                sourceTokens.append(token)
            }
            
            targetTokens.append(BOS_WORD)
            for token in target {
                targetTokens.append(token)
            }
            targetTokens.append(EOS_WORD)
            
        }
        
        // big questions, a list of the source and target tokens together? no
        // but it's multiple sources in one source.
        // that means they all need to be padded.
        //
        
        let padId = vocabulary.id(forToken: BLANK_WORD)!
        let tokenIds = sourceTokens.map { Int32(vocabulary.tokensToIds[$0]!)}
        let targetTokenIds = targetTokens.map { Int32(vocabulary.tokensToIds[$0]!)}
        // a batch contains several source sentences and the same amount of target sentences.
        // at this point a senence/ sequence is just a list of tokens.
        return TextBatch(source: Tensor(tokenIds), target: Tensor(targetTokenIds), pad: Int32(padId))
        // do I use tokenTypeIds? nope!
    }
    
    // the source is going to be a tensor of shape (sequence_count, longest_sequence_source)
    
    // the target (sequence_count, longest_sequence_of_target) where sequence_count is the same
    
    // it looks like the source batches all have the same size.
    // target is going to have <s> to start and </s> to end.
}
