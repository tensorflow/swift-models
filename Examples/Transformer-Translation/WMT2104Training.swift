//
//  File.swift
//  
//
//  Created by Andre Carrera on 3/26/20.
//

import Foundation
import TensorFlow
import ModelSupport
import Datasets

public struct TextProcessor {
    public let tokenizer: Tokenizer
    public var sourceVocabulary: Vocabulary
    public var targetVocabulary: Vocabulary
    private let sourcePadId: Int32
    private let targetPadId: Int32
    private let bosId: Int32
    private let eosId: Int32
    private let targetUnkId: Int32
    private let sourceUnkId: Int32
    private let maxSequenceLength: Int
    private let batchSize: Int
    public init(tokenizer: Tokenizer, sourceVocabulary: Vocabulary, targetVocabulary: Vocabulary, maxSequenceLength: Int,
    batchSize: Int) {
        self.tokenizer = tokenizer
        self.sourceVocabulary = sourceVocabulary
        self.targetVocabulary = targetVocabulary
        self.maxSequenceLength = maxSequenceLength
        self.batchSize = batchSize
        
        self.sourcePadId = Int32(self.sourceVocabulary.id(forToken: BLANK_WORD)!)
        self.targetPadId = Int32(self.targetVocabulary.id(forToken: BLANK_WORD)!)
        self.bosId = Int32(self.targetVocabulary.id(forToken: BOS_WORD)!)
        self.eosId = Int32(self.targetVocabulary.id(forToken: EOS_WORD)!)
        self.sourceUnkId = Int32(self.sourceVocabulary.id(forToken: UNKNOWN_WORD)!)
        self.targetUnkId = Int32(self.targetVocabulary.id(forToken: UNKNOWN_WORD)!)
    }
    // This will take all source and target sequenes
    // return batches where each batch is based on the size targets in the sequence.
//    public mutating func preprocess(source: [String], target:[String], maxSequenceLength: Int, batchSize: Int) -> [TextBatch] {
//        let sourcePadId = Int32(sourceVocabulary.add(token: BLANK_WORD))
//        let targetPadId = Int32(targetVocabulary.add(token: BLANK_WORD))
//        let bosId = Int32(targetVocabulary.add(token: BOS_WORD))
//        let eosId = Int32(targetVocabulary.add(token: EOS_WORD))
//
//        let tokenizedSource = source.map{ src -> [Int32] in
//            let tokenizedSequence = tokenizer
//                .tokenize(src)
//                .prefix(maxSequenceLength)
//            return tokenizedSequence.map { Int32(self.sourceVocabulary.add(token: $0))}
//        }
//        let tokenizedTarget = target.map{ tar -> [Int32] in
//            let tokenizedSequence = tokenizer
//                .tokenize(tar)
//                .prefix(maxSequenceLength)
//            return [bosId] + tokenizedSequence.map { Int32(self.targetVocabulary.add(token: $0))} + [eosId]
//        }
//
//        let sourceWithTarget = zip(tokenizedSource, tokenizedTarget).map{ $0 }
//
//        let groupedBySourceSize = Dictionary(grouping: sourceWithTarget, by: { $0.0.count}).values.flatMap { (group: [([Int32], [Int32])]) -> [TextBatch] in
//            let batchesFromGroup = group.chunked(into: batchSize)
//            return batchesFromGroup.map { (batch: [([Int32], [Int32])]) -> TextBatch in
//                // batch has multiple pairs of sources and targets
//                let sourceTensor = Tensor(batch.map{ Tensor<Int32>.init($0.0) })
//                let maxTargetLength = batch.map{ $0.1.count}.max() ?? 0
//                // pad target length up to largest max.
//                let targetTensor = Tensor(batch.map{ Tensor<Int32>.init($0.1 + [Int32](repeating: targetPadId, count: (maxTargetLength - $0.1.count))) }) // taraget tensor needs to be padded
//                let textBatch = TextBatch(source: sourceTensor, target: targetTensor, sourcePadId: sourcePadId, targetPadId: targetPadId)
//                return textBatch
//            }
//        }
//
//        return groupedBySourceSize
//    }
    /// only pads target sequence to max sequence length,
    public func preprocess(example: WMT2014EnDe.Example) -> TranslationBatch {
        
        let encodedSource = self.tokenizer.tokenize(example.sourceSentence)
            .prefix(self.maxSequenceLength)
            .map{ Int32(self.sourceVocabulary.id(forToken: $0) ?? Int(self.sourceUnkId))}
        
        var encodedTarget = self.tokenizer.tokenize(example.targetSentence)
            .prefix(self.maxSequenceLength - 2)
            .map{ Int32(self.targetVocabulary.id(forToken: $0) ?? Int(self.targetUnkId))}
        encodedTarget = [bosId] + encodedTarget + [eosId]
        let paddingCount = encodedTarget.count < maxSequenceLength ? maxSequenceLength - encodedTarget.count : 0
        let padding = [Int32](repeating: targetPadId, count: paddingCount)
        encodedTarget = encodedTarget + padding
        assert(encodedTarget.count == maxSequenceLength, "encodedTarget.count \(encodedTarget.count) does not equal maxSequenceLength \(maxSequenceLength)")
        
        let sourceTensor = Tensor<Int32>.init(encodedSource).expandingShape(at: 0)
        
        // add padding to target since it will be grouped by sourcelength
        
        // padding is going to be equal to the difference between maxSequence length and the totalEncod
        let targetTensor = Tensor<Int32>.init( encodedTarget).expandingShape(at: 0)
        let singleBatch = TranslationBatch(source: sourceTensor, target: targetTensor, sourcePadId: sourcePadId, targetPadId: targetPadId)
        
//        print("original source:", example.sourceSentence)
//        print("decoded source:", decode(tensor: singleBatch.tokenIds, vocab: sourceVocabulary))
//        
//        print("max len = \(maxSequenceLength)")
//        print("encoded target \(encodedTarget.count) last: \(encodedTarget.last!)")
//        print("original target:", example.targetSentence)
//        print("decoded target:", decode(tensor: singleBatch.targetTokenIds, vocab: targetVocabulary))
//        print("decoded truth:", decode(tensor: singleBatch.targetTruth, vocab: targetVocabulary))
        return singleBatch
    }
    
}

func decode(tensor: Tensor<Int32>, vocab: Vocabulary) -> String {
  let endId = Int32(vocab.id(forToken: "</s>")!)
   var words = [String]()
   for scalar in tensor.scalars {
//       if Int(scalar) == endId {
//           break
//       } else
        if let token = vocab.token(forId: Int(scalar)) {
           words.append(token)
       }
   }
   return words.joined(separator: " ")
}

extension Vocabulary {
    
    public init(fromFile fileURL: URL, specialTokens: [String]) throws {
        let vocabItems = try ( String(contentsOfFile: fileURL.path, encoding: .utf8))
        .components(separatedBy: .newlines)
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        let dictionary = [String: Int](
                (specialTokens + vocabItems)
                .filter { $0.count > 0 }
                .enumerated().map { ($0.element, $0.offset) },
            uniquingKeysWith: { (v1, v2) in max(v1, v2) })
        self.init(tokensToIds: dictionary )
    }
}
