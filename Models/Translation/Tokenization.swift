//
//  Tokenization.swift
//  TranslationTransformer
//
//  Created by Andre Carrera on 2/13/20.
//  Copyright © 2020 Lambdo. All rights reserved.
//

import Foundation
import TensorFlow
import Datasets

let BOS_WORD = "<s>"
let EOS_WORD = "</s>"
let BLANK_WORD = "<blank>"

public struct TextProcessor {
    public let tokenizer: Tokenizer
    public var sourceVocabulary: Vocabulary
    public var targetVocabulary: Vocabulary
    private let sourcePadId: Int32
    private let targetPadId: Int32
    private let bosId: Int32
    private let eosId: Int32
    private let maxSequenceLength: Int
    private let batchSize: Int
    public init(tokenizer: Tokenizer, sourceVocabulary: Vocabulary, targetVocabulary: Vocabulary, maxSequenceLength: Int,
    batchSize: Int) {
        self.tokenizer = tokenizer
        self.sourceVocabulary = sourceVocabulary
        self.targetVocabulary = targetVocabulary
        self.maxSequenceLength = maxSequenceLength
        self.batchSize = batchSize
        
        self.sourcePadId = Int32(self.sourceVocabulary.add(token: BLANK_WORD))
        self.targetPadId = Int32(self.targetVocabulary.add(token: BLANK_WORD))
        self.bosId = Int32(self.targetVocabulary.add(token: BOS_WORD))
        self.eosId = Int32(self.targetVocabulary.add(token: EOS_WORD))
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
    
    public func preprocess(example: WMT2014EnDe.Example) -> WMT2014EnDe.TextBatch {
        
        let encodedSource = self.tokenizer.tokenize(example.sourceSentence)
            .prefix(self.maxSequenceLength)
            .map{ Int32(self.sourceVocabulary.id(forToken: $0) ?? 0)}
        
        var encodedTarget = self.tokenizer.tokenize(example.targetSentence)
            .prefix(self.maxSequenceLength - 2)
            .map{ Int32(self.targetVocabulary.id(forToken: $0) ?? 0)}
        encodedTarget = [bosId] + encodedTarget + [eosId]
        let paddingCount = encodedTarget.count < maxSequenceLength ? maxSequenceLength - encodedTarget.count : 0
        let padding = [Int32](repeating: targetPadId, count: paddingCount)
        encodedTarget = encodedTarget + padding
        assert(encodedTarget.count == maxSequenceLength, "encodedTarget.count \(encodedTarget.count) does not equal maxSequenceLength \(maxSequenceLength)")
        
        let sourceTensor = Tensor<Int32>.init(encodedSource).expandingShape(at: 0)
        
        // add padding to target since it will be grouped by sourcelength
        
        // padding is going to be equal to the difference between maxSequence length and the totalEncod
        let targetTensor = Tensor<Int32>.init( encodedTarget).expandingShape(at: 0)
        return WMT2014EnDe.TextBatch(source: sourceTensor, target: targetTensor, sourcePadId: sourcePadId, targetPadId: targetPadId)
    }
    
}

public struct Vocabulary {
    internal var tokensToIds: [String: Int]
    internal var idsToTokens: [Int: String]
    
    public var count: Int { tokensToIds.count }
    
    public init(tokensToIds: [String: Int]) {
        self.tokensToIds = tokensToIds
        self.idsToTokens = [Int: String](uniqueKeysWithValues: tokensToIds.map { ($1, $0) })
    }
    
    public init(idsToTokens: [Int: String]) {
        self.tokensToIds = [String: Int](uniqueKeysWithValues: idsToTokens.map { ($1, $0) })
        self.idsToTokens = idsToTokens
    }
    
    public init() {
        self.tokensToIds = [:]
        self.idsToTokens = [:]
    }
    
    public func contains(_ token: String) -> Bool {
        tokensToIds.keys.contains(token)
    }
    
    public func id(forToken token: String) -> Int? {
        tokensToIds[token]
    }
    
    public func token(forId id: Int) -> String? {
        idsToTokens[id]
    }
    public mutating func add(token: String) -> Int {
        if let result = tokensToIds[token] {
            return result
        }
        let newId = self.count
        idsToTokens[newId] = token
        tokensToIds[token] = newId
        assert(idsToTokens.count == tokensToIds.count, "Vocabulary doesn't match, tokensToIds \(tokensToIds.count) , idsTotokens \(idsToTokens.count)")
        return newId
    }
}

extension Vocabulary {
    public init(fromFile fileURL: URL) throws {
        self.init(
        tokensToIds: [String: Int](
            (try String(contentsOfFile: fileURL.path, encoding: .utf8))
                .components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { $0.count > 0 }
                .enumerated().map { ($0.element, $0.offset) },
            uniquingKeysWith: { (v1, v2) in max(v1, v2) }))
    }

    public func save(toFile fileURL: URL) throws {
        try idsToTokens
            .sorted { $0.key < $1.key }
            .map { $0.1 }
            .joined(separator: "\n")
            .write(to: fileURL, atomically: true, encoding: .utf8)
    }
}


/// Text tokenizer which is used to split strings into arrays of tokens.
public protocol Tokenizer {
    func tokenize(_ text: String) -> [String]
}

/// Basic text tokenizer that performs some simple preprocessing to clean the provided text and
/// then performs tokenization based on whitespaces.
public struct BasicTokenizer: Tokenizer {
    public let caseSensitive: Bool
    
    /// Creates a basic text tokenizer.
    ///
    /// Arguments:
    ///   - caseSensitive: Specifies whether or not to ignore case.
    public init(caseSensitive: Bool = false) {
        self.caseSensitive = caseSensitive
    }
    
    public func tokenize(_ text: String) -> [String] {
        clean(text).split(separator: " ").flatMap { token -> [String] in
            var processed = String(token)
            if !caseSensitive {
                processed = processed.lowercased()
                
                // Normalize unicode characters.
                processed = processed.decomposedStringWithCanonicalMapping
                
                // Strip accents.
                processed = processed.replacingOccurrences(
                    of: #"\p{Mn}"#,
                    with: "",
                    options: .regularExpression)
            }
            
            //             Split punctuation. We treat all non-letter/number ASCII as punctuation. Characters
            //             such as "$" are not in the Unicode Punctuation class but we treat them as
            //             punctuation anyways for consistency.
            processed = processed.replacingOccurrences(
                of: #"([\p{P}!-/:-@\[-`{-~])"#,
                with: " $1 ",
                options: .regularExpression)
            
            return processed.split(separator: " ").map(String.init)
        }
    }
}


/// Returns a cleaned version of the provided string. Cleaning in this case consists of normalizing
/// whitespaces, removing control characters and adding whitespaces around CJK characters.
///
/// - Parameters:
///   - text: String to clean.
///
/// - Returns: Cleaned version of `text`.
internal func clean(_ text: String) -> String {
    // Normalize whitespaces.
    let afterWhitespace = text.replacingOccurrences(
        of: #"\s+"#,
        with: " ",
        options: .regularExpression)
    
    // Remove control characters.
    let afterControl = afterWhitespace.replacingOccurrences(
        of: #"[\x{0000}\x{fffd}\p{C}]"#,
        with: "",
        options: .regularExpression)
    
    // Add whitespace around CJK characters.
    //
    // The regular expression that we use defines a "chinese character" as anything in the
    // [CJK Unicode block](https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)).
    //
    // Note that the CJK Unicode block is not all Japanese and Korean characters, despite its name.
    // The modern Korean Hangul alphabet is a different block, as is Japanese Hiragana and
    // Katakana. Those alphabets are used to write space-separated words, and so they are not
    // treated specially and are instead handled like all of the other languages.
    let afterCJK = afterControl.replacingOccurrences(
        of: #"([\p{InCJK_Unified_Ideographs}"# +
            #"\p{InCJK_Unified_Ideographs_Extension_A}"# +
            #"\p{InCJK_Compatibility_Ideographs}"# +
            #"\x{20000}-\x{2a6df}"# +
            #"\x{2a700}-\x{2b73f}"# +
            #"\x{2b740}-\x{2b81f}"# +
            #"\x{2b820}-\x{2ceaf}"# +
        #"\x{2f800}-\x{2fa1f}])"#,
        with: " $1 ",
        options: .regularExpression)
    
    return afterCJK
}

