//
//  Tokenization.swift
//  TranslationTransformer
//
//  Created by Andre Carrera on 2/13/20.
//  Copyright © 2020 Lambdo. All rights reserved.
//

import Foundation
import TensorFlow
import Python
/// Tokenized text passage.
public struct TextBatch: KeyPathIterable {
    /// IDs that correspond to the vocabulary used while tokenizing.
    /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
    public var tokenIds: Tensor<Int32> // TODO: !!! Mutable in order to allow for batching.
    // aka src
    
    public var targetTokenIds: Tensor<Int32>
    // aka tgt

    /// IDs of the token types (e.g., sentence A and sentence B in BERT).
    /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
//    public var tokenTypeIds: Tensor<Int32> // TODO: !!! Mutable in order to allow for batching.

    /// Mask over the sequence of tokens specifying which ones are "real" as opposed to "padding".
    /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
    public var mask: Tensor<Int32> // TODO: !!! Mutable in order to allow for batching.
    
    public var targetMask: Tensor<Int32> // TODO: !!! Mutable in order to allow for batching.
    
    public var targetTruth: Tensor<Int32>
    
    public var tokenCount: Int32
//    if I want my batch to have sentence of unequal length, the shorter sentences have to be padded with the `<pad>` token so that every sentence is the same length
//    The mask for the input listen tensor covers the padded elements
//    The shape of the target mask is (batch_size, 1, input_sequence_length)
//    The batch size can be anything but it looks like in the translation example, it is only 1.
    init(source: Tensor<Int32>, target: Tensor<Int32>, sourcePadId: Int32, targetPadId: Int32) {
        self.tokenIds = source
        self.mask = Tensor(zerosLike: source)
            .replacing(with: Tensor(onesLike: source), where: source .!= Tensor.init(sourcePadId)) // Tensor.init(0) might need dto be expanded to Tensor(zerosLike: source)
            .expandingShape(at: 1)
        // if target is not None?
        let rangeExceptLast = 0..<(target.shape[1] - 1)
        self.targetTokenIds = target[0...,rangeExceptLast] // not sure if this is right. just means except last one
        self.targetTruth = target[0..., 1...]
        self.targetMask = TextBatch.makeStandardMask(target: self.targetTokenIds, pad: targetPadId)
        self.tokenCount = Tensor(zerosLike: targetTruth)
        .replacing(with: Tensor(onesLike: targetTruth), where: self.targetTruth .!= Tensor.init(targetPadId))
            .sum().scalar! // .sum() returns a vector.. Maybe that's what I want??
        
    }
    
    static func makeStandardMask(target: Tensor<Int32>, pad: Int32) -> Tensor<Int32> {
        var targetMask = Tensor(zerosLike: target)
            .replacing(with: Tensor(onesLike: target), where: target .!= Tensor.init(pad))
            .expandingShape(at: -2)
        targetMask *= subsequentMask(size: target.shape.last!)
        // tgt_mask = tgt_mask & Variable(
//        subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return targetMask
    }
}
func subsequentMask(size: Int) -> Tensor<Int32> {
    let attentionShape = [1, size, size]
    return Tensor<Int32>(ones: TensorShape(attentionShape))
        .bandPart(subdiagonalCount: 0, superdiagonalCount: -1)
}
/// Returns a 3-D attention mask that correspond to the 2-D mask of the provided text batch.
///
/// - Parameters:
///   - text: Text batch for which to create an attention mask. `input.mask` has shape
///     `[batchSize, sequenceLength]`.
///
/// - Returns: Attention mask with shape `[batchSize, sequenceLength, sequenceLength]`.
internal func createAttentionMask(forTextBatch text: TextBatch) -> Tensor<Float> {
    let batchSize = text.tokenIds.shape[0]
    let fromSequenceLength = text.tokenIds.shape[1]
    let toSequenceLength = text.mask.shape[1]
    let reshapedMask = Tensor<Float>(text.mask.reshaped(to: [batchSize, 1, toSequenceLength]))

    // We do not assume that `input.tokenIds` is a mask. We do not actually care if we attend
    // *from* padding tokens (only *to* padding tokens) so we create a tensor of all ones.
    let broadcastOnes = Tensor<Float>(ones: [batchSize, fromSequenceLength, 1])

    // We broadcast along two dimensions to create the mask.
    return broadcastOnes * reshapedMask
}

internal func createTargetAttentionMask(forTextBatch text: TextBatch) -> Tensor<Float> {
    let batchSize = text.targetTokenIds.shape[0]
    let fromSequenceLength = text.targetTokenIds.shape[1]
    let toSequenceLength = text.targetMask.shape[1]
    let reshapedMask = Tensor<Float>(text.targetMask.reshaped(to: [batchSize, 1, toSequenceLength]))

    // We do not assume that `input.tokenIds` is a mask. We do not actually care if we attend
    // *from* padding tokens (only *to* padding tokens) so we create a tensor of all ones.
    let broadcastOnes = Tensor<Float>(ones: [batchSize, fromSequenceLength, 1])

    // We broadcast along two dimensions to create the mask.
    return broadcastOnes * reshapedMask
}

//// TODO: Add documentation.
//internal func padAndBatch(textBatches: [TextBatch]) -> TextBatch {
//    if textBatches.count == 1 { return textBatches.first! }
//    let maxLength = textBatches.map { $0.tokenIds.shape[1] }.max()!
//    let paddedBatches = textBatches.map { batch -> TextBatch in
//        let paddingSize = maxLength - batch.tokenIds.shape[1]
//        return TextBatch(
//            tokenIds: batch.tokenIds.padded(forSizes: [
//                (before: 0, after: 0),
//                (before: 0, after: paddingSize)]),
//            tokenTypeIds: batch.tokenTypeIds.padded(forSizes: [
//                (before: 0, after: 0),
//                (before: 0, after: paddingSize)]),
//            mask: batch.mask.padded(forSizes: [
//                (before: 0, after: 0),
//                (before: 0, after: paddingSize)]))
//    }
//    return TextBatch(
//        tokenIds: Tensor<Int32>(
//            concatenating: paddedBatches.map { $0.tokenIds }, alongAxis: 0),
//        tokenTypeIds: Tensor<Int32>(
//            concatenating: paddedBatches.map { $0.tokenTypeIds }, alongAxis: 0),
//        mask: Tensor<Int32>(
//            concatenating: paddedBatches.map { $0.mask }, alongAxis: 0))
//}
//
/// Vocabulary that can be used for tokenizing strings.
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
        idsToTokens[count] = token
        tokensToIds[token] = count
        return tokensToIds[token]!
    }
}

//extension Vocabulary {
//    public init(fromFile fileURL: URL) throws {
//        self.init(
//        tokensToIds: [String: Int](
//            (try String(contentsOfFile: fileURL.path, encoding: .utf8))
//                .components(separatedBy: .newlines)
//                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
//                .filter { $0.count > 0 }
//                .enumerated().map { ($0.element, $0.offset) },
//            uniquingKeysWith: { (v1, v2) in max(v1, v2) }))
//    }
//
//    public func save(toFile fileURL: URL) throws {
//        try idsToTokens
//            .sorted { $0.key < $1.key }
//            .map { $0.1 }
//            .joined(separator: "\n")
//            .write(to: fileURL, atomically: true, encoding: .utf8)
//    }
//}
//
//extension Vocabulary {
//    public init(fromSentencePieceModel fileURL: URL) throws {
//        self.init(
//            tokensToIds: [String: Int](
//                (try Sentencepiece_ModelProto(serializedData: Data(contentsOf: fileURL)))
//                    .pieces
//                    .map { $0.piece.replacingOccurrences(of: "▁", with: "##") }
//                    .map { $0 == "<unk>" ? "[UNK]" : $0 }
//                    .enumerated().map { ($0.element, $0.offset) },
//                uniquingKeysWith: { (v1, v2) in max(v1, v2) }))
//    }
//
//    public init(fromJSONFile fileURL: URL) throws {
//        let json = try String(contentsOfFile: fileURL.path)
//        let tokensToIds = try JSONDecoder().decode(
//            [String: Int].self,
//            from: json.data(using: .utf8)!)
//        self.init(tokensToIds: tokensToIds)
//    }
//}
//
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

///// Greedy subword tokenizer.
/////
///// This tokenizer uses a greedy longest-match-first algorithm to perform tokenization using the
///// provided vocabulary. For example, `"unaffable"` could be tokenized as
///// `["un", "##aff", "##able"]`.
//public struct GreedySubwordTokenizer: Tokenizer {
//    public let vocabulary: Vocabulary
//    public let unknownToken: String
//    public let maxTokenLength: Int?
//
//    /// Creates a subword tokenizer.
//    ///
//    /// - Parameters:
//    ///   - vocabulary: Vocabulary containing all supported tokens.
//    ///   - unknownToken: Token used to represent unknown tokens (i.e., tokens that are not in the
//    ///     provided vocabulary or whose length is longer than `maxTokenLength`).
//    ///   - maxTokenLength: Maximum allowed token length.
//    public init(vocabulary: Vocabulary, unknownToken: String = "[UNK]", maxTokenLength: Int?) {
//        self.vocabulary = vocabulary
//        self.unknownToken = unknownToken
//        self.maxTokenLength = maxTokenLength
//    }
//
//    public func tokenize(_ text: String) -> [String] {
//        clean(text).split(separator: " ").flatMap { token -> [String] in
//            if let maxLength = maxTokenLength, token.count > maxLength { return [unknownToken] }
//            var isBad = false
//            var start = token.startIndex
//            var subTokens = [String]()
//            while start < token.endIndex {
//                // Find the longest matching substring.
//                var end = token.endIndex
//                var currentSubstring = ""
//                while start < end {
//                    var substring = String(token[start..<end])
//                    if start > token.startIndex {
//                        substring = "##" + substring
//                    }
//                    if vocabulary.contains(substring) {
//                        currentSubstring = substring
//                        start = end
//                    } else {
//                        end = token.index(end, offsetBy: -1)
//                    }
//                }
//
//                // Check if the substring is good.
//                if currentSubstring.isEmpty {
//                    isBad = true
//                    start = token.endIndex
//                } else {
//                    subTokens.append(currentSubstring)
//                    start = end
//                }
//            }
//            return isBad ? [unknownToken] : subTokens
//        }
//    }
//}
//
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

