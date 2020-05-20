// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation

public struct BytePairEncoder {
    public enum BPEError: Error {
        case unsupported
    }

    public let vocabulary: Vocabulary
    public let mergePairs: [Pair: Int]
    public let reversedMergePairs: [String: Pair]
    public let useCache: Bool

    // TODO: Find a nice way to support caching.
    /// A cache used to store encoded tokens and thus speed up encoding.
    //  private var cache: [String: [String]]

    public init(
        vocabularyFile: URL, mergesFile: URL,
        encoding: String.Encoding = .utf8
    ) throws {
        let vocabulary: Vocabulary = try Vocabulary(fromJSONFile: vocabularyFile)

        let lines: ArraySlice<String> =
            try String(contentsOfFile: mergesFile.path, encoding: encoding)
            .components(separatedBy: .newlines)
            .dropFirst()

        let pairs: [BytePairEncoder.Pair: Int] =
            [BytePairEncoder.Pair: Int](
                uniqueKeysWithValues: lines.enumerated().compactMap {
                    (index, line) -> (BytePairEncoder.Pair, Int)? in
                    let tokens = line.split(separator: " ")
                    guard tokens.count >= 2 else { return nil }
                    return (BytePairEncoder.Pair(String(tokens[0]), String(tokens[1])), index)
                })

        self.init(vocabulary: vocabulary, mergePairs: pairs)
    }

    public init(vocabulary: Vocabulary, mergePairs: [Pair: Int], useCache: Bool = true) {
        self.vocabulary = vocabulary
        self.mergePairs = mergePairs
        self.reversedMergePairs = [String: Pair](
            uniqueKeysWithValues: mergePairs.map {
                ($0.key.left + $0.key.right, $0.key)
            })
        self.useCache = useCache
        // self.cache = [:]
    }

    /// Encodes the provided token to a sequence of BPE-coded tokens.
    ///
    /// - Parameters:
    ///   - token: Token to encode.
    ///   - variant: Type of model (default: .roberta).
    /// - Returns: Array containing the BPE-coded tokens.
    public func encode(token: String, variant: Variant? = .roberta) -> [String] {
        // if let cached = cache[token] { return cached }
        // let token = " " + token
        var parts = [String]()

        switch variant {
        case .gpt2:
            // Split into parts before encoding.
            let unencodedTokens = BytePairEncoder.splittingWithDelimiters(
                token: token,
                glossaryRegex: BytePairEncoder.gpt2GlossaryRegex,
                variant: .gpt2)
            // Encode each token.
            let tokens = unencodedTokens.map({ BytePairEncoder.encodedToken($0) })
            // Separate each character.
            for token in tokens {
                for i in (0..<token.count) {
                    let index = token.index(token.startIndex, offsetBy: i)
                    parts.append(String(token[index]))
                }
            }
            if parts.count < 2 { return parts }
        case .roberta, .none:
            // Encode before splitting into parts.
            let encodedToken = BytePairEncoder.encodedToken(token)
            parts = BytePairEncoder.splittingWithDelimiters(
                token: encodedToken,
                glossaryRegex: BytePairEncoder.defaultGlossaryRegex,
                variant: .roberta)
            if parts.count < 2 { return parts }
        }

        // Create pairs of parts.
        var pairs = (0..<parts.count - 1).map { index in Pair(parts[index], parts[index + 1]) }
        while !pairs.isEmpty {
            let pair = pairs.min { mergePairs[$0] ?? Int.max < mergePairs[$1] ?? Int.max }!
            if !mergePairs.keys.contains(pair) { break }
            parts = BytePairEncoder.replacePair(pair: pair, tokenParts: parts)
            if parts.count < 2 { break }
            pairs = (0..<parts.count - 1).map { index in Pair(parts[index], parts[index + 1]) }
        }

        // Check if the new word parts are in the vocabulary, and backtrack if necessary.
        let encoded = parts.flatMap { part -> [String] in
            if vocabulary.contains(part) { return [part] }
            return splitRecursively(part)
        }

        // Update the cache and return.
        // if useCache { cache[token] = encoded }
        return encoded
    }

}

extension BytePairEncoder {
    public struct Pair: Hashable {
        public let left: String
        public let right: String

        public init(_ left: String, _ right: String) {
            self.left = left
            self.right = right
        }
    }

    internal static let defaultGlossary: [String] = [
        "e.g", "i.e", "&amp;", "&#124;", "&lt;", "&gt;", "&apos;", "&quot;", "&#91;", "&#93;",
    ]

    internal static let defaultGlossaryRegex: NSRegularExpression = {
        let escapedGlossary = defaultGlossary.map { "\\Q\($0)\\E" }.joined(separator: "|")
        return try! NSRegularExpression(pattern: "(?:\(escapedGlossary))|(?!\(escapedGlossary))")
    }()

    /// Regular expression matching the OpenAI GPT-2 implementation.
    internal static let gpt2Glossary = [
        "'s", "'t", "'re", "'ve", "'m", "'ll", "'d", " ?\\p{L}+", " ?\\p{N}+",
        " ?[^\\s\\p{L}\\p{N}]+", "\\s+(?!\\S)", "\\s+",
    ]

    internal static let gpt2GlossaryRegex: NSRegularExpression = {
        let escapedGlossary = gpt2Glossary.joined(separator: "|")
        return try! NSRegularExpression(pattern: "(?:\(escapedGlossary))")
    }()


    // TODO: Add documentation.
    internal static let bytesToUnicode: [UInt8: UnicodeScalar] = {
        var bytes = [UInt8](33...126) + [UInt8](161...172) + [UInt8](174...255)
        var characters = bytes.map(UInt32.init)
        var offset = UInt32(0)
        for byte in 0...UInt8(255) {
            if !bytes.contains(byte) {
                bytes.append(byte)
                characters.append(UInt32(offset + 256))
                offset += 1
            }
        }
        return [UInt8: UnicodeScalar](
            uniqueKeysWithValues: zip(bytes, characters.map { UnicodeScalar($0)! }))
    }()

    // The inverse of bytesToUnicode.
    internal static let unicodeToBytes: [UnicodeScalar: UInt8] = {
        [UnicodeScalar: UInt8](
            uniqueKeysWithValues: BytePairEncoder.bytesToUnicode.map { ($1, $0) })
    }()

    /// Recursively splits `token` into smaller units (by reversing BPE merges) until all units
    /// are either in the provided vocabulary, or cannot be split further.
    ///
    /// - Parameters:
    ///   - token: Token that needs to be split.
    internal func splitRecursively(_ token: String) -> [String] {
        guard let pair = reversedMergePairs[token] else { return [token] }
        let leftParts = vocabulary.contains(pair.left) ? [pair.left] : splitRecursively(pair.left)
        let rightParts =
            vocabulary.contains(pair.right) ? [pair.right] : splitRecursively(pair.right)
        return leftParts + rightParts
    }

    /// Uses the given regex to split a token into individual glossary terms.
    ///
    /// - Parameters:
    ///   - token: Full text.
    ///   - glossaryRegex: Regular expression for segmenting the given token.
    ///   - variant: The type of model (default: .roberta).
    /// - Returns: Array of substrings that match the given regex.
    internal static func splittingWithDelimiters(
        token: String,
        glossaryRegex: NSRegularExpression,
        keepEmpty: Bool = false,
        variant: Variant? = .roberta
    ) -> [String] {
        let matches = glossaryRegex.matches(
            in: token,
            range: NSRange(token.startIndex..., in: token))
        var parts = [String]()
        parts.reserveCapacity(token.count)
        switch variant {
        case .gpt2:
            for match in matches {
                if let start = token.index(
                    token.startIndex, offsetBy: match.range.lowerBound, limitedBy: token.endIndex),
                   let end = token.index(
                    token.startIndex, offsetBy: match.range.upperBound, limitedBy: token.endIndex)
                {
                  parts.append(String(token[start..<end]))
                }
            }
        case .roberta, .none:
            var lastEnd = token.startIndex
            for match in matches {
                let start = token.index(token.startIndex, offsetBy: match.range.lowerBound)
                if lastEnd != start { parts.append(String(token[lastEnd..<start])) }
                lastEnd = token.index(token.startIndex, offsetBy: match.range.upperBound)
            }
            if lastEnd != token.endIndex {
                parts.append(String(token[lastEnd...]))
            }
        }

        return parts
    }

    /// Replaces all occurrences of the provided symbol pair in `token` with the joined symbol.
    ///
    /// - Parameters:
    ///   - pair: Symbol pair to replace in `token`.
    ///   - token: Token as a sequence of symbols.
    /// - Returns: New token with the provided pair replaced for the joined symbol.
    internal static func replacePair(pair: Pair, tokenParts: [String]) -> [String] {
        var newTokenParts = [String]()
        newTokenParts.reserveCapacity(tokenParts.count)
        var j = 0
        while j < tokenParts.count - 1 {
            let part1 = tokenParts[j]
            let part2 = tokenParts[j + 1]
            if part1 == pair.left && part2 == pair.right {
                let joinedPair = part1 + part2
                newTokenParts.append(joinedPair)
                j += 2
            } else {
                newTokenParts.append(tokenParts[j])
                j += 1
            }
        }
        if j == tokenParts.count - 1 {
            newTokenParts.append(tokenParts[j])
        }
        return newTokenParts
    }

    internal static func encodedToken(_ token: String) -> String {
        String(String.UnicodeScalarView(token.utf8.map { BytePairEncoder.bytesToUnicode[$0]! }))
    }
}

extension BytePairEncoder {
    public enum Variant {
        /// Default variant.
        /// - Source: [RoBERTa: A Robustly Optimized BERT Pretraining Approach](
        ///             https://arxiv.org/pdf/1907.11692.pdf).
        case roberta
        /// - Source: [Language Models are Unsupervised Multitask Learners](
        ///             https://cdn.openai.com/better-language-models/
        ///             language_models_are_unsupervised_multitask_learners.pdf).
        case gpt2
    }

    /// Decodes the provided BPE-coded token to a sequence of tokens.
    ///
    /// - Parameters:
    ///   - token: BPE-coded token to decode.
    /// - Returns: String containing the decoded tokens.
    public static func decode(token: String) throws -> String {
        var buffer = [UInt8]()

        for scalar in token.unicodeScalars {
            buffer.append(BytePairEncoder.unicodeToBytes[scalar]!)
        }

        // TODO(#385): Support unicode strings with multibyte scalars.
        guard let decodedToken = String(bytes: buffer, encoding: .utf8) else {
            throw BPEError.unsupported
        }

        return decodedToken
    }
}
