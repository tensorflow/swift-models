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
import TensorFlow
import Datasets
import ModelSupport

// TODO: [AD] Avoid using token type embeddings for RoBERTa once optionals are supported in AD.
// TODO: [AD] Similarly for the embedding projection used in ALBERT.

/// BERT layer for encoding text.
///
/// - Sources:
///   - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](
///       https://arxiv.org/pdf/1810.04805.pdf).
///   - [RoBERTa: A Robustly Optimized BERT Pretraining Approach](
///       https://arxiv.org/pdf/1907.11692.pdf).
///   - [ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations](
///       https://arxiv.org/pdf/1909.11942.pdf).
public struct BERT: Module, Regularizable {
    // TODO: Convert to a generic constraint once TF-427 is resolved.
    public typealias Scalar = Float

    @noDerivative public let variant: Variant
    @noDerivative public let vocabulary: Vocabulary
    @noDerivative public let tokenizer: Tokenizer
    @noDerivative public let caseSensitive: Bool
    @noDerivative public let hiddenSize: Int
    @noDerivative public let hiddenLayerCount: Int
    @noDerivative public let attentionHeadCount: Int
    @noDerivative public let intermediateSize: Int
    @noDerivative public let intermediateActivation: Activation<Scalar>
    @noDerivative public let hiddenDropoutProbability: Scalar
    @noDerivative public let attentionDropoutProbability: Scalar
    @noDerivative public let maxSequenceLength: Int
    @noDerivative public let typeVocabularySize: Int
    @noDerivative public let initializerStandardDeviation: Scalar

    public var tokenEmbedding: Embedding<Scalar>
    public var tokenTypeEmbedding: Embedding<Scalar>
    public var positionEmbedding: Embedding<Scalar>
    public var embeddingLayerNorm: LayerNorm<Scalar>
    @noDerivative public var embeddingDropout: Dropout<Scalar>
    public var embeddingProjection: [Dense<Scalar>]
    public var encoderLayers: [TransformerEncoderLayer]

    public var regularizationValue: TangentVector {
        TangentVector(
            tokenEmbedding: tokenEmbedding.regularizationValue,
            tokenTypeEmbedding: tokenTypeEmbedding.regularizationValue,
            positionEmbedding: positionEmbedding.regularizationValue,
            embeddingLayerNorm: embeddingLayerNorm.regularizationValue,
            embeddingProjection: [Dense<Scalar>].TangentVector(
                embeddingProjection.map { $0.regularizationValue }),
            encoderLayers: [TransformerEncoderLayer].TangentVector(
                encoderLayers.map { $0.regularizationValue }))
    }

    /// TODO: [DOC] Add a documentation string and fix the parameter descriptions.
    ///
    /// - Parameters:
    ///   - hiddenSize: Size of the encoder and the pooling layers.
    ///   - hiddenLayerCount: Number of hidden layers in the encoder.
    ///   - attentionHeadCount: Number of attention heads for each encoder attention layer.
    ///   - intermediateSize: Size of the encoder "intermediate" (i.e., feed-forward) layer.
    ///   - intermediateActivation: Activation function used in the encoder and the pooling layers.
    ///   - hiddenDropoutProbability: Dropout probability for all fully connected layers in the
    ///     embeddings, the encoder, and the pooling layers.
    ///   - attentionDropoutProbability: Dropout probability for the attention scores.
    ///   - maxSequenceLength: Maximum sequence length that this model might ever be used with.
    ///     Typically, this is set to something large, just in case (e.g., 512, 1024, or 2048).
    ///   - typeVocabularySize: Vocabulary size for the token type IDs passed into the BERT model.
    ///   - initializerStandardDeviation: Standard deviation of the truncated Normal initializer
    ///     used for initializing all weight matrices.
    public init(
        variant: Variant,
        vocabulary: Vocabulary,
        tokenizer: Tokenizer,
        caseSensitive: Bool,
        hiddenSize: Int = 768,
        hiddenLayerCount: Int = 12,
        attentionHeadCount: Int = 12,
        intermediateSize: Int = 3072,
        intermediateActivation: @escaping Activation<Scalar> = gelu,
        hiddenDropoutProbability: Scalar = 0.1,
        attentionDropoutProbability: Scalar = 0.1,
        maxSequenceLength: Int = 512,
        typeVocabularySize: Int = 2,
        initializerStandardDeviation: Scalar = 0.02,
        useOneHotEmbeddings: Bool = false
    ) {
        self.variant = variant
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.caseSensitive = caseSensitive
        self.hiddenSize = hiddenSize
        self.hiddenLayerCount = hiddenLayerCount
        self.attentionHeadCount = attentionHeadCount
        self.intermediateSize = intermediateSize
        self.intermediateActivation = intermediateActivation
        self.hiddenDropoutProbability = hiddenDropoutProbability
        self.attentionDropoutProbability = attentionDropoutProbability
        self.maxSequenceLength = maxSequenceLength
        self.typeVocabularySize = typeVocabularySize
        self.initializerStandardDeviation = initializerStandardDeviation

        if case let .albert(_, hiddenGroupCount) = variant {
            precondition(
                hiddenGroupCount <= hiddenLayerCount,
                "The number of hidden groups must be smaller than the number of hidden layers.")
        }

        let embeddingSize: Int = {
            switch variant {
            case .bert, .roberta: return hiddenSize
            case let .albert(embeddingSize, _): return embeddingSize
            }
        }()

        self.tokenEmbedding = Embedding<Scalar>(
            vocabularySize: vocabulary.count,
            embeddingSize: embeddingSize,
            embeddingsInitializer: truncatedNormalInitializer(
                standardDeviation: Tensor<Scalar>(initializerStandardDeviation)),
            useOneHotEmbeddings: useOneHotEmbeddings)

        // The token type vocabulary will always be small and so we use the one-hot approach here
        // as it is always faster for small vocabularies.
        self.tokenTypeEmbedding = Embedding<Scalar>(
            vocabularySize: typeVocabularySize,
            embeddingSize: embeddingSize,
            embeddingsInitializer: truncatedNormalInitializer(
                standardDeviation: Tensor<Scalar>(initializerStandardDeviation)),
            useOneHotEmbeddings: true)

        // Since the position embeddings table is a learned variable, we create it using a (long)
        // sequence length, `maxSequenceLength`. The actual sequence length might be shorter than
        // this, for faster training of tasks that do not have long sequences. So,
        // `positionEmbedding` effectively contains an embedding table for positions
        // [0, 1, 2, ..., maxPositionEmbeddings - 1], and the current sequence may have positions
        // [0, 1, 2, ..., sequenceLength - 1], so we can just perform a slice.
        let positionPaddingIndex = { () -> Int in
            switch variant {
            case .bert, .albert: return 0
            case .roberta: return 2
            }
        }()
        self.positionEmbedding = Embedding(
            vocabularySize: positionPaddingIndex + maxSequenceLength,
            embeddingSize: embeddingSize,
            embeddingsInitializer: truncatedNormalInitializer(
                standardDeviation: Tensor(initializerStandardDeviation)),
            useOneHotEmbeddings: false)

        self.embeddingLayerNorm = LayerNorm<Scalar>(
            featureCount: hiddenSize,
            axis: -1)
        // TODO: Make dropout generic over the probability type.
        self.embeddingDropout = Dropout(probability: Double(hiddenDropoutProbability))

        // Add an embedding projection layer if using the ALBERT variant.
        self.embeddingProjection = {
            switch variant {
            case .bert, .roberta: return []
            case let .albert(embeddingSize, _):
                // TODO: [AD] Change to optional once supported.
                return [Dense<Scalar>(
                    inputSize: embeddingSize,
                    outputSize: hiddenSize,
                    weightInitializer: truncatedNormalInitializer(
                        standardDeviation: Tensor(initializerStandardDeviation)))]
            }
        }()

        switch variant {
        case .bert, .roberta:
        self.encoderLayers = (0..<hiddenLayerCount).map { _ in
            TransformerEncoderLayer(
                hiddenSize: hiddenSize,
                attentionHeadCount: attentionHeadCount,
                attentionQueryActivation: { $0 },
                attentionKeyActivation: { $0 },
                attentionValueActivation: { $0 },
                intermediateSize: intermediateSize,
                intermediateActivation: intermediateActivation,
                hiddenDropoutProbability: hiddenDropoutProbability,
                attentionDropoutProbability: attentionDropoutProbability)
        }
        case let .albert(_, hiddenGroupCount):
        self.encoderLayers = (0..<hiddenGroupCount).map { _ in
            TransformerEncoderLayer(
                hiddenSize: hiddenSize,
                attentionHeadCount: attentionHeadCount,
                attentionQueryActivation: { $0 },
                attentionKeyActivation: { $0 },
                attentionValueActivation: { $0 },
                intermediateSize: intermediateSize,
                intermediateActivation: intermediateActivation,
                hiddenDropoutProbability: hiddenDropoutProbability,
                attentionDropoutProbability: attentionDropoutProbability)
        }
        }
    }

    /// Preprocesses an array of text sequences and prepares them for processing with BERT.
    /// Preprocessing mainly consists of tokenization.
    ///
    /// - Parameters:
    ///   - sequences: Text sequences (not tokenized).
    ///   - maxSequenceLength: Maximum sequence length supported by the text perception module.
    ///     This is mainly used for padding the preprocessed sequences. If not provided, it
    ///     defaults to this model's maximum supported sequence length.
    ///   - tokenizer: Tokenizer to use while preprocessing.
    ///
    /// - Returns: Text batch that can be processed by BERT.
    public func preprocess(sequences: [String], maxSequenceLength: Int? = nil) -> TextBatch {
        let maxSequenceLength = maxSequenceLength ?? self.maxSequenceLength
        var sequences = sequences.map(tokenizer.tokenize)

        // Truncate the sequences based on the maximum allowed sequence length, while accounting
        // for the '[CLS]' token and for `sequences.count` '[SEP]' tokens. The following is a
        // simple heuristic which will truncate the longer sequence one token at a time. This makes 
        // more sense than truncating an equal percent of tokens from each sequence, since if one
        // sequence is very short then each token that is truncated likely contains more
        // information than respective tokens in longer sequences.
        var totalLength = sequences.map { $0.count }.reduce(0, +)
        let totalLengthLimit = { () -> Int in
            switch variant {
            case .bert, .albert: return maxSequenceLength - 1 - sequences.count
            case .roberta: return maxSequenceLength - 1 - 2 * sequences.count
            }
        }()
        while totalLength >= totalLengthLimit {
            let maxIndex = sequences.enumerated().max(by: { $0.1.count < $1.1.count })!.0
            sequences[maxIndex] = [String](sequences[maxIndex].dropLast())
            totalLength = sequences.map { $0.count }.reduce(0, +)
        }

        // The convention in BERT is:
        //   (a) For sequence pairs:
        //       tokens:       [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        //       tokenTypeIds: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        //   (b) For single sequences:
        //       tokens:       [CLS] the dog is hairy . [SEP]
        //       tokenTypeIds: 0     0   0   0  0     0 0
        // where "tokenTypeIds" are used to indicate whether this is the first sequence or the
        // second sequence. The embedding vectors for `tokenTypeId = 0` and `tokenTypeId = 1` were
        // learned during pre-training and are added to the WordPiece embedding vector (and
        // position vector). This is not *strictly* necessary since the [SEP] token unambiguously
        // separates the sequences. However, it makes it easier for the model to learn the concept
        // of sequences.
        //
        // For classification tasks, the first vector (corresponding to `[CLS]`) is used as the
        // "sentence embedding". Note that this only makes sense because the entire model is
        // fine-tuned under this assumption.
        var tokens = ["[CLS]"]
        var tokenTypeIds = [Int32(0)]
        for (sequenceId, sequence) in sequences.enumerated() {
            for token in sequence {
                tokens.append(token)
                tokenTypeIds.append(Int32(sequenceId))
            }
            tokens.append("[SEP]")
            tokenTypeIds.append(Int32(sequenceId))
            if case .roberta = variant, sequenceId < sequences.count - 1 {
                tokens.append("[SEP]")
                tokenTypeIds.append(Int32(sequenceId))
            }
        }
        let tokenIds = tokens.map { Int32(vocabulary.id(forToken: $0)!) }

        // The mask is set to `true` for real tokens and `false` for padding tokens. This is so
        // that only real tokens are attended to.
        let mask = [Int32](repeating: 1, count: tokenIds.count)

        return TextBatch(
            tokenIds: Tensor(tokenIds).expandingShape(at: 0),
            tokenTypeIds: Tensor(tokenTypeIds).expandingShape(at: 0),
            mask: Tensor(mask).expandingShape(at: 0))
    }

    @differentiable(wrt: self)
    public func callAsFunction(_ input: TextBatch) -> Tensor<Scalar> {
        let sequenceLength = input.tokenIds.shape[1]
        let variant = withoutDerivative(at: self.variant)

        // Compute the input embeddings and apply layer normalization and dropout on them.
        let tokenEmbeddings = tokenEmbedding(input.tokenIds)
        let tokenTypeEmbeddings = tokenTypeEmbedding(input.tokenTypeIds)
        let positionPaddingIndex: Int
        switch variant {
        case .bert, .albert: positionPaddingIndex = 0
        case .roberta: positionPaddingIndex = 2
        }
        let positionEmbeddings = positionEmbedding.embeddings.slice(
            lowerBounds: [positionPaddingIndex, 0],
            upperBounds: [positionPaddingIndex + sequenceLength, -1]
        ).expandingShape(at: 0)
        var embeddings = tokenEmbeddings + positionEmbeddings

        // Add token type embeddings if needed, based on which BERT variant is being used.
        switch variant {
        case .bert, .albert: embeddings = embeddings + tokenTypeEmbeddings
        case .roberta: break
        }

        embeddings = embeddingLayerNorm(embeddings)
        embeddings = embeddingDropout(embeddings)

        if case .albert = variant {
            embeddings = embeddingProjection[0](embeddings)
        }

        // Create an attention mask for the inputs with shape
        // `[batchSize, sequenceLength, sequenceLength]`.
        let attentionMask = createAttentionMask(forTextBatch: input)

        // We keep the representation as a 2-D tensor to avoid reshaping it back and forth from a
        // 3-D tensor to a 2-D tensor. Reshapes are normally free on GPUs/CPUs but may not be free
        // on TPUs, and so we want to minimize them to help the optimizer.
        var transformerInput = embeddings.reshapedToMatrix()
        let batchSize = embeddings.shape[0]

        // Run the stacked transformer.
        switch variant {
        case .bert, .roberta:
            for layerIndex in 0..<(withoutDerivative(at: encoderLayers) { $0.count }) {
                transformerInput = encoderLayers[layerIndex](TransformerInput(
                sequence: transformerInput,
                attentionMask: attentionMask,
                batchSize: batchSize))
        }
        case let .albert(_, hiddenGroupCount):
            let groupsPerLayer = Float(hiddenGroupCount) / Float(hiddenLayerCount)
            for layerIndex in 0..<hiddenLayerCount {
                let groupIndex = Int(Float(layerIndex) * groupsPerLayer)
                transformerInput = encoderLayers[groupIndex](TransformerInput(
                    sequence: transformerInput,
                    attentionMask: attentionMask,
                    batchSize: batchSize))
            }
        }

        // Reshape back to the original tensor shape.
        return transformerInput.reshapedFromMatrix(originalShape: embeddings.shape)
    }
}

extension BERT {
    public enum Variant: CustomStringConvertible {
        /// - Source: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](
        ///             https://arxiv.org/pdf/1810.04805.pdf).
        case bert

        /// - Source: [RoBERTa: A Robustly Optimized BERT Pretraining Approach](
        ///             https://arxiv.org/pdf/1907.11692.pdf).
        case roberta

        /// - Source: [ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations](
        ///             https://arxiv.org/pdf/1909.11942.pdf).
        case albert(embeddingSize: Int, hiddenGroupCount: Int)

        public var description: String {
            switch self {
            case .bert:
                return "bert"
            case .roberta:
                return "roberta"
            case let .albert(embeddingSize, hiddenGroupCount):
                return "albert-E-\(embeddingSize)-G-\(hiddenGroupCount)"
            }
        }
    }
}

//===-----------------------------------------------------------------------------------------===//
// Tokenization
//===-----------------------------------------------------------------------------------------===//

/// BERT tokenizer that is simply defined as the composition of the basic text tokenizer and the
/// greedy subword tokenizer.
public struct BERTTokenizer: Tokenizer {
    public let caseSensitive: Bool
    public let vocabulary: Vocabulary
    public let unknownToken: String
    public let maxTokenLength: Int?

    private let basicTextTokenizer: BasicTokenizer
    private let greedySubwordTokenizer: GreedySubwordTokenizer

    /// Creates a BERT tokenizer.
    ///
    /// - Parameters:
    ///   - vocabulary: Vocabulary containing all supported tokens.
    ///   - caseSensitive: Specifies whether or not to ignore case.
    ///   - unknownToken: Token used to represent unknown tokens (i.e., tokens that are not in the
    ///     provided vocabulary or whose length is longer than `maxTokenLength`).
    ///   - maxTokenLength: Maximum allowed token length.
    public init(
        vocabulary: Vocabulary,
        caseSensitive: Bool = false,
        unknownToken: String = "[UNK]",
        maxTokenLength: Int? = nil
    ) {
        self.caseSensitive = caseSensitive
        self.vocabulary = vocabulary
        self.unknownToken = unknownToken
        self.maxTokenLength = maxTokenLength
        self.basicTextTokenizer = BasicTokenizer(caseSensitive: caseSensitive)
        self.greedySubwordTokenizer = GreedySubwordTokenizer(
            vocabulary: vocabulary,
            unknownToken: unknownToken,
            maxTokenLength: maxTokenLength)
    }

    public func tokenize(_ text: String) -> [String] {
        basicTextTokenizer.tokenize(text).flatMap(greedySubwordTokenizer.tokenize)
    }
}

/// RoBERTa tokenizer that is simply defined as the composition of the basic text tokenizer and the
/// byte pair encoder.
public struct RoBERTaTokenizer: Tokenizer {
    public let caseSensitive: Bool
    public let unknownToken: String

    private let bytePairEncoder: BytePairEncoder

    private let tokenizationRegex: NSRegularExpression = try! NSRegularExpression(
        pattern: "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+")

    /// Creates a full text tokenizer.
    ///
    /// - Parameters:
    ///   - bytePairEncoder: Byte pair encoder to use.
    ///   - caseSensitive: Specifies whether or not to ignore case.
    ///   - unknownToken: Token used to represent unknown tokens (i.e., tokens that are not in the
    ///     provided vocabulary or whose length is longer than `maxTokenLength`).
    public init(
        bytePairEncoder: BytePairEncoder,
        caseSensitive: Bool = false,
        unknownToken: String = "[UNK]"
    ) {
        self.caseSensitive = caseSensitive
        self.unknownToken = unknownToken
        self.bytePairEncoder = bytePairEncoder
    }

    public func tokenize(_ text: String) -> [String] {
        let matches = tokenizationRegex.matches(
            in: text,
            range: NSRange(text.startIndex..., in: text))
        return matches.flatMap { match -> [String] in
            if let range = Range(match.range, in: text) {
                return bytePairEncoder.encode(token: String(text[range]))
            } else {
                return []
            }
        }
    }
}

//===-----------------------------------------------------------------------------------------===//
// Pre-Trained Models
//===-----------------------------------------------------------------------------------------===//

extension BERT {
    public enum PreTrainedModel {
        case bertBase(cased: Bool, multilingual: Bool)
        case bertLarge(cased: Bool, wholeWordMasking: Bool)
        case robertaBase
        case robertaLarge
        case albertBase
        case albertLarge
        case albertXLarge
        case albertXXLarge

        /// The name of this pre-trained model.
        public var name: String {
            switch self {
            case .bertBase(false, false): return "BERT Base Uncased"
            case .bertBase(true, false): return "BERT Base Cased"
            case .bertBase(false, true): return "BERT Base Multilingual Uncased"
            case .bertBase(true, true): return "BERT Base Multilingual Cased"
            case .bertLarge(false, false): return "BERT Large Uncased"
            case .bertLarge(true, false): return "BERT Large Cased"
            case .bertLarge(false, true): return "BERT Large Whole-Word-Masking Uncased"
            case .bertLarge(true, true): return "BERT Large Whole-Word-Masking Cased"
            case .robertaBase: return "RoBERTa Base"
            case .robertaLarge: return "RoBERTa Large"
            case .albertBase: return "ALBERT Base"
            case .albertLarge: return "ALBERT Large"
            case .albertXLarge: return "ALBERT xLarge"
            case .albertXXLarge: return "ALBERT xxLarge"
            }
        }

        /// The URL where this pre-trained model can be downloaded from.
        public var url: URL {
            let bertPrefix = "https://storage.googleapis.com/bert_models/2018_"
            let robertaPrefix = "https://storage.googleapis.com/s4tf-hosted-binaries/checkpoints/Text/RoBERTa"
            let albertPrefix = "https://storage.googleapis.com/tfhub-modules/google/albert"
            switch self {
            case .bertBase(false, false):
                return URL(string: "\(bertPrefix)10_18/\(subDirectory).zip")!
            case .bertBase(true, false):
                return URL(string: "\(bertPrefix)10_18/\(subDirectory).zip")!
            case .bertBase(false, true):
                return URL(string: "\(bertPrefix)11_03/\(subDirectory).zip")!
            case .bertBase(true, true):
                return URL(string: "\(bertPrefix)11_23/\(subDirectory).zip")!
            case .bertLarge(false, false):
                return URL(string: "\(bertPrefix)10_18/\(subDirectory).zip")!
            case .bertLarge(true, false):
                return URL(string: "\(bertPrefix)10_18/\(subDirectory).zip")!
            case .bertLarge(false, true):
                return URL(string: "\(bertPrefix)05_30/\(subDirectory).zip")!
            case .bertLarge(true, true):
                return URL(string: "\(bertPrefix)05_30/\(subDirectory).zip")!
            case .robertaBase:
                return URL(string: "\(robertaPrefix)/base.zip")!
            case .robertaLarge:
                return URL(string: "\(robertaPrefix)/large.zip")!
            case .albertBase, .albertLarge, .albertXLarge, .albertXXLarge:
                return URL(string: "\(albertPrefix)_\(subDirectory)/1.tar.gz")!
            }
        }

        public var variant: Variant {
            switch self {
            case .bertBase, .bertLarge:
                return .bert
            case .robertaBase, .robertaLarge:
                return .roberta
            case .albertBase, .albertLarge, .albertXLarge, .albertXXLarge:
                return .albert(embeddingSize: 128, hiddenGroupCount: 1)
            }
        }

        public var caseSensitive: Bool {
            switch self {
            case let .bertBase(cased, _): return cased
            case let .bertLarge(cased, _): return cased
            case .robertaBase, .robertaLarge: return true
            case .albertBase, .albertLarge, .albertXLarge, .albertXXLarge: return false
            }
        }

        public var hiddenSize: Int {
            switch self {
            case .bertBase: return 768
            case .bertLarge: return 1024
            case .robertaBase: return 768
            case .robertaLarge: return 1024
            case .albertBase: return 768
            case .albertLarge: return 1024
            case .albertXLarge: return 2048
            case .albertXXLarge: return 4096
            }
        }

        public var hiddenLayerCount: Int {
            switch self {
            case .bertBase: return 12
            case .bertLarge: return 24
            case .robertaBase: return 12
            case .robertaLarge: return 24
            case .albertBase: return 12
            case .albertLarge: return 24
            case .albertXLarge: return 24
            case .albertXXLarge: return 12
            }
        }

        public var attentionHeadCount: Int {
            switch self {
            case .bertBase: return 12
            case .bertLarge: return 16
            case .robertaBase: return 12
            case .robertaLarge: return 16
            case .albertBase: return 12
            case .albertLarge: return 16
            case .albertXLarge: return 16
            case .albertXXLarge: return 64
            }
        }

        public var intermediateSize: Int {
            switch self {
            case .bertBase: return 3072
            case .bertLarge: return 4096
            case .robertaBase: return 3072
            case .robertaLarge: return 4096
            case .albertBase: return 3072
            case .albertLarge: return 4096
            case .albertXLarge: return 8192
            case .albertXXLarge: return 16384
            }
        }

        /// The sub-directory of this pre-trained model.
        internal var subDirectory: String {
            switch self {
            case .bertBase(false, false): return "uncased_L-12_H-768_A-12"
            case .bertBase(true, false): return "cased_L-12_H-768_A-12"
            case .bertBase(false, true): return "multilingual_L-12_H-768_A-12"
            case .bertBase(true, true): return "multi_cased_L-12_H-768_A-12"
            case .bertLarge(false, false): return "uncased_L-24_H-1024_A-16"
            case .bertLarge(true, false): return "cased_L-24_H-1024_A-16"
            case .bertLarge(false, true): return "wwm_uncased_L-24_H-1024_A-16"
            case .bertLarge(true, true): return "wwm_cased_L-24_H-1024_A-16"
            case .robertaBase: return "base"
            case .robertaLarge: return "large"
            case .albertBase: return "base"
            case .albertLarge: return "large"
            case .albertXLarge: return "xLarge"
            case .albertXXLarge: return "xxLarge"
            }
        }

        /// Loads this pre-trained BERT model from the specified directory.
        ///
        /// - Note: This function will download the pre-trained model files to the specified
        //    directory, if they are not already there.
        ///
        /// - Parameters:
        ///   - directory: Directory to load the pretrained model from.
        public func load(from directory: URL) throws -> BERT {
            print("Loading BERT pre-trained model '\(name)'.")
            let directory = directory.appendingPathComponent(variant.description, isDirectory: true)
            try maybeDownload(to: directory)

            // Load the appropriate vocabulary file.
            let vocabulary: Vocabulary = {
                switch self {
                case .bertBase, .bertLarge:
                    let vocabularyURL = directory
                        .appendingPathComponent(subDirectory)
                        .appendingPathComponent("vocab.txt")
                    return try! Vocabulary(fromFile: vocabularyURL)
                case .robertaBase, .robertaLarge:
                    let vocabularyURL = directory
                        .appendingPathComponent(subDirectory)
                        .appendingPathComponent("vocab.json")
                    let dictionaryURL = directory
                        .appendingPathComponent(subDirectory)
                        .appendingPathComponent("dict.txt")
                    return try! Vocabulary(
                        fromRoBERTaJSONFile: vocabularyURL,
                        dictionaryFile: dictionaryURL)
                case .albertBase, .albertLarge, .albertXLarge, .albertXXLarge:
                    let vocabularyURL = directory
                        .appendingPathComponent(subDirectory)
                        .appendingPathComponent("assets")
                        .appendingPathComponent("30k-clean.model")
                    return try! Vocabulary(fromSentencePieceModel: vocabularyURL)
                }
            }()

            // Create the tokenizer and load any necessary files.
            let tokenizer: Tokenizer = try {
                switch self {
                case .bertBase, .bertLarge, .albertBase, .albertLarge, .albertXLarge, .albertXXLarge:
                    return BERTTokenizer(
                        vocabulary: vocabulary,
                        caseSensitive: caseSensitive,
                        unknownToken: "[UNK]",
                        maxTokenLength: nil)
                case .robertaBase, .robertaLarge:
                    let mergePairsFileURL = directory
                        .appendingPathComponent(subDirectory)
                        .appendingPathComponent("merges.txt")
                    let mergePairs = [BytePairEncoder.Pair: Int](
                        uniqueKeysWithValues:
                            (try String(contentsOfFile: mergePairsFileURL.path, encoding: .utf8))
                                .components(separatedBy: .newlines)
                                .dropFirst()
                                .enumerated()
                                .compactMap { (index, line) -> (BytePairEncoder.Pair, Int)? in
                                    let lineParts = line.split(separator: " ")
                                    if lineParts.count < 2 { return nil }
                                    return (
                                        BytePairEncoder.Pair(
                                            String(lineParts[0]),
                                            String(lineParts[1])),
                                        index)
                                })
                    return RoBERTaTokenizer(
                        bytePairEncoder: BytePairEncoder(
                            vocabulary: vocabulary,
                            mergePairs: mergePairs),
                        caseSensitive: caseSensitive,
                        unknownToken: "[UNK]")
                }
            }()

            // Create a BERT model.
            var model = BERT(
                variant: variant,
                vocabulary: vocabulary,
                tokenizer: tokenizer,
                caseSensitive: caseSensitive,
                hiddenSize: hiddenSize,
                hiddenLayerCount: hiddenLayerCount,
                attentionHeadCount: attentionHeadCount,
                intermediateSize: intermediateSize,
                intermediateActivation: gelu,
                hiddenDropoutProbability: 0.1,
                attentionDropoutProbability: 0.1,
                maxSequenceLength: 512,
                typeVocabularySize: 2,
                initializerStandardDeviation: 0.02,
                useOneHotEmbeddings: false)

            // Load the pre-trained model checkpoint.
            switch self {
            case .bertBase, .bertLarge:
                model.load(fromTensorFlowCheckpoint: directory
                    .appendingPathComponent(subDirectory)
                    .appendingPathComponent("bert_model.ckpt"))
            case .robertaBase, .robertaLarge:
                model.load(fromTensorFlowCheckpoint: directory
                    .appendingPathComponent(subDirectory)
                    .appendingPathComponent("roberta_\(subDirectory).ckpt"))
            case .albertBase, .albertLarge, .albertXLarge, .albertXXLarge:
                model.load(fromTensorFlowCheckpoint: directory
                    .appendingPathComponent(subDirectory)
                    .appendingPathComponent("variables")
                    .appendingPathComponent("variables"))
            }
            return model
        }

        /// Downloads this pre-trained model to the specified directory, if it's not already there.
        public func maybeDownload(to directory: URL) throws {
            switch self {
            case .bertBase, .bertLarge, .robertaBase, .robertaLarge:
                // Download and extract the pretrained model, if necessary.
                DatasetUtilities.downloadResource(filename: "\(subDirectory)", fileExtension: "zip",
                                                  remoteRoot: url.deletingLastPathComponent(),
                                                  localStorageDirectory: directory)
            case .albertBase, .albertLarge, .albertXLarge, .albertXXLarge:
                // Download the model, if necessary.
                let compressedFileURL = directory.appendingPathComponent("\(subDirectory).tar.gz")
                try download(from: url, to: compressedFileURL)

                // Extract the data, if necessary.
                let extractedDirectoryURL = directory.appendingPathComponent(subDirectory)
                if !FileManager.default.fileExists(atPath: extractedDirectoryURL.path) {
                    try extract(tarGZippedFileAt: compressedFileURL, to: extractedDirectoryURL)
                }
            }
        }
  }

    /// Loads a BERT model from the provided TensorFlow checkpoint file into this BERT model.
    ///
    /// - Parameters:
    ///   - fileURL: Path to the checkpoint file. Note that TensorFlow checkpoints typically
    ///     consist of multiple files (e.g., `bert_model.ckpt.index`, `bert_model.ckpt.meta`, and
    ///     `bert_model.ckpt.data-00000-of-00001`). In this case, the file URL should be specified
    ///     as their common prefix (e.g., `bert_model.ckpt`).
    public mutating func load(fromTensorFlowCheckpoint fileURL: URL) {
        let checkpointReader = TensorFlowCheckpointReader(checkpointPath: fileURL.path)
        tokenEmbedding.embeddings =
            Tensor(checkpointReader.loadTensor(named: "bert/embeddings/word_embeddings"))
        positionEmbedding.embeddings =
            Tensor(checkpointReader.loadTensor(named: "bert/embeddings/position_embeddings"))
        embeddingLayerNorm.offset =
            Tensor(checkpointReader.loadTensor(named: "bert/embeddings/LayerNorm/beta"))
        embeddingLayerNorm.scale =
            Tensor(checkpointReader.loadTensor(named: "bert/embeddings/LayerNorm/gamma"))
        switch variant {
        case .bert, .albert:
            tokenTypeEmbedding.embeddings =
                Tensor(checkpointReader.loadTensor(named: "bert/embeddings/token_type_embeddings"))
        case .roberta: ()
        }
        switch variant {
        case .bert, .roberta:
            for layerIndex in encoderLayers.indices {
                let prefix = "bert/encoder/layer_\(layerIndex)"
                encoderLayers[layerIndex].multiHeadAttention.queryWeight =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention/self/query/kernel"))
                encoderLayers[layerIndex].multiHeadAttention.queryBias =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention/self/query/bias"))
                encoderLayers[layerIndex].multiHeadAttention.keyWeight =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention/self/key/kernel"))
                encoderLayers[layerIndex].multiHeadAttention.keyBias =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention/self/key/bias"))
                encoderLayers[layerIndex].multiHeadAttention.valueWeight =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention/self/value/kernel"))
                encoderLayers[layerIndex].multiHeadAttention.valueBias =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention/self/value/bias"))
                encoderLayers[layerIndex].attentionWeight =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention/output/dense/kernel"))
                encoderLayers[layerIndex].attentionBias =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention/output/dense/bias"))
                encoderLayers[layerIndex].attentionLayerNorm.offset =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention/output/LayerNorm/beta"))
                encoderLayers[layerIndex].attentionLayerNorm.scale =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention/output/LayerNorm/gamma"))
                encoderLayers[layerIndex].intermediateWeight =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/intermediate/dense/kernel"))
                encoderLayers[layerIndex].intermediateBias =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/intermediate/dense/bias"))
                encoderLayers[layerIndex].outputWeight =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/output/dense/kernel"))
                encoderLayers[layerIndex].outputBias =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/output/dense/bias"))
                encoderLayers[layerIndex].outputLayerNorm.offset =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/output/LayerNorm/beta"))
                encoderLayers[layerIndex].outputLayerNorm.scale =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/output/LayerNorm/gamma"))
            }
        case .albert:
            embeddingProjection[0].weight =
                Tensor(checkpointReader.loadTensor(
                    named: "bert/encoder/embedding_hidden_mapping_in/kernel"))
            embeddingProjection[0].bias =
                Tensor(checkpointReader.loadTensor(
                    named: "bert/encoder/embedding_hidden_mapping_in/bias"))
            for layerIndex in encoderLayers.indices {
                let prefix = "bert/encoder/transformer/group_\(layerIndex)/inner_group_0"
                encoderLayers[layerIndex].multiHeadAttention.queryWeight =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention_1/self/query/kernel"))
                encoderLayers[layerIndex].multiHeadAttention.queryBias =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention_1/self/query/bias"))
                encoderLayers[layerIndex].multiHeadAttention.keyWeight =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention_1/self/key/kernel"))
                encoderLayers[layerIndex].multiHeadAttention.keyBias =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention_1/self/key/bias"))
                encoderLayers[layerIndex].multiHeadAttention.valueWeight =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention_1/self/value/kernel"))
                encoderLayers[layerIndex].multiHeadAttention.valueBias =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention_1/self/value/bias"))
                encoderLayers[layerIndex].attentionWeight =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention_1/output/dense/kernel"))
                encoderLayers[layerIndex].attentionBias =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention_1/output/dense/bias"))
                encoderLayers[layerIndex].attentionLayerNorm.offset =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/LayerNorm/beta"))
                encoderLayers[layerIndex].attentionLayerNorm.scale =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/LayerNorm/gamma"))
                encoderLayers[layerIndex].intermediateWeight =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/ffn_1/intermediate/dense/kernel"))
                encoderLayers[layerIndex].intermediateBias =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/ffn_1/intermediate/dense/bias"))
                encoderLayers[layerIndex].outputWeight =
                    Tensor(checkpointReader.loadTensor(
                        named: "\(prefix)/ffn_1/intermediate/output/dense/kernel"))
                encoderLayers[layerIndex].outputBias =
                    Tensor(checkpointReader.loadTensor(
                        named: "\(prefix)/ffn_1/intermediate/output/dense/bias"))
                encoderLayers[layerIndex].outputLayerNorm.offset =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/LayerNorm_1/beta"))
                encoderLayers[layerIndex].outputLayerNorm.scale =
                    Tensor(checkpointReader.loadTensor(named: "\(prefix)/LayerNorm_1/gamma"))
            }
        }
    }
}

extension Vocabulary {
    internal init(fromRoBERTaJSONFile fileURL: URL, dictionaryFile dictionaryURL: URL) throws {
        let dictionary = [Int: Int](
            uniqueKeysWithValues:
                (try String(contentsOfFile: dictionaryURL.path, encoding: .utf8))
                    .components(separatedBy: .newlines)
                    .compactMap { line in
                        let lineParts = line.split(separator: " ")
                        if lineParts.count < 1 { return nil }
                        return Int(lineParts[0])
                    }
                    .enumerated()
                    .map { ($1, $0 + 4) })
        let json = try String(contentsOfFile: fileURL.path)
        var tokensToIds = try JSONDecoder().decode(
            [String: Int].self,
            from: json.data(using: .utf8)!)
        tokensToIds = tokensToIds.mapValues { dictionary[$0]! }
        tokensToIds.merge(["[CLS]": 0, "[PAD]": 1, "[SEP]": 2, "[UNK]": 3]) { (_, new) in new }
        self.init(tokensToIds: tokensToIds)
    }
}
