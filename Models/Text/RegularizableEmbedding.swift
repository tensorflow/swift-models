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

import TensorFlow

/// An embedding layer.
///
/// `Embedding` is effectively a lookup table that maps indices from a fixed vocabulary to
/// fixed-size vector representations, e.g. `[[0], [3]] -> [[0.25, 0.1], [0.6, -0.2]]`.
public struct RegularizableEmbedding<Scalar: TensorFlowFloatingPoint>: Module, Regularizable {
    /// Embeddings lookup table with shape `[vocabularySize, embeddingSize]`.
    public var embeddings: Tensor<Scalar>

    /// Number of distinct indices (e.g., words) in the vocabulary.
    @noDerivative public let vocabularySize: Int

    /// Embedding size.
    @noDerivative public let embeddingSize: Int

    /// Indicator specifying whether to use the one-hot method for embedding lookups, which consists
    /// of a matrix multiplication. If `false`, a regular gather operation is used. The one-hot
    /// approach is better for TPUs.
    @noDerivative public let useOneHotEmbeddings: Bool

    public var regularizationValue: TangentVector {
        TangentVector(embeddings: embeddings)
    }

    /// Creates an embedding layer.
    ///
    /// - Parameters:
    ///   - vocabularySize: Number of distinct indices (e.g., words) in the vocabulary.
    ///   - embeddingSize: Embedding size.
    ///   - embeddingsInitializer: Initializer for the embedding table.
    ///   - useOneHotEmbeddings: If `true`, the one-hot method is used for embedding lookups, which
    ///     consists of a matrix multiplication. Otherwise, a regular gather operation is used. The
    ///     one-hot approach is better for TPUs.
    public init(
        vocabularySize: Int,
        embeddingSize: Int,
        embeddingsInitializer: ParameterInitializer<Scalar> = defaultInitializer,
        useOneHotEmbeddings: Bool = false
    ) {
        self.embeddings = embeddingsInitializer([vocabularySize, embeddingSize])
        self.vocabularySize = vocabularySize
        self.embeddingSize = embeddingSize
        self.useOneHotEmbeddings = useOneHotEmbeddings
    }

    /// Returns the output obtained from applying this layer to the given input.
    ///
    /// Note that if the input tensor has rank 3 and shape `[batchSize, length, depth]`, the output
    /// tensor will have shape `[batchSize, length, depth * embeddingSize]`. If it has rank 2 and
    /// shape `[batchSize, length]`, the output tensor will have shape
    /// `[batchSize, length, embeddingSize]`.
    ///
    /// - Parameter input: Input to this layer.
    /// - Returns: Output of this layer.
    /// - Precondition: The input tensor rank must be either 2 or 3.
    @differentiable(wrt: self)
    public func callAsFunction(_ input: Tensor<Int32>) -> Tensor<Scalar> {
        precondition(input.rank > 1 && input.rank < 4, "The input tensor rank must be either 2 or 3.")
        let flatInput = input.flattened()
        // TODO: [TPU] `useOneHotEmbeddings` is currently being ignored due to an AutoDiff bug.
        // let output = useOneHotEmbeddings ?
        //   matmul(Tensor<Scalar>(oneHotAtIndices: flatInput, depth: vocabularySize), embeddings) :
        //   embeddings.gathering(atIndices: flatInput)
        let output = embeddings.gathering(atIndices: flatInput)
        let outputShape = input.rank == 2 ?
            TensorShape(input.shape.dimensions + [embeddingSize]) :
            TensorShape(input.shape.dimensions[0..<2] + [input.shape[2] * embeddingSize])
        return output.reshaped(to: outputShape)
    }
}

extension RegularizableEmbedding {
    /// Default initializer to use for the embeddings lookup table.
    public static var defaultInitializer: ParameterInitializer<Scalar> {
        truncatedNormalInitializer(standardDeviation: Tensor<Scalar>(0.02))
    }
}
