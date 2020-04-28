// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
import ModelSupport

/// The DLRM model is parameterized to support multiple ways of combining the latent spaces of the inputs.
public enum InteractionType {
    /// Concatenate the tensors representing the latent spaces of the inputs together.
    ///
    /// This operation is the fastest, but does not encode any higher-order feature interactions.
    case concatenate

    /// Compute the dot product of every input latent space with every other input latent space
    /// and concatenate the results.
    ///
    /// This computation encodes 2nd-order feature interactions.
    ///
    /// If `selfInteraction` is true, 2nd-order self-interactions occur. If false,
    /// self-interactions are excluded.
    case dot(selfInteraction: Bool)
}

/// DLRM is the deep learning recommendation model and is used for recommendation tasks.
///
/// DLRM handles inputs that contain both sparse categorical data and numerical data.
/// Original Paper:
/// "Deep Learning Recommendation Model for Personalization and Recommendation Systems"
/// Maxim Naumov et al.
/// https://arxiv.org/pdf/1906.00091.pdf
public struct DLRM: TypedModule {
    public typealias Scalar = Float

    public var mlpBottom: Self.MLP
    public var mlpTop: Self.MLP
    public var latentFactors: [Self.Embedding]
    @noDerivative public let nDense: Int
    @noDerivative public let interaction: InteractionType

    /// Randomly initialize a DLRM model from the given hyperparameters.
    ///
    /// - Parameters:
    ///    - nDense: The number of continuous or dense inputs for each example.
    ///    - mSpa: The "width" of all embedding tables.
    ///    - lnEmb: Defines the "heights" of each of each embedding table.
    ///    - lnBot: The size of the hidden layers in the bottom MLP.
    ///    - lnTop: The size of the hidden layers in the top MLP.
    ///    - interaction: The type of interactions between the hidden  features.
    public init(nDense: Int, mSpa: Int, lnEmb: [Int], lnBot: [Int], lnTop: [Int],
                interaction: InteractionType = .concatenate) {
        self.nDense = nDense
        mlpBottom = MLP(dims: [nDense] + lnBot)
        let topInput = lnEmb.count * mSpa + lnBot.last!
        mlpTop = MLP(dims: [topInput] + lnTop + [1], sigmoidLastLayer: true)
        latentFactors = lnEmb.map { embeddingSize -> Embedding<Float> in
            // Use a random uniform initialization to match the reference implementation.
            let weights = Tensor(
                randomUniform: [embeddingSize, mSpa],
                lowerBound: Tensor(Float(-1.0)/Float(embeddingSize)),
                upperBound: Tensor(Float(1.0)/Float(embeddingSize)))
            return Embedding(embeddings: weights)
        }
        self.interaction = interaction
    }

    @differentiable
    public func callAsFunction(_ input: DLRMInput) -> Tensor<Float> {
        callAsFunction(denseInput: input.dense, sparseInput: input.sparse)
    }

    @differentiable(wrt: self)
    public func callAsFunction(
        denseInput: Tensor<Float>,
        sparseInput: [Tensor<Int32>]
    ) -> Tensor<Float> {
        precondition(denseInput.shape.last! == nDense)
        precondition(sparseInput.count == latentFactors.count)
        let denseEmbVec = mlpBottom(denseInput)
        let sparseEmbVecs = computeEmbeddings(sparseInputs: sparseInput,
                                              latentFactors: latentFactors)
        let topInput = computeInteractions(
            denseEmbVec: denseEmbVec, sparseEmbVecs: sparseEmbVecs)
        let prediction = mlpTop(topInput)

        // TODO: loss threshold clipping
        return prediction.reshaped(to: [-1])
    }

    @differentiable(wrt: (denseEmbVec, sparseEmbVecs))
    public func computeInteractions(
        denseEmbVec:  Tensor<Float>,
        sparseEmbVecs: [Tensor<Float>]
    ) -> Tensor<Float> {
        switch self.interaction {
        case .concatenate:
            return Tensor(concatenating: sparseEmbVecs + [denseEmbVec], alongAxis: 1)
        case let .dot(selfInteraction):
            let batchSize = denseEmbVec.shape[0]
            let allEmbeddings = Tensor(
                concatenating: sparseEmbVecs + [denseEmbVec],
                alongAxis: 1).reshaped(to: [batchSize, -1, denseEmbVec.shape[1]])
            // Use matmul to efficiently compute all dot products
            let higherOrderInteractions = matmul(
                allEmbeddings, allEmbeddings.transposed(permutation: 0, 2, 1))
            // Gather relevant indices
            let flattenedHigherOrderInteractions = higherOrderInteractions.reshaped(
                to: [batchSize, -1])
            let desiredIndices = makeIndices(
                n: Int32(higherOrderInteractions.shape[1]),
                selfInteraction: selfInteraction)
            let desiredInteractions =
                flattenedHigherOrderInteractions.batchGathering(atIndices: desiredIndices)
            return Tensor(concatenating: [desiredInteractions, denseEmbVec], alongAxis: 1)
        }
    }
}

/// DLRMInput represents the categorical and numerical input
public struct DLRMInput {

    /// dense represents a mini-batch of continuous inputs.
    ///
    /// It should have shape `[batchSize, continuousCount]`
    public let dense: Tensor<Float>

    /// sparse represents the categorical inputs to the mini-batch.
    ///
    /// The array should be of length `numCategoricalInputs`.
    /// Each tensor within the array should be a vector of length `batchSize`.
    public let sparse: [Tensor<Int32>]
}

// Work-around for lack of inout support
fileprivate func computeEmbeddings(
    sparseInputs: [Tensor<Int32>],
    latentFactors: [Embedding<Float>]
) -> [Tensor<Float>] {
    var sparseEmbVecs: [Tensor<Float>] = []
    for i in 0..<sparseInputs.count {
        sparseEmbVecs.append(latentFactors[i](sparseInputs[i]))
    }
    return sparseEmbVecs
}

// TODO: remove computeEmbeddingsVJP once inout differentiation is supported!
@derivative(of: computeEmbeddings)
fileprivate func computeEmbeddingsVJP(
    sparseInput: [Tensor<Int32>],
    latentFactors: [Embedding<Float>]
) -> (
    value: [Tensor<Float>],
    pullback: (Array<Tensor<Float>>.TangentVector) -> Array<Embedding<Float>>.TangentVector
) {
    var sparseEmbVecs = [Tensor<Float>]()
    var pullbacks = [(Tensor<Float>.TangentVector) -> Embedding<Float>.TangentVector]()
    for i in 0..<sparseInput.count {
        let (fwd, pullback) = valueWithPullback(at: latentFactors[i]) { $0(sparseInput[i]) }
        sparseEmbVecs.append(fwd)
        pullbacks.append(pullback)
    }
    return (
        value: sparseEmbVecs,
        pullback: { v in
            let arr = zip(v, pullbacks).map { $0.1($0.0) }
            return Array.DifferentiableView(arr)
        }
    )
}

/// Compute indices for the upper triangle (optionally including the diagonal) in a flattened representation.
///
/// - Parameter n: Size of the square matrix.
/// - Parameter selfInteraction: Include the diagonal iff selfInteraction is true.
fileprivate func makeIndices(n: Int32, selfInteraction: Bool) -> Tensor<Int32> {
    let interactionOffset: Int32
    if selfInteraction {
        interactionOffset = 0
    } else {
        interactionOffset = 1
    }
    var result = [Int32]()
    for i in 0..<n {
        for j in (i + interactionOffset)..<n {
            result.append(i*n + j)
        }
    }
    return Tensor(result)
}
