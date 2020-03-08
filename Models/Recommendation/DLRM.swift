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

/// DLRM is the deep learning recommendation model and is used for recommendation tasks.
///
/// DLRM handles inputs that contain both sparse categorical data and numerical data.
/// Original Paper:
/// "Deep Learning Recommendation Model for Personalization and Recommendation Systems"
/// Maxim Naumov et al.
/// https://arxiv.org/pdf/1906.00091.pdf
public struct DLRM: Module {

    public var mlpBottom: MLP
    public var mlpTop: MLP
    public var latentFactors: [Embedding<Float>]
    @noDerivative public let nDense: Int

    public init(nDense: Int, mSpa: Int, lnEmb: [Int], lnBot: [Int], lnTop: [Int]) {
        self.nDense = nDense
        mlpBottom = MLP(dims: [nDense] + lnBot)
        let topInput = lnEmb.count * mSpa + lnBot.last!
        mlpTop = MLP(dims: [topInput] + lnTop + [1])
        latentFactors = lnEmb.map { Embedding(vocabularySize: $0, embeddingSize: mSpa) }

        // TODO: Dot interactions
    }

    @differentiable
    public func callAsFunction(_ input: DLRMInput) -> Tensor<Float> {
        callAsFunction(denseInput: input.dense, sparseInput: input.sparse)
    }

    @differentiable(wrt: self)
    public func callAsFunction(denseInput: Tensor<Float>, sparseInput: [Tensor<Int32>]) -> Tensor<Float> {
        precondition(denseInput.shape.last! == nDense)
        assert(sparseInput.count == latentFactors.count)
        let denseEmbVec = mlpBottom(denseInput)
        let sparseEmbVecs = computeEmbeddings(sparseInputs: sparseInput, latentFactors: latentFactors)
        let topInput = Tensor(concatenating: sparseEmbVecs + [denseEmbVec], alongAxis: 1)
        let prediction = mlpTop(topInput)

        // TODO: loss threshold clipping
        return prediction.reshaped(to: [-1])
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
