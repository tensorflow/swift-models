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

struct TimeDistributed: Layer {
    var dense: Dense<Float>

    init(_ wrapped: Dense<Float>) {
        self.dense = wrapped
    }

    @differentiable(wrt: (self,input))
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let (batchSize, timeSteps, features) = (input.shape[0], input.shape[1], input.shape[2])
        let reshaped = input.reshaped(to: [batchSize * timeSteps, features])
        let output = dense(reshaped)
        let outputFeatures = output.shape[1]
        return output.reshaped(to: [batchSize, timeSteps, outputFeatures])
    }
}

@differentiable
func timeDistributed(_ input: Tensor<Float>, _ weight: Tensor<Float>) -> Tensor<Float> {
    let (batchSize, timeSteps, features) = (input.shape[0], input.shape[1], input.shape[2])
    let reshaped = input.reshaped(to: [batchSize * timeSteps, features])
    let output = matmul(reshaped, weight)
    let outputFeatures = output.shape[1]
    return output.reshaped(to: [batchSize, timeSteps, outputFeatures])
}

struct FeedForward: Layer {
    var dense1: TimeDistributed
    var dense2: TimeDistributed

    init(size: Int, hidden: Int) {
        dense1 = TimeDistributed(
            Dense<Float>(inputSize: size, outputSize: hidden, activation: gelu))
        dense2 = TimeDistributed(Dense<Float>(inputSize: hidden, outputSize: size))
    }

    @differentiable(wrt: (self,input))
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: dense1, dense2)
    }
}

struct AttentionInputGPT2: Differentiable {
    var query: Tensor<Float>
    var key: Tensor<Float>
    var value: Tensor<Float>
}

@differentiable(wrt: (query,key,value))
func makeAttentionInput(query: Tensor<Float>, key: Tensor<Float>, value: Tensor<Float>)
    -> AttentionInputGPT2
{
    return AttentionInputGPT2(query: query, key: key, value: value)
}

@derivative(of: makeAttentionInput, wrt: (query,key,value))
func _vjpMakeAttentionInput(query: Tensor<Float>, key: Tensor<Float>, value: Tensor<Float>)
    -> (
        value: AttentionInputGPT2,
        pullback: (AttentionInputGPT2.TangentVector)
            -> (Tensor<Float>, Tensor<Float>, Tensor<Float>)
    )
{
    let result = AttentionInputGPT2(query: query, key: key, value: value)
    return (result, { seed in (seed.query, seed.key, seed.value) })
}

public struct AttentionContext: Differentiable {
    var key: Tensor<Float>
    var value: Tensor<Float>

    public init(key: Tensor<Float>, value: Tensor<Float>) {
        self.key = key
        self.value = value
    }
}

@differentiable(wrt: (key,value))
func makeAttentionContext(key: Tensor<Float>, value: Tensor<Float>) -> AttentionContext {
    return AttentionContext(key: key, value: value)
}

@derivative(of: makeAttentionContext, wrt: (key,value))
func _vjpMakeAttentionContext(key: Tensor<Float>, value: Tensor<Float>)
    -> (
        value: AttentionContext,
        pullback: (AttentionContext.TangentVector)
            -> (Tensor<Float>, Tensor<Float>)
    )
{
    let result = AttentionContext(key: key, value: value)
    return (result, { seed in (seed.key, seed.value) })
}

@differentiable(wrt: dotProducts)
func causallyMasked(_ dotProducts: Tensor<Float>, enable: Bool = false) -> Tensor<Float> {
    if !enable {
        return dotProducts
    }
    let (queryTimeSteps, keyTimeSteps) = (dotProducts.shape[1], dotProducts.shape[2])
    let device = dotProducts.device
    let ones = Tensor<Float>(repeating: 1, shape: [1, queryTimeSteps, keyTimeSteps], on: device)
    let mask = ones.bandPart(
        subdiagonalCount: -1, superdiagonalCount: queryTimeSteps - keyTimeSteps)
    return dotProducts * mask - 1e10 * (1 - mask)
}

// causal mask is intentionally invisible to differentiation
@derivative(of: causallyMasked, wrt: dotProducts)
func _vjpCausallyMasked(_ dotProducts: Tensor<Float>, enable: Bool)
    -> (value: Tensor<Float>, pullback: (Tensor<Float>) -> Tensor<Float>)
{
    return (causallyMasked(dotProducts, enable: enable), identity)
}

struct Attention: ParameterlessLayer {
    typealias TangentVector = EmptyTangentVector

    @noDerivative var dropout: Dropout<Float>
    @noDerivative var scale: Tensor<Float>
    @noDerivative var causal: Bool

    init(size: Int, causal: Bool = false, dropProbability: Double) {
        scale = Tensor(sqrtf(Float(size)))
        dropout = Dropout<Float>(probability: dropProbability)
        self.causal = causal
    }

    @differentiable(wrt: (self,input))
    func callAsFunction(_ input: AttentionInputGPT2) -> Tensor<Float> {
        var dotProducts = batchedMatmul(input.query, input.key, adjointRight: true)
        dotProducts = causallyMasked(dotProducts, enable: causal) / scale
        return batchedMatmul(dropout(softmax(dotProducts)), input.value)
    }

    func callAsFunction(_ input: AttentionInputGPT2, state: inout AttentionContext) -> Tensor<Float>
    {
        state = AttentionContext(
            key: state.key.concatenated(with: input.key, alongAxis: 1),
            value: state.value.concatenated(with: input.value, alongAxis: 1))
        var dotProducts = batchedMatmul(input.query, state.key, adjointRight: true)
        dotProducts = causallyMasked(dotProducts, enable: causal) / scale
        return batchedMatmul(dropout(softmax(dotProducts)), state.value)
    }
}

@differentiable(wrt: input)
func splitHeads(_ input: Tensor<Float>, headCount: Int) -> Tensor<Float> {
    let (batchSize, timeSteps, features) = (input.shape[0], input.shape[1], input.shape[2])
    let featuresPerHead = features / headCount
    let splitLastDim = input.reshaped(to: [batchSize, timeSteps, headCount, featuresPerHead])
    let movedToFront = splitLastDim.transposed(permutation: 0, 2, 1, 3)
    return movedToFront.reshaped(to: [batchSize * headCount, timeSteps, featuresPerHead])
}

@differentiable(wrt: input)
func joinHeads(_ input: Tensor<Float>, headCount: Int) -> Tensor<Float> {
    let (generalizedBatch, timeSteps, featuresPerHead) = (
        input.shape[0], input.shape[1], input.shape[2]
    )
    let batchSize = generalizedBatch / headCount
    let features = featuresPerHead * headCount
    let splitFirstDim = input.reshaped(to: [batchSize, headCount, timeSteps, featuresPerHead])
    let movedToBack = splitFirstDim.transposed(permutation: 0, 2, 1, 3)
    return movedToBack.reshaped(to: [batchSize, timeSteps, features])
}

@differentiable(wrt: input)
func splitQKV(_ input: Tensor<Float>) -> AttentionInputGPT2 {
    let (generalizedBatch, timeSteps, featuresPerHead) = (
        input.shape[0], input.shape[1], input.shape[2] / 3
    )
    let query = input.slice(
        lowerBounds: [0, 0, 0],
        upperBounds: [generalizedBatch, timeSteps, featuresPerHead])
    let key = input.slice(
        lowerBounds: [0, 0, featuresPerHead],
        upperBounds: [generalizedBatch, timeSteps, 2 * featuresPerHead])
    let value = input.slice(
        lowerBounds: [0, 0, 2 * featuresPerHead],
        upperBounds: [generalizedBatch, timeSteps, 3 * featuresPerHead])
    return makeAttentionInput(query: query, key: key, value: value)
}

@derivative(of: splitQKV, wrt: input)
func _vjpSplitQKV(_ input: Tensor<Float>)
    -> (value: AttentionInputGPT2, pullback: (AttentionInputGPT2.TangentVector) -> Tensor<Float>)
{
    let value = splitQKV(input)
    return (
        value,
        { seed in
            return Tensor(concatenating: [seed.query, seed.key, seed.value], alongAxis: 2)
        }
    )
}

struct MultiHeadAttentionGPT2: Layer {
    var attention: Attention
    var wqkv: TimeDistributed
    var wo: TimeDistributed
    @noDerivative var headCount: Int

    init(attention: Attention, size: Int, headCount: Int) {
        self.attention = attention
        wqkv = TimeDistributed(
            Dense<Float>(
                inputSize: size, outputSize: size * 3, activation: identity))
        wo = TimeDistributed(Dense<Float>(inputSize: size, outputSize: size, activation: identity))
        self.headCount = headCount
    }

    @differentiable(wrt: (self,input))
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let qkvProjected = wqkv(input)
        let qkvSplit = splitQKV(qkvProjected)
        let attentionInput = makeAttentionInput(
            query: splitHeads(qkvSplit.query, headCount: headCount),
            key: splitHeads(qkvSplit.key, headCount: headCount),
            value: splitHeads(qkvSplit.value, headCount: headCount)
        )
        let outputs = attention(attentionInput)
        return wo(joinHeads(outputs, headCount: headCount))
    }

    func callAsFunction(_ input: Tensor<Float>, state: inout AttentionContext) -> Tensor<Float> {
        let qkvProjected = wqkv(input)
        let qkvSplit = splitQKV(qkvProjected)
        let attentionInput = makeAttentionInput(
            query: splitHeads(qkvSplit.query, headCount: headCount),
            key: splitHeads(qkvSplit.key, headCount: headCount),
            value: splitHeads(qkvSplit.value, headCount: headCount)
        )
        let outputs = attention(attentionInput, state: &state)
        return wo(joinHeads(outputs, headCount: headCount))
    }
}

public struct EncoderLayer: Layer {
    var selfAttention: MultiHeadAttentionGPT2
    var selfAttentionDropout: Dropout<Float>
    var selfAttentionNorm: LayerNorm<Float>
    var feedForward: FeedForward
    var feedForwardDropout: Dropout<Float>
    var feedForwardNorm: LayerNorm<Float>

    init(size: Int, headCount: Int, dropProbability: Double) {
        selfAttention = MultiHeadAttentionGPT2(
            attention: Attention(size: size, dropProbability: dropProbability),
            size: size,
            headCount: headCount)
        selfAttentionDropout = Dropout(probability: dropProbability)
        selfAttentionNorm = LayerNorm(featureCount: size, axis: 2, epsilon: 1e-5)
        feedForward = FeedForward(size: size, hidden: 4 * size)
        feedForwardDropout = Dropout(probability: dropProbability)
        feedForwardNorm = LayerNorm(featureCount: size, axis: 2, epsilon: 1e-5)
    }

    @differentiable(wrt: (self,input))
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let attended =
            input
            + input.sequenced(
                through: selfAttentionNorm, selfAttention, selfAttentionDropout)
        return attended
            + attended.sequenced(
                through: feedForwardNorm, feedForward, feedForwardDropout)
    }

    func callAsFunction(_ input: Tensor<Float>, state: inout AttentionContext) -> Tensor<Float> {
        var tmp = input
        tmp = selfAttentionNorm(tmp)
        tmp = selfAttention(tmp, state: &state)
        tmp = selfAttentionDropout(tmp)
        let attended = tmp + input
        return attended
            + attended.sequenced(
                through: feedForwardNorm, feedForward, feedForwardDropout)
    }
}

public struct TransformerLM: Module {
    var embedding: Embedding<Float>
    var positionalEmbeddings: Tensor<Float>
    var embeddingDropout: Dropout<Float> 
    var layers: [EncoderLayer]
    var norm: LayerNorm<Float>

    public init(
        embedding: Embedding<Float>, positionalEmbeddings: Tensor<Float>,
        dropProbability: Double,
        layers: [EncoderLayer], norm: LayerNorm<Float>
    ) {
        self.embedding = embedding
        self.positionalEmbeddings = positionalEmbeddings
        self.embeddingDropout = Dropout(probability: dropProbability)
        self.layers = layers
        self.norm = norm
    }

    // Used for generation, where state transference is important.
    public func callAsFunction(_ tokens: Tensor<Int32>, states: inout [AttentionContext]) -> Tensor<
        Float
    > {
        let positions = (0..<tokens.shape[1]).map { Int32($0 + states[0].key.shape[1]) }
        let positionsTensor = Tensor<Int32>(shape: [1, tokens.shape[1]], scalars: positions)
        var h = embedding(tokens)
        h = h + positionalEmbeddings.gathering(atIndices: positionsTensor)
        h = embeddingDropout(h)
        for i in 0..<layers.count {
            // Remove the .call when TF-516 is fixed.
            h = layers[i].callAsFunction(h, state: &states[i])
        }
        h = norm(h)
        // A somewhat hacky way to share weights.
        let logits = timeDistributed(h, embedding.embeddings.transposed())
        return logits
    }

    // Used for training, where examples are independent.
    @differentiable
    public func callAsFunction(_ tokens: Tensor<Int32>) -> Tensor<Float> {
        let positions = { (0..<tokens.shape[1]).map { Int32($0) } }()
        let positionsTensor = Tensor<Int32>(shape: [1, tokens.shape[1]], scalars: positions, on: tokens.device)
        var h = embedding(tokens)
        h = h + positionalEmbeddings.gathering(atIndices: positionsTensor)
        h = embeddingDropout(h)
        h = layers.differentiableReduce(h) { $1($0) }
        h = norm(h)
        // A somewhat hacky way to share weights.
        let logits = timeDistributed(h, embedding.embeddings.transposed())
        return logits
    }
}
