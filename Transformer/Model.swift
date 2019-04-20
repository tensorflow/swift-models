import TensorFlow
import Python

struct TimeDistributed: Layer {
    var dense: Dense<Float>

    init(_ wrapped: Dense<Float>) {
        self.dense = wrapped
    }

    @differentiable(wrt: (self, input))
    func call(_ input: Tensor<Float>) -> Tensor<Float> {
        let (batchSize, timeSteps, features) = (input.shape[0], input.shape[1], input.shape[2])
        let reshaped = input.reshaped(to: [batchSize * timeSteps, features])
        let output = dense(reshaped)
        let outputFeatures = output.shape[1]
        return output.reshaped(to: [batchSize, timeSteps, outputFeatures])
    }
}

struct FeedForward: Layer {
    var dense1: TimeDistributed
    var dense2: TimeDistributed
    @noDerivative let dropout: Dropout<Float>

    init(size: Int, hidden: Int, dropProbability: Double) {
        dense1 = TimeDistributed(
            Dense<Float>(inputSize: size, outputSize: hidden, activation: gelu))
        dense2 = TimeDistributed(Dense<Float>(inputSize: hidden, outputSize: size))
        dropout = Dropout<Float>(probability: dropProbability)
    }

    @differentiable(wrt: (self, input))
    func call(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: dense1, dropout, dense2)
    }
}

struct AttentionInput: Differentiable {
    let query: Tensor<Float>
    let key: Tensor<Float>
    let value: Tensor<Float>
}

@differentiable(wrt: (query, key, value), vjp: _vjpMakeAttentionInput)
func makeAttentionInput(query: Tensor<Float>, key: Tensor<Float>, value: Tensor<Float>)
    -> AttentionInput {
    return AttentionInput(query: query, key: key, value: value)
}

func _vjpMakeAttentionInput(query: Tensor<Float>, key: Tensor<Float>, value: Tensor<Float>)
    -> (AttentionInput, (AttentionInput.CotangentVector) -> (Tensor<Float>, Tensor<Float>, Tensor<Float>)) {
    let result = AttentionInput(query: query, key: key, value: value)
    return (result, { seed in (seed.query, seed.key, seed.value) })
}

struct AttentionContext: Differentiable {
    let key: Tensor<Float>
    let value: Tensor<Float>
}

@differentiable(wrt: (key, value), vjp: _vjpMakeAttentionContext)
func makeAttentionContext(key: Tensor<Float>, value: Tensor<Float>)
    -> AttentionContext {
    return AttentionContext(key: key, value: value)
}

func _vjpMakeAttentionContext(key: Tensor<Float>, value: Tensor<Float>)
    -> (AttentionContext, (AttentionContext.CotangentVector) -> (Tensor<Float>, Tensor<Float>)) {
    let result = AttentionContext(key: key, value: value)
    return (result, { seed in (seed.key, seed.value) })
}

@differentiable(wrt: dotProducts, vjp: _vjpCausallyMasked)
func causallyMasked(_ dotProducts: Tensor<Float>, enable: Bool = false) -> Tensor<Float> {
    if !enable {
        return dotProducts
    }
    let (queryTimeSteps, keyTimeSteps) = (dotProducts.shape[1], dotProducts.shape[2])
    let ones = Tensor<Float>(ones: [1, queryTimeSteps, keyTimeSteps])
    let mask = Raw.matrixBandPart(
        ones,
        numLower: Tensor(Int32(-1)),
        numUpper: Tensor(Int32(queryTimeSteps - keyTimeSteps)))
    return dotProducts * mask - 1e10 * (1 - mask)
}

// causal mask is intentionally invisible to differentiation
func _vjpCausallyMasked(_ dotProducts: Tensor<Float>, enable: Bool)
    -> (Tensor<Float>, (Tensor<Float>) -> Tensor<Float>) {
    return (causallyMasked(dotProducts), identity)
}

struct Attention: Layer {
    @noDerivative let dropout: Dropout<Float>
    @noDerivative let scale: Tensor<Float>
    @noDerivative let causal: Bool
    
    init(size: Int, causal: Bool = false, dropProbability: Double) {
        scale = Tensor(sqrt(Float(size)))
        dropout = Dropout<Float>(probability: dropProbability)
        self.causal = causal
    }
    
    @differentiable(wrt: (self, input))
    func call(_ input: AttentionInput) -> Tensor<Float> {
        var dotProducts = batchedMatmul(input.query, input.key, adjointRight: true)
        dotProducts = causallyMasked(dotProducts, enable: causal) / scale
        return batchedMatmul(dropout(softmax(dotProducts)), input.value)
    }
    
    func call(_ input: AttentionInput, state: inout AttentionContext) -> Tensor<Float> {
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
    let movedToFront = splitLastDim.transposed(withPermutations: 0, 2, 1, 3)
    return movedToFront.reshaped(to: [batchSize * headCount, timeSteps, featuresPerHead])
}

@differentiable(wrt: input)
func joinHeads(_ input: Tensor<Float>, headCount: Int) -> Tensor<Float> {
    let (generalizedBatch, timeSteps, featuresPerHead) = (
        input.shape[0], input.shape[1], input.shape[2])
    let batchSize = generalizedBatch / headCount
    let features = featuresPerHead * headCount
    let splitFirstDim = input.reshaped(to: [batchSize, headCount, timeSteps, featuresPerHead])
    let movedToBack = splitFirstDim.transposed(withPermutations: 0, 2, 1, 3)
    return movedToBack.reshaped(to: [batchSize, timeSteps, features])
}

@differentiable(wrt: input, vjp: _vjpSplitQKV)
func splitQKV(_ input: Tensor<Float>) -> AttentionInput {
    let (generalizedBatch, timeSteps, featuresPerHead) = (
        input.shape[0], input.shape[1], input.shape[2] / 3)
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

func _vjpSplitQKV(_ input: Tensor<Float>)
    -> (AttentionInput, (AttentionInput.CotangentVector) -> Tensor<Float>) {
    let value = splitQKV(input)
    return (value, { seed in
        return Raw.concatV2([seed.query, seed.key, seed.value], axis: Tensor<Int32>(2))
    })
}

struct MultiHeadAttention: Layer {
    var attention: Attention
    var wqkv: TimeDistributed
    var wo: TimeDistributed
    @noDerivative let headCount: Int
    
    init(attention: Attention, size: Int, headCount: Int) {
        self.attention = attention
        wqkv = TimeDistributed(Dense<Float>(
            inputSize: size, outputSize: size * 3, activation: identity))
        wo = TimeDistributed(Dense<Float>(inputSize: size, outputSize: size, activation: identity))
        self.headCount = headCount
    }
    
    @differentiable(wrt: (self, input))
    func call(_ input: Tensor<Float>) -> Tensor<Float> {
        let qkvProjected = wqkv(input)
        let qkvSplit = splitHeads(qkvProjected, headCount: headCount)
        let attentionInput = splitQKV(qkvSplit)
        let outputs = attention(attentionInput)
        return wo(joinHeads(outputs, headCount: headCount))
    }
    
    func call(_ input: Tensor<Float>, state: inout AttentionContext) -> Tensor<Float> {
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

struct EncoderLayer: Layer {
    var selfAttention: MultiHeadAttention
    var selfAttentionDropout: Dropout<Float>
    var selfAttentionNorm: LayerNorm<Float>
    var feedForward: FeedForward
    var feedForwardDropout: Dropout<Float>
    var feedForwardNorm: LayerNorm<Float>

    init(size: Int, headCount: Int, dropProbability: Double) {
        selfAttention = MultiHeadAttention(
            attention: Attention(size: size, dropProbability: dropProbability),
            size: size,
            headCount: headCount)
        selfAttentionDropout = Dropout(probability: dropProbability)
        selfAttentionNorm = LayerNorm(featureCount: size, axis: 2, epsilon: Tensor<Float>(1e-5))
        feedForward = FeedForward(size: size, hidden: 4 * size, dropProbability: dropProbability)
        feedForwardDropout = Dropout(probability: dropProbability)
        feedForwardNorm = LayerNorm(featureCount: size, axis: 2, epsilon: Tensor<Float>(1e-5))
    }

    @differentiable(wrt: (self, input))
    func call(_ input: Tensor<Float>) -> Tensor<Float> {
        let attended = input + input.sequenced(
            through: selfAttentionNorm, selfAttention, selfAttentionDropout)
        return attended + attended.sequenced(
            through: feedForwardNorm, feedForward, feedForwardDropout)
    }

    func call(_ input: Tensor<Float>, state: inout AttentionContext) -> Tensor<Float> {
        var tmp = input
        tmp = selfAttentionNorm(tmp)
        tmp = selfAttention(tmp, state: &state)
        tmp = selfAttentionDropout(tmp)
        let attended = tmp + input
        return attended + attended.sequenced(
            through: feedForwardNorm, feedForward, feedForwardDropout)
    }
}

struct Embedding: Differentiable {
    var weight: Tensor<Float>
    
    init(weight: Tensor<Float>) {
        self.weight = weight
    }
    
    init(vocabSize: Int, size: Int) {
        self.weight = Tensor(randomUniform: [vocabSize, size])
    }

    @differentiable(wrt: self)
    func call(_ input: Tensor<Int32>) -> Tensor<Float> {
        return weight.gathering(atIndices: input)
    }
}

struct TransformerLM {
    var embedding: Embedding
    var positionalEmbeddings: Tensor<Float>
    var layers: [EncoderLayer]
    var norm: LayerNorm<Float>

    func call(_ tokens: Tensor<Int32>, states: inout [AttentionContext]) -> Tensor<Float> {
        let positions = (0..<tokens.shape[1]).map { Int32($0 + states[0].key.shape[1]) }
        let positionsTensor = Tensor<Int32>(shape: [1, tokens.shape[1]], scalars: positions)
        var h = embedding(tokens)
        h = h + positionalEmbeddings.gathering(atIndices: positionsTensor)
        for i in 0..<layers.count {
            h = layers[i](h, state: &states[i])
        }
        h = norm(h)
        let tmp = TimeDistributed(
            Dense(weight: embedding.weight.transposed(), bias: Tensor(0.0), activation: identity))
        let logits = tmp(h) // a somewhat hacky way to share weights
        return logits
    }
}
