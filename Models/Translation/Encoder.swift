//
//  Encoder.swift
//  TranslationTransformer
//
//  Created by Andre Carrera on 2/11/20.
//  Copyright © 2020 Lambdo. All rights reserved.
//

import TensorFlow


// composed of multihead, feedforward in origina implementation.

//class EncoderLayer(nn.Module):
//"Encoder is made up of self-attn and feed forward "
//def __init__(self, size, self_attn, feed_forward, dropout):
//    super(EncoderLayer, self).__init__()
//    self.self_attn = self_attn
//    self.feed_forward = feed_forward
//    self.sublayer = clones(SublayerConnection(size, dropout), 2)
//    self.size = size
//
//def forward(self, x, mask):
//    "Follow Figure 1 (left) for connections."
//    x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
//    return self.sublayer[1](x, self.feed_forward)
struct TransformerEncoderLayer:Layer {
    var selfAttention: MultiHeadAttention,
    feedForward: PositionwiseFeedForward,
    sublayers: [SublayerConnection]
    
    init(size: Int, selfAttention: MultiHeadAttention, feedForward: PositionwiseFeedForward, dropoutProb: Double) {
        self.selfAttention = selfAttention
        self.feedForward = feedForward
        self.sublayers = [SublayerConnection](repeating: .init(size: size, droputProb: dropoutProb), count: 2)
    }
    
    @differentiable
    func callAsFunction(_ input: TransformerInput<Float>) -> Tensor<Float> {
        // SR-11882
        let selfNoDerivative = withoutDerivative(at: self)
        let inputNoDerivative = withoutDerivative(at: input)
        let output = self.sublayers[0](.init(sequence: input.sequence, activation: {
            selfNoDerivative.selfAttention.callAsFunction(.init(source: $0, target: $0, mask: inputNoDerivative.attentionMask))
        }))
        return self.sublayers[1](.init(sequence: output, activation: {
            selfNoDerivative.feedForward.callAsFunction($0)
        }))
    }
}

struct Encoder: Layer {
    var layers: [TransformerEncoderLayer]
    var norm: LayerNorm<Float>
    init(layer: TransformerEncoderLayer, layerCount: Int) {
        self.layers = [TransformerEncoderLayer](repeating: layer, count: layerCount)
        self.norm = LayerNorm(featureCount: layerCount, axis: 2)
    }
    
    // do I need to pass the extra arguments?? I think I do
    @differentiable
    func callAsFunction(_ input: TransformerInput<Float>) -> Tensor<Float> {
        var transformerInput = input.sequence.reshapedToMatrix()
        let batchSize = input.sequence.shape[0]
        
        for layerIndex in 0..<(withoutDerivative(at: layers) { $0.count }) {
            transformerInput = layers[layerIndex](TransformerInput(
                sequence: transformerInput,
                attentionMask: input.attentionMask,
                batchSize: batchSize))
        }
        
        return transformerInput.reshapedFromMatrix(originalShape: input.sequence.shape)
    }
}
