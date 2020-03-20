//
//  Encoder.swift
//  TranslationTransformer
//
//  Created by Andre Carrera on 2/11/20.
//  Copyright © 2020 Lambdo. All rights reserved.
//

import TensorFlow

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
        let batchSizeNotDerivative = withoutDerivative(at: input.batchSize)
        let output = self.sublayers[0](.init(sequence: input.sequence, activation: {
            let attentionInput = AttentionInput(source: $0, target: $0, mask: inputNoDerivative.attentionMask, batchSize: batchSizeNotDerivative)
            return selfNoDerivative.selfAttention.callAsFunction(attentionInput)
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
    
    @differentiable
    func callAsFunction(_ input: TransformerInput<Float>) -> Tensor<Float> {
        var transformerInput = input.sequence//.reshapedToMatrix().debugIdentity()
//        let batchSize = input.sequence.shape[0]
        
        for layerIndex in 0..<(withoutDerivative(at: layers) { $0.count }) {
            transformerInput = layers[layerIndex](TransformerInput(
                sequence: transformerInput,
                attentionMask: input.attentionMask))
        }
        
        return transformerInput//.reshapedFromMatrix(originalShape: input.sequence.shape).debugIdentity()
    }
}
