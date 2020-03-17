//
//  Decoder.swift
//  TranslationTransformer
//
//  Created by Andre Carrera on 2/11/20.
//  Copyright © 2020 Lambdo. All rights reserved.
//

import TensorFlow

// this on has self attention, source attention, and a feed forward
// sublayer connection (dropout and norm) for each.

// I can probably create a sublayer connection using "Activation"


struct TransformerDecoderLayer: Layer {
    var selfAttention: MultiHeadAttention,
    sourceAttention: MultiHeadAttention,
    feedForward: PositionwiseFeedForward,
    sublayers: [SublayerConnection]
    
    init(size: Int, selfAttention: MultiHeadAttention, sourceAttention: MultiHeadAttention, feedForward: PositionwiseFeedForward, dropoutProb: Double) {
        self.selfAttention = selfAttention
        self.sourceAttention = sourceAttention
        self.feedForward = feedForward
        self.sublayers = [SublayerConnection](repeating: .init(size: size, droputProb: dropoutProb), count: 3)
    }
    
    @differentiable
    func callAsFunction(_ input: DecoderInput<Float>) -> Tensor<Float> {
        // SR-11882
        let selfNoDerivative = withoutDerivative(at: self)
        let inputNoDerivative = withoutDerivative(at: input)
        let batchSizeNotDerivative = withoutDerivative(at: input.batchSize)
        // not sure if source and target are right for any of these.
        var output = self.sublayers[0](.init(sequence: input.sequence, activation: {
            selfNoDerivative.selfAttention(.init(source: $0, target: $0, mask: inputNoDerivative.targetMask, batchSize: batchSizeNotDerivative))
        }))
        output = self.sublayers[1](.init(sequence: output, activation: {
            selfNoDerivative.sourceAttention(.init(source: $0, target: inputNoDerivative.memory, mask: inputNoDerivative.sourceMask, batchSize: batchSizeNotDerivative))
        }))
        output = self.sublayers[2](.init(sequence: output, activation: {
            selfNoDerivative.feedForward($0)
        }))
        return output
    }
}

// based on DecoderLayer.forward parameters
public struct DecoderInput<Scalar: TensorFlowFloatingPoint>: Differentiable {
    /// Sequence that the transformer encoder operates over. The shape of this tensor is
    /// `[batchSize, sequenceLength, depth]` or `[batchSize, sequenceLength * depth]`.
    public var sequence: Tensor<Scalar>
    
    /// Mask to apply on the attention scores. This is a tensor with shape
    /// `[batchSize, sourceSequenceLength, targetSequenceLength]` or
    /// `[batchSize, sourceSequenceLength * targetSequenceLength]`. The values should be `1` or
    /// `0`. The attention scores will effectively be set to negative infinity for any positions in
    /// the mask that are set to `0`, and will be unchanged for positions that are set to `1`.
    public var sourceMask: Tensor<Scalar>
    
    public var targetMask: Tensor<Scalar>
    
    public var memory: Tensor<Scalar>
    
    /// The batch size of this input. This is optional because it is only needed if the input
    /// sequences have been reshaped to matrices.
    @noDerivative let batchSize: Int?
    
    @differentiable
    public init(sequence: Tensor<Scalar>, sourceMask: Tensor<Scalar>,targetMask: Tensor<Scalar>, memory: Tensor<Scalar>, batchSize: Int? = nil) {
        self.sequence = sequence
        self.sourceMask = sourceMask
        self.targetMask = targetMask
        self.memory = memory
        self.batchSize = batchSize
    }
}



struct Decoder: Layer {
    var layers: [TransformerDecoderLayer]
    var norm: LayerNorm<Float>
    init(layer: TransformerDecoderLayer, layerCount: Int) {
        self.layers = [TransformerDecoderLayer](repeating: layer, count: layerCount)
        self.norm = LayerNorm(featureCount: layerCount, axis: 2)
    }
    
    @differentiable
    func callAsFunction(_ input: DecoderInput<Float>) -> Tensor<Float> {
        var transformerInput = input.sequence.reshapedToMatrix()
        let batchSize = input.sequence.shape[0]
        
        for layerIndex in 0..<(withoutDerivative(at: layers) { $0.count }) {
            transformerInput = layers[layerIndex](DecoderInput(sequence: transformerInput, sourceMask: input.sourceMask, targetMask: input.targetMask, memory: input.memory.reshapedToMatrix(), batchSize: batchSize))
        }
        
        return transformerInput.reshapedFromMatrix(originalShape: input.sequence.shape)
    }
}
