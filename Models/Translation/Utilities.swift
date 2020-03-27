//
//  Utilities.swift
//  TranslationTransformer
//
//  Created by Andre Carrera on 2/11/20.
//  Copyright © 2020 Lambdo. All rights reserved.
//

import TensorFlow
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif


@differentiable(wrt: logits)
public func softmaxCrossEntropy(logits: Tensor<Float>, labels: Tensor<Int32>, ignoreIndex: Int32) -> Tensor<Float> {
    let ids = Tensor<Int32>(rangeFrom: 0, to: Int32(labels.shape.first!), stride: 1)
    let indices = ids.gathering(where: labels .!= Tensor(ignoreIndex))
    let maskedLogits = logits.gathering(atIndices: indices, alongAxis: 0)
    let maskedTargets = labels.gathering(atIndices: indices, alongAxis: 0)
    return softmaxCrossEntropy(logits: maskedLogits, labels: maskedTargets)
}

public typealias Activation<Scalar: TensorFlowFloatingPoint> =
    @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

public typealias ActivationInput<Input: Differentiable,Scalar: TensorFlowFloatingPoint> =
    @differentiable (Input) -> Tensor<Scalar>

struct DecoderContext: Differentiable {
    var decoder: TransformerDecoderLayer,
    input: DecoderInput<Float>
    
    @differentiable
    init(decoder: TransformerDecoderLayer,
         input: DecoderInput<Float>) {
        self.decoder = decoder
        self.input = input
    }
}

struct SubLayerInput<Scalar: TensorFlowFloatingPoint >: Differentiable {
    var sequence: Tensor<Scalar>
    @noDerivative public let activation: SubLayerInput<Scalar>.Activation
    /// The element-wise activation function type.
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
    
    @differentiable
    init(sequence: Tensor<Scalar>, activation: @escaping SubLayerInput<Scalar>.Activation) {
        self.sequence = sequence
        self.activation = activation
    }
}

struct DecoderSubLayerInput<Scalar: TensorFlowFloatingPoint >: Differentiable {
    var sequence: Tensor<Scalar>
    var decoderContext: DecoderInput<Scalar>
    @noDerivative public let activation: DecoderSubLayerInput<Scalar>.Activation
    /// The element-wise activation function type.
    public typealias Activation = @differentiable (Tensor<Scalar>,DecoderInput<Scalar>) -> Tensor<Scalar>
    
    @differentiable
    init(sequence: Tensor<Scalar>, decoderContext: DecoderInput<Scalar>, activation: @escaping DecoderSubLayerInput<Scalar>.Activation) {
        self.sequence = sequence
        self.activation = activation
        self.decoderContext = decoderContext
    }
}

public struct TextBatchInput: KeyPathIterable {
    /// IDs that correspond to the vocabulary used while tokenizing.
    /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
    public var tokenIds: Tensor<Int32> // TODO: !!! Mutable in order to allow for batching.
    // aka src
    
    public var targetTokenIds: Tensor<Int32>
    // aka tgt
    
    /// IDs of the token types (e.g., sentence A and sentence B in BERT).
    /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
    //    public var tokenTypeIds: Tensor<Int32> // TODO: !!! Mutable in order to allow for batching.
    
    /// Mask over the sequence of tokens specifying which ones are "real" as opposed to "padding".
    /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
    public var mask: Tensor<Float> // TODO: !!! Mutable in order to allow for batching.
    
    public var targetMask: Tensor<Float> // TODO: !!! Mutable in order to allow for batching.
    
    public var targetTruth: Tensor<Int32>
    
    public var tokenCount: Int32
    
    
    public init(tokenIds: Tensor<Int32>, targetTokenIds: Tensor<Int32>, mask: Tensor<Float>, targetMask: Tensor<Float>, targetTruth: Tensor<Int32>, tokenCount: Int32) {
        self.tokenIds = tokenIds
        self.targetTokenIds = targetTokenIds
        self.mask = mask
        self.targetMask = targetMask
        self.targetTruth = targetTruth
        self.tokenCount = tokenCount
    }
}

struct SublayerConnection: Layer {
    var norm: LayerNorm<Float>
    var dropout: Dropout<Float>
    init(size: Int, droputProb: Double) {
        self.norm = LayerNorm(featureCount: size, axis: -1, epsilon: 1e-6)
        self.dropout = Dropout(probability: droputProb)
    }
    @differentiable
    func callAsFunction(_ input: SubLayerInput< Float>) -> Tensor<Float> {
        return input.sequence + self.dropout(input.activation(self.norm(input.sequence)))
    }
    
    @differentiable
    func decoderForward(_ input: DecoderSubLayerInput< Float>) -> Tensor<Float> {
        return input.sequence + self.dropout(input.activation(self.norm(input.sequence), input.decoderContext))
    }
}

struct PositionwiseFeedForward: Layer {
    // "Implements FFN equation."
    var dense1: Dense<Float>
    var dense2: Dense<Float>
    @noDerivative let dropout: Dropout<Float>
    
    init(dimensionalityModel:Int, innerLayerDimensionality:Int, dropProbability: Double=0.1) {
        dense1 = Dense(inputSize: dimensionalityModel, outputSize: innerLayerDimensionality, weightInitializer: glorotUniform())
        dense2 = Dense(inputSize: innerLayerDimensionality, outputSize: dimensionalityModel, weightInitializer: glorotUniform())
        dropout = Dropout<Float>(probability: dropProbability)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return relu(dense1(input)).sequenced(through: dense2, dropout)
    }
}


struct PositionalEncoding: ParameterlessLayer {
    @noDerivative var encoding: Parameter<Float> // maybe should be an embedding?
    @noDerivative var dropout: Dropout<Float>
    init(size: Int, dropoutProbability: Double = 0, maxLength:Int=5000) {
        dropout = Dropout(probability: dropoutProbability)
        let position = Tensor(rangeFrom: 0, to: Float(maxLength), stride: 1).expandingShape(at: 1)
        let divStart = stride(from: 0, to: size, by: 2).map{ Float($0)}
        let divTerm = Tensor(divStart.map{ 1.0 / pow(10000.0, $0 / Float(size)) })
        
        var positionalEncoding = Tensor<Float>(zeros: [maxLength, size])
        // just alternating the value that is placed inside the tensor between to arrays.
        positionalEncoding[0..., 0..<positionalEncoding.shape[1]..2] = sin(position * divTerm)
        positionalEncoding[0..., 1..<positionalEncoding.shape[1]..2] = cos(position * divTerm)
        encoding = Parameter(positionalEncoding.expandingShape(at: 0))
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return self.dropout(input + encoding.value[0..., 0..<input.shape[1]])
    }
}


public struct TransformerInput<Scalar: TensorFlowFloatingPoint>: Differentiable {
    /// Sequence that the transformer encoder operates over. The shape of this tensor is
    /// `[batchSize, sequenceLength, depth]` or `[batchSize, sequenceLength * depth]`.
    public var sequence: Tensor<Scalar>
    
    /// Mask to apply on the attention scores. This is a tensor with shape
    /// `[batchSize, sourceSequenceLength, targetSequenceLength]` or
    /// `[batchSize, sourceSequenceLength * targetSequenceLength]`. The values should be `1` or
    /// `0`. The attention scores will effectively be set to negative infinity for any positions in
    /// the mask that are set to `0`, and will be unchanged for positions that are set to `1`.
    public var attentionMask: Tensor<Scalar>
    
    /// The batch size of this input. This is optional because it is only needed if the input
    /// sequences have been reshaped to matrices.
    @noDerivative let batchSize: Int?
    
    @differentiable
    public init(sequence: Tensor<Scalar>, attentionMask: Tensor<Scalar>, batchSize: Int? = nil) {
        self.sequence = sequence
        self.attentionMask = attentionMask
        self.batchSize = batchSize
    }
}

public struct DecoderInput<Scalar: TensorFlowFloatingPoint>: Differentiable {
    /// Sequence that the transformer encoder operates over. The shape of this tensor is
    /// `[batchSize, sequenceLength, depth]` or `[batchSize, sequenceLength * depth]`.
    public var sequence: Tensor<Scalar>
    
    public var memory: Tensor<Scalar>
    
    /// Mask to apply on the attention scores. This is a tensor with shape
    /// `[batchSize, sourceSequenceLength, targetSequenceLength]` or
    /// `[batchSize, sourceSequenceLength * targetSequenceLength]`. The values should be `1` or
    /// `0`. The attention scores will effectively be set to negative infinity for any positions in
    /// the mask that are set to `0`, and will be unchanged for positions that are set to `1`.
    public var sourceMask: Tensor<Scalar>
    
    public var targetMask: Tensor<Scalar>
    
    
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

extension Tensor {
    /// Returns this tensor reshaped to a matrix (i.e., a rank-2 tensor).
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    internal func reshapedToMatrix() -> Tensor {
        reshaped(to: [-1, shape[rank - 1]])
    }
    
    /// Returns this previously-reshaped rank-2 tensor reshaped back to its original shape.
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    internal func reshapedFromMatrix(originalShape: TensorShape) -> Tensor {
        reshaped(to: TensorShape(
            originalShape[0..<originalShape.count - 1].dimensions + [shape[rank - 1]]))
    }
    
    /// Returns this previously-reshaped rank-2 tensor reshaped back to its original shape.
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    internal func reshapedFromMatrix(originalShape: Tensor<Int32>) -> Tensor {
        reshaped(toShape: Tensor<Int32>(concatenating: [
            originalShape[0..<originalShape.shape[0] - 1],
            Tensor<Int32>([Int32(shape[rank - 1])])
        ]))
    }
}
