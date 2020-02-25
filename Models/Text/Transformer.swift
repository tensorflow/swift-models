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

/// Input to a transformer layer.
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

/// Multi-headed and multi-layer transformer encoder.
///
/// - Note: This layer returns a tensor with shape `[batchSize, sequenceLength, hiddenSize]`.
///
/// - Source: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).
public struct TransformerEncoder: Layer, Regularizable {
    // TODO: Convert to a generic constraint once TF-427 is resolved.
    public typealias Scalar = Float

    @noDerivative public let hiddenSize: Int

    public var encoderLayers: [TransformerEncoderLayer]

    public var regularizationValue: TangentVector {
        TangentVector(encoderLayers: [TransformerEncoderLayer].TangentVector(
            encoderLayers.map { $0.regularizationValue }))
    }

    /// Creates a transformer encoder.
    ///
    /// - Parameters:
    ///   - hiddenSize: Size/depth of the transformer hidden representation.
    ///   - layerCount: Number of transformer layers.
    ///   - attentionHeadCount: Number of attention heads.
    ///   - attentionQueryActivation: Activation function applied to the attention query tensor.
    ///   - attentionKeyActivation: Activation function applied to the attention key tensor.
    ///   - attentionValueActivation: Activation function applied to the attention value tensor.
    ///   - intermediateSize: Size/depth of the transformer intermediate representation.
    ///   - intermediateActivation: Activation function applied to the intermediate representation.
    ///   - hiddenDropoutProbability: Dropout probability for the hidden representations.
    ///   - attentionDropoutProbability: Dropout probability for the attention scores.
    ///   - queryWeightInitializer: Initializer for the query transformation weight.
    ///   - queryBiasInitializer: Initializer for the query transformation bias.
    ///   - keyWeightInitializer: Initializer for the key transformation weight.
    ///   - keyBiasInitializer: Initializer for the key transformation bias.
    ///   - valueWeightInitializer: Initializer for the value transformation weight.
    ///   - valueBiasInitializer: Initializer for the value transformation bias.
    ///   - attentionWeightInitializer: Initializer for the attention transformation weight.
    ///   - attentionBiasInitializer: Initializer for the attention transformation bias.
    ///   - intermediateWeightInitializer: Initializer for the intermediate transformation weight.
    ///   - intermediateBiasInitializer: Initializer for the intermediate transformation bias.
    ///   - outputWeightInitializer: Initializer for the output transformation weight.
    ///   - outputBiasInitializer: Initializer for the output transformation bias.
    public init(
        hiddenSize: Int,
        layerCount: Int,
        attentionHeadCount: Int,
        attentionQueryActivation: @escaping Activation<Scalar>,
        attentionKeyActivation: @escaping Activation<Scalar>,
        attentionValueActivation: @escaping Activation<Scalar>,
        intermediateSize: Int,
        intermediateActivation: @escaping Activation<Scalar>,
        hiddenDropoutProbability: Scalar,
        attentionDropoutProbability: Scalar,
        queryWeightInitializer: ParameterInitializer<Scalar> = defaultWeightInitializer,
        queryBiasInitializer: ParameterInitializer<Scalar> = defaultBiasInitializer,
        keyWeightInitializer: ParameterInitializer<Scalar> = defaultWeightInitializer,
        keyBiasInitializer: ParameterInitializer<Scalar> = defaultBiasInitializer,
        valueWeightInitializer: ParameterInitializer<Scalar> = defaultWeightInitializer,
        valueBiasInitializer: ParameterInitializer<Scalar> = defaultBiasInitializer,
        attentionWeightInitializer: ParameterInitializer<Scalar> = defaultWeightInitializer,
        attentionBiasInitializer: ParameterInitializer<Scalar> = defaultBiasInitializer,
        intermediateWeightInitializer: ParameterInitializer<Scalar> = defaultWeightInitializer,
        intermediateBiasInitializer: ParameterInitializer<Scalar> = defaultBiasInitializer,
        outputWeightInitializer: ParameterInitializer<Scalar> = defaultWeightInitializer,
        outputBiasInitializer: ParameterInitializer<Scalar> = defaultBiasInitializer
    ) {
        self.hiddenSize = hiddenSize
        self.encoderLayers = (0..<layerCount).map { _ in
            TransformerEncoderLayer(
                hiddenSize: hiddenSize,
                attentionHeadCount: attentionHeadCount,
                attentionQueryActivation: attentionQueryActivation,
                attentionKeyActivation: attentionKeyActivation,
                attentionValueActivation: attentionValueActivation,
                intermediateSize: intermediateSize,
                intermediateActivation: intermediateActivation,
                hiddenDropoutProbability: hiddenDropoutProbability,
                attentionDropoutProbability: attentionDropoutProbability,
                queryWeightInitializer: queryWeightInitializer,
                queryBiasInitializer: queryBiasInitializer,
                keyWeightInitializer: keyWeightInitializer,
                keyBiasInitializer: keyBiasInitializer,
                valueWeightInitializer: valueWeightInitializer,
                valueBiasInitializer: valueBiasInitializer,
                attentionWeightInitializer: attentionWeightInitializer,
                attentionBiasInitializer: attentionBiasInitializer,
                intermediateWeightInitializer: intermediateWeightInitializer,
                intermediateBiasInitializer: intermediateBiasInitializer,
                outputWeightInitializer: outputWeightInitializer,
                outputBiasInitializer: outputBiasInitializer)
        }
    }

    @differentiable
    public func callAsFunction(_ input: TransformerInput<Scalar>) -> Tensor<Scalar> {
        // The transformer performs sum residuals on all layers and so the input needs to have the
        // same depth as hidden size of the transformer.
        precondition(
            input.sequence.shape[2] == hiddenSize,
            "The depth of the input tensor (\(input.sequence.shape[2]) is different " +
                "than the hidden size (\(hiddenSize).")

        // We keep the representation as a 2-D tensor to avoid reshaping it back and forth from a
        // 3-D tensor to a 2-D tensor. Reshapes are normally free on GPUs/CPUs but may not be free
        // on TPUs, and so we want to minimize them to help the optimizer.
        var transformerInput = input.sequence.reshapedToMatrix()
        let batchSize = input.sequence.shape[0]
        for layerIndex in 0..<(withoutDerivative(at: encoderLayers) { $0.count }) {
            transformerInput = encoderLayers[layerIndex](TransformerInput(
                sequence: transformerInput,
                attentionMask: input.attentionMask,
                batchSize: batchSize))
        }

        return transformerInput.reshapedFromMatrix(originalShape: input.sequence.shape)
    }
}

extension TransformerEncoder {
    /// Default initializer to use for the linear transform weights.
    public static var defaultWeightInitializer: ParameterInitializer<Scalar> {
        truncatedNormalInitializer(standardDeviation: Tensor<Scalar>(0.02))
    }

    /// Default initializer to use for the linear transform biases.
    public static var defaultBiasInitializer: ParameterInitializer<Scalar> {
        zeros()
    }
}

/// Transformer encoder layer.
///
/// - Source: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).
public struct TransformerEncoderLayer: Layer, Regularizable {
    // TODO: Convert to a generic constraint once TF-427 is resolved.
    public typealias Scalar = Float

    @noDerivative public let hiddenSize: Int
    @noDerivative public let intermediateActivation: Activation<Scalar>

    public var multiHeadAttention: MultiHeadAttention
    @noDerivative public var hiddenDropout: Dropout<Scalar>
    public var attentionWeight: Tensor<Scalar>
    public var attentionBias: Tensor<Scalar>
    public var attentionLayerNorm: LayerNorm<Scalar>
    public var intermediateWeight: Tensor<Scalar>
    public var intermediateBias: Tensor<Scalar>
    public var outputWeight: Tensor<Scalar>
    public var outputBias: Tensor<Scalar>
    public var outputLayerNorm: LayerNorm<Scalar>

    public var regularizationValue: TangentVector {
        TangentVector(
            multiHeadAttention: multiHeadAttention.regularizationValue,
            attentionWeight: attentionWeight,
            attentionBias: Tensor(Scalar(0)),
            attentionLayerNorm: attentionLayerNorm.regularizationValue,
            intermediateWeight: intermediateWeight,
            intermediateBias: Tensor(Scalar(0)),
            outputWeight: outputWeight,
            outputBias: Tensor(Scalar(0)),
            outputLayerNorm: outputLayerNorm.regularizationValue)
    }

    /// Creates a transformer encoder layer.
    ///
    /// - Parameters:
    ///   - hiddenSize: Size/depth of the transformer hidden representation.
    ///   - attentionHeadCount: Number of attention heads.
    ///   - attentionQueryActivation: Activation function applied to the attention query tensor.
    ///   - attentionKeyActivation: Activation function applied to the attention key tensor.
    ///   - attentionValueActivation: Activation function applied to the attention value tensor.
    ///   - intermediateSize: Size/depth of the transformer intermediate representation.
    ///   - intermediateActivation: Activation function applied to the intermediate representation.
    ///   - hiddenDropoutProbability: Dropout probability for the hidden representations.
    ///   - attentionDropoutProbability: Dropout probability for the attention scores.
    ///   - queryWeightInitializer: Initializer for the query transformation weight.
    ///   - queryBiasInitializer: Initializer for the query transformation bias.
    ///   - keyWeightInitializer: Initializer for the key transformation weight.
    ///   - keyBiasInitializer: Initializer for the key transformation bias.
    ///   - valueWeightInitializer: Initializer for the value transformation weight.
    ///   - valueBiasInitializer: Initializer for the value transformation bias.
    ///   - attentionWeightInitializer: Initializer for the attention transformation weight.
    ///   - attentionBiasInitializer: Initializer for the attention transformation bias.
    ///   - intermediateWeightInitializer: Initializer for the intermediate transformation weight.
    ///   - intermediateBiasInitializer: Initializer for the intermediate transformation bias.
    ///   - outputWeightInitializer: Initializer for the output transformation weight.
    ///   - outputBiasInitializer: Initializer for the output transformation bias.
    public init(
        hiddenSize: Int,
        attentionHeadCount: Int,
        attentionQueryActivation: @escaping Activation<Scalar>,
        attentionKeyActivation: @escaping Activation<Scalar>,
        attentionValueActivation: @escaping Activation<Scalar>,
        intermediateSize: Int,
        intermediateActivation: @escaping Activation<Scalar>,
        hiddenDropoutProbability: Scalar,
        attentionDropoutProbability: Scalar,
        queryWeightInitializer: ParameterInitializer<Scalar> = defaultWeightInitializer,
        queryBiasInitializer: ParameterInitializer<Scalar> = defaultBiasInitializer,
        keyWeightInitializer: ParameterInitializer<Scalar> = defaultWeightInitializer,
        keyBiasInitializer: ParameterInitializer<Scalar> = defaultBiasInitializer,
        valueWeightInitializer: ParameterInitializer<Scalar> = defaultWeightInitializer,
        valueBiasInitializer: ParameterInitializer<Scalar> = defaultBiasInitializer,
        attentionWeightInitializer: ParameterInitializer<Scalar> = defaultWeightInitializer,
        attentionBiasInitializer: ParameterInitializer<Scalar> = defaultBiasInitializer,
        intermediateWeightInitializer: ParameterInitializer<Scalar> = defaultWeightInitializer,
        intermediateBiasInitializer: ParameterInitializer<Scalar> = defaultBiasInitializer,
        outputWeightInitializer: ParameterInitializer<Scalar> = defaultWeightInitializer,
        outputBiasInitializer: ParameterInitializer<Scalar> = defaultBiasInitializer
    ) {
        precondition(
            hiddenSize % attentionHeadCount == 0,
            "The hidden size of the transformer (\(hiddenSize)) must be a multiple of the " +
                "attention head count (\(attentionHeadCount)).")
        self.hiddenSize = hiddenSize
        self.intermediateActivation = intermediateActivation
        self.multiHeadAttention = MultiHeadAttention(
            sourceSize: hiddenSize,
            targetSize: hiddenSize,
            headCount: attentionHeadCount,
            headSize: hiddenSize / attentionHeadCount,
            queryActivation: attentionQueryActivation,
            keyActivation: attentionKeyActivation,
            valueActivation: attentionValueActivation,
            attentionDropoutProbability: attentionDropoutProbability,
            matrixResult: true,
            queryWeightInitializer: queryWeightInitializer,
            queryBiasInitializer: queryBiasInitializer,
            keyWeightInitializer: keyWeightInitializer,
            keyBiasInitializer: keyBiasInitializer,
            valueWeightInitializer: valueWeightInitializer,
            valueBiasInitializer: valueBiasInitializer)
        // TODO: Make dropout generic over the probability type.
        self.hiddenDropout = Dropout(probability: Double(hiddenDropoutProbability))
        self.attentionWeight = attentionWeightInitializer(
            [attentionHeadCount * hiddenSize / attentionHeadCount, hiddenSize])
        self.attentionBias = attentionBiasInitializer([hiddenSize])
        self.attentionLayerNorm = LayerNorm(
            featureCount: hiddenSize,
            axis: -1)
        self.intermediateWeight = intermediateWeightInitializer([hiddenSize, intermediateSize])
        self.intermediateBias = intermediateBiasInitializer([intermediateSize])
        self.outputWeight = intermediateWeightInitializer([intermediateSize, hiddenSize])
        self.outputBias = intermediateBiasInitializer([hiddenSize])
        self.outputLayerNorm = LayerNorm(featureCount: hiddenSize, axis: -1)
    }

    @differentiable
    public func callAsFunction(_ input: TransformerInput<Scalar>) -> Tensor<Scalar> {
        let attentionInput = AttentionInput(
            source: input.sequence,
            target: input.sequence,
            mask: input.attentionMask,
            batchSize: input.batchSize)
        var attentionOutput = multiHeadAttention(attentionInput)

        // Run a linear projection of `hiddenSize` and then add a residual connection to the input.
        attentionOutput = matmul(attentionOutput, attentionWeight) + attentionBias
        attentionOutput = hiddenDropout(attentionOutput)
        attentionOutput = attentionLayerNorm(attentionOutput + input.sequence)

        // The activation is only applied to the "intermediate" hidden layer.
        var intermediateOutput = matmul(attentionOutput, intermediateWeight) + intermediateBias
        intermediateOutput = intermediateActivation(intermediateOutput)

        // Project back to `hiddenSize` and add the residual.
        var output = matmul(intermediateOutput, outputWeight) + outputBias
        output = hiddenDropout(output)
        output = outputLayerNorm(output + attentionOutput)

        return output
    }
}

extension TransformerEncoderLayer {
    /// Default initializer to use for the linear transform weights.
    public static var defaultWeightInitializer: ParameterInitializer<Scalar> {
        truncatedNormalInitializer(standardDeviation: Tensor<Scalar>(0.02))
    }

    /// Default initializer to use for the linear transform biases.
    public static var defaultBiasInitializer: ParameterInitializer<Scalar> {
        zeros()
    }
}
