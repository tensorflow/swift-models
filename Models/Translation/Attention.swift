//
//  Attention.swift
//  TranslationTransformer
//
//  Created by Andre Carrera on 2/11/20.
//  Copyright © 2020 Lambdo. All rights reserved.
//

import TensorFlow

/// Input to an attention layer.
public struct AttentionInput<Scalar: TensorFlowFloatingPoint>: Differentiable {
    /// Source tensor that we are attending from, with shape
    /// `[batchSize, sourceSequenceLength, sourceDepth]` or
    /// `[batchSize, sourceSequenceLength * sourceDepth]`.
    public var source: Tensor<Scalar>

    /// Target tensor that we are attending to, with shape
    /// `[batchSize, targetSequenceLength, targetDepth]` or
    /// `[batchSize, targetSequenceLength * targetDepth]`.
    public var target: Tensor<Scalar>

    /// Mask to apply on the attention scores. This is a tensor with shape
    /// `[batchSize, sourceSequenceLength, targetSequenceLength]` or
    /// `[batchSize, sourceSequenceLength * targetSequenceLength]`. The values should be `1` or `0`.
    /// The attention scores will effectively be set to negative infinity for any positions in the
    /// mask that are set to `0`, and will be unchanged for positions that are set to `1`.
    public var mask: Tensor<Scalar>

    /// The batch size of this input. This is optional because it is only needed if the input
    /// sequences have been reshaped to matrices.
    @noDerivative let batchSize: Int?

    @differentiable
    public init(
        source: Tensor<Scalar>,
        target: Tensor<Scalar>,
        mask: Tensor<Scalar>,
        batchSize: Int? = nil
    ) {
        precondition(
            source.rank == target.rank,
            "The rank of the attention source and target tensors must match.")
        self.source = source
        self.target = target
        self.mask = mask
        self.batchSize = batchSize
    }
}

/// Multi-head attention layer.
///
/// This implementation is based on the
/// ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper. If the source and target
/// tensors are the same, then this layer behaves as a self-attention layer. Each sequence step in
/// the source tensor attends to the corresponding sequence in the target tensor and returns a
/// fixed-size vector.
///
/// This function first projects the source tensor into a "query" tensor and the target tensor into
/// "key" and "value" tensors. These are (effectively) a list of tensors of length `headCount`,
/// where each tensor has shape `[batchSize, sequenceLength, headSize]`. It then performs a dot
/// product between the query and they key tensors and scales them. Finally, they are passed
/// through the softmax function to obtain attention probabilities. The value tensors are then
/// interpolated by these probabilities, and then concatenated back to a single result tensor.
///
/// In practice, the multi-head attention is implemented using transpose and reshape operations,
/// rather than using separate tensors.
///
/// - Source: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).
public struct MultiHeadAttention: Layer {
    // TODO: Convert to a generic constraint once TF-427 is resolved.
    public typealias Scalar = Float

    @noDerivative public let sourceSize: Int
    @noDerivative public let targetSize: Int
    @noDerivative public let headCount: Int
    @noDerivative public let headSize: Int
    @noDerivative public let queryActivation: Activation<Scalar>
    @noDerivative public let keyActivation: Activation<Scalar>
    @noDerivative public let valueActivation: Activation<Scalar>
    @noDerivative public let matrixResult: Bool

    public var queryWeight: Tensor<Scalar>
    public var queryBias: Tensor<Scalar>
    public var keyWeight: Tensor<Scalar>
    public var keyBias: Tensor<Scalar>
    public var valueWeight: Tensor<Scalar>
    public var valueBias: Tensor<Scalar>
    @noDerivative public var attentionDropout: Dropout<Scalar>

    public var regularizationValue: TangentVector {
        TangentVector(
        queryWeight: queryWeight,
        queryBias: Tensor(Scalar(0)),
        keyWeight: keyWeight,
        keyBias: Tensor(Scalar(0)),
        valueWeight: valueWeight,
        valueBias: Tensor(Scalar(0)))
    }

    /// Creates a multi-head attention layer.
    ///
    /// - Parameters:
    ///   - sourceSize: Size/depth of the source tensor this layer is attending from.
    ///   - targetSize: Size/depth of the target tensor this layer is attending to.
    ///   - headCount: Number of attention heads.
    ///   - headSize: Size/depth of each attention head.
    ///   - queryActivation: Activation function applied to the attention query tensor.
    ///   - keyActivation: Activation function applied to the attention key tensor.
    ///   - valueActivation: Activation function applied to the attention value tensor.
    ///   - attentionDropoutProbability: Dropout probability for the attention scores.
    ///   - matrixResult: If `true`, the resulting tensor will have shape
    ///     `[batchSize * sourceSequenceLength, headCount * headSize]`. Otherwise, it will have shape
    ///     `[batchSize, sourceSequenceLength, headCount * headSize]`.
    ///   - queryWeightInitializer: Initializer for the query transformation weight.
    ///   - queryBiasInitializer: Initializer for the query transformation bias.
    ///   - keyWeightInitializer: Initializer for the key transformation weight.
    ///   - keyBiasInitializer: Initializer for the key transformation bias.
    ///   - valueWeightInitializer: Initializer for the value transformation weight.
    ///   - valueBiasInitializer: Initializer for the value transformation bias.
    public init(
        sourceSize: Int,
        targetSize: Int,
        headCount: Int = 1,
        headSize: Int = 512,
        queryActivation: @escaping Activation<Scalar> = identity,
        keyActivation: @escaping Activation<Scalar> = identity,
        valueActivation: @escaping Activation<Scalar> = identity,
        attentionDropoutProbability: Scalar = 0,
        matrixResult: Bool = false,
        queryWeightInitializer: ParameterInitializer<Scalar> = defaultWeightInitializer,
        queryBiasInitializer: ParameterInitializer<Scalar> = defaultBiasInitializer,
        keyWeightInitializer: ParameterInitializer<Scalar> = defaultWeightInitializer,
        keyBiasInitializer: ParameterInitializer<Scalar> = defaultBiasInitializer,
        valueWeightInitializer: ParameterInitializer<Scalar> = defaultWeightInitializer,
        valueBiasInitializer: ParameterInitializer<Scalar> = defaultBiasInitializer
    ) {
        self.sourceSize = sourceSize
        self.targetSize = targetSize
        self.headCount = headCount
        self.headSize = headSize
        self.queryActivation = queryActivation
        self.keyActivation = keyActivation
        self.valueActivation = valueActivation
        self.matrixResult = matrixResult
        self.queryWeight = queryWeightInitializer([sourceSize, headCount * headSize])
        self.queryBias = queryBiasInitializer([headCount * headSize])
        self.keyWeight = keyWeightInitializer([targetSize, headCount * headSize])
        self.keyBias = keyBiasInitializer([headCount * headSize])
        self.valueWeight = valueWeightInitializer([targetSize, headCount * headSize])
        self.valueBias = valueBiasInitializer([headCount * headSize])
        // TODO: Make dropout generic over the probability type.
        self.attentionDropout = Dropout(probability: Double(attentionDropoutProbability))
    }

    @differentiable
    public func callAsFunction(_ input: AttentionInput<Scalar>) -> Tensor<Scalar> {
        precondition(
            input.source.rank == 3 || input.batchSize != nil,
            "Whenever the input is provided in matrix form, the batch size must also be provided.")
        // Scalar dimensions referenced here:
        //   - B = batch size (number of sequences)
        //   - F = `input.source` sequence length
        //   - T = `input.target` sequence length
        //   - N = number of attention heads
        //   - H = size per attention head
        let matrixInput = input.source.rank < 3
        let B = matrixInput ? input.batchSize! : input.source.shape[0]
        let F = matrixInput ? input.source.shape[0] / B : input.source.shape[1]
        let T = matrixInput ? input.target.shape[0] / B : input.target.shape[1]
        let N = headCount
        let H = headSize

        let source = input.source.reshapedToMatrix()
        let target = input.target.reshapedToMatrix()

        var q = queryActivation(matmul(source, queryWeight) + queryBias) // [B * F, N * H]
        var k = keyActivation(matmul(target, keyWeight) + keyBias)       // [B * T, N * H]
        var v = valueActivation(matmul(target, valueWeight) + valueBias) // [B * T, N * H]

        q = q.reshaped(to: [B, F, N, H]).transposed(permutation: 0, 2, 1, 3) // [B, N, F, H]
        k = k.reshaped(to: [B, T, N, H]).transposed(permutation: 0, 2, 1, 3) // [B, N, T, H]
        v = v.reshaped(to: [B, T, N, H]).transposed(permutation: 0, 2, 1, 3) // [B, N, T, H]

        // Take the dot product between the query and the key to get the raw attention scores.
        var attentionScores = matmul(q, transposed: false, k, transposed: true) // [B, N, F, T]
        attentionScores = attentionScores / sqrtf(Scalar(headSize))

        // Since the attention mask is set to 1.0 for positions we want to attend to and 0.0 for
        // masked positions, we create a tensor which is 0.0 for positions we want to attend to and
        // -10000.0 for masked positions. Since we are adding this tensor to the raw scores before
        // the softmax, this is effectively the same as removing the masked entries entirely.
        let attentionMask = input.mask.expandingShape(at: 1) // [B, 1, F, T]
        attentionScores = attentionScores - 10000 * (1 - attentionMask)

        // Normalize the attention scores to convert them to probabilities. We are also dropping
        // out entire tokens to attend to, which might seem a bit unusual, but it is taken from the
        // original Transformer paper.
        let attentionProbabilities = attentionDropout(softmax(attentionScores)) // [B, N, F, T]

        let result = matmul(attentionProbabilities, v) // [B, N, F, H]
            .transposed(permutation: 0, 2, 1, 3)       // [B, F, N, H]
        return matrixResult ?
            result.reshaped(to: [B * F, N * H]) :
            result.reshaped(to: [B, F, N * H])
    }
}

extension MultiHeadAttention {
    /// Default initializer to use for the linear transform weights.
    public static var defaultWeightInitializer: ParameterInitializer<Scalar> {
        truncatedNormalInitializer(standardDeviation: Tensor<Scalar>(0.02))
    }

    /// Default initializer to use for the linear transform biases.
    public static var defaultBiasInitializer: ParameterInitializer<Scalar> {
        zeros()
    }
}
