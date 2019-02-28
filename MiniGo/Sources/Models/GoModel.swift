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
// implements the same architecture as https://github.com/tensorflow/minigo/blob/master/dual_net.py

public struct ModelConfiguration {
  /// size of Go board (typically 9 or 19)
  let boardSize: Int
  /// output feature count of conv layers in shared trunk
  let convWidth: Int
  /// output feature count of conv layer in policy head
  let policyConvWidth: Int
  /// output feature count of conv layer in value head
  let valueConvWidth: Int
  /// output feature count of dense layer in value head
  let valueDenseWidth: Int
  /// number of layers (typically equal to boardSize)
  let layerCount: Int

  public init(boardSize: Int) {
    self.boardSize = boardSize
    self.convWidth = boardSize == 19 ? 256 : 32
    self.policyConvWidth = 2
    self.valueConvWidth = 1
    self.valueDenseWidth = boardSize == 19 ? 256 : 64
    self.layerCount = boardSize
  }
}

struct ConvBN: Layer {
  var conv: Conv2D<Float>
  var norm: BatchNorm<Float>

  init(
    filterShape: (Int, Int, Int, Int),
    strides: (Int, Int) = (1, 1),
    padding: Padding,
    bias: Bool = true,
    affine: Bool = true) {
    // TODO(jekbradbury): thread through bias and affine boolean arguments
    // (behavior is correct for inference but this should be changed for training)
    self.conv = Conv2D(filterShape: filterShape, strides: strides, padding: padding)
    self.norm = BatchNorm(
      featureCount: filterShape.3,
      momentum: Tensor<Float>(0.95),
      epsilon: Tensor<Float>(1e-5))
  }

  @differentiable
  func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
    return norm.applied(to: conv.applied(to: input, in: context), in: context)
  }
}

extension ConvBN: LoadableFromPythonCheckpoint {
  mutating func load(from reader: PythonCheckpointReader) {
    conv.load(from: reader)
    norm.load(from: reader)
  }
}

struct ResidualIdentityBlock: Layer {
  var layer1: ConvBN
  var layer2: ConvBN

  public init(featureCounts: (Int, Int), kernelSize: Int = 3) {
    self.layer1 = ConvBN(
      filterShape: (kernelSize, kernelSize, featureCounts.0, featureCounts.1),
      padding: .same,
      bias: false)

    self.layer2 = ConvBN(
      filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.1),
      padding: .same,
      bias: false)
  }

  @differentiable
  func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
    var tmp = relu(layer1.applied(to: input, in: context))
    tmp = layer2.applied(to: tmp, in: context)
    return relu(tmp + input)
  }
}

extension ResidualIdentityBlock: LoadableFromPythonCheckpoint {
  mutating func load(from reader: PythonCheckpointReader) {
    layer1.load(from: reader)
    layer2.load(from: reader)
  }
}

// This is needed because we can't conform tuples to protocols
public struct GoModelOutput: Differentiable {
  public let policy: Tensor<Float>
  public let value: Tensor<Float>
  public let logits: Tensor<Float>
}

// This might be needed when we add training to work around an AD bug for memberwise initializers
// @differentiable(wrt: (policy, value, logits), vjp: _vjpMakeGoModelOutput)
// func makeGoModelOutput(
//   policy: Tensor<Float>, value: Tensor<Float>, logits: Tensor<Float>)
//   -> GoModelOutput {
//   return GoModelOutput(policy: policy, value: value, logits: logits)
// }
// func _vjpMakeGoModelOutput(
//   policy: Tensor<Float>, value: Tensor<Float>, logits: Tensor<Float>)
//   -> (GoModelOutput, (GoModelOutput.CotangentVector)
//   -> (Tensor<Float>, Tensor<Float>, Tensor<Float>)) {
//   let result = GoModelOutput(policy: policy, value: value, logits: logits)
//   return (result, { seed in (seed.policy, seed.value, seed.logits) })
// }

public struct GoModel: Layer {
  @noDerivative let configuration: ModelConfiguration
  var initialConv: ConvBN
  // TODO(jekbradbury): support differentiation wrt residualBlocks
  // [T] where T: Differentiable doesn't (shouldn't?) conform to Differentiable,
  // so we will likely need a LayerArray<T> where T: Layer type. But this
  // itself won't work until we have better generics support, and even then
  // T can't be an existential Layer. So it's @noDerivative for now.
  @noDerivative var residualBlocks: [ResidualIdentityBlock]
  var policyConv: ConvBN
  var policyDense: Dense<Float>
  var valueConv: ConvBN
  var valueDense1: Dense<Float>
  var valueDense2: Dense<Float>

  public init(configuration: ModelConfiguration) {
    self.configuration = configuration
    initialConv = ConvBN(
      filterShape: (3, 3, 17, configuration.convWidth),
      padding: .same,
      bias: false)

    residualBlocks = (1...configuration.boardSize).map { _ in
      ResidualIdentityBlock(featureCounts: (configuration.convWidth, configuration.convWidth))
    }

    policyConv = ConvBN(
      filterShape: (1, 1, configuration.convWidth, configuration.policyConvWidth),
      padding: .same,
      bias: false,
      affine: false)
    policyDense = Dense<Float>(
      inputSize: configuration.policyConvWidth * configuration.boardSize * configuration.boardSize,
      outputSize: configuration.boardSize * configuration.boardSize + 1,
      activation: {$0})

    valueConv = ConvBN(
      filterShape: (1, 1, configuration.convWidth, configuration.valueConvWidth),
      padding: .same,
      bias: false,
      affine: false)
    valueDense1 = Dense<Float>(
      inputSize: configuration.valueConvWidth * configuration.boardSize * configuration.boardSize,
      outputSize: configuration.valueDenseWidth,
      activation: relu)
    valueDense2 = Dense<Float>(
      inputSize: configuration.valueDenseWidth,
      outputSize: 1,
      activation: tanh)
  }

  @differentiable(wrt: (self, input), vjp: _vjpApplied)
  public func applied(to input: Tensor<Float>, in context: Context) -> GoModelOutput {
    let batchSize = input.shape[0]
    var output = relu(initialConv.applied(to: input, in: context))

    for i in 0..<configuration.boardSize {
      output = residualBlocks[i].applied(to: output, in: context)
    }

    let policyConvOutput = relu(policyConv.applied(to: output, in: context))
    let logits = policyDense.applied(
      to: policyConvOutput.reshaped(toShape: Tensor<Int32>(
        [batchSize,
         Int32(configuration.policyConvWidth * configuration.boardSize * configuration.boardSize),
        ])),
      in: context)
    let policyOutput = softmax(logits)

    let valueConvOutput = relu(valueConv.applied(to: output, in: context))
    let valueHidden = valueDense1.applied(
      to: valueConvOutput.reshaped(toShape: Tensor<Int32>(
        [batchSize,
         Int32(configuration.valueConvWidth * configuration.boardSize * configuration.boardSize)
        ])),
      in: context)
    let valueOutput = valueDense2.applied(to: valueHidden, in: context).reshaped(
      toShape: Tensor<Int32>([batchSize]))

    return GoModelOutput(policy: policyOutput, value: valueOutput, logits: logits)
  }

  @usableFromInline
  func _vjpApplied(to input: Tensor<Float>, in context: Context)
    -> (GoModelOutput, (GoModelOutput.CotangentVector)
      -> (GoModel.CotangentVector, Tensor<Float>)) {
    // TODO(jekbradbury): add a real VJP
    // (we're only interested in inference for now and have control flow in our applied(to:) method)
    return (applied(to: input, in: context), {
      seed in (GoModel.CotangentVector.zero, Tensor<Float>(0))
    })
  }
}

extension GoModel: InferenceModel {
  public func prediction(input: Tensor<Float>) -> GoModelOutput {
    return applied(to: input, in: Context(learningPhase: .inference))
  }
}

extension GoModel: LoadableFromPythonCheckpoint {
  public mutating func load(from reader: PythonCheckpointReader) {
    initialConv.load(from: reader)
    for i in 0..<configuration.boardSize {
      residualBlocks[i].load(from: reader)
    }

    // Special-case the two batchnorms that lack affine weights.
    policyConv.conv.load(from: reader)
    policyConv.norm.runningMean.value = reader.readTensor(
      layerName: "batch_normalization",
      weightName: "moving_mean")!
    policyConv.norm.runningVariance.value = reader.readTensor(
      layerName: "batch_normalization",
      weightName: "moving_variance")!
    reader.increment(layerName: "batch_normalization")

    policyDense.load(from: reader)

    valueConv.conv.load(from: reader)
    valueConv.norm.runningMean.value = reader.readTensor(
      layerName: "batch_normalization",
      weightName: "moving_mean")!
    valueConv.norm.runningVariance.value = reader.readTensor(
      layerName: "batch_normalization",
      weightName: "moving_variance")!
    reader.increment(layerName: "batch_normalization")

    valueDense1.load(from: reader)
    valueDense2.load(from: reader)
  }
}
