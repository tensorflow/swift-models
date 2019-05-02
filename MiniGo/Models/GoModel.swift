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

// Implements the same architecture as https://github.com/tensorflow/minigo/blob/master/dual_net.py

import TensorFlow

public struct ModelConfiguration {
    /// The size of the Go board (typically `9` or `19`).
    public let boardSize: Int
    /// The number of output features of conv layers in shared trunk.
    let convWidth: Int
    /// The output feature count of conv layer in policy head.
    let policyConvWidth: Int
    /// The output feature count of conv layer in value head.
    let valueConvWidth: Int
    /// The output feature count of dense layer in value head.
    let valueDenseWidth: Int
    /// The number of layers (typically equal to `boardSize`).
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

extension Array: Layer where Element: Layer, Element.Input == Element.Output {
    public typealias Input = Element.Input
    public typealias Output = Element.Output

    @differentiable(vjp: _gradCall)
    public func call(_ input: Input) -> Output {
        var activation = input
        for layer in self {
            activation = layer(activation)
        }
        return activation
    }

    public func _gradCall(_ input: Input)
        -> (Output, (Output.CotangentVector) -> (Array.CotangentVector, Input.CotangentVector))
    {
        var activation = input
        var pullbacks: [(Input.CotangentVector) -> (Element.CotangentVector, Input.CotangentVector)] = []
        for layer in self {
            let (newActivation, newPullback) = layer.valueWithPullback(at: activation) { $0($1) }
            activation = newActivation
            pullbacks.append(newPullback)
        }
        func pullback(_ v: Input.CotangentVector) -> (Array.CotangentVector, Input.CotangentVector) {
            var activationGradient = v
            var layerGradients: [Element.CotangentVector] = []
            for pullback in pullbacks.reversed() {
                let (newLayerGradient, newActivationGradient) = pullback(activationGradient)
                activationGradient = newActivationGradient
                layerGradients.append(newLayerGradient)
            }
            return (Array.CotangentVector(layerGradients.reversed()), activationGradient)
        }
        return (activation, pullback)
    }
}

struct ConvBN: Layer {
    var conv: Conv2D<Float>
    var norm: BatchNorm<Float>

    init(
        _ kernelSize: Int,
        from inputFeatures: Int,
        to outputFeatures: Int,
        stride: Int = 1,
        padding: Padding = .same,
        bias: Bool = false,
        affine: Bool = true
    ) {
        self.conv = Conv2D(
            filterShape: (kernelSize, kernelSize, inputFeatures, outputFeatures),
            strides: (stride, stride),
            padding: padding)
        self.norm = BatchNorm(
            featureCount: outputFeatures,
            momentum: Tensor(0.95),
            epsilon: Tensor(1e-5))
    }

    @differentiable
    public func call(_ input: Tensor<Float>) -> Tensor<Float> {
        return norm(conv(input))
    }
}

struct ResidualBlock: Layer {
    var layer1: ConvBN
    var layer2: ConvBN

    public init(_ kernelSize: Int = 3, from inputFeatures: Int, to outputFeatures: Int) {
        self.layer1 = ConvBN(kernelSize, from: inputFeatures, to: outputFeatures)
        self.layer2 = ConvBN(kernelSize, from: outputFeatures, to: outputFeatures)
    }

    @differentiable
    public func call(_ input: Tensor<Float>) -> Tensor<Float> {
        return relu(layer2(relu(layer1(input))) + input)
    }
}

public struct MLP: Layer {
    var conv: ConvBN
    var flatten: Flatten<Float> = Flatten()
    var dense: [Dense<Float>]

    public init(
        _ featureCounts: [Int],
        activation: @escaping @differentiable (Input) -> Output = identity
    ) {
        conv = ConvBN(1, from: featureCounts[0], to: featureCounts[1])
        dense = [Dense(inputSize: featureCounts[1], outputSize: featureCounts[2], activation: tanh)]
        if featureCounts.count == 2 {
            dense.append(Dense(
                inputSize: featureCounts[2],
                outputSize: featureCounts[3],
                activation: activation))
        }
    }

    @differentiable
    public func call(_ input: Tensor<Float>) -> Tensor<Float> {
        return dense(flatten(relu(conv(input))))
    }
}

public struct GoModelOutput: Differentiable {
    public let policy: Tensor<Float>
    public let value: Tensor<Float>
    public let logits: Tensor<Float>
}

public struct GoModel: Layer {
    var initialConv: ConvBN
    var residualBlocks: [ResidualBlock]
    var policyHead: MLP
    var valueHead: MLP

    public init(configuration: ModelConfiguration) {
        let cfg = configuration
        initialConv = ConvBN(3, from: 17, to: cfg.convWidth)
        residualBlocks = (0..<cfg.boardSize).map { _ in
            ResidualBlock(from: cfg.convWidth, to: cfg.convWidth)
        }
        policyHead = MLP(
            [cfg.convWidth, cfg.policyConvWidth, cfg.boardSize * cfg.boardSize + 1])
        valueHead = MLP(
            [cfg.convWidth, cfg.valueConvWidth, cfg.valueDenseWidth, 1],
            activation: tanh)
    }

    @differentiable
    public func call(_ input: Tensor<Float>) -> GoModelOutput {
        let shared = residualBlocks(relu(initialConv(input)))

        let logits = policyHead(shared)
        let policyOutput = softmax(logits)
        let valueOutput = valueHead(shared)

        return GoModelOutput(policy: policyOutput, value: valueOutput, logits: logits)
    }

    @differentiating(call)
    func _vjpCall(_ input: Tensor<Float>)
        -> (value: GoModelOutput, pullback: (GoModelOutput.CotangentVector)
        -> (GoModel.CotangentVector, Tensor<Float>)) {
        return (self(input), {
            seed in (GoModel.CotangentVector.zero, Tensor<Float>(0))
        })
    }
}

extension GoModel: InferenceModel {
    public func prediction(for input: Tensor<Float>) -> GoModelOutput {
        return self(input)
    }
}
