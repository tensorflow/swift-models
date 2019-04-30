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

// Original Paper:
// "Wide Residual Networks"
// Sergey Zagoruyko, Nikos Komodakis
// https://arxiv.org/abs/1605.07146
// https://github.com/szagoruyko/wide-residual-networks

struct BatchNormConv2DBlock: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var norm1: BatchNorm<Float>
    var conv1: Conv2D<Float>
    var norm2: BatchNorm<Float>
    var conv2: Conv2D<Float>

    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .same
    ) {
        self.norm1 = BatchNorm(featureCount: filterShape.2)
        self.conv1 = Conv2D(filterShape: filterShape, strides: strides, padding: padding)
        self.norm2 = BatchNorm(featureCount: filterShape.3)
        self.conv2 = Conv2D(filterShape: filterShape, strides: (1, 1), padding: padding)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        let firstLayer = conv1(relu(norm1(input)))
        return conv2(relu(norm2(firstLayer)))
    }
}

struct WideResNetBasicBlock: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var blocks: [BatchNormConv2DBlock]
    var shortcut: Conv2D<Float>

    init(
        featureCounts: (Int, Int), 
        kernelSize: Int = 3,
        depthFactor: Int = 2,
        widenFactor: Int = 1,
        initialStride: (Int, Int) = (2, 2)
    ) {
        if initialStride == (1, 1) {
            self.blocks = [BatchNormConv2DBlock(
                filterShape: (kernelSize, kernelSize,
                    featureCounts.0, featureCounts.1 * widenFactor),
                strides: initialStride)]
            self.shortcut = Conv2D(
                filterShape: (1, 1, featureCounts.0, featureCounts.1 * widenFactor),
                strides: initialStride)
        } else {
            self.blocks = [BatchNormConv2DBlock(
                filterShape: (kernelSize, kernelSize,
                    featureCounts.0 * widenFactor, featureCounts.1 * widenFactor),
                strides: initialStride)]
            self.shortcut = Conv2D(
                filterShape: (1, 1, featureCounts.0 * widenFactor, featureCounts.1 * widenFactor),
                strides: initialStride)
        }
        for _ in 1..<depthFactor {
            self.blocks += [BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize,
                featureCounts.1 * widenFactor, featureCounts.1 * widenFactor),
            strides: (1, 1))]
        }
    }

    @differentiable
    func call(_ input: Input) -> Output {
        let blocksReduced = blocks.differentiableReduce(input) { last, layer in
            relu(layer(last))
        }
        return relu(blocksReduced + shortcut(input))
    }
}

struct WideResNet: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var l1: Conv2D<Float>

    var l2: WideResNetBasicBlock
    var l3: WideResNetBasicBlock
    var l4: WideResNetBasicBlock
 
    var norm: BatchNorm<Float>
    var avgPool: AvgPool2D<Float>
    var flatten = Flatten<Float>()
    var classifier: Dense<Float>

    init(depthFactor: Int = 2, widenFactor: Int = 8) {
        self.l1 = Conv2D(filterShape: (3, 3, 3, 16), strides: (1, 1), padding: .same)

        l2 = WideResNetBasicBlock(featureCounts: (16, 16), depthFactor: depthFactor,
            widenFactor: widenFactor, initialStride: (1, 1))
        l3 = WideResNetBasicBlock(featureCounts: (16, 32), depthFactor: depthFactor,
            widenFactor: widenFactor)
        l4 = WideResNetBasicBlock(featureCounts: (32, 64), depthFactor: depthFactor,
            widenFactor: widenFactor)
        
        self.norm = BatchNorm(featureCount: 64 * widenFactor)
        self.avgPool = AvgPool2D(poolSize: (8, 8), strides: (8, 8))
        self.classifier = Dense(inputSize: 64 * widenFactor, outputSize: 10)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        let inputLayer = input.sequenced(through: l1, l2, l3, l4)
        let finalNorm = relu(norm(inputLayer))
        return finalNorm.sequenced(through: avgPool, flatten, classifier)
    }
}

extension WideResNet {
    enum Kind {
        case wideResNet16
        case wideResNet16k8
        case wideResNet16k10
        case wideResNet22
        case wideResNet22k8
        case wideResNet22k10
        case wideResNet28
        case wideResNet28k10
        case wideResNet28k12
        case wideResNet40k1
        case wideResNet40k2
        case wideResNet40k4
        case wideResNet40k8
    }

    init(kind: Kind) {
        switch kind {
        case .wideResNet16, .wideResNet16k8:
            self.init(depthFactor: 2, widenFactor: 8)
        case .wideResNet16k10:
            self.init(depthFactor: 2, widenFactor: 10)
        case .wideResNet22, .wideResNet22k8:
            self.init(depthFactor: 3, widenFactor: 8)
        case .wideResNet22k10:
            self.init(depthFactor: 3, widenFactor: 10)
        case .wideResNet28, .wideResNet28k10:
            self.init(depthFactor: 4, widenFactor: 10)
        case .wideResNet28k12:
            self.init(depthFactor: 4, widenFactor: 12)
        case .wideResNet40k1:
            self.init(depthFactor: 6, widenFactor: 1)
        case .wideResNet40k2:
            self.init(depthFactor: 6, widenFactor: 2)
        case .wideResNet40k4:
            self.init(depthFactor: 6, widenFactor: 4)
        case .wideResNet40k8:
            self.init(depthFactor: 6, widenFactor: 8)
        }
    }
}

// TODO: remove this when TF supports differentiableReduce, thanks @rxwei!
public extension Array where Element: Differentiable {
    func differentiableReduce<Result: Differentiable>(
        _ initialResult: Result,
        _ nextPartialResult: @differentiable (Result, Element) -> Result
    ) -> Result {
        return reduce(initialResult, nextPartialResult)
    }

    @usableFromInline
    @differentiating(differentiableReduce(_:_:), wrt: (self, initialResult))
    internal func reduceDerivative<Result: Differentiable>(
        _ initialResult: Result,
        _ nextPartialResult: @differentiable (Result, Element) -> Result
    ) -> (value: Result,
          pullback: (Result.CotangentVector) -> (Array.CotangentVector, Result.CotangentVector)) {
        var pullbacks: [(Result.CotangentVector) -> (Result.CotangentVector, Element.CotangentVector)] = []
        let count = self.count
        pullbacks.reserveCapacity(count)
        var result = initialResult
        for element in self {
            let (y, pb) = Swift.valueWithPullback(at: result, element, in: nextPartialResult)
            result = y
            pullbacks.append(pb)
        }
        return (value: result, pullback: { cotangent in
            var resultCotangent = cotangent
            var elementCotangents = CotangentVector([])
            elementCotangents.base.reserveCapacity(count)
            for pullback in pullbacks.reversed() {
                let (newResultCotangent, elementCotangent) = pullback(resultCotangent)
                resultCotangent = newResultCotangent
                elementCotangents.base.append(elementCotangent)
            }
            return (CotangentVector(elementCotangents.base.reversed()), resultCotangent)
        })
    }
}
