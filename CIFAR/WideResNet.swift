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

struct IdentityLayer: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var norm1: BatchNorm<Float>
    var conv1: Conv2D<Float>
    var norm2: BatchNorm<Float>
    let dropout = Dropout<Float>(probability: 0.3)
    var conv2: Conv2D<Float>
    

    init(
        filterShape: (Int, Int, Int, Int),
        padding: Padding = .same
    ) {
        self.norm1 = BatchNorm(featureCount: filterShape.3)
        self.conv1 = Conv2D(filterShape: filterShape, strides: (1,1), padding: padding)
        self.norm2 = BatchNorm(featureCount: filterShape.3)
        self.conv2 = Conv2D(filterShape: filterShape, strides: (1, 1), padding: padding)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        let preactivation1 = relu(norm1(input))
        let firstLayer = conv1(preactivation1)
        let preactivation2 = dropout(relu(norm2(firstLayer)))
        return conv2(preactivation2) + input
    }
}

struct ExpansionLayer: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var norm1: BatchNorm<Float>
    var conv1: Conv2D<Float>
    var norm2: BatchNorm<Float>
    let dropout = Dropout<Float>(probability: 0.3)
    var conv2: Conv2D<Float>
    var shortcut: Conv2D<Float>

    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .same
    ) {
        self.norm1 = BatchNorm(featureCount: filterShape.2)
        self.conv1 = Conv2D(filterShape: filterShape, strides: strides, padding: padding)
        self.norm2 = BatchNorm(featureCount: filterShape.3)
        self.conv2 = Conv2D(filterShape: (filterShape.0, filterShape.1,
                                          filterShape.3, filterShape.3),
                            strides: (1, 1), padding: padding)
        self.shortcut = Conv2D(filterShape: (1, 1, filterShape.2, filterShape.3), 
                               strides: strides, padding: padding)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        let preactivation1 = relu(norm1(input))
        let firstLayer = conv1(preactivation1)
        let preactivation2 = dropout(relu(norm2(firstLayer)))
        return conv2(preactivation2) + shortcut(preactivation1)
    }
}

struct WideResNetBasicBlock: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>
    var expansion: ExpansionLayer
    var blocks: [IdentityLayer]

    init(
        featureCounts: (Int, Int), 
        kernelSize: Int = 3,
        depthFactor: Int = 2,
        initialStride: (Int, Int) = (2, 2)
    ) {
        self.expansion = ExpansionLayer(
            filterShape: (kernelSize, kernelSize,
                featureCounts.0, featureCounts.1),
            strides: initialStride)
        self.blocks = []
        for _ in 1..<depthFactor {
            self.blocks += [IdentityLayer(
            filterShape: (kernelSize, kernelSize,
                          featureCounts.1, featureCounts.1))]
        }
    }

    @differentiable
    func call(_ input: Input) -> Output {
        var net = expansion(input)
        net = blocks.differentiableReduce(net) { last, layer in
            layer(last)
        }
        return net
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
        let featureCount1 = 16
        let featureCount2 = 16 * widenFactor
        let featureCount3 = 32 * widenFactor
        let featureCount4 = 64 * widenFactor
        self.l1 = Conv2D(filterShape: (3, 3, 3, featureCount1), strides: (1, 1), padding: .same)

        l2 = WideResNetBasicBlock(featureCounts: (featureCount1, featureCount2), 
                                  depthFactor: depthFactor,
                                  initialStride: (1, 1))
        l3 = WideResNetBasicBlock(featureCounts: (featureCount2, featureCount3), 
                                  depthFactor: depthFactor)
        l4 = WideResNetBasicBlock(featureCounts: (featureCount3, featureCount4), 
                                  depthFactor: depthFactor)
        
        self.norm = BatchNorm(featureCount: featureCount4)
        self.avgPool = AvgPool2D(poolSize: (8, 8), strides: (8, 8))
        self.classifier = Dense(inputSize: featureCount4, outputSize: 10)
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
