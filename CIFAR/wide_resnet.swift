import TensorFlow

// Original Paper:
// "Wide Residual Networks"
// Sergey Zagoruyko, Nikos Komodakis
// https://arxiv.org/abs/1605.07146
// https://github.com/szagoruyko/wide-residual-networks

struct BatchNormConv2DBlock: Layer {
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
    func applied(to input: Tensor<Float>) -> Tensor<Float> {
        let firstLayer = conv1.applied(to: relu(norm1.applied(to: input)))
        return conv2.applied(to: relu(norm2.applied(to: firstLayer)))
    }
}

struct WideResnet16FirstBasicBlock: Layer {
    var block1: BatchNormConv2DBlock
    var block2: BatchNormConv2DBlock
    var shortcut: Conv2D<Float>

    init(
        featureCounts: (Int, Int), 
        kernelSize: Int = 3, 
        widenFactor: Int = 1, 
        initialStride: (Int, Int) = (2, 2)
    ) {
        self.block1 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize, featureCounts.0, featureCounts.1 * widenFactor),
            strides: initialStride)
        self.block2 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize, featureCounts.1 * widenFactor, featureCounts.1 * widenFactor),
            strides: (1, 1))
        self.shortcut = Conv2D(
            filterShape: (1, 1, featureCounts.0, featureCounts.1 * widenFactor),
            strides: initialStride)
    }

    @differentiable
    func applied(to input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: block1, block2) + shortcut.applied(to: input)
    }
}

struct WideResnet16BasicBlock: Layer {
    var block1: BatchNormConv2DBlock
    var block2: BatchNormConv2DBlock
    var shortcut: Conv2D<Float>

    init(
        featureCounts: (Int, Int), 
        kernelSize: Int = 3, 
        widenFactor: Int = 1, 
        initialStride: (Int, Int) = (2, 2)
    ) {
        self.block1 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize, featureCounts.0 * widenFactor, featureCounts.1 * widenFactor),
            strides: initialStride)
        self.block2 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize, featureCounts.1 * widenFactor, featureCounts.1 * widenFactor),
            strides: (1, 1))
        self.shortcut = Conv2D(
            filterShape: (1, 1, featureCounts.0 * widenFactor, featureCounts.1 * widenFactor),
            strides: initialStride)
    }

    @differentiable
    func applied(to input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: block1, block2) + shortcut.applied(to: input)
    }
}

struct WideResNet16: Layer {
    var l1: Conv2D<Float>

    var l2a = WideResnet16FirstBasicBlock(featureCounts: (16, 16), widenFactor: 4, initialStride: (1, 1))
    var l3a = WideResnet16BasicBlock(featureCounts: (16, 32), widenFactor: 4)
    var l4a = WideResnet16BasicBlock(featureCounts: (32, 64), widenFactor: 4)
 
    var norm: BatchNorm<Float>
    var avgPool: AvgPool2D<Float>
    var flatten = Flatten<Float>()
    var classifier: Dense<Float>

    init() {
        self.l1 = Conv2D(filterShape: (3, 3, 3, 16), strides: (1, 1), padding: .same)
        self.norm = BatchNorm(featureCount: 64 * 4)
        self.avgPool = AvgPool2D(poolSize: (8, 8), strides: (8, 8))
        self.classifier = Dense(inputSize: 64 * 4, outputSize: 10)
    }

    @differentiable
    func applied(to input: Tensor<Float>) -> Tensor<Float> {
        let inputLayer = input.sequenced(through: l1, l2a, l3a, l4a)
        let finalNorm = relu(norm.applied(to: inputLayer))
        return finalNorm.sequenced(through: avgPool, flatten, classifier)
    }
}

struct WideResnet28FirstBasicBlock: Layer {
    var block1: BatchNormConv2DBlock
    var block2: BatchNormConv2DBlock
    var block3: BatchNormConv2DBlock
    var block4: BatchNormConv2DBlock
    var shortcut: Conv2D<Float>

    init(
        featureCounts: (Int, Int), 
        kernelSize: Int = 3, 
        widenFactor: Int = 1, 
        initialStride: (Int, Int) = (2, 2)
    ) {
        self.block1 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize, featureCounts.0, featureCounts.1 * widenFactor),
            strides: initialStride)
        self.block2 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize, featureCounts.1 * widenFactor, featureCounts.1 * widenFactor),
            strides: (1, 1))
        self.block3 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize, featureCounts.1 * widenFactor, featureCounts.1 * widenFactor),
            strides: (1, 1))
        self.block4 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize, featureCounts.1 * widenFactor, featureCounts.1 * widenFactor),
            strides: (1, 1))
        self.shortcut = Conv2D(
            filterShape: (1, 1, featureCounts.0, featureCounts.1 * widenFactor),
            strides: initialStride)
    }

    @differentiable
    func applied(to input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: block1, block2, block3, block4) + shortcut.applied(to: input)
    }
}

struct WideResnet28BasicBlock: Layer {
    var block1: BatchNormConv2DBlock
    var block2: BatchNormConv2DBlock
    var block3: BatchNormConv2DBlock
    var block4: BatchNormConv2DBlock
    var shortcut: Conv2D<Float>

    init(
        featureCounts: (Int, Int), 
        kernelSize: Int = 3, 
        widenFactor: Int = 1, 
        initialStride: (Int, Int) = (2, 2)
    ) {
        self.block1 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize, featureCounts.0 * widenFactor, featureCounts.1 * widenFactor),
            strides: initialStride)
        self.block2 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize, featureCounts.1 * widenFactor, featureCounts.1 * widenFactor),
            strides: (1, 1))
        self.block3 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize, featureCounts.1 * widenFactor, featureCounts.1 * widenFactor),
            strides: (1, 1))
        self.block4 = BatchNormConv2DBlock(
            filterShape: (kernelSize, kernelSize, featureCounts.1 * widenFactor, featureCounts.1 * widenFactor),
            strides: (1, 1))
        self.shortcut = Conv2D(
            filterShape: (1, 1, featureCounts.0 * widenFactor, featureCounts.1 * widenFactor),
            strides: initialStride)
    }

    @differentiable
    func applied(to input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: block1, block2) + shortcut.applied(to: input)
    }
}

struct WideResNet28: Layer {
    var l1: Conv2D<Float>

    var l2a = WideResnet28FirstBasicBlock(featureCounts: (16, 16), widenFactor: 10, initialStride: (1, 1))
    var l3a = WideResnet28BasicBlock(featureCounts: (16, 32), widenFactor: 10)
    var l4a = WideResnet28BasicBlock(featureCounts: (32, 64), widenFactor: 10)
 
    var norm: BatchNorm<Float>
    var avgPool: AvgPool2D<Float>
    var flatten = Flatten<Float>()
    var classifier: Dense<Float>

    init() {
        self.l1 = Conv2D(filterShape: (3, 3, 3, 16), strides: (1, 1), padding: .same)
        self.norm = BatchNorm(featureCount: 64 * 10)
        self.avgPool = AvgPool2D(poolSize: (8, 8), strides: (8, 8))
        self.classifier = Dense(inputSize: 64 * 10, outputSize: 10)
    }

    @differentiable
    func applied(to input: Tensor<Float>) -> Tensor<Float> {
        let inputLayer = input.sequenced(through: l1, l2a, l3a, l4a)
        let finalNorm = relu(norm.applied(to: inputLayer))
        return finalNorm.sequenced(through: avgPool, flatten, classifier)
    }
}
