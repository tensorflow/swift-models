import TensorFlow

// Original Paper:
// "Deep Residual Learning for Image Recognition"
// Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
// https://arxiv.org/abs/1512.03385
// using shortcut layer to connect BasicBlock layers (aka Option (B))
// see https://github.com/akamaster/pytorch_resnet_cifar10 for explanation

struct Conv2DBatchNorm: Layer {
    var conv: Conv2D<Float>
    var norm: BatchNorm<Float>

    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid
    ) {
        self.conv = Conv2D(filterShape: filterShape, strides: strides, padding: padding)
        self.norm = BatchNorm(featureCount: filterShape.3)
    }

    @differentiable
    func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        var tmp = input
        tmp = conv.applied(to: tmp, in: context)
        tmp = norm.applied(to: tmp, in: context)
        return tmp
    }
}

struct BasicBlock20: Layer {
    var layer1: Conv2DBatchNorm
    var layer2: Conv2DBatchNorm
    var shortcut: Conv2DBatchNorm

    init(
        featureCounts: (Int, Int, Int, Int),
        kernelSize: Int = 3,
        strides: (Int, Int) = (2, 2)
    ) {
        self.layer1 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.0, featureCounts.1),
            strides: strides,
            padding: .same)

        self.layer2 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)

        self.shortcut = Conv2DBatchNorm(
            filterShape: (1, 1, featureCounts.0, featureCounts.3),
            strides: strides,
            padding: .same)
    }

    @differentiable
    func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        var tmp = relu(layer1.applied(to: input, in: context))
        tmp = relu(layer2.applied(to: tmp, in: context))
        return relu(tmp + shortcut.applied(to: input, in: context))
    }
}

struct ResNet20: Layer {
    var inputLayer = Conv2DBatchNorm(filterShape: (3, 3, 3, 16), padding: .same)

    var basicBlock1 = BasicBlock20(featureCounts:(16, 16, 16, 16), strides: (1,1))
    var basicBlock2 = BasicBlock20(featureCounts:(16, 32, 32, 32))
    var basicBlock3 = BasicBlock20(featureCounts:(32, 64, 64, 64))

    var averagePool = AvgPool2D<Float>(poolSize: (8, 8), strides: (8, 8))
    var flatten = Flatten<Float>()
    var classifier = Dense<Float>(inputSize: 64, outputSize: 10, activation: softmax)

    @differentiable
    func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        var tmp = relu(inputLayer.applied(to: input, in: context))

        tmp = basicBlock1.applied(to: tmp, in: context)
        tmp = basicBlock2.applied(to: tmp, in: context)
        tmp = basicBlock3.applied(to: tmp, in: context)

        tmp = averagePool.applied(to: tmp, in: context)
        tmp = flatten.applied(to: tmp, in: context)
        return classifier.applied(to: tmp, in: context)
    }
}

struct BasicBlock32: Layer {
    var layer1: Conv2DBatchNorm
    var layer2: Conv2DBatchNorm
    var layer3: Conv2DBatchNorm
    var layer4: Conv2DBatchNorm
    var shortcut: Conv2DBatchNorm

    init(
        featureCounts: (Int, Int, Int, Int),
        kernelSize: Int = 3,
        strides: (Int, Int) = (2, 2)
    ) {
        self.layer1 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.0, featureCounts.1),
            strides: strides,
            padding: .same)

        self.layer2 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)

        self.layer3 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)

        self.layer4 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)

        self.shortcut = Conv2DBatchNorm(
            filterShape: (1, 1, featureCounts.0, featureCounts.3),
            strides: strides,
            padding: .same)
    }

    @differentiable
    func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        var tmp = relu(layer1.applied(to: input, in: context))
        tmp = relu(layer2.applied(to: tmp, in: context))
        tmp = relu(layer3.applied(to: tmp, in: context))
        tmp = relu(layer4.applied(to: tmp, in: context))
        return relu(tmp + shortcut.applied(to: input, in: context))
    }
}

struct ResNet32: Layer {
    var inputLayer = Conv2DBatchNorm(filterShape: (3, 3, 3, 16), padding: .same)

    var basicBlock1 = BasicBlock32(featureCounts:(16, 16, 16, 16), strides: (1,1))
    var basicBlock2 = BasicBlock32(featureCounts:(16, 32, 32, 32))
    var basicBlock3 = BasicBlock32(featureCounts:(32, 64, 64, 64))

    var averagePool = AvgPool2D<Float>(poolSize: (8, 8), strides: (8, 8))
    var flatten = Flatten<Float>()
    var classifier = Dense<Float>(inputSize: 64, outputSize: 10, activation: softmax)

    @differentiable
    func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        var tmp = relu(inputLayer.applied(to: input, in: context))

        tmp = basicBlock1.applied(to: tmp, in: context)
        tmp = basicBlock2.applied(to: tmp, in: context)
        tmp = basicBlock3.applied(to: tmp, in: context)

        tmp = averagePool.applied(to: tmp, in: context)
        tmp = flatten.applied(to: tmp, in: context)
        return classifier.applied(to: tmp, in: context)
    }
}

struct BasicBlock44: Layer {
    var layer1: Conv2DBatchNorm
    var layer2: Conv2DBatchNorm
    var layer3: Conv2DBatchNorm
    var layer4: Conv2DBatchNorm
    var layer5: Conv2DBatchNorm
    var layer6: Conv2DBatchNorm
    var shortcut: Conv2DBatchNorm

    init(
        featureCounts: (Int, Int, Int, Int),
        kernelSize: Int = 3,
        strides: (Int, Int) = (2, 2)
    ) {
        self.layer1 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.0, featureCounts.1),
            strides: strides,
            padding: .same)

        self.layer2 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)

        self.layer3 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)

        self.layer4 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)

        self.layer5 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)

        self.layer6 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)

        self.shortcut = Conv2DBatchNorm(
            filterShape: (1, 1, featureCounts.0, featureCounts.3),
            strides: strides,
            padding: .same)
    }

    @differentiable
    func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        var tmp = relu(layer1.applied(to: input, in: context))
        tmp = relu(layer2.applied(to: tmp, in: context))
        tmp = relu(layer3.applied(to: tmp, in: context))
        tmp = relu(layer4.applied(to: tmp, in: context))
        tmp = relu(layer5.applied(to: tmp, in: context))
        tmp = relu(layer6.applied(to: tmp, in: context))
        return relu(tmp + shortcut.applied(to: input, in: context))
    }
}

struct ResNet44: Layer {
    var inputLayer = Conv2DBatchNorm(filterShape: (3, 3, 3, 16), padding: .same)

    var basicBlock1 = BasicBlock44(featureCounts:(16, 16, 16, 16), strides: (1,1))
    var basicBlock2 = BasicBlock44(featureCounts:(16, 32, 32, 32))
    var basicBlock3 = BasicBlock44(featureCounts:(32, 64, 64, 64))

    var averagePool = AvgPool2D<Float>(poolSize: (8, 8), strides: (8, 8))
    var flatten = Flatten<Float>()
    var classifier = Dense<Float>(inputSize: 64, outputSize: 10, activation: softmax)

    @differentiable
    func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        var tmp = relu(inputLayer.applied(to: input, in: context))

        tmp = basicBlock1.applied(to: tmp, in: context)
        tmp = basicBlock2.applied(to: tmp, in: context)
        tmp = basicBlock3.applied(to: tmp, in: context)

        tmp = averagePool.applied(to: tmp, in: context)
        tmp = flatten.applied(to: tmp, in: context)
        return classifier.applied(to: tmp, in: context)
    }
}

struct BasicBlock56: Layer {
    var layer1: Conv2DBatchNorm
    var layer2: Conv2DBatchNorm
    var layer3: Conv2DBatchNorm
    var layer4: Conv2DBatchNorm
    var layer5: Conv2DBatchNorm
    var layer6: Conv2DBatchNorm
    var layer7: Conv2DBatchNorm
    var layer8: Conv2DBatchNorm
    var shortcut: Conv2DBatchNorm

    init(
        featureCounts: (Int, Int, Int, Int),
        kernelSize: Int = 3,
        strides: (Int, Int) = (2, 2)
    ) {
        self.layer1 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.0, featureCounts.1),
            strides: strides,
            padding: .same)

        self.layer2 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)

        self.layer3 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)

        self.layer4 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)

        self.layer5 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)

        self.layer6 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)

        self.layer7 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)

        self.layer8 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)

        self.shortcut = Conv2DBatchNorm(
            filterShape: (1, 1, featureCounts.0, featureCounts.3),
            strides: strides,
            padding: .same)
    }

    @differentiable
    func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        var tmp = relu(layer1.applied(to: input, in: context))
        tmp = relu(layer2.applied(to: tmp, in: context))
        tmp = relu(layer3.applied(to: tmp, in: context))
        tmp = relu(layer4.applied(to: tmp, in: context))
        tmp = relu(layer5.applied(to: tmp, in: context))
        tmp = relu(layer6.applied(to: tmp, in: context))
        tmp = relu(layer7.applied(to: tmp, in: context))
        tmp = relu(layer8.applied(to: tmp, in: context))
        return relu(tmp + shortcut.applied(to: input, in: context))
    }
}

struct ResNet56: Layer {
    var inputLayer = Conv2DBatchNorm(filterShape: (3, 3, 3, 16), padding: .same)

    var basicBlock1 = BasicBlock56(featureCounts:(16, 16, 16, 16), strides: (1,1))
    var basicBlock2 = BasicBlock56(featureCounts:(16, 32, 32, 32))
    var basicBlock3 = BasicBlock56(featureCounts:(32, 64, 64, 64))

    var averagePool = AvgPool2D<Float>(poolSize: (8, 8), strides: (8, 8))
    var flatten = Flatten<Float>()
    var classifier = Dense<Float>(inputSize: 64, outputSize: 10, activation: softmax)

    @differentiable
    func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        var tmp = relu(inputLayer.applied(to: input, in: context))

        tmp = basicBlock1.applied(to: tmp, in: context)
        tmp = basicBlock2.applied(to: tmp, in: context)
        tmp = basicBlock3.applied(to: tmp, in: context)

        tmp = averagePool.applied(to: tmp, in: context)
        tmp = flatten.applied(to: tmp, in: context)
        return classifier.applied(to: tmp, in: context)
    }
}
