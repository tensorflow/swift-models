import TensorFlow

// Original Paper:
// "Deep Residual Learning for Image Recognition"
// Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
// https://arxiv.org/abs/1512.03385
// using shortcut layer to connect BasicBlock layers (aka Option (B))
// see https://github.com/akamaster/pytorch_resnet_cifar10 for explanation

struct Conv2DBatchNorm: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var conv: Conv2D<Float>
    var norm: BatchNorm<Float>

    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1)
    ) {
        self.conv = Conv2D(filterShape: filterShape, strides: strides, padding: .same)
        self.norm = BatchNorm(featureCount: filterShape.3)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        return input.sequenced(through: conv, norm)
    }
}

struct BasicBlock20: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

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
            strides: strides)
        self.layer2 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2))
        self.shortcut = Conv2DBatchNorm(
            filterShape: (1, 1, featureCounts.0, featureCounts.3),
            strides: strides)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        var tmp = relu(layer1(input))
        tmp = relu(layer2(tmp))
        return relu(tmp + shortcut(input))
    }
}

struct ResNet20: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var inputLayer = Conv2DBatchNorm(filterShape: (3, 3, 3, 16))

    var basicBlock1 = BasicBlock20(featureCounts:(16, 16, 16, 16), strides: (1,1))
    var basicBlock2 = BasicBlock20(featureCounts:(16, 32, 32, 32))
    var basicBlock3 = BasicBlock20(featureCounts:(32, 64, 64, 64))

    var averagePool = AvgPool2D<Float>(poolSize: (8, 8), strides: (8, 8))
    var flatten = Flatten<Float>()
    var classifier = Dense<Float>(inputSize: 64, outputSize: 10, activation: softmax)

    @differentiable
    func call(_ input: Input) -> Output {
        let tmp = relu(inputLayer(input))
        let convolved = tmp.sequenced(through: basicBlock1, basicBlock2, basicBlock3)
        return convolved.sequenced(through: averagePool, flatten, classifier)
    }
}

struct BasicBlock32: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

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
            strides: strides)
        self.layer2 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2))
        self.layer3 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2))
        self.layer4 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2))
        self.shortcut = Conv2DBatchNorm(
            filterShape: (1, 1, featureCounts.0, featureCounts.3),
            strides: strides)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        var tmp = relu(layer1(input))
        tmp = relu(layer2(tmp))
        tmp = relu(layer3(tmp))
        tmp = relu(layer4(tmp))
        return relu(tmp + shortcut(input))
    }
}

struct ResNet32: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var inputLayer = Conv2DBatchNorm(filterShape: (3, 3, 3, 16))

    var basicBlock1 = BasicBlock32(featureCounts:(16, 16, 16, 16), strides: (1,1))
    var basicBlock2 = BasicBlock32(featureCounts:(16, 32, 32, 32))
    var basicBlock3 = BasicBlock32(featureCounts:(32, 64, 64, 64))

    var averagePool = AvgPool2D<Float>(poolSize: (8, 8), strides: (8, 8))
    var flatten = Flatten<Float>()
    var classifier = Dense<Float>(inputSize: 64, outputSize: 10, activation: softmax)

    @differentiable
    func call(_ input: Input) -> Output {
        let tmp = relu(inputLayer(input))
        let convolved = tmp.sequenced(through: basicBlock1, basicBlock2, basicBlock3)
        return convolved.sequenced(through: averagePool, flatten, classifier)
    }
}

struct BasicBlock44: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

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
            strides: strides)
        self.layer2 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2))
        self.layer3 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2))
        self.layer4 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2))
        self.layer5 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2))
        self.layer6 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2))
        self.shortcut = Conv2DBatchNorm(
            filterShape: (1, 1, featureCounts.0, featureCounts.3),
            strides: strides)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        var tmp = relu(layer1(input))
        tmp = relu(layer2(tmp))
        tmp = relu(layer3(tmp))
        tmp = relu(layer4(tmp))
        tmp = relu(layer5(tmp))
        tmp = relu(layer6(tmp))
        return relu(tmp + shortcut(input))
    }
}

struct ResNet44: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var inputLayer = Conv2DBatchNorm(filterShape: (3, 3, 3, 16))

    var basicBlock1 = BasicBlock44(featureCounts:(16, 16, 16, 16), strides: (1,1))
    var basicBlock2 = BasicBlock44(featureCounts:(16, 32, 32, 32))
    var basicBlock3 = BasicBlock44(featureCounts:(32, 64, 64, 64))

    var averagePool = AvgPool2D<Float>(poolSize: (8, 8), strides: (8, 8))
    var flatten = Flatten<Float>()
    var classifier = Dense<Float>(inputSize: 64, outputSize: 10, activation: softmax)

    @differentiable
    func call(_ input: Input) -> Output {
        let tmp = relu(inputLayer(input))
        let convolved = tmp.sequenced(through: basicBlock1, basicBlock2, basicBlock3)
        return convolved.sequenced(through: averagePool, flatten, classifier)
    }
}

struct BasicBlock56: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

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
            strides: strides)
        self.layer2 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2))
        self.layer3 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2))
        self.layer4 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2))
        self.layer5 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2))
        self.layer6 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2))
        self.layer7 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2))
        self.layer8 = Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2))
        self.shortcut = Conv2DBatchNorm(
            filterShape: (1, 1, featureCounts.0, featureCounts.3),
            strides: strides)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        var tmp = relu(layer1(input))
        tmp = relu(layer2(tmp))
        tmp = relu(layer3(tmp))
        tmp = relu(layer4(tmp))
        tmp = relu(layer5(tmp))
        tmp = relu(layer6(tmp))
        tmp = relu(layer7(tmp))
        tmp = relu(layer8(tmp))
        return relu(tmp + shortcut(input))
    }
}

struct ResNet56: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var inputLayer = Conv2DBatchNorm(filterShape: (3, 3, 3, 16))

    var basicBlock1 = BasicBlock56(featureCounts:(16, 16, 16, 16), strides: (1,1))
    var basicBlock2 = BasicBlock56(featureCounts:(16, 32, 32, 32))
    var basicBlock3 = BasicBlock56(featureCounts:(32, 64, 64, 64))

    var averagePool = AvgPool2D<Float>(poolSize: (8, 8), strides: (8, 8))
    var flatten = Flatten<Float>()
    var classifier = Dense<Float>(inputSize: 64, outputSize: 10, activation: softmax)

    @differentiable
    func call(_ input: Input) -> Output {
        let tmp = relu(inputLayer(input))
        let convolved = tmp.sequenced(through: basicBlock1, basicBlock2, basicBlock3)
        return convolved.sequenced(through: averagePool, flatten, classifier)
    }
}
