import TensorFlow

struct ConvBN: Layer {
    var conv: Conv2D<Float>
    var norm: BatchNorm<Float>

    public init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid
    ) {
        self.conv = Conv2D(filterShape: filterShape, strides: strides, padding: padding)
        self.norm = BatchNorm(featureCount: filterShape.3)
    }

    @differentiable
    public func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        return norm.applied(to: conv.applied(to: input, in: context), in: context)
    }
}

struct ResidualConvBlock: Layer {
    var layer1: ConvBN
    var layer2: ConvBN
    var layer3: ConvBN
    var shortcut: ConvBN

    public init(
        featureCounts: (Int, Int, Int, Int),
        kernelSize: Int = 3,
        strides: (Int, Int) = (2, 2)
    ) {
        self.layer1 = ConvBN(
            filterShape: (1, 1, featureCounts.0, featureCounts.1),
            strides: strides)

        self.layer2 = ConvBN(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)

        self.layer3 = ConvBN(filterShape: (1, 1, featureCounts.2, featureCounts.3))

        self.shortcut = ConvBN(
            filterShape: (1, 1, featureCounts.0, featureCounts.3),
            strides: strides,
            padding: .same)
    }

    @differentiable
    func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        var tmp = relu(layer1.applied(to: input, in: context))
        tmp = relu(layer2.applied(to: tmp, in: context))
        tmp = layer3.applied(to: tmp, in: context)
        return relu(tmp + shortcut.applied(to: input, in: context))
    }
}

struct ResidualIdentityBlock: Layer {
    var layer1: ConvBN
    var layer2: ConvBN
    var layer3: ConvBN

    public init(featureCounts: (Int, Int, Int, Int), kernelSize: Int = 3) {
        self.layer1 = ConvBN(filterShape: (1, 1, featureCounts.0, featureCounts.1))

        self.layer2 = ConvBN(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)

        self.layer3 = ConvBN(filterShape: (1, 1, featureCounts.2, featureCounts.3))
    }

    @differentiable
    func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        var tmp = relu(layer1.applied(to: input, in: context))
        tmp = relu(layer2.applied(to: tmp, in: context))
        tmp = layer3.applied(to: tmp, in: context)
        return relu(tmp + input)
    }
}

public struct ResNet50: Layer {
    var l1: ConvBN
    var maxPool: MaxPool2D<Float>

    var l2a = ResidualConvBlock(featureCounts: (64, 64, 64, 256), strides: (1, 1))
    var l2b = ResidualIdentityBlock(featureCounts: (256, 64, 64, 256))
    var l2c = ResidualIdentityBlock(featureCounts: (256, 64, 64, 256))

    var l3a = ResidualConvBlock(featureCounts: (256, 128, 128, 512))
    var l3b = ResidualIdentityBlock(featureCounts: (512, 128, 128, 512))
    var l3c = ResidualIdentityBlock(featureCounts: (512, 128, 128, 512))
    var l3d = ResidualIdentityBlock(featureCounts: (512, 128, 128, 512))

    var l4a = ResidualConvBlock(featureCounts: (512, 256, 256, 1024))
    var l4b = ResidualIdentityBlock(featureCounts: (1024, 256, 256, 1024))
    var l4c = ResidualIdentityBlock(featureCounts: (1024, 256, 256, 1024))
    var l4d = ResidualIdentityBlock(featureCounts: (1024, 256, 256, 1024))
    var l4e = ResidualIdentityBlock(featureCounts: (1024, 256, 256, 1024))
    var l4f = ResidualIdentityBlock(featureCounts: (1024, 256, 256, 1024))

    var l5a = ResidualConvBlock(featureCounts: (1024, 512, 512, 2048))
    var l5b = ResidualIdentityBlock(featureCounts: (2048, 512, 512, 2048))
    var l5c = ResidualIdentityBlock(featureCounts: (2048, 512, 512, 2048))
    var avgPool: AvgPool2D<Float>

    var flatten = Flatten<Float>()
    var classifier: Dense<Float>

    public init(imageSize: Int, classCount: Int) {
        // default to the ImageNet case where imageSize == 224
        // Swift requires that all properties get initialized outside control flow
        l1 = ConvBN(filterShape: (7, 7, 3, 64), strides: (2, 2), padding: .same)
        maxPool = MaxPool2D(poolSize: (3, 3), strides: (2, 2))
        avgPool = AvgPool2D(poolSize: (7, 7), strides: (7, 7))
        if imageSize == 32 {
            l1 = ConvBN(filterShape: (3, 3, 3, 64), padding: .same)
            maxPool = MaxPool2D(poolSize: (1, 1), strides: (1, 1)) // no-op
            avgPool = AvgPool2D(poolSize: (4, 4), strides: (4, 4))
        }
        classifier = Dense(inputSize: 2048, outputSize: classCount)
    }

    @differentiable
    public func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        var tmp = input
        tmp = maxPool.applied(to: relu(l1.applied(to: input, in: context)), in: context)
        tmp = l2a.applied(to: tmp, in: context)
        tmp = l2b.applied(to: tmp, in: context)
        tmp = l2c.applied(to: tmp, in: context)

        tmp = l3a.applied(to: tmp, in: context)
        tmp = l3b.applied(to: tmp, in: context)
        tmp = l3c.applied(to: tmp, in: context)
        tmp = l3d.applied(to: tmp, in: context)

        tmp = l4a.applied(to: tmp, in: context)
        tmp = l4b.applied(to: tmp, in: context)
        tmp = l4c.applied(to: tmp, in: context)
        tmp = l4d.applied(to: tmp, in: context)
        tmp = l4e.applied(to: tmp, in: context)
        tmp = l4f.applied(to: tmp, in: context)

        tmp = l5a.applied(to: tmp, in: context)
        tmp = l5b.applied(to: tmp, in: context)
        tmp = avgPool.applied(to: l5c.applied(to: tmp, in: context), in: context)
        tmp = flatten.applied(to: tmp, in: context)
        return classifier.applied(to: tmp, in: context)
    }
}
