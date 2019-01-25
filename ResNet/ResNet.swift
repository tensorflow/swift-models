import TensorFlow

struct ConvBN: Layer {
    var conv: Conv2D<Float>
    var norm: BatchNorm<Float>

    public init(filterShape: (Int, Int, Int, Int), strides: (Int, Int) = (1, 1),
        padding: Padding, modeRef: ModeRef) {
        self.conv = Conv2D(
            filterShape: filterShape, strides: strides, padding: padding)
        self.norm = BatchNorm(featureCount: filterShape.3, modeRef: modeRef)
    }

    public func applied(to input: Tensor<Float>) -> Tensor<Float> {
        return norm.applied(to: conv.applied(to: input))
    }
}

// AD on indirect passing would let us have ResidualBlock<Shortcut : Layer>
// where Shortcut is one of ConvBN or
// struct Identity: Layer {
//     public func applied(to input: Tensor<Float>) -> Tensor<Float> {
//         return input
//     }
// }

struct ResidualConvBlock: Layer {
    var layer1: ConvBN
    var layer2: ConvBN
    var layer3: ConvBN
    var shortcut: ConvBN

    public init(
        featureCounts: (Int, Int, Int, Int),
        kernelSize: Int = 3,
        strides: (Int, Int) = (2, 2),
        modeRef: ModeRef
    ) {
        self.layer1 = ConvBN(
            filterShape: (1, 1, featureCounts.0, featureCounts.1),
            strides: strides, padding: .valid, modeRef: modeRef)

        self.layer2 = ConvBN(
            filterShape: (
                kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same, modeRef: modeRef)

        self.layer3 = ConvBN(
            filterShape: (1, 1, featureCounts.2, featureCounts.3),
            padding: .valid, modeRef: modeRef)

        self.shortcut = ConvBN(
            filterShape: (1, 1, featureCounts.0, featureCounts.3),
            strides: strides,
            padding: .same, modeRef: modeRef)
    }

    func applied(to input: Tensor<Float>) -> Tensor<Float> {
        var tmp = relu(layer1.applied(to: input))
        tmp = relu(layer2.applied(to: tmp))
        tmp = layer3.applied(to: tmp)
        return relu(tmp + shortcut.applied(to: input))
    }
}

struct ResidualIdentityBlock: Layer {
    var layer1: ConvBN
    var layer2: ConvBN
    var layer3: ConvBN

    public init(featureCounts: (Int, Int, Int, Int), kernelSize: Int = 3,
                modeRef: ModeRef) {
        self.layer1 = ConvBN(
            filterShape: (1, 1, featureCounts.0, featureCounts.1),
            padding: .valid, modeRef: modeRef)

        self.layer2 = ConvBN(
            filterShape: (
                kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same, modeRef: modeRef)

        self.layer3 = ConvBN(
            filterShape: (1, 1, featureCounts.2, featureCounts.3),
            padding: .valid, modeRef: modeRef)
    }

    func applied(to input: Tensor<Float>) -> Tensor<Float> {
        var tmp = relu(layer1.applied(to: input))
        tmp = relu(layer2.applied(to: tmp))
        tmp = layer3.applied(to: tmp)
        return relu(tmp + input)
    }
}

public struct ResNet50: Layer {
    var l1: ConvBN
    @noDerivative let maxPool: MaxPool2D<Float>

    var l2a: ResidualConvBlock
    var l2b: ResidualIdentityBlock
    var l2c: ResidualIdentityBlock

    var l3a: ResidualConvBlock
    var l3b: ResidualIdentityBlock
    var l3c: ResidualIdentityBlock
    var l3d: ResidualIdentityBlock

    var l4a: ResidualConvBlock
    var l4b: ResidualIdentityBlock
    var l4c: ResidualIdentityBlock
    var l4d: ResidualIdentityBlock
    var l4e: ResidualIdentityBlock
    var l4f: ResidualIdentityBlock

    var l5a: ResidualConvBlock
    var l5b: ResidualIdentityBlock
    var l5c: ResidualIdentityBlock
    @noDerivative let avgPool: AvgPool2D<Float>

    var classifier: Dense<Float>

    public init(classCount: Int, modeRef: ModeRef) {
        self.l1 = ConvBN(filterShape: (7, 7, 3, 64), strides: (2, 2),
                         padding: .same, modeRef: modeRef)
        self.maxPool = MaxPool2D<Float>(
            poolSize: (3, 3), strides: (2, 2), padding: .valid)

        self.l2a = ResidualConvBlock(
            featureCounts: (64, 64, 64, 256), strides: (1, 1), modeRef: modeRef)
        self.l2b = ResidualIdentityBlock(
            featureCounts: (256, 64, 64, 256), modeRef: modeRef)
        self.l2c = ResidualIdentityBlock(
            featureCounts: (256, 64, 64, 256), modeRef: modeRef)

        self.l3a = ResidualConvBlock(
            featureCounts: (256, 128, 128, 512), modeRef: modeRef)
        self.l3b = ResidualIdentityBlock(
            featureCounts: (512, 128, 128, 512), modeRef: modeRef)
        self.l3c = ResidualIdentityBlock(
            featureCounts: (512, 128, 128, 512), modeRef: modeRef)
        self.l3d = ResidualIdentityBlock(
            featureCounts: (512, 128, 128, 512), modeRef: modeRef)

        self.l4a = ResidualConvBlock(
            featureCounts: (512, 256, 256, 1024), modeRef: modeRef)
        self.l4b = ResidualIdentityBlock(
            featureCounts: (1024, 256, 256, 1024), modeRef: modeRef)
        self.l4c = ResidualIdentityBlock(
            featureCounts: (1024, 256, 256, 1024), modeRef: modeRef)
        self.l4d = ResidualIdentityBlock(
            featureCounts: (1024, 256, 256, 1024), modeRef: modeRef)
        self.l4e = ResidualIdentityBlock(
            featureCounts: (1024, 256, 256, 1024), modeRef: modeRef)
        self.l4f = ResidualIdentityBlock(
            featureCounts: (1024, 256, 256, 1024), modeRef: modeRef)

        self.l5a = ResidualConvBlock(
            featureCounts: (1024, 512, 512, 2048), modeRef: modeRef)
        self.l5b = ResidualIdentityBlock(
            featureCounts: (2048, 512, 512, 2048), modeRef: modeRef)
        self.l5c = ResidualIdentityBlock(
            featureCounts: (2048, 512, 512, 2048), modeRef: modeRef)
        self.avgPool = AvgPool2D<Float>(
            poolSize: (7, 7), strides: (7, 7), padding: .valid)

        self.classifier = Dense<Float>(
            inputSize: 2048, outputSize: classCount, activation: { $0 })
    }

    public func applied(to input: Tensor<Float>) -> Tensor<Float> {
        var tmp = input
        tmp = maxPool.applied(to: relu(l1.applied(to: input)))
        tmp = l2a.applied(to: tmp)
        tmp = l2b.applied(to: tmp)
        tmp = l2c.applied(to: tmp)

        tmp = l3a.applied(to: tmp)
        tmp = l3b.applied(to: tmp)
        tmp = l3c.applied(to: tmp)
        tmp = l3d.applied(to: tmp)

        tmp = l4a.applied(to: tmp)
        tmp = l4b.applied(to: tmp)
        tmp = l4c.applied(to: tmp)
        tmp = l4d.applied(to: tmp)
        tmp = l4e.applied(to: tmp)
        tmp = l4f.applied(to: tmp)

        tmp = l5a.applied(to: tmp)
        tmp = l5b.applied(to: tmp)
        tmp = avgPool.applied(to: l5c.applied(to: tmp))

        return classifier.applied(to: tmp)
    }
}
