import TensorFlow

struct ConvBN: Layer {
    var conv: Conv2D<Float>
    var norm: BatchNorm<Float>

    public init(filterShape: (Int, Int, Int, Int), strides: (Int, Int) = (1, 1), padding: Padding) {
        self.conv = Conv2D(filterShape: filterShape, strides: strides, padding: padding)
        self.norm = BatchNorm(featureCount: filterShape.3)
    }

    @differentiable
    public func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        return norm.applied(to: conv.applied(to: input, in: context), in: context)
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
        strides: (Int, Int) = (2, 2)
    ) {
        self.layer1 = ConvBN(
            filterShape: (1, 1, featureCounts.0, featureCounts.1),
            strides: strides, padding: .valid)

        self.layer2 = ConvBN(
            filterShape: (
                kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)

        self.layer3 = ConvBN(
            filterShape: (1, 1, featureCounts.2, featureCounts.3),
            padding: .valid)

        self.shortcut = ConvBN(
            filterShape: (1, 1, featureCounts.0, featureCounts.3),
            strides: strides, padding: .same)
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
        self.layer1 = ConvBN(
            filterShape: (1, 1, featureCounts.0, featureCounts.1),
            padding: .valid)

        self.layer2 = ConvBN(
            filterShape: (
                kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)

        self.layer3 = ConvBN(
            filterShape: (1, 1, featureCounts.2, featureCounts.3),
            padding: .valid)
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

    public init(
        classCount: Int
    ) {
        self.l1 = ConvBN(
            filterShape: (7, 7, 3, 64), strides: (2, 2),
            padding: .same)
        self.maxPool = MaxPool2D<Float>(
            poolSize: (3, 3), strides: (2, 2), padding: .valid)

        self.l2a = ResidualConvBlock(
            featureCounts: (64, 64, 64, 256), strides: (1, 1))
        self.l2b = ResidualIdentityBlock(
            featureCounts: (256, 64, 64, 256))
        self.l2c = ResidualIdentityBlock(
            featureCounts: (256, 64, 64, 256))

        self.l3a = ResidualConvBlock(
            featureCounts: (256, 128, 128, 512))
        self.l3b = ResidualIdentityBlock(
            featureCounts: (512, 128, 128, 512))
        self.l3c = ResidualIdentityBlock(
            featureCounts: (512, 128, 128, 512))
        self.l3d = ResidualIdentityBlock(
            featureCounts: (512, 128, 128, 512))

        self.l4a = ResidualConvBlock(
            featureCounts: (512, 256, 256, 1024))
        self.l4b = ResidualIdentityBlock(
            featureCounts: (1024, 256, 256, 1024))
        self.l4c = ResidualIdentityBlock(
            featureCounts: (1024, 256, 256, 1024))
        self.l4d = ResidualIdentityBlock(
            featureCounts: (1024, 256, 256, 1024))
        self.l4e = ResidualIdentityBlock(
            featureCounts: (1024, 256, 256, 1024))
        self.l4f = ResidualIdentityBlock(
            featureCounts: (1024, 256, 256, 1024))

        self.l5a = ResidualConvBlock(
            featureCounts: (1024, 512, 512, 2048))
        self.l5b = ResidualIdentityBlock(
            featureCounts: (2048, 512, 512, 2048))
        self.l5c = ResidualIdentityBlock(
            featureCounts: (2048, 512, 512, 2048))
        self.avgPool = AvgPool2D<Float>(
            poolSize: (7, 7), strides: (7, 7), padding: .valid)

        self.classifier = Dense<Float>(
            inputSize: 2048, outputSize: classCount, activation: { $0 })
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
        tmp = tmp.reshaped(toShape: Tensor<Int32>(
            [tmp.shape[Int32(0)], tmp.shape[Int32(3)]]))
        return classifier.applied(to: tmp, in: context)
    }
}

let batchSize: Int32 = 128
let classCount = 1000

let fakeImageBatch = Tensor<Float>(zeros: [batchSize, 224, 224, 3])
let fakeLabelBatch = Tensor<Int32>(zeros: [batchSize])

var resnet = ResNet50(classCount: classCount)
let context = Context(learningPhase: .training)
let optimizer = SGD<ResNet50, Float>(learningRate: 0.1, momentum: 0.9)

for _ in 0..<10 {
    let gradients = gradient(at: resnet) { model -> Tensor<Float> in
        let logits = model.applied(to: fakeImageBatch, in: context)
        let oneHotLabels = Tensor<Float>(oneHotAtIndices: fakeLabelBatch, depth: logits.shape[1])
        return softmaxCrossEntropy(logits: logits, oneHotLabels: oneHotLabels)
    }
    optimizer.update(&resnet.allDifferentiableVariables, along: gradients)
}
