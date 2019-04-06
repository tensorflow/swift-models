import TensorFlow

// Pre-activated Resnet (aka Resnet v2):
// Original Paper:
// "Identity Mappings in Deep Residual Networks"
// Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun
// https://arxiv.org/abs/1603.05027
// https://github.com/KaimingHe/resnet-1k-layers/

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
        return norm.applied(to: conv.applied(to: input, in: context), in: context)
    }
}

struct BatchNormConv2D: Layer {
    var norm: BatchNorm<Float>
    var conv: Conv2D<Float>

    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid
    ) {
        self.norm = BatchNorm(featureCount: filterShape.2)
        self.conv = Conv2D(filterShape: filterShape, strides: strides, padding: padding)
    }

    @differentiable
    func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        let tmp = relu(norm.applied(to: input, in: context))
        return conv.applied(to: tmp, in: context)
    }
}

struct PreActivatedResidualBasicBlock: Layer {
    var layer1: BatchNormConv2D
    var layer2: BatchNormConv2D

    init(
        featureCounts: (Int, Int, Int, Int),
        kernelSize: Int = 3,
        strides: (Int, Int) = (1, 1)
    ) {
        self.layer1 = BatchNormConv2D(
            filterShape: (kernelSize, kernelSize, featureCounts.0, featureCounts.1),
            strides: strides,
            padding: .same)
        self.layer2 = BatchNormConv2D(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.3),
            strides: strides,
            padding: .same)
    }

    @differentiable
    func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        let tmp = layer1.applied(to: input, in: context)
        return layer2.applied(to: tmp, in: context)
    }
}

struct PreActivatedResidualBasicBlockShortcut: Layer {
    var layer1: BatchNormConv2D
    var layer2: BatchNormConv2D
    var shortcut: Conv2D<Float>

    init(featureCounts: (Int, Int, Int, Int), kernelSize: Int = 3) {
        self.layer1 = BatchNormConv2D(
            filterShape: (kernelSize, kernelSize, featureCounts.0, featureCounts.1),
            strides: (2, 2),
            padding: .same)
        self.layer2 = BatchNormConv2D(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            strides: (1, 1),
            padding: .same)
        self.shortcut = Conv2D(
            filterShape: (1, 1, featureCounts.0, featureCounts.3),
            strides: (2, 2),
            padding: .same)
    }

    @differentiable
    func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        var tmp = layer1.applied(to: input, in: context)
        tmp = layer2.applied(to: tmp, in: context)
        return tmp + shortcut.applied(to: input, in: context)
    }
}

struct PreActivatedResNet18: Layer {
    var l1: Conv2DBatchNorm
    var maxPool: MaxPool2D<Float>

    var l2a = PreActivatedResidualBasicBlock(featureCounts: (64, 64, 64, 64))
    var l2b = PreActivatedResidualBasicBlock(featureCounts: (64, 64, 64, 64))

    var l3a = PreActivatedResidualBasicBlockShortcut(featureCounts: (64, 128, 128, 128))
    var l3b = PreActivatedResidualBasicBlock(featureCounts: (128, 128, 128, 128))

    var l4a = PreActivatedResidualBasicBlockShortcut(featureCounts: (128, 256, 256, 256))
    var l4b = PreActivatedResidualBasicBlock(featureCounts: (256, 256, 256, 256))

    var l5a = PreActivatedResidualBasicBlockShortcut(featureCounts: (256, 512, 512, 512))
    var l5b = PreActivatedResidualBasicBlock(featureCounts: (512, 512, 512, 512))
 
    var norm: BatchNorm<Float>
    var avgPool: AvgPool2D<Float>
    var flatten = Flatten<Float>()
    var classifier: Dense<Float>

    init(imageSize: Int, classCount: Int) {
        // default to the ImageNet case where imageSize == 224
        // Swift requires that all properties get initialized outside control flow
        l1 = Conv2DBatchNorm(filterShape: (7, 7, 3, 64), strides: (2, 2), padding: .same)
        maxPool = MaxPool2D(poolSize: (3, 3), strides: (2, 2))
        avgPool = AvgPool2D(poolSize: (7, 7), strides: (7, 7))
        if imageSize == 32 {
            l1 = Conv2DBatchNorm(filterShape: (3, 3, 3, 64), padding: .same)
            maxPool = MaxPool2D(poolSize: (1, 1), strides: (1, 1)) // no-op
            avgPool = AvgPool2D(poolSize: (4, 4), strides: (4, 4))
        }
        norm = BatchNorm(featureCount: 512)
        classifier = Dense(inputSize: 512, outputSize: classCount)
    }

    @differentiable
    func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        var tmp = l1.applied(to: input, in: context)
        tmp = maxPool.applied(to: tmp, in: context)

        tmp = l2a.applied(to: tmp, in: context)
        tmp = l2b.applied(to: tmp, in: context)

        tmp = l3a.applied(to: tmp, in: context)
        tmp = l3b.applied(to: tmp, in: context)

        tmp = l4a.applied(to: tmp, in: context)
        tmp = l4b.applied(to: tmp, in: context)

        tmp = l5a.applied(to: tmp, in: context)
        tmp = l5b.applied(to: tmp, in: context)

        tmp = relu(norm.applied(to: tmp, in: context))
        tmp = avgPool.applied(to: tmp, in: context)
        tmp = flatten.applied(to: tmp, in: context)
        return classifier.applied(to: tmp, in: context)
    }
}

struct PreActivatedResNet34: Layer {
    var l1: Conv2DBatchNorm
    var maxPool: MaxPool2D<Float>

    var l2a = PreActivatedResidualBasicBlock(featureCounts: (64, 64, 64, 64))
    var l2b = PreActivatedResidualBasicBlock(featureCounts: (64, 64, 64, 64))
    var l2c = PreActivatedResidualBasicBlock(featureCounts: (64, 64, 64, 64))

    var l3a = PreActivatedResidualBasicBlockShortcut(featureCounts: (64, 128, 128, 128))
    var l3b = PreActivatedResidualBasicBlock(featureCounts: (128, 128, 128, 128))
    var l3c = PreActivatedResidualBasicBlock(featureCounts: (128, 128, 128, 128))
    var l3d = PreActivatedResidualBasicBlock(featureCounts: (128, 128, 128, 128))

    var l4a = PreActivatedResidualBasicBlockShortcut(featureCounts: (128, 256, 256, 256))
    var l4b = PreActivatedResidualBasicBlock(featureCounts: (256, 256, 256, 256))
    var l4c = PreActivatedResidualBasicBlock(featureCounts: (256, 256, 256, 256))
    var l4d = PreActivatedResidualBasicBlock(featureCounts: (256, 256, 256, 256))
    var l4e = PreActivatedResidualBasicBlock(featureCounts: (256, 256, 256, 256))
    var l4f = PreActivatedResidualBasicBlock(featureCounts: (256, 256, 256, 256))

    var l5a = PreActivatedResidualBasicBlockShortcut(featureCounts: (256, 512, 512, 512))
    var l5b = PreActivatedResidualBasicBlock(featureCounts: (512, 512, 512, 512))
    var l5c = PreActivatedResidualBasicBlock(featureCounts: (512, 512, 512, 512))
 
    var norm: BatchNorm<Float>
    var avgPool: AvgPool2D<Float>
    var flatten = Flatten<Float>()
    var classifier: Dense<Float>

    init(imageSize: Int, classCount: Int) {
        // default to the ImageNet case where imageSize == 224
        // Swift requires that all properties get initialized outside control flow
        l1 = Conv2DBatchNorm(filterShape: (7, 7, 3, 64), strides: (2, 2), padding: .same)
        maxPool = MaxPool2D(poolSize: (3, 3), strides: (2, 2))
        avgPool = AvgPool2D(poolSize: (7, 7), strides: (7, 7))
        if imageSize == 32 {
            l1 = Conv2DBatchNorm(filterShape: (3, 3, 3, 64), padding: .same)
            maxPool = MaxPool2D(poolSize: (1, 1), strides: (1, 1)) // no-op
            avgPool = AvgPool2D(poolSize: (4, 4), strides: (4, 4))
        }
        norm = BatchNorm(featureCount: 512)
        classifier = Dense(inputSize: 512, outputSize: classCount)
    }

    @differentiable
    func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        var tmp = l1.applied(to: input, in: context)
        tmp = maxPool.applied(to: tmp, in: context)

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
        tmp = l5c.applied(to: tmp, in: context)

        tmp = relu(norm.applied(to: tmp, in: context))
        tmp = avgPool.applied(to: tmp, in: context)
        tmp = flatten.applied(to: tmp, in: context)
        return classifier.applied(to: tmp, in: context)
    }
}
