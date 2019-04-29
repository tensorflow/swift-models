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

struct BasicBlock: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var blocks: [Conv2DBatchNorm]
    var shortcut: Conv2DBatchNorm

    init(
        featureCounts: (Int, Int),
        kernelSize: Int = 3,
        strides: (Int, Int) = (2, 2),
        blockCount: Int = 3
    ) {
        self.blocks = [Conv2DBatchNorm(
            filterShape: (kernelSize, kernelSize, featureCounts.0, featureCounts.1),
            strides: strides)]
        for _ in 2..<blockCount {
            self.blocks += [Conv2DBatchNorm(
                filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.1))]
        }
        self.shortcut = Conv2DBatchNorm(
            filterShape: (1, 1, featureCounts.0, featureCounts.1),
            strides: strides)
    }

    @differentiable
    func call(_ input: Input) -> Output {
        let blocksReduced = blocks.differentiableReduce(input) { last, layer in
            relu(layer(last))
        }
        return relu(blocksReduced + shortcut(input))
    }
}

struct ResNet: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var inputLayer = Conv2DBatchNorm(filterShape: (3, 3, 3, 16))

    var basicBlock1: BasicBlock
    var basicBlock2: BasicBlock
    var basicBlock3: BasicBlock

    init(blockCount: Int = 3) {
        basicBlock1 = BasicBlock(featureCounts:(16, 16), strides: (1, 1), blockCount: blockCount)
        basicBlock2 = BasicBlock(featureCounts:(16, 32), blockCount: blockCount)
        basicBlock3 = BasicBlock(featureCounts:(32, 64), blockCount: blockCount)
    }

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

func ResNet20() -> ResNet {
    return ResNet(blockCount: 3)
}

func ResNet32() -> ResNet {
    return ResNet(blockCount: 5)
}

func ResNet44() -> ResNet {
    return ResNet(blockCount: 7)
}

func ResNet56() -> ResNet {
    return ResNet(blockCount: 9)
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
    @differentiating(differentiableReduce, wrt: self)
    internal func reduceDerivative<Result: Differentiable>(
        _ initialResult: Result,
        _ nextPartialResult: @differentiable (Result, Element) -> Result
    ) -> (value: Result, pullback: (Result.CotangentVector) -> Array.CotangentVector) {
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
            return CotangentVector(elementCotangents.base.reversed())
        })
    }
}
