import TensorFlow

public struct DoubleConv: Layer {

    var conv1: Conv2D<Float>
    var conv2: Conv2D<Float>

    public init(inputFilters: Int, outputFilters: Int) {
        conv1 = Conv2D<Float>(
            filterShape: (3, 3, inputFilters, outputFilters),
            strides: (1, 1),
            padding: .same,
            activation: relu
        )
        conv2 = Conv2D<Float>(
            filterShape: (3, 3, outputFilters, outputFilters),
            strides: (1, 1),
            padding: .same,
            activation: relu
        )
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        input.sequenced(through: conv1, conv2)
    }
}

public struct Down: Layer {
    var maxPool = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    var doubleConv: DoubleConv

    public init(inputFilters: Int) {
        doubleConv = DoubleConv(inputFilters: inputFilters, outputFilters: inputFilters * 2)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        input.sequenced(through: maxPool, doubleConv)
    }
}

public struct Up: Layer {
    var doubleConv: DoubleConv
    var upConv: TransposedConv2D<Float>

    public init(inputFilters: Int) {
        doubleConv = DoubleConv(inputFilters: inputFilters, outputFilters: inputFilters/2)
        upConv = TransposedConv2D<Float>(
            filterShape: (2, 2, inputFilters/4, inputFilters/2),
            strides: (2, 2),
            padding: .same
        )
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        input.sequenced(through: doubleConv, upConv)
    }
}

public struct UNet: Layer {
    var doubleConv1 = DoubleConv(inputFilters: 3, outputFilters: 64)
    var down1 = Down(inputFilters: 64)
    var down2 = Down(inputFilters: 128)
    var down3 = Down(inputFilters: 256)
    var down4 = Down(inputFilters: 512)
    var upConv1 = TransposedConv2D<Float>(
        filterShape: (2, 2, 512, 1024),
        strides: (2, 2),
        padding: .same
    )
    var up4 = Up(inputFilters: 1024)
    var up3 = Up(inputFilters: 512)
    var up2 = Up(inputFilters: 256)
    var doubleConvFinal = DoubleConv(inputFilters: 128, outputFilters: 64)
    var convFinal = Conv2D<Float>(
        filterShape: (1, 1, 64, 3),
        strides: (1, 1),
        padding: .same,
        activation: softmax
    )

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let output1 = doubleConv1(input)
        let outputDown1 = down1(output1)
        let outputDown2 = down2(outputDown1)
        let outputDown3 = down3(outputDown2)
        let outputDown4 = down4(outputDown3)
        let outputUpConv1 = upConv1(outputDown4)
        var x = up4(outputUpConv1.concatenated(with: outputDown3, alongAxis: -1))
        x = up3(x.concatenated(with: outputDown2, alongAxis: -1))
        x = up2(x.concatenated(with: outputDown1, alongAxis: -1))
        x = doubleConvFinal(x.concatenated(with: output1, alongAxis: -1))
        x = convFinal(x)
        return x

    }
}


var t = Tensor<Float>(randomNormal: [1, 224, 224, 3])
var model = UNet()
print(model(t).shape)

