import TensorFlow
import ModelSupport

func h(_ tensor: Tensor<Float>) {
    print("[\(tensor.flattened().sum()), \(tensor[0, 0, 0, 0]) \(tensor[0, -1, 1, 1]), \(tensor[0, 1, -1, 0]), \(tensor[0, -1, -1, -1])]")
}


struct Config {
    let inputSize = (height:241, width: 289)
    let testImagePath = "/Users/joaqo/swift-models/pose.jpg"
}


extension CheckpointReader {
    func load(from name: String) -> Tensor<Float> {
        return Tensor(self.loadTensor(named: "MobilenetV1/\(name)"))
    }
}


public struct DepthwiseSeparableConvBlock: Layer {
    var dConv: DepthwiseConv2D<Float>
    var conv: Conv2D<Float>
    var depthWiseFilter: Tensor<Float>
    var depthWiseBias: Tensor<Float>
    var pointWiseFilter: Tensor<Float>
    var pointWiseBias: Tensor<Float>

    @noDerivative let strides: (Int, Int)

    public init(
        depthWiseFilter: Tensor<Float>,
        depthWiseBias: Tensor<Float>,
        pointWiseFilter: Tensor<Float>,
        pointWiseBias: Tensor<Float>,
        strides: (Int, Int)
    ) {
        self.depthWiseFilter =  depthWiseFilter
        self.depthWiseBias =  depthWiseBias
        self.pointWiseFilter =  pointWiseFilter
        self.pointWiseBias =  pointWiseBias
        self.strides = strides

        dConv = DepthwiseConv2D<Float>(
            filter: depthWiseFilter,
            bias: depthWiseBias,
            activation: relu6,
            strides: strides,
            padding: .same
        )

        conv = Conv2D<Float>(
            filter: pointWiseFilter,
            bias: pointWiseBias,
            activation: relu6,
            padding: .same
        )
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: dConv, conv)
    }
}
