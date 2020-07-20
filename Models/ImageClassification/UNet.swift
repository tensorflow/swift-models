import TensorFlow
import PythonKit

// TODO: remove this terrible way of saving/loading weights
public extension Layer {
    func weights() -> PythonObject {
        let np = Python.import("numpy")
        var tensors: [PythonObject] = []
        for kp in self.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            tensors.append(self[keyPath: kp].makeNumpyArray())
        }
        return np.array(tensors)
    }

    mutating func load(weights: PythonObject) {
        for (kpIdx, kp) in self.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self).enumerated() {
            self[keyPath: kp] = Tensor<Float>(numpy: weights[kpIdx].astype("float32"))!
        }
    }
}

public struct UNet: Layer {
    struct ConvBlock: Layer {
        var conv: Conv2D<Float>
        var bn: BatchNorm<Float>

        @noDerivative var bnMean: Tensor<Float> = Tensor(0)
        @noDerivative var bnVariance: Tensor<Float> = Tensor(0)

        mutating func save() {
          bnMean = bn.runningMean.value
          bnVariance = bn.runningVariance.value
        }
        mutating func restore() {
          bn.runningMean.value = bnMean
          bn.runningVariance.value = bnVariance
        }

        init(filterShape: (Int, Int, Int, Int)) {
            self.conv = Conv2D(filterShape: filterShape, padding: .same)
            self.bn = BatchNorm(featureCount: filterShape.3)
        }

        func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
            return relu(input.sequenced(through: conv, bn))
        }
    }

    @noDerivative let encoderPooling = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    var encoderConv1 = ConvBlock(filterShape: (3, 3, 3, 64))
    var encoderConv2 = ConvBlock(filterShape: (3, 3, 64, 128))
    var encoderConv3 = ConvBlock(filterShape: (3, 3, 128, 256))
    var encoderConv4 = ConvBlock(filterShape: (3, 3, 256, 512))

    @noDerivative let decoderUpsampling = UpSampling2D<Float>(size: 2)
    var decoderConv1 = ConvBlock(filterShape: (3, 3, 768, 256))
    var decoderConv2 = ConvBlock(filterShape: (3, 3, 384, 128))
    var decoderConv3 = ConvBlock(filterShape: (3, 3, 192, 64))
    var decoderConv4 = Conv2D<Float>(filterShape: (1, 1, 64, 1))

    public init() {}

    public mutating func save() {
      encoderConv1.save()
      encoderConv2.save()
      encoderConv3.save()
      encoderConv4.save()
      decoderConv1.save()
      decoderConv2.save()
      decoderConv3.save()
    }

    public mutating func restore() {
      encoderConv1.restore()
      encoderConv2.restore()
      encoderConv3.restore()
      encoderConv4.restore()
      decoderConv1.restore()
      decoderConv2.restore()
      decoderConv3.restore()
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let e1 = encoderConv1(input)
        let e2 = e1.sequenced(through: encoderConv2, encoderPooling)
        let e3 = e2.sequenced(through: encoderConv3, encoderPooling)
        let e4 = e3.sequenced(through: encoderConv4, encoderPooling)
        let d1i = decoderUpsampling(e4).concatenated(with: e3, alongAxis: 3)
        let d1 = decoderConv1(d1i)
        let d2i = decoderUpsampling(d1).concatenated(with: e2, alongAxis: 3)
        let d2 = decoderConv2(d2i)
        let d3i = decoderUpsampling(d2).concatenated(with: e1, alongAxis: 3)
        let d3 = decoderConv3(d3i)
        return decoderConv4(d3)
    }
}
