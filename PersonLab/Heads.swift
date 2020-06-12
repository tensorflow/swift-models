import ModelSupport
import TensorFlow

public struct PersonlabHeadsResults: Differentiable {
  public var heatmap: Tensor<Float>
  public var offsets: Tensor<Float>
  public var displacementsFwd: Tensor<Float>
  public var displacementsBwd: Tensor<Float>
}

public struct PersonlabHeads: Layer {
  @noDerivative let ckpt: CheckpointReader

  public var heatmap: Conv2D<Float>
  public var offsets: Conv2D<Float>
  public var displacementsFwd: Conv2D<Float>
  public var displacementsBwd: Conv2D<Float>

  public init(checkpoint: CheckpointReader) {
    self.ckpt = checkpoint

    self.heatmap = Conv2D<Float>(
      filter: ckpt.load(from: "heatmap_2/weights"),
      bias: ckpt.load(from: "heatmap_2/biases"),
      padding: .same
    )
    self.offsets = Conv2D<Float>(
      filter: ckpt.load(from: "offset_2/weights"),
      bias: ckpt.load(from: "offset_2/biases"),
      padding: .same
    )
    self.displacementsFwd = Conv2D<Float>(
      filter: ckpt.load(from: "displacement_fwd_2/weights"),
      bias: ckpt.load(from: "displacement_fwd_2/biases"),
      padding: .same
    )
    self.displacementsBwd = Conv2D<Float>(
      filter: ckpt.load(from: "displacement_bwd_2/weights"),
      bias: ckpt.load(from: "displacement_bwd_2/biases"),
      padding: .same
    )
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> PersonlabHeadsResults {
    return PersonlabHeadsResults(
      heatmap: sigmoid(self.heatmap(input)),
      offsets: self.offsets(input),
      displacementsFwd: self.displacementsFwd(input),
      displacementsBwd: self.displacementsBwd(input)
    )
  }
}
