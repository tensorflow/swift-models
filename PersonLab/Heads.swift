// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
