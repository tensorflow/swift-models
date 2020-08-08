
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
import TensorFlow

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @differentiable(wrt: self)
  public func dimensionGathering<Index: TensorFlowIndex>(
    atIndices indices: Tensor<Index>
  ) -> Tensor {
    return _Raw.gatherNd(params: self, indices: indices)
  }

  /// Derivative of `_Raw.gatherNd`.
  ///
  /// Ported from TensorFlow Python reference implementation:
  /// https://github.com/tensorflow/tensorflow/blob/r2.2/tensorflow/python/ops/array_grad.py#L691-L701
  @inlinable
  @derivative(of: dimensionGathering)
  func _vjpDimensionGathering<Index: TensorFlowIndex>(
    atIndices indices: Tensor<Index>
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let shapeTensor = Tensor<Index>(self.shapeTensor)
    let value = _Raw.gatherNd(params: self, indices: indices)
    return (
      value,
      { v in
        let dparams = _Raw.scatterNd(indices: indices, updates: v, shape: shapeTensor)
        return dparams
      }
    )
  }
}
