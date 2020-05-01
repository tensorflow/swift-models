// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import Batcher
import TensorFlow

/// A generic tuple of two tensors `Tensor`.
/// 
/// - Note: `TensorPair` has a generic name and provides little semantic information, to conform to
/// `Collatable`. You can use it for most basic datasets with one tensor of inputs and one tensor of
/// labels but you should write your own struct for more complex tasks (or if you want more descriptive
/// names).
public struct TensorPair<S1: TensorFlowScalar, S2: TensorFlowScalar>: _Collatable, KeyPathIterable {
    public var first: Tensor<S1>
    public var second: Tensor<S2>

    /// Creates from `first` and `second` tensors.
    public init(first: Tensor<S1>, second: Tensor<S2>) {
        self.first = first
        self.second = second
    }
}
