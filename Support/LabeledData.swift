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

// Note: This is a struct and not a tuple because we need the `Collatable`
// conformance below.
/// A tuple (data, label) that can be used to train a deep learning model.
///
/// - Parameter `Data`: the type of the input.
/// - Parameter `Label`: the type of the target.
public struct LabeledData<Data, Label> {
  /// The `data` of our sample (usually used as input for a model).
  public let data: Data
  /// The `label` of our sample (usually used as target for a model).
  public let label: Label

  /// Creates an instance from `data` and `label`.
  public init(data: Data, label: Label) {
    self.data = data
    self.label = label
  }
}

extension LabeledData: Collatable where Data: Collatable, Label: Collatable {
  /// Creates an instance from collating `samples`.
  public init<BatchSamples: Collection>(collating samples: BatchSamples)
  where BatchSamples.Element == Self {
    self.init(data: .init(collating: samples.map(\.data)),
              label: .init(collating: samples.map(\.label)))
  }
}
